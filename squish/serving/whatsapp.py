"""
squish/serving/whatsapp.py — WhatsApp integration via Meta (WhatsApp Business) Cloud API.

Free, direct Meta API — no Twilio, no third-party intermediaries.
Free tier: 1,000 service conversations / month at no cost.

Setup (one-time — Meta Developer Dashboard)
───────────────────────────────────────────
  1. Create a Meta Developer account: https://developers.facebook.com
  2. Create a new App → Add Product → WhatsApp
  3. In WhatsApp → Configuration:
       - Add your server URL as the Webhook URL:
           https://your-domain.example.com/webhook/whatsapp
       - Set a Verify Token (any secret string you choose)
       - Subscribe to the "messages" webhook field
  4. Collect from the Dashboard:
       - App Secret          (App Settings → Basic)
       - Access Token        (WhatsApp → API Setup → temporary token, or generate permanent)
       - Phone Number ID     (WhatsApp → API Setup)

  The server must be publicly reachable over HTTPS so Meta can POST to it.
  Recommended options (no third-party tunnel services):
    • VPS + Caddy (full control):
        your-domain.example.com {
            reverse_proxy localhost:11435
        }
    • Tailscale Funnel (self-controlled, end-to-end encrypted):
        tailscale funnel 11435

Usage
─────
  squish run 7b \\
      --whatsapp \\
      --whatsapp-verify-token   my_secret_verify_token \\
      --whatsapp-app-secret     <app-secret-from-meta-dashboard> \\
      --whatsapp-access-token   <access-token-from-meta-dashboard> \\
      --whatsapp-phone-number-id  <phone-number-id-from-meta-dashboard> \\
      --host 0.0.0.0 --port 11435

WhatsApp commands (sent as normal messages):
  /reset   — clear conversation history for your number
  /status  — show model name, avg TPS, uptime
  /help    — list available commands

Authentication
──────────────
  Incoming requests are validated using the X-Hub-Signature-256 header
  (HMAC-SHA256 of the raw request body signed with your Meta App Secret)
  when --whatsapp-app-secret is set. No extra packages required — uses
  Python's stdlib hmac + hashlib only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import threading
import time
import urllib.error as _urlerr
import urllib.request as _urlreq
from typing import Any

# ── Optional FastAPI (required when actually mounted) ────────────────────────
try:
    from fastapi import Query, Request
    from fastapi.responses import Response
    _FASTAPI = True
except ImportError:  # pragma: no cover
    _FASTAPI = False
    Request = Any  # type: ignore[assignment,misc]
    Response = Any  # type: ignore[assignment,misc]
    Query = Any

# ── Conversation store ───────────────────────────────────────────────────────
# Keyed by the sender's E.164 phone number as returned by Meta (e.g. "15551234567")
_sessions: dict[str, list[dict[str, str]]] = {}
_sessions_ts: dict[str, float] = {}   # last-activity timestamp per number
_sessions_lock = threading.Lock()
_MAX_HISTORY = 20        # total messages kept per session
_SESSION_TIMEOUT = 3600  # seconds of inactivity before session is cleared

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Keep replies short and suitable for a mobile messaging app. "
    "Avoid markdown formatting unless the user asks for it."
)

# ── Per-sender rate limiting ──────────────────────────────────────────────────
_RATE_LIMIT: int = 10       # max messages per sender per _RATE_WINDOW seconds
_RATE_WINDOW: float = 60.0
_rate_counts: dict[str, list[float]] = {}
_rate_lock = threading.Lock()

# ── Message deduplication ────────────────────────────────────────────────────
# Prevents duplicate processing when Meta retries a webhook (e.g. our 200 was late).
_SEEN_IDS_MAX: int = 500
_SEEN_IDS: set[str] = set()
_SEEN_IDS_ORDER: list[str] = []
_SEEN_IDS_LOCK = threading.Lock()

# ── Reply chunking ───────────────────────────────────────────────────────────
_MAX_MSG_LEN: int = 4096     # WhatsApp Cloud API per-message character limit

# ── Session helpers ──────────────────────────────────────────────────────────

def _expire_old_sessions() -> None:
    """Remove sessions inactive for longer than _SESSION_TIMEOUT (call with lock held)."""
    now = time.time()
    dead = [k for k, ts in _sessions_ts.items() if now - ts > _SESSION_TIMEOUT]
    for k in dead:
        _sessions.pop(k, None)
        _sessions_ts.pop(k, None)


def _get_or_create_session(
    sender: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Return (possibly new) message list for the given sender."""
    with _sessions_lock:
        _expire_old_sessions()
        if sender not in _sessions:
            _sessions[sender] = [{"role": "system", "content": system_prompt}]
        _sessions_ts[sender] = time.time()
        return _sessions[sender]


def _reset_session(sender: str, system_prompt: str) -> None:
    with _sessions_lock:
        _sessions[sender] = [{"role": "system", "content": system_prompt}]
        _sessions_ts[sender] = time.time()


def _apply_max_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep all system messages + only the last (_MAX_HISTORY-1) non-system turns."""
    system = [m for m in messages if m["role"] == "system"]
    non_sys = [m for m in messages if m["role"] != "system"]
    keep = non_sys[-(_MAX_HISTORY - 1):] if len(non_sys) > _MAX_HISTORY - 1 else non_sys
    return system + keep


def _is_duplicate(msg_id: str) -> bool:
    """
    Return True if msg_id was already processed; otherwise record it and return False.

    Keeps a bounded FIFO of the last _SEEN_IDS_MAX message IDs so Meta webhook
    retries (which occur when our 200 response arrives late) are silently dropped.
    An empty msg_id is never treated as a duplicate.
    """
    if not msg_id:
        return False
    with _SEEN_IDS_LOCK:
        if msg_id in _SEEN_IDS:
            return True
        _SEEN_IDS.add(msg_id)
        _SEEN_IDS_ORDER.append(msg_id)
        while len(_SEEN_IDS_ORDER) > _SEEN_IDS_MAX:
            evict = _SEEN_IDS_ORDER.pop(0)
            _SEEN_IDS.discard(evict)
        return False


def _is_rate_limited(sender: str) -> bool:
    """
    Return True if sender has exceeded _RATE_LIMIT messages in the last _RATE_WINDOW seconds.

    Uses a sliding window of per-sender timestamps. Thread-safe.
    """
    now = time.time()
    with _rate_lock:
        timestamps = _rate_counts.get(sender, [])
        timestamps = [t for t in timestamps if now - t < _RATE_WINDOW]
        if len(timestamps) >= _RATE_LIMIT:
            _rate_counts[sender] = timestamps
            return True
        timestamps.append(now)
        _rate_counts[sender] = timestamps
        return False


def _chunk_reply(text: str) -> list[str]:
    """
    Split text into a list of strings each at most _MAX_MSG_LEN characters.

    Splits preferentially at newlines, then spaces, then performs a hard cut
    if no whitespace boundary is found within the limit.
    """
    if len(text) <= _MAX_MSG_LEN:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= _MAX_MSG_LEN:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, _MAX_MSG_LEN)
        if split_at < 1:
            split_at = text.rfind(" ", 0, _MAX_MSG_LEN)
        if split_at < 1:
            split_at = _MAX_MSG_LEN
        chunks.append(text[:split_at].rstrip())
        text = text[split_at:].lstrip()
    return chunks


# ── Meta API helpers ─────────────────────────────────────────────────────────

def _validate_meta_signature(
    app_secret: str,
    raw_body: bytes,
    signature_header: str,
) -> bool:
    """
    Validate the X-Hub-Signature-256 header sent by Meta on every webhook POST.

    Meta computes HMAC-SHA256 of the raw request body using the app secret as
    the key and sends it as "sha256=<hex_digest>".
    """
    if not signature_header.startswith("sha256="):
        return False
    expected = signature_header[7:]
    computed = hmac.new(app_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(computed, expected)


def _send_whatsapp_reply(
    phone_number_id: str,
    access_token: str,
    to: str,
    body: str,
) -> None:
    """
    Send a text reply via the Meta WhatsApp Cloud API.

    POST https://graph.facebook.com/v19.0/{phone_number_id}/messages
    """
    url = f"https://graph.facebook.com/v19.0/{phone_number_id}/messages"
    payload = json.dumps({
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }).encode("utf-8")
    req = _urlreq.Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with _urlreq.urlopen(req, timeout=15):
            pass
    except _urlerr.URLError as exc:
        print(f"[squish/whatsapp] send failed: {exc}", flush=True)
    except Exception as exc:
        print(f"[squish/whatsapp] unexpected send error: {exc}", flush=True)


# ── Message processing ───────────────────────────────────────────────────────

def _handle_message(
    sender: str,
    text: str,
    phone_number_id: str,
    access_token: str,
    get_state: Any,
    get_generate: Any,
    get_tokenizer: Any,
    system_prompt: str,
    msg_id: str = "",
) -> None:
    """
    Process one incoming WhatsApp message and send a reply.

    Runs in a daemon thread — Meta requires the webhook to return 200 quickly,
    so generation happens here asynchronously.
    """
    def _reply(msg: str) -> None:
        _send_whatsapp_reply(phone_number_id, access_token, sender, msg)

    # Drop Meta webhook retries — same wamid already handled
    if _is_duplicate(msg_id):
        return

    # Throttle senders that exceed the rate limit
    if _is_rate_limited(sender):
        _reply("You're sending messages too quickly. Please wait a moment before sending again.")
        return

    # Special commands
    cmd = text.strip()
    if cmd == "/reset":
        _reset_session(sender, system_prompt)
        _reply("Session cleared. Starting fresh!")
        return
    if cmd == "/help":
        _reply(
            "Available commands:\n"
            "/reset  — clear conversation history\n"
            "/status — show model info\n"
            "/help   — this message"
        )
        return
    if cmd == "/status":
        state = get_state()
        if state.model is None:
            _reply("Model not loaded yet.")
        else:
            uptime = int(time.time() - state.loaded_at)
            _reply(
                f"Model: {state.model_name}\n"
                f"Speed: {state.avg_tps:.1f} tok/s\n"
                f"Requests: {state.requests}\n"
                f"Uptime: {uptime // 60}m {uptime % 60}s"
            )
        return

    # Model must be loaded
    state = get_state()
    if state.model is None:
        _reply("The AI model is still loading. Please try again in a moment.")
        return

    # Build prompt from history
    messages = _get_or_create_session(sender, system_prompt)
    with _sessions_lock:
        messages.append({"role": "user", "content": text})
        trimmed = _apply_max_history(messages)
        _sessions[sender] = trimmed

    tokenizer = get_tokenizer()
    try:
        prompt = tokenizer.apply_chat_template(
            trimmed, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in trimmed) + "\nASSISTANT:"

    # Generate (synchronous generator, non-streaming)
    _generate = get_generate()
    tokens: list[str] = []
    try:
        for tok_text, finish in _generate(
            prompt, max_tokens=512, temperature=0.7, top_p=0.9, stop=None, seed=None
        ):
            if tok_text:
                tokens.append(tok_text)
            if finish is not None:
                break
    except Exception as exc:
        _reply(f"Sorry, a generation error occurred: {exc}")
        return

    reply_text = "".join(tokens).strip() or "Sorry, I could not generate a response."

    with _sessions_lock:
        _sessions[sender].append({"role": "assistant", "content": reply_text})
        _sessions_ts[sender] = time.time()

    for chunk in _chunk_reply(reply_text):
        _reply(chunk)


# ── mount_whatsapp ───────────────────────────────────────────────────────────

def mount_whatsapp(
    app: Any,
    get_state: Any,
    get_generate: Any,
    get_tokenizer: Any,
    verify_token: str = "",
    app_secret: str = "",
    access_token: str = "",
    phone_number_id: str = "",
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> None:
    """
    Register WhatsApp webhook endpoints on ``app`` using the Meta Cloud API.

    Called from server.py when --whatsapp is set:

        mount_whatsapp(
            app,
            get_state        = lambda: _state,
            get_generate     = lambda: _generate_tokens,
            get_tokenizer    = lambda: _state.tokenizer,
            verify_token     = args.whatsapp_verify_token,
            app_secret       = args.whatsapp_app_secret,
            access_token     = args.whatsapp_access_token,
            phone_number_id  = args.whatsapp_phone_number_id,
            system_prompt    = args.system_prompt or "",
        )
    """
    if not _FASTAPI:  # pragma: no cover
        return

    # ── GET /webhook/whatsapp ─────────────────────────────────────────────
    # Meta sends a GET with hub.mode, hub.verify_token, hub.challenge to
    # confirm the webhook URL is under your control. Return hub.challenge.
    @app.get("/webhook/whatsapp")
    async def whatsapp_challenge(
        hub_mode: str | None = Query(None, alias="hub.mode"),
        hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
        hub_challenge: str | None = Query(None, alias="hub.challenge"),
    ) -> Response:  # pragma: no cover
        if hub_mode == "subscribe" and hub_verify_token == verify_token:
            return Response(content=hub_challenge or "", media_type="text/plain")
        return Response(content="Forbidden", status_code=403)

    # ── POST /webhook/whatsapp ────────────────────────────────────────────
    # Meta POSTs JSON. We must return 200 immediately; reply is sent
    # asynchronously via the Graph API in a daemon thread.
    @app.post("/webhook/whatsapp")
    async def whatsapp_incoming(request: Request) -> Response:  # pragma: no cover
        raw_body = await request.body()

        # Signature validation (HMAC-SHA256, stdlib only)
        if app_secret:
            sig = request.headers.get("X-Hub-Signature-256", "")
            if not sig:
                return Response(content="missing signature", status_code=403)
            if not _validate_meta_signature(app_secret, raw_body, sig):
                return Response(content="invalid signature", status_code=403)

        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            return Response(content="", status_code=200)

        # Walk the Meta webhook payload structure
        for entry in body.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                for msg in value.get("messages", []):
                    if msg.get("type") != "text":
                        continue  # skip media/reactions/etc.
                    msg_id = msg.get("id", "")
                    sender = msg.get("from", "")
                    text = (msg.get("text") or {}).get("body", "")
                    if not sender or not text:
                        continue
                    threading.Thread(
                        target=_handle_message,
                        args=(
                            sender, text.strip(),
                            phone_number_id, access_token,
                            get_state, get_generate, get_tokenizer,
                            system_prompt, msg_id,
                        ),
                        daemon=True,
                    ).start()

        # Always return 200 — Meta will retry if we don't
        return Response(content="", status_code=200)

    print("[squish] WhatsApp webhook mounted at /webhook/whatsapp (Meta Cloud API)", flush=True)
