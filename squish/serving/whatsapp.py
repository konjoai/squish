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
) -> None:
    """
    Process one incoming WhatsApp message and send a reply.

    Runs in a daemon thread — Meta requires the webhook to return 200 quickly,
    so generation happens here asynchronously.
    """
    def _reply(msg: str) -> None:
        _send_whatsapp_reply(phone_number_id, access_token, sender, msg)

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

    _reply(reply_text)


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
                            system_prompt,
                        ),
                        daemon=True,
                    ).start()

        # Always return 200 — Meta will retry if we don't
        return Response(content="", status_code=200)

    print("[squish] WhatsApp webhook mounted at /webhook/whatsapp (Meta Cloud API)", flush=True)
