"""
squish/serving/whatsapp.py — WhatsApp/Twilio webhook integration for the squish server.

Mounts two endpoints on the existing FastAPI app:
  GET  /webhook/whatsapp  — Twilio challenge handshake (returns 200 OK)
  POST /webhook/whatsapp  — incoming WhatsApp message → squish reply → TwiML response

When the --whatsapp flag is passed to `squish run`, this module is loaded and
`mount_whatsapp(app, ...)` is called from server.py's module-level init block,
following the same pattern as `squish/serving/ollama_compat.py`.

Usage
─────
  squish run 7b \
      --whatsapp \
      --twilio-account-sid  ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
      --twilio-auth-token   your_auth_token \
      --host 0.0.0.0 --port 11435

  # Expose publicly for Twilio callback:
  ngrok http 11435
  # Set Twilio WhatsApp sandbox URL → https://<ngrok-id>.ngrok.io/webhook/whatsapp

WhatsApp commands (sent as normal messages):
  /reset   — clear conversation history for your number
  /status  — show model name, avg TPS, uptime
  /help    — list available commands

Authentication
──────────────
  Twilio HMAC-SHA1 signature validation is performed when --twilio-auth-token is set
  and the twilio package is installed (pip install 'squish[whatsapp]').
  If the twilio package is absent, signature validation is skipped with a warning.
  Request bodies are always validated for required fields regardless.
"""
from __future__ import annotations

import hashlib
import hmac
import threading
import time
import urllib.parse
from typing import Any

# ── Optional FastAPI (required when actually mounted) ────────────────────────
try:
    from fastapi import Request
    from fastapi.responses import Response
    _FASTAPI = True
except ImportError:  # pragma: no cover
    _FASTAPI = False
    Request = Any  # type: ignore[assignment,misc]
    Response = Any  # type: ignore[assignment,misc]

# ── Optional Twilio SDK ──────────────────────────────────────────────────────
try:
    from twilio.request_validator import RequestValidator as _TwilioValidator
    _TWILIO_SDK = True
except ImportError:  # pragma: no cover
    _TWILIO_SDK = False
    _TwilioValidator = None  # type: ignore[assignment,misc]

# ── Conversation store ───────────────────────────────────────────────────────
# Keyed by Twilio's E.164 'From' string, e.g. "whatsapp:+15551234567"
_sessions: dict[str, list[dict[str, str]]] = {}
_sessions_ts: dict[str, float] = {}   # last-activity timestamp per number
_sessions_lock = threading.Lock()
_MAX_HISTORY = 20       # messages per session (matches squish chat default)
_SESSION_TIMEOUT = 3600  # seconds of inactivity before session is reset

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Keep replies short and suitable for a mobile messaging app. "
    "Avoid markdown formatting unless the user asks for it."
)


def _twiml_reply(body: str) -> str:
    """Wrap a plain-text reply in a Twilio Messaging Response TwiML envelope."""
    escaped = (
        body
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f"<Response><Message>{escaped}</Message></Response>"
    )


def _expire_old_sessions() -> None:
    """Remove sessions that have been inactive for longer than _SESSION_TIMEOUT."""
    now = time.time()
    to_delete = [k for k, ts in _sessions_ts.items() if now - ts > _SESSION_TIMEOUT]
    for k in to_delete:
        _sessions.pop(k, None)
        _sessions_ts.pop(k, None)


def _get_or_create_session(
    from_number: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """Return (possibly new) message list for the given number."""
    with _sessions_lock:
        _expire_old_sessions()
        if from_number not in _sessions:
            _sessions[from_number] = [{"role": "system", "content": system_prompt}]
        _sessions_ts[from_number] = time.time()
        return _sessions[from_number]


def _reset_session(from_number: str, system_prompt: str) -> None:
    with _sessions_lock:
        _sessions[from_number] = [{"role": "system", "content": system_prompt}]
        _sessions_ts[from_number] = time.time()


def _apply_max_history(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep system prompt + last (_MAX_HISTORY - 1) non-system messages."""
    system = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]
    trimmed = non_system[-(_MAX_HISTORY - 1):] if len(non_system) > _MAX_HISTORY - 1 else non_system
    return system + trimmed


def _validate_twilio_signature(
    auth_token: str,
    url: str,
    form_data: dict[str, str],
    signature: str,
) -> bool:
    """
    Validate a Twilio webhook signature.

    Uses the twilio SDK if available, otherwise falls back to a manual
    HMAC-SHA1 implementation that matches Twilio's documented algorithm.
    """
    if _TWILIO_SDK:
        validator = _TwilioValidator(auth_token)
        return validator.validate(url, form_data, signature)

    # Manual implementation of Twilio's HMAC-SHA1 signature scheme:
    # 1. Start with the full URL of the request URL.
    # 2. Sort all POST parameters alphabetically, append key+value with no separators.
    # 3. Sign with HMAC-SHA1 using the auth_token as the key.
    # 4. Base64-encode the result.
    import base64
    import hashlib
    import hmac as _hmac

    s = url
    for k in sorted(form_data.keys()):
        s += k + form_data[k]
    mac = _hmac.new(auth_token.encode("utf-8"), s.encode("utf-8"), hashlib.sha1).digest()
    computed = base64.b64encode(mac).decode("utf-8")
    return _hmac.compare_digest(computed, signature)


def mount_whatsapp(
    app: Any,
    get_state: Any,
    get_generate: Any,
    get_tokenizer: Any,
    account_sid: str = "",
    auth_token: str = "",
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> None:
    """
    Register WhatsApp webhook endpoints on ``app``.

    Called from server.py's module-level init block when --whatsapp is set:

        mount_whatsapp(
            app,
            get_state     = lambda: _state,
            get_generate  = lambda: _generate_tokens,
            get_tokenizer = lambda: _state.tokenizer,
            account_sid   = args.twilio_account_sid,
            auth_token    = args.twilio_auth_token,
            system_prompt = args.system_prompt or _DEFAULT_SYSTEM_PROMPT,
        )
    """
    if not _FASTAPI:  # pragma: no cover
        return

    if not _TWILIO_SDK and auth_token:
        print(
            "[squish/whatsapp] twilio package not installed — "
            "signature validation will use fallback HMAC implementation.\n"
            "  Install with: pip install 'squish[whatsapp]'",
            flush=True,
        )

    # ── GET /webhook/whatsapp ─────────────────────────────────────────────
    @app.get("/webhook/whatsapp")
    async def whatsapp_challenge(request: Request) -> Response:  # pragma: no cover
        """
        Twilio sends a GET to verify the webhook URL is reachable.
        Always return 200 OK with an empty TwiML envelope.
        """
        return Response(
            content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
            media_type="application/xml",
        )

    # ── POST /webhook/whatsapp ────────────────────────────────────────────
    @app.post("/webhook/whatsapp")
    async def whatsapp_incoming(request: Request) -> Response:  # pragma: no cover
        """
        Handle an incoming WhatsApp message from Twilio.

        Request body: application/x-www-form-urlencoded with at minimum:
          From=whatsapp:+15551234567
          Body=Hello, how are you?
        """
        # 1. Parse form body
        raw_body = await request.body()
        try:
            form = dict(urllib.parse.parse_qsl(raw_body.decode("utf-8"), keep_blank_values=True))
        except Exception:
            return Response(
                content=_twiml_reply("Sorry, could not parse your message."),
                media_type="application/xml",
                status_code=400,
            )

        from_number = form.get("From", "").strip()
        user_text   = form.get("Body", "").strip()

        if not from_number or not user_text:
            return Response(
                content=_twiml_reply("No message received."),
                media_type="application/xml",
            )

        # 2. Twilio signature validation
        if auth_token:
            sig = request.headers.get("X-Twilio-Signature", "")
            # Build the full request URL as Twilio signed it
            scheme = request.headers.get("X-Forwarded-Proto", request.url.scheme)
            host   = request.headers.get("X-Forwarded-Host", request.headers.get("host", ""))
            full_url = f"{scheme}://{host}/webhook/whatsapp"

            if not sig:
                return Response(
                    content=_twiml_reply("Unauthorized: missing signature."),
                    media_type="application/xml",
                    status_code=403,
                )
            valid = _validate_twilio_signature(auth_token, full_url, form, sig)
            if not valid:
                return Response(
                    content=_twiml_reply("Unauthorized: invalid signature."),
                    media_type="application/xml",
                    status_code=403,
                )

        # 3. Handle special commands
        cmd = user_text.lstrip("/").lower()
        if user_text.startswith("/"):
            if cmd == "reset":
                _reset_session(from_number, system_prompt)
                return Response(
                    content=_twiml_reply("Session cleared. Starting fresh!"),
                    media_type="application/xml",
                )
            if cmd == "help":
                help_text = (
                    "Available commands:\n"
                    "/reset  — clear conversation history\n"
                    "/status — show model info\n"
                    "/help   — this message"
                )
                return Response(content=_twiml_reply(help_text), media_type="application/xml")
            if cmd == "status":
                ms = get_state()
                if ms.model is None:
                    status_text = "Model not loaded yet."
                else:
                    status_text = (
                        f"Model: {ms.model_name}\n"
                        f"Avg TPS: {ms.avg_tps:.1f}\n"
                        f"Requests: {ms.requests}\n"
                        f"Uptime: {int(time.time() - ms.loaded_at)}s"
                    )
                return Response(content=_twiml_reply(status_text), media_type="application/xml")

        # 4. Check model is loaded
        ms = get_state()
        if ms.model is None:
            return Response(
                content=_twiml_reply(
                    "The AI model is still loading. Please try again in a moment."
                ),
                media_type="application/xml",
            )

        # 5. Build prompt from conversation history
        messages = _get_or_create_session(from_number, system_prompt)
        with _sessions_lock:
            messages.append({"role": "user", "content": user_text})
            trimmed = _apply_max_history(messages)
            _sessions[from_number] = trimmed

        # Convert message list to a single prompt string using the tokenizer's
        # chat template (same approach as server.py chat_completions handler).
        tokenizer = get_tokenizer()
        try:
            prompt = tokenizer.apply_chat_template(
                trimmed,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback: simple concatenation
            prompt = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in trimmed
            ) + "\nASSISTANT:"

        # 6. Generate reply (synchronous, non-streaming — WhatsApp requires complete message)
        _generate = get_generate()
        reply_tokens: list[str] = []
        try:
            for tok_text, finish in _generate(
                prompt,
                max_tokens=512,
                temperature=0.7,
                top_p=0.9,
                stop=None,
                seed=None,
            ):
                if tok_text:
                    reply_tokens.append(tok_text)
                if finish is not None:
                    break
        except Exception as exc:
            return Response(
                content=_twiml_reply(f"Sorry, an error occurred: {exc}"),
                media_type="application/xml",
            )

        reply_text = "".join(reply_tokens).strip() or "I couldn't generate a response."

        # 7. Append assistant reply to session history
        with _sessions_lock:
            _sessions[from_number].append({"role": "assistant", "content": reply_text})
            _sessions_ts[from_number] = time.time()

        # 8. Return TwiML
        return Response(content=_twiml_reply(reply_text), media_type="application/xml")

    print("[squish] WhatsApp webhook mounted at /webhook/whatsapp", flush=True)
