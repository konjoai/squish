"""
squish/serving/signal_cli.py — Signal integration via signal-cli JSON-RPC daemon.

Connects to a running signal-cli daemon over TCP or a UNIX socket, subscribes to
incoming messages, generates replies using the squish model, and sends them back.
Also mounts a /signal/status endpoint on the FastAPI app.

Prerequisites
─────────────
  1. Install signal-cli (≥ 0.12 recommended):
       https://github.com/AsamK/signal-cli/releases
       brew install signal-cli   # macOS via Homebrew (community tap)

  2. Register your phone number (one-time):
       signal-cli -a +YOUR_NUMBER register
       signal-cli -a +YOUR_NUMBER verify THE_CODE

  3. Start the signal-cli daemon in JSON-RPC socket mode (keeps running in background):
       # UNIX socket (recommended — stays local to the machine):
       signal-cli -a +YOUR_NUMBER jsonRpc \\
           --socket /tmp/signal-cli.sock

       # TCP — useful if signal-cli and squish run on different hosts:
       #   signal-cli doesn't expose TCP natively; bridge with socat:
       signal-cli -a +YOUR_NUMBER jsonRpc \\
           --socket /tmp/signal-cli.sock &
       socat TCP-LISTEN:7583,fork UNIX-CONNECT:/tmp/signal-cli.sock

  4. Start squish with the Signal integration:
       squish run 7b \\
           --signal \\
           --signal-account  +YOUR_NUMBER \\
           --signal-socket   127.0.0.1:7583   # or /tmp/signal-cli.sock

Everything runs on your own hardware — no third-party tunnel required.

Signal commands (sent as normal messages):
  /reset   — clear conversation history for this contact
  /status  — show model name, avg TPS, uptime
  /help    — list available commands
"""
from __future__ import annotations

import json
import socket
import threading
import time
from typing import Any

# ── Optional FastAPI (required when actually mounted) ────────────────────────
try:
    from fastapi.responses import JSONResponse as _JSONResponse
    _FASTAPI = True
except ImportError:  # pragma: no cover
    _FASTAPI = False
    _JSONResponse = None  # type: ignore[assignment,misc]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise AI assistant. "
    "Keep replies short and suitable for a mobile messaging app. "
    "Avoid markdown formatting unless the user asks for it."
)

# ── Session store ─────────────────────────────────────────────────────────────
# Keyed by sender phone number (E.164, e.g. "+15551234567").
_sessions:    dict[str, list[dict[str, str]]] = {}
_sessions_ts: dict[str, float]                 = {}
_sessions_lock = threading.Lock()
_MAX_HISTORY   = 20
_SESSION_TIMEOUT = 3600  # seconds


def _get_or_create_session(sender: str, system_prompt: str) -> list[dict[str, str]]:
    with _sessions_lock:
        now = time.time()
        if sender not in _sessions:
            _sessions[sender]    = [{"role": "system", "content": system_prompt}]
            _sessions_ts[sender] = now
        else:
            _sessions_ts[sender] = now
        return _sessions[sender]


def _reset_session(sender: str, system_prompt: str) -> None:
    with _sessions_lock:
        _sessions[sender]    = [{"role": "system", "content": system_prompt}]
        _sessions_ts[sender] = time.time()


def _expire_old_sessions() -> None:
    cutoff = time.time() - _SESSION_TIMEOUT
    with _sessions_lock:
        expired = [k for k, ts in _sessions_ts.items() if ts < cutoff]
        for k in expired:
            del _sessions[k]
            del _sessions_ts[k]


def _apply_max_history(
    msgs: list[dict[str, str]],
) -> list[dict[str, str]]:
    system_msgs  = [m for m in msgs if m["role"] == "system"]
    non_system   = [m for m in msgs if m["role"] != "system"]
    keep = _MAX_HISTORY - len(system_msgs)
    if keep < 0:
        keep = 0
    return system_msgs + non_system[-keep:]


# ── signal-cli JSON-RPC client ────────────────────────────────────────────────

class _SignalRPC:
    """
    Thin JSON-RPC-over-socket wrapper for signal-cli's daemon mode.

    Supports TCP (``host:port`` string) and UNIX sockets (absolute path).
    Each call is identified by a monotonically increasing integer id.
    Incoming unsolicited notifications (``method == "receive"``) are delivered
    to a registered callback.
    """

    def __init__(self, socket_addr: str) -> None:
        self._addr      = socket_addr
        self._sock: socket.socket | None = None
        self._lock      = threading.Lock()
        self._rx_lock   = threading.Lock()
        self._next_id   = 1
        self._callbacks: list[Any] = []

    def connect(self) -> None:
        with self._lock:
            if self._sock:
                return
            if self._addr.startswith("/") or self._addr.startswith("./"):
                # UNIX socket
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                sock.connect(self._addr)
            else:
                # TCP  host:port
                host, _, port_str = self._addr.rpartition(":")
                host = host or "127.0.0.1"
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect((host, int(port_str)))
            self._sock = sock
            self._file = sock.makefile("rb")

    def close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except OSError:
                    pass
                self._sock = None

    def on_receive(self, callback: Any) -> None:
        self._callbacks.append(callback)

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        with self._lock:
            if not self._sock:
                raise RuntimeError("Not connected")
            req_id = self._next_id
            self._next_id += 1
            payload = {"jsonrpc": "2.0", "method": method, "id": req_id}
            if params:
                payload["params"] = params
            raw = (json.dumps(payload) + "\n").encode()
            self._sock.sendall(raw)
        return req_id

    def send_message(self, account: str, recipient: str, message: str) -> None:
        self.call("send", {
            "account":   account,
            "recipient": [recipient],
            "message":   message,
        })

    def subscribe(self, account: str) -> None:
        self.call("subscribeReceive", {"account": account})

    def read_loop(self) -> None:
        """
        Blocking loop — reads newline-delimited JSON from the daemon and dispatches
        ``receive`` notifications to registered callbacks.  Exits on socket error.
        """
        try:
            for raw_line in self._file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("method") == "receive":
                    for cb in self._callbacks:
                        try:
                            cb(msg.get("params", {}))
                        except Exception:  # noqa: BLE001
                            pass
        except OSError:
            pass  # socket closed


# ── Main integration class ────────────────────────────────────────────────────

class _SignalBot:
    """
    Bridges signal-cli ↔ squish.

    Runs two background threads:
      • reader  — forwards signal-cli JSON-RPC notifications
      • watchdog — reconnects on disconnect, expires stale sessions
    """

    def __init__(
        self,
        socket_addr:   str,
        account:       str,
        get_state:     Any,
        get_generate:  Any,
        get_tokenizer: Any,
        system_prompt: str,
    ) -> None:
        self._addr          = socket_addr
        self._account       = account
        self._get_state     = get_state
        self._get_generate  = get_generate
        self._get_tokenizer = get_tokenizer
        self._system_prompt = system_prompt
        self._rpc: _SignalRPC | None = None
        self._running       = False
        self._reconnect_delay = 5   # seconds

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        t = threading.Thread(target=self._watchdog, daemon=True, name="squish-signal")
        t.start()

    def stop(self) -> None:
        self._running = False
        if self._rpc:
            self._rpc.close()

    def is_running(self) -> bool:
        return self._running

    # ── Internal ──────────────────────────────────────────────────────────────

    def _watchdog(self) -> None:
        while self._running:
            rpc = _SignalRPC(self._addr)
            try:
                rpc.connect()
                rpc.on_receive(self._handle_message)
                rpc.subscribe(self._account)
                self._rpc = rpc
                print(
                    f"[squish/signal] Connected to signal-cli at {self._addr} "
                    f"(account {self._account})",
                    flush=True,
                )
                rpc.read_loop()  # blocks until socket closes
            except Exception as exc:
                print(f"[squish/signal] Connection error: {exc}", flush=True)
            finally:
                rpc.close()
                self._rpc = None
            if self._running:
                print(
                    f"[squish/signal] Reconnecting in {self._reconnect_delay}s…",
                    flush=True,
                )
                time.sleep(self._reconnect_delay)

    def _handle_message(self, params: dict[str, Any]) -> None:
        _expire_old_sessions()

        envelope = params.get("envelope", {})
        sender: str = (
            envelope.get("sourceNumber")
            or envelope.get("source")
            or ""
        )
        data_msg: dict[str, Any] = envelope.get("dataMessage", {})
        text: str = (data_msg.get("message") or "").strip()

        if not sender or not text:
            return

        # ── Special commands ──────────────────────────────────────────────────
        cmd = text.lower()
        if cmd == "/reset":
            _reset_session(sender, self._system_prompt)
            self._send(sender, "Session cleared. Starting a fresh conversation.")
            return
        if cmd == "/help":
            self._send(sender,
                "Available commands:\n"
                "  /reset   — clear conversation history\n"
                "  /status  — server status\n"
                "  /help    — this message"
            )
            return
        if cmd == "/status":
            state = self._get_state()
            if state is None or getattr(state, "model", None) is None:
                self._send(sender, "Server is loading. Try again in a moment.")
            else:
                uptime_s = int(time.time() - getattr(state, "loaded_at", time.time()))
                self._send(sender,
                    f"Model: {getattr(state, 'model_name', '?')}\n"
                    f"Avg TPS: {getattr(state, 'avg_tps', 0):.1f}\n"
                    f"Requests: {getattr(state, 'requests', 0)}\n"
                    f"Uptime: {uptime_s // 60}m {uptime_s % 60}s"
                )
            return

        # ── Guard: model must be loaded ───────────────────────────────────────
        state = self._get_state()
        if state is None or getattr(state, "model", None) is None:
            self._send(sender,
                "The model is still loading. Please try again in a moment."
            )
            return

        # ── Build prompt ──────────────────────────────────────────────────────
        session = _get_or_create_session(sender, self._system_prompt)
        session.append({"role": "user", "content": text})
        trimmed = _apply_max_history(session)

        tokenizer = self._get_tokenizer()
        prompt: str = tokenizer.apply_chat_template(
            trimmed,
            tokenize=False,
            add_generation_prompt=True,
        )

        # ── Generate (synchronous, no streaming for Signal) ───────────────────
        _generate = self._get_generate()
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
            print(f"[squish/signal] Generation error for {sender}: {exc}", flush=True)
            self._send(sender, "Sorry, something went wrong generating a response.")
            return

        reply = "".join(reply_tokens).strip()
        if not reply:
            reply = "(no response generated)"

        session.append({"role": "assistant", "content": reply})
        self._send(sender, reply)

    def _send(self, recipient: str, message: str) -> None:
        if self._rpc is None:
            print(
                f"[squish/signal] Cannot send to {recipient} — not connected.",
                flush=True,
            )
            return
        try:
            self._rpc.send_message(self._account, recipient, message)
        except Exception as exc:
            print(f"[squish/signal] Send error: {exc}", flush=True)


# ── Module-level bot registry (at most one per process) ──────────────────────

_bot: _SignalBot | None = None


# ── Public API ────────────────────────────────────────────────────────────────

def mount_signal(
    app: Any,
    get_state:     Any,
    get_generate:  Any,
    get_tokenizer: Any,
    account:       str = "",
    socket_addr:   str = "127.0.0.1:7583",
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
) -> None:
    """
    Start a background Signal bot and register ``/signal/status`` on ``app``.

    Parameters
    ──────────
    app            FastAPI application instance.
    get_state      Zero-arg callable returning the server's ``ModelState``.
    get_generate   Zero-arg callable returning the ``_generate_tokens`` generator fn.
    get_tokenizer  Zero-arg callable returning the loaded tokenizer.
    account        E.164 phone number registered in signal-cli (``+1234567890``).
    socket_addr    ``host:port`` (e.g. ``127.0.0.1:7583``) or UNIX socket path
                   (absolute path, e.g. ``/tmp/signal-cli.sock``).
    system_prompt  System prompt injected at the start of every chat session.
    """
    global _bot

    if not account:
        print(
            "[squish/signal] --signal-account is required. "
            "Provide the E.164 phone number registered with signal-cli.",
            flush=True,
        )
        return

    _bot = _SignalBot(
        socket_addr   = socket_addr,
        account       = account,
        get_state     = get_state,
        get_generate  = get_generate,
        get_tokenizer = get_tokenizer,
        system_prompt = system_prompt,
    )
    _bot.start()

    print(
        f"[squish] Signal bot started — account {account}, "
        f"socket {socket_addr}",
        flush=True,
    )

    if not _FASTAPI:  # pragma: no cover
        return

    # ── GET /signal/status ────────────────────────────────────────────────────
    @app.get("/signal/status")
    async def signal_status() -> _JSONResponse:
        """Return Signal bot connection state."""
        return _JSONResponse({
            "running":      _bot.is_running() if _bot else False,
            "account":      account,
            "socket":       socket_addr,
            "sessions":     len(_sessions),
        })
