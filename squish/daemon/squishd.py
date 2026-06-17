"""squish/daemon/squishd.py — persistent squish inference daemon.

Architecture
------------
The daemon is a long-running process that:
  1. Binds a Unix domain socket at /tmp/squish.sock (faster than TCP for local
     IPC; no port conflicts; no firewall rules needed).
  2. Holds up to MAX_MODELS models resident in memory, routed by model name.
  3. Accepts JSON-framed request/response messages over the socket (newline-
     delimited, each message a single JSON object).
  4. Exposes a /v1/chat/completions-compatible JSON API so the thin client
     in squish/daemon/client.py needs no special protocol knowledge.

Wire protocol (internal; clients use DaemonClient)
----------------------------------------------------
Request  → { "model": str, "messages": [...], "max_tokens": int, ... }
Response → { "text": str, "tokens": int, "tok_s": float, "finish": str }
  or     → { "error": str }
Control  → { "_cmd": "ping"|"reload"|"status" }
           Ping response: { "status": "ok", "models": [...], "pid": int }

The protocol is intentionally *not* HTTP — the goal is absolute minimal latency
on the hot path (no HTTP parser, no header overhead, one JSON frame = one
inference).  The companion client (DaemonClient) handles framing.

Multi-model support
-------------------
DaemonServer maintains an ordered dict of { model_key → _LoadedModel }.  When
a request arrives with a model name not in the cache AND the cache is full, the
LRU model is evicted (unloaded) to make room.  MAX_MODELS defaults to 2.

The "model already resident" path costs only the Python dict lookup + inference.
The "cold model load" path costs the usual model-load time; subsequent requests
for that model are hot.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SOCK_PATH: str = os.environ.get("SQUISH_SOCK", "/tmp/squish.sock")
PID_FILE:  str = str(Path.home() / ".squish" / "daemon.pid")
LOG_FILE:  str = str(Path.home() / ".squish" / "daemon.log")
MAX_MODELS: int = int(os.environ.get("SQUISH_MAX_MODELS", "2"))

# Maximum bytes per JSON frame (8 MB — large enough for any realistic prompt)
_MAX_FRAME = 8 * 1024 * 1024


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_running(sock_path: str = SOCK_PATH) -> bool:
    """Return True if a squishd is listening at *sock_path*."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        s.connect(sock_path)
        _send_frame(s, {"_cmd": "ping"})
        data = _recv_frame(s)
        return bool(data and data.get("status") == "ok")
    except (OSError, ValueError, RuntimeError) as exc:
        logger.debug("Daemon ping probe failed: %s", exc)
        return False
    finally:
        try:
            s.close()
        except OSError as exc:
            logger.debug("Socket close failed: %s", exc)


def send_request(
    payload: dict[str, Any],
    sock_path: str = SOCK_PATH,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Send an inference request to the daemon; return the response dict.

    Raises ``ConnectionRefusedError`` if the daemon is not running.
    Raises ``RuntimeError`` on wire or inference errors.
    """
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(sock_path)
    except (FileNotFoundError, ConnectionRefusedError) as exc:
        raise ConnectionRefusedError(
            f"squishd not running at {sock_path} — "
            "start with: squishd start <model>"
        ) from exc
    try:
        _send_frame(s, payload)
        resp = _recv_frame(s)
        if resp is None:
            raise RuntimeError("daemon closed connection without response")
        if "error" in resp:
            raise RuntimeError(f"daemon error: {resp['error']}")
        return resp
    finally:
        try:
            s.close()
        except OSError as exc:
            logger.debug("Socket close failed: %s", exc)


def _send_frame(sock: socket.socket, obj: dict) -> None:
    """Encode *obj* as JSON and write a length-prefixed frame."""
    raw = json.dumps(obj).encode()
    if len(raw) > _MAX_FRAME:
        raise ValueError(f"frame too large: {len(raw)} bytes")
    # 4-byte big-endian length prefix
    header = len(raw).to_bytes(4, "big")
    sock.sendall(header + raw)


def _recv_frame(sock: socket.socket) -> dict[str, Any] | None:
    """Read one length-prefixed frame; return parsed dict or None on EOF."""
    header = _recv_exact(sock, 4)
    if header is None:
        return None
    length = int.from_bytes(header, "big")
    if length > _MAX_FRAME:
        raise RuntimeError(f"frame length {length} exceeds limit")
    data = _recv_exact(sock, length)
    if data is None:
        return None
    return json.loads(data.decode())


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    """Read exactly *n* bytes; return None on clean EOF, raise on error."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            if len(buf) == 0:
                return None
            raise RuntimeError("connection closed mid-frame")
        buf.extend(chunk)
    return bytes(buf)


# ── Loaded model record ────────────────────────────────────────────────────────

class _LoadedModel:
    """One resident model + tokenizer.  Created by DaemonServer._load_model."""

    def __init__(self, key: str, model_dir: str, compressed_dir: str = ""):
        self.key            = key
        self.model_dir      = model_dir
        self.compressed_dir = compressed_dir
        self.model          = None
        self.tokenizer      = None
        self.loaded_at      = 0.0
        self.last_used      = 0.0
        self.n_requests     = 0


# ── Daemon server ──────────────────────────────────────────────────────────────

class DaemonServer:
    """Unix-domain-socket inference daemon.

    Parameters
    ----------
    sock_path : str
        Path for the UDS socket.
    max_models : int
        Maximum number of models held resident simultaneously.  When a new
        model is requested and the cache is full, the LRU model is evicted.
    default_model_dir : str
        Model directory loaded at startup (optional).
    default_compressed_dir : str
        Compressed dir for the default model (optional).
    """

    def __init__(
        self,
        sock_path: str = SOCK_PATH,
        max_models: int = MAX_MODELS,
        default_model_dir: str = "",
        default_compressed_dir: str = "",
    ) -> None:
        self._sock_path         = sock_path
        self._max_models        = max(1, max_models)
        self._models: OrderedDict[str, _LoadedModel] = OrderedDict()
        self._lock              = threading.Lock()
        self._server_sock: socket.socket | None = None
        self._running           = False
        self._default_model_dir = default_model_dir
        self._default_comp_dir  = default_compressed_dir

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Bind the socket and serve requests (blocks until stop() is called)."""
        _ensure_dir(Path(PID_FILE).parent)
        # Remove stale socket if present
        try:
            os.unlink(self._sock_path)
        except FileNotFoundError:
            pass

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(self._sock_path)
        srv.listen(64)
        self._server_sock = srv
        self._running = True

        Path(PID_FILE).write_text(str(os.getpid()))
        logger.info("squishd listening at %s (pid=%d)", self._sock_path, os.getpid())

        # Pre-load default model if configured
        if self._default_model_dir:
            try:
                self._load_model(
                    _model_key(self._default_model_dir),
                    self._default_model_dir,
                    self._default_comp_dir,
                )
            except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
                logger.exception("Failed to pre-load default model %s: %s",
                                 self._default_model_dir, exc)

        try:
            while self._running:
                srv.settimeout(1.0)
                try:
                    conn, _ = srv.accept()
                except TimeoutError:
                    continue
                t = threading.Thread(
                    target=self._handle_connection,
                    args=(conn,),
                    daemon=True,
                )
                t.start()
        finally:
            srv.close()
            try:
                os.unlink(self._sock_path)
            except FileNotFoundError:
                pass
            try:
                Path(PID_FILE).unlink(missing_ok=True)
            except OSError as exc:
                logger.debug("PID file unlink failed: %s", exc)
            logger.info("squishd stopped")

    def stop(self) -> None:
        """Signal the server loop to exit."""
        self._running = False
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except OSError as exc:
                logger.debug("Server socket close failed: %s", exc)

    # ── Connection handler ────────────────────────────────────────────────────

    def _handle_connection(self, conn: socket.socket) -> None:
        conn.settimeout(120.0)
        try:
            req = _recv_frame(conn)
            if req is None:
                return
            resp = self._dispatch(req)
            _send_frame(conn, resp)
        except Exception:  # noqa: BLE001 — daemon boundary, must not crash
            logger.exception("Error handling connection")
            try:
                _send_frame(conn, {"error": "internal server error"})
            except OSError as exc:
                logger.debug("Failed to send error frame: %s", exc)
        finally:
            try:
                conn.close()
            except OSError as exc:
                logger.debug("Connection close failed: %s", exc)

    def _dispatch(self, req: dict[str, Any]) -> dict[str, Any]:
        # Control commands
        cmd = req.get("_cmd")
        if cmd == "ping":
            return self._cmd_ping()
        if cmd == "reload":
            return self._cmd_reload(req)
        if cmd == "status":
            return self._cmd_status()

        # Inference request
        return self._infer(req)

    # ── Control commands ──────────────────────────────────────────────────────

    def _cmd_ping(self) -> dict[str, Any]:
        with self._lock:
            models = list(self._models.keys())
        return {"status": "ok", "models": models, "pid": os.getpid()}

    def _cmd_reload(self, req: dict[str, Any]) -> dict[str, Any]:
        model_dir  = req.get("model_dir", self._default_model_dir)
        comp_dir   = req.get("compressed_dir", self._default_comp_dir)
        key        = _model_key(model_dir)
        try:
            with self._lock:
                self._models.pop(key, None)
            self._load_model(key, model_dir, comp_dir)
            return {"status": "ok", "reloaded": key}
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            logger.warning("Reload failed for %s: %s", key, exc)
            return {"error": str(exc)}

    def _cmd_status(self) -> dict[str, Any]:
        with self._lock:
            info = {
                k: {
                    "model_dir": m.model_dir,
                    "n_requests": m.n_requests,
                    "loaded_at": m.loaded_at,
                    "last_used": m.last_used,
                }
                for k, m in self._models.items()
            }
        return {"status": "ok", "pid": os.getpid(), "models": info}

    # ── Inference ─────────────────────────────────────────────────────────────

    def _infer(self, req: dict[str, Any]) -> dict[str, Any]:
        model_dir  = req.get("model_dir", self._default_model_dir)
        comp_dir   = req.get("compressed_dir", "")
        key        = req.get("model_key") or _model_key(model_dir)

        loaded = self._get_or_load(key, model_dir, comp_dir)
        if loaded is None:
            return {"error": f"model not available: {model_dir}"}

        messages   = req.get("messages", [])
        max_tokens = int(req.get("max_tokens", 512))
        temperature = float(req.get("temperature", 0.7))
        top_p      = float(req.get("top_p", 0.9))

        # Build prompt string from messages
        prompt = _messages_to_prompt(messages, loaded.tokenizer)

        t0 = time.perf_counter()
        try:
            text, n_tok = self._run_inference(
                loaded, prompt, max_tokens, temperature, top_p
            )
        except (RuntimeError, ValueError, TypeError, AttributeError, ImportError) as exc:
            logger.exception("Inference error for model %s: %s", key, exc)
            return {"error": str(exc)}
        elapsed  = time.perf_counter() - t0
        tok_s    = n_tok / elapsed if elapsed > 0 else 0.0

        with self._lock:
            loaded.n_requests += 1
            loaded.last_used   = time.time()

        return {
            "text":   text,
            "tokens": n_tok,
            "tok_s":  round(tok_s, 2),
            "finish": "stop",
        }

    def _run_inference(
        self,
        loaded: _LoadedModel,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[str, int]:
        """Run inference; return (text, token_count)."""
        # Use mlx_lm generate when available (lowest overhead, no HTTP roundtrip)
        try:
            from mlx_lm import generate as _mlx_generate
            # mlx_lm >= 0.21 replaced temp/top_p kwargs with a sampler callable.
            # Always use make_sampler when present; legacy fall-through below.
            _gen_kwargs: dict[str, Any] = {
                "prompt":     prompt,
                "max_tokens": max_tokens,
                "verbose":    False,
            }
            try:
                from mlx_lm.sample_utils import make_sampler as _ms
                _gen_kwargs["sampler"] = _ms(temp=temperature, top_p=top_p)
            except ImportError:
                _gen_kwargs["temp"]  = temperature
                _gen_kwargs["top_p"] = top_p
            result = _mlx_generate(loaded.model, loaded.tokenizer, **_gen_kwargs)
            text = result if isinstance(result, str) else result.get("text", "")
            n_tok = len(loaded.tokenizer.encode(text)) if hasattr(loaded.tokenizer, "encode") else 1
            return text, n_tok
        except ImportError:
            pass

        # Fallback: use SpeculativeGenerator plain stream
        from squish.speculative.speculative import SpeculativeGenerator
        gen = SpeculativeGenerator(loaded.model, loaded.tokenizer)
        parts: list[str] = []
        n_tok = 0
        for tok_text, _ in gen.stream(prompt, max_tokens=max_tokens,
                                       temperature=temperature, top_p=top_p):
            parts.append(tok_text)
            n_tok += 1
        return "".join(parts), n_tok

    # ── Model cache management ────────────────────────────────────────────────

    def _get_or_load(
        self,
        key: str,
        model_dir: str,
        compressed_dir: str,
    ) -> _LoadedModel | None:
        with self._lock:
            if key in self._models:
                self._models.move_to_end(key)  # LRU bump
                return self._models[key]

        # Load outside the lock (slow path)
        try:
            lm = self._load_model(key, model_dir, compressed_dir)
            return lm
        except (OSError, ValueError, RuntimeError, ImportError, KeyError) as exc:
            logger.exception("Failed to load model %s from %s: %s", key, model_dir, exc)
            return None

    def _load_model(
        self, key: str, model_dir: str, compressed_dir: str
    ) -> _LoadedModel:
        t0 = time.perf_counter()
        lm = _LoadedModel(key, model_dir, compressed_dir)

        # v4.1 Fix 5: detect mlx-native quantized models (config.json has a
        # "quantization" field) and load them via mlx_lm.load() directly.
        # The legacy compressed_loader path expects a `<model>-compressed/`
        # dir with manifest.json — that only exists for the squish npy-dir
        # format, so without this check squishd cannot load any mlx-native
        # quantized model (e.g. mlx-community Qwen2.5-7B-Instruct-int4).
        if _model_is_mlx_native_quant(model_dir):
            from mlx_lm import load as _mlx_load
            lm.model, lm.tokenizer = _mlx_load(model_dir)
        else:
            from squish.quant.compressed_loader import load_compressed_model
            comp = compressed_dir or (model_dir + "-compressed")
            lm.model, lm.tokenizer = load_compressed_model(
                model_dir=model_dir,
                npz_path=comp,
                verbose=False,
            )
        lm.loaded_at = time.time()
        lm.last_used = lm.loaded_at

        elapsed = time.perf_counter() - t0
        logger.info("loaded model %s in %.2f s", key, elapsed)

        with self._lock:
            # Evict LRU if at capacity
            while len(self._models) >= self._max_models:
                evict_key, _ = next(iter(self._models.items()))
                self._models.pop(evict_key)
                logger.info("evicted model %s (LRU)", evict_key)
            self._models[key] = lm

        return lm


# ── Utility helpers ────────────────────────────────────────────────────────────

def _model_key(model_dir: str) -> str:
    """Stable short key for a model directory (basename + hash prefix)."""
    name = Path(model_dir).name
    h    = hashlib.sha1(model_dir.encode()).hexdigest()[:8]
    return f"{name}:{h}"


def _model_is_mlx_native_quant(model_dir: str) -> bool:
    """Return True if config.json declares this is an mlx-native quantized model.

    mlx-community / mlx_lm.convert produce models with a ``quantization``
    field in config.json (e.g. Qwen2.5-7B-Instruct-int4).  These must be
    loaded via ``mlx_lm.load()`` — the squish ``compressed_loader`` only
    handles the npy-dir format and crashes looking for ``manifest.json``.
    """
    cfg = Path(model_dir) / "config.json"
    if not cfg.exists():
        return False
    try:
        with open(cfg) as f:
            return "quantization" in json.load(f)
    except (OSError, json.JSONDecodeError):
        return False


def _messages_to_prompt(messages: list[dict], tokenizer) -> str:
    """Convert OpenAI-style messages list to a flat prompt string."""
    if not messages:
        return ""
    # Try tokenizer's chat_template first
    try:
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    except Exception as exc:  # noqa: BLE001 — chat template is arbitrary Jinja; any failure must fall back
        logger.debug("chat_template apply failed, using fallback prompt: %s", exc)
    # Simple fallback: role-prefixed concatenation
    parts: list[str] = []
    for m in messages:
        role    = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<|{role}|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ── squishd CLI entry point ────────────────────────────────────────────────────

def main() -> None:  # pragma: no cover
    """Entry point for the ``squishd`` script."""
    import argparse
    import signal
    import sys

    ap = argparse.ArgumentParser(
        prog="squishd",
        description="Squish persistent inference daemon (Unix domain socket)",
    )
    sub = ap.add_subparsers(dest="action")

    p_start = sub.add_parser("start", help="Start the daemon")
    p_start.add_argument("model_dir", nargs="?", default="",
                         help="Model directory to pre-load at startup")
    p_start.add_argument("--compressed-dir", default="",
                         help="Compressed weights directory")
    p_start.add_argument("--max-models", type=int, default=MAX_MODELS,
                         help="Max resident models (default: %(default)s)")
    p_start.add_argument("--sock", default=SOCK_PATH,
                         help="Unix socket path (default: %(default)s)")
    p_start.add_argument("--foreground", action="store_true",
                         help="Run in the foreground (don't daemonise)")

    sub.add_parser("stop",   help="Stop the daemon")
    sub.add_parser("status", help="Show daemon status")
    sub.add_parser("reload", help="Reload the default model")

    args = ap.parse_args()
    action = args.action or "status"

    if action == "status":
        if is_running(SOCK_PATH):
            try:
                resp = send_request({"_cmd": "status"}, SOCK_PATH)
                models = resp.get("models", {})
                print(f"  ✓  squishd running  (pid {resp.get('pid', '?')})")
                for k, info in models.items():
                    print(f"     {k}: {info.get('n_requests', 0)} requests")
            except (OSError, RuntimeError, ValueError) as exc:
                logger.debug("Status query failed: %s", exc)
                print("  ✓  squishd running")
        else:
            print("  ✗  squishd not running  (start with: squishd start <model>)")
        return

    if action == "stop":
        pid_path = Path(PID_FILE)
        try:
            pid = int(pid_path.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            time.sleep(1)
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            pid_path.unlink(missing_ok=True)
            print(f"  ✓  squishd stopped  (pid {pid})")
        except FileNotFoundError:
            print("  squishd is not running (no pid file)")
        except ValueError:
            print("  squishd pid file is corrupt")
        return

    if action == "reload":
        if not is_running(SOCK_PATH):
            print("  squishd is not running")
            sys.exit(1)
        try:
            resp = send_request({"_cmd": "reload"}, SOCK_PATH)
            print(f"  ✓  Reloaded: {resp.get('reloaded', '?')}")
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"  ✗  Reload failed: {exc}")
            sys.exit(1)
        return

    if action == "start":
        if is_running(args.sock):
            print(f"  squishd already running at {args.sock}")
            return

        srv = DaemonServer(
            sock_path=args.sock,
            max_models=args.max_models,
            default_model_dir=args.model_dir,
            default_compressed_dir=args.compressed_dir,
        )

        if not args.foreground:
            # Daemonise: fork → detach → run
            pid = os.fork()
            if pid > 0:
                # Parent: wait briefly then verify
                time.sleep(1.0)
                if is_running(args.sock):
                    print(f"  ✓  squishd started  (pid {pid})")
                else:
                    print(f"  ⚠  squishd started (pid {pid}) but not yet ready")
                return
            # Child: detach from terminal
            os.setsid()

        # Set up graceful shutdown
        def _shutdown(signum, frame):
            logger.info("squishd received signal %d — shutting down", signum)
            srv.stop()
            sys.exit(0)

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT,  _shutdown)

        srv.start()


if __name__ == "__main__":  # pragma: no cover
    main()
