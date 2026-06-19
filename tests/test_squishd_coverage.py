"""Behavioral coverage for the wire-error handlers, server lifecycle,
command dispatch, inference orchestration, and model-cache management of
``squish.daemon.squishd`` left untested by the baseline suite.

The model loaders and inference engine are injected as fakes (fake ``mlx_lm``,
patched ``load_compressed_model`` / ``SpeculativeGenerator``) so every path is
exercised without real weights — host-agnostic, no MLX required.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import types
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.daemon import squishd
from squish.daemon.squishd import (
    DaemonServer,
    _LoadedModel,
    _model_is_mlx_native_quant,
    _recv_frame,
    is_running,
    send_request,
)


def _short_sock(label: str = "cov") -> str:
    return f"/tmp/sqd_{label}_{uuid.uuid4().hex[:8]}.sock"


# ── Wire-protocol error handlers ────────────────────────────────────────────


def test_is_running_swallows_close_error(tmp_path):
    sock = str(tmp_path / "nope.sock")  # no daemon → connect fails
    with patch.object(socket.socket, "close", side_effect=OSError("close boom")):
        # connect fails → False; the finally close() error is swallowed (81-82).
        assert is_running(sock) is False


def test_recv_frame_none_when_body_truncated_to_eof():
    a, b = socket.socketpair()
    try:
        b.sendall((10).to_bytes(4, "big"))  # header promises 10 bytes
        b.close()                            # ...but none arrive → clean EOF
        assert _recv_frame(a) is None        # data is None (139)
    finally:
        a.close()


def test_recv_frame_raises_on_mid_frame_close():
    a, b = socket.socketpair()
    try:
        b.sendall((10).to_bytes(4, "big") + b"abc")  # 3 of 10 bytes, then close
        b.close()
        with pytest.raises(RuntimeError, match="closed mid-frame"):
            _recv_frame(a)  # _recv_exact sees partial buffer (151)
    finally:
        a.close()


def _accept_then(handler, sock_path):
    """Start a one-shot UDS server that runs handler(conn) and return its thread."""
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)

    def run():
        conn, _ = srv.accept()
        try:
            handler(conn)
        finally:
            conn.close()
            srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_send_request_raises_when_daemon_closes_silently():
    sock = _short_sock("silent")
    t = _accept_then(lambda conn: conn.recv(64), sock)  # read, never reply
    time.sleep(0.05)
    with pytest.raises(RuntimeError, match="closed connection without response"):
        send_request({"_cmd": "ping"}, sock)  # resp is None (108)
    t.join(timeout=2.0)


def test_send_request_swallows_close_error(monkeypatch):
    body = json.dumps({"status": "ok"}).encode()
    framed = [len(body).to_bytes(4, "big"), body]  # header, then body
    fake = MagicMock()
    fake.recv.side_effect = lambda n: framed.pop(0) if framed else b""
    fake.close.side_effect = OSError("close boom")
    monkeypatch.setattr(squishd.socket, "socket", lambda *a, **k: fake)
    # Successful roundtrip, but the finally close() raises → swallowed (115-116).
    resp = send_request({"_cmd": "ping"}, "/ignored")
    assert resp["status"] == "ok"


# ── _model_is_mlx_native_quant ──────────────────────────────────────────────


def test_mlx_native_quant_detection(tmp_path):
    # No config.json → False (504-505).
    assert _model_is_mlx_native_quant(str(tmp_path)) is False
    # config.json with quantization → True.
    (tmp_path / "config.json").write_text(json.dumps({"quantization": {"bits": 4}}))
    assert _model_is_mlx_native_quant(str(tmp_path)) is True


def test_mlx_native_quant_without_field(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 4096}))
    assert _model_is_mlx_native_quant(str(tmp_path)) is False


def test_mlx_native_quant_corrupt_config(tmp_path):
    (tmp_path / "config.json").write_text("{ not json")
    assert _model_is_mlx_native_quant(str(tmp_path)) is False  # JSONDecodeError (509-510)


# ── Command dispatch (direct, no socket) ────────────────────────────────────


def _server():
    return DaemonServer(sock_path=_short_sock())


def test_dispatch_routes_reload():
    srv = _server()
    with patch.object(srv, "_load_model") as ml:
        resp = srv._dispatch({"_cmd": "reload", "model_dir": "/m"})
    assert resp["status"] == "ok" and "reloaded" in resp
    ml.assert_called_once()


def test_cmd_reload_failure_returns_error():
    srv = _server()
    with patch.object(srv, "_load_model", side_effect=RuntimeError("load fail")):
        resp = srv._cmd_reload({"model_dir": "/m"})
    assert "error" in resp and "load fail" in resp["error"]


# ── _infer orchestration ────────────────────────────────────────────────────


def test_infer_returns_error_when_model_unavailable():
    srv = _server()
    with patch.object(srv, "_get_or_load", return_value=None):
        resp = srv._infer({"model_dir": "/missing"})
    assert "error" in resp and "not available" in resp["error"]


def test_infer_success_updates_stats():
    srv = _server()
    loaded = _LoadedModel("k", "/m")
    loaded.tokenizer = MagicMock()
    with patch.object(srv, "_get_or_load", return_value=loaded), \
         patch.object(srv, "_run_inference", return_value=("hello world", 2)), \
         patch.object(squishd, "_messages_to_prompt", return_value="p"):
        resp = srv._infer({"messages": [{"role": "user", "content": "hi"}]})
    assert resp["text"] == "hello world" and resp["tokens"] == 2
    assert resp["finish"] == "stop" and loaded.n_requests == 1


def test_infer_handles_inference_exception():
    srv = _server()
    loaded = _LoadedModel("k", "/m")
    loaded.tokenizer = MagicMock()
    with patch.object(srv, "_get_or_load", return_value=loaded), \
         patch.object(srv, "_run_inference", side_effect=RuntimeError("boom")), \
         patch.object(squishd, "_messages_to_prompt", return_value="p"):
        resp = srv._infer({"messages": []})
    assert "error" in resp and "boom" in resp["error"]


# ── _run_inference (fake mlx_lm + fallback) ─────────────────────────────────


def _install_fake_mlx_lm(monkeypatch, generate_ret="generated text", with_sampler=True):
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.generate = lambda model, tok, **kw: generate_ret
    mlx_lm.load = lambda path: (MagicMock(name="model"), MagicMock(name="tok"))
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    if with_sampler:
        su = types.ModuleType("mlx_lm.sample_utils")
        su.make_sampler = lambda temp, top_p: MagicMock(name="sampler")
        monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", su)
    else:
        # Make `from mlx_lm.sample_utils import make_sampler` raise ImportError.
        monkeypatch.setitem(sys.modules, "mlx_lm.sample_utils", None)
    return mlx_lm


def test_run_inference_uses_mlx_lm_with_sampler(monkeypatch):
    _install_fake_mlx_lm(monkeypatch, generate_ret="abc")
    srv = _server()
    loaded = _LoadedModel("k", "/m")
    loaded.model = MagicMock()
    loaded.tokenizer = MagicMock()
    loaded.tokenizer.encode.return_value = [1, 2, 3]
    text, n_tok = srv._run_inference(loaded, "prompt", 16, 0.7, 0.9)
    assert text == "abc" and n_tok == 3


def test_run_inference_legacy_kwargs_when_no_sampler(monkeypatch):
    _install_fake_mlx_lm(monkeypatch, generate_ret={"text": "dict-result"}, with_sampler=False)
    srv = _server()
    loaded = _LoadedModel("k", "/m")
    loaded.model = MagicMock()
    loaded.tokenizer = MagicMock()
    loaded.tokenizer.encode.return_value = [9]
    # generate returns a dict → text extracted via .get; sampler import failed.
    text, n_tok = srv._run_inference(loaded, "prompt", 16, 0.7, 0.9)
    assert text == "dict-result" and n_tok == 1


def test_run_inference_falls_back_to_speculative(monkeypatch):
    # mlx_lm import fails → SpeculativeGenerator fallback path.
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    fake_spec = types.ModuleType("squish.speculative.speculative")

    class _Gen:
        def __init__(self, model, tok):
            pass

        def stream(self, prompt, max_tokens, temperature, top_p):
            yield ("foo", None)
            yield ("bar", None)

    fake_spec.SpeculativeGenerator = _Gen
    monkeypatch.setitem(sys.modules, "squish.speculative.speculative", fake_spec)
    srv = _server()
    loaded = _LoadedModel("k", "/m")
    loaded.model = MagicMock()
    loaded.tokenizer = MagicMock()
    text, n_tok = srv._run_inference(loaded, "prompt", 16, 0.7, 0.9)
    assert text == "foobar" and n_tok == 2


# ── _get_or_load + _load_model cache management ─────────────────────────────


def test_get_or_load_lru_hit():
    srv = _server()
    existing = _LoadedModel("k", "/m")
    srv._models["k"] = existing
    # Cache hit returns the resident model and bumps LRU (434-436).
    assert srv._get_or_load("k", "/m", "") is existing


def test_get_or_load_failure_returns_none():
    srv = _server()
    with patch.object(srv, "_load_model", side_effect=RuntimeError("nope")):
        assert srv._get_or_load("k", "/m", "") is None  # 442-444


def test_load_model_mlx_native(monkeypatch, tmp_path):
    fake = MagicMock(), MagicMock()
    mlx_lm = types.ModuleType("mlx_lm")
    mlx_lm.load = lambda path: fake
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    srv = _server()
    with patch.object(squishd, "_model_is_mlx_native_quant", return_value=True):
        lm = srv._load_model("k", str(tmp_path), "")
    assert lm.model is fake[0] and lm.tokenizer is fake[1]
    assert srv._models["k"] is lm and lm.loaded_at > 0


def test_load_model_compressed_path_and_lru_eviction(monkeypatch):
    srv = DaemonServer(sock_path=_short_sock(), max_models=1)
    srv._models["old"] = _LoadedModel("old", "/old")  # fill cache to capacity

    def fake_loader(model_dir, npz_path, verbose):
        return MagicMock(name="m"), MagicMock(name="t")

    fake_cl = types.ModuleType("squish.quant.compressed_loader")
    fake_cl.load_compressed_model = fake_loader
    monkeypatch.setitem(sys.modules, "squish.quant.compressed_loader", fake_cl)
    with patch.object(squishd, "_model_is_mlx_native_quant", return_value=False):
        srv._load_model("new", "/new", "")
    # LRU eviction removed "old"; only "new" remains (475-483).
    assert "old" not in srv._models and "new" in srv._models


# ── Live server: preload, handle_connection, lifecycle ──────────────────────


def _wait_ready(sock, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_running(sock):
            return True
        time.sleep(0.05)
    return False


def _run_server(srv):
    """Run start() swallowing the EBADF the accept loop raises when stop()
    closes the socket mid-accept (matches the existing live-server tests)."""
    try:
        srv.start()
    except Exception:  # noqa: BLE001 - test helper, server teardown is racy
        pass


def test_start_preloads_default_model_and_handles_bad_frame(monkeypatch, tmp_path):
    sock = _short_sock("live")
    monkeypatch.setattr(squishd, "PID_FILE", str(tmp_path / "squishd.pid"))
    srv = DaemonServer(sock_path=sock, default_model_dir="/default")
    preloaded = {"called": False}

    def fake_load(key, model_dir, comp):
        preloaded["called"] = True
        lm = _LoadedModel(key, model_dir)
        srv._models[key] = lm
        return lm

    monkeypatch.setattr(srv, "_load_model", fake_load)
    t = threading.Thread(target=_run_server, args=(srv,), daemon=True)
    t.start()
    try:
        assert _wait_ready(sock), "daemon did not start"
        assert preloaded["called"], "default model was not preloaded"
        # Send a non-dict frame → _dispatch raises → error frame returned (283-288).
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.connect(sock)
        squishd._send_frame(c, [1, 2, 3])  # a list, not a dict
        resp = squishd._recv_frame(c)
        c.close()
        assert resp == {"error": "internal server error"}
        # A connection that closes before sending → handler returns early (280).
        c2 = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c2.connect(sock)
        c2.close()
        time.sleep(0.05)
    finally:
        srv.stop()
        t.join(timeout=3.0)
    assert not Path(sock).exists()  # start() finally unlinked the socket (255-256)


def test_start_preload_failure_is_logged(monkeypatch, tmp_path):
    sock = _short_sock("preloadfail")
    monkeypatch.setattr(squishd, "PID_FILE", str(tmp_path / "squishd.pid"))
    srv = DaemonServer(sock_path=sock, default_model_dir="/bad")
    monkeypatch.setattr(srv, "_load_model", MagicMock(side_effect=RuntimeError("preload boom")))
    t = threading.Thread(target=_run_server, args=(srv,), daemon=True)
    t.start()
    try:
        assert _wait_ready(sock), "daemon did not start despite preload failure"
        # Server is up and answers despite the preload exception (235-237).
        assert send_request({"_cmd": "ping"}, sock)["status"] == "ok"
    finally:
        srv.stop()
        t.join(timeout=3.0)


def test_stop_swallows_server_socket_close_error():
    srv = _server()
    fake_sock = MagicMock()
    fake_sock.close.side_effect = OSError("close boom")
    srv._server_sock = fake_sock
    srv.stop()  # close error swallowed (270-271)
    assert srv._running is False


def test_stop_when_no_server_socket():
    srv = _server()  # _server_sock is None by default
    srv.stop()  # the `if self._server_sock is not None` is False (267->exit)
    assert srv._running is False


def test_get_or_load_loads_on_cache_miss():
    srv = _server()
    lm = _LoadedModel("k", "/m")
    with patch.object(srv, "_load_model", return_value=lm):
        # Not in cache → load path returns the new model (440-441).
        assert srv._get_or_load("k", "/m", "") is lm


# ── _handle_connection (direct, fake conn) ──────────────────────────────────


def _fake_conn(frame_obj=None):
    """Fake socket whose recv() yields one length-prefixed frame, then EOF."""
    conn = MagicMock()
    if frame_obj is None:
        conn.recv.return_value = b""  # immediate EOF
    else:
        body = json.dumps(frame_obj).encode()
        chunks = [len(body).to_bytes(4, "big"), body]
        conn.recv.side_effect = lambda n: chunks.pop(0) if chunks else b""
    return conn


def test_handle_connection_empty_request_returns_early():
    srv = _server()
    conn = _fake_conn(None)  # EOF → _recv_frame None → early return (280)
    srv._handle_connection(conn)
    conn.sendall.assert_not_called()


def test_handle_connection_send_error_frame_failure():
    srv = _server()
    conn = _fake_conn([1, 2, 3])  # a list → _dispatch raises AttributeError
    conn.sendall.side_effect = OSError("send fail")  # error-frame send also fails
    srv._handle_connection(conn)  # both excepts swallowed (283-288)
    conn.close.assert_called()


def test_handle_connection_close_failure_is_swallowed():
    srv = _server()
    conn = _fake_conn({"_cmd": "ping"})  # valid → dispatch ok, response sent
    conn.close.side_effect = OSError("close fail")
    srv._handle_connection(conn)  # finally close() error swallowed (292-293)


def test_start_finally_tolerates_missing_socket_and_pid_unlink_error(monkeypatch, tmp_path):
    import os as _os
    sock = _short_sock("finally")
    monkeypatch.setattr(squishd, "PID_FILE", str(tmp_path / "squishd.pid"))
    srv = DaemonServer(sock_path=sock)
    t = threading.Thread(target=_run_server, args=(srv,), daemon=True)
    t.start()
    try:
        assert _wait_ready(sock), "daemon did not start"
        # Remove the socket so start()'s finally os.unlink raises
        # FileNotFoundError (256-257).
        _os.unlink(sock)
        # Make the PID-file unlink raise OSError (260-261).
        monkeypatch.setattr(
            Path, "unlink",
            lambda self, *a, **k: (_ for _ in ()).throw(OSError("pid unlink boom")),
        )
    finally:
        srv.stop()
        t.join(timeout=3.0)
    assert srv._running is False
