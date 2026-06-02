"""Unit tests for squish.daemon — squishd UDS daemon, client, and LaunchAgent.

All tests run without a live model and without starting the daemon process.
They test:
  - Wire protocol (frame encode/decode)
  - DaemonServer dispatch logic (mocked model)
  - DaemonClient interface
  - LaunchAgent plist generation
  - is_running() when daemon is not present
"""
from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _short_sock(label: str = "sq") -> str:
    """Return an AF_UNIX socket path short enough for macOS (limit: 104 chars).

    pytest's ``tmp_path`` lands under ``~/Library/Caches/...`` on macOS CI
    runners, easily blowing the 104-byte limit. ``/tmp`` keeps us under it.
    """
    return tempfile.mktemp(suffix=".sock", prefix=f"sqt_{label}_", dir="/tmp")


# ── Wire protocol helpers ──────────────────────────────────────────────────────

class TestFrameProtocol:
    def _encode(self, obj: dict) -> bytes:
        raw = json.dumps(obj).encode()
        return len(raw).to_bytes(4, "big") + raw

    def test_encode_decode_roundtrip(self):
        from squish.daemon.squishd import _send_frame, _recv_frame

        a, b = socket.socketpair()
        try:
            _send_frame(a, {"hello": "world", "n": 42})
            result = _recv_frame(b)
            assert result == {"hello": "world", "n": 42}
        finally:
            a.close()
            b.close()

    def test_ping_dict(self):
        from squish.daemon.squishd import _send_frame, _recv_frame

        a, b = socket.socketpair()
        try:
            _send_frame(a, {"_cmd": "ping"})
            result = _recv_frame(b)
            assert result["_cmd"] == "ping"
        finally:
            a.close()
            b.close()

    def test_recv_eof_returns_none(self):
        from squish.daemon.squishd import _recv_frame

        a, b = socket.socketpair()
        a.close()
        # Reading from b should get EOF → None
        result = _recv_frame(b)
        assert result is None
        b.close()

    def test_large_frame_rejected(self):
        from squish.daemon.squishd import _send_frame

        a, b = socket.socketpair()
        try:
            # Manually send a too-large header
            big_header = (9 * 1024 * 1024).to_bytes(4, "big")
            a.sendall(big_header)
            from squish.daemon.squishd import _recv_frame
            with pytest.raises(RuntimeError, match="exceeds limit"):
                _recv_frame(b)
        finally:
            a.close()
            b.close()

    def test_send_frame_too_large_raises(self):
        from squish.daemon.squishd import _send_frame

        a, _ = socket.socketpair()
        big_payload = {"data": "x" * (9 * 1024 * 1024)}
        with pytest.raises(ValueError, match="frame too large"):
            _send_frame(a, big_payload)
        a.close()

    def test_unicode_roundtrip(self):
        from squish.daemon.squishd import _send_frame, _recv_frame

        a, b = socket.socketpair()
        try:
            msg = {"text": "こんにちは 🦾 Ñoño"}
            _send_frame(a, msg)
            assert _recv_frame(b) == msg
        finally:
            a.close()
            b.close()


# ── is_running when no daemon ─────────────────────────────────────────────────

class TestIsRunning:
    def test_not_running_no_socket(self, tmp_path):
        from squish.daemon.squishd import is_running

        fake_path = str(tmp_path / "squish_test.sock")
        assert is_running(fake_path) is False

    def test_not_running_stale_socket(self):
        from squish.daemon.squishd import is_running

        # Create a socket path with no listener (short path for macOS AF_UNIX)
        p = _short_sock("stale")
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(p)
        s.close()
        try:
            assert is_running(p) is False
        finally:
            if os.path.exists(p):
                os.unlink(p)


# ── DaemonServer control commands ─────────────────────────────────────────────

class TestDaemonServerDispatch:
    """Test DaemonServer._dispatch without starting the actual server."""

    def _make_server(self):
        from squish.daemon.squishd import DaemonServer
        return DaemonServer(sock_path="/tmp/squish_test_never_bind.sock")

    def test_ping_returns_ok(self):
        srv = self._make_server()
        resp = srv._dispatch({"_cmd": "ping"})
        assert resp["status"] == "ok"
        assert isinstance(resp["models"], list)
        assert isinstance(resp["pid"], int)

    def test_status_returns_ok(self):
        srv = self._make_server()
        resp = srv._dispatch({"_cmd": "status"})
        assert resp["status"] == "ok"
        assert "models" in resp

    def test_unknown_model_dir_returns_error(self):
        srv = self._make_server()
        resp = srv._dispatch({
            "messages": [{"role": "user", "content": "hello"}],
            "model_dir": "/nonexistent/model/dir",
            "max_tokens": 5,
        })
        assert "error" in resp

    def test_ping_pid_is_current_process(self):
        srv = self._make_server()
        resp = srv._dispatch({"_cmd": "ping"})
        assert resp["pid"] == os.getpid()


# ── Model key generation ───────────────────────────────────────────────────────

class TestModelKey:
    def test_same_dir_same_key(self):
        from squish.daemon.squishd import _model_key

        k1 = _model_key("/models/Qwen2.5-7B-Instruct")
        k2 = _model_key("/models/Qwen2.5-7B-Instruct")
        assert k1 == k2

    def test_different_dirs_different_keys(self):
        from squish.daemon.squishd import _model_key

        k1 = _model_key("/models/Qwen2.5-7B-Instruct")
        k2 = _model_key("/models/Qwen2.5-1.5B-Instruct")
        assert k1 != k2

    def test_key_contains_basename(self):
        from squish.daemon.squishd import _model_key

        k = _model_key("/models/Qwen2.5-7B-Instruct")
        assert "Qwen2.5-7B-Instruct" in k

    def test_key_has_hash_suffix(self):
        from squish.daemon.squishd import _model_key

        k = _model_key("/models/Qwen2.5-7B-Instruct")
        # Format: "basename:8hexchars"
        parts = k.split(":")
        assert len(parts) == 2
        assert len(parts[1]) == 8


# ── Messages → prompt conversion ──────────────────────────────────────────────

class TestMessagesToPrompt:
    def test_empty_messages(self):
        from squish.daemon.squishd import _messages_to_prompt

        result = _messages_to_prompt([], tokenizer=None)
        assert result == ""

    def test_chat_template_preferred(self):
        from squish.daemon.squishd import _messages_to_prompt

        tok = MagicMock()
        tok.apply_chat_template.return_value = "<CHAT>Hello</CHAT>"
        messages = [{"role": "user", "content": "Hello"}]
        result = _messages_to_prompt(messages, tok)
        assert result == "<CHAT>Hello</CHAT>"
        tok.apply_chat_template.assert_called_once()

    def test_fallback_when_no_template(self):
        from squish.daemon.squishd import _messages_to_prompt

        tok = MagicMock()
        tok.apply_chat_template.side_effect = Exception("no template")
        messages = [{"role": "user", "content": "Hello"}]
        result = _messages_to_prompt(messages, tok)
        assert "user" in result
        assert "Hello" in result
        assert "assistant" in result

    def test_fallback_with_none_tokenizer(self):
        from squish.daemon.squishd import _messages_to_prompt

        messages = [{"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Explain Python."}]
        result = _messages_to_prompt(messages, tokenizer=None)
        assert "system" in result
        assert "You are helpful." in result


# ── DaemonClient interface ─────────────────────────────────────────────────────

class TestDaemonClient:
    def test_not_available_when_no_daemon(self, tmp_path):
        from squish.daemon.client import DaemonClient

        c = DaemonClient(sock_path=str(tmp_path / "nope.sock"))
        assert c.available() is False

    def test_chat_raises_when_unavailable(self):
        from squish.daemon.client import DaemonClient

        c = DaemonClient(sock_path=_short_sock("nope"))
        with pytest.raises(ConnectionRefusedError):
            c.chat([{"role": "user", "content": "hi"}])

    def test_complete_raises_when_unavailable(self):
        from squish.daemon.client import DaemonClient

        c = DaemonClient(sock_path=_short_sock("nope"))
        with pytest.raises(ConnectionRefusedError):
            c.complete("Hello")

    def test_ttft_raises_when_unavailable(self):
        from squish.daemon.client import DaemonClient

        c = DaemonClient(sock_path=_short_sock("nope"))
        with pytest.raises(ConnectionRefusedError):
            c.ttft([{"role": "user", "content": "hi"}])

    def test_ping_raises_when_unavailable(self):
        from squish.daemon.client import DaemonClient

        c = DaemonClient(sock_path=_short_sock("nope"))
        with pytest.raises(ConnectionRefusedError):
            c.ping()

    def test_default_sock_path(self):
        from squish.daemon.client import DaemonClient
        from squish.daemon.squishd import SOCK_PATH

        c = DaemonClient()
        assert c._sock_path == SOCK_PATH


# ── Live server round-trip (with mock model) ───────────────────────────────────

class TestLiveSocketRoundtrip:
    """Start a minimal DaemonServer in a thread; test round-trips without a model."""

    def _run_server(self, srv):
        try:
            srv.start()
        except Exception:
            pass

    def _wait_ready(self, sock: str, timeout: float = 5.0) -> bool:
        """Poll until the daemon is reachable or timeout expires."""
        from squish.daemon.squishd import is_running
        deadline = time.time() + timeout
        while time.time() < deadline:
            if is_running(sock):
                return True
            time.sleep(0.05)
        return False

    def test_ping_roundtrip(self):
        from squish.daemon.squishd import DaemonServer, send_request

        sock = _short_sock("ping")
        srv  = DaemonServer(sock_path=sock)
        t    = threading.Thread(target=self._run_server, args=(srv,), daemon=True)
        t.start()
        assert self._wait_ready(sock), "daemon did not start in time"

        try:
            resp = send_request({"_cmd": "ping"}, sock)
            assert resp["status"] == "ok"
            assert isinstance(resp["models"], list)
        finally:
            srv.stop()
            t.join(timeout=2.0)

    def test_status_roundtrip(self):
        from squish.daemon.squishd import DaemonServer, send_request

        sock = _short_sock("stat")
        srv  = DaemonServer(sock_path=sock)
        t    = threading.Thread(target=self._run_server, args=(srv,), daemon=True)
        t.start()
        assert self._wait_ready(sock), "daemon did not start in time"

        try:
            resp = send_request({"_cmd": "status"}, sock)
            assert resp["status"] == "ok"
        finally:
            srv.stop()
            t.join(timeout=2.0)

    def test_unknown_model_returns_error_frame(self):
        from squish.daemon.squishd import DaemonServer, send_request

        sock = _short_sock("unkn")
        srv  = DaemonServer(sock_path=sock)
        t    = threading.Thread(target=self._run_server, args=(srv,), daemon=True)
        t.start()
        assert self._wait_ready(sock), "daemon did not start in time"

        try:
            with pytest.raises(RuntimeError, match="daemon error"):
                send_request({
                    "messages": [{"role": "user", "content": "hi"}],
                    "model_dir": "/nonexistent/path/to/model",
                }, sock, timeout=5.0)
        finally:
            srv.stop()
            t.join(timeout=2.0)

    def test_multiple_sequential_pings(self):
        from squish.daemon.squishd import DaemonServer, send_request

        sock = _short_sock("mult")
        srv  = DaemonServer(sock_path=sock)
        t    = threading.Thread(target=self._run_server, args=(srv,), daemon=True)
        t.start()
        assert self._wait_ready(sock), "daemon did not start in time"

        try:
            for _ in range(5):
                resp = send_request({"_cmd": "ping"}, sock)
                assert resp["status"] == "ok"
        finally:
            srv.stop()
            t.join(timeout=2.0)


# ── LaunchAgent plist generation ──────────────────────────────────────────────

class TestLaunchAgentPlist:
    def test_plist_contains_label(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(squishd_bin="/usr/local/bin/squishd")
        assert "ai.konjo.squishd" in xml

    def test_plist_contains_bin(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(squishd_bin="/usr/local/bin/squishd")
        assert "/usr/local/bin/squishd" in xml

    def test_plist_contains_model_dir(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(
            squishd_bin="/usr/local/bin/squishd",
            model_dir="/models/Qwen2.5-7B",
        )
        assert "/models/Qwen2.5-7B" in xml

    def test_plist_contains_run_at_load(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(squishd_bin="/usr/local/bin/squishd")
        assert "RunAtLoad" in xml
        assert "<true/>" in xml

    def test_plist_contains_keep_alive(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(squishd_bin="/usr/local/bin/squishd")
        assert "KeepAlive" in xml

    def test_plist_contains_log_path(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(
            squishd_bin="/usr/local/bin/squishd",
            log_path="/tmp/squish_test.log",
        )
        assert "/tmp/squish_test.log" in xml

    def test_plist_env_vars(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(
            squishd_bin="/usr/local/bin/squishd",
            sock_path="/tmp/mysock.sock",
            max_models=3,
        )
        assert "SQUISH_SOCK" in xml
        assert "/tmp/mysock.sock" in xml
        assert "SQUISH_MAX_MODELS" in xml

    def test_is_installed_false_by_default(self, tmp_path, monkeypatch):
        from squish.daemon import launchagent as la

        monkeypatch.setattr(la, "PLIST_PATH", tmp_path / "test.plist")
        assert la.is_installed() is False

    def test_non_macos_install_raises(self, monkeypatch):
        from squish.daemon import launchagent as la

        monkeypatch.setattr(la, "_is_macos", lambda: False)
        with pytest.raises(RuntimeError, match="macOS-only"):
            la.install()

    def test_non_macos_uninstall_raises(self, monkeypatch):
        from squish.daemon import launchagent as la

        monkeypatch.setattr(la, "_is_macos", lambda: False)
        with pytest.raises(RuntimeError, match="macOS-only"):
            la.uninstall()

    def test_plist_xml_is_valid_structure(self):
        from squish.daemon.launchagent import plist_content

        xml = plist_content(squishd_bin="/usr/bin/squishd")
        assert xml.startswith("<?xml")
        assert "<plist" in xml
        assert "</plist>" in xml
        assert "<dict>" in xml
        assert "<array>" in xml
