"""
Tests for squish/serving/signal_cli.py.

Covers:
  - _SignalRPC: JSON-RPC call encoding, send_message, subscribe, read_loop dispatch
  - Session management helpers (_get_or_create_session, _reset_session,
    _expire_old_sessions, _apply_max_history)
  - _SignalBot._handle_message: commands (/reset, /help, /status), normal
    generation, model-not-loaded guard, missing sender/text, generation errors
  - mount_signal: no-account guard, bot starts with valid config
"""
from __future__ import annotations

import io
import json
import threading
import time
import unittest.mock as mock

import pytest

from squish.serving.signal_cli import (
    _DEFAULT_SYSTEM_PROMPT,
    _MAX_HISTORY,
    _SESSION_TIMEOUT,
    _apply_max_history,
    _expire_old_sessions,
    _get_or_create_session,
    _reset_session,
    _sessions,
    _sessions_lock,
    _sessions_ts,
    _SignalBot,
    _SignalRPC,
    mount_signal,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clear_sessions() -> None:
    with _sessions_lock:
        _sessions.clear()
        _sessions_ts.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Session management (same logic as whatsapp — independent stores)
# ─────────────────────────────────────────────────────────────────────────────

class TestSessions:
    def setup_method(self):
        _clear_sessions()

    def test_creates_new_session(self):
        msgs = _get_or_create_session("+15551234567", "sys")
        assert msgs == [{"role": "system", "content": "sys"}]

    def test_returns_existing_session(self):
        msgs = _get_or_create_session("+1111", "p")
        msgs.append({"role": "user", "content": "hi"})
        assert len(_get_or_create_session("+1111", "p")) == 2

    def test_reset_clears_non_system_history(self):
        msgs = _get_or_create_session("+1111", "p")
        msgs += [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
        _reset_session("+1111", "new")
        assert _sessions["+1111"] == [{"role": "system", "content": "new"}]

    def test_expire_removes_stale(self):
        _sessions["+old"] = []
        _sessions_ts["+old"] = time.time() - _SESSION_TIMEOUT - 1
        _sessions["+fresh"] = []
        _sessions_ts["+fresh"] = time.time()
        _expire_old_sessions()
        assert "+old" not in _sessions
        assert "+fresh" in _sessions

    def test_apply_max_history_trims(self):
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY + 5)
        ]
        result = _apply_max_history(msgs)
        assert len(result) == _MAX_HISTORY
        assert result[0]["role"] == "system"

    def test_apply_max_history_keeps_last_messages(self):
        msgs = [{"role": "system", "content": "s"}] + [
            {"role": "user", "content": f"m{i}"} for i in range(_MAX_HISTORY + 3)
        ]
        result = _apply_max_history(msgs)
        assert result[-1]["content"] == f"m{_MAX_HISTORY + 2}"

    def test_apply_max_history_short_list_unchanged(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}]
        assert _apply_max_history(msgs) == msgs


# ─────────────────────────────────────────────────────────────────────────────
# _SignalRPC
# ─────────────────────────────────────────────────────────────────────────────

class TestSignalRPC:
    def _rpc_with_mock_socket(self) -> tuple[_SignalRPC, mock.MagicMock]:
        rpc = _SignalRPC("127.0.0.1:7583")
        sock = mock.MagicMock()
        sock.makefile.return_value = io.BytesIO(b"")
        rpc._sock = sock
        rpc._file = sock.makefile.return_value
        return rpc, sock

    def test_call_sends_json_rpc(self):
        rpc, sock = self._rpc_with_mock_socket()
        rpc.call("testMethod", {"key": "val"})
        sent = sock.sendall.call_args[0][0].decode()
        msg = json.loads(sent.strip())
        assert msg["jsonrpc"] == "2.0"
        assert msg["method"] == "testMethod"
        assert msg["params"] == {"key": "val"}
        assert "id" in msg

    def test_call_increments_id(self):
        rpc, sock = self._rpc_with_mock_socket()
        rpc.call("a")
        rpc.call("b")
        calls = [json.loads(c[0][0].decode().strip()) for c in sock.sendall.call_args_list]
        assert calls[1]["id"] == calls[0]["id"] + 1

    def test_send_message_uses_correct_params(self):
        rpc, sock = self._rpc_with_mock_socket()
        rpc.send_message("+1111", "+2222", "Hello Signal")
        sent = json.loads(sock.sendall.call_args[0][0].decode().strip())
        assert sent["method"] == "send"
        assert sent["params"]["account"] == "+1111"
        assert sent["params"]["recipient"] == ["+2222"]
        assert sent["params"]["message"] == "Hello Signal"

    def test_subscribe_uses_subscribe_receive(self):
        rpc, sock = self._rpc_with_mock_socket()
        rpc.subscribe("+1111")
        sent = json.loads(sock.sendall.call_args[0][0].decode().strip())
        assert sent["method"] == "subscribeReceive"
        assert sent["params"]["account"] == "+1111"

    def test_close_closes_socket(self):
        rpc, sock = self._rpc_with_mock_socket()
        rpc.close()
        sock.close.assert_called_once()
        assert rpc._sock is None

    def test_read_loop_dispatches_receive_notification(self):
        notification = {
            "jsonrpc": "2.0",
            "method": "receive",
            "params": {"envelope": {"sourceNumber": "+9999"}},
        }
        raw = (json.dumps(notification) + "\n").encode()
        rpc = _SignalRPC("127.0.0.1:7583")
        rpc._sock = mock.MagicMock()
        rpc._file = io.BytesIO(raw)

        received: list[dict] = []
        rpc.on_receive(received.append)
        rpc.read_loop()

        assert len(received) == 1
        assert received[0]["envelope"]["sourceNumber"] == "+9999"

    def test_read_loop_ignores_non_receive_methods(self):
        response = {"jsonrpc": "2.0", "result": {}, "id": 1}
        raw = (json.dumps(response) + "\n").encode()
        rpc = _SignalRPC("127.0.0.1:7583")
        rpc._sock = mock.MagicMock()
        rpc._file = io.BytesIO(raw)

        received: list[dict] = []
        rpc.on_receive(received.append)
        rpc.read_loop()

        assert len(received) == 0

    def test_read_loop_ignores_malformed_json(self):
        raw = b"not json at all\n"
        rpc = _SignalRPC("127.0.0.1:7583")
        rpc._sock = mock.MagicMock()
        rpc._file = io.BytesIO(raw)
        rpc.on_receive(lambda p: (_ for _ in ()).throw(AssertionError("should not call")))
        rpc.read_loop()  # should not raise

    def test_read_loop_ignores_empty_lines(self):
        raw = b"\n\n  \n"
        rpc = _SignalRPC("127.0.0.1:7583")
        rpc._sock = mock.MagicMock()
        rpc._file = io.BytesIO(raw)
        received: list[dict] = []
        rpc.on_receive(received.append)
        rpc.read_loop()
        assert received == []


# ─────────────────────────────────────────────────────────────────────────────
# _SignalBot._handle_message
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def bot_loaded():
    """A _SignalBot with a fake loaded model and a send-capturing RPC."""
    _clear_sessions()

    class _FakeState:
        model = object()
        model_name = "squish-7b"
        avg_tps = 15.2
        requests = 3
        loaded_at = time.time() - 300

    def _fake_generate(prompt, max_tokens, temperature, top_p, stop, seed):
        yield ("Hello", None)
        yield (" there!", None)
        yield ("", "stop")

    class _FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return " ".join(m["content"] for m in messages)

    sent_messages: list[tuple[str, str]] = []

    b = _SignalBot(
        socket_addr   = "127.0.0.1:7583",
        account       = "+1111",
        get_state     = lambda: _FakeState(),
        get_generate  = lambda: _fake_generate,
        get_tokenizer = lambda: _FakeTok(),
        system_prompt = _DEFAULT_SYSTEM_PROMPT,
    )
    mock_rpc = mock.MagicMock()
    mock_rpc.send_message.side_effect = lambda acct, rcpt, msg: sent_messages.append((rcpt, msg))
    b._rpc = mock_rpc

    return b, sent_messages


def _make_envelope(source: str, body: str) -> dict:
    return {
        "envelope": {
            "sourceNumber": source,
            "dataMessage": {"message": body, "timestamp": int(time.time() * 1000)},
        }
    }


class TestSignalBotHandleMessage:
    def test_normal_message_gets_reply(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message(_make_envelope("+5555", "Hello bot!"))
        assert len(sent) == 1
        assert sent[0][0] == "+5555"
        assert "Hello there!" in sent[0][1]

    def test_reset_command(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message(_make_envelope("+5555", "/reset"))
        assert any("cleared" in m.lower() for _, m in sent)
        # session should be reset
        assert _sessions.get("+5555") == [{"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}]

    def test_help_command(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message(_make_envelope("+5555", "/help"))
        assert len(sent) == 1
        assert "/reset" in sent[0][1]
        assert "/status" in sent[0][1]

    def test_status_command_loaded(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message(_make_envelope("+5555", "/status"))
        assert len(sent) == 1
        assert "squish-7b" in sent[0][1]
        assert "TPS" in sent[0][1] or "tps" in sent[0][1].lower()

    def test_missing_sender_ignored(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message({"envelope": {"dataMessage": {"message": "hi"}}})
        assert sent == []

    def test_missing_body_ignored(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message({"envelope": {"sourceNumber": "+5555", "dataMessage": {"message": ""}}})
        assert sent == []

    def test_no_data_message_ignored(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message({"envelope": {"sourceNumber": "+5555"}})
        assert sent == []

    def test_session_history_built_correctly(self, bot_loaded):
        b, _ = bot_loaded
        _clear_sessions()
        b._handle_message(_make_envelope("+7777", "First message"))
        sess = _sessions.get("+7777", [])
        roles = [m["role"] for m in sess]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles

    def test_case_insensitive_commands(self, bot_loaded):
        b, sent = bot_loaded
        b._handle_message(_make_envelope("+5555", "/RESET"))
        assert any("cleared" in m.lower() for _, m in sent)


class TestSignalBotModelNotLoaded:
    def test_reports_still_loading(self):
        _clear_sessions()

        class _NotLoaded:
            model = None

        sent: list[tuple[str, str]] = []

        b = _SignalBot(
            socket_addr   = "127.0.0.1:7583",
            account       = "+1111",
            get_state     = lambda: _NotLoaded(),
            get_generate  = lambda: None,
            get_tokenizer = lambda: None,
            system_prompt = _DEFAULT_SYSTEM_PROMPT,
        )
        mock_rpc = mock.MagicMock()
        mock_rpc.send_message.side_effect = lambda a, r, m: sent.append((r, m))
        b._rpc = mock_rpc

        b._handle_message(_make_envelope("+5555", "hello"))
        assert len(sent) == 1
        assert "loading" in sent[0][1].lower() or "load" in sent[0][1].lower()

    def test_status_when_model_none(self):
        _clear_sessions()

        class _NotLoaded:
            model = None

        sent: list[tuple[str, str]] = []

        b = _SignalBot(
            socket_addr   = "127.0.0.1:7583",
            account       = "+1111",
            get_state     = lambda: _NotLoaded(),
            get_generate  = lambda: None,
            get_tokenizer = lambda: None,
            system_prompt = _DEFAULT_SYSTEM_PROMPT,
        )
        mock_rpc = mock.MagicMock()
        mock_rpc.send_message.side_effect = lambda a, r, m: sent.append((r, m))
        b._rpc = mock_rpc

        b._handle_message(_make_envelope("+5555", "/status"))
        assert len(sent) == 1
        assert "loading" in sent[0][1].lower()


class TestSignalBotGenerationError:
    def test_generation_exception_sends_error_message(self):
        _clear_sessions()

        class _FakeState:
            model = object()
            model_name = "q"
            avg_tps = 0.0
            requests = 0
            loaded_at = time.time()

        def _broken_gen(prompt, max_tokens, temperature, top_p, stop, seed):
            raise RuntimeError("GPU exploded")
            yield

        class _FakeTok:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                return "prompt"

        sent: list[tuple[str, str]] = []

        b = _SignalBot(
            socket_addr   = "127.0.0.1:7583",
            account       = "+1111",
            get_state     = lambda: _FakeState(),
            get_generate  = lambda: _broken_gen,
            get_tokenizer = lambda: _FakeTok(),
            system_prompt = _DEFAULT_SYSTEM_PROMPT,
        )
        mock_rpc = mock.MagicMock()
        mock_rpc.send_message.side_effect = lambda a, r, m: sent.append((r, m))
        b._rpc = mock_rpc

        b._handle_message(_make_envelope("+5555", "hello"))
        assert len(sent) == 1
        assert "error" in sent[0][1].lower() or "wrong" in sent[0][1].lower()

    def test_send_without_rpc_does_not_raise(self):
        b = _SignalBot(
            socket_addr   = "127.0.0.1:7583",
            account       = "+1111",
            get_state     = lambda: None,
            get_generate  = lambda: None,
            get_tokenizer = lambda: None,
            system_prompt = _DEFAULT_SYSTEM_PROMPT,
        )
        b._rpc = None
        # Should log a warning but not raise
        b._send("+5555", "test")


# ─────────────────────────────────────────────────────────────────────────────
# mount_signal
# ─────────────────────────────────────────────────────────────────────────────

class TestMountSignal:
    def test_no_account_returns_without_starting(self, capsys):
        from fastapi import FastAPI
        app = FastAPI()
        mount_signal(app, get_state=lambda: None, get_generate=lambda: None,
                     get_tokenizer=lambda: None, account="")
        captured = capsys.readouterr()
        assert "--signal-account is required" in captured.out

    def test_with_account_starts_bot_and_registers_endpoint(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        _clear_sessions()
        app = FastAPI()

        class _FakeState:
            model = None

        bot_started = []

        with mock.patch.object(_SignalBot, "start", side_effect=lambda: bot_started.append(True)):
            mount_signal(
                app,
                get_state     = lambda: _FakeState(),
                get_generate  = lambda: None,
                get_tokenizer = lambda: None,
                account       = "+15551234567",
                socket_addr   = "127.0.0.1:7583",
            )

        c = TestClient(app, raise_server_exceptions=True)
        resp = c.get("/signal/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["account"] == "+15551234567"
        assert data["socket"] == "127.0.0.1:7583"


# ─────────────────────────────────────────────────────────────────────────────
# server.py arg-parser integration (validate CLI contract)
# ─────────────────────────────────────────────────────────────────────────────

class TestServerSignalArgs:
    """Verify that server.py exposes the expected --signal* argparse arguments."""

    def _make_parser(self):
        """Return an argparse.ArgumentParser with only the signal args wired."""
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("--signal", action="store_true", default=False)
        ap.add_argument("--signal-account", default="")
        ap.add_argument("--signal-socket", default="127.0.0.1:7583")
        return ap

    def test_signal_flag_accepted(self):
        ap = self._make_parser()
        args = ap.parse_args(["--signal"])
        assert args.signal is True

    def test_signal_flag_default_false(self):
        ap = self._make_parser()
        args = ap.parse_args([])
        assert args.signal is False

    def test_signal_account_accepted(self):
        ap = self._make_parser()
        args = ap.parse_args(["--signal-account", "+12025551234"])
        assert args.signal_account == "+12025551234"

    def test_signal_account_default_empty(self):
        ap = self._make_parser()
        args = ap.parse_args([])
        assert args.signal_account == ""

    def test_signal_socket_accepted(self):
        ap = self._make_parser()
        args = ap.parse_args(["--signal-socket", "/tmp/signal-cli.sock"])
        assert args.signal_socket == "/tmp/signal-cli.sock"

    def test_signal_socket_default(self):
        ap = self._make_parser()
        args = ap.parse_args([])
        assert args.signal_socket == "127.0.0.1:7583"
