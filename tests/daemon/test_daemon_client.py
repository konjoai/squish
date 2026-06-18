"""Tests for DaemonClient — the thin Unix-socket client for squishd.

send_request / is_running are monkeypatched, so no live daemon is needed.
"""
from __future__ import annotations

import pytest

from squish.daemon import client as client_mod
from squish.daemon.client import DaemonClient


@pytest.fixture
def capture(monkeypatch):
    """Patch send_request to capture the payload and return a canned response."""
    seen = {}

    def _fake_send(payload, sock_path, timeout):
        seen["payload"] = payload
        seen["sock_path"] = sock_path
        seen["timeout"] = timeout
        return {"text": "hi", "tokens": 1, "tok_s": 2.0, "finish": "stop"}

    monkeypatch.setattr(client_mod, "send_request", _fake_send)
    return seen


class TestAvailability:
    def test_available_true(self, monkeypatch):
        monkeypatch.setattr(client_mod, "is_running", lambda p: True)
        assert DaemonClient().available() is True

    def test_available_false(self, monkeypatch):
        monkeypatch.setattr(client_mod, "is_running", lambda p: False)
        assert DaemonClient(sock_path="/tmp/x.sock").available() is False


class TestRequests:
    def test_ping(self, capture):
        out = DaemonClient(sock_path="/s", timeout=9).ping()
        assert capture["payload"] == {"_cmd": "ping"}
        assert capture["sock_path"] == "/s" and capture["timeout"] == 9
        assert out["finish"] == "stop"

    def test_chat_includes_dirs_when_set(self, capture):
        c = DaemonClient(model_dir="/m", compressed_dir="/c")
        c.chat([{"role": "user", "content": "x"}], max_tokens=10, temperature=0.1, top_p=0.5)
        p = capture["payload"]
        assert p["messages"] == [{"role": "user", "content": "x"}]
        assert p["max_tokens"] == 10 and p["temperature"] == 0.1 and p["top_p"] == 0.5
        assert p["model_dir"] == "/m" and p["compressed_dir"] == "/c"

    def test_chat_omits_dirs_when_empty(self, capture):
        DaemonClient().chat([{"role": "user", "content": "x"}])
        p = capture["payload"]
        assert "model_dir" not in p and "compressed_dir" not in p

    def test_chat_arg_dirs_override_instance(self, capture):
        c = DaemonClient(model_dir="/default")
        c.chat([{"role": "user", "content": "x"}], model_dir="/override", compressed_dir="/cc")
        assert capture["payload"]["model_dir"] == "/override"
        assert capture["payload"]["compressed_dir"] == "/cc"

    def test_complete_wraps_prompt_as_user_message(self, capture):
        DaemonClient().complete("hello", max_tokens=7)
        p = capture["payload"]
        assert p["messages"] == [{"role": "user", "content": "hello"}]
        assert p["max_tokens"] == 7


class TestTiming:
    def test_ttft_returns_float(self, capture):
        t = DaemonClient().ttft([{"role": "user", "content": "x"}])
        assert isinstance(t, float) and t >= 0.0
        assert capture["payload"]["max_tokens"] == 1

    def test_wall_latency_percentiles(self, capture):
        out = DaemonClient().wall_latency(
            [{"role": "user", "content": "x"}], max_tokens=4, n_runs=5
        )
        assert set(out) == {"p50", "p95", "p99", "mean"}
        assert all(isinstance(v, float) for v in out.values())
        # Percentiles are monotonic non-decreasing for sorted timings.
        assert out["p50"] <= out["p95"] <= out["p99"]
        assert out["mean"] >= 0.0

    def test_wall_latency_single_run(self, capture):
        # n_runs=1 exercises the _pct hi-clamp (len-1 == 0).
        out = DaemonClient().wall_latency([{"role": "user", "content": "x"}], n_runs=1)
        assert out["p50"] == out["p99"]  # only one sample
