"""Tests for the isolated, resource-capped squish_python_repl executor.

Covers the spawn-based subprocess path (memory cap, error capture, output) and
the in-process fallback used when process isolation is unavailable.
"""
from __future__ import annotations

import sys

import pytest

from squish.agent import builtin_tools as bt


class TestReplIsolated:
    def test_basic_output(self):
        assert bt.squish_python_repl("print('hello')").strip() == "hello"

    def test_no_output_marker(self):
        assert bt.squish_python_repl("x = 1 + 1") == "[no output]"

    def test_error_is_captured_not_raised(self):
        out = bt.squish_python_repl("1 / 0")
        assert "[ERROR]" in out
        assert "ZeroDivisionError" in out

    def test_partial_stdout_before_error(self):
        out = bt.squish_python_repl("print('before'); 1/0")
        assert "before" in out
        assert "ZeroDivisionError" in out

    def test_import_is_blocked(self):
        # __import__ is not exposed in the restricted namespace.
        out = bt.squish_python_repl("import os")
        assert "[ERROR]" in out

    def test_empty_code_rejected(self):
        with pytest.raises(ValueError):
            bt.squish_python_repl("   ")

    @pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only")
    def test_memory_limit_enforced(self):
        # Allocating ~320 MB under a 64 MB cap must trip the memory limit
        # rather than exhausting the host.
        out = bt.squish_python_repl("x = [0] * (40_000_000)", max_memory_mb=64)
        assert "[MEMORY LIMIT EXCEEDED]" in out or "terminated" in out

    @pytest.mark.skipif(not hasattr(__import__("signal"), "SIGALRM"),
                        reason="needs POSIX RLIMIT_CPU/SIGALRM")
    def test_runaway_loop_terminates(self):
        out = bt.squish_python_repl("\n".join(["while True:", "    pass"]), timeout=2)
        # CPU limit or wall-clock guard must stop it; exact message varies.
        assert any(tok in out for tok in ("TIMEOUT", "terminated", "ERROR"))


class TestReplInProcessFallback:
    def test_fallback_runs_when_isolation_unavailable(self, monkeypatch):
        # Force the isolated path to report "unavailable" → in-process fallback.
        monkeypatch.setattr(bt, "_repl_run_isolated", lambda *a, **k: None)
        assert bt.squish_python_repl("print('fallback')").strip() == "fallback"

    def test_fallback_captures_error(self, monkeypatch):
        monkeypatch.setattr(bt, "_repl_run_isolated", lambda *a, **k: None)
        out = bt.squish_python_repl("1/0")
        assert "[ERROR]" in out and "ZeroDivisionError" in out

    def test_namespace_shared_between_paths(self):
        # The restricted namespace must expose the documented builtins.
        ns = bt._repl_namespace()
        assert "print" in ns["__builtins__"]
        assert "__import__" not in ns["__builtins__"]
