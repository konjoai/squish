"""Behavioral coverage for the background-mlx_lm-import mechanics of
``squish._fast_imports`` left untested by the baseline suite. Pure-Python;
mlx_lm is faked so no real import occurs. Module-level import-state globals are
reset per test via monkeypatch.
"""
from __future__ import annotations

import sys
import threading
import types

import pytest

from squish import _fast_imports as fi


@pytest.fixture
def fresh_bg_state(monkeypatch):
    """Reset the background-import globals so each test starts clean."""
    monkeypatch.setattr(fi, "_mlx_lm_import_thread", None)
    monkeypatch.setattr(fi, "_mlx_lm_import_done", threading.Event())
    monkeypatch.setattr(fi, "_mlx_lm_import_error", None)


def _fake_mlx_lm(monkeypatch):
    mlx_lm = types.ModuleType("mlx_lm")
    utils = types.ModuleType("mlx_lm.utils")
    utils.load_model = lambda *a, **k: None
    utils.load_tokenizer = lambda *a, **k: None
    utils.load_config = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", utils)


# ── _do_bg_mlx_lm_import ─────────────────────────────────────────────────────


def test_bg_import_success(fresh_bg_state, monkeypatch):
    _fake_mlx_lm(monkeypatch)
    fi._do_bg_mlx_lm_import()  # imports the fakes cleanly (97)
    assert fi._mlx_lm_import_done.is_set()
    assert fi._mlx_lm_import_error is None


def test_bg_import_failure_captured(fresh_bg_state, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)  # import → ImportError
    fi._do_bg_mlx_lm_import()
    assert fi._mlx_lm_import_done.is_set()
    assert fi._mlx_lm_import_error is not None  # error captured for the consumer


# ── start_background_mlx_lm_import ───────────────────────────────────────────


def test_start_bg_noop_when_thread_running(fresh_bg_state, monkeypatch):
    monkeypatch.setattr(fi, "_mlx_lm_import_thread", object())  # pretend already started
    fi.start_background_mlx_lm_import()  # early return — no new thread


def test_start_bg_when_already_imported(fresh_bg_state, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    fi.start_background_mlx_lm_import()  # mlx_lm present → set done + return (119)
    assert fi._mlx_lm_import_done.is_set()
    assert fi._mlx_lm_import_thread is None  # no thread spawned


def test_start_bg_spawns_thread_and_awaits(fresh_bg_state, monkeypatch):
    monkeypatch.delitem(sys.modules, "mlx_lm", raising=False)  # force the spawn branch

    def _fake_worker():
        fi._mlx_lm_import_done.set()  # no-op worker (avoids a real mlx_lm import)

    monkeypatch.setattr(fi, "_do_bg_mlx_lm_import", _fake_worker)
    fi.start_background_mlx_lm_import()
    assert fi._mlx_lm_import_thread is not None  # a daemon thread was spawned
    fi.await_mlx_lm_import(timeout=5.0)  # joins the spawned worker
    assert fi._mlx_lm_import_done.is_set()


# ── await_mlx_lm_import ──────────────────────────────────────────────────────


def test_await_noop_when_no_thread(fresh_bg_state):
    fi.await_mlx_lm_import()  # thread is None → immediate return


def test_await_returns_when_done_without_error(fresh_bg_state, monkeypatch):
    ev = threading.Event()
    ev.set()
    monkeypatch.setattr(fi, "_mlx_lm_import_thread", object())
    monkeypatch.setattr(fi, "_mlx_lm_import_done", ev)
    monkeypatch.setattr(fi, "_mlx_lm_import_error", None)
    fi.await_mlx_lm_import(timeout=1.0)  # done + no error → exit (141→exit)


def test_await_raises_on_error(fresh_bg_state, monkeypatch):
    ev = threading.Event()
    ev.set()
    monkeypatch.setattr(fi, "_mlx_lm_import_thread", object())
    monkeypatch.setattr(fi, "_mlx_lm_import_done", ev)
    monkeypatch.setattr(fi, "_mlx_lm_import_error", "ImportError: boom")
    with pytest.raises(RuntimeError, match="background mlx_lm import failed"):
        fi.await_mlx_lm_import(timeout=1.0)


# ── background_import_status ─────────────────────────────────────────────────


def test_background_import_status(fresh_bg_state):
    status = fi.background_import_status()
    assert set(status) == {"thread_started", "completed", "error", "mlx_lm_in_sys"}
    assert status["thread_started"] is False and status["completed"] is False


# ── stub helpers ─────────────────────────────────────────────────────────────


def test_stub_roc_curve_raises():
    with pytest.raises(RuntimeError, match="stubbed in squish runtime"):
        fi._stub_roc_curve()


def test_apply_stubs_idempotent_when_already_stubbed(monkeypatch):
    stub = types.ModuleType("sklearn")
    stub.__version__ = fi._STUB_VERSION
    monkeypatch.setitem(sys.modules, "sklearn", stub)
    assert fi.apply_load_path_stubs() is True  # our stub already present
    assert fi.stubs_active() is True


def test_apply_stubs_defers_to_real_sklearn(monkeypatch):
    real = types.ModuleType("sklearn")
    real.__version__ = "1.5.0"  # not the stub marker
    monkeypatch.setitem(sys.modules, "sklearn", real)
    assert fi.apply_load_path_stubs() is False  # real sklearn → don't touch
    assert fi.stubs_active() is False


def test_apply_stubs_fresh(monkeypatch):
    monkeypatch.delitem(sys.modules, "sklearn", raising=False)
    monkeypatch.delitem(sys.modules, "sklearn.metrics", raising=False)
    assert fi.apply_load_path_stubs() is True
    assert sys.modules["sklearn"].__version__ == fi._STUB_VERSION
    assert callable(sys.modules["sklearn.metrics"].roc_curve)
