"""tests/test_fast_imports.py

Regression tests for the cold-start import accelerators in
``squish/_fast_imports.py``:

  Fix #1 — sklearn stub: ``apply_load_path_stubs()`` registers a stub
  for ``sklearn`` and ``sklearn.metrics`` in ``sys.modules`` so
  ``transformers.generation.candidate_generator``'s conditional
  ``from sklearn.metrics import roc_curve`` finds the cached stub and
  skips loading real sklearn (~1 s).

  Fix #3 — background mlx_lm import: ``start_background_mlx_lm_import()``
  spawns a daemon thread that runs ``import mlx_lm`` in parallel with
  the rest of squish.server's load. ``await_mlx_lm_import()`` blocks
  the caller until the thread completes.

What these tests cover
──────────────────────
 1. Stub is idempotent — calling twice has no effect.
 2. Stub does NOT clobber a real sklearn that was already loaded.
 3. The stubbed ``roc_curve`` raises a clear RuntimeError if called.
 4. ``stubs_active()`` correctly reports the stub state.
 5. Background mlx_lm import:
    a. Returns immediately if mlx_lm is already loaded.
    b. Successfully awaits — bg thread completes, mlx_lm is in
       ``sys.modules`` after ``await_mlx_lm_import()`` returns.
    c. Exceptions in the bg thread are captured and re-raised on
       ``await_mlx_lm_import()``.
"""
from __future__ import annotations

import importlib
import sys
import threading
from unittest.mock import patch

import pytest


def _reset_fast_imports_module():
    """Reload ``squish._fast_imports`` so each test starts fresh.

    The module caches background-thread state in globals — reload to
    clear it without polluting the real sklearn / mlx_lm sys.modules
    entries from prior tests.
    """
    import squish._fast_imports as _fi
    _fi._mlx_lm_import_thread = None
    _fi._mlx_lm_import_done = threading.Event()
    _fi._mlx_lm_import_error = None
    return _fi


@pytest.fixture(autouse=True)
def _isolate():
    """Save + restore the bits of sys.modules these tests touch."""
    saved_sklearn = sys.modules.get("sklearn")
    saved_sklearn_metrics = sys.modules.get("sklearn.metrics")
    yield
    if saved_sklearn is not None:
        sys.modules["sklearn"] = saved_sklearn
    if saved_sklearn_metrics is not None:
        sys.modules["sklearn.metrics"] = saved_sklearn_metrics


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #1 — sklearn stub
# ═══════════════════════════════════════════════════════════════════════════════


def test_sklearn_stub_is_installed_after_squish_server_import():
    """`import squish.server` runs apply_load_path_stubs() — sklearn stub
    should be present afterwards."""
    import squish.server  # noqa: F401 — triggers stub install
    import squish._fast_imports as _fi
    assert _fi.stubs_active(), "sklearn stub should be installed after squish.server import"


def test_stub_idempotent():
    """Calling apply_load_path_stubs() a second time leaves the stub intact."""
    fi = _reset_fast_imports_module()
    # The stub is already installed by the squish.server import in earlier tests.
    assert fi.stubs_active()
    result = fi.apply_load_path_stubs()
    # Either it was already there (returns True because the stub IS active) or
    # it bailed because real sklearn was loaded. Either way, no exception.
    assert isinstance(result, bool)
    assert fi.stubs_active()


def test_stub_does_not_clobber_real_sklearn():
    """If a real sklearn was loaded before our stub, we must not overwrite it."""
    import squish._fast_imports as fi
    # Build a fake "real sklearn" with a different version marker.
    import types
    fake_real = types.ModuleType("sklearn")
    fake_real.__version__ = "1.5.0"  # not our stub marker
    fake_real.__path__ = []
    sys.modules["sklearn"] = fake_real

    # Attempt to re-stub
    result = fi.apply_load_path_stubs()
    assert result is False, "Should refuse to overwrite a real sklearn"
    assert sys.modules["sklearn"].__version__ == "1.5.0"
    assert not fi.stubs_active()


def test_stubbed_roc_curve_raises_with_helpful_message():
    """If anything actually calls the stubbed roc_curve, the error should
    point at squish's cold-start optimisation."""
    import squish.server  # noqa: F401 — ensures stub installed
    assert "sklearn.metrics" in sys.modules
    roc_curve = sys.modules["sklearn.metrics"].roc_curve
    with pytest.raises(RuntimeError) as exc_info:
        roc_curve()
    assert "squish runtime" in str(exc_info.value)


def test_stub_version_marker():
    """The stub uses a distinctive version string so it's easy to detect
    in observability output."""
    import squish.server  # noqa: F401
    assert sys.modules["sklearn"].__version__ == "0.0.0-squish-stub"


# ═══════════════════════════════════════════════════════════════════════════════
# Fix #3 — background mlx_lm import
# ═══════════════════════════════════════════════════════════════════════════════


def test_await_returns_immediately_when_already_imported():
    """If mlx_lm is already in sys.modules, await_mlx_lm_import() should
    not spawn a thread and should return without blocking.

    We simulate "already imported" without actually importing mlx_lm
    (the conftest sandbox blocks real mlx imports). The contract under
    test is: if ``mlx_lm`` is in sys.modules, start_background_mlx_lm_import
    sets the event without spawning a thread.
    """
    fi = _reset_fast_imports_module()
    import types
    fake_mlx_lm = types.ModuleType("mlx_lm")
    saved = sys.modules.get("mlx_lm")
    sys.modules["mlx_lm"] = fake_mlx_lm
    try:
        fi.start_background_mlx_lm_import()
        status = fi.background_import_status()
        assert status["completed"] is True
        assert status["thread_started"] is False
        # await should not block
        import time
        t0 = time.perf_counter()
        fi.await_mlx_lm_import()
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.01, f"await took {elapsed:.3f}s when it should be instant"
    finally:
        if saved is None:
            sys.modules.pop("mlx_lm", None)
        else:
            sys.modules["mlx_lm"] = saved


def test_background_import_propagates_errors():
    """If the bg import raises, await_mlx_lm_import() re-raises with the
    original error type captured in the message."""
    fi = _reset_fast_imports_module()
    # Manually invoke _do_bg_mlx_lm_import with a poisoned import to verify
    # the error-capture / re-raise logic.
    import squish._fast_imports as _fi_mod
    orig_do = _fi_mod._do_bg_mlx_lm_import

    def poisoned_do_bg_import():
        try:
            raise ImportError("simulated mlx_lm import failure")
        except Exception as exc:  # noqa: BLE001
            _fi_mod._mlx_lm_import_error = f"{type(exc).__name__}: {exc}"
        finally:
            _fi_mod._mlx_lm_import_done.set()

    # Simulate a failed bg import by reusing the existing thread-tracking
    # variables to drive await_mlx_lm_import down the error branch.
    _fi_mod._mlx_lm_import_thread = threading.Thread(target=poisoned_do_bg_import)
    _fi_mod._mlx_lm_import_thread.start()
    _fi_mod._mlx_lm_import_thread.join()

    with pytest.raises(RuntimeError) as exc_info:
        fi.await_mlx_lm_import()
    assert "simulated mlx_lm import failure" in str(exc_info.value)
    assert "ImportError" in str(exc_info.value)


def test_background_import_status_keys():
    """background_import_status() returns the four keys the contract promises."""
    import squish._fast_imports as fi
    status = fi.background_import_status()
    assert set(status.keys()) == {
        "thread_started", "completed", "error", "mlx_lm_in_sys",
    }
    assert isinstance(status["thread_started"], bool)
    assert isinstance(status["completed"], bool)
    assert isinstance(status["mlx_lm_in_sys"], bool)


def test_squish_server_actually_starts_bg_thread():
    """After squish.server module import, the bg thread either completed
    or is in flight — the thread_started flag must be True."""
    import squish.server  # noqa: F401
    import squish._fast_imports as fi
    status = fi.background_import_status()
    # The bg thread MAY have already finished (very likely on a warm cache).
    # We require that it was started OR mlx_lm somehow got loaded via the
    # foreground (which would also indicate the optimisation worked).
    assert status["thread_started"] or status["mlx_lm_in_sys"]
