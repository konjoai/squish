"""Cold-start import accelerators.

`import mlx_lm` transitively pulls in transformers → sklearn → torch
(~4 s cold on M3). Squish never uses sklearn at runtime — it's imported
only by `transformers.generation.candidate_generator` for HF's assisted
decoding feature, which squish does not invoke.

This module pre-populates ``sys.modules["sklearn"]`` and
``sys.modules["sklearn.metrics"]`` with no-op stubs so the
``if is_sklearn_available(): from sklearn.metrics import roc_curve``
line at the top of candidate_generator finds them in cache and skips
the real loaders.

Call ``apply_load_path_stubs()`` once, BEFORE any code path imports
``mlx_lm`` or ``transformers``. Idempotent — second and subsequent
calls are no-ops. Safe to call from inside ``squish.server`` at module
load time (squish doesn't otherwise depend on sklearn).

Tests for the public contract live in ``tests/test_fast_imports.py``.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

# Names this module commits to caching as stubs.
_STUBBED_MODULES = ("sklearn", "sklearn.metrics")

_STUB_VERSION = "0.0.0-squish-stub"


def _stub_pkg(name: str, *, is_pkg: bool = True) -> types.ModuleType:
    m = types.ModuleType(name)
    if is_pkg:
        m.__path__ = []  # type: ignore[attr-defined]
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_pkg)
    m.__version__ = _STUB_VERSION
    sys.modules[name] = m
    return m


def _stub_roc_curve(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError(
        "sklearn.metrics.roc_curve is stubbed in squish runtime. "
        "Install sklearn explicitly if your code needs it; squish's "
        "cold-start path skips it to save ~1 s of import time."
    )


def apply_load_path_stubs() -> bool:
    """Install no-op stubs for heavy transitive imports.

    Returns True if stubs were applied, False if they were already present
    (idempotent fast path) or if the real module is already loaded
    (too late — caller must run this before importing mlx_lm/transformers).
    """
    if "sklearn" in sys.modules:
        # Either we already stubbed (look for the version marker) or the
        # real sklearn was loaded first. Either way: don't touch it.
        return getattr(sys.modules["sklearn"], "__version__", None) == _STUB_VERSION

    _stub_pkg("sklearn")
    m = _stub_pkg("sklearn.metrics", is_pkg=False)
    m.roc_curve = _stub_roc_curve  # type: ignore[attr-defined]
    return True


def stubs_active() -> bool:
    """Return True iff our stub is currently installed in sys.modules."""
    return (
        "sklearn" in sys.modules
        and getattr(sys.modules["sklearn"], "__version__", None) == _STUB_VERSION
    )


# ── Background mlx_lm import ─────────────────────────────────────────────────
# `import mlx_lm` is ~2.6 s after the sklearn stub (was 3.8 s before).
# It's a pure-CPU import chain (Python source parsing + module init) that
# can run concurrently with squish.server's own heavy imports (FastAPI,
# uvicorn, the wave modules) — Python releases the GIL during file I/O
# inside each child import, so we get real wall-time overlap.

import threading as _threading  # noqa: E402

_mlx_lm_import_thread: "_threading.Thread | None" = None
_mlx_lm_import_done = _threading.Event()
_mlx_lm_import_error: "str | None" = None


def _do_bg_mlx_lm_import() -> None:
    global _mlx_lm_import_error
    try:
        # The stubs MUST already be applied — they are, by virtue of
        # apply_load_path_stubs() running before start_background_mlx_lm_import().
        import mlx_lm  # noqa: F401, PLC0415
        from mlx_lm.utils import load_model, load_tokenizer, load_config  # noqa: F401, PLC0415
    except (ImportError, RuntimeError, OSError, AttributeError, ValueError) as exc:
        # Captured here and re-raised by await_mlx_lm_import() on the consumer thread.
        import logging  # noqa: PLC0415 — kept lazy to avoid cold-start import overhead
        logging.getLogger("squish._fast_imports").debug(
            "background mlx_lm import failed: %s", exc
        )
        _mlx_lm_import_error = f"{type(exc).__name__}: {exc}"
    finally:
        _mlx_lm_import_done.set()


def start_background_mlx_lm_import() -> None:
    """Kick off ``import mlx_lm`` on a daemon thread.

    Safe to call multiple times — only the first call spawns the thread.
    Callers that need mlx_lm guaranteed-loaded should ``await_mlx_lm_import()``
    before using it; if the background import is already done, that call
    returns immediately.
    """
    global _mlx_lm_import_thread
    if _mlx_lm_import_thread is not None:
        return
    if "mlx_lm" in sys.modules:
        # Already imported on the foreground — nothing to do.
        _mlx_lm_import_done.set()
        return
    _mlx_lm_import_thread = _threading.Thread(
        target=_do_bg_mlx_lm_import,
        name="squish-mlx-lm-import",
        daemon=False,
    )
    _mlx_lm_import_thread.start()


def await_mlx_lm_import(timeout: "float | None" = None) -> None:
    """Block until the background mlx_lm import completes.

    If no background import was started (or it already finished), returns
    immediately. Raises RuntimeError if the background import raised.
    """
    if _mlx_lm_import_thread is None:
        return
    _mlx_lm_import_done.wait(timeout=timeout)
    if _mlx_lm_import_error is not None:
        raise RuntimeError(
            f"background mlx_lm import failed: {_mlx_lm_import_error}"
        )


def background_import_status() -> dict[str, "object"]:
    """For tests + observability. Returns the current state of the bg import."""
    return {
        "thread_started":   _mlx_lm_import_thread is not None,
        "completed":        _mlx_lm_import_done.is_set(),
        "error":            _mlx_lm_import_error,
        "mlx_lm_in_sys":    "mlx_lm" in sys.modules,
    }
