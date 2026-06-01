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
