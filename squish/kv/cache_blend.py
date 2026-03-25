"""
squish.kv.cache_blend — backwards-compatibility shim.

Module moved to ``squish.experimental.kv.cache_blend``.
This shim transparently replaces itself with the canonical module so that
ALL names (including private ``_names``) remain importable from the original path.
"""
import sys as _sys
import importlib as _importlib
_sys.modules[__name__] = _importlib.import_module(
    "squish.experimental.kv.cache_blend"
)
