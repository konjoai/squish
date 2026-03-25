"""
squish.speculative.online_sd — backwards-compatibility shim.

Module moved to ``squish.experimental.speculative.online_sd``.
This shim transparently replaces itself with the canonical module so that
ALL names (including private ``_names``) remain importable from the original path.
"""
import sys as _sys
import importlib as _importlib
_sys.modules[__name__] = _importlib.import_module(
    "squish.experimental.speculative.online_sd"
)
