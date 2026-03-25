"""
squish.speculative.rest_spec — backwards-compatibility shim.

Module moved to ``squish.experimental.speculative.rest_spec``.
This shim transparently replaces itself with the canonical module so that
ALL names (including private ``_names``) remain importable from the original path.
"""
import sys as _sys
import importlib as _importlib
_sys.modules[__name__] = _importlib.import_module(
    "squish.experimental.speculative.rest_spec"
)
