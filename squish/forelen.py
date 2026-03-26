"""squish/forelen.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.token.forelen.
This shim makes ``import squish.forelen`` and ``from squish.forelen import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.token.forelen")
