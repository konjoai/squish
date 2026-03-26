"""squish/gemfilter.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.token.gemfilter.
This shim makes ``import squish.gemfilter`` and ``from squish.gemfilter import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.token.gemfilter")
