"""squish/ipw.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.token.ipw.
This shim makes ``import squish.ipw`` and ``from squish.ipw import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.token.ipw")
