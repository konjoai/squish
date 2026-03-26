"""squish/paris_kv.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.kv.paris_kv.
This shim makes ``import squish.paris_kv`` and ``from squish.paris_kv import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.kv.paris_kv")
