"""squish/smallkv.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.kv.smallkv.
This shim makes ``import squish.smallkv`` and ``from squish.smallkv import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.kv.smallkv")
