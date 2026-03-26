"""squish/diffkv.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.kv.diffkv.
This shim makes ``import squish.diffkv`` and ``from squish.diffkv import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.kv.diffkv")
