"""squish/kv_cache.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.kv.kv_cache.
This shim makes ``from squish.kv_cache import ...`` work
without duplicating any code.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.kv.kv_cache")
