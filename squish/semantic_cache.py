"""squish/semantic_cache.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.kv.semantic_cache.
This shim makes ``from squish.semantic_cache import ...`` work
without duplicating any code.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.kv.semantic_cache")
