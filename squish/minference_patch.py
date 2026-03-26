"""squish/minference_patch.py — backwards-compatibility shim (wave 108).

The canonical implementation lives at squish.attention.minference_patch.
This shim makes ``from squish.minference_patch import ...`` work
without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.attention.minference_patch")
