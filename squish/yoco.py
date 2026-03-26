"""squish/yoco.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.attention.yoco.
This shim makes ``import squish.yoco`` and ``from squish.yoco import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.attention.yoco")
