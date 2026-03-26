"""squish/sage_attention2.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.attention.sage_attention2.
This shim makes ``import squish.sage_attention2`` and ``from squish.sage_attention2 import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.attention.sage_attention2")
