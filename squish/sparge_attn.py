"""squish/sparge_attn.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.attention.sparge_attn.
This shim makes ``import squish.sparge_attn`` and ``from squish.sparge_attn import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.attention.sparge_attn")
