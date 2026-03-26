"""squish/svdq.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.quant.svdq.
This shim makes ``import squish.svdq`` and ``from squish.svdq import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.quant.svdq")
