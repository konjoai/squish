"""squish/cla.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.attention.cla.
This shim makes ``import squish.cla`` and ``from squish.cla import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.attention.cla")
