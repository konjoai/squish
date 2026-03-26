"""squish/specontext.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.speculative.specontext.
This shim makes ``import squish.specontext`` and ``from squish.specontext import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.speculative.specontext")
