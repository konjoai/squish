"""squish/robust_scheduler.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.serving.robust_scheduler.
This shim makes ``import squish.robust_scheduler`` and ``from squish.robust_scheduler import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.serving.robust_scheduler")
