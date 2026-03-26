"""squish/scheduler.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.serving.scheduler.
This shim makes ``from squish.scheduler import ...`` work
without duplicating any code.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.serving.scheduler")
