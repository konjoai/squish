"""squish/ada_serve.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.serving.ada_serve.
This shim makes ``import squish.ada_serve`` and ``from squish.ada_serve import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.serving.ada_serve")
