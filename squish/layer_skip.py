"""squish/layer_skip.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.token.layer_skip.
This shim makes ``import squish.layer_skip`` and ``from squish.layer_skip import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.token.layer_skip")
