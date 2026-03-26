"""squish/lookahead_reasoning.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.token.lookahead_reasoning.
This shim makes ``import squish.lookahead_reasoning`` and ``from squish.lookahead_reasoning import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.token.lookahead_reasoning")
