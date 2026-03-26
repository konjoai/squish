"""squish/grammar_engine.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.grammar.grammar_engine.
This shim makes ``from squish.grammar_engine import GrammarEngine`` work
without duplicating any code.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.grammar.grammar_engine")
