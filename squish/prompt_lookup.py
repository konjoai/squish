"""squish/prompt_lookup.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.speculative.prompt_lookup.
This shim makes ``import squish.prompt_lookup`` and ``from squish.prompt_lookup import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.speculative.prompt_lookup")
