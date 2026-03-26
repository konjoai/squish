"""squish/prompt_compressor.py — backwards-compatibility shim (wave 108).

The canonical implementation lives at squish.context.prompt_compressor.
This shim makes ``from squish.prompt_compressor import ...`` work
without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.context.prompt_compressor")
