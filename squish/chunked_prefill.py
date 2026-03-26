"""squish/chunked_prefill.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.streaming.chunked_prefill.
This shim makes ``from squish.chunked_prefill import ...`` work
without duplicating any code.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.streaming.chunked_prefill")
