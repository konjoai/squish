"""squish/sparse_verify.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.speculative.sparse_verify.
This shim makes ``import squish.sparse_verify`` and ``from squish.sparse_verify import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.speculative.sparse_verify")
