"""squish/sparse_spec.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.speculative.sparse_spec.
This shim makes ``import squish.sparse_spec`` and ``from squish.sparse_spec import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.speculative.sparse_spec")
