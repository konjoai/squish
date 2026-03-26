"""squish/seq_packing.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.streaming.seq_packing.
This shim makes ``import squish.seq_packing`` and ``from squish.seq_packing import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.streaming.seq_packing")
