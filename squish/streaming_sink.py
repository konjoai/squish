"""squish/streaming_sink.py — backwards-compatibility shim (wave 107).

The canonical implementation lives at squish.streaming.streaming_sink.
This shim makes ``import squish.streaming_sink`` and ``from squish.streaming_sink import ...``
work without duplicating any code or module state.
"""
import sys as _sys
import importlib as _il

_sys.modules[__name__] = _il.import_module("squish.streaming.streaming_sink")
