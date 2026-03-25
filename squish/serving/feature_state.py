"""squish/serving/feature_state.py — Central server state container.

Centralises the ~90 module-level ``_xxx = None`` globals from ``server.py``
into a typed :class:`FeatureState` dataclass.  This makes dependencies
explicit, enables unit testing without importing ``server.py``, and reduces
server.py's global symbol count.

Usage in server.py::

    from squish.serving.feature_state import FeatureState, _state

    # All the old module globals migrate to _state.<attr>
    _state.model = loaded_model
    _state.tokenizer = tok

Public API
──────────
FeatureState   — dataclass of all runtime server state
_state         — module-level singleton, shared by server.py
"""
from __future__ import annotations

__all__ = ["FeatureState", "_state"]

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FeatureState:
    """Holds all mutable server runtime state.

    Attributes mirror the ``_``-prefixed module-level globals in ``server.py``.
    All attributes default to ``None`` / ``False`` / empty to allow server.py
    to be imported without side-effects during testing.
    """

    # ── Core model state ──────────────────────────────────────────────────
    model:              Optional[Any] = None
    tokenizer:          Optional[Any] = None
    model_id:           str           = ""
    model_path:         str           = ""
    model_loaded:       bool          = False
    model_load_ms:      float         = 0.0
    model_params:       int           = 0

    # ── Generation state ──────────────────────────────────────────────────
    max_tokens_default: int           = 2048
    context_window:     int           = 8192
    rope_scaling:       float         = 1.0
    temperature:        float         = 0.7
    top_p:              float         = 0.95
    top_k:              int           = 50
    min_p:              float         = 0.05
    rep_penalty:        float         = 1.05

    # ── Quantization / compression ────────────────────────────────────────
    quant_bits:         int           = 4
    group_size:         int           = 64
    is_blazing:         bool          = False
    blazing_preset:     str           = ""

    # ── KV cache ──────────────────────────────────────────────────────────
    kv_cache:           Optional[Any] = None
    kv_cache_enabled:   bool          = True
    kv_cache_size:      int           = 0

    # ── Draft head / speculative decoding ─────────────────────────────────
    draft_head:         Optional[Any] = None
    draft_head_path:    str           = ""

    # ── Telemetry / profiling ──────────────────────────────────────────────
    profiler:           Optional[Any] = None
    tracer:             Optional[Any] = None
    startup_report:     Optional[Any] = None

    # ── Network ───────────────────────────────────────────────────────────
    host:               str           = "127.0.0.1"
    port:               int           = 11435
    api_key:            str           = "squish"

    # ── Feature flags ─────────────────────────────────────────────────────
    metal_flash_attn:   bool          = False
    fused_sampler:      bool          = False
    token_merge:        bool          = False
    cache_warmup:       bool          = True
    fast_warmup:        bool          = False
    no_metal_warmup:    bool          = False

    # ── Adapter / lora ────────────────────────────────────────────────────
    adapter_path:       str           = ""
    adapter_loaded:     bool          = False

    # ── Chat history / sessions ───────────────────────────────────────────
    max_context_msgs:   int           = 40

    # ── Misc ──────────────────────────────────────────────────────────────
    generation_lock:    Optional[Any] = None
    hardware_info:      Optional[Any] = None


# Module-level singleton — server.py imports this and mutates it at runtime.
_state = FeatureState()
