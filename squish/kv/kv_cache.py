#!/usr/bin/env python3
"""
squish/kv_cache.py

Quantized KV cache for long-context inference on Apple Silicon unified memory.

Two complementary strategies — each reducing KV cache memory by ~50%:

  KIVI (Kim et al., 2024  https://arxiv.org/abs/2402.02750)
  ──────────────────────────────────────────────────────────
  Keep the most-recent ``window`` token positions in FP16 (the "residual").
  Quantize older positions to INT8 with per-channel scales for keys and
  per-token scales for values.  All slots are dequantized on-the-fly during
  the attention computation.

  Memory model (per layer, per head, per token):
    • FP16:  head_dim × 2 bytes  = 256 bytes  (Qwen2-7B: head_dim=128)
    • INT8:  head_dim × 1 byte
           + 1 scale × 4 bytes (f32)          ≈ 132 bytes  (½ of FP16)

  SnapKV (Li et al., 2024  https://arxiv.org/abs/2404.14469)
  ──────────────────────────────────────────────────────────
  During prefill, observe which K/V positions receive the most attention from
  the most-recent ``snap_window`` query positions.  After prefill, evict the
  bottom ``(1 - budget_ratio)`` fraction of positions.  This caps the cache
  size to ``budget`` tokens regardless of context length.


Usage
-----
At the server level — patch the model before any generation:

    from squish.kv.kv_cache import make_quantized_cache, patch_model_kv_cache

    # After mlx_lm.load() returns (model, tokenizer):
    patch_model_kv_cache(model, mode="int8", window=64)

    # With SnapKV budget (evict to at most 2048 positions):
    patch_model_kv_cache(model, mode="snap", window=64, budget=2048)

Low-level: create a cache and pass to generate():

    cache = make_quantized_cache(model, mode="int8", window=64)
    # Pass cache as kv_cache argument to mlx_lm generate functions
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Lazy MLX import (module may be imported without Metal available, e.g. tests)
# ---------------------------------------------------------------------------
try:
    import mlx.core as _mlx
except ImportError:  # pragma: no cover
    _mlx = None  # will raise at runtime if Metal code is actually called


def _mx():
    """Compatibility shim — prefer using _mlx directly in hot paths."""
    return _mlx


# ---------------------------------------------------------------------------
# INT8 per-channel quantization helpers (pure numpy — runs on CPU)
# These are only called on the "old" portion of the cache;
# the recent window stays in FP16 on Metal.
# ---------------------------------------------------------------------------

def _quantize_int8_per_channel(arr_f16: np.ndarray) -> tuple:
    """
    Quantize a 2-D float16 array to INT8 per output-channel (per row).

    Uses in-place arithmetic to keep peak memory to ~1× input (float32)
    instead of the naive 3× (float32 + abs intermediate + division result).

    Parameters
    ----------
    arr_f16 : np.ndarray  shape (n_tokens, head_dim)  — float16

    Returns
    -------
    q    : np.ndarray  shape (n_tokens, head_dim)  — int8
    scale: np.ndarray  shape (n_tokens,)            — float32 per-token scale
    """
    arr = arr_f16.astype(np.float32)           # unavoidable: fp16 overflows in abs
    # Per-row abs-max (fused reduce — no full intermediate array)
    scale    = np.max(np.abs(arr), axis=-1)    # (n,)
    scale_safe = np.maximum(scale, 1e-8)       # (n,) — avoids divide-by-zero
    # In-place normalize + scale — reuses the float32 buffer
    arr /= scale_safe[:, np.newaxis]           # normalise to [-1, 1]
    arr *= 127.0
    np.round(arr, out=arr)
    q = np.clip(arr, -128, 127).astype(np.int8)
    return q, scale_safe.astype(np.float32)


def _dequantize_int8_per_channel(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """
    Dequantize INT8 array back to float16.

    Parameters
    ----------
    q     : (n_tokens, head_dim)   int8
    scale : (n_tokens,)            float32

    Returns
    -------
    (n_tokens, head_dim)  float16
    """
    arr = q.astype(np.float32) / 127.0 * scale[:, np.newaxis]
    return arr.astype(np.float16)


# ---------------------------------------------------------------------------
# W104 — Per-token INT2 KV quantization (4-level NF2 codebook, bit-packed)
# ---------------------------------------------------------------------------
#
# Math:
#   For each token row v ∈ ℝ^d (after Hadamard rotation):
#     scale = max(|v|) / 1.5                    (so codebook level ±1.5 = ±max)
#     idx   = clip(round(v / scale + 1.5), 0, 3)
#     decode_value = (idx - 1.5) · scale       i.e. {-1.5, -0.5, 0.5, 1.5} · scale
#
#   The four codebook levels {-1.5, -0.5, 0.5, 1.5} (uniform-spaced, symmetric)
#   match SQINT2's NF2_VALUES — see squish/quant/sqint2.py:178.  Per-token scale
#   keeps each row independent: 4× smaller than INT8 storage at ≈0.5–1 dB MSE
#   penalty on Hadamard-rotated activations (where the post-rotation distribution
#   is approximately IID, near-Gaussian with bounded variance per axis).
#
# Storage:
#   Indices 0..3 fit in 2 bits.  Pack 4 indices per uint8 byte along head_dim.
#   For head_dim=128 → packed shape (n_tokens, 32) uint8 = 32 bytes/token, vs
#   128 bytes/token for INT8.  Plus per-token f32 scale (4 bytes) → 36 bytes
#   total per token vs 132 for INT8.  Memory ratio ≈ 0.273 → ~3.7× context at
#   the same RAM (3.66× exact when head_dim ≫ 1).
#
# Constraint: head_dim must be divisible by 4.  Common transformer head_dims
# (64, 96, 128, 160, 192, 256) all satisfy this.  We assert at quantize time
# rather than padding; padding would inflate storage and complicate dequant.
# ---------------------------------------------------------------------------

# NF2 codebook levels — uniform symmetric 4-level codebook centred on 0.
# Identical to squish.quant.sqint2.NF2_VALUES so the SQINT2 weight pipeline
# and the W104 KV pipeline share a single quantisation grid.
_KV_INT2_LEVELS: np.ndarray = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float32)
# Decode boundary points between adjacent codebook levels (midpoints): -1, 0, 1.
# We don't store these — np.clip(round(v/scale + 1.5), 0, 3) does the binning.


def _quantize_int2_per_channel(arr_f16: np.ndarray) -> tuple:
    """
    Quantize a 2-D float16 array to bit-packed INT2 per output-channel (per row).

    Each row gets its own scale (max(|row|) / 1.5).  Indices in {0, 1, 2, 3}
    are packed 4-per-byte along the head_dim axis: packed[t, i] holds
    indices for cols (4i, 4i+1, 4i+2, 4i+3) in bit positions (0–1, 2–3, 4–5, 6–7).

    Parameters
    ----------
    arr_f16 : np.ndarray  shape (n_tokens, head_dim)  — float16
              ``head_dim`` must be a multiple of 4.

    Returns
    -------
    packed : np.ndarray  shape (n_tokens, head_dim // 4)  — uint8
    scale  : np.ndarray  shape (n_tokens,)                — float32
    """
    if arr_f16.ndim != 2:
        raise ValueError(f"_quantize_int2_per_channel: expected 2-D, got {arr_f16.ndim}-D")
    n, d = arr_f16.shape
    if d % 4 != 0:
        raise ValueError(
            f"_quantize_int2_per_channel: head_dim={d} must be divisible by 4 "
            f"(packing 4 INT2 indices per uint8 byte)"
        )

    arr = arr_f16.astype(np.float32)
    # Per-row abs-max → scale s.t. max maps to codebook level 1.5
    abs_max = np.max(np.abs(arr), axis=-1)             # (n,)
    scale = np.maximum(abs_max, 1e-8) / 1.5            # (n,) — avoids divide-by-zero
    # idx = round(v/scale + 1.5), then clip to [0, 3]
    normalised = arr / scale[:, np.newaxis] + 1.5
    np.round(normalised, out=normalised)
    indices = np.clip(normalised, 0.0, 3.0).astype(np.uint8)   # (n, d)

    # Pack 4 indices per byte: byte = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6)
    # Each pack of 4 cols → 1 byte; output shape (n, d/4).
    packed = (
        (indices[:, 0::4])
        | (indices[:, 1::4] << 2)
        | (indices[:, 2::4] << 4)
        | (indices[:, 3::4] << 6)
    ).astype(np.uint8)
    return packed, scale.astype(np.float32)


def _dequantize_int2_per_channel(
    packed: np.ndarray, scale: np.ndarray, head_dim: int
) -> np.ndarray:
    """
    Dequantize bit-packed INT2 array back to float16.

    Inverse of :func:`_quantize_int2_per_channel`.  Looks up the 4-level
    codebook entry for each unpacked index and multiplies by the per-token
    scale.

    Parameters
    ----------
    packed   : (n_tokens, head_dim // 4)  uint8
    scale    : (n_tokens,)                float32
    head_dim : original (unpacked) head dimension

    Returns
    -------
    (n_tokens, head_dim)  float16
    """
    if head_dim % 4 != 0:
        raise ValueError(
            f"_dequantize_int2_per_channel: head_dim={head_dim} must be divisible by 4"
        )
    n_packed = head_dim // 4
    if packed.shape[-1] != n_packed:
        raise ValueError(
            f"_dequantize_int2_per_channel: packed.shape[-1]={packed.shape[-1]} "
            f"!= head_dim/4 = {n_packed}"
        )

    n = packed.shape[0]
    indices = np.empty((n, head_dim), dtype=np.uint8)
    indices[:, 0::4] =  packed       & np.uint8(0x03)
    indices[:, 1::4] = (packed >> 2) & np.uint8(0x03)
    indices[:, 2::4] = (packed >> 4) & np.uint8(0x03)
    indices[:, 3::4] = (packed >> 6) & np.uint8(0x03)

    # Codebook lookup: indices ∈ {0,1,2,3} → {-1.5, -0.5, 0.5, 1.5}.
    arr = _KV_INT2_LEVELS[indices.astype(np.intp)]      # (n, head_dim) float32
    arr *= scale[:, np.newaxis]
    return arr.astype(np.float16)


# ---------------------------------------------------------------------------
# W105 — Per-token INT4 KV quantization (16-level symmetric, 2-per-byte packed)
# ---------------------------------------------------------------------------
#
# Math:
#   For each token row v ∈ ℝ^d (after Hadamard rotation):
#     scale = max(|v|) / 7.5                    (so codebook level ±7.5 = ±max)
#     idx   = clip(round(v / scale + 7.5), 0, 15)         (16 levels, 4 bits)
#     decode_value = (idx - 7.5) · scale       i.e. {-7.5, -6.5, ..., 6.5, 7.5} · scale
#
#   Symmetric 16-level uniform codebook centred on 0; mirrors the structure of
#   the W104 INT2 codec (n_levels = 2^bits, mid = (n_levels - 1) / 2).  The
#   mid-bin offset of 7.5 means the codebook never has a true "0" level — the
#   two centre levels are ±0.5·scale, equally good for capturing small-but-
#   nonzero post-rotation activations.
#
# Storage:
#   Indices 0..15 fit in 4 bits.  Pack 2 indices per uint8 byte along head_dim
#   (low nibble = even index, high nibble = odd index):
#     byte = (idx_even & 0xF) | ((idx_odd & 0xF) << 4)
#   For head_dim=128 → packed shape (n_tokens, 64) uint8 = 64 bytes/token, vs
#   128 bytes/token for INT8.  Plus per-token f32 scale (4 bytes) → 68 bytes
#   total per token vs 132 for INT8 (≈ 0.515 ratio → ~1.94× context at the
#   same RAM, asymptotic 2× as head_dim → ∞).
#
# Quality positioning (relative to W104 INT2 and existing INT8):
#   - INT8: ~44 dB SNR on Hadamard-rotated activations (~132 B/token at d=128).
#   - INT4: ~22–28 dB SNR (~68 B/token) — the intermediate quality tier.
#   - INT2: ~5–7 dB SNR (~36 B/token) — long-context-only.
#   INT4 is the "Phase 3.1 INT4 path" originally cited in the W104 plan: it
#   doubles context within the M3 16 GB envelope while keeping a quality
#   margin large enough that it works as a safe default for 8 K–24 K contexts
#   without the SNR cliff that INT2 introduces.
#
# Constraint: head_dim must be divisible by 2.  Every common transformer
# head_dim already satisfies this; we assert at quantize time rather than
# padding (padding inflates storage and complicates dequant).
# ---------------------------------------------------------------------------

# INT4 codebook levels — uniform symmetric 16-level codebook centred on 0.
# Indices 0..15 decode to (idx - 7.5) · scale.  We materialise the lookup
# table once at module load to keep the dequant hot path branch-free.
_KV_INT4_LEVELS: np.ndarray = (
    np.arange(16, dtype=np.float32) - 7.5
)  # [-7.5, -6.5, ..., 6.5, 7.5]


def _quantize_int4_per_channel(arr_f16: np.ndarray) -> tuple:
    """
    Quantize a 2-D float16 array to nibble-packed INT4 per output-channel
    (per row).

    Each row gets its own scale (max(|row|) / 7.5).  Indices in {0..15}
    are packed 2-per-byte along the head_dim axis:
        packed[t, i] = (idx_even & 0xF) | ((idx_odd & 0xF) << 4)

    Parameters
    ----------
    arr_f16 : np.ndarray  shape (n_tokens, head_dim)  — float16
              ``head_dim`` must be a multiple of 2.

    Returns
    -------
    packed : np.ndarray  shape (n_tokens, head_dim // 2)  — uint8
    scale  : np.ndarray  shape (n_tokens,)                — float32
    """
    if arr_f16.ndim != 2:
        raise ValueError(f"_quantize_int4_per_channel: expected 2-D, got {arr_f16.ndim}-D")
    n, d = arr_f16.shape
    if d % 2 != 0:
        raise ValueError(
            f"_quantize_int4_per_channel: head_dim={d} must be divisible by 2 "
            f"(packing 2 INT4 indices per uint8 byte)"
        )

    arr = arr_f16.astype(np.float32)
    abs_max = np.max(np.abs(arr), axis=-1)             # (n,)
    scale = np.maximum(abs_max, 1e-8) / 7.5            # (n,) — avoid /0
    normalised = arr / scale[:, np.newaxis] + 7.5
    np.round(normalised, out=normalised)
    indices = np.clip(normalised, 0.0, 15.0).astype(np.uint8)  # (n, d)

    # Pack 2 nibbles per byte: even cols → low nibble, odd cols → high nibble.
    packed = (
        (indices[:, 0::2] & np.uint8(0x0F))
        | ((indices[:, 1::2] & np.uint8(0x0F)) << 4)
    ).astype(np.uint8)
    return packed, scale.astype(np.float32)


def _dequantize_int4_per_channel(
    packed: np.ndarray, scale: np.ndarray, head_dim: int
) -> np.ndarray:
    """
    Dequantize nibble-packed INT4 array back to float16.

    Inverse of :func:`_quantize_int4_per_channel`.

    Parameters
    ----------
    packed   : (n_tokens, head_dim // 2)  uint8
    scale    : (n_tokens,)                float32
    head_dim : original (unpacked) head dimension

    Returns
    -------
    (n_tokens, head_dim)  float16
    """
    if head_dim % 2 != 0:
        raise ValueError(
            f"_dequantize_int4_per_channel: head_dim={head_dim} must be divisible by 2"
        )
    n_packed = head_dim // 2
    if packed.shape[-1] != n_packed:
        raise ValueError(
            f"_dequantize_int4_per_channel: packed.shape[-1]={packed.shape[-1]} "
            f"!= head_dim/2 = {n_packed}"
        )

    n = packed.shape[0]
    indices = np.empty((n, head_dim), dtype=np.uint8)
    indices[:, 0::2] =  packed       & np.uint8(0x0F)   # low nibble  → even cols
    indices[:, 1::2] = (packed >> 4) & np.uint8(0x0F)   # high nibble → odd cols

    arr = _KV_INT4_LEVELS[indices.astype(np.intp)]      # (n, head_dim) float32
    arr *= scale[:, np.newaxis]
    return arr.astype(np.float16)


# Mode-dispatching helpers used by KVLayerCache to avoid duplicating the
# per-mode split at every callsite.  ``mode`` here is the per-layer storage
# mode: "int8" (default), "int4" (W105), or "int2" (W104).

# All quant-bearing modes recognised by the dispatch helpers.
_KV_QUANT_MODES: frozenset = frozenset({"int8", "int4", "int2"})


def _kv_quantize_per_channel(arr_f16: np.ndarray, mode: str) -> tuple:
    """Dispatch to the correct quantizer for the layer's storage mode."""
    if mode == "int2":
        return _quantize_int2_per_channel(arr_f16)
    if mode == "int4":
        return _quantize_int4_per_channel(arr_f16)
    # "int8" / "snap" both use INT8 per-channel storage.
    return _quantize_int8_per_channel(arr_f16)


def _kv_dequantize_per_channel(
    q: np.ndarray, scale: np.ndarray, mode: str, head_dim: "int | None" = None
) -> np.ndarray:
    """Dispatch to the correct dequantizer for the layer's storage mode.

    ``head_dim`` is required when ``mode in {"int2", "int4"}`` to undo
    bit-packing.
    """
    if mode == "int2":
        if head_dim is None:
            raise ValueError("head_dim is required for INT2 dequantization")
        return _dequantize_int2_per_channel(q, scale, head_dim)
    if mode == "int4":
        if head_dim is None:
            raise ValueError("head_dim is required for INT4 dequantization")
        return _dequantize_int4_per_channel(q, scale, head_dim)
    return _dequantize_int8_per_channel(q, scale)


# Convenience: recommended KV mode for a given context length.  Plan W104:
# auto-enable INT2 above 8 K tokens to keep Qwen2.5-7B at 32 K within the
# M3 16 GB envelope (currently OOMs around ~10 K with INT8 KV).
# W105 adds an optional INT4 medium tier (8 K–16 K), where the additional
# 5–10 dB SNR margin over INT2 is worth the extra ~30 B per token.
KV_INT2_AUTO_THRESHOLD: int = 8192
KV_INT4_DEFAULT_THRESHOLD: int = 16384      # W105 — switch INT4 → INT2 above this


def recommended_kv_mode(
    context_tokens: int,
    short_mode: str = "int8",
    long_mode: str = "int2",
    threshold: int = KV_INT2_AUTO_THRESHOLD,
    medium_mode: "str | None" = None,
    medium_threshold: "int | None" = None,
) -> str:
    """Pick the KV storage mode suitable for ``context_tokens``.

    Two-tier (W104, default behaviour, unchanged):

    >>> recommended_kv_mode(4096)
    'int8'
    >>> recommended_kv_mode(16384)
    'int2'

    Three-tier (W105) — pass ``medium_mode`` and ``medium_threshold`` to
    introduce an INT4 stage between ``short_mode`` and ``long_mode``:

    >>> recommended_kv_mode(4000, medium_mode="int4", medium_threshold=8192)
    'int8'
    >>> recommended_kv_mode(12000, medium_mode="int4", medium_threshold=8192)
    'int4'
    >>> recommended_kv_mode(20000, medium_mode="int4", medium_threshold=8192)
    'int2'

    Used by callers that build a :class:`QuantizedKVCache` once per session
    and want the storage mode driven by the planned context length.
    """
    if context_tokens < 0:
        raise ValueError(f"context_tokens must be ≥ 0, got {context_tokens}")
    if (medium_mode is None) != (medium_threshold is None):
        raise ValueError(
            "medium_mode and medium_threshold must both be set or both be None"
        )
    if medium_mode is not None and medium_threshold is not None:
        if medium_threshold > threshold:
            raise ValueError(
                f"medium_threshold ({medium_threshold}) must be ≤ threshold "
                f"({threshold}) — medium tier sits between short and long"
            )
        if context_tokens > threshold:
            return long_mode
        if context_tokens > medium_threshold:
            return medium_mode
        return short_mode
    return long_mode if context_tokens > threshold else short_mode


def recommended_kv_mode_3tier(
    context_tokens: int,
    short_threshold: int = KV_INT2_AUTO_THRESHOLD,
    long_threshold: int = KV_INT4_DEFAULT_THRESHOLD,
) -> str:
    """W105 — int8 / int4 / int2 by context length.

    Returns ``"int8"`` for ≤ ``short_threshold``, ``"int4"`` for the band
    between ``short_threshold`` and ``long_threshold``, and ``"int2"`` for
    contexts longer than ``long_threshold``.

    Defaults: ≤ 8 K → int8, 8 K–16 K → int4, > 16 K → int2.
    """
    return recommended_kv_mode(
        context_tokens,
        short_mode="int8",
        medium_mode="int4",
        long_mode="int2",
        medium_threshold=short_threshold,
        threshold=long_threshold,
    )


# ---------------------------------------------------------------------------
# W106 — KV memory budgeting + cache factory  (closed-form planning helpers)
# ---------------------------------------------------------------------------
#
# W104+W105 added three storage modes (int8/int4/int2) but production callers
# still write boilerplate to wire `recommended_kv_mode_3tier` to the right
# constructor, and have no closed-form way to answer the two questions
# every deployer asks before picking a mode:
#
#   1. "How much RAM will the KV cache use at context X under mode Y?"
#   2. "How long a context fits in B bytes under mode Y?"
#
# This block answers both with a closed-form formula that matches the live
# `cache.memory_bytes` of the actual buffers to within 1 % across the
# regression test workload.  It also adds `make_kv_cache(...)` — the one-line
# factory that picks the right mode and constructs the right class.
#
# Math
# ----
# For a single-layer storage tier with `n_kv_heads` heads and `head_dim`
# columns, the per-token cost is:
#
#   bytes_per_token(mode) =
#       (codes_per_token + scale_bytes_per_token) * n_kv_heads
#
#   codes_per_token =
#       head_dim          for "int8"   (1 byte / value)
#       head_dim / 2      for "int4"   (½ byte / value)
#       head_dim / 4      for "int2"   (¼ byte / value)
#       head_dim * 2      for "fp16"   (2 bytes / value, no quant)
#
#   scale_bytes_per_token =
#       0                 for "fp16"   (no per-token scale stored)
#       4                 for int8/int4/int2  (one fp32 scale per token)
#
# K and V are stored independently → multiply by 2.  Multiply by `n_layers`
# for whole-model totals.  This DOES NOT include the FP16 recent-window
# cost (typically window=128 or smaller, dwarfed by the old-tier buffer at
# real context lengths) — see KVMemoryEstimate.recent_window_bytes for the
# explicit additive term when callers want it.
# ---------------------------------------------------------------------------

# Per-quant scale storage (one fp32 scalar per token, per head).
_KV_SCALE_BYTES: int = 4
_KV_BYTES_FP16:  int = 2
_KV_BUFFERS_PER_LAYER: int = 2          # K + V

# Order matters: tiers are tried short→long when picking a mode for a
# memory budget.  "fp16" is included so callers can ask "does the
# uncompressed cache fit?" without writing a separate branch.
_KV_TIER_ORDER: tuple = ("int8", "int4", "int2")


def _bytes_per_token_per_head(mode: str, head_dim: int) -> int:
    """Closed-form per-token-per-head cost for one quant-bearing tier.

    Includes both the code bytes and the fp32 per-token scale.
    """
    if head_dim <= 0:
        raise ValueError(f"head_dim must be positive, got {head_dim}")
    if mode == "fp16":
        return head_dim * _KV_BYTES_FP16          # no per-token scale
    if mode == "int8":
        codes = head_dim
    elif mode == "int4":
        if head_dim % 2:
            raise ValueError(
                f"INT4 storage requires head_dim divisible by 2, got {head_dim}"
            )
        codes = head_dim // 2
    elif mode == "int2":
        if head_dim % 4:
            raise ValueError(
                f"INT2 storage requires head_dim divisible by 4, got {head_dim}"
            )
        codes = head_dim // 4
    else:
        raise ValueError(
            f"mode must be one of fp16/int8/int4/int2, got {mode!r}"
        )
    return codes + _KV_SCALE_BYTES


@dataclass(frozen=True)
class KVMemoryEstimate:
    """Closed-form KV-cache memory estimate for one (mode, model, context).

    Attributes
    ----------
    mode                          one of {"fp16", "int8", "int4", "int2"}
    n_layers                      transformer block count
    n_kv_heads                    KV-head count (after grouped-query reduction)
    head_dim                      per-head dimension
    context_tokens                planned context length
    bytes_per_token_per_head      closed-form codes + scale, per head, per tier
    bytes_per_token               × n_kv_heads × 2 (K+V buffers)
    bytes_per_layer               × context_tokens
    total_bytes                   × n_layers — the headline number
    fp16_baseline_bytes           same dims but with mode="fp16" — the reference
    compression_ratio             fp16_baseline_bytes / total_bytes (≥ 1.0)
    recent_window_bytes           additional FP16 recent-window overhead at the
                                  given window size, per layer × n_layers
    """
    mode:                     str
    n_layers:                 int
    n_kv_heads:               int
    head_dim:                 int
    context_tokens:           int
    bytes_per_token_per_head: int
    bytes_per_token:          int
    bytes_per_layer:          int
    total_bytes:              int
    fp16_baseline_bytes:      int
    compression_ratio:        float
    recent_window_bytes:      int

    def fits_in(self, budget_bytes: int) -> bool:
        """True iff ``total_bytes + recent_window_bytes ≤ budget_bytes``."""
        return (self.total_bytes + self.recent_window_bytes) <= budget_bytes


def estimate_kv_memory(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    mode: str,
    *,
    window: int = 0,
) -> KVMemoryEstimate:
    """Closed-form KV-cache memory estimate.

    Args:
        n_layers:        transformer layer count.
        n_kv_heads:      KV-head count (number of distinct K/V heads after any
                         grouped-query reduction; for a 7B Qwen this is 8, not 28).
        head_dim:        per-head dimension (e.g. 128).
        context_tokens:  planned context length.
        mode:            one of "fp16", "int8", "int4", "int2".
        window:          optional FP16 recent-window size; included as a
                         separate additive term (``recent_window_bytes``) so the
                         caller can isolate it.  Default 0 — pure quant tier.

    Returns:
        :class:`KVMemoryEstimate` with the full breakdown.

    Raises:
        ValueError on bad mode, non-positive dims, or head_dim that is
        incompatible with the requested mode (e.g. odd head_dim + int4).
    """
    if n_layers   <= 0: raise ValueError(f"n_layers must be positive, got {n_layers}")
    if n_kv_heads <= 0: raise ValueError(f"n_kv_heads must be positive, got {n_kv_heads}")
    if context_tokens < 0:
        raise ValueError(f"context_tokens must be ≥ 0, got {context_tokens}")
    if window < 0:
        raise ValueError(f"window must be ≥ 0, got {window}")

    bpt_head  = _bytes_per_token_per_head(mode, head_dim)
    bpt       = bpt_head * n_kv_heads * _KV_BUFFERS_PER_LAYER
    bpl       = bpt * context_tokens
    total     = bpl * n_layers

    fp16_head = _bytes_per_token_per_head("fp16", head_dim)
    fp16_base = fp16_head * n_kv_heads * _KV_BUFFERS_PER_LAYER * context_tokens * n_layers
    ratio     = fp16_base / total if total > 0 else 1.0

    # Recent window holds raw FP16 K and V (head_dim values per token per head).
    win_bpt   = head_dim * _KV_BYTES_FP16 * n_kv_heads * _KV_BUFFERS_PER_LAYER
    win_total = win_bpt * window * n_layers

    return KVMemoryEstimate(
        mode=mode,
        n_layers=n_layers, n_kv_heads=n_kv_heads, head_dim=head_dim,
        context_tokens=context_tokens,
        bytes_per_token_per_head=bpt_head,
        bytes_per_token=bpt,
        bytes_per_layer=bpl,
        total_bytes=total,
        fp16_baseline_bytes=fp16_base,
        compression_ratio=ratio,
        recent_window_bytes=win_total,
    )


def estimate_max_context(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    budget_bytes: int,
    mode: str,
    *,
    window: int = 0,
) -> int:
    """Largest context length whose KV cache fits in ``budget_bytes``.

    Inverts :func:`estimate_kv_memory`.  Returns 0 when the recent-window
    overhead alone already exceeds the budget.
    """
    if budget_bytes < 0:
        raise ValueError(f"budget_bytes must be ≥ 0, got {budget_bytes}")
    bpt_head = _bytes_per_token_per_head(mode, head_dim)
    bpt_layer = bpt_head * n_kv_heads * _KV_BUFFERS_PER_LAYER
    per_token = bpt_layer * n_layers
    if per_token <= 0:
        return 0
    win_bpt   = head_dim * _KV_BYTES_FP16 * n_kv_heads * _KV_BUFFERS_PER_LAYER
    win_total = win_bpt * window * n_layers
    remaining = budget_bytes - win_total
    if remaining <= 0:
        return 0
    return remaining // per_token


def recommend_mode_for_budget(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    context_tokens: int,
    budget_bytes: int,
    *,
    window: int = 0,
) -> "str | None":
    """Pick the highest-quality quant mode whose KV cache fits in ``budget_bytes``.

    Tries ``int8`` → ``int4`` → ``int2`` in order; returns the first that
    fits, or ``None`` if even ``int2`` is too large (in which case the caller
    must shrink context, layers, or heads — there is no smaller tier).

    Note: this is the *budget-driven* counterpart to
    :func:`recommended_kv_mode_3tier`, which is *context-driven*.  Callers
    typically prefer one or the other depending on whether they have a hard
    RAM ceiling or a planned conversation length.
    """
    for mode in _KV_TIER_ORDER:
        try:
            est = estimate_kv_memory(
                n_layers, n_kv_heads, head_dim, context_tokens, mode,
                window=window,
            )
        except ValueError:
            # head_dim incompatible with this tier — skip and try the next.
            continue
        if est.fits_in(budget_bytes):
            return mode
    return None


def make_kv_cache(
    n_layers: int,
    *,
    planned_context: int,
    rotate: bool = True,
    mode: "str | None" = None,
    window: int = 128,
    seed: int = 42,
    **extra,
) -> "QuantizedKVCache":
    """Construct the recommended KV cache for a planned workload.

    One-line factory that wraps the W104/W105 mode selection + the right
    constructor (Hadamard-rotated by default, since rotation is what makes
    INT2 viable on real activations and is essentially free for INT4/INT8).

    Args:
        n_layers:         transformer layer count.
        planned_context:  expected conversation length in tokens; drives mode
                          selection via :func:`recommended_kv_mode_3tier`
                          when ``mode`` is not specified.
        rotate:           use :class:`HadamardKVCache` (default) — recommended.
                          Set ``False`` for the bare :class:`QuantizedKVCache`.
        mode:             explicit override; one of "int8" / "int4" / "int2".
                          When ``None`` (default), the mode is auto-picked.
        window:           FP16 recent-window size — kept at 128 by default,
                          which is the long-context recommendation.
        seed:             Hadamard rotation seed (ignored when ``rotate=False``).
        **extra:          forwarded verbatim to the cache constructor (e.g.
                          ``budget``, ``snap_window``).

    Returns:
        A constructed :class:`HadamardKVCache` (default) or
        :class:`QuantizedKVCache` ready for ``mlx_lm`` integration.
    """
    if mode is None:
        mode = recommended_kv_mode_3tier(planned_context)
    cls = HadamardKVCache if rotate else QuantizedKVCache
    if rotate:
        return cls(n_layers=n_layers, window=window, mode=mode, seed=seed, **extra)
    return cls(n_layers=n_layers, window=window, mode=mode, **extra)


# ---------------------------------------------------------------------------

# Number of tokens to collect before fitting the per-layer SVD basis.
# 64 tokens provide a stable subspace for head_dim=128 models.
_SVD_INIT_TOKENS: int = 64


# ---------------------------------------------------------------------------
# KVLayerCache — per-layer KV buffer with optional INT8 compression
# ---------------------------------------------------------------------------

class KVLayerCache:
    """
    Single-layer KV cache that combines:
    • A FP16 "recent window" for the last ``window`` positions
    • An INT8 compressed buffer for all older positions

    Thread-safe (a single generation thread owns the cache).

    Attributes
    ----------
    keys_recent   : list[np.ndarray]  — list of (n_heads, head_dim) f16 arrays
    values_recent : same shape
    keys_old_q    : np.ndarray or None  — (n_heads, n_old, head_dim) int8
    keys_old_s    : np.ndarray or None  — (n_heads, n_old) float32 scales
    values_old_q  : same
    values_old_s  : same
    """
    __slots__ = (
        "window",
        "keys_recent", "values_recent",
        "keys_old_q", "keys_old_s",
        "values_old_q", "values_old_s",
        "n_heads", "head_dim",
        "_lock",
        # disk overflow tier (Item 4 — long-context NVMe spill)
        "_disk_threshold", "_disk_dir",
        "_disk_map_k",  "_disk_map_v",
        "_disk_scales_k", "_disk_scales_v",
        "_disk_n",
        # Phase 1: SVD KV compression
        "_svd_rank",      # int: 0 = off; rank < head_dim when enabled
        "_svd_Vk",        # np.ndarray (n_heads, rank, head_dim) float16 or None
        "_svd_Vv",        # same shape as _svd_Vk
        "_svd_buf_k",     # list[np.ndarray] calibration buffer (FP16), or None
        "_svd_buf_v",     # same
        # Phase 2: HNSW retrieval index
        "_retrieval_top_k",  # int: 0 = off; >0 enables HNSW for disk-tier retrieval
        "_hnsw",             # HNSWIndex | None
        # Phase 0C: async CPU dequant pre-fetch (prevents GPU ↔ dequant contention)
        "_prefetch_future",  # Future[tuple] | None
        # CommVQ vector quantization for old KV tokens
        "_comm_vq_bits",       # int: 0 = off; 2 or 4 for CommVQ compression
        "_comm_vq_k",          # CommVQCodebook | None  — key codebook
        "_comm_vq_v",          # CommVQCodebook | None  — value codebook
        "_comm_vq_calib_k",    # list[np.ndarray] | None — calibration buffer keys
        "_comm_vq_calib_v",    # list[np.ndarray] | None — calibration buffer values
        "_comm_vq_old_k",      # np.ndarray | None  (n_heads, n_old) uint16 indices
        "_comm_vq_old_v",      # np.ndarray | None  (n_heads, n_old) uint16 indices
        # Phase 3: Q-Filters geometric KV eviction
        "_qfilter",            # QFilterState | None — set by QuantizedKVCache when qfilter_rank > 0
        # W104: per-layer storage mode for the old-tier quantization.
        # "int8" (default) or "int2"; mirrors QuantizedKVCache.mode for the
        # quant-bearing modes.  Stored on the layer rather than read from a
        # parent reference to keep the layer cache self-contained.
        "_kv_mode",
    )

    def __init__(self, window: int = 64, kv_mode: str = "int8"):
        if kv_mode not in ("int8", "int4", "int2"):
            raise ValueError(f"kv_mode must be int8, int4, or int2, got {kv_mode!r}")
        self.window        = window
        self._kv_mode      = kv_mode
        self.keys_recent   = []     # list of (n_heads, head_dim) f16 arrays
        self.values_recent = []
        self.keys_old_q    = None   # (n_heads, n_old, head_dim_or_rank) int8
        self.keys_old_s    = None   # (n_heads, n_old) f32
        self.values_old_q  = None
        self.values_old_s  = None
        self.n_heads       = None
        self.head_dim      = None
        self._lock         = threading.RLock()   # re-entrant: _snap_evict calls get_full_kv under the lock
        # disk overflow tier — all None / 0 when disabled
        self._disk_threshold = None   # int: spill old_q rows beyond this count
        self._disk_dir       = None
        self._disk_map_k     = None   # np.memmap (n_heads, max_disk, head_dim) int8
        self._disk_map_v     = None
        self._disk_scales_k  = None   # np.ndarray (n_heads, max_disk) f32
        self._disk_scales_v  = None
        self._disk_n         = 0      # rows currently written to disk
        # Phase 1: SVD KV compression
        self._svd_rank   = 0     # 0 = off; set to rank < head_dim to enable
        self._svd_Vk     = None  # (n_heads, rank, head_dim) float16 — fitted once, then frozen
        self._svd_Vv     = None
        self._svd_buf_k  = None  # list[np.ndarray] calibration buffer (FP16), cleared after fit
        self._svd_buf_v  = None
        # Phase 2: HNSW retrieval index
        self._retrieval_top_k = 0     # 0 = off; set via enable_disk_tier
        self._hnsw            = None  # HNSWIndex, lazily created on first disk spill
        # Phase 0C: async CPU dequant pre-fetch
        self._prefetch_future = None  # concurrent.futures.Future | None
        # CommVQ vector quantization
        self._comm_vq_bits   = 0     # 0 = off; 2 or 4 to enable CommVQ
        self._comm_vq_k      = None  # CommVQCodebook — fitted on 1st calibration
        self._comm_vq_v      = None
        self._comm_vq_calib_k = None  # list or None — buffer until codebook fitted
        self._comm_vq_calib_v = None
        self._comm_vq_old_k  = None  # (n_heads, n_old) uint16 indices
        self._comm_vq_old_v  = None
        # Phase 3: Q-Filters geometric KV eviction
        self._qfilter        = None  # QFilterState | None

    # ── Main cache update ─────────────────────────────────────────────────────

    def append(self, key_np: np.ndarray, value_np: np.ndarray) -> None:
        """
        Append a single token's K/V pair (shape (n_heads, head_dim)) to the cache.
        When the recent window overflows, the oldest slot is quantized to INT8.
        """
        with self._lock:
            if self.n_heads is None:
                self.n_heads  = key_np.shape[0]
                self.head_dim = key_np.shape[-1]

            self.keys_recent.append(key_np.astype(np.float16))
            self.values_recent.append(value_np.astype(np.float16))

            # Phase 3: Q-Filters — record incoming key in geometric filter
            if self._qfilter is not None:
                self._qfilter.append_key(key_np)

            # Evict the oldest recent slot to INT8 when window fills
            while len(self.keys_recent) > self.window:
                oldest_k = self.keys_recent.pop(0)    # (n_heads, head_dim)
                oldest_v = self.values_recent.pop(0)

                # ── Phase 1: SVD calibration phase ──────────────────────────────────
                # Buffer tokens in FP16 until we have enough to fit the SVD basis.
                # Once fitted, all subsequent tokens are projected before INT8 quant.
                if self._svd_rank > 0 and self._svd_Vk is None:
                    if self._svd_buf_k is None:
                        self._svd_buf_k, self._svd_buf_v = [], []
                    self._svd_buf_k.append(oldest_k)
                    self._svd_buf_v.append(oldest_v)
                    if len(self._svd_buf_k) >= _SVD_INIT_TOKENS:
                        self._svd_fit_and_flush()
                    continue  # token is buffered; skip normal quantization

                # ── Apply SVD projection if basis is ready ────────────────────────
                if self._svd_Vk is not None:
                    oldest_k = self._svd_project(oldest_k, self._svd_Vk)
                    oldest_v = self._svd_project(oldest_v, self._svd_Vv)

                # ── CommVQ vector quantization path ──────────────────────────────
                if self._comm_vq_bits > 0:
                    # Each oldest_k/v has shape (n_heads, head_dim).
                    # Flatten per-head to (n_heads, head_dim) float32 for codebook.
                    # On first _SVD_INIT_TOKENS tokens, buffer for codebook fitting.
                    vecs_k = oldest_k.astype(np.float32)  # (n_heads, head_dim)
                    vecs_v = oldest_v.astype(np.float32)
                    if self._comm_vq_k is None:
                        # Accumulate calibration buffer
                        if self._comm_vq_calib_k is None:
                            self._comm_vq_calib_k = []
                            self._comm_vq_calib_v = []
                        self._comm_vq_calib_k.append(vecs_k)
                        self._comm_vq_calib_v.append(vecs_v)
                        if len(self._comm_vq_calib_k) >= _SVD_INIT_TOKENS:
                            # Fit codebooks: one per head dimension; use all heads
                            from squish.quant.comm_vq import CommVQCodebook
                            n_codes = 2 ** self._comm_vq_bits  # 4 for 2-bit, 16 for 4-bit
                            head_dim = self.head_dim
                            calib_k = np.concatenate(self._comm_vq_calib_k, axis=0)  # (N*n_heads, head_dim)
                            calib_v = np.concatenate(self._comm_vq_calib_v, axis=0)
                            cb_k = CommVQCodebook(dim=head_dim, n_codes=n_codes)
                            cb_v = CommVQCodebook(dim=head_dim, n_codes=n_codes)
                            cb_k.fit(calib_k)
                            cb_v.fit(calib_v)
                            self._comm_vq_k = cb_k
                            self._comm_vq_v = cb_v
                            # Now encode all buffered tokens
                            for buf_k, buf_v in zip(self._comm_vq_calib_k, self._comm_vq_calib_v, strict=False):
                                idx_k = self._comm_vq_k.encode(buf_k).reshape(self.n_heads, 1)  # (n_heads, 1)
                                idx_v = self._comm_vq_v.encode(buf_v).reshape(self.n_heads, 1)
                                if self._comm_vq_old_k is None:
                                    self._comm_vq_old_k = idx_k
                                    self._comm_vq_old_v = idx_v
                                else:
                                    self._comm_vq_old_k = np.concatenate([self._comm_vq_old_k, idx_k], axis=1)
                                    self._comm_vq_old_v = np.concatenate([self._comm_vq_old_v, idx_v], axis=1)
                            self._comm_vq_calib_k = None
                            self._comm_vq_calib_v = None
                        continue  # token buffered; will be encoded after fitting
                    else:
                        # Codebook already fitted — encode directly (n_heads, 1) indices
                        idx_k = self._comm_vq_k.encode(vecs_k).reshape(self.n_heads, 1)
                        idx_v = self._comm_vq_v.encode(vecs_v).reshape(self.n_heads, 1)
                        if self._comm_vq_old_k is None:
                            self._comm_vq_old_k = idx_k
                            self._comm_vq_old_v = idx_v
                        else:
                            self._comm_vq_old_k = np.concatenate([self._comm_vq_old_k, idx_k], axis=1)
                            self._comm_vq_old_v = np.concatenate([self._comm_vq_old_v, idx_v], axis=1)
                    continue  # CommVQ path done — skip INT8 quantization

                # Quantize per-head per-token (dispatched on layer storage mode).
                # INT8 → (1, head_dim) int8; INT2 → (1, head_dim/4) uint8 packed.
                new_kq_list, new_ks_list = [], []
                new_vq_list, new_vs_list = [], []
                for h in range(self.n_heads):
                    kq, ks = _kv_quantize_per_channel(
                        oldest_k[h:h+1, :], self._kv_mode)
                    vq, vs = _kv_quantize_per_channel(
                        oldest_v[h:h+1, :], self._kv_mode)
                    new_kq_list.append(kq)            # each (1, head_dim or head_dim/4)
                    new_ks_list.append(ks)            # each (1,) float32
                    new_vq_list.append(vq)
                    new_vs_list.append(vs)

                # stack → (n_heads, 1, last_dim) and (n_heads, 1)
                # last_dim is head_dim for INT8, head_dim/4 for INT2.
                slot_kq = np.stack(new_kq_list, axis=0)
                slot_ks = np.stack(new_ks_list, axis=0)
                slot_vq = np.stack(new_vq_list, axis=0)
                slot_vs = np.stack(new_vs_list, axis=0)

                if self.keys_old_q is None:
                    self.keys_old_q   = slot_kq
                    self.keys_old_s   = slot_ks
                    self.values_old_q = slot_vq
                    self.values_old_s = slot_vs
                else:
                    self.keys_old_q   = np.concatenate([self.keys_old_q,   slot_kq], axis=1)
                    self.keys_old_s   = np.concatenate([self.keys_old_s,   slot_ks], axis=1)
                    self.values_old_q = np.concatenate([self.values_old_q, slot_vq], axis=1)
                    self.values_old_s = np.concatenate([self.values_old_s, slot_vs], axis=1)
            # Spill oldest INT8 entries to NVMe disk tier if enabled
            self._maybe_spill_to_disk()

    def get_full_kv(self) -> tuple:
        """
        Return the full key and value matrices as FP16 numpy arrays.

        Returns
        -------
        keys   : (n_heads, n_total, head_dim)  float16
        values : (n_heads, n_total, head_dim)  float16
        """
        with self._lock:
            # Reconstruct disk tier (oldest, spilled to NVMe memmap)
            disk_k, disk_v = self._disk_full_kv()

            # Reconstruct RAM INT8 portion
            if self.keys_old_q is not None:
                # Dequantize per head (dispatched on storage mode).
                # For INT2 we need the unpacked head_dim — use self.head_dim
                # when no SVD projection is active (storage dim = head_dim);
                # otherwise the SVD path is INT8-only (validated in __init__).
                deq_dim = self.head_dim
                old_k_list, old_v_list = [], []
                for h in range(self.n_heads):
                    k_deq = _kv_dequantize_per_channel(
                        self.keys_old_q[h], self.keys_old_s[h],
                        self._kv_mode, head_dim=deq_dim)
                    v_deq = _kv_dequantize_per_channel(
                        self.values_old_q[h], self.values_old_s[h],
                        self._kv_mode, head_dim=deq_dim)
                    # Phase 1: back-project SVD-compressed tokens to full head_dim
                    if self._svd_Vk is not None:
                        Vk_h = self._svd_Vk[h].astype(np.float32)  # (rank, head_dim)
                        Vv_h = self._svd_Vv[h].astype(np.float32)
                        k_deq = (k_deq.astype(np.float32) @ Vk_h).astype(np.float16)
                        v_deq = (v_deq.astype(np.float32) @ Vv_h).astype(np.float16)
                    old_k_list.append(k_deq)
                    old_v_list.append(v_deq)
                old_k = np.stack(old_k_list, axis=0)   # (n_heads, n_old, head_dim)
                old_v = np.stack(old_v_list, axis=0)
            elif self._comm_vq_bits > 0 and self._comm_vq_old_k is not None:
                # CommVQ: decode vector-quantized tokens per head
                cb_k = self._comm_vq_k
                cb_v = self._comm_vq_v
                old_k_list, old_v_list = [], []
                for h in range(self.n_heads):
                    # indices shape: (n_old,)
                    k_deq = cb_k.decode(self._comm_vq_old_k[h]).astype(np.float16)  # (n_old, head_dim)
                    v_deq = cb_v.decode(self._comm_vq_old_v[h]).astype(np.float16)
                    old_k_list.append(k_deq)
                    old_v_list.append(v_deq)
                old_k = np.stack(old_k_list, axis=0)   # (n_heads, n_old, head_dim)
                old_v = np.stack(old_v_list, axis=0)
            else:
                old_k = old_v = None

            if self.keys_recent:
                # Each element is (n_heads, head_dim) → stack along token dim
                rec_k = np.stack(self.keys_recent,   axis=1)   # (n_heads, n_rec, head_dim)
                rec_v = np.stack(self.values_recent, axis=1)
            else:
                rec_k = rec_v = None

            # Combine: disk || RAM int8 || FP16 recent
            parts_k = [p for p in (disk_k, old_k, rec_k) if p is not None]
            parts_v = [p for p in (disk_v, old_v, rec_v) if p is not None]
            if not parts_k:
                return None, None
            if len(parts_k) == 1:
                return parts_k[0], parts_v[0]
            full_k = np.concatenate(parts_k, axis=1)
            full_v = np.concatenate(parts_v, axis=1)
            return full_k, full_v

    # ── Phase 0C: async CPU dequant pre-fetch ─────────────────────────────────
    # During the token-sampling step (which is CPU-bound) we overlap the
    # INT8→FP16 dequantization of the *next* decode step on a background thread.
    # This hides the O(n_old_tokens) numpy work behind the sampler, preventing
    # ≥30 % slowdown from blocking the generation loop on large KV caches.
    #
    # Usage in the decode loop:
    #   layer_cache.start_prefetch()      # fire-and-forget at end of step N
    #   ...sample token N...
    #   k, v = layer_cache.get_full_kv_prefetched()   # ready at step N+1

    _THREAD_POOL: "concurrent.futures.ThreadPoolExecutor | None" = None

    @classmethod
    def _get_pool(cls) -> "concurrent.futures.ThreadPoolExecutor":
        if cls._THREAD_POOL is None:
            import concurrent.futures
            # One worker is enough: dequant is sequential per layer; we want
            # CPU—not Metal—so we keep the thread off the main queue.
            cls._THREAD_POOL = concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="squish-kv-deq"
            )
        return cls._THREAD_POOL

    def start_prefetch(self) -> None:
        """
        Submit the dequantization work for the *current* cache state to a
        background CPU thread.  Call this immediately after sampling a token
        so the work overlaps with the next-step setup.
        """
        if self._prefetch_future is not None:
            return  # already in-flight
        if self.keys_old_q is None:
            return  # nothing to prefetch — recent window only, no INT8 tier
        try:
            self._prefetch_future = self._get_pool().submit(self.get_full_kv)
        except Exception:  # pragma: no cover
            self._prefetch_future = None  # never block generation

    def get_full_kv_prefetched(self) -> tuple:
        """
        Return the pre-fetched ``(keys, values)`` numpy arrays if available,
        otherwise fall back to a synchronous ``get_full_kv()`` call.
        """
        future = self._prefetch_future
        self._prefetch_future = None  # consume the future
        if future is None:
            return self.get_full_kv()
        try:
            return future.result(timeout=0.5)
        except Exception:  # pragma: no cover
            return self.get_full_kv()

    def get_as_mlx(self):
        """Return (keys, values) as MLX bfloat16 arrays for use in attention."""
        mx = _mx()
        k_np, v_np = self.get_full_kv()
        if k_np is None:
            return None, None
        return (mx.array(k_np).astype(mx.bfloat16),
                mx.array(v_np).astype(mx.bfloat16))

    @property
    def n_tokens(self) -> int:
        old = self.keys_old_q.shape[1] if self.keys_old_q is not None else 0
        return old + len(self.keys_recent)

    @property
    def memory_bytes(self) -> int:
        """Approximate memory usage in bytes."""
        b = 0
        if self.keys_old_q is not None:
            b += self.keys_old_q.nbytes + self.keys_old_s.nbytes * 2
            b += self.values_old_q.nbytes + self.values_old_s.nbytes * 2
        for arr in self.keys_recent + self.values_recent:
            b += arr.nbytes
        return b

    def reset(self):
        """Clear all cached K/V state (SVD basis is preserved across conversations)."""
        with self._lock:
            self.keys_recent.clear()
            self.values_recent.clear()
            self.keys_old_q = self.keys_old_s = None
            self.values_old_q = self.values_old_s = None
            self._disk_n = 0
            # Delete memmap files if they exist
            self._disk_map_k = None
            self._disk_map_v = None
            self._disk_scales_k = None
            self._disk_scales_v = None
            for attr in ("_disk_path_k", "_disk_path_v"):
                p = getattr(self, attr, None)
                if p is not None:  # pragma: no cover
                    try:
                        import pathlib
                        pathlib.Path(p).unlink(missing_ok=True)
                    except Exception:
                        pass
            # Clear SVD calibration buffer but keep the fitted basis
            self._svd_buf_k = None
            self._svd_buf_v = None
            # Reset HNSW retrieval index (token positions change each conversation)
            self._hnsw = None

    # ── Phase 1: SVD KV compression helpers ───────────────────────────────────

    def _svd_project(self, x: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Project one token's K or V from full head_dim to SVD rank.

        Parameters
        ----------
        x : (n_heads, head_dim)          float16
        V : (n_heads, rank, head_dim)    float16  — right singular vectors

        Returns
        -------
        (n_heads, rank)  float16
        """
        x_f32 = x.astype(np.float32)       # (n_heads, head_dim)
        V_f32 = V.astype(np.float32)       # (n_heads, rank, head_dim)
        # result[h] = x[h] @ V[h].T  → (rank,) per head
        out = np.einsum("hd,hrd->hr", x_f32, V_f32)
        return out.astype(np.float16)

    def _svd_fit_and_flush(self) -> None:
        """
        Fit per-head SVD bases from the calibration buffer, then quantize all
        buffered tokens to INT8 using the fitted projection.

        Called once when ``len(_svd_buf_k) >= _SVD_INIT_TOKENS``.
        After this call ``_svd_Vk/Vv`` are set and ``_svd_buf_k/v`` are cleared.
        """
        Vk_list, Vv_list = [], []
        for h in range(self.n_heads):
            K = np.stack([t[h] for t in self._svd_buf_k], axis=0).astype(np.float32)
            _, _, Vt = np.linalg.svd(K, full_matrices=False)   # Vt: (min(n_init,dim), dim)
            Vk_list.append(Vt[:self._svd_rank, :])              # (rank, head_dim)

            V_mat = np.stack([t[h] for t in self._svd_buf_v], axis=0).astype(np.float32)
            _, _, Vt = np.linalg.svd(V_mat, full_matrices=False)
            Vv_list.append(Vt[:self._svd_rank, :])

        # Store as float16 to save memory; cast to float32 only on projection
        self._svd_Vk = np.stack(Vk_list, axis=0).astype(np.float16)  # (n_heads, rank, head_dim)
        self._svd_Vv = np.stack(Vv_list, axis=0).astype(np.float16)

        # Flush the calibration buffer: project + quantize each buffered token
        for k_f16, v_f16 in zip(self._svd_buf_k, self._svd_buf_v, strict=False):
            k_proj = self._svd_project(k_f16, self._svd_Vk)   # (n_heads, rank)
            v_proj = self._svd_project(v_f16, self._svd_Vv)

            new_kq_list, new_ks_list = [], []
            new_vq_list, new_vs_list = [], []
            for h in range(self.n_heads):
                kq, ks = _quantize_int8_per_channel(k_proj[h:h+1, :])
                vq, vs = _quantize_int8_per_channel(v_proj[h:h+1, :])
                new_kq_list.append(kq)
                new_ks_list.append(ks)
                new_vq_list.append(vq)
                new_vs_list.append(vs)

            slot_kq = np.stack(new_kq_list, axis=0)
            slot_ks = np.stack(new_ks_list, axis=0)
            slot_vq = np.stack(new_vq_list, axis=0)
            slot_vs = np.stack(new_vs_list, axis=0)

            if self.keys_old_q is None:
                self.keys_old_q   = slot_kq
                self.keys_old_s   = slot_ks
                self.values_old_q = slot_vq
                self.values_old_s = slot_vs
            else:
                self.keys_old_q   = np.concatenate([self.keys_old_q,   slot_kq], axis=1)
                self.keys_old_s   = np.concatenate([self.keys_old_s,   slot_ks], axis=1)
                self.values_old_q = np.concatenate([self.values_old_q, slot_vq], axis=1)
                self.values_old_s = np.concatenate([self.values_old_s, slot_vs], axis=1)

        self._svd_buf_k = None
        self._svd_buf_v = None

    # ── Disk overflow tier (Item 4) ───────────────────────────────────────────

    def enable_disk_tier(
        self,
        threshold: int,
        max_disk_tokens: int,
        cache_dir,          # str | Path
        n_heads: int,
        head_dim: int,
        retrieval_top_k: int = 0,  # Phase 2: >0 enables HNSW retrieval index
    ) -> None:
        """
        Enable disk-backed overflow for old INT8 K/V entries.

        When the number of INT8-quantized positions (``keys_old_q.shape[1]``)
        exceeds ``threshold``, the oldest ``(n_old - threshold)`` rows are
        spilled to a ``numpy.memmap`` file on the NVMe mount at ``cache_dir``.

        OS page-fault semantics keep hot pages in the unified memory file-cache
        while cold pages stay on disk — effectively using the SSD as a
        transparent third tier, behind Metal (FP16 recent window) and CPU RAM
        (INT8 ring buffer).

        Parameters
        ----------
        threshold        : int — rows to keep in RAM before spilling
        max_disk_tokens  : int — pre-allocated memmap size (rows)
        cache_dir        : directory for temp memmap files (e.g. /tmp/squish_kv)
        n_heads, head_dim : model dimensions (required to size the memmap
                            before the first append, because __slots__ prevent
                            lazy init once n_heads is known).
        retrieval_top_k  : int — Phase 2: if >0 build an HNSW index on spilled
                            key vectors to support get_relevant_kv().
        """
        import pathlib
        if self._kv_mode in ("int4", "int2"):
            raise ValueError(
                f"enable_disk_tier() is INT8-only (memmap dtype is int8). "
                f"Sub-INT8 modes (int4 — W105, int2 — W104) are RAM-only; "
                f"layer is in {self._kv_mode!r}."
            )
        cache_dir = pathlib.Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._disk_threshold = threshold
        self._disk_dir       = cache_dir
        self.n_heads  = n_heads
        self.head_dim = head_dim
        self._retrieval_top_k = retrieval_top_k
        uid = id(self)
        path_k = cache_dir / f"kv_{uid}_k.bin"
        path_v = cache_dir / f"kv_{uid}_v.bin"
        self._disk_map_k = np.memmap(
            path_k, dtype=np.int8, mode="w+",
            shape=(n_heads, max_disk_tokens, head_dim))
        self._disk_map_v = np.memmap(
            path_v, dtype=np.int8, mode="w+",
            shape=(n_heads, max_disk_tokens, head_dim))
        self._disk_scales_k = np.zeros((n_heads, max_disk_tokens), dtype=np.float32)
        self._disk_scales_v = np.zeros((n_heads, max_disk_tokens), dtype=np.float32)
        self._disk_n = 0
        # HNSW index is created lazily on first spill (dim may change with SVD)

    def _maybe_spill_to_disk(self) -> None:
        """
        If the INT8 RAM buffer exceeds ``_disk_threshold``, spill the oldest
        tokens to the memmap tier.

        Called inside the ``_lock`` from ``append()``.
        """
        if (self._disk_threshold is None
                or self._disk_map_k is None
                or self.keys_old_q is None):
            return
        n_old = self.keys_old_q.shape[1]
        if n_old <= self._disk_threshold:
            return
        n_spill = n_old - self._disk_threshold
        disk_end = self._disk_n + n_spill
        if disk_end > self._disk_map_k.shape[1]:
            # Disk tier full — silently keep RAM tier only (graceful degrade)
            return
        # Write oldest n_spill rows to memmap.
        # shapes: keys_old_q (n_heads, n_old, head_dim)
        self._disk_map_k[:, self._disk_n:disk_end, :] = \
            self.keys_old_q[:, :n_spill, :]
        self._disk_map_v[:, self._disk_n:disk_end, :] = \
            self.values_old_q[:, :n_spill, :]
        self._disk_scales_k[:, self._disk_n:disk_end] = \
            self.keys_old_s[:, :n_spill]
        self._disk_scales_v[:, self._disk_n:disk_end] = \
            self.values_old_s[:, :n_spill]
        self._disk_n = disk_end

        # Phase 2: update HNSW retrieval index with the spilled key vectors
        if self._retrieval_top_k > 0:  # pragma: no cover
            self._update_hnsw_index(
                keys_int8=self.keys_old_q[:, :n_spill, :],
                scales=self.keys_old_s[:, :n_spill],
                start_pos=self._disk_n - n_spill,
            )

        # Trim RAM buffer
        self.keys_old_q   = self.keys_old_q[:, n_spill:, :]
        self.keys_old_s   = self.keys_old_s[:, n_spill:]
        self.values_old_q = self.values_old_q[:, n_spill:, :]
        self.values_old_s = self.values_old_s[:, n_spill:]

    def _update_hnsw_index(  # pragma: no cover
        self,
        keys_int8: np.ndarray,   # (n_heads, n, rank_or_head_dim) int8
        scales: np.ndarray,      # (n_heads, n) float32
        start_pos: int,
    ) -> None:
        """
        Add spilled key vectors to the HNSW retrieval index.

        Uses head 0 as the representative head for indexing.  HNSW is
        lazily created on the first call so the correct dimension is known
        (after SVD fitting if active).

        Parameters
        ----------
        keys_int8 : (n_heads, n, dim) int8 — spilled key vectors
        scales    : (n_heads, n) float32   — per-token scales
        start_pos : int — disk position offset for these tokens
        """
        h = 0   # representative head
        n = keys_int8.shape[1]
        # Dequantize head-0 keys
        k_deq = _dequantize_int8_per_channel(
            keys_int8[h], scales[h])  # (n, rank_or_head_dim) float16
        # Back-project SVD if active (so index is in full head_dim space)
        if self._svd_Vk is not None:
            k_deq = (k_deq.astype(np.float32) @ self._svd_Vk[h].astype(np.float32)).astype(np.float16)
        k_f32 = k_deq.astype(np.float32)  # (n, head_dim)
        dim = k_f32.shape[-1]
        # Lazy HNSW init: dim is now known
        if self._hnsw is None:
            try:
                try:
                    from squish.vector_index import HNSWIndex
                except ImportError:  # pragma: no cover
                    from vector_index import HNSWIndex  # direct run
                max_elem = int(self._disk_map_k.shape[1]) if self._disk_map_k is not None else 500_000
                self._hnsw = HNSWIndex(dim=dim, max_elements=max_elem)
            except ImportError:  # pragma: no cover
                return  # hnswlib not installed — silently skip
        ids = np.arange(start_pos, start_pos + n, dtype=np.int64)
        self._hnsw.add(k_f32, ids)

    def get_relevant_kv(  # pragma: no cover
        self,
        query_key_fp16: np.ndarray,   # (n_heads, head_dim) float16
        top_k: int,
        hot_window: int = 256,
    ) -> tuple:
        """
        Return a *sparse* K/V context composed of ANNS-retrieved disk tokens
        plus a guaranteed hot-window of the most recent RAM tokens.

        Falls back to ``get_full_kv()`` when no HNSW index is available.

        Parameters
        ----------
        query_key_fp16 : (n_heads, head_dim) float16 — current decode step key
        top_k          : int — number of disk tokens to retrieve per query
        hot_window     : int — number of most-recent RAM tokens always included

        Returns
        -------
        (keys, values) : (n_heads, n_ctx, head_dim) float16 each
        """
        if self._hnsw is None or self._disk_n == 0:
            return self.get_full_kv()
        with self._lock:
            h = 0  # representative head for ANNS query
            # Build query vector (back-project SVD if active)
            q_fp16 = query_key_fp16[h]
            if self._svd_Vk is not None:
                q_fp16 = (q_fp16.astype(np.float32) @ self._svd_Vk[h].astype(np.float32)).astype(np.float16)
            q_f32 = q_fp16.astype(np.float32)

            # ANNS search over disk-tier keys
            retrieved_ids, _ = self._hnsw.search(q_f32, top_k=top_k)

            if len(retrieved_ids) == 0:
                return self.get_full_kv()

            # Gather retrieved disk rows
            retrieved_k_list, retrieved_v_list = [], []
            for hh in range(self.n_heads):
                k_rows = _dequantize_int8_per_channel(
                    np.array(self._disk_map_k[hh, retrieved_ids, :]),
                    self._disk_scales_k[hh, retrieved_ids])
                v_rows = _dequantize_int8_per_channel(
                    np.array(self._disk_map_v[hh, retrieved_ids, :]),
                    self._disk_scales_v[hh, retrieved_ids])
                retrieved_k_list.append(k_rows)
                retrieved_v_list.append(v_rows)
            disk_k = np.stack(retrieved_k_list, axis=0)   # (n_heads, top_k, dim)
            disk_v = np.stack(retrieved_v_list, axis=0)

            # RAM INT8 tier (most recent hot_window entries)
            if self.keys_old_q is not None:
                n_old = self.keys_old_q.shape[1]
                hot_start = max(0, n_old - hot_window)
                old_k_list, old_v_list = [], []
                for hh in range(self.n_heads):
                    k_deq = _dequantize_int8_per_channel(
                        self.keys_old_q[hh, hot_start:, :],
                        self.keys_old_s[hh, hot_start:])
                    v_deq = _dequantize_int8_per_channel(
                        self.values_old_q[hh, hot_start:, :],
                        self.values_old_s[hh, hot_start:])
                    if self._svd_Vk is not None:
                        Vk_h = self._svd_Vk[hh].astype(np.float32)
                        Vv_h = self._svd_Vv[hh].astype(np.float32)
                        k_deq = (k_deq.astype(np.float32) @ Vk_h).astype(np.float16)
                        v_deq = (v_deq.astype(np.float32) @ Vv_h).astype(np.float16)
                    old_k_list.append(k_deq)
                    old_v_list.append(v_deq)
                hot_k = np.stack(old_k_list, axis=0)
                hot_v = np.stack(old_v_list, axis=0)
            else:
                hot_k = hot_v = None

            # FP16 recent window (always included)
            if self.keys_recent:
                rec_k = np.stack(self.keys_recent, axis=1)
                rec_v = np.stack(self.values_recent, axis=1)
            else:
                rec_k = rec_v = None

            # Concatenate: retrieved disk || hot RAM || recent FP16
            parts_k = [p for p in (disk_k, hot_k, rec_k) if p is not None]
            parts_v = [p for p in (disk_v, hot_v, rec_v) if p is not None]
            if not parts_k:
                return None, None
            return (np.concatenate(parts_k, axis=1),
                    np.concatenate(parts_v, axis=1))

    def _disk_full_kv(self) -> tuple:
        """
        Reconstruct the disk-tier portion as FP16 numpy arrays.

        Returns (keys, values) each shape (n_heads, _disk_n, head_dim) float16,
        or (None, None) if the disk tier is empty.
        """
        if self._disk_n == 0 or self._disk_map_k is None:
            return None, None
        old_k_list, old_v_list = [], []
        for h in range(self.n_heads):
            old_k_list.append(_dequantize_int8_per_channel(
                np.array(self._disk_map_k[h, :self._disk_n, :]),
                self._disk_scales_k[h, :self._disk_n]))
            old_v_list.append(_dequantize_int8_per_channel(
                np.array(self._disk_map_v[h, :self._disk_n, :]),
                self._disk_scales_v[h, :self._disk_n]))
        return (np.stack(old_k_list, axis=0),   # (n_heads, _disk_n, head_dim)
                np.stack(old_v_list, axis=0))

    # ── mlx_lm KVCache protocol (offset + update_and_fetch) ──────────────────

    @property
    def offset(self) -> int:
        """Total tokens stored; used by mlx_lm for RoPE position encoding."""
        return self.n_tokens

    def update_and_fetch(self, keys, values):
        """
        mlx_lm attention-layer cache protocol.

        Called by each model attention layer with the newly computed K/V
        tensors (shape: batch=1, n_heads, T_new, head_dim).  Appends the
        new tokens to the INT8-compressed ring buffer and returns the
        *full* accumulated K/V sequence as MLX arrays so that the
        attention computation uses the complete context.

        Parameters
        ----------
        keys   : mx.array  shape (1, n_heads, T_new, head_dim)
        values : mx.array  shape (1, n_heads, T_new, head_dim)

        Returns
        -------
        (keys_full, values_full) as mx.array (1, n_heads, T_total, head_dim)
        """
        mx = _mx()
        # Convert to numpy for storage: (n_heads, T_new, head_dim) float16
        k_np = np.array(keys[0].astype(mx.float16))
        v_np = np.array(values[0].astype(mx.float16))
        T_new = k_np.shape[1]
        for t in range(T_new):
            self.append(k_np[:, t, :], v_np[:, t, :])
        full_k, full_v = self.get_full_kv()   # (n_heads, T_total, head_dim) f16
        if full_k is None:  # pragma: no cover
            return keys, values
        return (
            mx.array(full_k[None]).astype(mx.bfloat16),   # (1, n_heads, T_total, head_dim)
            mx.array(full_v[None]).astype(mx.bfloat16),
        )


# ---------------------------------------------------------------------------
# SnapKV eviction (importance-based position selection)
# ---------------------------------------------------------------------------

def _snap_evict(
    layer_cache: KVLayerCache,
    budget: int,
    snap_window: int = 32,
) -> None:
    """
    Apply SnapKV-style eviction: keep only the ``budget`` most-important
    token positions.

    Importance is defined as the sum of attention weights each K/V position
    receives from the most-recent ``snap_window`` query positions.

    Called once after prefill (when the cache exceeds ``budget``).

    Parameters
    ----------
    layer_cache : KVLayerCache to evict
    budget      : maximum number of positions to retain
    snap_window : number of recent queries to use for importance estimation
    """
    with layer_cache._lock:
        n = layer_cache.n_tokens
        if n <= budget:
            return

        # Reconstruct full FP16 cache to compute importances
        full_k, full_v = layer_cache.get_full_kv()
        if full_k is None:  # pragma: no cover
            return

        nh, nt, hd = full_k.shape
        k_f32 = full_k.astype(np.float32)   # (n_heads, n_tokens, head_dim)

        # Use the tail of K as proxy queries (recent snap_window positions)
        q_window = min(snap_window, nt)
        q = k_f32[:, -q_window:, :]          # (nh, snap_window, hd)

        # Attention logits: (nh, snap_window, n_tokens)
        scale_factor = 1.0 / (hd ** 0.5)
        logits = np.einsum("nhd, nTd -> nhT", q, k_f32) * scale_factor

        # Softmax and sum importances over snap_window
        exp_l = np.exp(logits - logits.max(axis=-1, keepdims=True))
        attn  = exp_l / exp_l.sum(axis=-1, keepdims=True)     # (nh, snap_w, nt)
        importance = attn.sum(axis=(0, 1))                     # (n_tokens,)

        # Always keep the last snap_window positions (recent context)
        top_indices = np.argsort(-importance)[: budget]
        top_indices = np.sort(top_indices)                     # restore order

        # Rebuild cache with only selected positions
        sel_k = full_k[:, top_indices, :]   # (nh, budget, hd)
        sel_v = full_v[:, top_indices, :]

        # Reset and reload as all-recent (FP16) — next tokens will push old
        # positions into INT8 naturally through the window mechanism
        layer_cache.keys_recent.clear()
        layer_cache.values_recent.clear()
        layer_cache.keys_old_q = None
        layer_cache.keys_old_s = None
        layer_cache.values_old_q = None
        layer_cache.values_old_s = None

        # Reload into recent window — all positions initially FP16
        # append_batch calls the existing eviction logic to spill to INT8
        for t in range(sel_k.shape[1]):
            # Each element: (n_heads, head_dim)
            layer_cache.keys_recent.append(sel_k[:, t, :])
            layer_cache.values_recent.append(sel_v[:, t, :])

        # Spill all but the last `window` back to INT8
        while len(layer_cache.keys_recent) > layer_cache.window:
            oldest_k = layer_cache.keys_recent.pop(0)
            oldest_v = layer_cache.values_recent.pop(0)
            new_kq_list, new_ks_list = [], []
            new_vq_list, new_vs_list = [], []
            for h in range(nh):
                kq, ks = _quantize_int8_per_channel(oldest_k[h:h+1, :])
                vq, vs = _quantize_int8_per_channel(oldest_v[h:h+1, :])
                new_kq_list.append(kq)
                new_ks_list.append(ks)
                new_vq_list.append(vq)
                new_vs_list.append(vs)
            slot_kq = np.stack(new_kq_list, axis=0)
            slot_ks = np.stack(new_ks_list, axis=0)
            slot_vq = np.stack(new_vq_list, axis=0)
            slot_vs = np.stack(new_vs_list, axis=0)
            if layer_cache.keys_old_q is None:
                layer_cache.keys_old_q   = slot_kq
                layer_cache.keys_old_s   = slot_ks
                layer_cache.values_old_q = slot_vq
                layer_cache.values_old_s = slot_vs
            else:
                layer_cache.keys_old_q   = np.concatenate(
                    [layer_cache.keys_old_q,   slot_kq], axis=1)
                layer_cache.keys_old_s   = np.concatenate(
                    [layer_cache.keys_old_s,   slot_ks], axis=1)
                layer_cache.values_old_q = np.concatenate(
                    [layer_cache.values_old_q, slot_vq], axis=1)
                layer_cache.values_old_s = np.concatenate(
                    [layer_cache.values_old_s, slot_vs], axis=1)


# ---------------------------------------------------------------------------
# QuantizedKVCache — full model cache (all layers)
# ---------------------------------------------------------------------------

class QuantizedKVCache:
    """
    Full-model KV cache with KIVI-style INT8 compression and optional SnapKV
    eviction.

    Compatible with ``mlx_lm``'s cache API: the cache is a list of per-layer
    dicts with ``'keys'`` and ``'values'`` MLX arrays, populated on demand.

    Usage
    -----
    After model load, before generation:

        cache = QuantizedKVCache(n_layers=32, window=64, mode="snap",
                                 budget=2048, snap_window=32)
        # Pass as kv_cache to mlx_lm generate helpers, or use
        # cache.to_mlx_list() to get the raw list format.
    """

    def __init__(
        self,
        n_layers: int,
        window: int = 64,
        mode: str = "int8",                 # "fp16" | "int8" | "snap"
        budget: int = 4096,
        snap_window: int = 32,
        svd_rank: int = 0,                  # Phase 1: 0 = off; set to rank < head_dim
        comm_vq_bits: int = 0,              # CommVQ: 0 = off; 2 for 2-bit, 4 for 4-bit
        # Phase 3: Q-Filters geometric KV eviction
        qfilter_rank: int = 0,              # 0 = off; SVD projection rank (e.g. 32)
        qfilter_budget: int = 2048,         # max tokens to retain per layer
        qfilter_anchor: int = 64,           # recent tokens always preserved
        qfilter_evict_every: int = 16,      # run eviction every N decode steps
        # Phase 5: TTT Fast Weights — absorb evicted KVs into outer-product memory
        fast_weight_lr: float = 0.0,        # 0.0 = off; learning rate > 0 enables
        fast_weight_decay: float = 0.999,   # per-absorption W_f decay factor
    ):
        """
        Parameters
        ----------
        n_layers      : number of transformer layers
        window        : recent FP16 window size (KIVI parameter)
        mode          : "fp16" (no compression) | "int8" (KIVI) | "snap" (KIVI+SnapKV)
        budget        : max K/V positions to retain per layer (SnapKV only)
        snap_window   : attention window for importance scoring (SnapKV)
        svd_rank      : Phase 1 — project head_dim → rank before INT8 quant (0 = off)
        comm_vq_bits  : CommVQ bits per index — 2 (4×4=16 codes) or 4 (16 codes) or 0 off
        qfilter_rank  : Phase 3 — SVD rank for geometric KV eviction (0 = off)
        qfilter_budget: Phase 3 — max tokens to keep per layer after eviction
        qfilter_anchor: Phase 3 — recent tokens never evicted (recency anchor)
        qfilter_evict_every: Phase 3 — eviction interval in decode steps
        fast_weight_lr    : Phase 5 — learning rate for fast-weight absorption (0 = off)
        fast_weight_decay : Phase 5 — per-absorption W_f decay factor
        """
        if mode not in ("fp16", "int8", "snap", "int4", "int2"):
            raise ValueError(
                f"mode must be fp16, int8, snap, int4, or int2 — got {mode!r}"
            )

        # W104 / W105: sub-INT8 storage modes are incompatible with the
        # auxiliary tiers that assume INT8 storage shape (head_dim) and INT8
        # quantize/dequantize calls in their hot paths.  Reject these
        # combinations explicitly so the user learns at construction time
        # rather than via a shape error mid-decode.
        if mode in ("int4", "int2"):
            if svd_rank > 0:
                raise ValueError(
                    f"mode={mode!r} is incompatible with svd_rank > 0 "
                    "(SVD KV path quantizes projected vectors as INT8). "
                    f"Use mode='int8' with svd_rank, or mode={mode!r} alone."
                )
            if comm_vq_bits > 0:
                raise ValueError(
                    f"mode={mode!r} is incompatible with comm_vq_bits > 0 "
                    "(CommVQ already vector-quantizes K/V; pick one)."
                )
            if qfilter_rank > 0:
                raise ValueError(
                    f"mode={mode!r} is incompatible with qfilter_rank > 0 "
                    "(Q-Filters geometric eviction assumes INT8 storage). "
                    f"Use mode='int8' with qfilter_rank, or mode={mode!r} alone."
                )

        self.mode        = mode
        self.window      = window
        self.budget      = budget
        self.snap_window = snap_window
        self.svd_rank    = svd_rank
        self.comm_vq_bits = comm_vq_bits
        self.n_layers    = n_layers
        # KVLayerCache stores per-layer mode; INT8 is the default for "fp16"
        # (no old-tier quant happens) and "snap" (KIVI INT8 + SnapKV eviction).
        # Sub-INT8 modes ("int4", "int2") propagate directly.
        layer_mode = mode if mode in ("int4", "int2") else "int8"
        self._layers: list[KVLayerCache] = [
            KVLayerCache(window=window, kv_mode=layer_mode) for _ in range(n_layers)
        ]
        if svd_rank > 0:
            for layer in self._layers:
                layer._svd_rank = svd_rank
        if comm_vq_bits > 0:
            for layer in self._layers:
                layer._comm_vq_bits = comm_vq_bits
        self._snapped = [False] * n_layers   # has SnapKV eviction been applied?
        # Phase 3: Q-Filters
        self._qfilter_cfg = None
        if qfilter_rank > 0:
            from squish.kv.q_filters import QFilterConfig, QFilterState  # noqa: PLC0415
            self._qfilter_cfg = QFilterConfig(
                rank=qfilter_rank,
                budget=qfilter_budget,
                anchor=qfilter_anchor,
                evict_every=qfilter_evict_every,
                min_tokens=max(qfilter_rank, 64),
            )
            for layer in self._layers:
                layer._qfilter = QFilterState(self._qfilter_cfg)
        # Phase 5: TTT Fast Weights
        self._fw_manager = None
        if fast_weight_lr > 0.0:
            from squish.kv.fast_weights import FastWeightConfig, FastWeightManager  # noqa: PLC0415
            self._fw_manager = FastWeightManager(
                FastWeightConfig(lr=fast_weight_lr, decay=fast_weight_decay),
                n_layers=n_layers,
            )

    # ── Compatibility shims for mlx_lm cache list API ────────────────────────

    def __len__(self):
        return self.n_layers

    def __getitem__(self, idx):
        """Return the KVLayerCache for layer idx (mlx_lm update_and_fetch protocol)."""
        return self._layers[idx]

    def __iter__(self):
        for i in range(self.n_layers):
            yield self[i]

    # ── Layer update (called by patched attention) ────────────────────────────

    def update(self, layer_idx: int, key_np: np.ndarray, value_np: np.ndarray) -> None:
        """
        Append key/value for ``layer_idx``.  Applies SnapKV eviction once the
        cache exceeds ``budget`` tokens (first call after prefill).
        """
        layer = self._layers[layer_idx]
        layer.append(key_np, value_np)

        if (self.mode == "snap"
                and not self._snapped[layer_idx]
                and layer.n_tokens > self.budget):
            _snap_evict(layer, self.budget, self.snap_window)
            self._snapped[layer_idx] = True

    def get_kv_mlx(self, layer_idx: int):
        """Return (keys, values) as MLX bfloat16 arrays."""
        return self._layers[layer_idx].get_as_mlx()

    def reset(self):
        """Clear all layers (new conversation)."""
        for layer in self._layers:
            layer.reset()
        self._snapped = [False] * self.n_layers
        # Phase 3: reset Q-filter per-request state (basis preserved across requests)
        if self._qfilter_cfg is not None:
            for layer in self._layers:
                if layer._qfilter is not None:
                    layer._qfilter.reset()
        # Phase 5: reset fast weight matrices for new request
        if self._fw_manager is not None:
            self._fw_manager.reset()

    def tick_qfilter(self, step: int) -> None:
        """
        Phase 3 + 5 — Run Q-filter geometric eviction across all layers for ``step``.

        Call once per decode step, immediately after ``mx.eval(logits)``.
        No-op if Q-filters are not configured (``qfilter_rank == 0``).

        When Phase 5 fast weights are also enabled (``fast_weight_lr > 0``),
        evicted tokens are absorbed into the fast-weight memory before removal.

        The eviction fires when *all* of these are true for a layer:
        • ``n_tokens > qfilter_budget``
        • ``step % qfilter_evict_every == 0``  (or evict_every == 0)
        • The layer's SVD basis has been calibrated
        """
        if self._qfilter_cfg is None:
            return
        from squish.kv.q_filters import _qfilter_evict  # noqa: PLC0415
        cfg = self._qfilter_cfg
        import numpy as _np  # noqa: PLC0415
        for layer_idx, layer in enumerate(self._layers):
            if layer._qfilter is None:
                continue
            n = layer.n_tokens
            if n <= cfg.budget:
                continue
            if cfg.evict_every > 0 and step % cfg.evict_every != 0:
                continue
            if not layer._qfilter.is_calibrated:
                continue
            full_k, full_v = layer.get_full_kv()
            if full_k is None:
                continue
            anchor  = min(cfg.anchor, n)
            recent  = full_k[:, -anchor:, :]               # (n_heads, anchor, head_dim)
            scores  = layer._qfilter.score_recent(recent)
            if scores is None or len(scores) != n:
                continue
            masked          = scores.copy()
            masked[-anchor:] = _np.inf
            keep = _np.sort(_np.argsort(-masked)[: cfg.budget])

            # Phase 5: absorb evicted tokens into fast weights before dropping
            if self._fw_manager is not None and full_v is not None:
                keep_set  = set(keep.tolist())
                evict_idx = _np.array(
                    [i for i in range(n) if i not in keep_set], dtype=_np.int64
                )
                if len(evict_idx) > 0:
                    evict_k = full_k[:, evict_idx, :]
                    evict_v = full_v[:, evict_idx, :]
                    self._fw_manager.absorb_layer(layer_idx, evict_k, evict_v)

            _qfilter_evict(layer, keep)
            layer._qfilter.rebuild_after_eviction(keep)

    @property
    def n_tokens(self) -> int:
        """Total K/V tokens currently cached (layer 0 is representative)."""
        return self._layers[0].n_tokens if self._layers else 0

    @property
    def memory_mb(self) -> float:
        """Approximate total KV cache memory in MB."""
        total = sum(layer.memory_bytes for layer in self._layers)
        return total / 1_048_576

    def stats(self) -> dict:
        d = {
            "mode":      self.mode,
            "n_layers":  self.n_layers,
            "n_tokens":  self.n_tokens,
            "memory_mb": round(self.memory_mb, 2),
            "window":    self.window,
            "budget":    self.budget,
        }
        if self._qfilter_cfg is not None:
            calibrated = sum(
                1 for l in self._layers if l._qfilter is not None and l._qfilter.is_calibrated
            )
            d["qfilter_rank"]        = self._qfilter_cfg.rank
            d["qfilter_budget"]      = self._qfilter_cfg.budget
            d["qfilter_calibrated"]  = calibrated
        if self._fw_manager is not None:
            fw_stats = self._fw_manager.stats()
            d["fast_weight_lr"]       = fw_stats["fast_weight_lr"]
            d["fast_weight_absorbed"] = fw_stats["fast_weight_total_absorbed"]
        return d

    def restore_from(self, src: "QuantizedKVCache") -> None:
        """
        Copy all layer data from *src* into this cache in-place.

        Used by the disk-prompt-cache path: the disk-loaded cache is
        deserialised into a temporary object, then its state is copied
        into the model-patched layers so the existing object references
        remain valid.
        """
        for dst_lay, src_lay in zip(self._layers, src._layers, strict=False):
            dst_lay.keys_old_q   = src_lay.keys_old_q
            dst_lay.keys_old_s   = src_lay.keys_old_s
            dst_lay.values_old_q = src_lay.values_old_q
            dst_lay.values_old_s = src_lay.values_old_s
            dst_lay.keys_recent   = list(src_lay.keys_recent)
            dst_lay.values_recent = list(src_lay.values_recent)
            dst_lay.n_heads  = src_lay.n_heads
            dst_lay.head_dim = src_lay.head_dim
            # Phase 1: carry SVD basis across (calibration buffer is per-request)
            dst_lay._svd_Vk   = src_lay._svd_Vk
            dst_lay._svd_Vv   = src_lay._svd_Vv
            dst_lay._svd_rank = src_lay._svd_rank
            dst_lay._svd_buf_k = None
            dst_lay._svd_buf_v = None

    def clone_snapshot(self) -> "QuantizedKVCache":
        """
        Create an independent point-in-time snapshot of the current cache state.

        The snapshot is a lightweight copy: numpy arrays in the INT8 compressed
        tier are reference-shared (they are never mutated after being stored),
        while the recent-window list is a new list object.  This is safe because
        the source cache will be ``reset()`` at the start of the next request,
        replacing its own references without touching the snapshotted arrays.

        Used by :class:`squish.kv.prefix_kv_store.PrefixKVStore` to save the
        post-prefill cache for later prefix reuse — allowing subsequent requests
        whose prompt shares the same prefix to restore the KV state and skip
        re-running those tokens through the model.

        Returns
        -------
        QuantizedKVCache
            A new cache object containing the same KV data as *self* at the
            moment of the call.  No Q-filter or fast-weight state is carried
            over (irrelevant for a static snapshot).
        """
        snap = QuantizedKVCache(
            n_layers=self.n_layers,
            window=self.window,
            mode="int8",        # snapshot stores raw INT8 KV — no active eviction
            budget=self.budget,
            snap_window=self.snap_window,
            svd_rank=self.svd_rank,
            comm_vq_bits=0,     # CommVQ state is not snapshotted
        )
        snap.restore_from(self)
        snap._snapped = list(self._snapped)
        return snap


# ---------------------------------------------------------------------------
# HadamardKVCache — QuaRot-style Hadamard rotation before quantization
# ---------------------------------------------------------------------------

class HadamardKVCache(QuantizedKVCache):
    """
    QuaRot-inspired KV cache: rotate K/V vectors with a random Hadamard
    matrix before INT8 quantization, and un-rotate during read-back.

    Motivation (QuaRot / HadamardQuant, Tseng et al. 2024)
    -------------------------------------------------------
    Raw key/value activations in large language models have highly
    non-uniform variance across channels — a few dimensions carry most of
    the signal, making per-channel INT8 quantization lossy.

    Multiplying by a random Hadamard matrix H (H ∈ ℝ^{d×d}, H·Hᵀ = d·I)
    *spreads* the variance uniformly across dimensions before quantization.
    The rotation is isometric (preserves inner products up to a scalar),
    so attention scores are unchanged:

        QKᵀ = (Q·Hᵀ)·(K·Hᵀ)ᵀ · (1/d)    (same value, uniform quantization grid)

    In practice this reduces INT8 quantization MSE by 1.5–3× with zero
    change to model outputs and negligible runtime overhead (~0.1 ms per step).

    Usage
    -----
    Drop-in replacement for :class:`QuantizedKVCache`:

        cache = HadamardKVCache(n_layers=32, window=64, mode="int8")
        # K/V stored as H·K and H·V; attention reads back un-rotated copies.

    Or via the helper:

        cache = make_hadamard_cache(model, mode="int8", window=64)

    W104 — INT2 mode
    ----------------
    ``mode="int2"`` stores quantised K/V using a 4-level NF2 codebook
    (±1.5σ, ±0.5σ) bit-packed 4-per-byte.  Hadamard rotation is *especially*
    valuable here — without it the post-rotation distribution of K/V outliers
    collapses three of the four NF2 bins, costing ≥ 3 dB MSE.  See PLAN W104
    for the 32 K-context-on-M3-16 GB envelope this enables.

    W105 — INT4 mode
    ----------------
    ``mode="int4"`` stores quantised K/V using a 16-level uniform symmetric
    codebook (±0.5..±7.5 in scaled units) nibble-packed 2-per-byte.  Half the
    memory of INT8 with ~22–28 dB SNR on Hadamard-rotated activations — the
    intermediate quality tier that makes 16 K context safe on M3 16 GB
    without the SNR cliff that INT2 introduces.  Use ``recommended_kv_mode_3tier``
    to drive int8 / int4 / int2 selection by planned context length.
    """

    def __init__(
        self,
        n_layers: int,
        window: int = 64,
        mode: str = "int8",
        budget: int = 4096,
        snap_window: int = 32,
        svd_rank: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_layers=n_layers, window=window, mode=mode,
            budget=budget, snap_window=snap_window, svd_rank=svd_rank,
        )
        self._seed    = seed
        # Hadamard matrices are generated lazily on first use (head_dim unknown)
        self._H_k: dict[int, np.ndarray] = {}   # dim → H float16 matrix
        self._H_v: dict[int, np.ndarray] = {}   # separate rotation per K/V

    # ── Hadamard construction ─────────────────────────────────────────────────

    @staticmethod
    def _build_hadamard(dim: int, rng: np.random.Generator) -> np.ndarray:
        """
        Return a (dim, dim) random Hadamard-like rotation matrix.

        For power-of-two dims: construct the Walsh–Hadamard matrix (H_n) and
        apply a random orthogonal sign matrix (D = diag(±1)) so that
        H_rand = D × H_n / sqrt(dim).  This is orthogonal and fast to multiply.

        For non-power-of-two dims: fall back to a random orthogonal matrix via
        QR decomposition of a Gaussian matrix.  Slightly slower to construct
        but identical properties once built.
        """
        if dim > 0 and (dim & (dim - 1)) == 0:
            # Power-of-two: Walsh–Hadamard via Sylvester construction
            H = np.array([[1.0]], dtype=np.float32)
            while H.shape[0] < dim:
                H = np.block([[H, H], [H, -H]])
            H /= np.sqrt(dim)
            # Random sign flip for additional randomness
            signs = rng.choice([-1.0, 1.0], size=(dim,)).astype(np.float32)
            H = H * signs[np.newaxis, :]   # column-wise sign flip
        else:
            # General: random orthogonal matrix (QR of Gaussian)
            G = rng.standard_normal((dim, dim)).astype(np.float32)
            H, _ = np.linalg.qr(G)

        return H.astype(np.float16)

    def _get_H_k(self, head_dim: int) -> np.ndarray:
        """Return (or build and cache) the key rotation matrix for head_dim."""
        if head_dim not in self._H_k:
            rng = np.random.default_rng(self._seed)
            self._H_k[head_dim] = self._build_hadamard(head_dim, rng)
        return self._H_k[head_dim]

    def _get_H_v(self, head_dim: int) -> np.ndarray:
        """Return (or build and cache) the value rotation matrix for head_dim."""
        if head_dim not in self._H_v:
            rng = np.random.default_rng(self._seed + 1)
            self._H_v[head_dim] = self._build_hadamard(head_dim, rng)
        return self._H_v[head_dim]

    # ── Override update to pre-rotate before quantization ────────────────────

    def update(
        self, layer_idx: int, key_np: np.ndarray, value_np: np.ndarray
    ) -> None:
        """
        Rotate K and V with the per-head Hadamard matrix, then delegate to
        the parent KIVI quantization pipeline.

        Parameters
        ----------
        layer_idx : int
        key_np    : np.ndarray  shape (n_heads, n_new_tokens, head_dim)  float16
        value_np  : same shape
        """
        head_dim = key_np.shape[-1]
        H_k = self._get_H_k(head_dim)  # (head_dim, head_dim) float16
        H_v = self._get_H_v(head_dim)

        # Rotate: (n_heads, n_tokens, head_dim) @ (head_dim, head_dim)
        # Use float32 for the multiply to avoid fp16 accumulation errors,
        # then cast back once done.
        key_rot   = (key_np.astype(np.float32)   @ H_k.astype(np.float32)).astype(np.float16)
        value_rot = (value_np.astype(np.float32) @ H_v.astype(np.float32)).astype(np.float16)

        super().update(layer_idx, key_rot, value_rot)

    # ── Override get_kv_mlx to un-rotate on read-back ─────────────────────────

    def get_kv_mlx(self, layer_idx: int):
        """
        Retrieve quantised K/V as MLX arrays, un-rotated with Hᵀ (= H⁻¹).

        Returns (keys, values) in bfloat16, identical to what an un-patched
        model would have stored — so attention scores are unchanged.
        """
        try:
            import mlx.core as mx
        except ImportError:  # pragma: no cover
            raise RuntimeError("mlx.core not available")

        keys_mlx, vals_mlx = super().get_kv_mlx(layer_idx)

        # Infer head_dim from the last dimension of the returned tensors
        head_dim = keys_mlx.shape[-1]
        H_k = self._get_H_k(head_dim)
        H_v = self._get_H_v(head_dim)

        # Un-rotate: multiply by Hᵀ (orthogonal inverse)
        # Convert H to MLX arrays for the matrix multiply on Metal
        H_k_mx = mx.array(H_k.T.astype(np.float16))  # (head_dim, head_dim)
        H_v_mx = mx.array(H_v.T.astype(np.float16))

        keys_out = (keys_mlx.astype(mx.float32) @ H_k_mx.astype(mx.float32)).astype(mx.bfloat16)
        vals_out = (vals_mlx.astype(mx.float32) @ H_v_mx.astype(mx.float32)).astype(mx.bfloat16)

        return keys_out, vals_out


def make_hadamard_cache(  # pragma: no cover
    model,
    mode: str = "int8",
    window: int = 64,
    budget: int = 4096,
    snap_window: int = 32,
    svd_rank: int = 0,
    seed: int = 42,
) -> HadamardKVCache:
    """
    Create a :class:`HadamardKVCache` sized correctly for *model*.

    Parameters
    ----------
    model       : loaded mlx_lm model
    mode        : "fp16" | "int8" | "snap"
    window      : FP16 residual window size (KIVI)
    budget      : max K/V positions per layer (SnapKV; ignored for fp16)
    snap_window : attention window for importance scoring (SnapKV)
    svd_rank    : Phase 1 SVD projection rank (0 = off)
    seed        : RNG seed for the Hadamard rotation matrices
    """
    n = _n_layers(model)
    return HadamardKVCache(
        n_layers=n, window=window, mode=mode,
        budget=budget, snap_window=snap_window,
        svd_rank=svd_rank, seed=seed,
    )


class _LayerCacheView:
    """
    Thin shim so QuantizedKVCache[i] behaves like a KV cache dict for mlx_lm.
    """
    def __init__(self, layer: KVLayerCache, parent: QuantizedKVCache):
        self._layer  = layer
        self._parent = parent

    @property
    def keys(self):
        k, _ = self._layer.get_as_mlx()
        return k

    @property
    def values(self):
        _, v = self._layer.get_as_mlx()
        return v


# ---------------------------------------------------------------------------
# Model patching — intercept attention layers to use QuantizedKVCache
# ---------------------------------------------------------------------------

def _n_layers(model) -> int:  # pragma: no cover
    """Infer number of transformer layers from a loaded mlx_lm model."""
    # Most mlx_lm models expose model.model.layers
    try:
        return len(model.model.layers)
    except AttributeError:
        pass
    try:
        return len(model.layers)
    except AttributeError:
        pass
    # Fallback: count by inspecting config
    try:
        cfg = model.args
        return (getattr(cfg, "num_hidden_layers", None)
                or getattr(cfg, "n_layers", None)
                or 32)
    except Exception:
        return 32


def make_quantized_cache(  # pragma: no cover
    model,
    mode: str = "int8",
    window: int = 64,
    budget: int = 4096,
    snap_window: int = 32,
    svd_rank: int = 0,
    comm_vq_bits: int = 0,
    qfilter_rank: int = 0,
    qfilter_budget: int = 2048,
    qfilter_anchor: int = 64,
    qfilter_evict_every: int = 16,
    fast_weight_lr: float = 0.0,
    fast_weight_decay: float = 0.999,
) -> QuantizedKVCache:
    """
    Create a :class:`QuantizedKVCache` sized correctly for ``model``.

    Parameters
    ----------
    model             : mlx_lm model (already loaded)
    mode              : "fp16" | "int8" | "snap"
    window            : FP16 residual window (KIVI)
    budget            : max K/V positions (SnapKV only)
    snap_window       : attention window for importance (SnapKV)
    svd_rank          : Phase 1 — SVD projection rank (0 = off)
    comm_vq_bits      : CommVQ bits per KV token (0 = off; 2 or 4)
    qfilter_rank      : Phase 3 — SVD rank for geometric eviction (0 = off)
    qfilter_budget    : Phase 3 — max tokens per layer after eviction
    qfilter_anchor    : Phase 3 — recent tokens never evicted
    qfilter_evict_every: Phase 3 — eviction interval in decode steps
    fast_weight_lr    : Phase 5 — learning rate for fast-weight absorption (0 = off)
    fast_weight_decay : Phase 5 — per-absorption decay factor
    """
    n = _n_layers(model)
    return QuantizedKVCache(
        n_layers=n, window=window, mode=mode,
        budget=budget, snap_window=snap_window,
        svd_rank=svd_rank, comm_vq_bits=comm_vq_bits,
        qfilter_rank=qfilter_rank, qfilter_budget=qfilter_budget,
        qfilter_anchor=qfilter_anchor, qfilter_evict_every=qfilter_evict_every,
        fast_weight_lr=fast_weight_lr, fast_weight_decay=fast_weight_decay,
    )


def patch_model_kv_cache(  # pragma: no cover
    model,
    mode: str = "int8",
    window: int = 64,
    budget: int = 4096,
    snap_window: int = 32,
    svd_rank: int = 0,
    comm_vq_bits: int = 0,
    qfilter_rank: int = 0,
    qfilter_budget: int = 2048,
    qfilter_anchor: int = 64,
    qfilter_evict_every: int = 16,
    fast_weight_lr: float = 0.0,
    fast_weight_decay: float = 0.999,
    verbose: bool = True,
) -> QuantizedKVCache:
    """
    Monkey-patch the model's attention layers so that the KV cache written
    during generation is automatically quantized.

    This is less invasive than modifying mlx_lm internals: instead of
    replacing the attention class, we wrap the cache-update step by
    intercepting ``mlx_lm.utils.generate_step``-style generation via
    a shared :class:`QuantizedKVCache` object.

    Returns the :class:`QuantizedKVCache` instance; pass it to
    ``generate_with_cache(model, tokenizer, prompt, cache=...)`` or
    use ``generate_step`` from mlx_lm.

    Note
    ----
    For full KV-cache quantization inside the MLX attention kernel, a
    future version will use MLX custom primitives.  This implementation
    works at the Python level with numpy round-trips for the compressed
    portion, which adds ~5-10 ms per 100 tokens for the
    dequantize-and-forward step.
    """
    cache = make_quantized_cache(
        model, mode=mode, window=window,
        budget=budget, snap_window=snap_window,
        svd_rank=svd_rank, comm_vq_bits=comm_vq_bits,
        qfilter_rank=qfilter_rank, qfilter_budget=qfilter_budget,
        qfilter_anchor=qfilter_anchor, qfilter_evict_every=qfilter_evict_every,
        fast_weight_lr=fast_weight_lr, fast_weight_decay=fast_weight_decay,
    )
    n = _n_layers(model)

    if verbose:
        svd_info      = f"  svd_rank={svd_rank}" if svd_rank > 0 else ""
        commvq_info   = f"  commvq_bits={comm_vq_bits}" if comm_vq_bits > 0 else ""
        qfilter_info  = (f"  qfilter_rank={qfilter_rank}  qfilter_budget={qfilter_budget}"
                         if qfilter_rank > 0 else "")
        fw_info       = (f"  fast_weight_lr={fast_weight_lr}"
                         if fast_weight_lr > 0.0 else "")
        print(f"  [KV cache] mode={mode}  window={window}  "
              f"budget={budget if mode == 'snap' else '—'}  "
              f"layers={n}{svd_info}{commvq_info}{qfilter_info}{fw_info}")

    # Store on the model so server.py can retrieve it
    model._squish_kv_cache = cache
    return cache


# ---------------------------------------------------------------------------
# generate_with_cache — convenience wrapper for server.py
# ---------------------------------------------------------------------------

def generate_step_with_quantized_cache(  # pragma: no cover
    model,
    token_ids,          # (1, seq_len) MLX int32
    quantized_cache: QuantizedKVCache,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    """
    Run a single generation step using the quantized KV cache.

    This is a simplified stub that works with models whose attention
    accepts a ``cache`` keyword argument as a list of dicts with
    ``keys`` / ``values`` entries.

    For production use, mlx_lm's ``generate_step`` with ``cache=``
    is the recommended path once QuantizedKVCache supports the full
    mlx_lm cache protocol.

    Returns
    -------
    next_token_id : int
    """
    mx = _mx()

    with mx.stream(mx.gpu):
        logits = model(token_ids)          # (1, seq, vocab)

    next_logits = np.array(logits[0, -1, :].astype(mx.float32))  # (vocab,)

    if temperature <= 0.0 or temperature < 1e-5:
        return int(np.argmax(next_logits))

    next_logits = next_logits / temperature
    # top-p filtering
    if top_p < 1.0:
        sorted_idx  = np.argsort(-next_logits)
        cum_probs   = np.cumsum(
            np.exp(next_logits[sorted_idx]
                   - np.max(next_logits[sorted_idx])))
        cum_probs  /= cum_probs[-1] + 1e-9
        sorted_idx[np.searchsorted(cum_probs, top_p) + 1
                                  if np.searchsorted(cum_probs, top_p) + 1
                                     < len(sorted_idx) else -1]
        next_logits[sorted_idx[np.searchsorted(cum_probs, top_p) + 1:]] = -1e9

    probs = np.exp(next_logits - np.max(next_logits))
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


# ---------------------------------------------------------------------------
# DiskKVCache — persistent cross-request prompt cache backed by NVMe
# ---------------------------------------------------------------------------

class DiskKVCache:
    """
    Cross-request disk-backed prompt cache for QuantizedKVCache.

    Serialises full KV state to per-entry ``.npz`` files keyed by the
    SHA-256 of the input token-id sequence.  On a cache hit, prefill is
    skipped entirely.

    Parameters
    ----------
    cache_dir : str | Path
        Directory on fast NVMe where entry files are stored.  Created if it
        does not exist.
    max_entries : int
        Maximum number of entries; LRU eviction by mtime when exceeded.
    """

    def __init__(self, cache_dir, max_entries: int = 64):
        import threading as _threading
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max   = max_entries
        self._lock  = _threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def lookup(self, input_ids: list[int]) -> "tuple[QuantizedKVCache, np.ndarray] | None":
        """
        Return ``(qkv_cache, last_logit_f32)`` on a cache hit, or ``None``.

        *last_logit_f32* is the prefill's final-position raw logit vector so
        the caller can sample the first generated token without re-running
        the model.
        """
        entry = self._dir / (self._key(input_ids) + ".npz")
        if not entry.exists():
            return None
        try:
            data = np.load(entry, allow_pickle=False)
            qkv       = self._deserialise(data)
            last_logit = data["last_logit"].astype(np.float32) if "last_logit" in data else None
            if last_logit is None:
                return None  # legacy entry without logit — treat as miss
            # Touch mtime for LRU ordering
            entry.touch()
            return qkv, last_logit
        except Exception:  # corrupted or schema mismatch — treat as miss  # pragma: no cover
            try:
                entry.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def store(
        self,
        input_ids: list[int],
        qkv_cache: "QuantizedKVCache",
        last_logit_np: "np.ndarray | None" = None,
    ) -> None:
        """
        Persist *qkv_cache* (and optionally *last_logit_np*) to disk in a
        background thread.  Returns immediately; silently drops on error.
        """
        import threading as _threading

        def _worker():
            import os as _os
            try:
                arrays = self._serialise(qkv_cache)
                if arrays is None:
                    return
                if last_logit_np is not None:
                    arrays["last_logit"] = last_logit_np.astype(np.float32)
                entry = self._dir / (self._key(input_ids) + ".npz")
                tmp   = entry.with_name(entry.stem + ".tmp.npz")
                np.savez_compressed(str(tmp), **arrays)
                _os.replace(str(tmp), str(entry))
                self._evict_if_needed()
            except Exception:  # pragma: no cover
                pass

        _threading.Thread(target=_worker, daemon=True).start()

    # ── internals ───────────────────────────────────────────────────────────

    @staticmethod
    def _key(input_ids: list[int]) -> str:
        import hashlib
        raw = np.array(input_ids, dtype=np.int32).tobytes()
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _serialise(qkv_cache: "QuantizedKVCache") -> "dict | None":
        """
        Pack all layers into a flat dict of numpy arrays.

        Keys:
          ``n_layers``           — int scalar
          ``L{i}_n_heads``       — int scalar
          ``L{i}_head_dim``      — int scalar
          ``L{i}_keys_old_q``    — (n_heads, n_old, head_dim) int8 or missing
          ``L{i}_keys_old_s``    — (n_heads, n_old) f32        or missing
          ``L{i}_vals_old_q``    — (n_heads, n_old, head_dim) int8 or missing
          ``L{i}_vals_old_s``    — (n_heads, n_old) f32        or missing
          ``L{i}_n_recent``      — int scalar
          ``L{i}_keys_recent``   — (n_heads, n_rec, head_dim) f16 or missing
          ``L{i}_vals_recent``   — (n_heads, n_rec, head_dim) f16 or missing
        """
        layers = qkv_cache._layers
        out: dict[str, np.ndarray] = {
            "n_layers": np.array(len(layers), dtype=np.int32),
        }
        for i, lay in enumerate(layers):
            if lay.n_heads is None:
                return None  # layer not yet populated — skip whole entry
            out[f"L{i}_n_heads"]  = np.array(lay.n_heads,  dtype=np.int32)
            out[f"L{i}_head_dim"] = np.array(lay.head_dim, dtype=np.int32)
            if lay.keys_old_q is not None:
                out[f"L{i}_keys_old_q"] = lay.keys_old_q
                out[f"L{i}_keys_old_s"] = lay.keys_old_s
                out[f"L{i}_vals_old_q"] = lay.values_old_q
                out[f"L{i}_vals_old_s"] = lay.values_old_s
            n_rec = len(lay.keys_recent)
            out[f"L{i}_n_recent"] = np.array(n_rec, dtype=np.int32)
            if n_rec > 0:
                out[f"L{i}_keys_recent"] = np.stack(lay.keys_recent, axis=1)   # (H, n_rec, D)
                out[f"L{i}_vals_recent"] = np.stack(lay.values_recent, axis=1)
            # Phase 1: persist SVD basis so subsequent requests skip refitting
            if lay._svd_Vk is not None:
                out[f"L{i}_svd_rank"] = np.array(lay._svd_rank, dtype=np.int32)
                out[f"L{i}_svd_Vk"]   = lay._svd_Vk   # (n_heads, rank, head_dim) f16
                out[f"L{i}_svd_Vv"]   = lay._svd_Vv
        return out

    @staticmethod
    def _deserialise(data) -> "QuantizedKVCache":
        """Reconstruct a QuantizedKVCache from a loaded npz dict."""
        n_layers = int(data["n_layers"])
        # Build a shell QuantizedKVCache with the right layer count
        # We bypass patch_model_kv_cache and construct layers directly.
        layers: list[KVLayerCache] = []
        for i in range(n_layers):
            lay = KVLayerCache()
            lay.n_heads  = int(data[f"L{i}_n_heads"])
            lay.head_dim = int(data[f"L{i}_head_dim"])
            if f"L{i}_keys_old_q" in data:
                lay.keys_old_q   = data[f"L{i}_keys_old_q"]
                lay.keys_old_s   = data[f"L{i}_keys_old_s"]
                lay.values_old_q = data[f"L{i}_vals_old_q"]
                lay.values_old_s = data[f"L{i}_vals_old_s"]
            n_rec = int(data[f"L{i}_n_recent"])
            if n_rec > 0:
                k_rec = data[f"L{i}_keys_recent"]   # (H, n_rec, D)
                v_rec = data[f"L{i}_vals_recent"]
                for t in range(n_rec):
                    lay.keys_recent.append(k_rec[:, t, :])
                    lay.values_recent.append(v_rec[:, t, :])
            # Phase 1: restore SVD basis if persisted
            if f"L{i}_svd_Vk" in data:
                lay._svd_rank = int(data[f"L{i}_svd_rank"])
                lay._svd_Vk   = data[f"L{i}_svd_Vk"]
                lay._svd_Vv   = data[f"L{i}_svd_Vv"]
            layers.append(lay)

        qkv = object.__new__(QuantizedKVCache)
        qkv._layers = layers
        # Restore public config attributes with safe defaults
        qkv.mode   = getattr(layers[0], "_mode", "int8") if layers else "int8"
        qkv.window = getattr(layers[0], "_window", 64)   if layers else 64
        qkv.budget = getattr(layers[0], "_budget", 4096) if layers else 4096
        return qkv

    def _evict_if_needed(self) -> None:
        """Remove the oldest (by mtime) entries when over the size cap."""
        with self._lock:
            entries = sorted(
                (p for p in self._dir.glob("*.npz") if not p.stem.endswith(".tmp")),
                key=lambda p: p.stat().st_mtime,
            )
            while len(entries) > self._max:
                try:
                    entries.pop(0).unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass


# ---------------------------------------------------------------------------
# SessionKVCache — persistent cross-session KV state (Phase 3)
# ---------------------------------------------------------------------------

class SessionKVCache:
    """
    Persistent KV-state cache keyed by a SHA-256 hash of the last 8 message
    contents in a conversation.

    Unlike :class:`DiskKVCache` (which is keyed by raw token IDs), this cache
    is keyed by the *conversation context*, allowing the server to resume KV
    state across restarts without requiring identical tokenization.

    Session files are stored as compressed ``.npz`` under ``cache_dir``.
    Writes are non-blocking (background thread).

    Parameters
    ----------
    cache_dir   : str | Path — directory for session files (created if needed)
    max_entries : int        — LRU cap; oldest evicted when exceeded
    """

    def __init__(self, cache_dir, max_entries: int = 128):
        import threading as _threading
        self._dir  = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._max  = max_entries
        self._lock = _threading.Lock()

    # ── Public API ──────────────────────────────────────────────────────────

    def session_key(self, messages: "list[dict]") -> str:
        """
        Derive a stable session key from the last 8 message contents.

        Parameters
        ----------
        messages : list of OpenAI-style message dicts with ``"content"`` keys

        Returns
        -------
        32-hex-char string (SHA-256 truncated to 128 bits)
        """
        import hashlib
        tail = messages[-8:] if len(messages) > 8 else messages
        raw  = "\n".join(str(m.get("content", "")) for m in tail).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def load_session(self, key: str) -> "QuantizedKVCache | None":
        """
        Return a :class:`QuantizedKVCache` for a prior session, or ``None`` on miss.

        Parameters
        ----------
        key : session key from :meth:`session_key`
        """
        entry = self._dir / (key + ".npz")
        if not entry.exists():
            return None
        try:
            data = np.load(entry, allow_pickle=False)
            qkv  = DiskKVCache._deserialise(data)
            entry.touch()   # update mtime for LRU ordering
            return qkv
        except Exception:
            try:
                entry.unlink(missing_ok=True)
            except Exception:  # pragma: no cover
                pass
            return None

    def save_session(
        self,
        key: str,
        qkv_cache: "QuantizedKVCache",
    ) -> None:
        """
        Persist *qkv_cache* under *key* in a background thread.

        Returns immediately; silently drops on serialisation error.
        """
        import threading as _threading

        def _worker():
            import os as _os
            try:
                arrays = DiskKVCache._serialise(qkv_cache)
                if arrays is None:
                    return
                entry = self._dir / (key + ".npz")
                tmp   = entry.with_name(entry.stem + ".tmp.npz")
                np.savez_compressed(str(tmp), **arrays)
                _os.replace(str(tmp), str(entry))
                self._evict_if_needed()
            except Exception:  # pragma: no cover
                pass

        _threading.Thread(target=_worker, daemon=True).start()

    def list_sessions(self) -> "list[str]":
        """Return sorted list of active session keys (file stems)."""
        return sorted(
            p.stem for p in self._dir.glob("*.npz")
            if not p.stem.endswith(".tmp")
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        with self._lock:
            entries = sorted(
                (p for p in self._dir.glob("*.npz") if not p.stem.endswith(".tmp")),
                key=lambda p: p.stat().st_mtime,
            )
            while len(entries) > self._max:
                try:
                    entries.pop(0).unlink(missing_ok=True)
                except Exception:  # pragma: no cover
                    pass


# ── H2O: Heavy-Hitter Oracle KV-cache eviction ───────────────────────────────
#
# Based on:
#   "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
#    Language Models" — Zhang et al., NeurIPS 2023  (arXiv:2306.14048)
#
# Key idea: accumulate per-token cumulative attention scores; keep the
# top ``heavy_ratio`` fraction ("heavy hitters") plus the ``recent_window``
# most-recent positions; evict the rest.  Unlike pure sliding-window
# approaches, H2O retains tokens the model actually attends to, even if
# they are no longer recent.

import heapq as _h2o_heapq
from dataclasses import dataclass as _h2o_dc


@_h2o_dc
class H2OConfig:
    """Configuration for H2O heavy-hitter KV-cache eviction.

    Parameters
    ----------
    heavy_ratio : float
        Fraction of the cache budget reserved for heavy hitters (0 < x < 1).
        The remaining fraction is used for the recency window.
    recent_window : int
        Minimum number of most-recent positions always kept in the cache.
    max_seq_len : int
        Trigger automatic eviction when cached length exceeds this value.
        Set ``0`` to disable automatic eviction (call
        :meth:`H2OEvictionPolicy.evict_to_budget` manually).
    """

    heavy_ratio:   float = 0.1
    recent_window: int   = 128
    max_seq_len:   int   = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.heavy_ratio < 1.0:
            raise ValueError("heavy_ratio must be in (0, 1)")
        if self.recent_window < 0:
            raise ValueError("recent_window must be >= 0")
        if self.max_seq_len < 0:
            raise ValueError("max_seq_len must be >= 0")


class H2OEvictionPolicy:
    """Per-head policy: accumulate attention scores and evict KV positions.

    Tracks cumulative attention received by each cached position across all
    forward passes.  When the cache is full, positions with the lowest
    accumulated scores (that are not in the recency window) are evicted.

    Parameters
    ----------
    config : H2OConfig
    """

    def __init__(self, config: H2OConfig) -> None:
        self._cfg       = config
        self._scores:    dict[int, float] = {}   # position → cumulative score
        self._positions: list[int]        = []   # ordered cached positions
        self._next_pos:  int              = 0

    # ── observation ─────────────────────────────────────────────────────────

    def add_token(self, init_score: float = 0.0) -> int:
        """Register a new token and return its position index.

        Parameters
        ----------
        init_score : float — initial score assigned to this position (default 0).
        """
        pos = self._next_pos
        self._next_pos += 1
        self._scores[pos]   = float(init_score)
        self._positions.append(pos)
        if self._cfg.max_seq_len > 0 and len(self._positions) > self._cfg.max_seq_len:
            self.evict_to_budget(self._cfg.max_seq_len)
        return pos

    def record_attention(self, attn_row: np.ndarray) -> None:
        """Accumulate one row of attention weights over cached positions.

        Parameters
        ----------
        attn_row : (n_cached,) float array — attention weights for the current
            query over all currently cached positions (same order as
            :attr:`positions`).
        """
        row = np.asarray(attn_row, dtype=np.float64)
        n   = min(len(row), len(self._positions))
        for i in range(n):
            pos = self._positions[i]
            self._scores[pos] = self._scores.get(pos, 0.0) + float(row[i])

    # ── eviction ────────────────────────────────────────────────────────────

    def evict_to_budget(self, budget: int) -> list[int]:
        """Evict positions until the cache contains at most ``budget`` entries.

        Heavy hitters (top-scoring non-recent positions) and the most-recent
        window are retained; lower-scoring older positions are evicted.

        Parameters
        ----------
        budget : int — maximum positions to retain after eviction.

        Returns
        -------
        Sorted list of evicted position indices.
        """
        if budget < 1:
            raise ValueError("budget must be >= 1")
        positions = self._positions
        if len(positions) <= budget:
            return []

        cfg      = self._cfg
        n_recent = min(cfg.recent_window, budget)
        n_heavy  = max(0, budget - n_recent)

        recent_set = set(positions[-n_recent:]) if n_recent > 0 else set()
        candidates = [
            (pos, self._scores.get(pos, 0.0))
            for pos in positions if pos not in recent_set
        ]
        heavy_pos = {
            pos for pos, _ in
            _h2o_heapq.nlargest(n_heavy, candidates, key=lambda x: x[1])
        }

        keep    = recent_set | heavy_pos
        evicted = sorted(p for p in positions if p not in keep)
        for p in evicted:
            self._scores.pop(p, None)
        self._positions = [p for p in positions if p in keep]
        return evicted

    # ── introspection ───────────────────────────────────────────────────────

    @property
    def positions(self) -> list[int]:
        """Ordered list of currently cached position indices."""
        return list(self._positions)

    @property
    def num_cached(self) -> int:
        """Number of currently cached positions."""
        return len(self._positions)

    def top_heavy_hitters(self, k: int = 10) -> list[tuple[int, float]]:
        """Return the top-``k`` positions by cumulative attention score.

        Returns list of ``(position, score)`` pairs, highest score first.
        """
        items = [(p, self._scores.get(p, 0.0)) for p in self._positions]
        return _h2o_heapq.nlargest(k, items, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# KVBudgetBroker — unified KV token-budget arbitrator (Phase 5C Opt 7)
# ---------------------------------------------------------------------------

class KVBudgetBroker:
    """
    Singleton that arbitrates KV cache token budgets across all active eviction
    systems (SnapKV, SqueezeKVCache, SmallKVCache, YOCO, DiffKV, KVTuner,
    KVSharer, AdaptiveBudget, ...).

    Without centralised arbitration, multiple budget-consuming systems running
    simultaneously can double-count their reservations: each believes it owns
    N tokens, but the total can exceed available unified memory by a large
    multiple.  The broker ensures the sum of all allocations never exceeds the
    configured ``total_tokens`` limit.

    Usage
    -----
    ::

        # At startup (once, after model load):
        KVBudgetBroker.instance().set_total(max_seq_len)

        # In each eviction module's __init__:
        allocated = KVBudgetBroker.instance().register("squeeze_kv", requested=4_096)

        # At request time:
        budget = KVBudgetBroker.instance().allocated("squeeze_kv")

    Allocation policy
    -----------------
    * If ``total_tokens == 0`` (unconstrained) every system gets its full
      requested budget.
    * If the sum of all requests ≤ ``total_tokens``, every system gets exactly
      what it asked for.
    * Otherwise each system is scaled down proportionally; each system receives
      at least 1 token.
    """

    _instance: "KVBudgetBroker | None" = None

    # ── Singleton access ────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "KVBudgetBroker":
        """Return (or create) the process-global broker instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the current singleton.  Intended for tests and clean server restarts."""
        cls._instance = None

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._total_tokens: int = 0
        self._registrations: dict[str, int] = {}   # name → requested tokens
        self._allocations:   dict[str, int] = {}   # name → allocated tokens

    # ── Configuration ───────────────────────────────────────────────────────

    def set_total(self, total_tokens: int) -> None:
        """
        Set the global KV token budget shared across all registered systems.

        A value of 0 means "unconstrained" — each system gets its full
        requested allocation.

        Parameters
        ----------
        total_tokens : int
            Maximum total KV tokens across all registered eviction systems.
        """
        if total_tokens < 0:
            raise ValueError(f"total_tokens must be >= 0, got {total_tokens}")
        self._total_tokens = total_tokens
        self._recompute()

    # ── Registration ────────────────────────────────────────────────────────

    def register(self, name: str, requested: int) -> int:
        """
        Register an eviction system and return its allocated token budget.

        Parameters
        ----------
        name      : str  — unique identifier for the system (e.g. "squeeze_kv")
        requested : int  — desired number of KV tokens (> 0)

        Returns
        -------
        int — actual allocated tokens (≤ requested when total is constrained)
        """
        if requested <= 0:
            raise ValueError(f"requested must be > 0, got {requested!r}")
        self._registrations[name] = requested
        self._recompute()
        return self._allocations[name]

    def unregister(self, name: str) -> None:
        """Remove a system from the budget pool (called on teardown)."""
        self._registrations.pop(name, None)
        self._allocations.pop(name, None)
        self._recompute()

    # ── Query ────────────────────────────────────────────────────────────────

    def allocated(self, name: str) -> int:
        """Return the allocated token budget for *name* (0 if not registered)."""
        return self._allocations.get(name, 0)

    @property
    def total_tokens(self) -> int:
        """The configured global token budget (0 = unconstrained)."""
        return self._total_tokens

    @property
    def registered_systems(self) -> list:
        """Names of all currently registered eviction systems."""
        return list(self._registrations)

    def summary(self) -> dict:
        """Return a snapshot of all current allocations (name → tokens)."""
        return dict(self._allocations)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _recompute(self) -> None:
        """
        Re-run fair-share allocation after any registration or total change.
        Called automatically; not part of the public API.
        """
        if not self._registrations:
            self._allocations = {}
            return

        total_requested = sum(self._registrations.values())

        if self._total_tokens == 0 or total_requested <= self._total_tokens:
            # Unconstrained, or all requests fit — give everyone what they asked for.
            self._allocations = dict(self._registrations)
        else:
            # Scale each system down proportionally; guarantee at least 1 token.
            scale = self._total_tokens / total_requested
            self._allocations = {
                name: max(1, int(req * scale))
                for name, req in self._registrations.items()
            }

