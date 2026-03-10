"""
squish/nf4_quant.py

NF4 — NormalFloat-4 Quantization for neural network weights.

Based on:
    Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
    arXiv:2305.14314

Background
----------
Standard INT4 quantization places 16 levels uniformly in the range [-1, +1].
This is suboptimal for neural network weights, which are approximately
normally distributed (zero-mean Gaussian): uniform spacing wastes resolution
on the tails and under-represents the dense central region.

NF4 places the 16 levels exactly at the quantiles of a N(0, 1) distribution
normalized to [-1, +1]:  level_k = Q(2k+0.5 / 16) for k=0..15.
This minimizes expected quantization error under the assumption that weights
are i.i.d. Gaussian, which holds well for linear layer weights in modern LLMs.

Per group (default 64 weights):
  1. Normalize: s = max(|w|) + ε;  w_norm = w / s  → range [-1, +1]
  2. Nearest-neighbor lookup in NF4_LEVELS (16 entries, sorted)
  3. Nibble-pack: two 4-bit indices per byte → 50% vs float32

Decompression:
  1. Unpack nibbles
  2. Lookup in NF4_LEVELS → float32 in [-1, +1]
  3. Rescale by stored scale: w_recon = v * s

Memory:
  - Packed weights: (n, d//2) uint8      (4 bits/weight)
  - Scales:         (n, d//group_size) float32  (overhead ≈ 0.5 bits/weight for gs=64)
  - Total: ~4.5 bits/weight (same as INT4; better per-bit quality)

Interface
---------
This module mirrors quantize_int4 / dequantize_int4 in squish/quantizer.py
for drop-in use in squish/convert.py.

    from squish.nf4_quant import quantize_nf4, dequantize_nf4, NF4_LEVELS

    packed, scales = quantize_nf4(weight_matrix, group_size=64)
    restored       = dequantize_nf4(packed, scales, group_size=64)
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "NF4_LEVELS",
    "quantize_nf4",
    "dequantize_nf4",
    "quantize_nf4_numpy",
    "dequantize_nf4_numpy",
]

# ---------------------------------------------------------------------------
# NF4 Codebook — 16 quantile levels of N(0,1) mapped to [-1, +1]
# ---------------------------------------------------------------------------
# These are the exact values from the QLoRA paper (Table 1), computed as the
# quantiles of the standard normal distribution at positions:
#   (2k + 0.5) / 16  for k = 0, 1, ..., 15
# then linearly normalized to fit [-1, +1].
# The values are symmetric: levels[15-k] = -levels[k].
NF4_LEVELS: np.ndarray = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)

assert len(NF4_LEVELS) == 16, "NF4_LEVELS must have exactly 16 entries"


# ---------------------------------------------------------------------------
# Pure-NumPy quantization
# ---------------------------------------------------------------------------

def quantize_nf4_numpy(
    embeddings: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Quantize a 2-D float32 array to NF4 nibble-packed format.

    Parameters
    ----------
    embeddings : np.ndarray  float32  shape (n_rows, n_cols)
    group_size : int
        Number of columns per quantization group; must divide n_cols evenly.
        Default 64 — same default as INT4 path in squish.

    Returns
    -------
    packed : np.ndarray  uint8  shape (n_rows, n_cols // 2)
        Nibble-packed indices: low nibble = even column, high nibble = odd column.
    scales : np.ndarray  float32  shape (n_rows, n_cols // group_size)
        Per-group abs-max scale factors.

    Raises
    ------
    ValueError
        If n_cols is not divisible by group_size, or group_size < 2.
    """
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    if emb.ndim != 2:
        raise ValueError(f"embeddings must be 1-D or 2-D, got shape {embeddings.shape}")

    n_rows, n_cols = emb.shape
    if group_size < 1:
        raise ValueError(f"group_size must be ≥ 1, got {group_size}")
    if n_cols % group_size != 0:
        raise ValueError(
            f"n_cols ({n_cols}) must be divisible by group_size ({group_size})"
        )
    if n_cols % 2 != 0:
        raise ValueError(
            f"n_cols ({n_cols}) must be even for nibble-packing"
        )

    n_groups = n_cols // group_size

    # Reshape to (n_rows * n_groups, group_size)
    grouped = emb.reshape(n_rows * n_groups, group_size)

    # Per-group scale = abs-max (+ ε to avoid div/0)
    abs_max = np.abs(grouped).max(axis=1, keepdims=True)  # (G, 1)
    abs_max = np.where(abs_max == 0, 1.0, abs_max).astype(np.float32)

    # Normalize to [-1, +1]
    normalized = grouped / abs_max                          # (G, group_size)

    # Nearest-neighbor lookup into NF4_LEVELS
    # Shape broadcast: (G, group_size, 1) vs (1, 1, 16)
    levels = NF4_LEVELS[np.newaxis, np.newaxis, :]         # (1, 1, 16)
    expanded = normalized[:, :, np.newaxis]                 # (G, group_size, 1)
    diffs = np.abs(expanded - levels)                       # (G, group_size, 16)
    indices = diffs.argmin(axis=-1).astype(np.uint8)        # (G, group_size) in [0..15]

    # Nibble-pack: two 4-bit indices per byte
    # Even columns → low nibble, odd columns → high nibble
    indices_2d = indices.reshape(n_rows * n_groups, group_size)

    # Interleave packing across the full row
    # Rebuild full-row view for packing
    indices_full = indices_2d.reshape(n_rows, n_cols)       # (n_rows, n_cols)
    low  = indices_full[:, 0::2].astype(np.uint8)           # even cols → low nibble
    high = indices_full[:, 1::2].astype(np.uint8)           # odd cols  → high nibble
    packed = (low & 0x0F) | ((high & 0x0F) << 4)           # (n_rows, n_cols//2)

    scales = abs_max.reshape(n_rows, n_groups).astype(np.float32)
    return packed, scales


def dequantize_nf4_numpy(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """
    Reconstruct float32 from NF4 nibble-packed weights.

    Parameters
    ----------
    packed     : np.ndarray  uint8  shape (n_rows, n_cols // 2)
    scales     : np.ndarray  float32  shape (n_rows, n_groups)
        where n_groups = (n_cols // group_size) and n_cols = packed.shape[1] * 2.
    group_size : int — must match the value used in quantize_nf4.

    Returns
    -------
    np.ndarray  float32  shape (n_rows, n_cols)
    """
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)

    n_rows        = packed.shape[0]
    n_cols        = packed.shape[1] * 2

    if n_cols % group_size != 0:
        raise ValueError(
            f"n_cols ({n_cols}) is not divisible by group_size ({group_size})"
        )

    # Unpack nibbles
    low  = (packed & 0x0F).astype(np.uint8)   # even original cols
    high = (packed >> 4).astype(np.uint8)      # odd original cols

    # Reconstruct original column order: interleave low (even) and high (odd)
    # low[:, i] was originally column 2*i; high[:, i] was column 2*i+1
    indices = np.empty((n_rows, n_cols), dtype=np.uint8)
    indices[:, 0::2] = low
    indices[:, 1::2] = high

    # Lookup NF4 values
    values = NF4_LEVELS[indices.astype(np.int32)]   # (n_rows, n_cols) float32

    # Rescale by per-group scale
    # scales shape: (n_rows, n_groups); expand to (n_rows, n_cols)
    scales_expanded = np.repeat(scales, group_size, axis=1)  # (n_rows, n_cols)
    return (values * scales_expanded).astype(np.float32)


# ---------------------------------------------------------------------------
# Public API (mirrors quantize_int4 / dequantize_int4 interface)
# ---------------------------------------------------------------------------

def quantize_nf4(
    embeddings: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """
    NF4 quantize a float32 weight matrix.

    This is a pure-NumPy implementation.  For very large models (>10 GB),
    consider processing in row chunks to limit peak RAM.

    Parameters
    ----------
    embeddings : np.ndarray  float32  shape (n_rows, n_cols)
    group_size : int  default 64

    Returns
    -------
    packed : np.ndarray  uint8  shape (n_rows, n_cols // 2)
    scales : np.ndarray  float32  shape (n_rows, n_cols // group_size)
    """
    return quantize_nf4_numpy(embeddings, group_size)


def dequantize_nf4(
    packed: np.ndarray,
    scales: np.ndarray,
    group_size: int = 64,
) -> np.ndarray:
    """
    Reconstruct float32 from NF4 nibble-packed format.

    Parameters
    ----------
    packed     : np.ndarray  uint8  shape (n_rows, n_cols // 2)
    scales     : np.ndarray  float32  shape (n_rows, n_cols // group_size)
    group_size : int — must match quantize_nf4 call

    Returns
    -------
    np.ndarray  float32  shape (n_rows, n_cols)
    """
    return dequantize_nf4_numpy(packed, scales, group_size)
