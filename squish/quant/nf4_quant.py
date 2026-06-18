"""squish/quant/nf4_quant.py — NormalFloat4 (NF4) weight quantization.

NF4 is the 4-bit NormalFloat scheme from QLoRA (Dettmers et al., 2023): a fixed
16-level codebook whose values sit at the quantiles of a unit normal, so it is
information-theoretically matched to the (roughly Gaussian) distribution of LLM
weights. Each group of ``group_size`` columns is normalized by its absmax to
[-1, 1], every weight is snapped to the nearest codebook level, and the 4-bit
indices are nibble-packed two-per-byte.

On-disk format (see ``squish/convert.py`` writer and
``squish/io/loader_utils.py`` reader):
  - ``packed``  : uint8,   shape ``(n, d // 2)`` — nibble-packed level indices,
                  even column in the low nibble, odd column in the high nibble.
  - ``scales``  : float32, shape ``(n, d // group_size)`` — per-group absmax.

The caller (`_pick_int4_group_size`) always chooses a ``group_size`` that evenly
divides ``d`` and ``d`` is even, so no padding is required and the reader can
recover ``group_size`` exactly as ``(packed.shape[1] * 2) // scales.shape[1]``.
"""

from __future__ import annotations

import numpy as np

# Canonical NF4 codebook (QLoRA / bitsandbytes): 16 levels on [-1, 1], asymmetric,
# with an exact 0.0, placed at the quantiles of a unit normal distribution.
_NF4_CODEBOOK: np.ndarray = np.array(
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


def _validate_2d(weight: np.ndarray, group_size: int) -> tuple[int, int, int]:
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got shape {weight.shape}")
    if group_size < 1:
        raise ValueError(f"group_size must be ≥ 1, got {group_size}")
    n, d = weight.shape
    if d % group_size != 0:
        raise ValueError(
            f"n_cols ({d}) must be divisible by group_size ({group_size})"
        )
    if d % 2 != 0:
        raise ValueError(f"n_cols ({d}) must be even for nibble packing")
    return n, d, d // group_size


def quantize_nf4(weight: np.ndarray, group_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a float (n, d) weight matrix to nibble-packed NF4.

    Args:
        weight: 2-D array of shape (n, d); d must be even and divisible by
            ``group_size``.
        group_size: Columns per quantization group (per-group absmax scale).

    Returns:
        ``(packed, scales)`` where ``packed`` is uint8 ``(n, d // 2)`` and
        ``scales`` is float32 ``(n, d // group_size)``.
    """
    w = np.ascontiguousarray(weight, dtype=np.float32)
    n, d, n_groups = _validate_2d(w, group_size)

    grouped = w.reshape(n, n_groups, group_size)
    # Per-group absmax scale; guard all-zero groups against divide-by-zero.
    absmax = np.abs(grouped).max(axis=-1)                       # (n, n_groups)
    scales = np.where(absmax > 0, absmax, 1.0).astype(np.float32)
    normed = grouped / scales[:, :, np.newaxis]                # → [-1, 1]

    # Nearest codebook level for every weight (argmin over the 16 levels).
    flat = normed.reshape(n, d)
    diffs = np.abs(flat[:, :, np.newaxis] - _NF4_CODEBOOK[np.newaxis, np.newaxis, :])
    idx = np.argmin(diffs, axis=-1).astype(np.uint8)           # (n, d) in [0, 15]

    # Nibble-pack: even column → low nibble, odd column → high nibble.
    packed = (idx[:, 0::2] | (idx[:, 1::2] << np.uint8(4))).astype(np.uint8)
    return packed, scales


def dequantize_nf4(
    packed: np.ndarray, scales: np.ndarray, group_size: int = 32
) -> np.ndarray:
    """Reconstruct a float32 (n, d) weight matrix from nibble-packed NF4.

    Args:
        packed: uint8 array of shape (n, d // 2) from :func:`quantize_nf4`.
        scales: float32 array of shape (n, d // group_size).
        group_size: Columns per group (matches the value used at quantize time).

    Returns:
        Reconstructed float32 array of shape (n, d).
    """
    p = np.ascontiguousarray(packed, dtype=np.uint8)
    s = np.ascontiguousarray(scales, dtype=np.float32)
    n, half = p.shape
    d = half * 2

    idx = np.empty((n, d), dtype=np.uint8)
    idx[:, 0::2] = p & np.uint8(0x0F)            # low nibble  → even columns
    idx[:, 1::2] = (p >> np.uint8(4)) & np.uint8(0x0F)  # high nibble → odd columns

    vals = _NF4_CODEBOOK[idx.astype(np.intp)]    # (n, d) in [-1, 1]
    # Broadcast each group's scale across its group_size columns.
    scales_expanded = np.repeat(s, group_size, axis=1)[:, :d]
    return (vals * scales_expanded).astype(np.float32)
