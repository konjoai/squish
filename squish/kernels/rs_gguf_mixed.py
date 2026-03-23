"""squish/kernels/rs_gguf_mixed.py — Rust-backed GGUF-style block quantisation.

Wraps ``squish_quant_rs.gguf_mixed_quant_f32`` with a NumPy fallback.

GGUF uses a two-tier scale hierarchy: each block of ``group_size`` weights has
its own ``scale`` and ``min``, and every 8 consecutive blocks share a
``super_scale`` (the mean of their block scales) used for further compression.

Reference: GGML/GGUF quantisation format as used in llama.cpp, 2023–.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "GGUFMixedConfig",
    "RustGGUFMixed",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "gguf_mixed_quant_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_quant(
    w: np.ndarray,
    bits: int,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-block min/max quant + super-block meta-scale.

    Args:
        w:          ``(rows, cols)`` float32.
        bits:       Quantisation bits.
        group_size: Elements per block.

    Returns:
        Tuple of quantised ``(rows, cols)`` int8,
        scales ``(n_blocks,)`` float32,
        mins ``(n_blocks,)`` float32,
        super_scales ``(n_super,)`` float32.
    """
    rows, cols = w.shape
    gs = max(1, group_size)
    levels = float((1 << bits) - 1)
    super_blk = 8
    q = np.empty_like(w, dtype=np.int8)
    scales_list: list[float] = []
    mins_list: list[float] = []
    for r in range(rows):
        row = w[r]
        for g_start in range(0, cols, gs):
            chunk = row[g_start:g_start + gs]
            mn = float(chunk.min())
            mx = float(chunk.max())
            rng = max(mx - mn, 1e-8)
            scale = rng / levels
            scales_list.append(scale)
            mins_list.append(mn)
            qchunk = np.round((chunk - mn) / rng * levels).clip(0, levels).astype(np.int8)
            q[r, g_start:g_start + gs] = qchunk
    scales_arr = np.array(scales_list, dtype=np.float32)
    mins_arr = np.array(mins_list, dtype=np.float32)
    n_blocks = len(scales_list)
    n_super = (n_blocks + super_blk - 1) // super_blk
    super_scales = np.array([
        max(1e-8, float(scales_arr[i * super_blk:(i + 1) * super_blk].mean()))
        for i in range(n_super)
    ], dtype=np.float32)
    return q, scales_arr, mins_arr, super_scales


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class GGUFMixedConfig:
    """Configuration for :class:`RustGGUFMixed`.

    Attributes:
        bits:       Default quantisation bits.
        group_size: Default elements per block.
    """

    bits: int = 4
    group_size: int = 32


class RustGGUFMixed:
    """Rust-accelerated GGUF-style mixed block quantisation.

    Quantises weight matrices with per-block min/max scaling and a two-level
    super-block meta-scale.  Falls back to NumPy when ``squish_quant_rs`` is
    unavailable.

    Example::

        gguf = RustGGUFMixed()
        q, scales, mins, ss = gguf.quantize(W)
        W_hat = gguf.dequantize(q, scales, mins)
    """

    def __init__(self, config: Optional[GGUFMixedConfig] = None) -> None:
        self._cfg = config or GGUFMixedConfig()

    def quantize(
        self,
        w: np.ndarray,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Quantise weight matrix with GGUF-style block scaling.

        Args:
            w:          ``(rows, cols)`` float32 weight matrix.
            bits:       Quantisation bits (overrides config).
            group_size: Elements per block (overrides config).

        Returns:
            Tuple of:
            - ``quantized``:    ``(rows, cols)`` int8.
            - ``scales``:       ``(n_blocks,)`` float32.
            - ``mins``:         ``(n_blocks,)`` float32.
            - ``super_scales``: ``(n_super,)`` float32.

        Raises:
            ValueError: If ``w`` is not 2-D.
        """
        arr = np.ascontiguousarray(w, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"w must be 2-D (rows, cols), got {arr.shape}")
        b = int(bits) if bits is not None else self._cfg.bits
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _HAS_RUST:
            q, s, m, ss = _sq.gguf_mixed_quant_f32(arr, b, gs)
            return (
                np.asarray(q, dtype=np.int8),
                np.asarray(s, dtype=np.float32),
                np.asarray(m, dtype=np.float32),
                np.asarray(ss, dtype=np.float32),
            )
        return _numpy_quant(arr, b, gs)

    def dequantize(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        mins: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct float32 weights from block-quantised form.

        Args:
            quantized:  ``(rows, cols)`` int8.
            scales:     ``(n_blocks,)`` float32.
            mins:       ``(n_blocks,)`` float32.
            group_size: Elements per block (overrides config).

        Returns:
            ``(rows, cols)`` float32 reconstructed weight matrix.
        """
        q = np.asarray(quantized, dtype=np.float32)
        rows, cols = q.shape
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        gs = max(1, gs)
        groups_per_row = (cols + gs - 1) // gs
        out = np.empty_like(q)
        for r in range(rows):
            for g_idx, g_start in enumerate(range(0, cols, gs)):
                scale = scales[r * groups_per_row + g_idx]
                mn = mins[r * groups_per_row + g_idx]
                chunk = q[r, g_start:g_start + gs]
                out[r, g_start:g_start + gs] = chunk * scale + mn
        return out.astype(np.float32)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
