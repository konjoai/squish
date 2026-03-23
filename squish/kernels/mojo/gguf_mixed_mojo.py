"""squish/kernels/mojo/gguf_mixed_mojo.py — Mojo-backed GGUF-style block quantisation.

Wraps the ``gguf_mixed_quant_kernel`` Mojo stub via MojoBridge with a NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "GGUFMixedMojoConfig",
    "MojoGGUFMixed",
]

_bridge = MojoBridge()
_quant_kernel = _bridge.load_kernel("gguf_mixed_quant_kernel")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_quant(
    w: np.ndarray, bits: int, group_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
            scales_list.append(rng / levels)
            mins_list.append(mn)
            q[r, g_start:g_start + gs] = np.round((chunk - mn) / rng * levels).clip(0, levels).astype(np.int8)
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
class GGUFMixedMojoConfig:
    """Configuration for :class:`MojoGGUFMixed`.

    Attributes:
        bits:       Default quantisation bits.
        group_size: Default elements per block.
    """

    bits: int = 4
    group_size: int = 32


class MojoGGUFMixed:
    """Mojo-backed GGUF-style mixed block quantisation.

    Falls back to NumPy when the Mojo runtime is absent.

    Example::

        gguf = MojoGGUFMixed()
        q, scales, mins, ss = gguf.quantize(W)
        W_hat = gguf.dequantize(q, scales, mins)
    """

    def __init__(self, config: Optional[GGUFMixedMojoConfig] = None) -> None:
        self._cfg = config or GGUFMixedMojoConfig()

    def quantize(
        self,
        w: np.ndarray,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """GGUF-style block quantisation with super-block meta-scales.

        Args:
            w:          ``(rows, cols)`` float32 weight matrix.
            bits:       Quantisation bits (overrides config).
            group_size: Elements per block (overrides config).

        Returns:
            Tuple of quantised ``(rows, cols)`` int8,
            scales ``(n_blocks,)`` float32,
            mins ``(n_blocks,)`` float32,
            super_scales ``(n_super,)`` float32.

        Raises:
            ValueError: If ``w`` is not 2-D.
        """
        arr = np.ascontiguousarray(w, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"w must be 2-D (rows, cols), got {arr.shape}")
        b = int(bits) if bits is not None else self._cfg.bits
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _quant_kernel is not None:
            rows, cols = arr.shape
            gpr = (cols + gs - 1) // gs
            n_blocks = rows * gpr
            n_super = (n_blocks + 7) // 8
            q = np.empty((rows, cols), dtype=np.int8)
            scales = np.empty(n_blocks, dtype=np.float32)
            mins = np.empty(n_blocks, dtype=np.float32)
            super_scales = np.empty(n_super, dtype=np.float32)
            _quant_kernel(
                arr.ctypes.data, q.ctypes.data, scales.ctypes.data,
                mins.ctypes.data, super_scales.ctypes.data,
                rows, cols, b, gs,
            )
            return q, scales, mins, super_scales
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
            ``(rows, cols)`` float32 reconstructed weights.
        """
        q = np.asarray(quantized, dtype=np.float32)
        rows, cols = q.shape
        gs = max(1, int(group_size) if group_size is not None else self._cfg.group_size)
        gpr = (cols + gs - 1) // gs
        out = np.empty_like(q)
        for r in range(rows):
            for g_idx, g_start in enumerate(range(0, cols, gs)):
                scale = scales[r * gpr + g_idx]
                mn = mins[r * gpr + g_idx]
                out[r, g_start:g_start + gs] = q[r, g_start:g_start + gs] * scale + mn
        return out.astype(np.float32)

    def backend(self) -> str:
        return "mojo" if _quant_kernel is not None else "numpy"
