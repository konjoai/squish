"""squish/kernels/mojo/bit_distiller_mojo.py — Mojo-backed BitDistiller weight quantisation.

Wraps the ``bit_distiller_quant_kernel`` Mojo stub via MojoBridge with a NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "BitDistillerMojoConfig",
    "MojoBitDistiller",
]

_bridge = MojoBridge()
_quant_kernel = _bridge.load_kernel("bit_distiller_quant_kernel")


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_quant(
    w: np.ndarray, bits: int, group_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = w.shape
    gs = max(1, group_size)
    levels = float((1 << bits) - 1)
    q = np.empty_like(w, dtype=np.int8)
    scales_list: list[float] = []
    zeros_list: list[float] = []
    for r in range(rows):
        row = w[r]
        for g_start in range(0, cols, gs):
            chunk = row[g_start:g_start + gs]
            mn = float(chunk.min())
            mx = float(chunk.max())
            rng = max(mx - mn, 1e-8)
            scale = rng / levels
            zero = -mn / scale
            scales_list.append(scale)
            zeros_list.append(zero)
            q[r, g_start:g_start + gs] = np.round((chunk - mn) / rng * levels).clip(0, levels).astype(np.int8)
    return q, np.array(scales_list, dtype=np.float32), np.array(zeros_list, dtype=np.float32)


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class BitDistillerMojoConfig:
    """Configuration for :class:`MojoBitDistiller`.

    Attributes:
        bits:       Default quantisation bits.
        group_size: Default elements per group.
    """

    bits: int = 4
    group_size: int = 128


class MojoBitDistiller:
    """Mojo-backed BitDistiller per-group weight quantisation.

    Falls back to NumPy when the Mojo runtime is absent.

    Example::

        bd = MojoBitDistiller()
        q, scales, zeros = bd.quantize(W)
        W_hat = bd.dequantize(q, scales, zeros)
    """

    def __init__(self, config: Optional[BitDistillerMojoConfig] = None) -> None:
        self._cfg = config or BitDistillerMojoConfig()

    def quantize(
        self,
        w: np.ndarray,
        bits: Optional[int] = None,
        group_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per-group asymmetric quantisation.

        Args:
            w:          ``(rows, cols)`` float32 weight matrix.
            bits:       Quantisation bits (overrides config).
            group_size: Elements per group (overrides config).

        Returns:
            Tuple of quantised ``(rows, cols)`` int8,
            scales ``(n_blocks,)`` float32,
            zeros ``(n_blocks,)`` float32.

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
            q = np.empty((rows, cols), dtype=np.int8)
            scales = np.empty(rows * gpr, dtype=np.float32)
            zeros = np.empty(rows * gpr, dtype=np.float32)
            _quant_kernel(
                arr.ctypes.data, q.ctypes.data, scales.ctypes.data, zeros.ctypes.data,
                rows, cols, b, gs,
            )
            return q, scales, zeros
        return _numpy_quant(arr, b, gs)

    def dequantize(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        zeros: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct float32 weights.

        Args:
            quantized:  ``(rows, cols)`` int8.
            scales:     ``(n_blocks,)`` float32.
            zeros:      ``(n_blocks,)`` float32.
            group_size: Elements per group (overrides config).

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
                zero = zeros[r * gpr + g_idx]
                out[r, g_start:g_start + gs] = (q[r, g_start:g_start + gs] - zero) * scale
        return out.astype(np.float32)

    def backend(self) -> str:
        return "mojo" if _quant_kernel is not None else "numpy"
