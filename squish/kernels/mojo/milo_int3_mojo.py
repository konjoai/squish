"""squish/kernels/mojo/milo_int3_mojo.py — Mojo-backed MILO INT3 quantisation + bitpacking.

Wraps the ``milo_int3_pack_kernel`` Mojo stub via MojoBridge with a NumPy fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge

__all__ = [
    "MiloINT3MojoConfig",
    "MojoMiloINT3",
]

_bridge = MojoBridge()
_pack_kernel = _bridge.load_kernel("milo_int3_pack_kernel")
_quant_kernel = _bridge.load_kernel("milo_quant_kernel")

_MAX_INT3 = 3.0


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_quant(w: np.ndarray, group_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = w.shape
    gs = max(1, group_size)
    q = np.empty_like(w, dtype=np.int8)
    scales_list: list[float] = []
    zeros_list: list[float] = []
    for r in range(rows):
        row = w[r]
        for g_start in range(0, cols, gs):
            chunk = row[g_start:g_start + gs]
            scale = max(float(np.abs(chunk).max()) / _MAX_INT3, 1e-8)
            scales_list.append(scale)
            zeros_list.append(0.0)
            q[r, g_start:g_start + gs] = np.round(chunk / scale).clip(-_MAX_INT3, _MAX_INT3).astype(np.int8)
    return q, np.array(scales_list, dtype=np.float32), np.array(zeros_list, dtype=np.float32)


def _numpy_pack(values: np.ndarray) -> np.ndarray:
    n = len(values)
    out_bytes = (n * 3 + 7) // 8
    packed = np.zeros(out_bytes, dtype=np.uint8)
    for i, v in enumerate(values):
        v3 = int(v) & 0x7
        bit_offset = i * 3
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8
        packed[byte_idx] |= (v3 << bit_shift) & 0xFF
        carry = v3 >> (8 - bit_shift)
        if carry and byte_idx + 1 < out_bytes:
            packed[byte_idx + 1] |= carry & 0xFF
    return packed


def _numpy_unpack(packed: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bit_offset = i * 3
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8
        v3 = (int(packed[byte_idx]) >> bit_shift) & 0x7
        if bit_shift > 5 and byte_idx + 1 < len(packed):
            v3 |= (int(packed[byte_idx + 1]) << (8 - bit_shift)) & 0x7
        out[i] = np.int8(v3 if v3 < 4 else v3 - 8)
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class MiloINT3MojoConfig:
    """Configuration for :class:`MojoMiloINT3`.

    Attributes:
        group_size: Default elements per quantisation group.
    """

    group_size: int = 128


class MojoMiloINT3:
    """Mojo-backed MILO INT3 quantisation and bitpacking.

    Falls back to NumPy when the Mojo runtime is absent.

    Example::

        milo = MojoMiloINT3()
        q, scales, zeros = milo.quantize(W)
        packed = milo.pack(q.ravel())
        q_back = milo.unpack(packed, W.size)
    """

    def __init__(self, config: Optional[MiloINT3MojoConfig] = None) -> None:
        self._cfg = config or MiloINT3MojoConfig()

    def quantize(
        self,
        w: np.ndarray,
        group_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Group-wise symmetric INT3 quantisation.

        Args:
            w:          ``(rows, cols)`` float32 weight matrix.
            group_size: Elements per group (overrides config).

        Returns:
            Tuple of quantised ``(rows, cols)`` int8,
            scales ``(n_groups,)`` float32,
            zeros ``(n_groups,)`` float32.

        Raises:
            ValueError: If ``w`` is not 2-D.
        """
        arr = np.ascontiguousarray(w, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"w must be 2-D (rows, cols), got {arr.shape}")
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _quant_kernel is not None:
            rows, cols = arr.shape
            gpr = (cols + gs - 1) // gs
            q = np.empty((rows, cols), dtype=np.int8)
            scales = np.empty(rows * gpr, dtype=np.float32)
            zeros = np.zeros(rows * gpr, dtype=np.float32)
            _quant_kernel(arr.ctypes.data, q.ctypes.data, scales.ctypes.data, rows, cols, gs)
            return q, scales, zeros
        return _numpy_quant(arr, gs)

    def pack(self, values: np.ndarray) -> np.ndarray:
        """Pack INT3 values (i8) into compact u8 bytes.

        Args:
            values: ``(N,)`` int8 values in range −4..3.

        Returns:
            ``(ceil(N × 3 / 8),)`` uint8 packed bytes.

        Raises:
            ValueError: If ``values`` is not 1-D.
        """
        arr = np.ascontiguousarray(values, dtype=np.int8)
        if arr.ndim != 1:
            raise ValueError(f"values must be 1-D, got {arr.shape}")
        if _pack_kernel is not None:
            n = len(arr)
            out_bytes = (n * 3 + 7) // 8
            out = np.zeros(out_bytes, dtype=np.uint8)
            _pack_kernel(arr.ctypes.data, out.ctypes.data, n)
            return out
        return _numpy_pack(arr)

    def unpack(self, packed: np.ndarray, n: int) -> np.ndarray:
        """Unpack u8 bytes back to INT3 values (i8).

        Args:
            packed: ``(ceil(N × 3 / 8),)`` uint8 packed bytes.
            n:      Number of INT3 values to restore.

        Returns:
            ``(n,)`` int8 values.
        """
        return _numpy_unpack(np.asarray(packed, dtype=np.uint8), n)

    def dequantize(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        group_size: Optional[int] = None,
    ) -> np.ndarray:
        """Reconstruct float32 weights from symmetric INT3 form.

        Args:
            quantized:  ``(rows, cols)`` int8.
            scales:     ``(n_groups,)`` float32.
            group_size: Elements per group (overrides config).

        Returns:
            ``(rows, cols)`` float32.
        """
        q = np.asarray(quantized, dtype=np.float32)
        rows, cols = q.shape
        gs = max(1, int(group_size) if group_size is not None else self._cfg.group_size)
        gpr = (cols + gs - 1) // gs
        out = np.empty_like(q)
        for r in range(rows):
            for g_idx, g_start in enumerate(range(0, cols, gs)):
                out[r, g_start:g_start + gs] = q[r, g_start:g_start + gs] * scales[r * gpr + g_idx]
        return out.astype(np.float32)

    def backend(self) -> str:
        return "mojo" if (_pack_kernel is not None and _quant_kernel is not None) else "numpy"
