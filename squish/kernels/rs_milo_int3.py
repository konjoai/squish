"""squish/kernels/rs_milo_int3.py — Rust-backed MILO INT3 quantisation + bitpacking.

Wraps ``squish_quant_rs.milo_quant_f32`` and ``squish_quant_rs.milo_pack_int3_u8``
with NumPy fallbacks.

MILO (Minimum-Information-Loss Quantisation) uses 3-bit symmetric group-wise
quantisation followed by compact bitpacking (8 INT3 values → 3 bytes) to
achieve sub-4-bit weight storage with minimal calibration overhead.

Reference: Inspired by MILO sub-4-bit quantisation techniques for LLM
inference efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "MiloINT3Config",
    "RustMiloINT3",
]

try:
    import squish_quant as _sq
    _HAS_QUANT = hasattr(_sq, "milo_quant_f32")
    _HAS_PACK = hasattr(_sq, "milo_pack_int3_u8")
    _HAS_RUST = _HAS_QUANT and _HAS_PACK
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_QUANT = _HAS_PACK = _HAS_RUST = False

_MAX_INT3 = 3.0  # symmetric INT3: clamp to ±3


# ── NumPy fallbacks ───────────────────────────────────────────────────────────


def _numpy_quant(
    w: np.ndarray,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-wise symmetric INT3 quantisation.

    Args:
        w:          ``(rows, cols)`` float32.
        group_size: Elements per group.

    Returns:
        Tuple of quantised ``(rows, cols)`` int8,
        scales ``(n_groups,)`` float32,
        zeros ``(n_groups,)`` float32 (always 0.0 for symmetric).
    """
    rows, cols = w.shape
    gs = max(1, group_size)
    q = np.empty_like(w, dtype=np.int8)
    scales_list: list[float] = []
    zeros_list: list[float] = []
    for r in range(rows):
        row = w[r]
        for g_start in range(0, cols, gs):
            chunk = row[g_start:g_start + gs]
            abs_max = float(np.abs(chunk).max())
            scale = max(abs_max / _MAX_INT3, 1e-8)
            scales_list.append(scale)
            zeros_list.append(0.0)
            qchunk = np.round(chunk / scale).clip(-_MAX_INT3, _MAX_INT3).astype(np.int8)
            q[r, g_start:g_start + gs] = qchunk
    return q, np.array(scales_list, dtype=np.float32), np.array(zeros_list, dtype=np.float32)


def _numpy_pack(values: np.ndarray) -> np.ndarray:
    """Pack 1-D array of INT3 values (i8) into u8 bytes.

    8 INT3 values → 3 bytes (24 bits).

    Args:
        values: ``(N,)`` int8 values in range −4..3.

    Returns:
        ``(ceil(N × 3 / 8),)`` uint8 packed bytes.
    """
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
    """Unpack u8 bytes back into INT3 values (i8).

    Args:
        packed: ``(ceil(N × 3 / 8),)`` uint8 packed bytes.
        n:      Number of INT3 values to restore.

    Returns:
        ``(n,)`` int8 values in range −4..3.
    """
    out = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bit_offset = i * 3
        byte_idx = bit_offset // 8
        bit_shift = bit_offset % 8
        v3 = (int(packed[byte_idx]) >> bit_shift) & 0x7
        if bit_shift > 5 and byte_idx + 1 < len(packed):
            v3 |= (int(packed[byte_idx + 1]) << (8 - bit_shift)) & 0x7
        # Sign-extend 3-bit signed: values 0–3 positive, 4–7 → −4..−1
        out[i] = np.int8(v3 if v3 < 4 else v3 - 8)
    return out


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class MiloINT3Config:
    """Configuration for :class:`RustMiloINT3`.

    Attributes:
        group_size: Default elements per quantisation group.
    """

    group_size: int = 128


class RustMiloINT3:
    """Rust-accelerated MILO INT3 quantisation and bitpacking.

    Provides group-wise symmetric f32→INT3 quantisation, compact 3-bit
    bitpacking (8 values → 3 bytes), and the inverse operations.  Falls back
    to NumPy when ``squish_quant_rs`` is unavailable.

    Example::

        milo = RustMiloINT3()
        q, scales, zeros = milo.quantize(W)
        packed = milo.pack(q.ravel())
        q_back = milo.unpack(packed, W.size)
    """

    def __init__(self, config: Optional[MiloINT3Config] = None) -> None:
        self._cfg = config or MiloINT3Config()

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
            Tuple of:
            - ``quantized``: ``(rows, cols)`` int8 in range −3..3.
            - ``scales``:    ``(n_groups,)`` float32.
            - ``zeros``:     ``(n_groups,)`` float32 (always 0.0).

        Raises:
            ValueError: If ``w`` is not 2-D.
        """
        arr = np.ascontiguousarray(w, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"w must be 2-D (rows, cols), got {arr.shape}")
        gs = int(group_size) if group_size is not None else self._cfg.group_size
        if _HAS_QUANT:
            q, s, z = _sq.milo_quant_f32(arr, gs)
            return (
                np.asarray(q, dtype=np.int8),
                np.asarray(s, dtype=np.float32),
                np.asarray(z, dtype=np.float32),
            )
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
        if _HAS_PACK:
            return np.asarray(_sq.milo_pack_int3_u8(arr), dtype=np.uint8)
        return _numpy_pack(arr)

    def unpack(self, packed: np.ndarray, n: int) -> np.ndarray:
        """Unpack u8 bytes back to INT3 values (i8).

        Args:
            packed: ``(ceil(N × 3 / 8),)`` uint8 packed bytes.
            n:      Number of INT3 values to restore.

        Returns:
            ``(n,)`` int8 values in range −4..3.
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
            quantized:  ``(rows, cols)`` int8 in range −3..3.
            scales:     ``(n_groups,)`` float32.
            group_size: Elements per group (overrides config).

        Returns:
            ``(rows, cols)`` float32 reconstructed weights.
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
                out[r, g_start:g_start + gs] = q[r, g_start:g_start + gs] * scale
        return out.astype(np.float32)

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
