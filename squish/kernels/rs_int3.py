"""Rust-backed INT3 packed quantization kernel.

Wraps ``squish_quant.{pack,unpack}_int3_grouped_{f32,}`` from the maturin-
compiled Rust extension.  Falls back to a pure-NumPy reference implementation
when the extension is unavailable.

Encoding: 3-bit symmetric signed range [-3, +3].  8 values packed into 3 bytes
(24 bits), low-index element at the least-significant bits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["INT3KernelConfig", "RustINT3Kernel"]

_INT3_MAX: int = 3  # max signed magnitude
_INT3_LEVELS: int = 7  # [-3, -2, -1, 0, 1, 2, 3]

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class INT3KernelConfig:
    """Configuration for :class:`RustINT3Kernel`.

    Attributes
    ----------
    group_size:
        Number of elements per quantization group (must divide ``n_cols``).
    """

    group_size: int = 128


class RustINT3Kernel:
    """INT3 quantization kernel backed by Rust (falls back to NumPy).

    Compresses float32 weights to 3-bit signed integers at ~5.3× lower
    storage than float32 (16 / 3 ≈ 5.3).

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`INT3KernelConfig`.
    """

    def __init__(self, config: INT3KernelConfig | None = None) -> None:
        self.config = config or INT3KernelConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def pack(
        self, W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize and pack a 2-D weight matrix to INT3.

        Parameters
        ----------
        W:
            Float32 array of shape ``(N, D)``.

        Returns
        -------
        packed:
            uint8 array of shape ``(N, ceil(D * 3 / 8) * (8 // 8))``.
            Precisely: ``ceil(D / 8) * 3`` bytes per row.
        scales:
            float32 array of shape ``(N, D // group_size)``.
        """
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        W32 = W.astype(np.float32, copy=False)
        gs = self.config.group_size

        if _RUST_AVAILABLE:
            try:
                packed, scales = _sq.pack_int3_grouped_f32(W32, gs)
                return packed, scales
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_pack(W32, gs)

    def unpack(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Unpack INT3 bytes back to float32.

        Parameters
        ----------
        packed:
            uint8 array of shape ``(N, *)``.
        scales:
            float32 array of shape ``(N, D // group_size)``.
        original_shape:
            ``(N, D)`` — the shape of the weight matrix before packing.

        Returns
        -------
        float32 array of shape ``(N, D)``.
        """
        n_rows, n_cols = original_shape

        if _RUST_AVAILABLE:
            try:
                return _sq.unpack_int3_grouped(
                    packed, scales, self.config.group_size, n_cols
                )
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_unpack(packed, scales, self.config.group_size, n_cols)

    def packed_size_bytes(self, n_elements: int) -> int:
        """Return the number of packed bytes for *n_elements* INT3 values.

        8 INT3 values occupy exactly 3 bytes (24 bits).
        """
        return math.ceil(n_elements * 3 / 8)

    # ------------------------------------------------------------------ #
    #  NumPy fallback implementations                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_pack(
        W: np.ndarray, group_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_rows, n_cols = W.shape
        n_groups = n_cols // group_size
        padded = ((n_cols + 7) // 8) * 8
        n_packed = padded * 3 // 8

        scales = np.empty((n_rows, n_groups), dtype=np.float32)
        packed = np.zeros((n_rows, n_packed), dtype=np.uint8)

        for row_idx in range(n_rows):
            row = W[row_idx]
            for g in range(n_groups):
                s = row[g * group_size : (g + 1) * group_size]
                abs_max = np.abs(s).max()
                scales[row_idx, g] = 1.0 if abs_max == 0 else abs_max / 3.0

            # Quantize to signed int3 [-3, 3] → stored as unsigned 3-bit [0, 7]
            quant = np.zeros(padded, dtype=np.uint8)
            for j in range(n_cols):
                g = j // group_size
                v = round(row[j] / scales[row_idx, g])
                v = max(-3, min(3, v))
                quant[j] = int(v) & 0x07

            # Pack 8 values → 3 bytes
            for chunk in range(padded // 8):
                b = quant[chunk * 8 : chunk * 8 + 8]
                byte0 = b[0] | (b[1] << 3) | ((b[2] & 0x03) << 6)
                byte1 = ((b[2] >> 2) & 0x01) | (b[3] << 1) | (b[4] << 4) | ((b[5] & 0x01) << 7)
                byte2 = ((b[5] >> 1) & 0x03) | (b[6] << 2) | (b[7] << 5)
                pb = chunk * 3
                packed[row_idx, pb]     = byte0
                packed[row_idx, pb + 1] = byte1
                packed[row_idx, pb + 2] = byte2

        return packed, scales

    @staticmethod
    def _numpy_unpack(
        packed: np.ndarray,
        scales: np.ndarray,
        group_size: int,
        n_cols: int,
    ) -> np.ndarray:
        n_rows = packed.shape[0]
        out = np.zeros((n_rows, n_cols), dtype=np.float32)

        for row_idx in range(n_rows):
            p = packed[row_idx]
            s = scales[row_idx]
            chunks = (n_cols + 7) // 8
            for chunk in range(chunks):
                pb = chunk * 3
                byte0 = int(p[pb])     if pb     < len(p) else 0
                byte1 = int(p[pb + 1]) if pb + 1 < len(p) else 0
                byte2 = int(p[pb + 2]) if pb + 2 < len(p) else 0

                vals = [
                    byte0 & 0x07,
                    (byte0 >> 3) & 0x07,
                    ((byte0 >> 6) & 0x03) | ((byte1 & 0x01) << 2),
                    (byte1 >> 1) & 0x07,
                    (byte1 >> 4) & 0x07,
                    ((byte1 >> 7) & 0x01) | ((byte2 & 0x03) << 1),
                    (byte2 >> 2) & 0x07,
                    (byte2 >> 5) & 0x07,
                ]
                for bit, raw in enumerate(vals):
                    j = chunk * 8 + bit
                    if j >= n_cols:
                        break
                    signed = raw - 8 if raw >= 4 else raw
                    g = j // group_size
                    out[row_idx, j] = signed * s[g]

        return out
