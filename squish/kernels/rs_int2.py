"""Rust-backed INT2 packed quantization kernel.

Wraps ``squish_quant.{quantize,dequantize}_int2_grouped_{f32,bf16}`` from the
maturin-compiled Rust extension.  Falls back to a pure-NumPy reference
implementation when the extension is unavailable.

Encoding: 2-bit unsigned values [0–3] with per-group zero-point + scale.
4 values packed per byte, low-index element at the least-significant 2 bits.
Yields a 16× compression ratio versus float32 (32 bits → 2 bits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["INT2KernelConfig", "RustINT2Kernel"]

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class INT2KernelConfig:
    """Configuration for :class:`RustINT2Kernel`.

    Attributes
    ----------
    group_size:
        Number of elements per quantization group (must divide ``n_cols``
        and ``n_cols`` must be divisible by 4).
    use_bf16_input:
        When ``True``, the Rust kernel accepts raw uint16 BF16 bits, avoiding
        a Python-side dtype conversion copy.
    """

    group_size: int = 64
    use_bf16_input: bool = False


class RustINT2Kernel:
    """INT2 quantization kernel backed by Rust (falls back to NumPy).

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`INT2KernelConfig`.
    """

    def __init__(self, config: INT2KernelConfig | None = None) -> None:
        self.config = config or INT2KernelConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def pack(
        self, W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize and pack a 2-D weight matrix to INT2.

        Parameters
        ----------
        W:
            Float32 or uint16 (BF16 bits) array of shape ``(N, D)``.
            ``D`` must be divisible by both ``group_size`` and 4.

        Returns
        -------
        packed:
            uint8 array of shape ``(N, D // 4)`` — four 2-bit values per byte.
        scales:
            float32 array of shape ``(N, D // group_size)`` — per-group range.
        zero_points:
            float32 array of shape ``(N, D // group_size)`` — per-group minimum.
        """
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        gs = self.config.group_size

        if _RUST_AVAILABLE:
            try:
                if self.config.use_bf16_input and W.dtype == np.uint16:
                    packed, scales, zp = _sq.quantize_int2_grouped_bf16(W, gs)
                else:
                    packed, scales, zp = _sq.quantize_int2_grouped_f32(
                        W.astype(np.float32, copy=False), gs
                    )
                return packed, scales, zp
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_pack(W.astype(np.float32, copy=False), gs)

    def unpack(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Unpack INT2 bytes back to float32.

        Parameters
        ----------
        packed:
            uint8 array of shape ``(N, D // 4)``.
        scales:
            float32 array of shape ``(N, D // group_size)``.
        zero_points:
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
                return _sq.dequantize_int2_grouped_f32(
                    packed, scales, zero_points, self.config.group_size
                )
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_unpack(packed, scales, zero_points, self.config.group_size, n_cols)

    def compression_ratio(self) -> float:
        """Return the theoretical storage compression ratio (16.0).

        float32 (32 bits) → INT2 (2 bits) = 16× reduction.
        """
        return 16.0

    # ------------------------------------------------------------------ #
    #  NumPy fallback implementations                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_pack(
        W: np.ndarray, group_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_rows, n_cols = W.shape
        n_groups = n_cols // group_size
        n_packed = n_cols // 4

        scales = np.empty((n_rows, n_groups), dtype=np.float32)
        zero_pts = np.empty((n_rows, n_groups), dtype=np.float32)
        packed = np.zeros((n_rows, n_packed), dtype=np.uint8)

        groups = W.reshape(n_rows, n_groups, group_size)
        gmin = groups.min(axis=-1)
        gmax = groups.max(axis=-1)
        g_range = gmax - gmin
        scales[:] = np.where(g_range == 0, 1.0, g_range / 3.0)
        zero_pts[:] = gmin

        for row_idx in range(n_rows):
            for i in range(n_packed):
                byte = 0
                for bit in range(4):
                    j = i * 4 + bit
                    g = j // group_size
                    q = round((W[row_idx, j] - zero_pts[row_idx, g]) / scales[row_idx, g])
                    q = int(max(0, min(3, q)))
                    byte |= (q & 0x03) << (bit * 2)
                packed[row_idx, i] = byte

        return packed, scales, zero_pts

    @staticmethod
    def _numpy_unpack(
        packed: np.ndarray,
        scales: np.ndarray,
        zero_points: np.ndarray,
        group_size: int,
        n_cols: int,
    ) -> np.ndarray:
        n_rows, n_packed = packed.shape
        out = np.zeros((n_rows, n_cols), dtype=np.float32)

        for row_idx in range(n_rows):
            for i in range(n_packed):
                byte = int(packed[row_idx, i])
                for bit in range(4):
                    j = i * 4 + bit
                    if j >= n_cols:
                        break
                    q = (byte >> (bit * 2)) & 0x03
                    g = j // group_size
                    out[row_idx, j] = q * scales[row_idx, g] + zero_points[row_idx, g]

        return out
