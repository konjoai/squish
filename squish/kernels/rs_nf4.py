"""Rust-backed NF4 (NormalFloat4) quantization kernel.

Wraps ``squish_quant.{quantize,dequantize}_nf4_grouped_{f32,bf16}`` from the
maturin-compiled Rust extension.  Falls back to a pure-NumPy reference
implementation when the extension is unavailable.

Reference: Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs",
arXiv 2305.14314.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

__all__ = ["NF4KernelConfig", "RustNF4Kernel"]

# Standard-normal quantile function — 16 uniformly spaced probability levels
# (QLoRA Table 1, arXiv 2305.14314).
_NF4_LUT: np.ndarray = np.array(
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

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class NF4KernelConfig:
    """Configuration for :class:`RustNF4Kernel`.

    Attributes
    ----------
    group_size:
        Number of elements per quantization group (must divide ``n_cols``).
    use_bf16_input:
        When ``True``, the Rust kernel accepts raw uint16 BF16 bits, avoiding
        a Python-side dtype conversion copy.
    """

    group_size: int = 64
    use_bf16_input: bool = False


class RustNF4Kernel:
    """NF4 quantization kernel backed by Rust (falls back to NumPy).

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`NF4KernelConfig`.
    """

    def __init__(self, config: NF4KernelConfig | None = None) -> None:
        self.config = config or NF4KernelConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def quantize(
        self, W: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize a 2-D weight matrix to NF4.

        Parameters
        ----------
        W:
            Float32 or uint16 (BF16 bits) weight matrix of shape ``(N, D)``.

        Returns
        -------
        packed:
            uint8 array of shape ``(N, D // 2)`` — two nibbles per byte.
        scales:
            float32 array of shape ``(N, D // group_size)`` — per-group
            absolute-maximum scales.
        """
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        gs = self.config.group_size

        if _RUST_AVAILABLE:
            try:
                if self.config.use_bf16_input and W.dtype == np.uint16:
                    packed, scales = _sq.quantize_nf4_grouped_bf16(W, gs)
                else:
                    packed, scales = _sq.quantize_nf4_grouped_f32(
                        W.astype(np.float32, copy=False), gs
                    )
                return packed, scales
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_quantize(W.astype(np.float32, copy=False), gs)

    def dequantize(
        self, packed: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        """Dequantize NF4 packed weights.

        Parameters
        ----------
        packed:
            uint8 array of shape ``(N, D // 2)``.
        scales:
            float32 array of shape ``(N, D // group_size)``.

        Returns
        -------
        float32 array of shape ``(N, D)``.
        """
        if _RUST_AVAILABLE:
            try:
                return _sq.dequantize_nf4_grouped_f32(
                    packed, scales, self.config.group_size
                )
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_dequantize(packed, scales, self.config.group_size)

    def speedup_estimate(self) -> float:
        """Return an estimated Rust-vs-NumPy throughput multiplier (~10×)."""
        return 10.0

    # ------------------------------------------------------------------ #
    #  NumPy fallback implementations                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_quantize(
        W: np.ndarray, group_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_rows, n_cols = W.shape
        n_groups = n_cols // group_size
        n_packed = n_cols // 2

        groups = W.reshape(n_rows, n_groups, group_size)
        abs_max = np.abs(groups).max(axis=-1)  # (n_rows, n_groups)
        abs_max[abs_max == 0] = 1.0
        # Broadcast scales to full (n_rows, n_cols) shape
        scale_full = np.repeat(abs_max, group_size, axis=1)  # (n_rows, n_cols)
        W_scaled = (W / scale_full).reshape(n_rows, n_cols, 1)
        # Nearest NF4 level: broadcast over 16-entry LUT
        indices = np.argmin(np.abs(W_scaled - _NF4_LUT), axis=-1).astype(np.uint8)

        packed = (indices[:, 0::2] & 0x0F) | ((indices[:, 1::2] & 0x0F) << 4)
        scales = abs_max.reshape(n_rows, n_groups)
        return packed.astype(np.uint8), scales.astype(np.float32)

    @staticmethod
    def _numpy_dequantize(
        packed: np.ndarray, scales: np.ndarray, group_size: int
    ) -> np.ndarray:
        n_rows, n_packed = packed.shape
        n_cols = n_packed * 2
        n_groups = n_cols // group_size

        lo = (packed & 0x0F).astype(np.int32)
        hi = ((packed >> 4) & 0x0F).astype(np.int32)
        indices = np.empty((n_rows, n_cols), dtype=np.int32)
        indices[:, 0::2] = lo
        indices[:, 1::2] = hi

        out = _NF4_LUT[indices]
        scale_full = np.repeat(scales, group_size, axis=1)
        return (out * scale_full).astype(np.float32)
