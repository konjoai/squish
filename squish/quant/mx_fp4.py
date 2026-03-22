"""squish/quant/mx_fp4.py

MxFP4 — OCP MXFP4 Block-Scaling 4-bit Quantization.

Reference
---------
OCP Microscaling (MX) Formats Specification v1.0, Open Compute Project, 2023.
Rouhani et al. "Microscaling Data Formats for Deep Learning." arXiv:2310.10537.

Algorithm
---------
MXFP4 uses *block scaling*: every ``block_size`` consecutive elements share a
single FP8 scale (E8M0).  Each element is stored as a 4-bit micro-scaled float
in E2M1 format (2 exponent bits, 1 mantissa bit).

E2M1 representable values (sign × 2^(e-1) × (1 + m/2)):
  0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0  (and negatives)

This module provides a NumPy reference implementation:
* ``quantize(x)`` → ``MxFP4Result`` with code array and per-block scales.
* ``dequantize(result)`` → approximate FP32 reconstruction.

Key properties
--------------
* Block size = 32 (OCP default).
* Per-block scale stored as FP8 (E8M0 simulation via float32 power-of-two).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "MxFP4Config",
    "MxFP4Result",
    "MxFP4",
]

# E2M1 representable magnitudes (excluding sign and 0)
_E2M1_VALUES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
# All signed values including 0
_E2M1_ALL = np.concatenate([[-v for v in reversed(_E2M1_VALUES)], [0.0], [float(v) for v in _E2M1_VALUES]], dtype=np.float32)
# Codes 0..15: 0 = -6.0, 7 = 0.0, 15 = 6.0
_E2M1_CODE_TABLE = np.array(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0],
    dtype=np.float32,
)  # 16 entries — code=15 reserved → 0.0 (NaN in E2M1)


@dataclass
class MxFP4Config:
    """Configuration for :class:`MxFP4`.

    Attributes:
        block_size: Number of elements sharing one FP8 block scale.
    """

    block_size: int = 32

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be ≥ 1")


@dataclass
class MxFP4Result:
    """Result of MXFP4 quantization.

    Attributes:
        codes: INT4 codes, shape ``(n_elements,)`` packed as uint8 (2 per byte).
        scales: Per-block FP8 scale (stored as float32 approx), shape
            ``(n_blocks,)``.
        original_shape: Shape of the original tensor.
    """

    codes: np.ndarray  # shape (n_elements,) int32 values 0..15
    scales: np.ndarray  # shape (n_blocks,) float32
    original_shape: tuple

    def dequantize(self) -> np.ndarray:
        """Reconstruct approximate FP32 values."""
        n_blocks = len(self.scales)
        bs = len(self.codes) // n_blocks
        vals = _E2M1_CODE_TABLE[self.codes] * np.repeat(self.scales, bs)
        return vals.reshape(self.original_shape)


class MxFP4:
    """OCP MXFP4 block-scaling quantizer.

    Parameters
    ----------
    config:
        MxFP4Config.
    """

    def __init__(self, config: Optional[MxFP4Config] = None) -> None:
        self._cfg = config or MxFP4Config()

    @property
    def config(self) -> MxFP4Config:
        return self._cfg

    def quantize(self, x: np.ndarray) -> MxFP4Result:
        """Quantize tensor ``x`` to MXFP4.

        Parameters
        ----------
        x:
            Input float32 tensor (any shape).

        Returns
        -------
        MxFP4Result
        """
        original_shape = x.shape
        flat = np.asarray(x, dtype=np.float32).ravel()
        bs = self._cfg.block_size
        pad = (-len(flat)) % bs
        if pad:
            flat = np.pad(flat, (0, pad))
        n = len(flat)
        n_blocks = n // bs
        blocks = flat.reshape(n_blocks, bs)

        # Per-block scale: nearest power-of-two that covers the block max abs
        abs_max = np.abs(blocks).max(axis=1)  # (n_blocks,)
        # E8M0: scale = 2^ceil(log2(abs_max / 6.0)) — ensures max fits within E2M1 range
        safe_max = np.maximum(abs_max, 1e-12)
        scales = np.exp2(np.ceil(np.log2(safe_max / 6.0))).astype(np.float32)
        scales = np.maximum(scales, 1e-12)

        # Scale blocks
        scaled = blocks / scales[:, None]  # (n_blocks, bs)
        scaled_flat = scaled.ravel()

        # Nearest neighbour in E2M1 code table (excluding code 15)
        codes = self._nearest_e2m1(scaled_flat)
        return MxFP4Result(
            codes=codes[:len(x.ravel()) if pad == 0 else n - pad],
            scales=scales,
            original_shape=original_shape,
        )

    @staticmethod
    def _nearest_e2m1(values: np.ndarray) -> np.ndarray:
        """Map each float32 value to its nearest E2M1 code (0..14)."""
        table = _E2M1_CODE_TABLE[:15]  # exclude code 15 (reserved)
        diffs = np.abs(values[:, None] - table[None, :])
        return np.argmin(diffs, axis=1).astype(np.int32)
