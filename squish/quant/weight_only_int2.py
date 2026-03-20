"""
squish/quant/weight_only_int2.py

2-bit group-wise weight-only quantization with asymmetric scaling.

Achieves 8× memory compression vs FP32 (4× vs FP16) by storing weights
as 2-bit unsigned integers with per-group scale/zero-point metadata.
Four weights are packed into each uint8 byte (pack-4 scheme).

This is a *weight-only* scheme: activations remain in full precision.
The dequantize-then-matmul pattern is used at inference time:

    weight_fp = dequantize(packed, scale, zero)   # FP32, shape (rows, cols)
    output     = activation @ weight_fp.T

On Apple Silicon with unified memory the bottleneck is DRAM bandwidth; a
4× reduction in weight bytes loaded per matmul translates directly to
faster token generation.

Key differences from existing squish quantizers
-----------------------------------------------
``int3_runtime.py``
    INT3 weight-only loader for MiLo-format model files.  Operates
    on pre-quantized .npy artefacts; no in-process calibration.
``any4.py``
    Learned 4-bit LUT (k-means codebook) quantization requiring
    calibration data to build a per-tensor codebook.  Non-uniform.
``fp8_quant.py``
    FP8 activation quantization at inference time, not weight-only.

This module provides uniform symmetric and asymmetric INT2 quantization
of weight matrices directly in Python/NumPy, with no external file
dependencies.  Suitable for on-the-fly model compression before loading.

References
----------
Chee et al. (2024). QuIP#: Even Better LLM Quantization with Hadamard
Incoherence and Lattice Codebooks. NeurIPS 2024. arXiv:2402.04396.

Egiazarian et al. (2024). Extreme Compression of Large Language Models via
Additive Quantization. ICLR 2024. arXiv:2401.06118.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Int2QuantConfig:
    """Configuration for 2-bit weight-only quantization.

    Parameters
    ----------
    group_size:
        Number of weights sharing a single scale/zero-point pair.
        Must be divisible by 4 (pack-4 constraint).  Larger groups
        compress more but reduce quantization accuracy.
    symmetric:
        When True, use symmetric quantization (zero_point = 0, range
        centred at 0).  False uses asymmetric with a learned zero_point.
    clip_threshold:
        Fraction (0.5–1.0) of the per-group value range to keep; the
        (1 - clip_threshold) * 50 %-ile from each tail is clipped before
        quantization to reduce outlier influence.  1.0 disables clipping.
    """

    group_size: int = 64
    symmetric: bool = False
    clip_threshold: float = 0.99

    def __post_init__(self) -> None:
        if self.group_size < 8:
            raise ValueError("group_size must be >= 8")
        if self.group_size % 4 != 0:
            raise ValueError(
                "group_size must be divisible by 4 (pack-4 alignment constraint)"
            )
        if not 0.5 <= self.clip_threshold <= 1.0:
            raise ValueError("clip_threshold must be in [0.5, 1.0]")


class WeightOnlyInt2Quant:
    """2-bit weight-only quantizer with group-wise asymmetric scaling.

    Quantizes a 2-D weight matrix to 2-bit integers using per-group
    scale and zero-point metadata, packs four weights per byte, and
    supports exact dequantization back to FP32 for matmul.

    Workflow
    --------
    ::

        q = WeightOnlyInt2Quant()
        packed, scale, zero = q.quantize(weight_fp32)
        # store packed (8× smaller), scale, zero alongside model layer
        ...
        weight_approx = q.dequantize(packed, scale, zero)
        output = activation @ weight_approx.T

    Pack-4 scheme
    -------------
    Each ``uint8`` byte stores four 2-bit weights in the low bits:

        byte = w0[1:0] | w1[1:0] << 2 | w2[1:0] << 4 | w3[1:0] << 6

    Dequantization (asymmetric)
    ---------------------------
    ``weight_fp = w_int * scale + zero``
    """

    N_BITS: int = 2
    N_LEVELS: int = 4       # 2 ** N_BITS
    PACK_FACTOR: int = 4    # weights per byte

    def __init__(self, config: Int2QuantConfig | None = None) -> None:
        self.config = config or Int2QuantConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self, weight: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize a weight matrix to 2-bit with group-wise metadata.

        Parameters
        ----------
        weight:
            FP32 or FP16 weight tensor, shape ``(rows, cols)``.
            ``cols`` must be divisible by ``config.group_size``.

        Returns
        -------
        packed : ndarray of uint8, shape ``(rows, cols // PACK_FACTOR)``
            Four 2-bit weights packed per byte.
        scale : ndarray of float32, shape ``(rows, n_groups)``
        zero : ndarray of float32, shape ``(rows, n_groups)``
            Zero-point per group.  All zeros when ``config.symmetric=True``.

        Raises
        ------
        ValueError
            If ``weight`` is not 2-D, or if ``cols`` is not divisible by
            ``group_size``.
        """
        weight = np.asarray(weight, dtype=np.float32)
        if weight.ndim != 2:
            raise ValueError("weight must be a 2-D tensor")
        rows, cols = weight.shape
        gs = self.config.group_size
        if cols % gs != 0:
            raise ValueError(
                f"cols ({cols}) must be divisible by group_size ({gs})"
            )

        n_groups = cols // gs
        wr = weight.reshape(rows, n_groups, gs)

        # Optional outlier clipping
        t = self.config.clip_threshold
        if t < 1.0:
            tail = (1.0 - t) * 50.0
            lo = np.percentile(wr, tail, axis=2, keepdims=True)
            hi = np.percentile(wr, 100.0 - tail, axis=2, keepdims=True)
            wr = np.clip(wr, lo, hi)

        if self.config.symmetric:
            absmax = np.max(np.abs(wr), axis=2, keepdims=True)
            absmax = np.where(absmax == 0.0, 1.0, absmax)
            half = self.N_LEVELS // 2 - 1  # = 1
            scale = (absmax / half).astype(np.float32).squeeze(2)
            zero = np.zeros_like(scale)
            w_int = np.round(wr / scale[:, :, None]).astype(np.int32)
            w_int = np.clip(w_int, -half, half)
            # Shift to unsigned [0, N_LEVELS-1]
            w_int = w_int + (self.N_LEVELS // 2)
        else:
            w_min = wr.min(axis=2, keepdims=True)
            w_max = wr.max(axis=2, keepdims=True)
            w_range = w_max - w_min
            w_range = np.where(w_range == 0.0, 1.0, w_range)
            scale = (w_range / (self.N_LEVELS - 1)).astype(np.float32).squeeze(2)
            zero = w_min.squeeze(2).astype(np.float32)
            w_int = np.round((wr - w_min) / w_range * (self.N_LEVELS - 1))
            w_int = np.clip(w_int, 0, self.N_LEVELS - 1).astype(np.int32)

        w_int_flat = w_int.reshape(rows, cols).astype(np.uint8)
        packed = self._pack(w_int_flat)
        return packed, scale, zero

    def dequantize(
        self,
        packed: np.ndarray,
        scale: np.ndarray,
        zero: np.ndarray,
    ) -> np.ndarray:
        """Dequantize packed INT2 weights back to FP32.

        Parameters
        ----------
        packed:
            uint8 array from ``quantize()``, shape ``(rows, cols // 4)``.
        scale:
            float32 scale per group, shape ``(rows, n_groups)``.
        zero:
            float32 zero-point per group, shape ``(rows, n_groups)``.

        Returns
        -------
        ndarray of float32, shape ``(rows, cols)``
        """
        w_int = self._unpack(packed)
        rows, cols = w_int.shape
        gs = self.config.group_size
        n_groups = cols // gs
        wr = w_int.reshape(rows, n_groups, gs).astype(np.float32)

        if self.config.symmetric:
            wr = wr - (self.N_LEVELS // 2)
            dequant = wr * scale[:, :, None]
        else:
            dequant = wr * scale[:, :, None] + zero[:, :, None]

        return dequant.reshape(rows, cols)

    def compression_ratio(self, original_dtype: str = "float16") -> float:
        """Theoretical compression ratio vs the given source dtype.

        Parameters
        ----------
        original_dtype:
            ``"float32"`` (32 bits) or ``"float16"`` / ``"bfloat16"`` (16 bits).
        """
        bits = {"float32": 32, "float16": 16, "bfloat16": 16}
        src = bits.get(original_dtype, 16)
        return float(src) / self.N_BITS

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _pack(self, w_int: np.ndarray) -> np.ndarray:
        """Pack a uint8 array with values in [0, 3] into 4-per-byte bytes."""
        rows, cols = w_int.shape
        assert cols % self.PACK_FACTOR == 0, "cols must be divisible by PACK_FACTOR"
        packed_cols = cols // self.PACK_FACTOR
        packed = np.zeros((rows, packed_cols), dtype=np.uint8)
        for i in range(self.PACK_FACTOR):
            packed |= (w_int[:, i :: self.PACK_FACTOR] & 0x03) << (2 * i)
        return packed

    def _unpack(self, packed: np.ndarray) -> np.ndarray:
        """Unpack 4-per-byte bytes into a uint8 array with values in [0, 3]."""
        rows, packed_cols = packed.shape
        cols = packed_cols * self.PACK_FACTOR
        w_int = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(self.PACK_FACTOR):
            w_int[:, i :: self.PACK_FACTOR] = (packed >> (2 * i)) & 0x03
        return w_int
