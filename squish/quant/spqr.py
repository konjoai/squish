"""squish/quant/spqr.py

SpQRQuantizer — Sparse-Quantized Representation for Near-Lossless LLM Weight Compression.

Reference
---------
Dettmers et al. "SpQR: A Sparse-Quantized Representation for Near-Lossless
LLM Weight Compression." NeurIPS 2023 (arXiv 2306.03078).

Algorithm
---------
SpQR identifies *outlier weight groups* (typically 1–2% of all groups) whose
elements, if quantized naively to INT3, would cause large reconstruction
errors. Those groups are kept in full-precision (FP16/FP32) in a sparse
matrix. The remaining "dense core" is quantized to 3-bit per weight with
per-group scale and zero-point.

At inference time the sparse outlier correction is added to the dense
dequantised GEMM output, recovering near-lossless quality at effectively
~2.1 bits per weight.

This module provides:

1. ``SpQRQuantizer.quantize(W)`` — quantize a weight matrix, returning
   ``SpQRResult`` containing the INT3 dense core, group metadata, and the
   FP32 sparse outlier residual.
2. ``SpQRQuantizer.dequantize(result)`` — reconstruct an approximate
   ``(rows, cols)`` FP32 weight matrix.
3. ``SpQRQuantizer.forward(x, result)`` — linear projection using SpQR weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "SpQRConfig",
    "SpQRResult",
    "SpQRQuantizer",
]

# INT3 symmetric range [-4, 3] (4 bits give [-8,7]; 3-bit unsigned here mapped as [-4,3])
_INT3_MIN = -4
_INT3_MAX = 3


@dataclass
class SpQRConfig:
    """Configuration for :class:`SpQRQuantizer`.

    Attributes:
        group_size: Number of weights per quantization group.
        outlier_threshold: Number of standard deviations above which a
            group's reconstruction error marks it as an outlier.
        outlier_fraction: Target fraction of outlier groups (overrides
            threshold if set to a positive value).
        bits: Bit-width for the dense core (default 3).
        seed: RNG seed.
    """

    group_size: int = 128
    outlier_threshold: float = 3.0
    outlier_fraction: float = 0.02
    bits: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1; got {self.group_size}")
        if self.bits < 1 or self.bits > 8:
            raise ValueError(f"bits must be in [1, 8]; got {self.bits}")
        if self.outlier_fraction < 0 or self.outlier_fraction >= 1:
            raise ValueError(f"outlier_fraction must be in [0, 1); got {self.outlier_fraction}")


@dataclass
class SpQRResult:
    """Quantized weight produced by :class:`SpQRQuantizer`.

    Attributes:
        quantized: ``(rows, cols)`` int8 array — dense INT3 core
            (values in ``[-4, 3]`` for 3-bit).
        scales: ``(n_groups,)`` float32 per-group scale.
        zeros: ``(n_groups,)`` float32 per-group zero-point.
        sparse_indices: ``(n_outlier, 2)`` int32 — (row, col) of outlier elements.
        sparse_values: ``(n_outlier,)`` float32 — FP32 residual for each outlier.
        shape: Original ``(rows, cols)`` shape.
        group_size: Quantization group size used.
    """

    quantized: np.ndarray
    scales: np.ndarray
    zeros: np.ndarray
    sparse_indices: np.ndarray
    sparse_values: np.ndarray
    shape: Tuple[int, int]
    group_size: int

    @property
    def effective_bits(self) -> float:
        """Effective bits per weight accounting for sparse outlier overhead."""
        rows, cols = self.shape
        total = rows * cols
        n_outlier = self.sparse_indices.shape[0]
        dense_bits = (total - n_outlier) * self.group_size / max(self.group_size, 1) * 3
        outlier_bits = n_outlier * 32
        return (dense_bits + outlier_bits) / max(total, 1)


class SpQRQuantizer:
    """SpQR weight quantizer.

    Example::

        cfg = SpQRConfig(group_size=16, outlier_fraction=0.05, bits=3)
        q = SpQRQuantizer(cfg)

        W = np.random.randn(64, 128).astype(np.float32)
        result = q.quantize(W)
        W_hat = q.dequantize(result)

        x = np.random.randn(8, 64).astype(np.float32)
        y = q.forward(x, result)
    """

    def __init__(self, config: Optional[SpQRConfig] = None) -> None:
        self._cfg = config or SpQRConfig()

    @property
    def config(self) -> SpQRConfig:
        return self._cfg

    def _int3_quant_group(
        self, g: np.ndarray
    ) -> Tuple[np.ndarray, float, float]:
        """Quantize a 1-D group to INT3, returning (q, scale, zero)."""
        bits = self._cfg.bits
        q_max = (1 << (bits - 1)) - 1   # e.g. 3 for 3-bit
        q_min = -(1 << (bits - 1))       # e.g. -4 for 3-bit
        w_min = float(g.min())
        w_max = float(g.max())
        rng = w_max - w_min
        scale = rng / (q_max - q_min) if rng > 1e-8 else 1.0
        zero = w_min - scale * q_min
        q = np.round((g - zero) / scale).clip(q_min, q_max).astype(np.int8)
        return q, scale, zero

    def quantize(self, W: np.ndarray) -> SpQRResult:
        """Quantize weight matrix *W*.

        1. Flatten into groups of ``group_size``.
        2. Quantize each group to ``bits``-bit with per-group scale/zero.
        3. Compute per-element reconstruction error.
        4. Mark elements whose absolute error exceeds ``outlier_fraction``
           of total elements (largest-error first) as outliers.
        5. Store outlier (row, col, value) in sparse format.

        Args:
            W: ``(rows, cols)`` float32 weight matrix.

        Returns:
            :class:`SpQRResult`.
        """
        W = np.asarray(W, dtype=np.float32)
        rows, cols = W.shape
        gs = self._cfg.group_size
        flat = W.reshape(-1)
        n_elem = flat.size
        n_groups = (n_elem + gs - 1) // gs

        pad = n_groups * gs - n_elem
        if pad:
            flat_pad = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
        else:
            flat_pad = flat

        quantized_flat = np.zeros_like(flat_pad, dtype=np.int8)
        scales = np.zeros(n_groups, dtype=np.float32)
        zeros = np.zeros(n_groups, dtype=np.float32)

        for g in range(n_groups):
            start = g * gs
            end = start + gs
            grp = flat_pad[start:end]
            q, s, z = self._int3_quant_group(grp)
            quantized_flat[start:end] = q
            scales[g] = s
            zeros[g] = z

        # Reconstruction error to find outliers
        dequant_flat = np.zeros_like(flat_pad)
        for g in range(n_groups):
            start = g * gs
            end = start + gs
            dequant_flat[start:end] = quantized_flat[start:end].astype(np.float32) * scales[g] + zeros[g]

        error = np.abs(flat_pad[:n_elem] - dequant_flat[:n_elem])
        n_outlier = max(0, int(self._cfg.outlier_fraction * n_elem))
        if n_outlier > 0:
            outlier_idx_flat = np.argpartition(error, -n_outlier)[-n_outlier:]
        else:
            outlier_idx_flat = np.array([], dtype=np.int64)

        # Convert flat indices back to (row, col)
        row_idx = outlier_idx_flat // cols
        col_idx = outlier_idx_flat % cols
        sparse_indices = np.stack([row_idx, col_idx], axis=-1).astype(np.int32)
        sparse_values = W[row_idx, col_idx] if len(row_idx) > 0 else np.array([], dtype=np.float32)

        quantized = quantized_flat[:n_elem].reshape(rows, cols)
        return SpQRResult(
            quantized=quantized,
            scales=scales,
            zeros=zeros,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values.astype(np.float32),
            shape=(rows, cols),
            group_size=gs,
        )

    def dequantize(self, result: SpQRResult) -> np.ndarray:
        """Reconstruct approximate FP32 weight from :class:`SpQRResult`.

        Returns:
            ``(rows, cols)`` float32 weight.
        """
        rows, cols = result.shape
        gs = result.group_size
        flat_q = result.quantized.reshape(-1).astype(np.float32)
        n_elem = flat_q.size
        n_groups = (n_elem + gs - 1) // gs

        dequant = np.zeros_like(flat_q)
        for g in range(n_groups):
            start = g * gs
            end = min(start + gs, n_elem)
            dequant[start:end] = flat_q[start:end] * result.scales[g] + result.zeros[g]

        W_hat = dequant.reshape(rows, cols)
        # Add sparse outlier correction
        if result.sparse_indices.shape[0] > 0:
            r = result.sparse_indices[:, 0]
            c = result.sparse_indices[:, 1]
            W_hat[r, c] = result.sparse_values
        return W_hat

    def forward(self, x: np.ndarray, result: SpQRResult) -> np.ndarray:
        """Linear projection using SpQR compressed weights.

        Args:
            x: ``(*, rows)`` input tensor.
            result: SpQR compressed weight.

        Returns:
            ``(*, cols)`` output.
        """
        W = self.dequantize(result)
        return np.asarray(x, dtype=np.float32) @ W
