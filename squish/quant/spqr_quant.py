"""squish/quant/spqr_quant.py

SpQR — Sparse-Quantized Representation Quantizer.

SpQR separates a weight matrix into:
* **Bulk weights** — quantized uniformly to ``bits``-bit integers with
  per-row scales.
* **Sensitive outlier weights** — stored at full float32 precision in COO
  sparse format, identified by their (scaled) Hessian sensitivity.

This delivers near-lossless quality at 3- to 4-bit average precision for
large transformer weight matrices.

Reference
---------
Dettmers, T. et al. "SpQR: A Sparse-Quantized Representation for Near-
Lossless LLM Weight Compression and Accelerated Inference."
arXiv:2306.03078, Jan 2024.
"""

from __future__ import annotations

__all__ = ["SpQRConfig", "SpQRQuantizer"]

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SpQRConfig:
    """Configuration for SpQRQuantizer.

    Parameters
    ----------
    bits:
        Bit-width for bulk quantization (integer in [2, 8]).
    outlier_fraction:
        Fraction of weights retained as full-precision outliers,
        selected by largest Hessian diagonal magnitude (or largest
        absolute value when no Hessian is provided).  Must be in (0, 0.5).
    seed:
        RNG seed (reserved for stochastic rounding extensions).
    """

    bits: int = 3
    outlier_fraction: float = 0.01
    seed: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.bits <= 8:
            raise ValueError("bits must be in [2, 8]")
        if not 0.0 < self.outlier_fraction < 0.5:
            raise ValueError("outlier_fraction must be in (0, 0.5)")


# ---------------------------------------------------------------------------
# Quantizer helpers
# ---------------------------------------------------------------------------

def _uniform_quantize(
    W_row: ndarray, bits: int
) -> Tuple[ndarray, float, float]:
    """Quantize a 1-D array to ``bits``-bit unsigned integers.

    Returns
    -------
    W_q:
        Quantized array, dtype uint8.
    scale:
        Per-row scale = (max − min) / (2^bits − 1).
    zero_point:
        Per-row zero-point = min.
    """
    q_levels = (1 << bits) - 1  # e.g., 7 for 3-bit
    w_min = float(W_row.min())
    w_max = float(W_row.max())
    scale = (w_max - w_min) / q_levels if w_max > w_min else 1.0
    W_q = np.clip(
        np.round((W_row - w_min) / scale), 0, q_levels
    ).astype(np.uint8)
    return W_q, scale, w_min


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------

class SpQRQuantizer:
    """Weight quantizer using the SpQR sparse-quantized representation.

    Parameters
    ----------
    config:
        ``SpQRConfig`` instance.
    """

    def __init__(self, config: SpQRConfig | None = None) -> None:
        self.config = config or SpQRConfig()

    def quantize(
        self,
        W: ndarray,
        hessian_diag: Optional[ndarray] = None,
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray, ndarray, ndarray]:
        """Quantize a weight matrix using SpQR.

        Parameters
        ----------
        W:
            2-D float weight matrix, shape ``(rows, cols)``.
        hessian_diag:
            Optional 1-D Hessian diagonal of length ``cols`` used to identify
            sensitive outlier columns.  When ``None``, column L2 norm is used
            as a proxy.

        Returns
        -------
        W_bulk_q:
            Bulk quantized matrix (uint8), same shape as ``W``.
        outlier_rows:
            Row indices (int) of outlier entries, shape ``(n_outliers,)``.
        outlier_cols:
            Column indices (int) of outlier entries, shape ``(n_outliers,)``.
        outlier_vals:
            Float32 values of outlier entries, shape ``(n_outliers,)``.
        scales:
            Per-row scale factors, shape ``(rows,)``.
        zero_points:
            Per-row zero-points, shape ``(rows,)``.
        """
        W = np.asarray(W, dtype=np.float32)
        rows, cols = W.shape
        n_outliers = max(1, int(W.size * self.config.outlier_fraction))

        # Identify outlier positions by sensitivity
        if hessian_diag is not None:
            hd = np.asarray(hessian_diag, dtype=np.float32)
            sensitivity = (W ** 2) * hd[np.newaxis, :]  # row-broadcast
        else:
            sensitivity = np.abs(W)

        # Flat indices of top-n_outliers sensitive weights
        flat_idx = np.argpartition(sensitivity.ravel(), -n_outliers)[-n_outliers:]
        outlier_mask = np.zeros(W.shape, dtype=bool)
        outlier_mask.flat[flat_idx] = True

        outlier_rows = np.where(outlier_mask)[0].astype(np.int32)
        outlier_cols = np.where(outlier_mask)[1].astype(np.int32)
        outlier_vals = W[outlier_mask].astype(np.float32)

        # Zero out outlier positions before bulk quantization
        W_bulk = W.copy()
        W_bulk[outlier_mask] = 0.0

        # Per-row uniform quantization of bulk weights
        W_bulk_q = np.zeros((rows, cols), dtype=np.uint8)
        scales = np.zeros(rows, dtype=np.float32)
        zero_points = np.zeros(rows, dtype=np.float32)
        for r in range(rows):
            row_q, s, zp = _uniform_quantize(W_bulk[r], self.config.bits)
            W_bulk_q[r] = row_q
            scales[r] = s
            zero_points[r] = zp

        return W_bulk_q, outlier_rows, outlier_cols, outlier_vals, scales, zero_points

    def dequantize(
        self,
        W_bulk_q: ndarray,
        outlier_rows: ndarray,
        outlier_cols: ndarray,
        outlier_vals: ndarray,
        scales: ndarray,
        zero_points: ndarray,
    ) -> ndarray:
        """Reconstruct approximate float weights from SpQR representation.

        Parameters
        ----------
        W_bulk_q:
            Bulk quantized matrix, shape ``(rows, cols)``.
        outlier_rows, outlier_cols:
            COO outlier indices.
        outlier_vals:
            COO outlier float32 values.
        scales:
            Per-row scale factors.
        zero_points:
            Per-row zero-points.

        Returns
        -------
        Approximate float32 weight matrix.
        """
        rows, cols = W_bulk_q.shape
        W_out = W_bulk_q.astype(np.float32) * scales[:, np.newaxis] + zero_points[:, np.newaxis]
        W_out[outlier_rows, outlier_cols] = outlier_vals.astype(np.float32)
        return W_out

    def matmul(
        self,
        x: ndarray,
        W_bulk_q: ndarray,
        outlier_rows: ndarray,
        outlier_cols: ndarray,
        outlier_vals: ndarray,
        scales: ndarray,
        zero_points: ndarray,
    ) -> ndarray:
        """Compute ``y = x @ W_dequant.T``.

        Parameters
        ----------
        x:
            Input activations, shape ``(batch, cols)``.

        Returns
        -------
        Output activations, shape ``(batch, rows)``.
        """
        W_float = self.dequantize(
            W_bulk_q, outlier_rows, outlier_cols, outlier_vals, scales, zero_points
        )
        return np.asarray(x, dtype=np.float32) @ W_float.T

    def effective_bits(self, W_shape: Tuple[int, int]) -> float:
        """Average effective bits per weight given the outlier fraction.

        Returns
        -------
        Weighted average of int bits (bulk) and 32 bits (outliers).
        """
        f = self.config.outlier_fraction
        return (1.0 - f) * self.config.bits + f * 32.0
