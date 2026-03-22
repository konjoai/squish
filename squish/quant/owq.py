"""OWQ quantization (arXiv 2306.05625, EMNLP 2023).

Activation-variance ranked column promotion: INT3 → INT4 for the
high-variance columns that most sensitive to rounding errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class OWQConfig:
    """Configuration for OWQ quantization."""

    bits: int = 3
    promoted_bits: int = 4
    group_size: int = 128
    promotion_fraction: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        if self.bits < 1:
            raise ValueError(f"bits must be >= 1, got {self.bits}")
        if self.promoted_bits <= self.bits:
            raise ValueError(
                f"promoted_bits ({self.promoted_bits}) must be > bits ({self.bits})"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")
        if not (0 < self.promotion_fraction < 1):
            raise ValueError(
                f"promotion_fraction must be in (0, 1), got {self.promotion_fraction}"
            )


@dataclass
class OWQResult:
    """Output of OWQ quantization."""

    quantized: np.ndarray       # (rows, cols) int8 — dense columns only
    scales: np.ndarray          # (n_groups,) float32
    zeros: np.ndarray           # (n_groups,) float32
    promoted_cols: np.ndarray   # (n_promoted,) int32 — column indices kept in FP32
    promoted_values: np.ndarray # (rows, n_promoted) float32 — original FP32 values
    shape: Tuple[int, int]
    group_size: int
    bits: int


class OWQQuantizer:
    """OWQ: activation-variance driven precision promotion.

    Columns with the highest input-activation variance are kept in
    ``promoted_bits`` precision (stored as plain FP32 originals here)
    while the remaining columns are quantized to ``bits`` bits.
    """

    def __init__(self, config: Optional[OWQConfig] = None) -> None:
        self._config = config or OWQConfig()

    @property
    def config(self) -> OWQConfig:
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quant_cols(
        self,
        W: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Uniform per-group quantization of a weight matrix.

        Parameters
        ----------
        W:
            ``(rows, cols)`` float32.  ``cols`` must be a multiple of
            ``group_size``.

        Returns
        -------
        quantized, scales, zeros
        """
        cfg = self._config
        rows, cols = W.shape
        n_col_groups = max(1, cols // cfg.group_size)

        q_min = -(1 << (cfg.bits - 1))
        q_max = (1 << (cfg.bits - 1)) - 1

        scales = np.empty(rows * n_col_groups, dtype=np.float32)
        zeros = np.empty(rows * n_col_groups, dtype=np.float32)
        quantized = np.empty((rows, cols), dtype=np.int8)

        for r in range(rows):
            for gc in range(n_col_groups):
                col_start = gc * cfg.group_size
                col_end = min(col_start + cfg.group_size, cols)
                g = W[r, col_start:col_end].astype(np.float32)

                g_min = float(g.min())
                g_max = float(g.max())
                scale = (g_max - g_min) / ((1 << cfg.bits) - 1) if g_max != g_min else 1.0
                zero = -g_min / scale

                q = np.round(g / scale + zero).astype(np.float32)
                q = np.clip(q, q_min, q_max).astype(np.int8)

                idx = r * n_col_groups + gc
                scales[idx] = scale
                zeros[idx] = zero
                quantized[r, col_start:col_end] = q

        return quantized, scales, zeros

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_activation_variance(self, activations: np.ndarray) -> np.ndarray:
        """Compute per-column variance of an activation matrix.

        Parameters
        ----------
        activations:
            ``(n_samples, cols)`` float32 activation tensor.

        Returns
        -------
        np.ndarray
            ``(cols,)`` float32 variance for each column.
        """
        a = np.asarray(activations, dtype=np.float32)
        return a.var(axis=0)

    def quantize(
        self,
        W: np.ndarray,
        activation_stats: Optional[np.ndarray] = None,
    ) -> OWQResult:
        """Quantize W with column promotion based on activation variance.

        Parameters
        ----------
        W:
            ``(rows, cols)`` float32 weight matrix.
        activation_stats:
            Optional ``(cols,)`` pre-computed per-column activation variance.
            When ``None``, column L2-norms are used as a proxy.

        Returns
        -------
        OWQResult
        """
        cfg = self._config
        W = np.asarray(W, dtype=np.float32)
        rows, cols = W.shape

        # Determine column sensitivity
        if activation_stats is not None:
            col_variance = np.asarray(activation_stats, dtype=np.float32)
            if col_variance.shape[0] != cols:
                raise ValueError(
                    f"activation_stats length {col_variance.shape[0]} != W cols {cols}"
                )
        else:
            col_variance = np.linalg.norm(W, axis=0).astype(np.float32)

        # Select top-promotion_fraction columns by variance
        n_promoted = max(1, int(np.ceil(cols * cfg.promotion_fraction)))
        promoted_col_indices = np.argsort(col_variance)[-n_promoted:].astype(np.int32)
        promoted_col_indices.sort()

        # Store original FP32 values for promoted columns
        promoted_values = W[:, promoted_col_indices].astype(np.float32)

        # Build dense-column weight for quantization (fill promoted cols with 0)
        W_dense = W.copy()
        W_dense[:, promoted_col_indices] = 0.0

        # Pad cols to multiple of group_size for grouping
        pad = (-cols) % cfg.group_size
        if pad:
            W_dense = np.pad(W_dense, ((0, 0), (0, pad)))

        quantized, scales, zeros = self._quant_cols(W_dense)
        quantized = quantized[:, :cols]

        return OWQResult(
            quantized=quantized,
            scales=scales,
            zeros=zeros,
            promoted_cols=promoted_col_indices,
            promoted_values=promoted_values,
            shape=(rows, cols),
            group_size=cfg.group_size,
            bits=cfg.bits,
        )

    def dequantize(self, result: OWQResult) -> np.ndarray:
        """Reconstruct float32 weight from OWQResult."""
        cfg = self._config
        rows, cols = result.shape
        pad = (-cols) % cfg.group_size
        padded_cols = cols + pad

        q_padded = np.pad(result.quantized, ((0, 0), (0, pad))) if pad else result.quantized

        n_col_groups = padded_cols // cfg.group_size
        W_hat = np.empty((rows, padded_cols), dtype=np.float32)
        for r in range(rows):
            for gc in range(n_col_groups):
                col_start = gc * cfg.group_size
                col_end = col_start + cfg.group_size
                idx = r * n_col_groups + gc
                W_hat[r, col_start:col_end] = (
                    q_padded[r, col_start:col_end].astype(np.float32) - result.zeros[idx]
                ) * result.scales[idx]

        W_hat = W_hat[:, :cols]
        # Restore promoted columns to their original FP32 values
        W_hat[:, result.promoted_cols] = result.promoted_values
        return W_hat

    def forward(self, x: np.ndarray, result: OWQResult) -> np.ndarray:
        """Matrix-multiply input ``x`` by the dequantized weight.

        Parameters
        ----------
        x:
            ``(*, rows)`` float32 input.
        result:
            Quantized weight produced by :meth:`quantize`.

        Returns
        -------
        np.ndarray
            ``(*, cols)`` output.
        """
        W_hat = self.dequantize(result)
        return np.tensordot(x, W_hat, axes=([-1], [0]))
