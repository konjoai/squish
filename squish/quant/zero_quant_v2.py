# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""ZeroQuantV2 — Groupwise weight quantisation with FP16 outlier residuals.

ZeroQuant-V2 quantises weights in groups of ``group_size`` elements using
local symmetric INT8 (or INT4) scales.  Weights whose absolute value exceeds
``outlier_threshold * group_max`` are considered *outliers* and are preserved
in full FP32 precision as a residual tensor, bypassing quantisation error on
extreme values that would otherwise dominate reconstruction quality.

Concretely, for each group ``g``:

1. Identify outlier mask: ``|w| > threshold * max(|w|)``.
2. Residual: ``r = w * outlier_mask`` (FP32).
3. Quantise ``w - r`` with symmetric INT8:
   ``scale = max(|w - r|) / 127``; ``q = round((w - r) / scale)``.
4. Dequantise: ``recon = q * scale + r``.

Reference:
    Yao et al., "ZeroQuant-V2: Exploring Post-Training Quantization in LLMs
    from Comprehensive Study to Low Rank Compensation", arXiv 2023.
    https://arxiv.org/abs/2303.08302

Usage::

    import numpy as np
    from squish.quant.zero_quant_v2 import ZQConfig, ZeroQuantV2, ZQStats

    rng     = np.random.default_rng(0)
    weights = rng.standard_normal((64, 256)).astype(np.float32)

    cfg  = ZQConfig(n_bits=8, group_size=128, outlier_threshold=0.95)
    zq   = ZeroQuantV2(cfg)

    q_int, scales, residual = zq.quantize(weights)
    recon = zq.dequantize(q_int, scales, residual)
    print(zq.stats)
"""

from __future__ import annotations

__all__ = ["ZQConfig", "ZeroQuantV2", "ZQStats"]

from dataclasses import dataclass

import numpy as np


@dataclass
class ZQConfig:
    """Configuration for ZeroQuant-V2 quantisation.

    Attributes:
        n_bits: Quantisation width; must be 4 or 8.
        group_size: Number of columns per quantisation group.  The column
            dimension of the weight matrix must be divisible by ``group_size``.
        outlier_threshold: Fraction of the per-group maximum above which a
            weight element is treated as an outlier and stored in FP32.
    """

    n_bits: int = 8
    group_size: int = 128
    outlier_threshold: float = 0.95

    def __post_init__(self) -> None:
        if self.n_bits not in (4, 8):
            raise ValueError(
                f"n_bits must be 4 or 8; got {self.n_bits}"
            )
        if self.group_size < 8:
            raise ValueError(
                f"group_size must be >= 8; got {self.group_size}"
            )
        if not (0.0 < self.outlier_threshold < 1.0):
            raise ValueError(
                f"outlier_threshold must be in (0, 1); got {self.outlier_threshold}"
            )

    # ------------------------------------------------------------------
    # Derived quantisation constants
    # ------------------------------------------------------------------

    @property
    def q_min(self) -> int:
        """Minimum signed integer value for the configured bit-width."""
        return -(1 << (self.n_bits - 1))

    @property
    def q_max(self) -> int:
        """Maximum signed integer value for the configured bit-width."""
        return (1 << (self.n_bits - 1)) - 1


@dataclass
class ZQStats:
    """Running statistics for a :class:`ZeroQuantV2` session.

    Attributes:
        total_quantize_calls: Number of :meth:`ZeroQuantV2.quantize`
            invocations.
        total_outlier_elements: Cumulative number of outlier weight elements
            routed to the FP32 residual.
        total_elements: Cumulative total number of weight elements processed.
    """

    total_quantize_calls:  int = 0
    total_outlier_elements: int = 0
    total_elements:         int = 0

    @property
    def outlier_rate(self) -> float:
        """Fraction of all processed elements that were outliers."""
        return self.total_outlier_elements / max(1, self.total_elements)


class ZeroQuantV2:
    """ZeroQuant-V2 groupwise weight quantiser with FP16 outlier residuals.

    Args:
        config: :class:`ZQConfig` specifying bit-width, group size, and
            outlier threshold.
    """

    def __init__(self, config: ZQConfig) -> None:
        self._config = config
        self._stats  = ZQStats()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def quantize(
        self,
        weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantise a 2-D weight matrix using ZeroQuant-V2.

        Args:
            weights: Float32 array of shape ``(rows, cols)`` where ``cols``
                is divisible by ``config.group_size``.

        Returns:
            Tuple ``(quantized_int, scales, residual)`` where:

            * ``quantized_int`` — INT8 array, shape ``(rows, cols)``.
            * ``scales``        — float32 per-group scales,
              shape ``(rows, cols // group_size)``.
            * ``residual``      — float32 outlier residuals,
              shape ``(rows, cols)``; zero for non-outlier elements.

        Raises:
            ValueError: If ``weights`` is not 2-D or ``cols`` is not divisible
                by ``group_size``.
        """
        weights = np.asarray(weights, dtype=np.float32)
        if weights.ndim != 2:
            raise ValueError(
                f"weights must be 2-D; got shape {weights.shape}"
            )
        rows, cols = weights.shape
        gs = self._config.group_size
        if cols % gs != 0:
            raise ValueError(
                f"cols ({cols}) must be divisible by group_size ({gs})"
            )

        n_groups   = cols // gs
        q_min      = self._config.q_min
        q_max      = self._config.q_max
        threshold  = self._config.outlier_threshold

        quantized = np.zeros_like(weights, dtype=np.int8)
        scales    = np.zeros((rows, n_groups), dtype=np.float32)
        residual  = np.zeros_like(weights, dtype=np.float32)

        total_outliers = 0

        for g in range(n_groups):
            col_start = g * gs
            col_end   = col_start + gs
            w_group   = weights[:, col_start:col_end]  # (rows, gs)

            group_max = np.max(np.abs(w_group), axis=1, keepdims=True)  # (rows, 1)

            # Outlier mask: (rows, gs) bool
            outlier_mask = np.abs(w_group) > (threshold * group_max)
            total_outliers += int(np.sum(outlier_mask))

            r_group = np.where(outlier_mask, w_group, 0.0).astype(np.float32)
            residual[:, col_start:col_end] = r_group

            # Quantise (w - residual) symmetrically
            w_residual_free = w_group - r_group  # (rows, gs)
            per_row_max = np.max(np.abs(w_residual_free), axis=1)  # (rows,)
            # Avoid division by zero for all-zero rows
            safe_max = np.where(per_row_max == 0.0, 1.0, per_row_max)
            scale = safe_max / float(q_max)  # (rows,)
            scales[:, g] = scale.astype(np.float32)

            q_group = np.round(w_residual_free / safe_max[:, np.newaxis] * q_max)
            q_group = np.clip(q_group, q_min, q_max).astype(np.int8)
            quantized[:, col_start:col_end] = q_group

        self._stats.total_quantize_calls   += 1
        self._stats.total_outlier_elements += total_outliers
        self._stats.total_elements         += weights.size

        return quantized, scales, residual

    def dequantize(
        self,
        quantized: np.ndarray,
        scales: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Reconstruct float32 weights from the quantised representation.

        Args:
            quantized: INT8 array, shape ``(rows, cols)``.
            scales:    Float32 per-group scales, shape ``(rows, n_groups)``.
            residual:  Float32 outlier residuals, shape ``(rows, cols)``.

        Returns:
            Reconstructed float32 weight matrix of shape ``(rows, cols)``.
        """
        quantized = np.asarray(quantized, dtype=np.float32)
        residual  = np.asarray(residual,  dtype=np.float32)
        scales    = np.asarray(scales,    dtype=np.float32)

        rows, cols = quantized.shape
        gs       = self._config.group_size
        n_groups = scales.shape[1]

        recon = np.empty((rows, cols), dtype=np.float32)
        for g in range(n_groups):
            col_start = g * gs
            col_end   = col_start + gs
            scale_bc  = scales[:, g:g + 1]  # (rows, 1)
            recon[:, col_start:col_end] = (
                quantized[:, col_start:col_end] * scale_bc
                + residual[:, col_start:col_end]
            )

        return recon

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> ZQStats:
        """Running quantisation statistics."""
        return self._stats
