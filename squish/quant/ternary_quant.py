# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""TernaryQuant — BitNet-style ternary weight quantisation ({-1, 0, +1}).

Maps floating-point weights to the ternary alphabet {−1, 0, +1} following the
BitNet b1.58 approach.  Weights whose absolute value falls below
``zero_threshold × mean(|W|)`` are mapped to 0; all remaining weights are
mapped to +1 or −1 according to their sign.

The compressed representation is stored as int8, which makes this format
directly usable with SIMD integer instructions.  The effective bit-width of a
ternary distribution is log₂(3) ≈ 1.58 bits per weight, yielding approximately
a 20× reduction vs float32.

Reference:
    Ma et al., "The Era of 1-bit LLMs: All Large Language Models are in
    1.58 Bits", arXiv 2024.  https://arxiv.org/abs/2402.17764

Usage::

    import numpy as np
    from squish.quant.ternary_quant import TernaryConfig, TernaryQuantizer

    cfg      = TernaryConfig(zero_threshold=0.5)
    qtz      = TernaryQuantizer(cfg)
    weights  = np.random.randn(4096, 4096).astype(np.float32)

    ternary, scale = qtz.quantize(weights)   # int8 array, float scale
    recon          = qtz.dequantize(ternary, scale)

    print(qtz.stats.sparsity)                # fraction of zero weights
"""

from __future__ import annotations

__all__ = [
    "AsymmetricTernaryQuantizer",
    "AsymmetricTernaryResult",
    "TernaryConfig",
    "TernaryQuantizer",
    "TernaryStats",
]

from dataclasses import dataclass, field

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TernaryConfig:
    """Configuration for ternary weight quantisation.

    Attributes:
        zero_threshold: Relative threshold controlling the width of the dead
            zone.  A weight *w* is mapped to 0 when
            ``|w| < zero_threshold × mean(|W|)``.  Higher values produce
            sparser weight tensors at the cost of greater quantisation error.
            Must be strictly positive.
    """

    zero_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.zero_threshold <= 0.0:
            raise ValueError(
                f"zero_threshold must be strictly positive; "
                f"got {self.zero_threshold}"
            )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class TernaryStats:
    """Cumulative statistics for a :class:`TernaryQuantizer` session.

    Attributes:
        total_quantize_calls: Number of times :meth:`TernaryQuantizer.quantize`
            has been called.
        total_weights_quantized: Cumulative number of individual weight values
            processed across all quantize calls.
        total_zeros: Cumulative number of weights that were mapped to 0 (the
            dead-zone).
    """

    total_quantize_calls: int = 0
    total_weights_quantized: int = 0
    total_zeros: int = 0

    @property
    def sparsity(self) -> float:
        """Fraction of all processed weights that were mapped to zero."""
        return self.total_zeros / max(1, self.total_weights_quantized)


# ── Quantizer ─────────────────────────────────────────────────────────────────

class TernaryQuantizer:
    """BitNet-style ternary quantiser for transformer weight matrices.

    All state is accumulated in :attr:`stats` so that callers can monitor
    average sparsity across multiple layers.

    Args:
        config: :class:`TernaryConfig` controlling the dead-zone threshold.
    """

    def __init__(self, config: TernaryConfig) -> None:
        self.config = config
        self._stats = TernaryStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(
        self, weights: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Quantise a floating-point weight tensor to ternary {-1, 0, +1}.

        The absolute-mean scale ``mean(|W|)`` is returned alongside the
        ternary codes so that callers can optionally reconstruct an
        approximate float representation via :meth:`dequantize`.

        Args:
            weights: Float32 array of any shape.

        Returns:
            A 2-tuple ``(ternary_weights, scale)`` where:

            * ``ternary_weights`` — int8 array with the same shape as
              *weights*, containing only values in {-1, 0, 1}.
            * ``scale`` — float32 scalar equal to ``mean(|weights|)``;
              used for optional dequantisation.
        """
        weights = np.asarray(weights, dtype=np.float32)
        scale = float(np.mean(np.abs(weights)))

        threshold = self.config.zero_threshold * scale

        ternary = np.zeros(weights.shape, dtype=np.int8)
        ternary[weights >  threshold] =  1
        ternary[weights < -threshold] = -1

        n_zeros = int(np.sum(ternary == 0))

        self._stats.total_quantize_calls     += 1
        self._stats.total_weights_quantized  += int(weights.size)
        self._stats.total_zeros              += n_zeros

        return ternary, scale

    def dequantize(
        self, ternary_weights: np.ndarray, scale: float
    ) -> np.ndarray:
        """Reconstruct an approximate float32 tensor from ternary codes.

        The reconstruction is simply ``ternary_weights × scale``; this
        does not recover the exact original weights but provides an
        unbiased estimate useful for error analysis.

        Args:
            ternary_weights: int8 array with values in {-1, 0, 1}.
            scale: Scale factor previously returned by :meth:`quantize`.

        Returns:
            Float32 array with the same shape as *ternary_weights*.
        """
        return ternary_weights.astype(np.float32) * float(scale)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> TernaryStats:
        """Cumulative quantisation statistics for this instance."""
        return self._stats


# ── Asymmetric result ─────────────────────────────────────────────────────────


@dataclass
class AsymmetricTernaryResult:
    """Result of :meth:`AsymmetricTernaryQuantizer.quantize`.

    Attributes:
        ternary: int8 2-D array (n_rows × n_cols) with protected columns zeroed.
        preserved_fp16: float16 2-D array (n_rows × n_protected), one column per
            protected weight column.
        preserved_cols: sorted int32 1-D array of protected column indices.
        scale: float scalar = mean(|W_unprotected|); used for dequantisation.
        original_shape: original shape of the input weight tensor.
    """

    ternary:        np.ndarray
    preserved_fp16: np.ndarray
    preserved_cols: np.ndarray
    scale:          float
    original_shape: tuple


# ── Asymmetric quantizer ──────────────────────────────────────────────────────


class AsymmetricTernaryQuantizer:
    """Ternary quantiser with FP16 preservation for super-weight columns.

    Protected columns are stored at FP16 precision while all other columns
    are quantised to {-1, 0, +1}.  The scale is computed from the unprotected
    columns only, preventing super-weight outliers from inflating the scale.

    Args:
        config: :class:`TernaryConfig` controlling the dead-zone threshold.
    """

    def __init__(self, config: TernaryConfig) -> None:
        self.config = config

    def quantize(
        self,
        weights: np.ndarray,
        protected_cols: list[int] | None = None,
    ) -> AsymmetricTernaryResult:
        """Quantise *weights* with optional FP16-preserved columns.

        Args:
            weights: Float32 array; non-2-D inputs are handled via internal
                reshaping to ``(n_rows, last_dim)``.
            protected_cols: Column indices to keep at FP16.  Out-of-range,
                negative, and duplicate indices are silently dropped.  Pass
                ``None`` or ``[]`` for pure ternary output.

        Returns:
            :class:`AsymmetricTernaryResult`.
        """
        weights = np.asarray(weights, dtype=np.float32)
        original_shape = weights.shape
        # Internal 2-D view: (n_rows, n_cols)
        w2d = weights.reshape(-1, original_shape[-1]) if weights.ndim > 1 else weights.reshape(1, -1)
        n_rows, n_cols = w2d.shape

        # Sanitise protected columns: deduplicate, sort, drop out-of-range
        if protected_cols is not None and len(protected_cols) > 0:
            valid = sorted(set(int(c) for c in protected_cols if 0 <= int(c) < n_cols))
        else:
            valid = []

        # Build unprotected mask for scale computation
        unprotected_mask = np.ones(n_cols, dtype=bool)
        for c in valid:
            unprotected_mask[c] = False

        # Scale from unprotected columns only (prevents outlier inflation)
        if unprotected_mask.any():
            scale = float(np.mean(np.abs(w2d[:, unprotected_mask])))
        else:
            scale = float(np.mean(np.abs(w2d)))

        threshold = self.config.zero_threshold * scale

        # Ternary codes
        ternary = np.zeros(w2d.shape, dtype=np.int8)
        ternary[w2d >  threshold] =  1
        ternary[w2d < -threshold] = -1
        for c in valid:
            ternary[:, c] = 0   # protected columns are zeroed in ternary

        # Preserved FP16 columns
        if valid:
            preserved_fp16 = w2d[:, valid].astype(np.float16)
        else:
            preserved_fp16 = np.empty((n_rows, 0), dtype=np.float16)

        return AsymmetricTernaryResult(
            ternary=ternary,
            preserved_fp16=preserved_fp16,
            preserved_cols=np.array(valid, dtype=np.int32),
            scale=scale,
            original_shape=original_shape,
        )

    def dequantize(self, result: AsymmetricTernaryResult) -> np.ndarray:
        """Reconstruct a float32 tensor from an :class:`AsymmetricTernaryResult`.

        Ternary columns → ``ternary × scale``; protected columns → FP16 values.

        Args:
            result: Output of a previous :meth:`quantize` call.

        Returns:
            Float32 array with shape ``result.original_shape``.
        """
        recon = result.ternary.astype(np.float32) * result.scale
        # Restore protected columns from stored FP16
        for i, col in enumerate(result.preserved_cols):
            recon[:, int(col)] = result.preserved_fp16[:, i].astype(np.float32)
        return recon.reshape(result.original_shape)
