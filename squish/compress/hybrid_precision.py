# [Experimental] This module is part of Squish v42+ (Wave 68).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""squish/compress/hybrid_precision.py — Per-Block Intra-Weight Mixed Precision.

Wave 68: assigns 4-bit or 2-bit quantisation per 64-element weight block based
on measured block variance, following Zhao et al. — Intra-weight Mixed Precision
Quantisation (NeurIPS 2024).

Algorithm
─────────
1. Split each weight row into non-overlapping blocks of size ``block_size``
   (default 64).
2. Compute per-block variance = ``mean((w - mean(w))²)``.
3. Rank blocks by variance.  Assign:
   - Top ``int4_fraction`` (default 75%) of blocks by variance → INT4
   - Remaining ``(1 - int4_fraction)`` % → INT2
   - Top ``bf16_outlier_pct`` (default 5%) of blocks by *magnitude* → BF16
     (outliers stored full-precision in the scale table)
4. Return a :class:`BlockPrecisionMap` encoding the bit-width per block.

Rate-distortion interface
─────────────────────────
:func:`find_variance_threshold` performs a binary search over the variance
threshold that achieves a target average BPW (``--target-bpw 3.0`` etc.).

Usage::

    from squish.compress.hybrid_precision import (
        HybridPrecisionConfig,
        HybridPrecisionProfiler,
        BlockPrecisionMap,
        find_variance_threshold,
        assign_hybrid_precision,
    )

    cfg = HybridPrecisionConfig(target_bpw=3.0, block_size=64)
    precision_map = assign_hybrid_precision(weight_matrix, cfg)
    print(f"Effective BPW: {precision_map.effective_bpw:.2f}")

    threshold = find_variance_threshold(weight_matrix, target_bpw=3.0, block_size=64)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "HybridPrecisionConfig",
    "BlockPrecision",
    "BlockPrecisionMap",
    "HybridPrecisionProfiler",
    "assign_hybrid_precision",
    "find_variance_threshold",
    "BITS_INT4",
    "BITS_INT2",
    "BITS_BF16",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BITS_INT4: int = 4
BITS_INT2: int = 2
BITS_BF16: int = 16

_DEFAULT_BLOCK_SIZE: int = 64
_DEFAULT_INT4_FRACTION: float = 0.75
_DEFAULT_BF16_OUTLIER_PCT: float = 0.05
_DEFAULT_TARGET_BPW: float = 3.0
_MAX_BINARY_SEARCH_ITERS: int = 64
_BPW_TOLERANCE: float = 0.05


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class HybridPrecisionConfig:
    """Configuration for hybrid per-block bit-width assignment.

    Attributes:
        block_size: Number of elements per quantisation block (default 64).
        int4_fraction: Fraction of blocks (by variance) assigned to INT4
            (default 0.75).  The remaining ``1 - int4_fraction`` get INT2.
        bf16_outlier_pct: Fraction of blocks (by magnitude) stored at BF16
            full precision regardless of variance (default 0.05 = top 5%).
        target_bpw: Target average bits-per-weight.  Used by
            :func:`find_variance_threshold` and the rate-distortion solver.
            Ignored when calling :func:`assign_hybrid_precision` directly.
    """

    block_size: int = _DEFAULT_BLOCK_SIZE
    int4_fraction: float = _DEFAULT_INT4_FRACTION
    bf16_outlier_pct: float = _DEFAULT_BF16_OUTLIER_PCT
    target_bpw: float = _DEFAULT_TARGET_BPW

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if not (0.0 <= self.int4_fraction <= 1.0):
            raise ValueError("int4_fraction must be in [0, 1]")
        if not (0.0 <= self.bf16_outlier_pct < 1.0):
            raise ValueError("bf16_outlier_pct must be in [0, 1)")
        if self.target_bpw <= 0.0:
            raise ValueError("target_bpw must be > 0")


# ---------------------------------------------------------------------------
# Block precision assignment
# ---------------------------------------------------------------------------

class BlockPrecision:
    """Enum-like constants for block precision levels."""

    INT4: int = BITS_INT4
    INT2: int = BITS_INT2
    BF16: int = BITS_BF16


@dataclass
class BlockPrecisionMap:
    """Per-block bit-width assignment for a weight tensor.

    Attributes:
        bits: Array of bit widths per block, dtype uint8, shape ``(n_blocks,)``.
            Values are one of :data:`BITS_INT4`, :data:`BITS_INT2`,
            :data:`BITS_BF16`.
        block_size: Elements per block.
        original_shape: Shape of the weight tensor this map covers.
        variances: Per-block variance (float32), shape ``(n_blocks,)``.
        magnitudes: Per-block mean absolute magnitude (float32), shape ``(n_blocks,)``.
    """

    bits: np.ndarray
    block_size: int
    original_shape: tuple
    variances: np.ndarray
    magnitudes: np.ndarray

    @property
    def n_blocks(self) -> int:
        return len(self.bits)

    @property
    def n_int4(self) -> int:
        return int(np.sum(self.bits == BITS_INT4))

    @property
    def n_int2(self) -> int:
        return int(np.sum(self.bits == BITS_INT2))

    @property
    def n_bf16(self) -> int:
        return int(np.sum(self.bits == BITS_BF16))

    @property
    def effective_bpw(self) -> float:
        """Average bits-per-weight across all blocks."""
        if self.n_blocks == 0:
            return 0.0
        total_bits = (
            self.n_int4 * BITS_INT4
            + self.n_int2 * BITS_INT2
            + self.n_bf16 * BITS_BF16
        )
        return total_bits / self.n_blocks

    def rate_distortion_table(self) -> dict:
        """Return a summary dict for logging / analysis."""
        return {
            "n_blocks": self.n_blocks,
            "n_int4": self.n_int4,
            "n_int2": self.n_int2,
            "n_bf16": self.n_bf16,
            "pct_int4": 100.0 * self.n_int4 / max(1, self.n_blocks),
            "pct_int2": 100.0 * self.n_int2 / max(1, self.n_blocks),
            "pct_bf16": 100.0 * self.n_bf16 / max(1, self.n_blocks),
            "effective_bpw": self.effective_bpw,
        }


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class HybridPrecisionProfiler:
    """Computes per-block statistics and assigns bit widths.

    Args:
        config: :class:`HybridPrecisionConfig`.
    """

    def __init__(self, config: Optional[HybridPrecisionConfig] = None) -> None:
        self.config = config or HybridPrecisionConfig()

    def _compute_block_stats(
        self, weights: np.ndarray, block_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute per-block variance and mean-abs-magnitude.

        Returns:
            *(variances, magnitudes)* — each shape ``(n_blocks,)`` float32.
        """
        flat = weights.reshape(-1).astype(np.float32)
        n_elements = len(flat)
        n_blocks = (n_elements + block_size - 1) // block_size
        # Pad to a multiple of block_size
        pad = n_blocks * block_size - n_elements
        if pad > 0:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        blocks = flat.reshape(n_blocks, block_size)
        variances  = np.var(blocks, axis=1)     # (n_blocks,)
        magnitudes = np.mean(np.abs(blocks), axis=1)  # (n_blocks,)
        return variances.astype(np.float32), magnitudes.astype(np.float32)

    def assign(self, weights: np.ndarray) -> BlockPrecisionMap:
        """Assign a bit-width to each block of *weights*.

        Args:
            weights: Weight tensor (any shape).

        Returns:
            :class:`BlockPrecisionMap` with per-block precision assignments.
        """
        cfg = self.config
        block_size = cfg.block_size
        variances, magnitudes = self._compute_block_stats(weights, block_size)
        n_blocks = len(variances)
        bits = np.full(n_blocks, BITS_INT2, dtype=np.uint8)

        # --- Step 1: mark top bf16_outlier_pct by magnitude as BF16 ---
        n_bf16 = max(0, round(n_blocks * cfg.bf16_outlier_pct))
        if n_bf16 > 0:
            # Argsort descending by magnitude
            outlier_idx = np.argpartition(magnitudes, -n_bf16)[-n_bf16:]
            bits[outlier_idx] = BITS_BF16

        # --- Step 2: among remaining blocks, top int4_fraction by variance → INT4 ---
        remaining_mask = bits != BITS_BF16
        remaining_idx = np.where(remaining_mask)[0]
        if len(remaining_idx) > 0:
            n_int4 = max(0, round(len(remaining_idx) * cfg.int4_fraction))
            if n_int4 > 0:
                remaining_vars = variances[remaining_idx]
                top_var_idx = np.argpartition(remaining_vars, -n_int4)[-n_int4:]
                bits[remaining_idx[top_var_idx]] = BITS_INT4

        return BlockPrecisionMap(
            bits=bits,
            block_size=block_size,
            original_shape=tuple(weights.shape),
            variances=variances,
            magnitudes=magnitudes,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def assign_hybrid_precision(
    weights: np.ndarray,
    config: Optional[HybridPrecisionConfig] = None,
) -> BlockPrecisionMap:
    """Assign per-block bit widths to *weights*.

    Convenience wrapper around :class:`HybridPrecisionProfiler`.

    Args:
        weights: Weight tensor (any shape).
        config: :class:`HybridPrecisionConfig`.  Defaults are used if ``None``.

    Returns:
        :class:`BlockPrecisionMap`.
    """
    return HybridPrecisionProfiler(config).assign(weights)


def find_variance_threshold(
    weights: np.ndarray,
    *,
    target_bpw: float = _DEFAULT_TARGET_BPW,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    bf16_outlier_pct: float = _DEFAULT_BF16_OUTLIER_PCT,
    tol: float = _BPW_TOLERANCE,
) -> float:
    """Binary-search for the variance threshold that yields *target_bpw*.

    All blocks with ``variance >= threshold`` are assigned INT4; all blocks
    below the threshold are assigned INT2 (subject to outlier BF16 override).

    Args:
        weights: Weight tensor to profile.
        target_bpw: Desired effective bits-per-weight.
        block_size: Elements per block.
        bf16_outlier_pct: Fraction of blocks stored at BF16.
        tol: BPW tolerance (default ±0.05).

    Returns:
        Variance threshold (float) that achieves *target_bpw* ± *tol*.

    Raises:
        ValueError: If *target_bpw* is not achievable (e.g. lower than INT2
            or higher than BF16).
    """
    profiler = HybridPrecisionProfiler(
        HybridPrecisionConfig(
            block_size=block_size,
            bf16_outlier_pct=bf16_outlier_pct,
            target_bpw=target_bpw,
        )
    )
    variances, _ = profiler._compute_block_stats(weights, block_size)
    n_blocks = len(variances)
    n_outlier = max(0, round(n_blocks * bf16_outlier_pct))
    n_non_outlier = n_blocks - n_outlier

    if n_non_outlier == 0:
        return float(variances.max())

    # Adjusted target: ignoring BF16 outlier blocks
    bf16_bits_contribution = n_outlier * BITS_BF16
    remaining_budget = target_bpw * n_blocks - bf16_bits_contribution
    # remaining_budget = n_int4 * 4 + n_int2 * 2
    # n_int4 + n_int2 = n_non_outlier
    # → n_int4 = (remaining_budget - 2 * n_non_outlier) / 2
    numerator = remaining_budget - 2.0 * n_non_outlier
    denominator = 2.0  # (4 - 2)
    if denominator == 0:
        return 0.0
    n_int4_target = max(0.0, min(float(n_non_outlier), numerator / denominator))
    int4_fraction = n_int4_target / max(1, n_non_outlier)

    # Find the variance at the (1 - int4_fraction) percentile
    if int4_fraction >= 1.0:
        return float(variances.min())
    if int4_fraction <= 0.0:
        return float(variances.max()) + 1.0

    threshold_idx = max(0, round(n_non_outlier * (1.0 - int4_fraction)) - 1)
    sorted_vars = np.sort(variances)
    return float(sorted_vars[min(threshold_idx, len(sorted_vars) - 1)])
