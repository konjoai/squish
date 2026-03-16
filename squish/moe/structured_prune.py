# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""StructuredPrune — 2:4 (N:M) structured magnitude-based weight pruning.

For every group of *M* adjacent weights along the last axis of a weight matrix,
keeps the *N* with the largest absolute value and zeros the rest.  The 2:4
(N=2, M=4) pattern is the canonical structured-sparsity format supported by
NVIDIA Ampere (sm80+) and later sparse tensor cores, which can execute sparse
matrix multiplications at up to 2× the throughput of their dense equivalents.

The pruning is purely magnitude-based: no gradient information is required,
making it applicable as a post-training compression step.

Reference:
    Pool et al., "Channel Sparsity of Neural Networks with Hardware-Friendly
    Pruning", NeurIPS 2021. https://arxiv.org/abs/2010.13720

Usage::

    import numpy as np
    from squish.moe.structured_prune import PruneConfig, StructuredPruner

    cfg     = PruneConfig(N=2, M=4)
    pruner  = StructuredPruner(cfg)
    weights = np.random.randn(4096, 4096).astype(np.float32)

    pruned, mask = pruner.prune(weights)   # 50% of weights zeroed
    print(pruner.stats.sparsity)           # ≈ 0.50
    print(pruner.sparsity_fraction(pruned))
"""

from __future__ import annotations

__all__ = [
    "PruneConfig",
    "StructuredPruner",
    "PruneStats",
]

from dataclasses import dataclass

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class PruneConfig:
    """Configuration for N:M structured magnitude pruning.

    Attributes:
        N: Number of weights to *keep* per group.  Must satisfy
            ``1 <= N < M``.
        M: Group size (number of consecutive weights per group).  Must be
            >= 2.
    """

    N: int = 2
    M: int = 4

    def __post_init__(self) -> None:
        if self.M < 2:
            raise ValueError(
                f"M must be >= 2; got {self.M}"
            )
        if self.N < 1:
            raise ValueError(
                f"N must be >= 1; got {self.N}"
            )
        if self.N >= self.M:
            raise ValueError(
                f"N must be strictly less than M; got N={self.N}, M={self.M}"
            )


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class PruneStats:
    """Cumulative statistics for a :class:`StructuredPruner` session.

    Attributes:
        total_prune_calls: Number of times :meth:`StructuredPruner.prune`
            has been called.
        total_weights_in: Cumulative number of weight elements processed.
        total_weights_zeroed: Cumulative number of weights set to zero.
    """

    total_prune_calls: int = 0
    total_weights_in: int = 0
    total_weights_zeroed: int = 0

    @property
    def sparsity(self) -> float:
        """Fraction of all processed weights that were zeroed."""
        return self.total_weights_zeroed / max(1, self.total_weights_in)


# ── Pruner ────────────────────────────────────────────────────────────────────

class StructuredPruner:
    """N:M structured magnitude pruner.

    Operates on the last dimension of an input weight tensor, partitioning
    it into non-overlapping groups of *M* elements and retaining only the
    *N* largest by absolute value.

    Args:
        config: :class:`PruneConfig` specifying N and M.
    """

    def __init__(self, config: PruneConfig) -> None:
        self.config = config
        self._stats = PruneStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply N:M structured pruning along the last axis.

        For every group of *M* consecutive elements in the last dimension,
        the *N* elements with the largest magnitude are kept and all others
        are set to zero.

        Args:
            weights: Float32 array of shape ``(..., dim)`` where ``dim``
                must be divisible by *M*.

        Returns:
            A 2-tuple ``(pruned_weights, mask)`` where:

            * ``pruned_weights`` — float32 array of the same shape as
              *weights* with ``(M - N) / M`` of elements zeroed.
            * ``mask`` — bool array of the same shape; ``True`` where a
              weight was kept, ``False`` where it was zeroed.

        Raises:
            ValueError: If the last dimension of *weights* is not
                divisible by *M*.
        """
        cfg = self.config
        N, M = cfg.N, cfg.M
        weights = np.asarray(weights, dtype=np.float32)
        last_dim = weights.shape[-1]

        if last_dim % M != 0:
            raise ValueError(
                f"Last dimension ({last_dim}) must be divisible by M={M}; "
                f"got weights.shape={weights.shape}"
            )

        original_shape = weights.shape
        n_groups = last_dim // M

        # Flatten all leading dims into one batch dimension.
        flat = weights.reshape(-1, n_groups, M)   # (batch, n_groups, M)
        n_rows = flat.shape[0]

        # Indices of top N by magnitude in each group (unsorted within top N).
        top_n_idx = np.argsort(-np.abs(flat), axis=-1)[:, :, :N]
        # Shape: (n_rows, n_groups, N)

        # Build boolean mask over the (n_rows, n_groups, M) space.
        mask_flat = np.zeros((n_rows, n_groups, M), dtype=bool)

        # Expand batch and group indices to broadcast with top_n_idx.
        batch_idx = np.arange(n_rows).reshape(n_rows, 1, 1)     # (n_rows, 1, 1)
        group_idx = np.arange(n_groups).reshape(1, n_groups, 1)  # (1, n_groups, 1)
        batch_idx_exp = np.broadcast_to(batch_idx, (n_rows, n_groups, N))
        group_idx_exp = np.broadcast_to(group_idx, (n_rows, n_groups, N))
        mask_flat[batch_idx_exp, group_idx_exp, top_n_idx] = True

        pruned_flat = np.where(mask_flat, flat, 0.0)

        pruned  = pruned_flat.reshape(original_shape)
        mask    = mask_flat.reshape(original_shape)

        n_zeroed = int(np.sum(~mask))
        self._stats.total_prune_calls   += 1
        self._stats.total_weights_in    += int(weights.size)
        self._stats.total_weights_zeroed += n_zeroed

        return pruned, mask

    def sparsity_fraction(self, weights: np.ndarray) -> float:
        """Compute the fraction of near-zero elements in *weights*.

        An element is considered near-zero when its absolute value is less
        than ``1e-12`` times the maximum absolute value of the entire array.
        This relative threshold avoids false positives on small but non-zero
        weights while remaining robust to magnitude differences across layers.

        Args:
            weights: Float32 array of any shape.

        Returns:
            Fraction of elements that are near-zero (scalar in [0, 1]).
        """
        weights = np.asarray(weights, dtype=np.float32)
        max_abs = float(np.max(np.abs(weights)))
        if max_abs == 0.0:
            return 1.0
        near_zero = np.abs(weights) < 1e-12 * max_abs
        return float(np.mean(near_zero))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> PruneStats:
        """Cumulative pruning statistics for this instance."""
        return self._stats
