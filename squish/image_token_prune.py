"""squish/image_token_prune.py

ImageTokenPrune — Entropy-based saliency pruning of image patch tokens for
efficient vision-language model inference.

Standard ViT encoders produce one patch token per image region (e.g., 196
tokens for a 14 × 14 grid over a 224 × 224 image).  Most of these tokens
attend in a strongly peaked, low-entropy pattern — a few reference tokens
receive the bulk of attention mass, while the remainder distribute attention
nearly uniformly, contributing little discriminative information.  Retaining
all tokens wastes KV cache capacity and increases attention FLOPs quadratically
with sequence length when text and vision tokens are concatenated.

ImageTokenPrune assigns each image patch a saliency score defined as the
*negative entropy* of its outgoing attention distribution averaged over heads:
``saliency(i) = mean_h Σ_j p_{h,i,j} · log(p_{h,i,j} + ε)``.  Negative
entropy is maximised when the distribution is a point mass (one token attends
with full confidence) and minimised when the distribution is uniform (diffuse,
low-information attention).  Tokens are ranked by this score and the bottom
``prune_ratio`` fraction — those with the most diffuse, least-confident
attention — are discarded.  The remaining tokens are returned as two sorted
index arrays: kept and pruned positions in ascending order.

The pruner is stateless between calls; saliency is recomputed from scratch on
each invocation.  This design is intentional: attention patterns shift
throughout generation, so caching saliency scores across decode steps would
introduce staleness artefacts.

Example usage::

    import numpy as np
    from squish.image_token_prune import PruneConfig, ImageTokenPruner

    cfg    = PruneConfig(n_tokens=196, prune_ratio=0.5, n_heads=8)
    pruner = ImageTokenPruner(cfg)

    # Simulate attention weights for 196 image tokens across 8 heads.
    raw    = np.random.rand(8, 196, 196).astype(np.float32)
    attn   = raw / raw.sum(axis=2, keepdims=True)   # row-normalise
    kept, pruned = pruner.prune(attn)
    print(f"kept={len(kept)}, pruned={len(pruned)}")
    print(pruner.stats)
"""

from __future__ import annotations

__all__ = ["PruneConfig", "ImageTokenPruner", "PruneStats"]

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PruneConfig:
    """Configuration for entropy-based image token pruning.

    Attributes:
        n_tokens:    Total number of image patch tokens expected per call.
        prune_ratio: Fraction of tokens to discard.  Must be in ``[0, 1)``.
        n_heads:     Number of attention heads in the weight matrix.
        eps:         Small constant added inside ``log`` for numerical
                     stability.  Must be positive.
    """

    n_tokens: int = 196
    prune_ratio: float = 0.5
    n_heads: int = 8
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.n_tokens < 1:
            raise ValueError(
                f"n_tokens must be >= 1, got {self.n_tokens}"
            )
        if not (0.0 <= self.prune_ratio < 1.0):
            raise ValueError(
                f"prune_ratio must be in [0, 1), got {self.prune_ratio}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {self.eps}")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class PruneStats:
    """Aggregate statistics for an :class:`ImageTokenPruner`.

    Attributes:
        total_prune_calls:   Total number of :meth:`~ImageTokenPruner.prune`
                             invocations.
        total_tokens_pruned: Cumulative count of discarded tokens across all
                             calls.
    """

    total_prune_calls: int = 0
    total_tokens_pruned: int = 0


# ---------------------------------------------------------------------------
# Pruner
# ---------------------------------------------------------------------------


class ImageTokenPruner:
    """Prunes image patch tokens by negative-entropy saliency scoring.

    Higher negative-entropy (more peaked attention distribution) indicates a
    more salient, information-dense token.  The bottom ``prune_ratio`` fraction
    of tokens by saliency — those with the most diffuse attention — are
    removed.

    Args:
        config: A :class:`PruneConfig` instance.
    """

    def __init__(self, config: PruneConfig) -> None:
        self._cfg = config
        self._total_prune_calls:   int = 0
        self._total_tokens_pruned: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prune(
        self,
        attn_weights: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Score and partition image tokens into kept and pruned index sets.

        Args:
            attn_weights: Attention weight matrix of shape
                          ``(n_heads, n_tokens, n_tokens)``.  Each row
                          ``attn_weights[h, i, :]`` should sum to 1 (i.e.,
                          represent a valid probability distribution).

        Returns:
            A ``(kept_indices, pruned_indices)`` tuple of 1-D int64 arrays,
            both sorted in ascending order.  Together they partition
            ``range(n_tokens)``.

        Raises:
            ValueError: If *attn_weights* does not have shape
                        ``(n_heads, n_tokens, n_tokens)``.
        """
        expected = (self._cfg.n_heads, self._cfg.n_tokens, self._cfg.n_tokens)
        if attn_weights.shape != expected:
            raise ValueError(
                f"attn_weights must have shape {expected}, "
                f"got {attn_weights.shape}."
            )

        n_tokens = self._cfg.n_tokens

        # Negative entropy per head per token: Σ_j p·log(p+ε)
        # Shape: (n_heads, n_tokens)
        log_p = np.log(attn_weights + self._cfg.eps)
        neg_entropy_per_head = np.sum(attn_weights * log_p, axis=2)

        # Mean negative entropy over heads → saliency score per token.
        # Higher = more peaked = more salient.
        saliency: np.ndarray = np.mean(neg_entropy_per_head, axis=0)  # (n_tokens,)

        n_prune = int(round(self._cfg.prune_ratio * n_tokens))
        n_keep  = n_tokens - n_prune

        if n_keep >= n_tokens:
            kept_indices   = np.arange(n_tokens, dtype=np.int64)
            pruned_indices = np.empty(0, dtype=np.int64)
        else:
            # Indices of the top n_keep tokens by saliency (highest scores).
            top_idx = np.argpartition(-saliency, n_keep - 1)[:n_keep]
            kept_indices = np.sort(top_idx.astype(np.int64))

            # Pruned = complement of kept.
            mask = np.ones(n_tokens, dtype=bool)
            mask[kept_indices] = False
            pruned_indices = np.where(mask)[0].astype(np.int64)

        self._total_prune_calls   += 1
        self._total_tokens_pruned += len(pruned_indices)

        return kept_indices, pruned_indices

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> PruneStats:
        """Return a snapshot of cumulative pruning statistics."""
        return PruneStats(
            total_prune_calls=self._total_prune_calls,
            total_tokens_pruned=self._total_tokens_pruned,
        )
