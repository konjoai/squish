"""FastVPruner: training-free visual token pruning after layer 2.

Luo et al. (ACL 2024, arXiv 2403.06764) showed that the cross-attention score
of visual tokens from text queries drops sharply after layer 2 of the LLM
decoder: most patch tokens become informationally redundant and can be dropped
without accuracy penalty.  FastV computes the mean cross-attention weight that
each visual token receives across all text query heads at a configurable layer
and discards the bottom-scoring fraction.

Reference: Luo et al., "An Image is Worth 1/2 Tokens After Layer 2: Practical
Training-Free Acceleration of Vision-Language Models", ACL 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "FastVConfig",
    "FastVPruneResult",
    "FastVPruner",
]


@dataclass
class FastVConfig:
    """Configuration for :class:`FastVPruner`.

    Attributes:
        keep_ratio: Fraction of visual tokens to retain (0, 1].  Default 0.5.
        prune_layer: LLM decoder layer index at which to apply pruning.
            Layer 2 is the empirically optimal choice (Luo et al.).
        min_keep: Minimum number of visual tokens to retain regardless of ratio.
        score_aggregation: How to aggregate multi-head attention scores.
            ``"mean"`` averages across heads; ``"max"`` uses the per-token max.
    """

    keep_ratio: float = 0.5
    prune_layer: int = 2
    min_keep: int = 1
    score_aggregation: str = "mean"

    def __post_init__(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")
        if self.prune_layer < 0:
            raise ValueError(f"prune_layer must be ≥ 0, got {self.prune_layer}")
        if self.min_keep < 1:
            raise ValueError(f"min_keep must be ≥ 1, got {self.min_keep}")
        valid = {"mean", "max"}
        if self.score_aggregation not in valid:
            raise ValueError(
                f"score_aggregation must be one of {valid}, got {self.score_aggregation!r}"
            )


@dataclass
class FastVPruneResult:
    """Result of one :meth:`FastVPruner.prune` call.

    Attributes:
        kept_indices: Indices of retained visual tokens (sorted).
        pruned_indices: Indices of dropped visual tokens.
        scores: Raw attention scores for all tokens.
    """

    kept_indices: np.ndarray
    pruned_indices: np.ndarray
    scores: np.ndarray

    @property
    def keep_count(self) -> int:
        return int(self.kept_indices.size)

    @property
    def prune_count(self) -> int:
        return int(self.pruned_indices.size)

    @property
    def actual_keep_ratio(self) -> float:
        total = self.keep_count + self.prune_count
        return self.keep_count / total if total > 0 else 0.0


class FastVPruner:
    """Drop low-importance visual tokens based on text-to-visual attention.

    Usage::

        cfg = FastVConfig(keep_ratio=0.5, prune_layer=2)
        pruner = FastVPruner(cfg)
        # attn_weights: (n_text_tokens, n_visual_tokens) or
        #               (n_heads, n_text_tokens, n_visual_tokens)
        result = pruner.prune(attn_weights)
        kept_visual_tokens = visual_tokens[result.kept_indices]
    """

    def __init__(self, config: FastVConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def prune(
        self,
        attn_weights: np.ndarray,
        n_visual: Optional[int] = None,
    ) -> FastVPruneResult:
        """Select visual tokens to keep based on *attn_weights*.

        Parameters
        ----------
        attn_weights:
            Attention weights of shape:
            - ``(n_text, n_visual)`` — single head or pre-aggregated
            - ``(n_heads, n_text, n_visual)`` — multi-head
            The last dimension indexes visual tokens.
        n_visual:
            Override for the number of visual tokens when *attn_weights*
            covers more columns.  If None, inferred from the last dimension.
        """
        attn = np.asarray(attn_weights, dtype=np.float32)
        if attn.ndim == 3:
            # (n_heads, n_text, n_visual)
            if self.config.score_aggregation == "mean":
                scores = attn.mean(axis=(0, 1))  # (n_visual,)
            else:
                scores = attn.max(axis=(0, 1))
        elif attn.ndim == 2:
            # (n_text, n_visual)
            if self.config.score_aggregation == "mean":
                scores = attn.mean(axis=0)
            else:
                scores = attn.max(axis=0)
        else:
            raise ValueError(
                f"attn_weights must be 2-D or 3-D, got shape {attn.shape}"
            )

        n_vis = n_visual if n_visual is not None else scores.size
        scores = scores[:n_vis]
        n_keep = max(
            self.config.min_keep,
            int(np.ceil(n_vis * self.config.keep_ratio)),
        )
        n_keep = min(n_keep, n_vis)

        sorted_idx = np.argsort(scores)[::-1]  # highest score first
        kept = np.sort(sorted_idx[:n_keep])
        pruned = np.sort(sorted_idx[n_keep:])
        return FastVPruneResult(kept_indices=kept, pruned_indices=pruned, scores=scores)

    def apply(
        self,
        visual_tokens: np.ndarray,
        attn_weights: np.ndarray,
    ) -> Tuple[np.ndarray, FastVPruneResult]:
        """Prune *visual_tokens* and return the reduced sequence.

        Parameters
        ----------
        visual_tokens:
            Token matrix of shape ``(n_visual, hidden_dim)``.
        attn_weights:
            Attention weights as accepted by :meth:`prune`.
        """
        result = self.prune(attn_weights, n_visual=visual_tokens.shape[0])
        return visual_tokens[result.kept_indices], result

    def compression_ratio(self, n_total: int) -> float:
        """Expected ratio of tokens kept / total under config.keep_ratio."""
        n_keep = max(self.config.min_keep, int(np.ceil(n_total * self.config.keep_ratio)))
        return min(n_keep, n_total) / n_total if n_total > 0 else 0.0
