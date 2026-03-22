"""VisionZip: context-dependent visual token compression.

Yang et al. (arXiv 2412.04467, 2024) observed that only a small subset of
visual tokens receive significant attention from the [CLS] or query tokens;
the rest carry minimal context-specific information.  VisionZip splits tokens
into *dominant* (high CLS attention) and *contextual* (low attention) sets,
retains all dominant tokens, and samples a configurable fraction of contextual
ones.

Composable with :class:`~squish.vision.fast_v.FastVPruner` for two-stage
95%+ reduction.

Reference: Yang et al., "VisionZip: Longer is Better but Not Necessary in
Vision Language Models", arXiv 2412.04467, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "VisionZipConfig",
    "VisionZipResult",
    "VisionZip",
]


@dataclass
class VisionZipConfig:
    """Configuration for :class:`VisionZip`.

    Attributes:
        dominant_ratio: Fraction of tokens classified as *dominant* by
            their attention score.  All dominant tokens are retained.
        contextual_keep_ratio: Fraction of *contextual* (low-attention)
            tokens to randomly sample and keep alongside dominant tokens.
        min_tokens: Absolute minimum total tokens to retain.
        seed: RNG seed for contextual sampling.
    """

    dominant_ratio: float = 0.1
    contextual_keep_ratio: float = 0.1
    min_tokens: int = 1
    seed: int = 0

    def __post_init__(self) -> None:
        for name, val in [
            ("dominant_ratio", self.dominant_ratio),
            ("contextual_keep_ratio", self.contextual_keep_ratio),
        ]:
            if not (0.0 < val <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {val}")
        if self.min_tokens < 1:
            raise ValueError(f"min_tokens must be ≥ 1, got {self.min_tokens}")


@dataclass
class VisionZipResult:
    """Result of one :meth:`VisionZip.compress` call.

    Attributes:
        kept_indices: Final set of retained token indices (sorted).
        dominant_indices: Subset classified as dominant.
        contextual_sampled_indices: Contextual subset that was sampled.
        scores: Full CLS-attention score array.
    """

    kept_indices: np.ndarray
    dominant_indices: np.ndarray
    contextual_sampled_indices: np.ndarray
    scores: np.ndarray

    @property
    def keep_count(self) -> int:
        return int(self.kept_indices.size)

    @property
    def compression_ratio(self) -> float:
        n_total = self.scores.size
        return self.keep_count / n_total if n_total > 0 else 0.0


class VisionZip:
    """Two-stage visual token compression: dominant selection + contextual sampling.

    Usage::

        cfg = VisionZipConfig(dominant_ratio=0.1, contextual_keep_ratio=0.1)
        vz = VisionZip(cfg)
        # cls_attn: mean attention from CLS/query token → each visual token
        result = vz.compress(cls_attn)
        kept_tokens = visual_tokens[result.kept_indices]
    """

    def __init__(self, config: VisionZipConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def compress(
        self,
        cls_attn: np.ndarray,
    ) -> VisionZipResult:
        """Compute kept token indices from CLS/query attention vector.

        Parameters
        ----------
        cls_attn:
            1-D array of shape ``(n_visual,)`` representing the attention
            weight each visual token receives from the CLS or query token.
        """
        scores = np.asarray(cls_attn, dtype=np.float32).ravel()
        n = scores.size

        n_dominant = max(1, int(np.ceil(n * self.config.dominant_ratio)))
        sorted_desc = np.argsort(scores)[::-1]
        dominant_idx = sorted_desc[:n_dominant]
        contextual_idx = sorted_desc[n_dominant:]

        n_ctx_keep = max(0, int(np.ceil(contextual_idx.size * self.config.contextual_keep_ratio)))
        if n_ctx_keep > 0 and contextual_idx.size > 0:
            sampled_ctx = self._rng.choice(contextual_idx, size=min(n_ctx_keep, contextual_idx.size), replace=False)
        else:
            sampled_ctx = np.empty(0, dtype=np.int64)

        kept = np.union1d(dominant_idx, sampled_ctx)
        # Enforce minimum
        if kept.size < self.config.min_tokens:
            # Pull in highest-score tokens until minimum met
            extra = sorted_desc[: self.config.min_tokens]
            kept = np.union1d(kept, extra)

        return VisionZipResult(
            kept_indices=np.sort(kept),
            dominant_indices=np.sort(dominant_idx),
            contextual_sampled_indices=np.sort(sampled_ctx),
            scores=scores,
        )

    def apply(
        self,
        visual_tokens: np.ndarray,
        cls_attn: np.ndarray,
    ) -> Tuple[np.ndarray, VisionZipResult]:
        """Filter *visual_tokens* and return the compressed sequence."""
        result = self.compress(cls_attn)
        return visual_tokens[result.kept_indices], result
