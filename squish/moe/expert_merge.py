"""squish/moe/expert_merge.py

ExpertMerger — Cosine-similarity-based expert consolidation.

Iteratively merges the most-similar expert weight pairs (by cosine similarity
of their flattened weight matrices) until the target compression ratio is
reached.  Each merge replaces two experts with their renormalised mean.

This is the compression phase of model-merging approaches described in:
  Goddard et al., "ARCEE's MergeKit." arXiv:2403.13257, 2024.
  (See also MoE-specific variants in the DeepSeek/Qwen distillation literature.)
"""

from __future__ import annotations

__all__ = ["ExpertMergeConfig", "ExpertMerger"]

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from numpy import ndarray


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ExpertMergeConfig:
    """Configuration for ExpertMerger.

    Parameters
    ----------
    similarity_threshold:
        Minimum cosine similarity required to consider merging two experts.
        Set to 0.0 to always merge until target_ratio is reached.
    target_ratio:
        Target fraction of original experts to retain after merging.
        E.g. 0.7 keeps 70 % of experts.
    seed:
        RNG seed (reserved for future tie-breaking).
    """

    similarity_threshold: float = 0.9
    target_ratio: float = 0.7
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0, 1]")
        if not (0.0 < self.target_ratio <= 1.0):
            raise ValueError("target_ratio must be in (0, 1]")


# ---------------------------------------------------------------------------
# ExpertMerger
# ---------------------------------------------------------------------------

class ExpertMerger:
    """Merge similar experts by pairwise cosine similarity.

    Parameters
    ----------
    config:
        ``ExpertMergeConfig`` instance.
    """

    def __init__(self, config: ExpertMergeConfig) -> None:
        self.config = config

    def merge(
        self, expert_weights: List[ndarray]
    ) -> Tuple[List[ndarray], Dict[int, int]]:
        """Merge experts until ``target_ratio`` is reached.

        Parameters
        ----------
        expert_weights:
            List of N weight matrices.  Each matrix is flattened for
            similarity computation; shapes may differ but must be consistent
            within the list.

        Returns
        -------
        merged_weights:
            Reduced list of expert weights after merging.
        merge_map:
            Dict mapping eliminated expert index → surviving expert index
            (in terms of *original* indices).
        """
        if len(expert_weights) == 0:
            return [], {}

        # Work on a mutable copy; track which original index each slot holds
        weights = [w.copy().astype(np.float32) for w in expert_weights]
        # canonical_id[i] = original expert index that slot i represents
        canonical_id: List[int] = list(range(len(weights)))
        merge_map: Dict[int, int] = {}

        n_original = len(weights)
        target_n = max(1, round(n_original * self.config.target_ratio))

        while len(weights) > target_n:
            sim = self.similarity_matrix(weights)
            # Zero diagonal so we don't pick self-similarity
            np.fill_diagonal(sim, -1.0)
            best_flat = int(np.argmax(sim))
            i, j = divmod(best_flat, len(weights))
            if sim[i, j] < self.config.similarity_threshold:
                break  # no pair above threshold; stop early
            # Merge: replace slot i with renormalised mean, remove slot j
            merged = (weights[i] + weights[j]) / 2.0
            norm = np.linalg.norm(merged)
            if norm > 0.0:
                merged = merged / norm * (
                    (np.linalg.norm(weights[i]) + np.linalg.norm(weights[j])) / 2.0
                )
            weights[i] = merged
            # Record that j's original index maps to i's original index
            merge_map[canonical_id[j]] = canonical_id[i]
            weights.pop(j)
            canonical_id.pop(j)

        return weights, merge_map

    def similarity_matrix(self, weights: List[ndarray]) -> ndarray:
        """Compute pairwise cosine similarity matrix.

        Parameters
        ----------
        weights:
            List of N weight matrices (arbitrary shape; flattened internally).

        Returns
        -------
        sim:
            Shape ``(N, N)`` symmetric matrix with values in ``[-1, 1]``.
        """
        if len(weights) == 0:
            return np.empty((0, 0), dtype=np.float32)

        # Flatten and normalise each weight vector
        vecs = np.stack([w.ravel().astype(np.float32) for w in weights])  # (N, D)
        norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        normed = vecs / norms  # (N, D)
        sim = normed @ normed.T  # (N, N)
        return sim.astype(np.float32)

    @staticmethod
    def compression_ratio(original_n: int, merged_n: int) -> float:
        """Fraction of experts retained after merging.

        Parameters
        ----------
        original_n:
            Number of experts before merging.
        merged_n:
            Number of experts after merging.

        Returns
        -------
        ratio in ``[0, 1]``.
        """
        if original_n <= 0:
            raise ValueError("original_n must be > 0")
        return merged_n / original_n
