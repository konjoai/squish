"""squish/model/layer_collapse.py

LayerCollapse — Efficient Depth Reduction via Cosine Similarity Pruning.

Reference
---------
Gromov et al. "The Unreasonable Inefficiency of Layer Removal in
Transformer Models." ICML 2025 (arXiv:2403.03853).

Algorithm
---------
Many transformer layers produce outputs that are highly similar to those
of their immediate neighbours.  Layers whose output cosine similarity
exceeds a threshold can be removed with minimal quality degradation:

1. Run a small calibration set of prompts through the model.
2. For each consecutive layer pair (l, l+1), compute:
   ``cos_sim = mean(cos(h_l, h_{l+1}))`` over all calibration tokens.
3. Rank pairs by cosine similarity.  Pairs above ``threshold`` are
   candidates for collapse.
4. Remove (collapse) up to ``max_prune_fraction`` of layers, starting
   from the most similar pair.

This module provides the calibration and collapse-mask logic without
modifying actual model weights, making it usable as an offline analyser
or as a prefix to a runtime layer-skipping pass.

Key properties
--------------
* NumPy-only.
* ``similarity_threshold`` — cosine similarity above which a layer is
  collapsible (default 0.97).
* ``max_prune_fraction`` — hard cap on fraction of layers to remove
  (default 0.40).
* ``n_layers`` — number of transformer layers in the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "LayerCollapseConfig",
    "CollapseSchedule",
    "LayerCollapse",
]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class LayerCollapseConfig:
    """Configuration for :class:`LayerCollapse`.

    Attributes:
        n_layers: Number of transformer layers.
        hidden_size: Hidden state dimension.
        similarity_threshold: Cosine similarity above which a layer pair
            is considered collapsible.
        max_prune_fraction: Maximum fraction of layers to remove.
    """

    n_layers: int = 32
    hidden_size: int = 4096
    similarity_threshold: float = 0.97
    max_prune_fraction: float = 0.40

    def __post_init__(self) -> None:
        if not 0.0 < self.similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if not 0.0 <= self.max_prune_fraction < 1.0:
            raise ValueError("max_prune_fraction must be in [0, 1)")


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class CollapseSchedule:
    """The set of layers to keep after collapse.

    Attributes:
        keep_mask: Boolean array of shape ``(n_layers,)``; True = keep.
        similarity_scores: Per-layer-pair cosine similarity scores.
        n_layers_removed: Number of layers that were removed.
    """

    keep_mask: np.ndarray
    similarity_scores: np.ndarray
    n_layers_removed: int

    @property
    def layers_kept(self) -> List[int]:
        return [i for i, keep in enumerate(self.keep_mask) if keep]

    @property
    def compression_ratio(self) -> float:
        return self.n_layers_removed / len(self.keep_mask)


# ── Module ────────────────────────────────────────────────────────────────────


class LayerCollapse:
    """Layer depth reduction via cosine-similarity calibration.

    Parameters
    ----------
    config:
        Layer collapse configuration.
    """

    def __init__(self, config: Optional[LayerCollapseConfig] = None) -> None:
        self._cfg = config or LayerCollapseConfig()
        # Accumulator: per adjacent-pair cosine similarity sums and counts
        self._sim_sum: np.ndarray = np.zeros(self._cfg.n_layers - 1, dtype=np.float64)
        self._sim_count: np.ndarray = np.zeros(self._cfg.n_layers - 1, dtype=np.int64)
        self._schedule: Optional[CollapseSchedule] = None

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def config(self) -> LayerCollapseConfig:
        return self._cfg

    @property
    def schedule(self) -> Optional[CollapseSchedule]:
        return self._schedule

    @property
    def calibration_sample_count(self) -> int:
        return int(self._sim_count.min()) if self._sim_count.any() else 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Per-token mean cosine similarity between two hidden-state sequences."""
        a = a.reshape(-1, self._cfg.hidden_size).astype(np.float64)
        b = b.reshape(-1, self._cfg.hidden_size).astype(np.float64)
        norm_a = np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-8)
        norm_b = np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-8)
        return float(((a / norm_a) * (b / norm_b)).sum(axis=1).mean())

    def calibrate(self, layer_hidden_states: List[np.ndarray]) -> None:
        """Update cosine similarity estimates from one forward pass.

        Parameters
        ----------
        layer_hidden_states:
            List of ``n_layers`` hidden-state tensors, each of shape
            ``(seq_len, hidden_dim)`` or ``(batch, seq_len, hidden_dim)``.
        """
        if len(layer_hidden_states) != self._cfg.n_layers:
            raise ValueError(
                f"Expected {self._cfg.n_layers} hidden states, "
                f"got {len(layer_hidden_states)}"
            )
        for i in range(self._cfg.n_layers - 1):
            sim = self._cosine_similarity(
                layer_hidden_states[i], layer_hidden_states[i + 1]
            )
            self._sim_sum[i] += sim
            self._sim_count[i] += 1
        self._schedule = None  # invalidate cached schedule

    def mean_similarities(self) -> np.ndarray:
        """Return the mean cosine similarity for each adjacent layer pair.

        Returns
        -------
        np.ndarray
            Shape ``(n_layers - 1,)``; NaN where no calibration data exists.
        """
        with np.errstate(invalid="ignore"):
            return np.where(
                self._sim_count > 0,
                self._sim_sum / self._sim_count,
                np.nan,
            )

    def compute_schedule(self) -> CollapseSchedule:
        """Compute which layers to keep.

        Returns
        -------
        CollapseSchedule
            Keep mask and statistics.
        """
        sims = self.mean_similarities()
        # Fill NaN with 0 (those pairs can't be pruned without calibration data)
        sims_filled = np.nan_to_num(sims, nan=0.0)
        n_layers = self._cfg.n_layers
        max_remove = int(n_layers * self._cfg.max_prune_fraction)

        # Greedily remove highest-similarity layers (later layer of each pair)
        keep_mask = np.ones(n_layers, dtype=bool)
        removed = 0

        # Sort pairs by similarity descending
        pair_order = np.argsort(sims_filled)[::-1]
        for pair_idx in pair_order:
            if removed >= max_remove:
                break
            if sims_filled[pair_idx] < self._cfg.similarity_threshold:
                break
            # Remove layer pair_idx+1 (the "later" layer)
            layer_to_remove = pair_idx + 1
            if keep_mask[layer_to_remove]:
                keep_mask[layer_to_remove] = False
                removed += 1

        self._schedule = CollapseSchedule(
            keep_mask=keep_mask,
            similarity_scores=sims_filled,
            n_layers_removed=removed,
        )
        return self._schedule

    def should_skip(self, layer_idx: int) -> bool:
        """Return True if the given layer should be skipped.

        Parameters
        ----------
        layer_idx:
            Zero-based layer index.

        Returns
        -------
        bool
            True if this layer is collapsed and should be skipped.
        """
        if self._schedule is None:
            self.compute_schedule()
        return not bool(self._schedule.keep_mask[layer_idx])

    def reset_calibration(self) -> None:
        """Clear accumulated calibration data."""
        self._sim_sum[:] = 0.0
        self._sim_count[:] = 0
        self._schedule = None
