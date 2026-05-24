"""
squish/kv/head_importance.py — Per-head importance scoring (P2 / FastGen-style).

Not every attention head is equally important.  Empirically (FastGen, Ge et al.
2023; ScissorHands, Liu et al. 2023; The Heads Hypothesis, Voita et al. 2019),
a substantial fraction of heads carry near-uniform or near-redundant attention
distributions and can be dropped or aggressively compressed with negligible
accuracy impact.

This module provides a **gradient-free** importance estimator based on the
empirical activation statistics of K (and optionally V) cache tensors:

  importance(h, l) = α · variance_score + β · concentration_score
                                              + γ · magnitude_score

where each term is computed per (layer, head) over a calibration sample:

  variance_score      = trace(Cov(K_lh))                 — total channel energy
  concentration_score = max_t ||K_lh[t]||_2 / mean_t ||K_lh[t]||_2
                                                          — outlier dominance
  magnitude_score     = mean_t ||K_lh[t]||_2             — average L2

All three rise when a head carries informative, anisotropic signal; all three
fall toward zero when a head produces near-uniform, low-energy activations
(those are the "streaming" / prunable heads in the FastGen taxonomy).

The scores are min-max normalised per layer before combining, so the final
importance is a unitless ∈ [0, 1] number, directly thresholdable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


_VALID_MIX = ("variance", "concentration", "magnitude")


@dataclass(frozen=True)
class HeadImportanceScores:
    """Per-(layer, head) importance scores.

    Attributes
    ----------
    per_layer : (n_layers, n_heads) float32 — importance ∈ [0, 1].
    n_layers  : int
    n_heads   : int
    weights   : dict[str, float] — relative weight of each scoring term.
    """
    per_layer: np.ndarray
    n_layers: int
    n_heads: int
    weights: dict = field(default_factory=dict)

    def head_mask(self, threshold: float) -> np.ndarray:
        """Boolean mask: True where importance > threshold (i.e. keep)."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be ∈ [0, 1], got {threshold}")
        return self.per_layer > threshold

    def pruned_count(self, threshold: float) -> int:
        """Number of heads that would be pruned at this threshold."""
        return int((self.per_layer <= threshold).sum())

    def top_k_per_layer(self, k: int) -> np.ndarray:
        """Boolean mask: True for the top-k most-important heads per layer."""
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if k >= self.n_heads:
            return np.ones_like(self.per_layer, dtype=bool)
        # argsort descending; keep top-k indices
        keep = np.zeros_like(self.per_layer, dtype=bool)
        order = np.argsort(-self.per_layer, axis=-1)         # (n_layers, n_heads)
        for li in range(self.n_layers):
            keep[li, order[li, :k]] = True
        return keep

    def to_json(self) -> dict:
        """Plain-Python representation suitable for JSON encoding."""
        return {
            "n_layers": self.n_layers,
            "n_heads":  self.n_heads,
            "weights":  dict(self.weights),
            "per_layer": [
                [float(x) for x in row] for row in self.per_layer
            ],
        }


def _normalise_per_layer(arr: np.ndarray) -> np.ndarray:
    """Min-max scale each row (layer) to [0, 1].  Constant rows → 0.5."""
    lo = arr.min(axis=-1, keepdims=True)
    hi = arr.max(axis=-1, keepdims=True)
    rng = hi - lo
    out = np.where(rng > 0, (arr - lo) / np.maximum(rng, 1e-12), 0.5)
    return out.astype(np.float32)


def _validate_layer_samples(
    samples: Sequence[Sequence[np.ndarray]], n_heads_expected: int | None,
) -> tuple[int, int, int]:
    """Returns (n_layers, n_heads, head_dim).  Raises ValueError on bad input."""
    if len(samples) == 0:
        raise ValueError("samples must contain at least one layer")
    first_layer = samples[0]
    if len(first_layer) == 0:
        raise ValueError("layer 0 has no sample tokens — need ≥ 1")
    probe = first_layer[0]
    if probe.ndim != 2:
        raise ValueError(
            f"each sample tensor must be 2-D (n_heads, head_dim); "
            f"got shape {probe.shape}"
        )
    n_heads, head_dim = probe.shape
    if n_heads_expected is not None and n_heads != n_heads_expected:
        raise ValueError(
            f"n_heads mismatch: expected {n_heads_expected}, got {n_heads}"
        )
    return len(samples), n_heads, head_dim


class HeadImportanceAnalyzer:
    """Gradient-free per-head importance estimator.

    Parameters
    ----------
    weights : dict with keys ``"variance"``, ``"concentration"``, ``"magnitude"``.
        Relative contribution of each scoring term.  Auto-normalised to sum to 1.
        Default: equal weight (1/3 each).
    n_heads_expected : optional sanity-check on calibration tensor shape.

    Notes
    -----
    A *single* analyzer instance is reusable across cache configurations — the
    scores it produces depend only on the calibration tensors, not on the cache.
    """

    def __init__(
        self,
        weights: dict | None = None,
        n_heads_expected: int | None = None,
    ) -> None:
        if weights is None:
            weights = {"variance": 1.0, "concentration": 1.0, "magnitude": 1.0}
        bad = [k for k in weights if k not in _VALID_MIX]
        if bad:
            raise ValueError(
                f"unknown weight keys {bad}; valid: {_VALID_MIX}"
            )
        total = sum(weights.get(k, 0.0) for k in _VALID_MIX)
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        self._weights = {k: float(weights.get(k, 0.0)) / total for k in _VALID_MIX}
        self._n_heads_expected = n_heads_expected

    @property
    def weights(self) -> dict:
        return dict(self._weights)

    def score(
        self, samples: Sequence[Sequence[np.ndarray]],
    ) -> HeadImportanceScores:
        """Score every (layer, head) pair.

        Parameters
        ----------
        samples : sequence of sequences.  ``samples[layer]`` is a list of
            calibration tokens, each shape ``(n_heads, head_dim)``.  All
            layers must have the same ``(n_heads, head_dim)`` shape.

        Returns
        -------
        HeadImportanceScores
        """
        n_layers, n_heads, head_dim = _validate_layer_samples(
            samples, self._n_heads_expected,
        )

        raw_variance      = np.zeros((n_layers, n_heads), dtype=np.float64)
        raw_concentration = np.zeros((n_layers, n_heads), dtype=np.float64)
        raw_magnitude     = np.zeros((n_layers, n_heads), dtype=np.float64)

        for li, layer in enumerate(samples):
            if len(layer) == 0:
                log.warning("head_importance: layer %d has no samples; "
                            "treating as zero importance", li)
                continue
            # Stack into (n_tokens, n_heads, head_dim) float64 — numeric safety.
            stack = np.stack(
                [t.astype(np.float64, copy=False) for t in layer], axis=0,
            )
            if stack.shape[1:] != (n_heads, head_dim):
                raise ValueError(
                    f"layer {li}: shape {stack.shape[1:]} differs from "
                    f"layer 0 {(n_heads, head_dim)}"
                )

            # Per-head channel variance (sum of variances across head_dim).
            # var over tokens, then sum over channels → scalar per head.
            raw_variance[li] = stack.var(axis=0).sum(axis=-1)

            # Per-token L2 norm: (n_tokens, n_heads)
            l2 = np.linalg.norm(stack, axis=-1)            # (n_tokens, n_heads)
            mean_l2 = l2.mean(axis=0)
            max_l2  = l2.max(axis=0)
            raw_concentration[li] = max_l2 / np.maximum(mean_l2, 1e-12)
            raw_magnitude[li]     = mean_l2

        var_norm  = _normalise_per_layer(raw_variance)
        conc_norm = _normalise_per_layer(raw_concentration)
        mag_norm  = _normalise_per_layer(raw_magnitude)

        combined = (
            self._weights["variance"]      * var_norm
            + self._weights["concentration"] * conc_norm
            + self._weights["magnitude"]     * mag_norm
        ).astype(np.float32)

        return HeadImportanceScores(
            per_layer=combined,
            n_layers=n_layers,
            n_heads=n_heads,
            weights=self.weights,
        )

    def prune_heads(
        self, samples: Sequence[Sequence[np.ndarray]], threshold: float = 0.1,
    ) -> tuple[HeadImportanceScores, np.ndarray]:
        """Score and return a keep-mask in one call.

        Returns
        -------
        scores : HeadImportanceScores
        keep   : (n_layers, n_heads) bool — True for heads to keep.
        """
        scores = self.score(samples)
        return scores, scores.head_mask(threshold)
