"""pyramid_kv.py — PyramidKV: Layer-wise Adaptive KV Budget Allocation

Allocates KV cache budget per transformer layer following a pyramidal schedule:
lower layers retain more KV entries, upper layers aggressively evict.

Based on: PyramidKV (Zhang et al., 2024) — Dynamic KV Cache Compression based
on Pyramidal Information Funneling.

Key insight: lower transformer layers encode absolute position and local syntax
and benefit from full KV retention; upper layers operate on abstract features
and tolerate aggressive eviction without quality degradation.

Budget schedule:
  layer_budget[l] = max(min_budget, round(base_budget * pyramid_factor(l, n)))
  where pyramid_factor approaches 1.0 at l=0 and min_alpha at l=n-1.

Eviction policy: tokens with lowest accumulated attention-sum importance score
(H2O-style) are evicted when the per-layer budget is exceeded.

Usage:
    cfg = PyramidKVConfig(n_layers=32, base_budget=1024, min_budget=128)
    mgr = PyramidKVManager(cfg)
    budget = mgr.layer_budget(layer_idx=16)          # -> int
    mgr.update_importance(layer_idx, attn_weights_2d) # (seq, seq)
    keep_mask = mgr.eviction_mask(layer_idx, current_seq_len)  # bool[seq]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PyramidKVConfig:
    """Configuration for PyramidKV layer-wise budget allocation.

    Args:
        n_layers:    Total number of transformer layers.
        base_budget: KV budget for the bottommost layer (layer 0).
        min_budget:  Minimum KV budget for any layer (applied at top layers).
        alpha:       Decay factor controlling how steeply the budget falls.
                     alpha=0.5 → top layer gets 50% of base_budget.
                     alpha=0.9 → top layer gets 10% of base_budget.
        warmup_steps: Steps before importance scores are reliable (no eviction).
        ema_alpha:   EMA decay for importance accumulation across requests.
    """
    n_layers: int = 32
    base_budget: int = 1024
    min_budget: int = 64
    alpha: float = 0.7
    warmup_steps: int = 4
    ema_alpha: float = 0.1


@dataclass
class PyramidKVStats:
    """Runtime statistics collected per PyramidKVManager instance."""
    total_evictions: int = 0
    eviction_calls: int = 0
    layer_budgets: List[int] = field(default_factory=list)

    @property
    def mean_eviction_rate(self) -> float:
        if self.eviction_calls == 0:
            return 0.0
        return self.total_evictions / self.eviction_calls

    def reset(self) -> None:
        self.total_evictions = 0
        self.eviction_calls = 0


class PyramidKVManager:
    """Manages layer-wise KV budgets following a pyramidal allocation schedule.

    The budget schedule is computed once at construction; importance scores are
    accumulated per layer via EMA across requests.
    """

    def __init__(self, config: Optional[PyramidKVConfig] = None) -> None:
        self.config = config or PyramidKVConfig()
        cfg = self.config
        if cfg.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if cfg.base_budget < cfg.min_budget:
            raise ValueError("base_budget must be >= min_budget")
        if not (0.0 < cfg.alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        self._budgets: List[int] = self._compute_budgets()
        # importance[l] = 1-D array of shape (seq_len,), EMA-averaged
        self._importance: Dict[int, np.ndarray] = {}
        self._step_counts: Dict[int, int] = {}
        self.stats = PyramidKVStats(layer_budgets=list(self._budgets))

    # ------------------------------------------------------------------
    # Budget computation
    # ------------------------------------------------------------------

    def _pyramid_factor(self, layer_idx: int) -> float:
        """Compute the budget scaling factor for a given layer index.

        Layer 0 gets factor=1.0; top layer gets factor=(1-alpha).
        """
        cfg = self.config
        if cfg.n_layers == 1:
            return 1.0
        # Linear decay from 1.0 at layer 0 to (1-alpha) at layer n-1
        t = layer_idx / (cfg.n_layers - 1)
        return 1.0 - cfg.alpha * t

    def _compute_budgets(self) -> List[int]:
        """Return list of per-layer KV budgets (length = n_layers)."""
        cfg = self.config
        budgets = []
        for l in range(cfg.n_layers):
            factor = self._pyramid_factor(l)
            b = max(cfg.min_budget, round(cfg.base_budget * factor))
            budgets.append(b)
        return budgets

    def layer_budget(self, layer_idx: int) -> int:
        """Return the KV token budget for the given layer index."""
        if layer_idx < 0 or layer_idx >= self.config.n_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self.config.n_layers})"
            )
        return self._budgets[layer_idx]

    def all_budgets(self) -> List[int]:
        """Return all per-layer budgets as a list."""
        return list(self._budgets)

    # ------------------------------------------------------------------
    # Importance accumulation
    # ------------------------------------------------------------------

    def update_importance(
        self, layer_idx: int, attn_weights: np.ndarray
    ) -> None:
        """Update importance scores for a layer using attention weights.

        Args:
            layer_idx:    Layer index (0-indexed).
            attn_weights: 2-D float array of shape (n_heads, seq_len) or
                          (seq_len, seq_len).  The column-sum (or row-sum of
                          the last row for causal attention) is used as the
                          per-token importance signal.
        """
        if attn_weights.ndim == 2:
            # Sum over heads (if shape is heads×seq) or over queries
            score = attn_weights.sum(axis=0).astype(np.float32)
        elif attn_weights.ndim == 3:
            # (n_heads, q_len, k_len) — take last query row per head then mean
            score = attn_weights[:, -1, :].mean(axis=0).astype(np.float32)
        else:
            raise ValueError(
                f"attn_weights must be 2-D or 3-D, got shape {attn_weights.shape}"
            )

        step = self._step_counts.get(layer_idx, 0)
        prev = self._importance.get(layer_idx)
        if prev is None or prev.shape != score.shape:
            self._importance[layer_idx] = score.copy()
        else:
            alpha = self.config.ema_alpha
            # EMA: blend new score with existing estimate
            if score.shape[0] > prev.shape[0]:
                # Sequence grew — pad prev with mean
                pad = np.full(score.shape[0] - prev.shape[0], prev.mean())
                prev_padded = np.concatenate([prev, pad])
                self._importance[layer_idx] = (
                    alpha * score + (1.0 - alpha) * prev_padded
                )
            elif score.shape[0] < prev.shape[0]:
                self._importance[layer_idx] = score.copy()
            else:
                self._importance[layer_idx] = (
                    alpha * score + (1.0 - alpha) * prev
                )
        self._step_counts[layer_idx] = step + 1

    # ------------------------------------------------------------------
    # Eviction
    # ------------------------------------------------------------------

    def eviction_mask(
        self, layer_idx: int, current_seq_len: int
    ) -> np.ndarray:
        """Return a boolean keep-mask of shape (current_seq_len,).

        True = keep this KV position; False = evict it.
        If current_seq_len <= layer_budget, all positions are kept.
        """
        budget = self.layer_budget(layer_idx)
        mask = np.ones(current_seq_len, dtype=bool)

        if current_seq_len <= budget:
            return mask

        # Warmup: do not evict if we haven't seen enough steps
        step = self._step_counts.get(layer_idx, 0)
        if step < self.config.warmup_steps:
            return mask

        importance = self._importance.get(layer_idx)
        if importance is None:
            # No importance data — keep the most recent `budget` tokens
            mask[: current_seq_len - budget] = False
            self.stats.total_evictions += current_seq_len - budget
            self.stats.eviction_calls += 1
            return mask

        # Align importance scores to current_seq_len
        if importance.shape[0] >= current_seq_len:
            scores = importance[:current_seq_len]
        else:
            pad_len = current_seq_len - importance.shape[0]
            scores = np.concatenate(
                [importance, np.full(pad_len, importance.mean())]
            )

        # Always keep the most recent token (position -1)
        # Rank by importance; keep top-budget positions
        threshold_idx = np.argsort(scores)[: current_seq_len - budget]
        mask[threshold_idx] = False
        # Ensure last token always kept
        mask[-1] = True

        n_evicted = int((~mask).sum())
        self.stats.total_evictions += n_evicted
        self.stats.eviction_calls += 1
        return mask

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def reset_layer(self, layer_idx: int) -> None:
        """Clear importance scores for a specific layer (e.g., new request)."""
        self._importance.pop(layer_idx, None)
        self._step_counts.pop(layer_idx, None)

    def reset_all(self) -> None:
        """Clear all importance scores."""
        self._importance.clear()
        self._step_counts.clear()

    def budget_reduction_ratio(self) -> float:
        """Return the mean budget / base_budget ratio across all layers."""
        return sum(self._budgets) / (
            self.config.base_budget * self.config.n_layers
        )
