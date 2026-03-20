"""squish/kv/green_kv.py

GreenKV — Accurate KV Cache Eviction with Per-Head Budget Adjustment.

Reference
---------
"GreenKV: Accurate and Efficient KV Cache Eviction with Budget Adjustment."
arXiv:2412.15838, 2024.

Algorithm
---------
GreenKV extends SnapKV-style top-K eviction with two enhancements:

1. **Accumulated attention scores** — instead of a single observation window,
   the importance of a KV position is the *sum* of attention weights it
   received across all query positions so far (or a recent window of them).
   This gives a richer importance signal than a single snapshot.

2. **Per-head budget transfer** — heads that concentrate attention on a small
   set of tokens (low *coverage*) need fewer KV slots.  Their unused budget is
   redistributed to heads with high coverage so the global token budget is
   preserved while per-head budgets adapt to actual attention patterns.

Key properties
--------------
* NumPy-only; no GPU dependency.
* ``global_budget`` — total KV positions kept across all heads.
* ``obs_window`` — number of recent positions used to accumulate scores.
* ``min_head_budget`` — each head keeps at least this many positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "GreenKVConfig",
    "GreenKVEviction",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class GreenKVConfig:
    """Configuration for :class:`GreenKVEviction`.

    Attributes:
        global_budget: Total KV positions to retain across all heads.
        obs_window: Recent query positions used for accumulation.
        min_head_budget: Per-head lower bound on retained positions.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    global_budget: int = 512
    obs_window: int = 32
    min_head_budget: int = 16
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.global_budget < 1:
            raise ValueError(f"global_budget must be ≥ 1; got {self.global_budget}")
        if self.obs_window < 1:
            raise ValueError(f"obs_window must be ≥ 1; got {self.obs_window}")
        if self.min_head_budget < 1:
            raise ValueError(
                f"min_head_budget must be ≥ 1; got {self.min_head_budget}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Core class ─────────────────────────────────────────────────────────────────


class GreenKVEviction:
    """Accumulated-score KV eviction with per-head budget redistribution.

    Example::

        cfg     = GreenKVConfig(global_budget=128, n_heads=4, head_dim=16)
        evict   = GreenKVEviction(cfg)

        K = np.random.randn(4, 256, 16).astype(np.float32)
        V = np.random.randn(4, 256, 16).astype(np.float32)
        Q_obs = np.random.randn(4, 32, 16).astype(np.float32)

        K_keep, V_keep, kept_idx = evict.compress(Q_obs, K, V)
    """

    def __init__(self, config: Optional[GreenKVConfig] = None) -> None:
        self.config = config or GreenKVConfig()

    # ── Core ──────────────────────────────────────────────────────────────────

    def _head_budgets(self, scores: np.ndarray) -> np.ndarray:
        """Adaptive per-head budgets from coverage statistics.

        Args:
            scores: ``(H, S)`` per-head accumulated importance scores.

        Returns:
            ``(H,)`` integer budget for each head (sums to global_budget).
        """
        H, S = scores.shape
        cfg = self.config
        # Coverage: fraction of score mass held by top-50% positions
        half = max(1, S // 2)
        coverage = np.zeros(H, dtype=np.float64)
        for h in range(H):
            s = scores[h]
            s_sort = np.sort(s)[::-1]
            total = s_sort.sum() + 1e-9
            coverage[h] = s_sort[:half].sum() / total

        # Inverse-coverage weighting: concentrated heads get fewer slots
        weights = 1.0 - coverage  # (H,)
        weights = np.clip(weights, 1e-9, None)
        weights /= weights.sum()

        # Distribute global budget; enforce per-head minimum
        raw = weights * cfg.global_budget
        budgets = np.maximum(raw, cfg.min_head_budget).astype(int)

        # Trim to global budget (greedy trim from largest)
        while budgets.sum() > cfg.global_budget:
            excess = budgets.sum() - cfg.global_budget
            h_max = budgets.argmax()
            trim = min(int(excess), budgets[h_max] - cfg.min_head_budget)
            if trim <= 0:
                break
            budgets[h_max] -= trim

        return budgets

    def compress(
        self,
        Q_obs: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
        """Compress K/V by evicting low-importance positions.

        Args:
            Q_obs: ``(n_heads, obs_window, head_dim)`` observation query window.
            K: ``(n_heads, S, head_dim)`` full key cache.
            V: ``(n_heads, S, head_dim)`` full value cache.

        Returns:
            Tuple of:
            - ``K_keep``: list of ``(budget_h, head_dim)`` per-head key arrays.
            - ``V_keep``: corresponding value arrays.
            - ``kept_indices``: list of ``(budget_h,)`` integer position arrays.
        """
        Q_obs = np.asarray(Q_obs, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        H, S, d = K.shape
        scale = 1.0 / np.sqrt(d)

        # Accumulated attention scores (H, S)
        scores_mat = np.zeros((H, S), dtype=np.float64)
        for h in range(H):
            q_h = Q_obs[h]  # (obs_window, d)
            logits = q_h @ K[h].T * scale  # (obs_window, S)
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
            scores_mat[h] = attn.mean(axis=0)

        budgets = self._head_budgets(scores_mat)

        K_keep: List[np.ndarray] = []
        V_keep: List[np.ndarray] = []
        kept_indices: List[np.ndarray] = []

        for h in range(H):
            b = min(int(budgets[h]), S)
            top_idx = np.argsort(-scores_mat[h])[:b]
            top_idx = np.sort(top_idx)
            K_keep.append(K[h][top_idx])
            V_keep.append(V[h][top_idx])
            kept_indices.append(top_idx)

        return K_keep, V_keep, kept_indices

    def __repr__(self) -> str:
        return (
            f"GreenKVEviction(global_budget={self.config.global_budget}, "
            f"n_heads={self.config.n_heads})"
        )
