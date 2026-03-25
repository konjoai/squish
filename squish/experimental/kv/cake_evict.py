"""squish/kv/cake_evict.py

CAKE — Cascading and Adaptive KV Cache Eviction with Layer-wise Budget.

Reference
---------
"CAKE: Cascading and Adaptive KV Cache Eviction with Layer Importance-based
Budget Allocation." NeurIPS 2024 workshop (arXiv:2410.22143).

Algorithm
---------
CAKE allocates the total KV budget non-uniformly across layers by measuring
each layer's **attention entropy**:

1. Layers with high attention entropy are doing broad retrieval — they need
   more KV slots.
2. Layers with low entropy are attending narrowly — their budget can be
   reduced.

The budget for layer ``l`` is:

    budget_l = global_budget * softmax(entropy_l / temperature)[l]

Within each layer, positions are ranked by their accumulated attention score
and the top-``budget_l`` kept.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``global_budget`` — total KV positions summed over all layers.
* ``n_layers`` — number of transformer layers.
* ``temperature`` — softmax temperature for budget distribution.
* ``obs_window`` — query positions used to compute per-layer entropy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "CAKEConfig",
    "CAKEEviction",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class CAKEConfig:
    """Configuration for :class:`CAKEEviction`.

    Attributes:
        global_budget: Total KV positions across all layers.
        n_layers: Number of transformer layers.
        obs_window: Recent query positions for entropy computation.
        temperature: Softmax temperature for budget distribution (higher → more uniform).
        min_layer_budget: Minimum tokens retained per layer.
        n_heads: Attention heads per layer.
        head_dim: Dimension per head.
    """

    global_budget: int = 2048
    n_layers: int = 12
    obs_window: int = 32
    temperature: float = 1.0
    min_layer_budget: int = 16
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.global_budget < 1:
            raise ValueError(f"global_budget must be ≥ 1; got {self.global_budget}")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1; got {self.n_layers}")
        if self.obs_window < 1:
            raise ValueError(f"obs_window must be ≥ 1; got {self.obs_window}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")
        if self.min_layer_budget < 1:
            raise ValueError(
                f"min_layer_budget must be ≥ 1; got {self.min_layer_budget}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Core class ─────────────────────────────────────────────────────────────────


class CAKEEviction:
    """Entropy-adaptive layer-wise KV eviction.

    Example::

        cfg   = CAKEConfig(global_budget=512, n_layers=4, n_heads=2, head_dim=8)
        cake  = CAKEEviction(cfg)

        # Per-layer inputs: list of (Q_obs, K, V) tuples
        layers = [
            (np.random.randn(2, 8, 8).astype(np.float32),
             np.random.randn(2, 128, 8).astype(np.float32),
             np.random.randn(2, 128, 8).astype(np.float32))
            for _ in range(4)
        ]
        budgets = cake.compute_budgets(layers)
        K_outs, V_outs, indices = cake.compress(layers, budgets)
    """

    def __init__(self, config: Optional[CAKEConfig] = None) -> None:
        self.config = config or CAKEConfig()

    # ── Budget computation ────────────────────────────────────────────────────

    def _layer_entropy(
        self,
        Q_obs: np.ndarray,
        K: np.ndarray,
    ) -> float:
        """Compute mean normalised attention entropy for one layer.

        Args:
            Q_obs: ``(n_heads, obs_window, head_dim)`` observation queries.
            K: ``(n_heads, S, head_dim)`` key cache.

        Returns:
            Scalar mean normalised entropy in [0, 1].
        """
        H, W, d = Q_obs.shape
        S = K.shape[1]
        scale = 1.0 / np.sqrt(d)
        log_S = np.log(float(S) + 1e-9)
        entropies = []
        for h in range(H):
            logits = Q_obs[h] @ K[h].T * scale  # (W, S)
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
            ent = -(attn * np.log(attn + 1e-9)).sum(axis=-1)  # (W,)
            entropies.append(float(ent.mean() / log_S))
        return float(np.mean(entropies))

    def compute_budgets(
        self,
        layers: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        """Compute per-layer KV budgets from attention entropy.

        Args:
            layers: List of ``(Q_obs, K, V)`` tuples, one per layer.

        Returns:
            ``(n_layers,)`` integer budget array summing to ≤ global_budget.
        """
        entropies = np.array(
            [self._layer_entropy(qo, k) for qo, k, _ in layers],
            dtype=np.float64,
        )
        temp = self.config.temperature
        # Softmax over entropies
        e = np.exp(entropies / temp - (entropies / temp).max())
        weights = e / e.sum()
        raw = weights * self.config.global_budget
        budgets = np.maximum(raw, self.config.min_layer_budget).astype(int)
        # Trim to global budget
        while budgets.sum() > self.config.global_budget:
            h_max = budgets.argmax()
            trim = min(budgets.sum() - self.config.global_budget,
                       budgets[h_max] - self.config.min_layer_budget)
            if trim <= 0:
                break
            budgets[h_max] -= trim
        return budgets

    # ── Compression ───────────────────────────────────────────────────────────

    def compress(
        self,
        layers: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        budgets: Optional[np.ndarray] = None,
    ) -> Tuple[List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
        """Compress each layer's KV to its budget.

        Args:
            layers: List of ``(Q_obs, K, V)`` tuples (one per layer).
            budgets: Optional pre-computed budgets from :meth:`compute_budgets`.
                If None, budgets are computed internally.

        Returns:
            ``(K_outs, V_outs, indices)`` — each is a list of per-layer lists
            of per-head arrays.
        """
        if budgets is None:
            budgets = self.compute_budgets(layers)

        K_outs: List[List[np.ndarray]] = []
        V_outs: List[List[np.ndarray]] = []
        idx_outs: List[List[np.ndarray]] = []

        for l_idx, (Q_obs, K, V) in enumerate(layers):
            Q_obs = np.asarray(Q_obs, dtype=np.float32)
            K = np.asarray(K, dtype=np.float32)
            V = np.asarray(V, dtype=np.float32)
            H, S, d = K.shape
            budget = min(int(budgets[l_idx]), S)
            scale = 1.0 / np.sqrt(d)

            kl, vl, il = [], [], []
            for h in range(H):
                logits = Q_obs[h] @ K[h].T * scale  # (W, S)
                e = np.exp(logits - logits.max(axis=-1, keepdims=True))
                attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
                scores = attn.mean(axis=0)  # (S,)
                top_idx = np.sort(np.argsort(-scores)[:budget])
                kl.append(K[h][top_idx])
                vl.append(V[h][top_idx])
                il.append(top_idx)

            K_outs.append(kl)
            V_outs.append(vl)
            idx_outs.append(il)

        return K_outs, V_outs, idx_outs

    def __repr__(self) -> str:
        return (
            f"CAKEEviction(global_budget={self.config.global_budget}, "
            f"n_layers={self.config.n_layers})"
        )
