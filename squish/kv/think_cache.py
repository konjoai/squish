"""squish/kv/think_cache.py

ThinKCache — Thinner Key Cache by Query-Driven Channel Pruning (Xu et al.,
EMNLP 2024 / arXiv:2407.21018).

Reference
---------
"ThinK: Thinner Key Cache by Query-Driven Pruning." Xu et al., EMNLP 2024
(arXiv:2407.21018).

Algorithm
---------
ThinK observes that not all head_dim channels of the K tensors contribute
equally to the attention output.  Per head, channels aligned with the current
query magnitude tend to dominate.  ThinK prunes the *least* query-aligned
channels from K, storing only ``keep_ratio`` fraction of key channels at full
precision, while the remaining channels are dropped (set to zero or omitted).

Steps per layer:
1. Receive Q and K tensors ``(H, T, d)``.
2. Compute per-channel importance = mean over T of |Q| * |K|: ``imp(h, c) = Σ_t |Q[h,t,c]| * |K[h,t,c]|``.
3. Keep top-``keep_ratio * d`` channels by importance; zero-out the rest.
4. Return pruned K.  V is not modified.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* 20% K-channel reduction by default; <0.1 PPL cost.
* Compatible with any downstream KV eviction or quantization module.
* ``keep_ratio`` in (0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "ThinKConfig",
    "ThinKCache",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class ThinKConfig:
    """Configuration for :class:`ThinKCache`.

    Attributes:
        keep_ratio: Fraction of key channels to retain (default 0.8 = keep 80%).
        n_heads: Number of attention heads.
        head_dim: Key/value dimension per head.
    """

    keep_ratio: float = 0.8
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if not (0.0 < self.keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0, 1]; got {self.keep_ratio}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── ThinKCache ────────────────────────────────────────────────────────────────


class ThinKCache:
    """Query-driven K-channel pruning cache.

    Example::

        cfg = ThinKConfig(keep_ratio=0.8, n_heads=4, head_dim=16)
        cache = ThinKCache(cfg)

        rng = np.random.default_rng(0)
        Q = rng.standard_normal((4, 8, 16)).astype(np.float32)
        K = rng.standard_normal((4, 8, 16)).astype(np.float32)
        K_pruned = cache.prune_k(Q, K)   # shape (4, 8, 16) with 20% channels zeroed
    """

    def __init__(self, config: Optional[ThinKConfig] = None) -> None:
        self.config = config or ThinKConfig()
        self._total_pruned_channels = 0
        self._total_channels = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def prune_k(
        self,
        Q: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Prune K channels not aligned to query magnitude.

        Args:
            Q: ``(n_heads, T, head_dim)`` query tensor.
            K: ``(n_heads, S, head_dim)`` key tensor.

        Returns:
            ``(n_heads, S, head_dim)`` pruned K with dropped channels zeroed.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        H, T, d = Q.shape
        _, S, _ = K.shape
        cfg = self.config
        keep_k = max(1, round(cfg.keep_ratio * d))

        K_pruned = K.copy()
        for h in range(H):
            # Per-channel importance: sum |Q[h,:,c]| * |K[h,:,c]| over T and S
            q_importance = np.abs(Q[h]).mean(axis=0)  # (d,)
            k_importance = np.abs(K[h]).mean(axis=0)  # (d,)
            importance = q_importance * k_importance

            keep_idx = np.argpartition(importance, -keep_k)[-keep_k:]
            drop_mask = np.ones(d, dtype=bool)
            drop_mask[keep_idx] = False
            K_pruned[h, :, drop_mask] = 0.0
            self._total_pruned_channels += int(drop_mask.sum())
            self._total_channels += d

        return K_pruned

    def keep_indices(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Return ``(H, keep_k)`` array of kept channel indices per head.

        Args:
            Q: ``(n_heads, T, head_dim)``.
            K: ``(n_heads, S, head_dim)``.

        Returns:
            int64 array of shape ``(H, keep_k)``.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        H, T, d = Q.shape
        keep_k = max(1, round(self.config.keep_ratio * d))
        indices = np.zeros((H, keep_k), dtype=np.int64)
        for h in range(H):
            importance = np.abs(Q[h]).mean(axis=0) * np.abs(K[h]).mean(axis=0)
            indices[h] = np.argpartition(importance, -keep_k)[-keep_k:]
        return indices

    def channel_reduction_ratio(self) -> float:
        """Fraction of channels pruned across all calls."""
        if self._total_channels == 0:
            return 0.0
        return self._total_pruned_channels / self._total_channels

    def reset_stats(self) -> None:
        """Reset pruning statistics."""
        self._total_pruned_channels = 0
        self._total_channels = 0

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"ThinKCache(keep_ratio={cfg.keep_ratio}, "
            f"n_heads={cfg.n_heads}, head_dim={cfg.head_dim})"
        )
