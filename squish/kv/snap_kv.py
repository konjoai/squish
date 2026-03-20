"""
squish/kv/snap_kv.py

SnapKV: Observation-Window KV Compression.

Reference
---------
Li et al. "SnapKV: LLM Knows What You Are Looking For Before Generation."
NeurIPS 2024.

Algorithm
---------
Before decode starts, SnapKV observes a short *observation window* of recent
prompt tokens (the last ``obs_window`` positions).  For each head it computes
attention scores between the observation window and ALL prior KV positions, then
pools the scores with a max-pooling kernel to produce one importance value per
KV position.  Only the top-``budget`` positions (plus the mandatory observation
window itself) are retained; the rest are discarded.

The net result is a compressed KV cache that is much smaller than the full
context, enabling faster decode attention while preserving the positions the
model actually looks at.

Key properties
--------------
* ``obs_window`` — number of recent positions used as observers (default 16).
* ``budget`` — maximum KV positions to keep (default 512).
* ``pool_kernel`` — max-pool window for importance smoothing (default 5).
* Retains the full observation window unconditionally.
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SnapKVConfig:
    """Configuration for SnapKV."""

    obs_window: int = 16
    """Recent tokens used as the observation window for importance scoring."""

    budget: int = 512
    """Maximum KV positions retained after compression."""

    pool_kernel: int = 5
    """Max-pool kernel size for smoothing attention importance scores."""

    head_dim: int = 64
    """Attention head dimension (used for score scaling)."""

    def __post_init__(self) -> None:
        if self.obs_window < 1:
            raise ValueError("obs_window must be >= 1")
        if self.budget < 1:
            raise ValueError("budget must be >= 1")
        if self.pool_kernel < 1:
            raise ValueError("pool_kernel must be >= 1")


@dataclass
class SnapKVStats:
    """Runtime statistics for SnapKV operations."""

    compress_calls: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0

    @property
    def mean_compression_ratio(self) -> float:
        if self.total_tokens_out == 0:
            return 0.0
        return self.total_tokens_in / self.total_tokens_out


class SnapKV:
    """Observation-window KV cache compressor.

    Usage
    -----
    ::

        snap = SnapKV()
        keys_c, values_c = snap.compress(keys, values)
        # keys_c, values_c have at most ``config.budget`` rows.
    """

    def __init__(self, config: Optional[SnapKVConfig] = None) -> None:
        self.config = config or SnapKVConfig()
        self.stats = SnapKVStats()

    # ------------------------------------------------------------------
    # Importance scoring
    # ------------------------------------------------------------------

    def _importance_scores(self, keys: np.ndarray) -> np.ndarray:
        """Compute per-position importance based on observation-window attention.

        Parameters
        ----------
        keys:
            Shape ``(seq_len, head_dim)``.

        Returns
        -------
        scores:
            Shape ``(seq_len,)`` — higher = more important.
        """
        cfg = self.config
        seq_len, hd = keys.shape
        obs_len = min(cfg.obs_window, seq_len)
        obs_queries = keys[-obs_len:]          # (obs_len, hd)
        scale = 1.0 / math.sqrt(hd)

        # Attention logits: (obs_len, seq_len)
        logits = (obs_queries @ keys.T) * scale
        # Softmax along seq_len axis
        logits -= logits.max(axis=1, keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-9

        # Pool across obs_len → (seq_len,)
        importance = weights.mean(axis=0)

        # Smooth with max-pool kernel
        k = cfg.pool_kernel
        if k > 1 and seq_len >= k:
            smoothed = np.empty_like(importance)
            half = k // 2
            for i in range(seq_len):
                lo = max(0, i - half)
                hi = min(seq_len, i + half + 1)
                smoothed[i] = importance[lo:hi].max()
            importance = smoothed

        return importance

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress(
        self,
        keys: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compress a KV cache by retaining the top-budget positions.

        Parameters
        ----------
        keys:
            Shape ``(seq_len, head_dim)``.
        values:
            Shape ``(seq_len, head_dim)``.

        Returns
        -------
        keys_compressed, values_compressed:
            Each of shape ``(min(budget, seq_len), head_dim)``.
        """
        self.stats.compress_calls += 1
        seq_len = keys.shape[0]
        self.stats.total_tokens_in += seq_len

        cfg = self.config
        if seq_len <= cfg.budget:
            self.stats.total_tokens_out += seq_len
            return keys.copy(), values.copy()

        obs_len = min(cfg.obs_window, seq_len)
        prior_len = seq_len - obs_len

        importance = self._importance_scores(keys)

        # Always keep the observation window
        obs_mask = np.zeros(seq_len, dtype=bool)
        obs_mask[prior_len:] = True

        # From prior positions, pick top-(budget - obs_len)
        prior_budget = max(0, cfg.budget - obs_len)
        prior_scores = importance[:prior_len]
        if prior_budget > 0 and prior_len > 0:
            top_prior = np.argpartition(prior_scores, -min(prior_budget, prior_len))[
                -min(prior_budget, prior_len) :
            ]
            selected_prior = np.sort(top_prior)
            keep_idx = np.concatenate([selected_prior, np.arange(prior_len, seq_len)])
        else:
            keep_idx = np.arange(prior_len, seq_len)

        self.stats.total_tokens_out += len(keep_idx)
        return keys[keep_idx], values[keep_idx]

    def reset_stats(self) -> None:
        """Reset runtime counters."""
        self.stats = SnapKVStats()
