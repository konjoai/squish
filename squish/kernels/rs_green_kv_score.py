"""squish/kernels/rs_green_kv_score.py — Rust-backed GreenKV importance scoring.

Wraps ``squish_quant_rs.green_kv_score_f32`` with a NumPy fallback.

GreenKV accumulates per-token KV-cache importance scores by computing,
for each head, the mean softmax attention weight over a sliding window
of recent query vectors.  Scores are used to evict low-importance cache
entries in long-context inference.
Rayon parallelises over attention heads.

Reference: Zhang et al., "GreenKV: Achieving KV Cache Compression with
Nearly-Lossless Accuracy for Large Language Models," 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "GreenKVConfig",
    "RustGreenKVScore",
]

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "green_kv_score_f32")
except ImportError:
    _sq = None  # type: ignore[assignment]
    _HAS_RUST = False


# ── NumPy fallback ────────────────────────────────────────────────────────────


def _numpy_green_kv_score(
    q_obs: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Compute mean softmax attention scores over obs_window queries.

    Args:
        q_obs: ``(H, obs_window, D)`` float32.
        k:     ``(H, seq_len,    D)`` float32.

    Returns:
        ``(H, seq_len)`` float32 importance scores.
    """
    n_heads, obs_window, head_dim = q_obs.shape
    seq_len = k.shape[1]
    scale = head_dim ** -0.5
    scores = np.zeros((n_heads, seq_len), dtype=np.float32)
    for h in range(n_heads):
        for qp in range(obs_window):
            logits = q_obs[h, qp] @ k[h].T * scale  # (seq_len,)
            logits -= logits.max()
            w = np.exp(logits)
            w /= w.sum() + 1e-9
            scores[h] += w
        scores[h] /= obs_window
    return scores


# ── Config / wrapper ──────────────────────────────────────────────────────────


@dataclass
class GreenKVConfig:
    """Configuration for :class:`RustGreenKVScore`.

    Attributes:
        obs_window: Number of recent queries used to estimate token importance.
    """

    obs_window: int = 32


class RustGreenKVScore:
    """Rust-accelerated GreenKV per-head KV cache importance scoring.

    For each head, computes mean softmax attention weights over an
    observation window of recent query vectors against the full key cache.
    Returns per-token importance scores for downstream cache eviction.
    Rayon parallelises over heads.
    Falls back to NumPy when ``squish_quant_rs`` is unavailable.
    """

    def __init__(self, config: Optional[GreenKVConfig] = None) -> None:
        self._cfg = config or GreenKVConfig()

    def score(
        self,
        q_obs: np.ndarray,
        k: np.ndarray,
    ) -> np.ndarray:
        """Compute per-token KV-cache importance scores.

        Args:
            q_obs: Recent query slice ``(n_heads, obs_window, head_dim)``
                   float32.
            k:     Full key cache ``(n_heads, seq_len, head_dim)`` float32.

        Returns:
            Importance scores ``(n_heads, seq_len)`` float32.

        Raises:
            ValueError: If head counts or head dims are inconsistent.
        """
        q = np.ascontiguousarray(q_obs, dtype=np.float32)
        kk = np.ascontiguousarray(k, dtype=np.float32)
        if q.ndim != 3 or kk.ndim != 3:
            raise ValueError("q_obs and k must be 3-D (H, T, D)")
        if q.shape[0] != kk.shape[0]:
            raise ValueError(
                f"n_heads mismatch: q_obs={q.shape[0]}, k={kk.shape[0]}"
            )
        if q.shape[2] != kk.shape[2]:
            raise ValueError(
                f"head_dim mismatch: q_obs={q.shape[2]}, k={kk.shape[2]}"
            )
        if _HAS_RUST:
            return np.asarray(_sq.green_kv_score_f32(q, kk), dtype=np.float32)
        return _numpy_green_kv_score(q, kk)

    def top_k_mask(
        self,
        scores: np.ndarray,
        budget: int,
    ) -> np.ndarray:
        """Return a boolean mask retaining the top-budget tokens per head.

        Args:
            scores: ``(n_heads, seq_len)`` float32 importance scores.
            budget: Number of tokens to keep.

        Returns:
            ``(n_heads, seq_len)`` bool mask — True = keep.
        """
        seq_len = scores.shape[1]
        budget = min(budget, seq_len)
        mask = np.zeros_like(scores, dtype=bool)
        for h in range(scores.shape[0]):
            top_idx = np.argpartition(-scores[h], budget - 1)[:budget]
            mask[h, top_idx] = True
        return mask

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
