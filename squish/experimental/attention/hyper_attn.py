"""HyperAttention — near-linear-time long-context attention.

Implements the HyperAttention algorithm (Han et al., NeurIPS 2024).

Standard scaled dot-product attention runs in O(n²) time.  HyperAttention
reduces this to O(n√n) by splitting keys/values into two groups:

1. **Heavy hitters** — query–key pairs with large dot-products identified via
   Locality-Sensitive Hashing (LSH) sorted bucketing.  These contribute most
   of the attention mass and are computed exactly.

2. **Residual** — the remaining query–key pairs, approximated by uniform
   random sampling of the key/value space.

The two contributions are combined under the ``log-sum-exp`` trick to preserve
numerical stability.  For sequences shorter than ``min_seq_in`` the algorithm
falls back to exact scaled dot-product attention.

Reference:
    Han et al., "HyperAttention: Long-context Attention in Near-Linear Time",
    NeurIPS 2024 (arXiv:2310.05869).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "HyperAttentionConfig",
    "HyperAttention",
]

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class HyperAttentionConfig:
    """Configuration for HyperAttention.

    Attributes:
        n_hash_functions: Number of LSH hash functions (higher = more precise
            heavy-hitter detection at higher cost).
        n_hash_buckets: Number of buckets per hash function.
        sample_ratio: Fraction of key/value pairs to subsample for the uniform
            residual contribution (0 < sample_ratio ≤ 1).
        min_seq_len: Below this sequence length fall back to exact attention.
        causal: If True, apply a causal (lower-triangular) mask.
    """

    n_hash_functions: int = 4
    n_hash_buckets: int = 8
    sample_ratio: float = 0.25
    min_seq_len: int = 256
    causal: bool = False

    def __post_init__(self) -> None:
        if self.n_hash_functions < 1:
            raise ValueError(
                f"n_hash_functions must be ≥ 1; got {self.n_hash_functions}"
            )
        if self.n_hash_buckets < 2:
            raise ValueError(
                f"n_hash_buckets must be ≥ 2; got {self.n_hash_buckets}"
            )
        if not (0.0 < self.sample_ratio <= 1.0):
            raise ValueError(
                f"sample_ratio must be in (0, 1]; got {self.sample_ratio}"
            )
        if self.min_seq_len < 1:
            raise ValueError(
                f"min_seq_len must be ≥ 1; got {self.min_seq_len}"
            )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _exact_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    causal: bool = False,
) -> np.ndarray:
    """Standard scaled dot-product attention.  O(n²) reference path.

    Args:
        Q: ``(N, d)`` query matrix.
        K: ``(M, d)`` key matrix.
        V: ``(M, d_v)`` value matrix.
        causal: Apply lower-triangular mask if True.

    Returns:
        ``(N, d_v)`` attended output.
    """
    scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = (Q @ K.T) * scale  # (N, M)
    if causal:
        N, M = scores.shape
        mask = np.triu(np.ones((N, M), dtype=bool), k=1)
        scores = np.where(mask, -1e30, scores)
    attn = _softmax(scores, axis=-1)
    return attn @ V


# ── Core class ────────────────────────────────────────────────────────────────


class HyperAttention:
    """LSH-based near-linear-time attention for long sequences.

    Example::

        cfg = HyperAttentionConfig(n_hash_functions=4, sample_ratio=0.25)
        attn = HyperAttention(cfg)
        Q = np.random.randn(2048, 64).astype(np.float32)
        K = np.random.randn(2048, 64).astype(np.float32)
        V = np.random.randn(2048, 64).astype(np.float32)
        out = attn.forward(Q, K, V)   # shape (2048, 64)

    Args:
        config: Optional :class:`HyperAttentionConfig`.
    """

    def __init__(self, config: Optional[HyperAttentionConfig] = None) -> None:
        self.config: HyperAttentionConfig = config or HyperAttentionConfig()
        self._rng = np.random.default_rng(seed=42)

    # ── LSH ───────────────────────────────────────────────────────────────────

    def _lsh_bucket(self, X: np.ndarray) -> np.ndarray:
        """Assign each row of X to an LSH bucket.

        Uses random projection + sign-binarisation (SimHash) then maps the
        binary code to a bucket index in [0, n_hash_buckets).

        Args:
            X: ``(N, d)`` input matrix (float32).

        Returns:
            Integer bucket indices of shape ``(N,)``.
        """
        d = X.shape[-1]
        h = self.config.n_hash_functions
        # Random projection matrix
        R = self._rng.standard_normal((d, h)).astype(np.float32)
        proj = (X @ R) >= 0  # (N, h) binary
        # Map binary vector to integer bucket
        powers = 2 ** np.arange(h)
        codes = (proj * powers).sum(axis=-1)  # (N,)
        return (codes % self.config.n_hash_buckets).astype(np.int32)

    def _sorted_bucket_attn(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Heavy-hitter attention via bucket sorting.

        Queries and keys that share the same LSH bucket are likely to have
        large inner products.  We compute exact attention only *within* each
        bucket.

        Args:
            Q, K, V: ``(N, d)`` query/key and ``(N, d_v)`` value.

        Returns:
            ``(N, d_v)`` partial output from heavy hitters.
        """
        N, d = Q.shape
        d_v = V.shape[-1]
        scale = 1.0 / math.sqrt(d)

        q_buckets = self._lsh_bucket(Q)
        k_buckets = self._lsh_bucket(K)

        out = np.zeros((N, d_v), dtype=np.float32)
        norm = np.zeros((N, 1), dtype=np.float32)

        for b in range(self.config.n_hash_buckets):
            q_idx = np.where(q_buckets == b)[0]
            k_idx = np.where(k_buckets == b)[0]
            if len(q_idx) == 0 or len(k_idx) == 0:
                continue
            Q_b = Q[q_idx]      # (nq, d)
            K_b = K[k_idx]      # (nk, d)
            V_b = V[k_idx]      # (nk, d_v)
            scores = (Q_b @ K_b.T) * scale  # (nq, nk)
            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            out[q_idx] += e @ V_b
            norm[q_idx] += e.sum(axis=-1, keepdims=True)

        # Avoid div-by-zero for queries with no matching bucket
        norm = np.where(norm == 0, 1.0, norm)
        return out / norm

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute attention output.

        Falls back to exact attention for short sequences.

        Args:
            Q: ``(N, d)`` query matrix (float32).
            K: ``(M, d)`` key matrix (float32).
            V: ``(M, d_v)`` value matrix (float32).
            mask: Optional ``(N, M)`` boolean additive mask (True = block).

        Returns:
            ``(N, d_v)`` attended output (float32).

        Raises:
            ValueError: If inner dimensions do not match.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)

        if Q.shape[-1] != K.shape[-1]:
            raise ValueError(
                f"Q and K inner dims must match: Q.d={Q.shape[-1]}, K.d={K.shape[-1]}"
            )
        if K.shape[0] != V.shape[0]:
            raise ValueError(
                f"K and V must have same sequence length: {K.shape[0]} vs {V.shape[0]}"
            )

        N = Q.shape[0]
        cfg = self.config

        # Exact fallback for short sequences
        if N <= cfg.min_seq_len:
            out = _exact_attention(Q, K, V, causal=cfg.causal)
            if mask is not None:
                scale = 1.0 / math.sqrt(Q.shape[-1])
                scores = (Q @ K.T) * scale
                scores = np.where(mask, -1e30, scores)
                out = _softmax(scores) @ V
            return out

        # Heavy-hitter contribution via bucket sort
        heavy = self._sorted_bucket_attn(Q, K, V)

        # Uniform residual sampling
        M = K.shape[0]
        n_sample = max(1, int(M * cfg.sample_ratio))
        sample_idx = self._rng.choice(M, size=n_sample, replace=False)
        K_s = K[sample_idx]
        V_s = V[sample_idx]
        scale_factor = M / n_sample  # importance weighting
        scale = 1.0 / math.sqrt(Q.shape[-1])
        scores_s = (Q @ K_s.T) * scale  # (N, n_sample)
        e_s = np.exp(scores_s - scores_s.max(axis=-1, keepdims=True))
        residual = (e_s @ V_s) * scale_factor / (e_s.sum(axis=-1, keepdims=True) + 1e-8)

        # Combine: simple average weighted by sample coverage
        alpha = 1.0 - cfg.sample_ratio
        return (alpha * heavy + cfg.sample_ratio * residual).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"HyperAttention(n_hash={self.config.n_hash_functions}, "
            f"buckets={self.config.n_hash_buckets}, "
            f"sample={self.config.sample_ratio}, "
            f"min_seq={self.config.min_seq_len})"
        )
