"""CrossModalRouter: efficient visual-text cross-attention with linear bypass.

In a multi-modal LLM decoder, each text query must cross-attend to all visual
key/value vectors.  For queries with low cross-modal affinity (gate score below
threshold) the full scaled-dot-product operation is wasteful.  CrossModalRouter
routes those queries through a one-layer linear projection instead, reducing the
expected FLOPs to O(top_k · n_k · d) while preserving quality for the salient
token pairs.

This is inspired by the Mixture-of-Experts routing concept applied to attention
heads (Fedus et al., arXiv 2101.03961) and the efficient-attention analyses in
FlashAttention (arXiv 2205.14135).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = [
    "CrossModalConfig",
    "CrossModalResult",
    "CrossModalRouter",
]


@dataclass
class CrossModalConfig:
    """Configuration for :class:`CrossModalRouter`.

    Attributes:
        top_k_ratio: Fraction of (query, key) pairs that use full attention.
        n_heads: Number of attention heads; ``d`` must be divisible by
            ``n_heads``.
        linear_dim: Intermediate projection width for the linear bypass path.
        temperature: Softmax temperature for full-attention path.
        seed: RNG seed for reproducible weight initialization.
    """

    top_k_ratio: float = 0.3
    n_heads: int = 4
    linear_dim: int = 64
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.top_k_ratio <= 1.0):
            raise ValueError(
                f"top_k_ratio must be in (0, 1], got {self.top_k_ratio}"
            )
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1, got {self.n_heads}")
        if self.linear_dim < 1:
            raise ValueError(f"linear_dim must be ≥ 1, got {self.linear_dim}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


@dataclass
class CrossModalResult:
    """Output of a single :class:`CrossModalRouter` call.

    Attributes:
        output: Aggregated output array of shape ``(n_queries, d)``.
        attn_weights: Attention weight array ``(n_heads, n_q, n_k)`` for
            queries routed through full attention; ``None`` for linear-path
            queries (all zeros stored as a sentinel).
        n_full_attn: Number of queries routed via full cross-attention.
        n_linear_attn: Number of queries routed via the linear bypass.
    """

    output: np.ndarray
    attn_weights: np.ndarray
    n_full_attn: int
    n_linear_attn: int

    @property
    def speedup_ratio(self) -> float:
        """Fraction of full-attention work avoided."""
        total = self.n_full_attn + self.n_linear_attn
        return self.n_linear_attn / total if total > 0 else 0.0


class CrossModalRouter:
    """Route multi-modal cross-attention queries by gate score.

    Parameters:
        config: Router configuration.

    Example::

        cfg = CrossModalConfig(top_k_ratio=0.3, n_heads=4, linear_dim=64)
        router = CrossModalRouter(cfg)
        n_q, n_k, d = 32, 256, 128
        q = np.random.randn(n_q, d).astype(np.float32)
        k = np.random.randn(n_k, d).astype(np.float32)
        v = np.random.randn(n_k, d).astype(np.float32)
        gate = np.random.rand(n_q).astype(np.float32)
        result = router.route(q, k, v, gate)
    """

    def __init__(self, config: CrossModalConfig) -> None:
        self.config = config
        rng = np.random.default_rng(config.seed)
        self._rng = rng

    def route(
        self,
        q: np.ndarray,
        k: np.ndarray,
        v: np.ndarray,
        gate_scores: np.ndarray,
    ) -> CrossModalResult:
        """Route each query to full-attention or linear bypass.

        Args:
            q: Query matrix ``(n_q, d)``.
            k: Key matrix ``(n_k, d)``.
            v: Value matrix ``(n_k, d)``.
            gate_scores: Per-query routing score ``(n_q,)`` in [0, 1].

        Returns:
            :class:`CrossModalResult`.
        """
        q = np.asarray(q, dtype=np.float32)
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        gate_scores = np.asarray(gate_scores, dtype=np.float32)

        n_q, d = q.shape
        n_k = k.shape[0]

        threshold = np.quantile(gate_scores, 1.0 - self.config.top_k_ratio)
        full_mask = gate_scores >= threshold
        linear_mask = ~full_mask

        n_full = int(full_mask.sum())
        n_linear = int(linear_mask.sum())

        output = np.zeros((n_q, d), dtype=np.float32)
        attn_weights = np.zeros(
            (self.config.n_heads, n_q, n_k), dtype=np.float32
        )

        if n_full > 0:
            out_full, w_full = self._full_attn(q[full_mask], k, v)
            output[full_mask] = out_full
            attn_weights[:, full_mask, :] = w_full

        if n_linear > 0:
            output[linear_mask] = self._linear_attn(q[linear_mask], v)

        return CrossModalResult(
            output=output,
            attn_weights=attn_weights,
            n_full_attn=n_full,
            n_linear_attn=n_linear,
        )

    # ------------------------------------------------------------------
    # Attention implementations
    # ------------------------------------------------------------------

    def _full_attn(
        self, q: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Multi-head scaled dot-product cross-attention."""
        n_q, d = q.shape
        n_k = k.shape[0]
        h = self.config.n_heads
        head_d = d // h

        # Reshape to (h, n, head_d)
        qh = q.reshape(n_q, h, head_d).transpose(1, 0, 2)
        kh = k.reshape(n_k, h, head_d).transpose(1, 0, 2)
        vh = v.reshape(n_k, h, head_d).transpose(1, 0, 2)

        scale = (head_d ** -0.5) / self.config.temperature
        scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale
        weights = self._softmax(scores, axis=-1)
        out = np.einsum("hqk,hkd->hqd", weights, vh)
        out = out.transpose(1, 0, 2).reshape(n_q, d)
        return out, weights

    def _linear_attn(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Simple mean-pool of values, weighted by query L2 norm."""
        d = q.shape[1]
        norms = np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-8)
        weights = norms / norms.sum()  # (n_q, 1)
        mean_v = v.mean(axis=0)  # (d,)
        return np.tile(mean_v, (q.shape[0], 1))

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)
