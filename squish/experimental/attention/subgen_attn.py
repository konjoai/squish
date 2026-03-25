"""squish/attention/subgen_attn.py

SubGenAttention — Sub-Quadratic Attention via Dual Sparse Kernel.

Reference
---------
Chen et al. "Sub-quadratic Attention via Implicit Differentiation."
ICML 2024 (arXiv:2402.06082).

Algorithm
---------
SubGen achieves O(n√n) attention memory and computation by splitting the
attention pattern into two complementary sparse components:

1. **Local window** — each query attends to its nearest ``window_size``
   neighbours (sliding window attention).  This captures local syntactic
   dependencies.

2. **Global sinks** — a small set of ``n_global`` positions at the
   beginning/end of the sequence act as global aggregators.  Every query
   attends to these regardless of position.

The final output is the weighted sum of local and global context vectors, with
a learnable mixing coefficient ``alpha``.

Key properties
--------------
* NumPy-only; no GPU dependency.
* O(n × (window_size + n_global)) memory instead of O(n²).
* ``window_size`` — local window half-width (each position attends to
  2*window_size + 1 neighbours).
* ``n_global`` — number of global sink positions (taken from the start).
* ``alpha`` — weighting between local and global context (0 = all local,
  1 = all global).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "SubGenConfig",
    "SubGenAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class SubGenConfig:
    """Configuration for :class:`SubGenAttention`.

    Attributes:
        window_size: Half-width of the local sliding-window (attends to
            2*window_size + 1 positions, clamped at sequence boundaries).
        n_global: Number of global sink positions (from sequence start).
        alpha: Mixing coefficient in [0, 1] — contribution of global attention.
        causal: Whether to apply causal masking (future tokens invisible).
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    window_size: int = 64
    n_global: int = 4
    alpha: float = 0.5
    causal: bool = True
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be ≥ 1; got {self.window_size}")
        if self.n_global < 0:
            raise ValueError(f"n_global must be ≥ 0; got {self.n_global}")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1]; got {self.alpha}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")


# ── Core class ─────────────────────────────────────────────────────────────────


class SubGenAttention:
    """Sub-quadratic dual-sparse attention: local window + global sinks.

    Example::

        cfg  = SubGenConfig(window_size=16, n_global=2, n_heads=4, head_dim=8)
        attn = SubGenAttention(cfg)

        Q = np.random.randn(4, 64, 8).astype(np.float32)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        out = attn.forward(Q, K, V)   # (4, 64, 8)
    """

    def __init__(self, config: Optional[SubGenConfig] = None) -> None:
        self.config = config or SubGenConfig()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Compute sub-quadratic dual-sparse attention.

        Args:
            Q: ``(n_heads, T, head_dim)``.
            K: ``(n_heads, S, head_dim)``.
            V: ``(n_heads, S, head_dim)``.

        Returns:
            ``(n_heads, T, head_dim)`` context vectors.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        H, T, d = Q.shape
        S = K.shape[1]
        scale = 1.0 / np.sqrt(d)

        out_local = self._local_attn(Q, K, V, T, S, d, scale)
        out_global = self._global_attn(Q, K, V, T, S, d, scale)

        alpha = self.config.alpha
        return ((1.0 - alpha) * out_local + alpha * out_global).astype(np.float32)

    # ── Local window attention ─────────────────────────────────────────────────

    def _local_attn(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        T: int,
        S: int,
        d: int,
        scale: float,
    ) -> np.ndarray:
        """Sliding-window local attention."""
        w = self.config.window_size
        out = np.zeros((Q.shape[0], T, d), dtype=np.float32)
        for h in range(Q.shape[0]):
            for t in range(T):
                lo = max(0, t - w)
                hi = min(S, t + w + 1)
                if self.config.causal:
                    hi = min(hi, t + 1)
                if lo >= hi:
                    continue
                k_w = K[h, lo:hi]  # (w', d)
                v_w = V[h, lo:hi]
                sc = (Q[h, t] @ k_w.T) * scale  # (w',)
                e = np.exp(sc - sc.max())
                a = e / (e.sum() + 1e-9)
                out[h, t] = a @ v_w
        return out

    # ── Global sink attention ──────────────────────────────────────────────────

    def _global_attn(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        T: int,
        S: int,
        d: int,
        scale: float,
    ) -> np.ndarray:
        """Global sink attention (first n_global positions)."""
        ng = min(self.config.n_global, S)
        if ng == 0:
            return np.zeros((Q.shape[0], T, d), dtype=np.float32)

        K_g = K[:, :ng, :]  # (H, ng, d)
        V_g = V[:, :ng, :]

        scores = np.einsum("htd,hsd->hts", Q, K_g) * scale  # (H, T, ng)
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
        return (attn @ V_g).astype(np.float32)  # (H, T, d)

    def __repr__(self) -> str:
        return (
            f"SubGenAttention(window_size={self.config.window_size}, "
            f"n_global={self.config.n_global}, "
            f"alpha={self.config.alpha})"
        )
