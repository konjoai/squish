"""squish/attention/razor_attn.py

RazorAttention — Efficient KV Cache Compression Through Retrieval Heads.

Reference
---------
He et al. "RazorAttention: Efficient KV Cache Compression Through Retrieval
Heads." NeurIPS 2024 (arXiv:2407.15891).

Algorithm
---------
Large-language-model attention heads naturally split into two populations:

* **Retrieval heads** — a small fraction (~10–20 %) that attend to a wide
  spread of context positions.  These must see the full KV cache to preserve
  factual accuracy.
* **Non-retrieval heads** — the majority, which predominantly attend to the
  most recent tokens.  Their contribution can be approximated by keeping only
  a *summary KV*: the very first token (attention sink) and the most recent
  token.

The module provides:

1. A calibration step that classifies each head from a small set of prompt
   examples by measuring attention entropy — high-entropy heads are retrieval
   heads.
2. A forward pass that routes each head to the full KV buffer or the
   2-token summary KV based on its classification.

Key properties
--------------
* NumPy-only; no GPU dependency.
* ``n_summary_tokens`` — number of summary tokens to keep for non-retrieval
  heads (default 2: sink + most-recent).
* ``entropy_threshold`` — heads with mean normalised attention entropy above
  this value are classified as retrieval heads.
* Calibration is idempotent — calling it multiple times updates running stats.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

__all__ = [
    "RazorAttentionConfig",
    "RazorHeadType",
    "RazorAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RazorAttentionConfig:
    """Configuration for :class:`RazorAttention`.

    Attributes:
        n_heads: Total number of attention heads.
        head_dim: Dimension per head.
        n_summary_tokens: Tokens kept for non-retrieval heads (sink + recent).
        entropy_threshold: Normalised entropy threshold for head classification.
        causal: Whether to apply a causal mask during calibration.
    """

    n_heads: int = 8
    head_dim: int = 64
    n_summary_tokens: int = 2
    entropy_threshold: float = 0.5
    causal: bool = True

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")
        if self.n_summary_tokens < 1:
            raise ValueError(
                f"n_summary_tokens must be ≥ 1; got {self.n_summary_tokens}"
            )
        if not (0.0 <= self.entropy_threshold <= 1.0):
            raise ValueError(
                f"entropy_threshold must be in [0, 1]; got {self.entropy_threshold}"
            )


# ── Head type enum ─────────────────────────────────────────────────────────────

class RazorHeadType:
    """Constants for head classification results."""

    RETRIEVAL = "retrieval"
    NON_RETRIEVAL = "non_retrieval"
    UNCLASSIFIED = "unclassified"


# ── Core class ─────────────────────────────────────────────────────────────────


class RazorAttention:
    """Retrieval-head-aware KV cache compression.

    Example::

        cfg    = RazorAttentionConfig(n_heads=4, head_dim=32)
        razor  = RazorAttention(cfg)

        # Calibrate on a few prompts: Q/K/V each (n_heads, T, head_dim)
        Q = np.random.randn(4, 128, 32).astype(np.float32)
        K = np.random.randn(4, 128, 32).astype(np.float32)
        V = np.random.randn(4, 128, 32).astype(np.float32)
        razor.calibrate(Q, K, V)

        # Forward: only retrieval heads see full KV
        out = razor.forward(Q, K, V)  # (n_heads, T, head_dim)
    """

    def __init__(self, config: Optional[RazorAttentionConfig] = None) -> None:
        self.config = config or RazorAttentionConfig()
        H = self.config.n_heads
        self._entropy_acc: np.ndarray = np.zeros(H, dtype=np.float64)
        self._n_calibration_calls: int = 0
        self._head_types: List[str] = [RazorHeadType.UNCLASSIFIED] * H

    # ── Calibration ───────────────────────────────────────────────────────────

    def calibrate(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> None:
        """Compute per-head attention entropy and update running averages.

        Args:
            Q: ``(n_heads, T, head_dim)`` query tensor.
            K: ``(n_heads, T, head_dim)`` key tensor.
            V: ``(n_heads, T, head_dim)`` value tensor (unused — entropy only).
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        H, T, d = Q.shape
        if H != self.config.n_heads:
            raise ValueError(
                f"Expected {self.config.n_heads} heads; got {H}"
            )

        scale = 1.0 / np.sqrt(d)
        scores = np.einsum("htd,hsd->hts", Q, K) * scale  # (H, T, T)

        if self.config.causal:
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            scores = np.where(mask[np.newaxis], -1e30, scores)

        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)  # (H, T, T)

        # Normalised entropy per head (averaged over query positions)
        log_T = np.log(float(T) + 1e-9)
        ent = -(attn * np.log(attn + 1e-9)).sum(axis=-1)  # (H, T)
        norm_ent = (ent / log_T).mean(axis=-1)  # (H,)

        # Running mean (equal-weight average across calls)
        n = self._n_calibration_calls
        self._entropy_acc = (self._entropy_acc * n + norm_ent) / (n + 1)
        self._n_calibration_calls += 1

        # Update head types
        thresh = self.config.entropy_threshold
        for h in range(H):
            if self._entropy_acc[h] >= thresh:
                self._head_types[h] = RazorHeadType.RETRIEVAL
            else:
                self._head_types[h] = RazorHeadType.NON_RETRIEVAL

    def head_types(self) -> List[str]:
        """Return current head classification list."""
        return list(self._head_types)

    def retrieval_head_indices(self) -> List[int]:
        """Return indices of retrieval heads."""
        return [h for h, t in enumerate(self._head_types) if t == RazorHeadType.RETRIEVAL]

    def non_retrieval_head_indices(self) -> List[int]:
        """Return indices of non-retrieval heads."""
        return [h for h, t in enumerate(self._head_types) if t == RazorHeadType.NON_RETRIEVAL]

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Compressed attention forward pass.

        Retrieval heads attend over the full K/V.  Non-retrieval heads attend
        over the ``n_summary_tokens`` summary KV (first + last tokens).

        Args:
            Q: ``(n_heads, T, head_dim)``.
            K: ``(n_heads, S, head_dim)`` (may differ from T for decode steps).
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
        outs = np.zeros((H, T, d), dtype=np.float32)

        n_sum = min(self.config.n_summary_tokens, S)
        # Summary indices: first tokens + last tokens
        if n_sum >= S:
            summary_idx = np.arange(S)
        else:
            n_front = max(1, n_sum // 2)
            n_back = n_sum - n_front
            summary_idx = np.concatenate([
                np.arange(n_front),
                np.arange(S - n_back, S),
            ])

        for h in range(H):
            if self._head_types[h] == RazorHeadType.NON_RETRIEVAL:
                k_h = K[h][summary_idx]  # (n_sum, d)
                v_h = V[h][summary_idx]
            else:
                k_h = K[h]
                v_h = V[h]

            scores = Q[h] @ k_h.T * scale  # (T, n_kv)

            if self.config.causal and self._head_types[h] != RazorHeadType.NON_RETRIEVAL:
                t_ids = np.arange(T)
                s_ids = np.arange(k_h.shape[0])
                mask = s_ids[np.newaxis, :] > t_ids[:, np.newaxis]
                scores = np.where(mask, -1e30, scores)

            e = np.exp(scores - scores.max(axis=-1, keepdims=True))
            attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
            outs[h] = attn @ v_h

        return outs

    def __repr__(self) -> str:
        ret = sum(1 for t in self._head_types if t == RazorHeadType.RETRIEVAL)
        return (
            f"RazorAttention(n_heads={self.config.n_heads}, "
            f"retrieval_heads={ret}, "
            f"n_calibration_calls={self._n_calibration_calls})"
        )
