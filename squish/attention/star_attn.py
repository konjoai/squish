"""squish/attention/star_attn.py

StarAttention — Block-Partitioned Star-Topology Local + Anchor Attention
(Acharya et al., NeurIPS 2024 / arXiv:2411.17116).

Reference
---------
"Star Attention: Efficient LLM Inference over Long Sequences." Acharya et al.,
NeurIPS 2024 (arXiv:2411.17116).

Algorithm
---------
Star Attention partitions the sequence into non-overlapping blocks.  The first
block is designated the **anchor** block.  Each block's queries compute:

1. **Local attention** within their own block (keys/values from the same block).
2. **Anchor attention** against the anchor block (keys/values from block 0).

The two attention outputs are combined via numerically stable log-sum-exp
renormalisation (weighted average using the two softmax normalisers).

This avoids global O(T²) attention while preserving long-range anchoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "StarAttentionConfig",
    "StarAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class StarAttentionConfig:
    """Configuration for :class:`StarAttention`.

    Attributes:
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
        block_size: Token block size for star-topology partitioning.
        causal: If True, apply causal masking within local blocks.
    """

    n_heads: int = 8
    head_dim: int = 64
    block_size: int = 256
    causal: bool = True

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be ≥ 1; got {self.block_size}")


# ── StarAttention ─────────────────────────────────────────────────────────────


class StarAttention:
    """Block-partitioned star topology attention.

    Example::

        cfg = StarAttentionConfig(n_heads=2, head_dim=8, block_size=4, causal=True)
        model = StarAttention(cfg)
        rng = np.random.default_rng(0)
        T = 12
        Q = rng.standard_normal((2, T, 8)).astype(np.float32)
        K = rng.standard_normal((2, T, 8)).astype(np.float32)
        V = rng.standard_normal((2, T, 8)).astype(np.float32)
        out = model.forward(Q, K, V)   # shape (2, T, 8)
    """

    def __init__(self, config: Optional[StarAttentionConfig] = None) -> None:
        self.config = config or StarAttentionConfig()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Compute star attention over input QKV.

        Args:
            Q: ``(H, T, d)`` query tensor.
            K: ``(H, T, d)`` key tensor.
            V: ``(H, T, d)`` value tensor.

        Returns:
            Output tensor ``(H, T, d)``.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        H, T, d = Q.shape
        bs = self.config.block_size
        scale = 1.0 / np.sqrt(max(d, 1))

        # Pad sequence to multiple of block_size.
        n_blocks = (T + bs - 1) // bs
        T_pad = n_blocks * bs
        pad_len = T_pad - T

        def _pad(x: np.ndarray) -> np.ndarray:
            if pad_len == 0:
                return x
            return np.concatenate(
                [x, np.zeros((H, pad_len, d), dtype=x.dtype)], axis=1
            )

        Q_p, K_p, V_p = _pad(Q), _pad(K), _pad(V)

        # Anchor block: first block.
        K_anchor = K_p[:, :bs, :]  # (H, bs, d)
        V_anchor = V_p[:, :bs, :]

        out_p = np.zeros((H, T_pad, d), dtype=np.float32)

        for b in range(n_blocks):
            s, e = b * bs, (b + 1) * bs
            Q_b = Q_p[:, s:e, :]  # (H, bs, d)
            K_b = K_p[:, s:e, :]
            V_b = V_p[:, s:e, :]

            # ── Local attention ──────────────────────────────────────────────
            A_local, denom_local = _masked_scaled_attn(
                Q_b, K_b, V_b, scale, causal=self.config.causal
            )

            # ── Anchor attention ─────────────────────────────────────────────
            # No causal masking for cross-block anchor (anchor precedes block).
            A_anchor, denom_anchor = _masked_scaled_attn(
                Q_b, K_anchor, V_anchor, scale, causal=False
            )

            # ── Combine via log-sum-exp renormalisation ───────────────────────
            denom_sum = denom_local + denom_anchor + 1e-9
            out_b = (A_local * denom_local + A_anchor * denom_anchor) / denom_sum
            out_p[:, s:e, :] = out_b

        return out_p[:, :T, :]

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"StarAttention(n_heads={cfg.n_heads}, head_dim={cfg.head_dim}, "
            f"block_size={cfg.block_size}, causal={cfg.causal})"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _masked_scaled_attn(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float,
    causal: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scaled dot-product attention; return (output, sum_exp_denom).

    Returns:
        out:   (H, T_q, d) — un-normalised weighted sum (multiplied by denom).
        denom: (H, T_q, 1) — sum of exp scores for renormalisation.
    """
    # scores: (H, T_q, T_k)
    scores = np.einsum("htd,hsd->hts", Q, K) * scale
    if causal:
        T_q, T_k = scores.shape[1], scores.shape[2]
        mask = np.triu(np.ones((T_q, T_k), dtype=bool), k=1)
        scores[:, mask] = -1e9
    scores -= scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    denom = exp_scores.sum(axis=-1, keepdims=True)  # (H, T_q, 1)
    attn = exp_scores / (denom + 1e-9)
    out = np.einsum("hts,hsd->htd", attn, V)
    return out * denom, denom
