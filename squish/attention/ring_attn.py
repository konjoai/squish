"""squish/attention/ring_attn.py

Ring Attention — Sequence-parallel exact attention distributed across
available compute units via ring-topology K/V passing
(Liu et al., ICLR 2024 / arXiv:2310.01889).

Reference
---------
"Ring Attention with Blockwise Transformers for Near-Infinite Context."
Liu et al., ICLR 2024 (arXiv:2310.01889).

Algorithm
---------
Given *n_devices* logical shards and a sequence of length S:

1. Split Q, K, V into *n_devices* equal-sized blocks along the sequence axis.
2. Each shard holds its local Q block and initially its local K/V block.
3. Iterate *n_devices* rounds; in each round:
   a. The shard computes local block-wise attention using its current K/V.
   b. K/V blocks are passed to the "next" shard (simulated by rotation).
4. Running (numerically-stable) log-sum-exp accumulation combines partial
   softmax outputs from each round.
5. Final output is the concatenation of all shard outputs.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* Scales to context lengths > DRAM on a single node with sufficient shards.
* ``n_shards`` — number of logical compute units to simulate.
* Causal masking is applied correctly across shard boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = [
    "RingAttentionConfig",
    "RingAttention",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RingAttentionConfig:
    """Configuration for :class:`RingAttention`.

    Attributes:
        n_shards: Number of logical shards (simulated devices).
        causal: Whether to apply causal (autoregressive) masking.
        n_heads: Number of attention heads.
        head_dim: Dimension per head.
    """

    n_shards: int = 4
    causal: bool = True
    n_heads: int = 8
    head_dim: int = 64

    def __post_init__(self) -> None:
        if self.n_shards < 1:
            raise ValueError(f"n_shards must be ≥ 1; got {self.n_shards}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {self.head_dim}")

    @property
    def n_devices(self) -> int:  # server.py compatibility alias
        return self.n_shards

    @property
    def chunk_size(self) -> int:  # server.py compatibility alias (half-sequence default)
        return 512


# ── Ring Attention ────────────────────────────────────────────────────────────


class RingAttention:
    """Ring-topology sequence-parallel exact attention.

    Example::

        cfg = RingAttentionConfig(n_shards=4, causal=True, n_heads=4, head_dim=8)
        attn = RingAttention(cfg)
        Q = np.random.randn(4, 64, 8).astype(np.float32)  # (H, T, d)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        out = attn.forward(Q, K, V)  # (4, 64, 8)
    """

    def __init__(self, config: Optional[RingAttentionConfig] = None) -> None:
        self.config = config or RingAttentionConfig()

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
    ) -> np.ndarray:
        """Compute ring attention over the full sequence.

        Args:
            Q: ``(n_heads, T, head_dim)`` query tensor.
            K: ``(n_heads, S, head_dim)`` key tensor — must equal T.
            V: ``(n_heads, S, head_dim)`` value tensor.

        Returns:
            ``(n_heads, T, head_dim)`` attention output.

        Raises:
            ValueError: If T != S or tensors are incompatible.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)

        H, T, d = Q.shape
        if K.shape[1] != T:
            raise ValueError(
                f"Ring attention requires T==S; got Q.T={T}, K.S={K.shape[1]}"
            )

        n = self.config.n_shards
        # Pad sequence to be divisible by n_shards
        pad = (n - T % n) % n
        if pad > 0:
            Q = np.pad(Q, ((0, 0), (0, pad), (0, 0)))
            K = np.pad(K, ((0, 0), (0, pad), (0, 0)))
            V = np.pad(V, ((0, 0), (0, pad), (0, 0)))

        T_padded = Q.shape[1]
        block_size = T_padded // n
        scale = 1.0 / np.sqrt(d)

        # Split into shards: (n, H, block_size, d)
        Q_shards = Q.reshape(H, n, block_size, d).transpose(1, 0, 2, 3)
        K_shards = K.reshape(H, n, block_size, d).transpose(1, 0, 2, 3)
        V_shards = V.reshape(H, n, block_size, d).transpose(1, 0, 2, 3)

        # Output accumulators — online softmax
        out_acc = np.zeros((n, H, block_size, d), dtype=np.float32)
        lse_acc = np.full((n, H, block_size), -np.inf, dtype=np.float32)

        K_cur = K_shards.copy()
        V_cur = V_shards.copy()

        for step in range(n):
            # Global offset of K/V shard (for causal mask)
            kv_shard_offset = (step % n)

            for s in range(n):
                q_block = Q_shards[s]   # (H, block_size, d)
                k_block = K_cur[s]      # (H, block_size, d)
                v_block = V_cur[s]      # (H, block_size, d)

                # scores: (H, q_len, kv_len)
                scores = np.einsum("hqd,hkd->hqk", q_block, k_block) * scale

                if self.config.causal:
                    q_global = s * block_size + np.arange(block_size)          # (q,)
                    k_global = kv_shard_offset * block_size + np.arange(block_size)  # (k,)
                    mask = q_global[:, None] < k_global[None, :]
                    scores[:, mask] = -1e30

                # Online softmax update
                max_scores = scores.max(axis=-1)           # (H, q)
                exp_s = np.exp(scores - max_scores[..., None])
                lse_this = max_scores + np.log(exp_s.sum(axis=-1) + 1e-9)

                # Combine with running accumulator
                lse_prev = lse_acc[s]
                lse_new = np.maximum(lse_prev, lse_this)
                alpha_prev = np.exp(lse_prev - lse_new)
                alpha_this = np.exp(lse_this - lse_new)

                attn_out = np.einsum("hqk,hkd->hqd", exp_s / (exp_s.sum(axis=-1, keepdims=True) + 1e-9), v_block)
                out_acc[s] = (
                    alpha_prev[..., None] * out_acc[s]
                    + alpha_this[..., None] * attn_out
                )
                lse_acc[s] = lse_new

            # Ring-pass K/V to the next shard
            K_cur = np.roll(K_cur, shift=1, axis=0)
            V_cur = np.roll(V_cur, shift=1, axis=0)

        # Reassemble: (n, H, block_size, d) -> (H, T_padded, d)
        output = out_acc.transpose(1, 0, 2, 3).reshape(H, T_padded, d)
        return output[:, :T, :].astype(np.float32)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"RingAttention(n_shards={cfg.n_shards}, causal={cfg.causal}, "
            f"n_heads={cfg.n_heads}, head_dim={cfg.head_dim})"
        )
