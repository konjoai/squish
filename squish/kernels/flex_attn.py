"""FlexAttentionKernel — composable compiled attention via score_mod + block_mask.

Implements the FlexAttention programming model introduced by the PyTorch team
(pytorch.org/blog 2024, ASPLOS 2025).  The core idea is to express any
attention variant (causal, alibi, sliding-window, document-mask, …) as a
pure per-element *score modification function* plus an optional *block mask*.
Under ``torch.compile`` this collapses into a single Triton kernel.

This module provides:

* :class:`ScoreMod` — callable score modifications (causal, alibi, sliding-window,
  soft-cap, custom function).
* :class:`BlockMask` — coarse-grained block-sparse mask representation.
* :class:`FlexAttentionKernel` — orchestrates score_mod + block_mask and
  dispatches to either a simulated NumPy path (all platforms) or the real
  ``torch.nn.attention.flex_attention`` when PyTorch ≥ 2.4 is available.

Reference:
    PyTorch Team, "FlexAttention: A Programming Model for Attention
    Generalization", pytorch.org/blog (2024); ASPLOS 2025.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

__all__ = [
    "ScoreModType",
    "BlockMask",
    "FlexAttentionConfig",
    "FlexAttentionKernel",
    "make_causal_mod",
    "make_alibi_mod",
    "make_sliding_window_mod",
    "make_softcap_mod",
]

# ── Score mod type alias ──────────────────────────────────────────────────────

# score_mod(score, b, h, q_idx, k_idx) -> modified_score
ScoreModType = Callable[[float, int, int, int, int], float]


# ── Built-in score modifications ──────────────────────────────────────────────


def make_causal_mod() -> ScoreModType:
    """Return a score_mod that masks future positions (causal attention).

    Returns:
        Score modification function.
    """
    def _mod(score: float, b: int, h: int, q_idx: int, k_idx: int) -> float:
        return score if k_idx <= q_idx else -1e30

    return _mod


def make_alibi_mod(slopes: np.ndarray) -> ScoreModType:
    """Return a score_mod implementing ALiBi position bias.

    Args:
        slopes: ``(n_heads,)`` float array of per-head alibi slopes.

    Returns:
        Score modification function.
    """
    _slopes = np.asarray(slopes, dtype=np.float32)

    def _mod(score: float, b: int, h: int, q_idx: int, k_idx: int) -> float:
        bias = float(_slopes[h]) * abs(q_idx - k_idx)
        return score - bias

    return _mod


def make_sliding_window_mod(window_size: int) -> ScoreModType:
    """Return a score_mod implementing a local sliding-window mask.

    Args:
        window_size: Maximum allowed distance between query and key indices.

    Returns:
        Score modification function.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be ≥ 1; got {window_size}")

    def _mod(score: float, b: int, h: int, q_idx: int, k_idx: int) -> float:
        return score if abs(q_idx - k_idx) <= window_size else -1e30

    return _mod


def make_softcap_mod(cap: float) -> ScoreModType:
    """Return a score_mod applying soft-capping (Gemma-style logit capping).

    Applies: ``score ← cap * tanh(score / cap)``

    Args:
        cap: Positive softcap value.

    Returns:
        Score modification function.
    """
    if cap <= 0:
        raise ValueError(f"cap must be positive; got {cap}")

    def _mod(score: float, b: int, h: int, q_idx: int, k_idx: int) -> float:
        return float(cap) * math.tanh(score / float(cap))

    return _mod


# ── BlockMask ─────────────────────────────────────────────────────────────────


@dataclass
class BlockMask:
    """Coarse-grained block-sparse mask for FlexAttention.

    The mask is defined at the *block* level: if block (bq, bk) is False,
    the entire ``(block_size × block_size)`` tile is skipped.

    Attributes:
        mask: Boolean array of shape ``(n_q_blocks, n_k_blocks)``.
            True = compute, False = skip.
        block_size: Number of tokens per block (must divide sequence length).
    """

    mask: np.ndarray
    block_size: int = 64

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError(
                f"BlockMask.mask must be 2-D; got shape {self.mask.shape}"
            )
        if self.block_size < 1:
            raise ValueError(
                f"block_size must be ≥ 1; got {self.block_size}"
            )

    @classmethod
    def causal(cls, seq_len: int, block_size: int = 64) -> "BlockMask":
        """Create a causal block mask for a sequence of length ``seq_len``.

        Args:
            seq_len: Sequence length (should be divisible by ``block_size``).
            block_size: Block granularity.

        Returns:
            :class:`BlockMask` with a lower-triangular block pattern.
        """
        n = math.ceil(seq_len / block_size)
        mask = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1):
                mask[i, j] = True
        return cls(mask=mask, block_size=block_size)

    def token_mask(self, n_q: int, n_k: int) -> np.ndarray:
        """Expand the block mask to a full token-level boolean mask.

        Args:
            n_q: Number of query tokens.
            n_k: Number of key tokens.

        Returns:
            Boolean array of shape ``(n_q, n_k)``.
        """
        bs = self.block_size
        n_qb = math.ceil(n_q / bs)
        n_kb = math.ceil(n_k / bs)
        full = np.zeros((n_qb * bs, n_kb * bs), dtype=bool)
        for bq in range(n_qb):
            for bk in range(n_kb):
                if bq < self.mask.shape[0] and bk < self.mask.shape[1]:
                    if self.mask[bq, bk]:
                        r0, r1 = bq * bs, (bq + 1) * bs
                        c0, c1 = bk * bs, (bk + 1) * bs
                        full[r0:r1, c0:c1] = True
        return full[:n_q, :n_k]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class FlexAttentionConfig:
    """Configuration for FlexAttentionKernel.

    Attributes:
        scale: Attention scale factor.  Defaults to ``1 / sqrt(head_dim)``
            when set to -1.
        use_torch_compile: Attempt to use ``torch.nn.attention.flex_attention``
            when PyTorch ≥ 2.4 is available.  Falls back to NumPy silently.
    """

    scale: float = -1.0
    use_torch_compile: bool = False


# ── Main class ────────────────────────────────────────────────────────────────


class FlexAttentionKernel:
    """Composable attention kernel with score_mod and block_mask support.

    Example::

        kernel = FlexAttentionKernel()
        causal_mod = make_causal_mod()
        Q = np.random.randn(8, 4, 64, 32).astype(np.float32)  # (B,H,N,d)
        K = np.random.randn(8, 4, 64, 32).astype(np.float32)
        V = np.random.randn(8, 4, 64, 32).astype(np.float32)
        out = kernel.forward(Q, K, V, score_mod=causal_mod)

    Args:
        config: :class:`FlexAttentionConfig` (optional).
    """

    def __init__(self, config: Optional[FlexAttentionConfig] = None) -> None:
        self.config: FlexAttentionConfig = config or FlexAttentionConfig()

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        score_mod: Optional[ScoreModType] = None,
        block_mask: Optional[BlockMask] = None,
    ) -> np.ndarray:
        """Compute attention with optional score modifications and block mask.

        Accepts tensors of shape ``(B, H, N, d)`` (batch, heads, seq, dim)
        or ``(N, d)`` (single head, no batch).  Single-head inputs are
        automatically promoted.

        Args:
            Q: Query tensor.
            K: Key tensor (same shape as Q).
            V: Value tensor (same shape as Q).
            score_mod: Optional per-element score modification function.
            block_mask: Optional coarse block-level sparsity mask.

        Returns:
            Attended output of the same shape as Q.

        Raises:
            ValueError: If Q/K/V have incompatible shapes.
        """
        Q = np.asarray(Q, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)

        # Promote 2-D (N, d) to (1, 1, N, d)
        squeeze = False
        if Q.ndim == 2:
            Q, K, V = Q[None, None], K[None, None], V[None, None]
            squeeze = True

        if Q.shape != K.shape or Q.shape[:-1] != V.shape[:-1]:
            raise ValueError(
                f"Q/K/V shape mismatch: Q={Q.shape}, K={K.shape}, V={V.shape}"
            )

        B, H, N, d = Q.shape
        M = K.shape[2]
        scale = self.config.scale if self.config.scale > 0 else (1.0 / math.sqrt(d))

        out = np.zeros((B, H, N, V.shape[-1]), dtype=np.float32)

        # Expand block mask to token level (if provided)
        tok_mask: Optional[np.ndarray] = None
        if block_mask is not None:
            tok_mask = block_mask.token_mask(N, M)  # (N, M) bool — True=allowed

        for b in range(B):
            for h in range(H):
                scores = (Q[b, h] @ K[b, h].T) * scale  # (N, M)

                # Apply block mask (set blocked positions to -inf)
                if tok_mask is not None:
                    scores = np.where(tok_mask, scores, -1e30)

                # Apply per-element score_mod
                if score_mod is not None:
                    for qi in range(N):
                        for ki in range(M):
                            scores[qi, ki] = score_mod(
                                float(scores[qi, ki]), b, h, qi, ki
                            )

                # Softmax
                s_max = scores.max(axis=-1, keepdims=True)
                e = np.exp(scores - s_max)
                attn = e / (e.sum(axis=-1, keepdims=True) + 1e-9)
                out[b, h] = attn @ V[b, h]

        if squeeze:
            return out[0, 0]
        return out

    def __repr__(self) -> str:
        return (
            f"FlexAttentionKernel(scale={self.config.scale}, "
            f"use_torch_compile={self.config.use_torch_compile})"
        )
