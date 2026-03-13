# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/gqa.py

GQA — Grouped Query Attention KV cache and attention computation.

This module implements the GQA memory-reduction technique described in:

    Ainslie et al., "GQA: Training Generalised Multi-Query Transformer Models
    from Multi-Head Checkpoints", EMNLP 2023.
    https://arxiv.org/abs/2305.13245

In standard Multi-Head Attention (MHA), each query head has its own dedicated
K and V projection.  GQA groups query heads into ``n_kv_heads`` groups;
every head in a group shares one K/V pair.  Setting ``n_kv_heads=1`` recovers
Multi-Query Attention (MQA).

Memory savings vs MHA at inference time:

    KV memory reduction = n_kv_heads / n_q_heads

    Example (Llama 3.1-8B): n_q_heads=32, n_kv_heads=8 → 4× KV reduction.

Example usage::

    import numpy as np
    from squish.gqa import GQAConfig, GQACache, grouped_query_attention

    config = GQAConfig(n_q_heads=32, n_kv_heads=8, head_dim=128)
    cache = GQACache(config)

    # Decode loop — append one token at a time.
    for step in range(10):
        k = np.random.randn(config.n_kv_heads, config.head_dim).astype(np.float32)
        v = np.random.randn(config.n_kv_heads, config.head_dim).astype(np.float32)
        cache.append(k, v)

    keys, values = cache.get_kv()  # (n_kv_heads, seq_len, head_dim)

    q = np.random.randn(config.n_q_heads, 1, config.head_dim).astype(np.float32)
    out = grouped_query_attention(q, keys, values, config)  # (n_q_heads, 1, head_dim)
"""

from __future__ import annotations

__all__ = [
    "GQAConfig",
    "GQACache",
    "grouped_query_attention",
    "GQAStats",
]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GQAConfig:
    """Configuration for Grouped Query Attention.

    Attributes:
        n_q_heads:      Number of query heads.
        n_kv_heads:     Number of KV heads (must evenly divide ``n_q_heads``).
        head_dim:       Dimension of each head vector.
        max_seq_len:    Maximum sequence length the cache will hold.
        softmax_scale:  Pre-softmax scaling factor.  Defaults to
                        ``1 / sqrt(head_dim)`` when ``None``.
    """

    n_q_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 4096
    softmax_scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_q_heads < 1:
            raise ValueError(f"n_q_heads must be >= 1, got {self.n_q_heads}")
        if self.n_kv_heads < 1:
            raise ValueError(f"n_kv_heads must be >= 1, got {self.n_kv_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}")
        if self.n_q_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({self.n_q_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})"
            )
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        if self.softmax_scale <= 0.0:
            raise ValueError(
                f"softmax_scale must be > 0, got {self.softmax_scale}"
            )

    @property
    def group_size(self) -> int:
        """Number of query heads sharing each KV head."""
        return self.n_q_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class GQACache:
    """Fixed-length KV cache storing grouped-query key/value projections.

    Stores up to ``config.max_seq_len`` tokens in pre-allocated float32
    arrays of shape ``(n_kv_heads, max_seq_len, head_dim)``.  Attempting to
    append beyond ``max_seq_len`` raises ``OverflowError``.

    Args:
        config: A :class:`GQAConfig` instance.
    """

    def __init__(self, config: GQAConfig) -> None:
        self._cfg = config
        self._keys = np.zeros(
            (config.n_kv_heads, config.max_seq_len, config.head_dim),
            dtype=np.float32,
        )
        self._values = np.zeros(
            (config.n_kv_heads, config.max_seq_len, config.head_dim),
            dtype=np.float32,
        )
        self._pos: int = 0
        self._stats = GQAStats(
            n_q_heads=config.n_q_heads,
            n_kv_heads=config.n_kv_heads,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one token's KV projections for all KV heads.

        Args:
            key:   Shape ``(n_kv_heads, head_dim)``.
            value: Shape ``(n_kv_heads, head_dim)``.

        Raises:
            ValueError: if shapes do not match the config.
            OverflowError: if ``max_seq_len`` has already been reached.
        """
        cfg = self._cfg
        expected = (cfg.n_kv_heads, cfg.head_dim)
        if key.shape != expected:
            raise ValueError(
                f"key shape {key.shape} does not match expected {expected}"
            )
        if value.shape != expected:
            raise ValueError(
                f"value shape {value.shape} does not match expected {expected}"
            )
        if self._pos >= cfg.max_seq_len:
            raise OverflowError(
                f"GQACache is full: max_seq_len={cfg.max_seq_len} reached. "
                "Call reset() before appending further tokens."
            )

        self._keys[:, self._pos, :] = key
        self._values[:, self._pos, :] = value
        self._pos += 1

        self._stats.n_appends += 1
        self._stats.total_kv_heads_saved += cfg.n_q_heads - cfg.n_kv_heads

    def get_kv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return all stored KV tensors as dense views.

        Returns:
            ``(keys, values)`` each of shape
            ``(n_kv_heads, seq_len, head_dim)``.  The returned arrays are
            views into the internal buffer; copy if mutation is needed.
        """
        return (
            self._keys[:, : self._pos, :],
            self._values[:, : self._pos, :],
        )

    def reset(self) -> None:
        """Clear the cache and reset the sequence position to zero."""
        self._keys[:] = 0.0
        self._values[:] = 0.0
        self._pos = 0

    @property
    def seq_len(self) -> int:
        """Number of tokens currently stored in the cache."""
        return self._pos

    @property
    def stats(self) -> GQAStats:
        """Cumulative GQA statistics (updated in place)."""
        return self._stats


# ---------------------------------------------------------------------------
# Attention computation
# ---------------------------------------------------------------------------

def grouped_query_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    config: GQAConfig,
) -> np.ndarray:
    """Compute Grouped Query Attention over a full context.

    Expands the KV heads by repeating each KV head ``group_size`` times so
    that the standard scaled-dot-product attention can be applied uniformly
    across all query heads.

    Args:
        q:      Query tensor, shape ``(n_q_heads, seq_q, head_dim)``.
        k:      Key tensor,   shape ``(n_kv_heads, seq_kv, head_dim)``.
        v:      Value tensor, shape ``(n_kv_heads, seq_kv, head_dim)``.
        config: A :class:`GQAConfig` instance (provides scale and head counts).

    Returns:
        Output tensor of shape ``(n_q_heads, seq_q, head_dim)``.

    Raises:
        ValueError: if any input shape is inconsistent with *config*.
    """
    n_q_heads, seq_q, head_dim = q.shape
    n_kv_heads, seq_kv, kv_head_dim = k.shape

    if n_q_heads != config.n_q_heads:
        raise ValueError(
            f"q has {n_q_heads} heads but config specifies {config.n_q_heads}"
        )
    if n_kv_heads != config.n_kv_heads:
        raise ValueError(
            f"k has {n_kv_heads} KV heads but config specifies {config.n_kv_heads}"
        )
    if head_dim != config.head_dim or kv_head_dim != config.head_dim:
        raise ValueError(
            f"head_dim mismatch: q={head_dim}, k={kv_head_dim}, "
            f"config={config.head_dim}"
        )
    if v.shape != k.shape:
        raise ValueError(
            f"k and v must have the same shape; got k={k.shape}, v={v.shape}"
        )

    group_size = config.group_size

    # Expand KV heads to match query heads by repeating each KV head
    # group_size times along the head axis.
    # Shape: (n_kv_heads, seq_kv, head_dim) → (n_q_heads, seq_kv, head_dim)
    k_expanded = np.repeat(k, group_size, axis=0)
    v_expanded = np.repeat(v, group_size, axis=0)

    # Scaled dot-product attention.
    # scores: (n_q_heads, seq_q, seq_kv)
    scores = np.matmul(q, k_expanded.transpose(0, 2, 1)) * config.softmax_scale

    # Numerically stable softmax along the key dimension.
    scores -= scores.max(axis=-1, keepdims=True)
    scores_exp = np.exp(scores)
    attn_weights = scores_exp / scores_exp.sum(axis=-1, keepdims=True)

    # Weighted sum over values: (n_q_heads, seq_q, head_dim)
    return np.matmul(attn_weights, v_expanded).astype(q.dtype)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class GQAStats:
    """Cumulative statistics for a :class:`GQACache` instance.

    Attributes:
        n_q_heads:             Query head count from the config (for ratio).
        n_kv_heads:            KV head count from the config (for ratio).
        n_appends:             Total number of token appends performed.
        total_kv_heads_saved:  Cumulative KV head-slots saved versus MHA.
                               Each append saves ``n_q_heads - n_kv_heads``
                               head writes compared to a full MHA cache.
    """

    n_q_heads: int = 32
    n_kv_heads: int = 8
    n_appends: int = 0
    total_kv_heads_saved: int = 0

    @property
    def memory_ratio(self) -> float:
        """KV memory used relative to an equivalent MHA cache (0.0–1.0).

        Returns ``n_kv_heads / n_q_heads``.  A value of 0.25 means the GQA
        cache uses 25% of the memory that MHA would require.
        """
        if self.n_q_heads == 0:
            return 0.0
        return self.n_kv_heads / self.n_q_heads
