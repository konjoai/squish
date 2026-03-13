# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/sliding_window_attn.py

SlidingWindowAttn — Mistral/Gemma sliding window KV cache.

This module implements the sliding-window attention (SWA) mechanism introduced
in Mistral and refined in Gemma:

    Jiang et al., "Mistral 7B", 2023.
    https://arxiv.org/abs/2310.06825

    Gemma Team, "Gemma: Open Models Based on Gemini Research and Technology",
    Google DeepMind 2024.
    https://arxiv.org/abs/2403.08295

SWA restricts each token's attention receptive field to the most recent
``window_size`` tokens.  This caps KV cache memory at O(window_size) rather
than O(seq_len), enabling arbitrarily long generation without growing memory.

The cache is implemented as a ring buffer: appends are O(1) and the oldest
token is evicted automatically once the window is full.

Example usage::

    import numpy as np
    from squish.sliding_window_attn import SWAConfig, SlidingWindowKVCache
    from squish.sliding_window_attn import sliding_window_attention

    config = SWAConfig(window_size=512, n_heads=32, head_dim=128, kv_n_heads=8)
    cache = SlidingWindowKVCache(config)

    for step in range(1000):
        k = np.random.randn(config.kv_n_heads, config.head_dim).astype(np.float32)
        v = np.random.randn(config.kv_n_heads, config.head_dim).astype(np.float32)
        cache.append(k, v)

    q = np.random.randn(config.n_heads, config.head_dim).astype(np.float32)
    out = sliding_window_attention(q, cache, config)  # (n_heads, head_dim)
    print(f"fill={cache.fill_ratio:.1%}, evicted={cache.stats.tokens_evicted}")
"""

from __future__ import annotations

__all__ = [
    "SWAConfig",
    "SlidingWindowKVCache",
    "sliding_window_attention",
    "SWAStats",
]

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SWAConfig:
    """Configuration for sliding-window KV cache and attention.

    Attributes:
        window_size:  Maximum number of recent tokens retained in the cache.
        n_heads:      Number of query attention heads.
        head_dim:     Dimension of each attention head vector.
        kv_n_heads:   Number of KV heads.  Defaults to ``n_heads`` when
                      ``None`` (MHA).  Set lower for GQA.
    """

    window_size: int = 4096
    n_heads: int = 32
    head_dim: int = 128
    kv_n_heads: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kv_n_heads is None:
            self.kv_n_heads = self.n_heads
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self.window_size}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.head_dim < 1:
            raise ValueError(f"head_dim must be >= 1, got {self.head_dim}")
        if self.kv_n_heads < 1:
            raise ValueError(f"kv_n_heads must be >= 1, got {self.kv_n_heads}")
        if self.n_heads % self.kv_n_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"kv_n_heads ({self.kv_n_heads})"
            )


# ---------------------------------------------------------------------------
# Ring-buffer KV cache
# ---------------------------------------------------------------------------

class SlidingWindowKVCache:
    """Ring-buffer KV cache that retains only the last ``window_size`` tokens.

    Internal layout
    ~~~~~~~~~~~~~~~
    Two arrays of shape ``(kv_n_heads, window_size, head_dim)`` serve as the
    backing store.  A ``_head`` pointer advances modulo ``window_size`` after
    each write, giving O(1) appends.  When the buffer is full the oldest slot
    is silently overwritten (evicted).

    ``get_kv()`` reconstructs the temporally ordered view (oldest → newest)
    by rotating the ring buffer to a contiguous slice at read time.

    Args:
        config: A :class:`SWAConfig` instance.
    """

    def __init__(self, config: SWAConfig) -> None:
        self._cfg = config
        w = config.window_size
        h = config.kv_n_heads
        d = config.head_dim
        self._key_buf = np.zeros((h, w, d), dtype=np.float32)
        self._val_buf = np.zeros((h, w, d), dtype=np.float32)
        # Index of the next write slot in the ring buffer.
        self._head: int = 0
        # Total tokens appended (monotonically increasing; not capped).
        self._total_tokens: int = 0
        self._stats = SWAStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append(self, key: np.ndarray, value: np.ndarray) -> None:
        """Append one token's KV vectors to the ring buffer.

        When the buffer is full the oldest token is evicted and the eviction
        counter in ``stats`` is incremented.

        Args:
            key:   Shape ``(kv_n_heads, head_dim)``.
            value: Shape ``(kv_n_heads, head_dim)``.

        Raises:
            ValueError: if shapes do not match the config.
        """
        cfg = self._cfg
        expected = (cfg.kv_n_heads, cfg.head_dim)
        if key.shape != expected:
            raise ValueError(
                f"key shape {key.shape} does not match expected {expected}"
            )
        if value.shape != expected:
            raise ValueError(
                f"value shape {value.shape} does not match expected {expected}"
            )

        # If already full this write evicts the slot at _head.
        if self._total_tokens >= cfg.window_size:
            self._stats.tokens_evicted += 1

        self._key_buf[:, self._head, :] = key
        self._val_buf[:, self._head, :] = value
        self._head = (self._head + 1) % cfg.window_size
        self._total_tokens += 1
        self._stats.total_tokens_seen += 1

    def get_kv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the stored KV tensors in temporal order (oldest to newest).

        Returns:
            ``(keys, values)`` each of shape
            ``(kv_n_heads, window_used, head_dim)``.

        Notes:
            When the buffer is not yet full the returned slice covers only the
            filled portion ``[:window_used]``.  When full, the ring is rotated
            so position 0 is the oldest surviving token.
        """
        used = self.window_used
        cfg = self._cfg

        if self._total_tokens <= cfg.window_size:
            # Buffer not yet full: valid data lives at indices [0, used).
            return (
                self._key_buf[:, :used, :].copy(),
                self._val_buf[:, :used, :].copy(),
            )

        # Buffer is full: oldest token is at _head, newest at _head-1 (mod w).
        # Build an index array that rotates the ring into chronological order.
        indices = (self._head + np.arange(cfg.window_size)) % cfg.window_size
        return (
            self._key_buf[:, indices, :],
            self._val_buf[:, indices, :],
        )

    def reset(self) -> None:
        """Clear the cache and reset all counters."""
        self._key_buf[:] = 0.0
        self._val_buf[:] = 0.0
        self._head = 0
        self._total_tokens = 0

    @property
    def seq_len(self) -> int:
        """Total number of tokens appended (not capped by window_size)."""
        return self._total_tokens

    @property
    def window_used(self) -> int:
        """Number of valid token slots in the current window."""
        return min(self._total_tokens, self._cfg.window_size)

    @property
    def fill_ratio(self) -> float:
        """Fraction of the window currently filled (0.0–1.0)."""
        return self.window_used / self._cfg.window_size

    @property
    def stats(self) -> SWAStats:
        """Cumulative sliding-window statistics (updated in place)."""
        return self._stats


# ---------------------------------------------------------------------------
# Attention computation (single decode step)
# ---------------------------------------------------------------------------

def sliding_window_attention(
    q: np.ndarray,
    cache: SlidingWindowKVCache,
    config: SWAConfig,
) -> np.ndarray:
    """Compute sliding-window attention for a single decode step.

    Attends over the ``window_used`` tokens currently stored in *cache*.
    Supports GQA: when ``config.kv_n_heads < config.n_heads`` the KV
    tensors are expanded by repeating each KV head ``group_size`` times.

    Args:
        q:      Query for the current token, shape ``(n_q_heads, head_dim)``.
        cache:  A :class:`SlidingWindowKVCache` instance holding context.
        config: The matching :class:`SWAConfig` instance.

    Returns:
        Attention output, shape ``(n_q_heads, head_dim)``, same dtype as *q*.

    Raises:
        ValueError: if ``q`` shape is inconsistent with *config*.
    """
    if q.shape != (config.n_heads, config.head_dim):
        raise ValueError(
            f"q shape {q.shape} does not match "
            f"expected ({config.n_heads}, {config.head_dim})"
        )

    keys, values = cache.get_kv()  # (kv_n_heads, window_used, head_dim)
    window_used = keys.shape[1]

    if window_used == 0:
        return np.zeros((config.n_heads, config.head_dim), dtype=q.dtype)

    group_size = config.n_heads // config.kv_n_heads

    # Expand KV heads to match query heads (GQA → MHA-equivalent view).
    if group_size > 1:
        keys = np.repeat(keys, group_size, axis=0)    # (n_q_heads, w, head_dim)
        values = np.repeat(values, group_size, axis=0)

    scale = 1.0 / math.sqrt(config.head_dim)

    # scores: (n_q_heads, window_used)
    # q[:, np.newaxis, :] @ keys.T → (n_q_heads, 1, window_used) → squeezed
    q_3d = q[:, np.newaxis, :]                            # (n_q_heads, 1, head_dim)
    scores = np.matmul(q_3d, keys.transpose(0, 2, 1))    # (n_q_heads, 1, window_used)
    scores = scores.squeeze(1) * scale                    # (n_q_heads, window_used)

    # Numerically stable softmax.
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)        # (n_q_heads, window_used)

    # Weighted sum over the value context.
    # weights[:, np.newaxis, :] @ values → (n_q_heads, 1, head_dim) → squeezed
    weights_3d = weights[:, np.newaxis, :]                # (n_q_heads, 1, window_used)
    output = np.matmul(weights_3d, values).squeeze(1)     # (n_q_heads, head_dim)

    return output.astype(q.dtype)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class SWAStats:
    """Cumulative statistics for a :class:`SlidingWindowKVCache` instance.

    Attributes:
        total_tokens_seen:  Total tokens appended since construction or last
                            ``reset()``.
        tokens_evicted:     Tokens that were overwritten because the window
                            was already full when they were appended.
    """

    total_tokens_seen: int = 0
    tokens_evicted: int = 0

    @property
    def eviction_rate(self) -> float:
        """Fraction of all tokens that were evicted from the window.

        Returns ``tokens_evicted / total_tokens_seen``, or 0.0 when no tokens
        have been seen yet.
        """
        if self.total_tokens_seen == 0:
            return 0.0
        return self.tokens_evicted / self.total_tokens_seen
