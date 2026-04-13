"""Attention Sink rolling-window KV cache for streaming infinite-length generation.

Reference:
    Xiao et al., "Efficient Streaming Language Models with Attention Sinks",
    arXiv:2309.17453 (2023).

Key insight: LLMs assign disproportionate attention mass to the first few tokens
regardless of their semantic content.  Keeping those "sink" tokens alive in the
KV cache — alongside a rolling window of recent tokens — prevents attention
score collapse and enables theoretically unbounded generation without re-fill or
sliding-window re-computation.

This module is a pure-NumPy simulation of the SinkKVCache mechanism, suitable
for offline analysis, benchmarking, and integration testing.  Production
inference uses the MLX-JIT path in squish.streaming.chunked_prefill.
"""

from __future__ import annotations

__all__ = ["SinkConfig", "SinkKVCache", "SinkStats"]

from collections import deque
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SinkConfig:
    """Immutable configuration for :class:`SinkKVCache`.

    Attributes:
        n_sink_tokens: Number of initial tokens whose KV entries are never
            evicted.  Xiao et al. found 4 sink tokens sufficient for all
            tested models.
        window_size: Number of *recent* tokens kept in the rolling buffer
            (not counting sink tokens).  Total KV cache size ≤
            ``n_sink_tokens + window_size``.
        dtype: NumPy dtype string for all stored key/value arrays.  Must be
            a floating-point type; ``"float32"`` is the default.
    """

    n_sink_tokens: int = 4
    window_size: int = 256
    dtype: str = "float32"

    def __post_init__(self) -> None:
        if self.n_sink_tokens < 0:
            raise ValueError(
                f"n_sink_tokens must be ≥ 0; got {self.n_sink_tokens}"
            )
        if self.window_size < 1:
            raise ValueError(
                f"window_size must be ≥ 1; got {self.window_size}"
            )
        try:
            np.dtype(self.dtype)
        except TypeError as exc:
            raise ValueError(f"dtype {self.dtype!r} is not a valid NumPy dtype") from exc
        if not np.issubdtype(np.dtype(self.dtype), np.floating):
            raise ValueError(
                f"dtype must be a floating-point type; got {self.dtype!r}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class SinkStats:
    """Accumulated statistics for a :class:`SinkKVCache` instance.

    Attributes:
        n_tokens_seen: Total tokens added since last :meth:`SinkKVCache.reset`.
        n_evictions: Number of tokens evicted from the rolling window.
    """

    n_tokens_seen: int = 0
    n_evictions: int = 0

    @property
    def util_fraction(self) -> float:
        """Fraction of the rolling window currently occupied (0–1).

        Returns 0.0 before any tokens are added.  Value is based on
        ``n_tokens_seen`` relative to the window size, clamped to [0.0, 1.0].
        This is a snapshot metric only — callers needing exact occupancy
        should inspect :attr:`SinkKVCache.n_recent` directly.
        """
        return min(1.0, float(self.n_tokens_seen) / max(1, self.n_tokens_seen))

    @property
    def total_tokens_held(self) -> int:
        """Tokens currently alive in the cache (seen minus evicted)."""
        return max(0, self.n_tokens_seen - self.n_evictions)


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class SinkKVCache:
    """Rolling-window KV cache with permanent attention sinks.

    The cache is split into two regions:

    1. **Sink region** — the first ``config.n_sink_tokens`` added are stored in
       fixed-size arrays and are *never* evicted.  If fewer tokens than
       ``n_sink_tokens`` have been added, only those are stored.

    2. **Recent region** — a ``collections.deque`` with ``maxlen=window_size``
       holds the most recent tokens.  New tokens past the window limit evict
       the oldest entry automatically.

    :meth:`get_kv` returns a concatenation of sink + recent arrays so callers
    receive a single contiguous view for attention computation.

    Args:
        config: :class:`SinkConfig` controlling sink/window sizes and dtype.
        n_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
    """

    def __init__(
        self,
        config: SinkConfig,
        n_heads: int,
        head_dim: int,
    ) -> None:
        if n_heads < 1:
            raise ValueError(f"n_heads must be ≥ 1; got {n_heads}")
        if head_dim < 1:
            raise ValueError(f"head_dim must be ≥ 1; got {head_dim}")

        self._config = config
        self._n_heads = n_heads
        self._head_dim = head_dim
        self._dtype = np.dtype(config.dtype)

        # Sink storage: list of (n_heads, head_dim) arrays, length ≤ n_sink_tokens
        self._sink_k: list[np.ndarray] = []
        self._sink_v: list[np.ndarray] = []

        # Recent storage: deque auto-evicts oldest when full
        self._recent_k: deque[np.ndarray] = deque(maxlen=config.window_size)
        self._recent_v: deque[np.ndarray] = deque(maxlen=config.window_size)

        self._n_tokens_seen: int = 0
        self._n_evictions: int = 0

    # ------------------------------------------------------------------ add_kv

    def add_kv(self, key: np.ndarray, value: np.ndarray) -> None:
        """Add a single token's key and value tensors to the cache.

        Args:
            key: Float array of shape ``(n_heads, head_dim)``.
            value: Float array of shape ``(n_heads, head_dim)``.

        Raises:
            ValueError: If ``key`` or ``value`` have an unexpected shape.
        """
        config = self._config
        expected = (self._n_heads, self._head_dim)

        key = np.asarray(key, dtype=self._dtype)
        value = np.asarray(value, dtype=self._dtype)

        if key.shape != expected:
            raise ValueError(
                f"key shape {key.shape} does not match (n_heads={self._n_heads}, "
                f"head_dim={self._head_dim})"
            )
        if value.shape != expected:
            raise ValueError(
                f"value shape {value.shape} does not match (n_heads={self._n_heads}, "
                f"head_dim={self._head_dim})"
            )

        if len(self._sink_k) < config.n_sink_tokens:
            # Still filling the sink region
            self._sink_k.append(key.copy())
            self._sink_v.append(value.copy())
        else:
            # Sink full — push into rolling window
            if len(self._recent_k) == config.window_size:
                self._n_evictions += 1
            self._recent_k.append(key.copy())
            self._recent_v.append(value.copy())

        self._n_tokens_seen += 1

    # --------------------------------------------------------------- get_kv

    def get_kv(self) -> tuple[np.ndarray, np.ndarray]:
        """Return concatenated sink + recent key and value tensors.

        Returns:
            ``(keys, values)`` each of shape
            ``(n_sink_tokens + n_recent, n_heads, head_dim)``.
            Returns zero-length arrays on an empty cache.
        """
        all_k = self._sink_k + list(self._recent_k)
        all_v = self._sink_v + list(self._recent_v)

        if not all_k:
            empty = np.empty((0, self._n_heads, self._head_dim), dtype=self._dtype)
            return empty, empty.copy()

        keys = np.stack(all_k, axis=0)    # (T, n_heads, head_dim)
        values = np.stack(all_v, axis=0)
        return keys, values

    # ------------------------------------------------------------------ reset

    def reset(self) -> None:
        """Clear both sink and recent regions, resetting all counters."""
        self._sink_k.clear()
        self._sink_v.clear()
        self._recent_k.clear()
        self._recent_v.clear()
        self._n_tokens_seen = 0
        self._n_evictions = 0

    # ------------------------------------------------------------------ stats

    def get_stats(self) -> SinkStats:
        """Return a snapshot of current cache statistics."""
        return SinkStats(
            n_tokens_seen=self._n_tokens_seen,
            n_evictions=self._n_evictions,
        )

    # ------------------------------------------------------------ convenience

    @property
    def n_sink(self) -> int:
        """Number of sink tokens currently stored (≤ ``config.n_sink_tokens``)."""
        return len(self._sink_k)

    @property
    def n_recent(self) -> int:
        """Number of recent tokens currently in the rolling window."""
        return len(self._recent_k)

    @property
    def total_tokens(self) -> int:
        """Total tokens currently held (sink + recent)."""
        return self.n_sink + self.n_recent

    @property
    def config(self) -> SinkConfig:
        """The :class:`SinkConfig` this cache was built with."""
        return self._config

    def __repr__(self) -> str:
        return (
            f"SinkKVCache(n_sink={self.n_sink}/{self._config.n_sink_tokens}, "
            f"n_recent={self.n_recent}/{self._config.window_size}, "
            f"seen={self._n_tokens_seen}, evictions={self._n_evictions})"
        )
