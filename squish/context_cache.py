#!/usr/bin/env python3
"""
squish/context_cache.py

ContextCache — Persistent cross-session context caching with TTL.

Many users submit the same context repeatedly — a long system prompt, a
shared document, or a fixed few-shot preamble.  Rather than re-encoding
those tokens on every request, a persistent context cache stores the
pre-computed KV tensors keyed by a deterministic hash of the input token
IDs, expiring entries after a configurable TTL.

The cache is bounded by ``max_entries``.  When that limit is reached, the
oldest entry by creation time is evicted to make room (FIFO on overflow).
TTL-expired entries are not automatically purged on every access; callers
should invoke :meth:`PersistentContextCache.evict_expired` periodically.

Example usage::

    import numpy as np
    from squish.context_cache import PersistentContextCache, CacheEntry

    cache = PersistentContextCache(max_entries=256, default_ttl_s=300.0)

    tokens = [1, 2, 3, 4, 5]
    kv = np.zeros((32, len(tokens), 128), dtype=np.float32)

    entry_id = cache.put(tokens, kv)
    result = cache.get(tokens)
    assert result is not None

    expired = cache.evict_expired()
    print(f"entry_id={entry_id!r}, n_entries={cache.n_entries}")
    print(cache.stats)
"""

from __future__ import annotations

__all__ = [
    "CacheEntry",
    "PersistentContextCache",
    "ContextCacheStats",
]

import hashlib
import time
import uuid
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hash_tokens(tokens: list[int]) -> int:
    """Compute a stable 128-bit integer hash of a token-ID list.

    Uses MD5 over the little-endian int64 byte representation so that
    the hash is reproducible across interpreter restarts (unlike Python's
    built-in ``hash()`` which is randomised by PYTHONHASHSEED).
    """
    data = np.array(tokens, dtype=np.int64).tobytes()
    digest = hashlib.md5(data).digest()  # noqa: S324 — non-crypto use
    return int.from_bytes(digest, byteorder="big")


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class CacheEntry:
    """A single cached context entry.

    Attributes:
        entry_id:     Unique string identifier assigned at insertion.
        token_hash:   Stable integer hash of the source token IDs.
        kv_data:      Pre-computed KV tensor (arbitrary shape, float32).
        created_at:   Unix timestamp when this entry was inserted.
        ttl_s:        Time-to-live in seconds.  Entry expires after
                      ``created_at + ttl_s``.
        access_count: Number of successful cache hits for this entry.
    """

    entry_id: str
    token_hash: int
    kv_data: np.ndarray
    created_at: float
    ttl_s: float = 300.0
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("entry_id must be a non-empty string")
        if self.ttl_s <= 0.0:
            raise ValueError(f"ttl_s must be > 0, got {self.ttl_s}")
        if not isinstance(self.kv_data, np.ndarray):
            raise TypeError(
                f"kv_data must be a numpy ndarray, got "
                f"{type(self.kv_data).__name__!r}"
            )

    @property
    def is_expired(self) -> bool:
        """``True`` if the entry's TTL has elapsed as of ``time.time()``."""
        return time.time() > self.created_at + self.ttl_s


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class ContextCacheStats:
    """Cumulative statistics for :class:`PersistentContextCache`.

    Attributes:
        total_puts:   Total calls to :meth:`PersistentContextCache.put`.
        total_gets:   Total calls to :meth:`PersistentContextCache.get`.
        hits:         Gets that returned a valid (non-expired) entry.
        misses:       Gets that returned ``None`` (miss or expired).
        evictions:    Entries removed by TTL expiry or capacity overflow.
    """

    total_puts: int = 0
    total_gets: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        """Fraction of get calls satisfied by a valid cache entry (0.0–1.0).

        Returns 0.0 when no gets have been issued yet.
        """
        if self.total_gets == 0:
            return 0.0
        return self.hits / self.total_gets


# ---------------------------------------------------------------------------
# Persistent context cache
# ---------------------------------------------------------------------------

class PersistentContextCache:
    """Persistent cross-session context cache with TTL eviction.

    Stores pre-computed KV tensors keyed by a deterministic hash of the
    input token IDs.  Each entry carries a TTL; expired entries are not
    served and can be evicted in bulk via :meth:`evict_expired`.

    When the number of live entries would exceed ``max_entries`` on a new
    :meth:`put`, the oldest entry (by ``created_at``) is evicted first.

    Args:
        max_entries:    Maximum number of live entries (>= 1).
        default_ttl_s:  Default TTL in seconds used when ``put`` is called
                        without an explicit ``ttl_s`` override (> 0).

    Raises:
        ValueError: if ``max_entries`` or ``default_ttl_s`` are out of range.
    """

    def __init__(
        self,
        max_entries: int = 256,
        default_ttl_s: float = 300.0,
    ) -> None:
        if max_entries < 1:
            raise ValueError(
                f"max_entries must be >= 1, got {max_entries}"
            )
        if default_ttl_s <= 0.0:
            raise ValueError(
                f"default_ttl_s must be > 0, got {default_ttl_s}"
            )
        self._max_entries = max_entries
        self._default_ttl_s = default_ttl_s
        # token_hash -> CacheEntry for fast O(1) lookup.
        self._entries: dict[int, CacheEntry] = {}
        self._stats = ContextCacheStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        tokens: list[int],
        kv_data: np.ndarray,
        ttl_s: Optional[float] = None,
    ) -> str:
        """Store a pre-computed KV tensor for *tokens*.

        If an entry for the same token sequence already exists it is
        overwritten.  When the cache is at capacity, the oldest live entry
        is evicted first.

        Args:
            tokens:  Source token IDs (non-empty list of ints).
            kv_data: Pre-computed KV tensor (numpy ndarray, any shape).
            ttl_s:   Optional TTL override in seconds.  Uses
                     ``default_ttl_s`` when ``None``.

        Returns:
            A unique string ``entry_id`` for the new entry.

        Raises:
            ValueError: if ``tokens`` is empty or ``ttl_s`` is not positive.
            TypeError:  if ``kv_data`` is not a numpy ndarray.
        """
        if not tokens:
            raise ValueError("tokens must be a non-empty list")
        if not isinstance(kv_data, np.ndarray):
            raise TypeError(
                f"kv_data must be a numpy ndarray, got "
                f"{type(kv_data).__name__!r}"
            )
        effective_ttl = ttl_s if ttl_s is not None else self._default_ttl_s
        if effective_ttl <= 0.0:
            raise ValueError(f"ttl_s must be > 0, got {effective_ttl}")

        token_hash = _hash_tokens(tokens)
        entry_id = str(uuid.uuid4())
        now = time.time()

        # Evict capacity overflow (oldest by created_at) before inserting.
        if token_hash not in self._entries:
            while len(self._entries) >= self._max_entries:
                self._evict_oldest()

        entry = CacheEntry(
            entry_id=entry_id,
            token_hash=token_hash,
            kv_data=kv_data,
            created_at=now,
            ttl_s=effective_ttl,
            access_count=0,
        )
        self._entries[token_hash] = entry
        self._stats.total_puts += 1
        return entry_id

    def get(self, tokens: list[int]) -> Optional[np.ndarray]:
        """Retrieve the cached KV tensor for *tokens*, if present and live.

        Args:
            tokens: Source token IDs to look up.

        Returns:
            The cached ``kv_data`` ndarray, or ``None`` on a miss or if
            the matching entry has expired (the expired entry is removed).

        Raises:
            ValueError: if ``tokens`` is empty.
        """
        if not tokens:
            raise ValueError("tokens must be a non-empty list")

        self._stats.total_gets += 1
        token_hash = _hash_tokens(tokens)
        entry = self._entries.get(token_hash)

        if entry is None:
            self._stats.misses += 1
            return None

        if entry.is_expired:
            del self._entries[token_hash]
            self._stats.evictions += 1
            self._stats.misses += 1
            return None

        entry.access_count += 1
        self._stats.hits += 1
        return entry.kv_data

    def evict_expired(self) -> int:
        """Remove all TTL-expired entries from the cache.

        Returns:
            The number of entries removed.
        """
        expired_keys = [
            k for k, entry in self._entries.items() if entry.is_expired
        ]
        for k in expired_keys:
            del self._entries[k]
        self._stats.evictions += len(expired_keys)
        return len(expired_keys)

    @property
    def n_entries(self) -> int:
        """Number of entries currently in the cache (including not-yet-expired)."""
        return len(self._entries)

    @property
    def stats(self) -> ContextCacheStats:
        """Cumulative cache statistics (updated in place)."""
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evict_oldest(self) -> None:
        """Evict the entry with the smallest ``created_at`` timestamp."""
        if not self._entries:
            return
        oldest_key = min(
            self._entries, key=lambda k: self._entries[k].created_at
        )
        del self._entries[oldest_key]
        self._stats.evictions += 1
