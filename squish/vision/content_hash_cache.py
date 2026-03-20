"""
ContentHashImageCache — Content-Hash-Based Image Prefix Cache.

Key idea (from vllm-mlx research): hash image bytes before firing vision encoding.
On a cache hit, reuse stored KV tensors from the prior encoding pass.
The vllm-mlx paper reports 28× speedup on repeated image queries.

VisionPrefixCache already exists in the codebase; this module adds the
deduplication layer that sits in front of it — a simple hash → KV lookup table.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ContentHashCacheConfig:
    """Configuration for the content-hash image KV cache."""
    max_entries: int = 256          # maximum cached images
    hash_fn: str = "sha256"         # "sha256" only (perceptual hash future work)
    ttl_seconds: Optional[float] = None   # None → entries live forever

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be >= 1, got {self.max_entries}")
        if self.hash_fn not in ("sha256",):
            raise ValueError(f"hash_fn must be 'sha256', got '{self.hash_fn}'")


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

@dataclass
class _CacheEntry:
    kv_cache: np.ndarray    # cached KV tensor(s) from vision encoder
    inserted_at: float      # wall-clock time of insertion


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

@dataclass
class ContentHashCacheStats:
    """Runtime statistics."""
    lookups: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    stores: int = 0

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups > 0 else 0.0

    @property
    def bytes_cached(self) -> int:
        """Reported externally by the cache instance."""
        return 0

    def __repr__(self) -> str:
        return (
            f"ContentHashCacheStats(lookups={self.lookups}, "
            f"hit_rate={self.hit_rate:.1%}, evictions={self.evictions})"
        )


class ContentHashImageCache:
    """LRU cache mapping SHA-256(image_bytes) → vision-encoder KV tensors.

    Usage::

        cache = ContentHashImageCache(ContentHashCacheConfig())
        kv = cache.lookup(image_bytes)
        if kv is None:
            kv = vision_encoder(image_bytes)
            cache.store(image_bytes, kv)
    """

    def __init__(self, config: ContentHashCacheConfig) -> None:
        self.config = config
        self.stats = ContentHashCacheStats()
        # OrderedDict used as LRU — most-recently used at the back
        self._store: "OrderedDict[str, _CacheEntry]" = OrderedDict()

    # ------------------------------------------------------------------
    # Hash
    # ------------------------------------------------------------------

    def hash_image(self, image_bytes: bytes) -> str:
        """Compute deterministic hash of raw image bytes."""
        return hashlib.sha256(image_bytes).hexdigest()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Return cached KV tensor if present and not expired; else None."""
        self.stats.lookups += 1
        key = self.hash_image(image_bytes)
        entry = self._store.get(key)

        if entry is None:
            self.stats.misses += 1
            return None

        # TTL check
        if (
            self.config.ttl_seconds is not None
            and time.monotonic() - entry.inserted_at > self.config.ttl_seconds
        ):
            del self._store[key]
            self.stats.evictions += 1
            self.stats.misses += 1
            return None

        # LRU update — move to back
        self._store.move_to_end(key)
        self.stats.hits += 1
        return entry.kv_cache

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(self, image_bytes: bytes, kv_cache: np.ndarray) -> None:
        """Cache KV tensor for image_bytes, evicting eldest entry if full."""
        key = self.hash_image(image_bytes)

        # evict oldest if at capacity
        while len(self._store) >= self.config.max_entries:
            self._store.popitem(last=False)
            self.stats.evictions += 1

        self._store[key] = _CacheEntry(
            kv_cache=kv_cache,
            inserted_at=time.monotonic(),
        )
        self._store.move_to_end(key)
        self.stats.stores += 1

    # ------------------------------------------------------------------
    # Eviction / management
    # ------------------------------------------------------------------

    def evict_lru(self) -> int:
        """Manually evict the least-recently-used entry.

        Returns:
            Bytes freed (0 if cache was empty).
        """
        if not self._store:
            return 0
        _, entry = self._store.popitem(last=False)
        self.stats.evictions += 1
        return int(entry.kv_cache.nbytes)

    def evict_expired(self) -> int:
        """Remove all TTL-expired entries.

        Returns:
            Number of entries removed.
        """
        if self.config.ttl_seconds is None:
            return 0
        now = time.monotonic()
        expired = [
            k for k, e in self._store.items()
            if now - e.inserted_at > self.config.ttl_seconds
        ]
        for k in expired:
            del self._store[k]
            self.stats.evictions += 1
        return len(expired)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of entries currently in cache."""
        return len(self._store)

    @property
    def bytes_cached(self) -> int:
        """Total bytes of KV data currently held."""
        return sum(e.kv_cache.nbytes for e in self._store.values())

    def __repr__(self) -> str:
        return (
            f"ContentHashImageCache(size={self.size}/{self.config.max_entries}, "
            f"bytes={self.bytes_cached:,}, {self.stats})"
        )
