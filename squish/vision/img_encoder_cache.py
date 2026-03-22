"""ImageEncoderCache: LRU cache for vision encoder output tensors.

Encoding the same image (e.g. system thumbnail, repeated frame, share icon) is
wasted compute.  By keying the cache on the SHA-256 of the raw image bytes,
subsequent requests get the cached token array in O(1) instead of O(encoder
latency).

The implementation is intentionally in-process and single-threaded to remain
dependency-free.  For production use, the caller can layer distributed caching
(Redis, memcached) on top by wrapping :meth:`encode_or_cached`.

This complements :class:`~squish.vision.content_hash_cache.ContentHashCache`
which caches at the text-token level; here we cache one step earlier — the
vision-encoder output before projection into the LLM embedding space.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

__all__ = [
    "ImageEncoderCacheConfig",
    "CacheEntry",
    "ImageEncoderCache",
]


@dataclass
class ImageEncoderCacheConfig:
    """Configuration for :class:`ImageEncoderCache`.

    Attributes:
        max_entries: Maximum number of images to retain in cache.
        token_dim: Expected token embedding dimension (used for validation).
        seed: Unused; for API consistency.
    """

    max_entries: int = 1000
    token_dim: int = 256
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError(f"max_entries must be ≥ 1, got {self.max_entries}")
        if self.token_dim < 1:
            raise ValueError(f"token_dim must be ≥ 1, got {self.token_dim}")


@dataclass
class CacheEntry:
    """A single cached encoder output.

    Attributes:
        image_hash: SHA-256 hex digest of the source image bytes.
        tokens: Encoder output array ``(n_tokens, token_dim)``.
        timestamp: Unix timestamp of last access (initialised at insertion).
        hit_count: Number of cache hits for this entry.
    """

    image_hash: str
    tokens: np.ndarray
    timestamp: float = field(default_factory=time.monotonic)
    hit_count: int = 0


class ImageEncoderCache:
    """Thread-unsafe LRU cache of vision encoder token arrays.

    Example::

        cfg = ImageEncoderCacheConfig(max_entries=256, token_dim=512)
        cache = ImageEncoderCache(cfg)

        def my_encoder(image_hash: str) -> np.ndarray:
            ...  # call CLIP / SigLIP here
            return tokens  # shape (n_tokens, 512)

        tokens = cache.encode_or_cached(image_sha256_hex, my_encoder)

    Stats::

        print(cache.stats())
        # {'hits': 3, 'misses': 1, 'entries': 1, 'hit_rate': 0.75}
    """

    def __init__(self, config: ImageEncoderCacheConfig) -> None:
        self.config = config
        self._store: Dict[str, CacheEntry] = {}
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def get(self, image_hash: str) -> Optional[np.ndarray]:
        """Return cached tokens for ``image_hash``, or ``None`` on miss."""
        entry = self._store.get(image_hash)
        if entry is None:
            self._misses += 1
            return None
        entry.hit_count += 1
        entry.timestamp = time.monotonic()
        self._hits += 1
        return entry.tokens

    def put(self, image_hash: str, tokens: np.ndarray) -> None:
        """Insert or replace the cache entry for ``image_hash``.

        Evicts the least-recently-used entry if at capacity.
        """
        tokens = np.asarray(tokens, dtype=np.float32)
        if image_hash in self._store:
            self._store[image_hash].tokens = tokens
            self._store[image_hash].timestamp = time.monotonic()
            return
        if len(self._store) >= self.config.max_entries:
            self._evict_lru()
        self._store[image_hash] = CacheEntry(
            image_hash=image_hash,
            tokens=tokens,
            timestamp=time.monotonic(),
        )

    def encode_or_cached(
        self,
        image_hash: str,
        encoder_fn: Callable[[str], np.ndarray],
    ) -> np.ndarray:
        """Return cached tokens if available, otherwise call ``encoder_fn``.

        Args:
            image_hash: SHA-256 hex digest identifying the image.
            encoder_fn: Callable ``(image_hash) → tokens`` used on cache miss.

        Returns:
            Token array ``(n_tokens, token_dim)``.
        """
        cached = self.get(image_hash)
        if cached is not None:
            return cached
        tokens = np.asarray(encoder_fn(image_hash), dtype=np.float32)
        self.put(image_hash, tokens)
        return tokens

    def stats(self) -> dict:
        """Return a snapshot of cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._store),
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }

    def clear(self) -> None:
        """Remove all entries and reset statistics."""
        self._store.clear()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Remove the entry with the oldest ``timestamp``."""
        if not self._store:
            return
        lru_key = min(self._store, key=lambda h: self._store[h].timestamp)
        del self._store[lru_key]

    # ------------------------------------------------------------------
    # Container protocol helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, image_hash: str) -> bool:
        return image_hash in self._store
