#!/usr/bin/env python3
"""
squish/vision_cache.py

Content-based hash cache for vision encoder outputs.

Vision encoders (CLIP, SigLIP, etc.) are expensive: a single 336×336 image
takes ~80 ms on an M3 Pro.  When the *same* image appears repeatedly
(product photos, repeated diagrams, system prompts with a logo) encoding it
every time wastes cycles.

:class:`VisionPrefixCache` de-duplicates encoder calls via SHA-256 of the raw
image bytes.  Identical bytes → identical encoding → return cached result.

Notes
─────
- Cache entries are keyed by ``SHA-256(image_bytes)``.
- Entry eviction is LRU (oldest-first) controlled by ``max_entries``, or
  explicit via :meth:`clear_lru`.
- ``get_or_encode`` calls ``vision_encoder(image_bytes)`` on cache miss.
  The encoder callable can be any Python callable that accepts ``bytes`` and
  returns an encoding (``mx.array``, numpy array, etc.).
- Thread safety: external synchronisation is the caller's responsibility.
  The cache itself uses a plain ``OrderedDict`` and is not lock-protected.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any, Callable


class VisionPrefixCache:
    """LRU cache for vision encoder outputs, keyed by image content hash.

    Parameters
    ----------
    max_entries:
        Maximum number of encoded images to hold in cache before LRU eviction.
        Defaults to ``64``.
    bytes_per_entry_estimate:
        Rough size estimate per cache entry in bytes, used by :meth:`clear_lru`
        to calculate how many entries to evict.  For a 336×336 SigLIP patch
        embedding, ~4 MB is a reasonable upper bound.  Defaults to
        ``4 * 1024 * 1024`` (4 MiB).
    """

    def __init__(
        self,
        max_entries: int = 64,
        bytes_per_entry_estimate: int = 4 * 1024 * 1024,
    ) -> None:
        self._max_entries = max_entries
        self._bytes_per_entry = bytes_per_entry_estimate
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._hits = 0
        self._misses = 0

    # ── Hashing ───────────────────────────────────────────────────────────────

    def _hash_image(self, image_bytes: bytes) -> str:
        """Return the SHA-256 hex digest of *image_bytes*."""
        return hashlib.sha256(image_bytes).hexdigest()

    # ── Core cache operations ─────────────────────────────────────────────────

    def get_or_encode(
        self,
        image_bytes: bytes,
        vision_encoder: Callable[[bytes], Any],
    ) -> Any:
        """Return the cached encoding for *image_bytes*, or encode and cache it.

        Parameters
        ----------
        image_bytes:
            Raw image bytes (JPEG, PNG, etc.).
        vision_encoder:
            Callable that accepts ``bytes`` and returns an encoding array.
            Called only on cache misses.

        Returns
        -------
        Any
            The encoder output, either from cache or freshly computed.
        """
        key = self._hash_image(image_bytes)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]

        self._misses += 1
        encoding = vision_encoder(image_bytes)
        self._cache[key] = encoding
        self._evict_if_needed()
        return encoding

    def _evict_if_needed(self) -> None:
        """Remove oldest entries until the cache is within ``max_entries``."""
        while len(self._cache) > self._max_entries:
            self._cache.popitem(last=False)

    def invalidate(self, image_bytes: bytes) -> bool:
        """Remove the cache entry for *image_bytes*.

        Returns ``True`` if an entry was removed, ``False`` if not present.
        """
        key = self._hash_image(image_bytes)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    # ── LRU size-based eviction ───────────────────────────────────────────────

    def clear_lru(self, target_size_mb: int) -> int:
        """Evict oldest entries until the estimated cache size is ≤ *target_size_mb*.

        Uses ``bytes_per_entry_estimate`` to calculate how many entries to
        retain.  Returns the number of entries evicted.

        Parameters
        ----------
        target_size_mb:
            Target cache size upper bound, in mebibytes.

        Returns
        -------
        int
            Number of entries that were evicted.
        """
        target_bytes = target_size_mb * 1024 * 1024
        target_entries = max(0, target_bytes // self._bytes_per_entry)
        evicted = 0
        while len(self._cache) > target_entries:
            self._cache.popitem(last=False)
            evicted += 1
        return evicted

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return cache statistics.

        Keys: ``hits``, ``misses``, ``total_images``, ``cache_entries``,
              ``hit_rate``, ``estimated_size_mb``.
        """
        total = self._hits + self._misses
        estimated_bytes = len(self._cache) * self._bytes_per_entry
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total_images": total,
            "cache_entries": len(self._cache),
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "estimated_size_mb": round(estimated_bytes / 1024 / 1024, 2),
        }

    def clear(self) -> None:
        """Remove all entries and reset hit/miss counters."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
