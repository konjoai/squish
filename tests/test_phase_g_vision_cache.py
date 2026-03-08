"""
tests/test_phase_g_vision_cache.py

Coverage tests for squish/vision_cache.py — Phase G3 (VisionPrefixCache).
"""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

import numpy as np
import pytest

from squish.vision_cache import VisionPrefixCache


def _make_image(value: int = 1, size: int = 100) -> bytes:
    """Return deterministic dummy image bytes."""
    return bytes([value % 256] * size)


def _dummy_encoder(img_bytes: bytes) -> np.ndarray:
    """Minimal vision encoder substitute that returns a fixed array."""
    return np.array([float(img_bytes[0])], dtype=np.float32)


class TestVisionPrefixCacheInit:
    def test_defaults(self):
        c = VisionPrefixCache()
        assert c._max_entries == 64
        assert c._bytes_per_entry == 4 * 1024 * 1024
        assert c.cache_size() if hasattr(c, "cache_size") else len(c._cache) == 0

    def _len(self, c):
        return len(c._cache)

    def test_custom_max_entries(self):
        c = VisionPrefixCache(max_entries=8)
        assert c._max_entries == 8

    def test_custom_bytes_per_entry(self):
        c = VisionPrefixCache(bytes_per_entry_estimate=1024)
        assert c._bytes_per_entry == 1024


class TestVisionPrefixCacheHashing:
    def test_hash_image_sha256(self):
        c = VisionPrefixCache()
        img = b"hello"
        expected = hashlib.sha256(img).hexdigest()
        assert c._hash_image(img) == expected

    def test_hash_image_different_inputs(self):
        c = VisionPrefixCache()
        assert c._hash_image(b"a") != c._hash_image(b"b")

    def test_hash_image_deterministic(self):
        c = VisionPrefixCache()
        img = b"consistent"
        assert c._hash_image(img) == c._hash_image(img)


class TestVisionPrefixCacheMissHit:
    def test_miss_calls_encoder(self):
        c = VisionPrefixCache()
        encoder = MagicMock(return_value=np.array([1.0]))
        img = _make_image(42)
        result = c.get_or_encode(img, encoder)
        encoder.assert_called_once_with(img)
        assert result is not None

    def test_hit_skips_encoder(self):
        c = VisionPrefixCache()
        encoder = MagicMock(return_value=np.array([7.0]))
        img = _make_image(7)
        c.get_or_encode(img, encoder)
        c.get_or_encode(img, encoder)  # second call → cache hit
        assert encoder.call_count == 1  # encoder called only once

    def test_miss_increments_miss_counter(self):
        c = VisionPrefixCache()
        img = _make_image(1)
        c.get_or_encode(img, _dummy_encoder)
        assert c._misses == 1
        assert c._hits == 0

    def test_hit_increments_hit_counter(self):
        c = VisionPrefixCache()
        img = _make_image(2)
        c.get_or_encode(img, _dummy_encoder)
        c.get_or_encode(img, _dummy_encoder)
        assert c._hits == 1
        assert c._misses == 1

    def test_different_images_separate_cache_entries(self):
        c = VisionPrefixCache()
        encoder = MagicMock(side_effect=lambda b: np.array([float(b[0])]))
        c.get_or_encode(_make_image(1), encoder)
        c.get_or_encode(_make_image(2), encoder)
        assert encoder.call_count == 2
        assert len(c._cache) == 2

    def test_cached_encoding_returned(self):
        """The exact encoding from the first call is returned on subsequent hits."""
        c = VisionPrefixCache()
        encoding = np.array([99.0])
        encoder = MagicMock(return_value=encoding)
        img = _make_image(99)
        r1 = c.get_or_encode(img, encoder)
        r2 = c.get_or_encode(img, encoder)
        assert (r1 == r2).all()


class TestVisionPrefixCacheEviction:
    def test_evicts_when_over_capacity(self):
        c = VisionPrefixCache(max_entries=2)
        for i in range(3):
            c.get_or_encode(_make_image(i), _dummy_encoder)
        assert len(c._cache) == 2

    def test_lru_order_evicts_oldest(self):
        """After filling to max, the oldest unchanged entry is evicted."""
        c = VisionPrefixCache(max_entries=2)
        img0 = _make_image(0)
        img1 = _make_image(1)
        img2 = _make_image(2)
        c.get_or_encode(img0, _dummy_encoder)
        c.get_or_encode(img1, _dummy_encoder)
        # Access img0 again to make it more recent
        c.get_or_encode(img0, _dummy_encoder)
        # Add img2 → evicts img1 (least recently used)
        c.get_or_encode(img2, _dummy_encoder)
        h1 = c._hash_image(img1)
        assert h1 not in c._cache


class TestVisionPrefixCacheInvalidate:
    def test_invalidate_existing(self):
        c = VisionPrefixCache()
        img = _make_image(5)
        c.get_or_encode(img, _dummy_encoder)
        removed = c.invalidate(img)
        assert removed is True
        assert len(c._cache) == 0

    def test_invalidate_nonexistent(self):
        c = VisionPrefixCache()
        removed = c.invalidate(_make_image(99))
        assert removed is False


class TestVisionPrefixCacheClearLRU:
    def test_clear_lru_reduces_to_target(self):
        """clear_lru evicts entries until estimated size fits target_size_mb."""
        # bytes_per_entry=1MB, 4 entries → 4MB; target=2MB → keep 2 entries
        c = VisionPrefixCache(max_entries=100, bytes_per_entry_estimate=1024 * 1024)
        for i in range(4):
            c.get_or_encode(_make_image(i), _dummy_encoder)
        assert len(c._cache) == 4
        evicted = c.clear_lru(target_size_mb=2)
        assert evicted == 2
        assert len(c._cache) == 2

    def test_clear_lru_no_eviction_needed(self):
        """If cache is already within target, nothing is evicted."""
        c = VisionPrefixCache(max_entries=100, bytes_per_entry_estimate=1024 * 1024)
        for i in range(2):
            c.get_or_encode(_make_image(i), _dummy_encoder)
        evicted = c.clear_lru(target_size_mb=10)
        assert evicted == 0
        assert len(c._cache) == 2

    def test_clear_lru_zero_target(self):
        """target_size_mb=0 evicts all entries."""
        c = VisionPrefixCache(max_entries=100, bytes_per_entry_estimate=1024 * 1024)
        for i in range(3):
            c.get_or_encode(_make_image(i), _dummy_encoder)
        evicted = c.clear_lru(target_size_mb=0)
        assert evicted == 3
        assert len(c._cache) == 0


class TestVisionPrefixCacheStats:
    def test_initial_stats(self):
        c = VisionPrefixCache()
        s = c.stats()
        assert s["hits"] == 0
        assert s["misses"] == 0
        assert s["total_images"] == 0
        assert s["cache_entries"] == 0
        assert s["hit_rate"] == 0.0
        assert s["estimated_size_mb"] == 0.0

    def test_stats_after_misses(self):
        c = VisionPrefixCache()
        for i in range(3):
            c.get_or_encode(_make_image(i), _dummy_encoder)
        s = c.stats()
        assert s["misses"] == 3
        assert s["hits"] == 0
        assert s["hit_rate"] == 0.0

    def test_stats_after_hits(self):
        c = VisionPrefixCache()
        img = _make_image(1)
        c.get_or_encode(img, _dummy_encoder)
        c.get_or_encode(img, _dummy_encoder)
        c.get_or_encode(img, _dummy_encoder)
        s = c.stats()
        assert s["hits"] == 2
        assert s["misses"] == 1
        assert s["hit_rate"] == pytest.approx(2 / 3)

    def test_stats_estimated_size(self):
        bpe = 2 * 1024 * 1024  # 2 MiB per entry
        c = VisionPrefixCache(bytes_per_entry_estimate=bpe)
        for i in range(3):
            c.get_or_encode(_make_image(i), _dummy_encoder)
        s = c.stats()
        assert s["estimated_size_mb"] == pytest.approx(6.0)

    def test_stats_total_images(self):
        c = VisionPrefixCache()
        img = _make_image(1)
        c.get_or_encode(img, _dummy_encoder)
        c.get_or_encode(img, _dummy_encoder)
        c.get_or_encode(_make_image(2), _dummy_encoder)
        s = c.stats()
        assert s["total_images"] == 3


class TestVisionPrefixCacheClear:
    def test_clear_empties_cache_and_resets_counters(self):
        c = VisionPrefixCache()
        img = _make_image(1)
        c.get_or_encode(img, _dummy_encoder)
        c.get_or_encode(img, _dummy_encoder)
        c.clear()
        assert len(c._cache) == 0
        assert c._hits == 0
        assert c._misses == 0

    def test_clear_then_reuse(self):
        """After clear(), the cache operates normally again."""
        c = VisionPrefixCache()
        img = _make_image(5)
        encoder = MagicMock(return_value=np.array([5.0]))
        c.get_or_encode(img, encoder)
        c.clear()
        c.get_or_encode(img, encoder)
        assert encoder.call_count == 2  # must re-encode after clear
