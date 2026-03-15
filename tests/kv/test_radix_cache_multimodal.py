"""
tests/kv/test_radix_cache_multimodal.py

Phase 1 — Content-hash multimodal prefix cache tests.

Covers all new functionality added to RadixTree in Phase 1:
  - RadixTree.content_hash() static method
  - insert_content_prefix() / find_content_prefix()
  - evict_content_lru() / content_size property
  - content_hits metric
  - LRU eviction policy
  - content_maxsize=0 disables the cache
  - clear() flushes the content cache
  - Thread safety

No MLX / Metal required.  Pure-Python + numpy only.
"""
import threading
import unittest


class TestContentHashStatic(unittest.TestCase):

    def test_returns_64_hex_string(self):
        from squish.kv.radix_cache import RadixTree
        h = RadixTree.content_hash(b"hello world")
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h))

    def test_same_bytes_same_hash(self):
        from squish.kv.radix_cache import RadixTree
        data = b"\x00\xff" * 500
        self.assertEqual(RadixTree.content_hash(data), RadixTree.content_hash(data))

    def test_different_bytes_different_hash(self):
        from squish.kv.radix_cache import RadixTree
        h1 = RadixTree.content_hash(b"image_a_jpeg_bytes")
        h2 = RadixTree.content_hash(b"image_b_jpeg_bytes")
        self.assertNotEqual(h1, h2)

    def test_empty_bytes(self):
        from squish.kv.radix_cache import RadixTree
        h = RadixTree.content_hash(b"")
        self.assertEqual(len(h), 64)

    def test_large_bytes(self):
        from squish.kv.radix_cache import RadixTree
        data = bytes(range(256)) * 1024  # 256 KB
        h = RadixTree.content_hash(data)
        self.assertEqual(len(h), 64)


class TestContentCacheInsertFind(unittest.TestCase):

    def _tree(self, content_maxsize=8):
        from squish.kv.radix_cache import RadixTree
        return RadixTree(content_maxsize=content_maxsize)

    def test_miss_returns_none(self):
        t = self._tree()
        self.assertIsNone(t.find_content_prefix("a" * 64))

    def test_hit_after_insert(self):
        t = self._tree()
        h = "a" * 64
        t.insert_content_prefix(h, [1, 2, 3], [10, 11])
        result = t.find_content_prefix(h)
        self.assertIsNotNone(result)
        token_ids, block_refs = result
        self.assertEqual(token_ids, [1, 2, 3])
        self.assertEqual(block_refs, [10, 11])

    def test_returns_copies_not_references(self):
        """Mutating the returned lists must not affect the stored entry."""
        t = self._tree()
        h = "b" * 64
        t.insert_content_prefix(h, [1, 2, 3], [10])
        result = t.find_content_prefix(h)
        result[0].append(999)
        result[1].append(999)
        result2 = t.find_content_prefix(h)
        self.assertEqual(result2[0], [1, 2, 3])
        self.assertEqual(result2[1], [10])

    def test_insert_updates_existing_entry(self):
        t = self._tree()
        h = "c" * 64
        t.insert_content_prefix(h, [1, 2], [10])
        t.insert_content_prefix(h, [3, 4, 5], [20, 21])
        token_ids, block_refs = t.find_content_prefix(h)
        self.assertEqual(token_ids, [3, 4, 5])
        self.assertEqual(block_refs, [20, 21])

    def test_content_hits_increments_on_hit(self):
        t = self._tree()
        h = "d" * 64
        t.insert_content_prefix(h, [1], [0])
        self.assertEqual(t.content_hits, 0)
        t.find_content_prefix(h)
        self.assertEqual(t.content_hits, 1)
        t.find_content_prefix(h)
        self.assertEqual(t.content_hits, 2)

    def test_content_hits_not_incremented_on_miss(self):
        t = self._tree()
        t.find_content_prefix("e" * 64)
        self.assertEqual(t.content_hits, 0)

    def test_empty_content_hash_returns_none(self):
        t = self._tree()
        self.assertIsNone(t.find_content_prefix(""))

    def test_empty_token_ids_not_stored(self):
        t = self._tree()
        h = "f" * 64
        t.insert_content_prefix(h, [], [10])
        self.assertIsNone(t.find_content_prefix(h))

    def test_multiple_entries(self):
        t = self._tree()
        entries = {("a" * 64): ([1, 2], [10]),
                   ("b" * 64): ([3, 4, 5], [20, 21, 22]),
                   ("c" * 64): ([6], [30])}
        for h, (tids, brefs) in entries.items():
            t.insert_content_prefix(h, tids, brefs)
        for h, (expected_tids, expected_brefs) in entries.items():
            result = t.find_content_prefix(h)
            self.assertIsNotNone(result)
            self.assertEqual(result[0], expected_tids)
            self.assertEqual(result[1], expected_brefs)


class TestContentCacheContentHash(unittest.TestCase):
    """Integration: use RadixTree.content_hash() as the key."""

    def test_round_trip_with_computed_hash(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        data = b"\xff\xd8\xff" + bytes(range(100))  # fake JPEG prefix
        h = RadixTree.content_hash(data)
        t.insert_content_prefix(h, [101, 102, 103, 104], [1, 2])
        result = t.find_content_prefix(h)
        self.assertEqual(result[0], [101, 102, 103, 104])
        self.assertEqual(result[1], [1, 2])

    def test_different_images_different_entries(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        h1 = RadixTree.content_hash(b"image_one")
        h2 = RadixTree.content_hash(b"image_two")
        t.insert_content_prefix(h1, [1, 2], [10])
        t.insert_content_prefix(h2, [3, 4], [20])
        r1 = t.find_content_prefix(h1)
        r2 = t.find_content_prefix(h2)
        self.assertEqual(r1[0], [1, 2])
        self.assertEqual(r2[0], [3, 4])


class TestContentCacheLRU(unittest.TestCase):

    def _tree(self, n):
        from squish.kv.radix_cache import RadixTree
        return RadixTree(content_maxsize=n)

    def test_capacity_respected(self):
        t = self._tree(3)
        for i in range(10):
            h = chr(ord("a") + i) * 64
            t.insert_content_prefix(h, [i], [i])
        self.assertLessEqual(t.content_size, 3)

    def test_lru_oldest_evicted(self):
        """After filling to capacity, inserting a new entry evicts the oldest."""
        t = self._tree(2)
        h1, h2, h3 = "a" * 64, "b" * 64, "c" * 64
        t.insert_content_prefix(h1, [1], [1])
        t.insert_content_prefix(h2, [2], [2])
        # Touch h2 to make h1 the LRU
        t.find_content_prefix(h2)
        # Insert h3 → h1 should be evicted
        t.insert_content_prefix(h3, [3], [3])
        self.assertIsNone(t.find_content_prefix(h1), "h1 should have been LRU-evicted")
        self.assertIsNotNone(t.find_content_prefix(h2))
        self.assertIsNotNone(t.find_content_prefix(h3))

    def test_access_prevents_eviction(self):
        """Recently accessed entry must survive eviction."""
        t = self._tree(2)
        h1, h2, h3 = "x" * 64, "y" * 64, "z" * 64
        t.insert_content_prefix(h1, [1], [1])
        t.insert_content_prefix(h2, [2], [2])
        # Access h1 to promote it
        t.find_content_prefix(h1)
        # Now h2 is LRU
        t.insert_content_prefix(h3, [3], [3])
        self.assertIsNotNone(t.find_content_prefix(h1), "h1 was accessed last, must survive")
        self.assertIsNone(t.find_content_prefix(h2), "h2 was LRU, must be evicted")

    def test_content_maxsize_zero_disables_cache(self):
        t = self._tree(0)
        h = "a" * 64
        t.insert_content_prefix(h, [1, 2], [10])  # should be silently ignored
        self.assertIsNone(t.find_content_prefix(h))
        self.assertEqual(t.content_size, 0)

    def test_content_size_property(self):
        t = self._tree(4)
        self.assertEqual(t.content_size, 0)
        t.insert_content_prefix("a" * 64, [1], [1])
        self.assertEqual(t.content_size, 1)
        t.insert_content_prefix("b" * 64, [2], [2])
        self.assertEqual(t.content_size, 2)


class TestContentCacheEvictLRU(unittest.TestCase):

    def _tree(self, n=8):
        from squish.kv.radix_cache import RadixTree
        return RadixTree(content_maxsize=n)

    def test_evict_lru_removes_one_by_default(self):
        t = self._tree()
        for i in range(3):
            t.insert_content_prefix(chr(ord("a") + i) * 64, [i], [i])
        count = t.evict_content_lru(1)
        self.assertEqual(count, 1)
        self.assertEqual(t.content_size, 2)

    def test_evict_lru_n_removes_n(self):
        t = self._tree()
        for i in range(5):
            t.insert_content_prefix(chr(ord("a") + i) * 64, [i], [i])
        count = t.evict_content_lru(3)
        self.assertEqual(count, 3)
        self.assertEqual(t.content_size, 2)

    def test_evict_more_than_stored(self):
        t = self._tree()
        t.insert_content_prefix("a" * 64, [1], [1])
        count = t.evict_content_lru(100)
        self.assertEqual(count, 1)
        self.assertEqual(t.content_size, 0)

    def test_evict_empty_cache(self):
        t = self._tree()
        count = t.evict_content_lru(5)
        self.assertEqual(count, 0)


class TestContentCacheClear(unittest.TestCase):

    def test_clear_flushes_content_cache(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        h = RadixTree.content_hash(b"some image data")
        t.insert_content_prefix(h, [1, 2, 3], [100])
        self.assertIsNotNone(t.find_content_prefix(h))
        t.clear()
        self.assertIsNone(t.find_content_prefix(h))
        self.assertEqual(t.content_size, 0)

    def test_clear_also_flushes_text_cache(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        t.put("hello", "world", "stop")
        t.clear()
        self.assertIsNone(t.get("hello"))

    def test_clear_also_flushes_trie(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        t.insert_prefix([1, 2, 3], [10])
        t.clear()
        prefix_len, block_refs = t.find_prefix([1, 2, 3])
        self.assertEqual(prefix_len, 0)


class TestContentCacheDoesNotInterfereWithTextCache(unittest.TestCase):

    def test_content_cache_and_text_cache_coexist(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        # Text cache
        t.put("hello", "response", "stop")
        # Content cache
        h = RadixTree.content_hash(b"img data")
        t.insert_content_prefix(h, [1, 2], [5])
        # Both should work independently
        self.assertEqual(t.get("hello"), ("response", "stop"))
        self.assertEqual(t.find_content_prefix(h)[0], [1, 2])

    def test_content_hits_separate_from_text_hits(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree()
        t.put("q", "a", "stop")
        h = RadixTree.content_hash(b"img")
        t.insert_content_prefix(h, [1], [1])
        t.get("q")
        t.find_content_prefix(h)
        self.assertEqual(t.hits, 1)
        self.assertEqual(t.content_hits, 1)


class TestContentCacheThreadSafety(unittest.TestCase):

    def test_concurrent_insert_and_find(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree(content_maxsize=16)
        errors = []

        def _insert(offset):
            try:
                for i in range(20):
                    h = (chr(ord("a") + (offset + i) % 26)) * 64
                    t.insert_content_prefix(h, [offset + i], [offset + i])
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def _find(offset):
            try:
                for i in range(20):
                    h = (chr(ord("a") + (offset + i) % 26)) * 64
                    t.find_content_prefix(h)   # may hit or miss — just must not raise
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = (
            [threading.Thread(target=_insert, args=(i * 3,)) for i in range(4)]
            + [threading.Thread(target=_find, args=(i * 3,)) for i in range(4)]
        )
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()
        self.assertEqual(errors, [], f"Thread errors: {errors}")

    def test_concurrent_evict_and_find(self):
        from squish.kv.radix_cache import RadixTree
        t = RadixTree(content_maxsize=8)
        for i in range(8):
            t.insert_content_prefix(chr(ord("a") + i) * 64, [i], [i])
        errors = []

        def _evict():
            try:
                for _ in range(10):
                    t.evict_content_lru(1)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def _find(i):
            try:
                for _ in range(20):
                    t.find_content_prefix(chr(ord("a") + i % 8) * 64)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = (
            [threading.Thread(target=_evict) for _ in range(2)]
            + [threading.Thread(target=_find, args=(i,)) for i in range(4)]
        )
        for thr in threads:
            thr.start()
        for thr in threads:
            thr.join()
        self.assertEqual(errors, [], f"Thread errors: {errors}")


if __name__ == "__main__":
    unittest.main()
