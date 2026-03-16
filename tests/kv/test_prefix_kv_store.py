"""
tests/kv/test_prefix_kv_store.py

Unit tests for Phase 13 — PrefixKVStore and QuantizedKVCache.clone_snapshot().

All tests are pure-numpy (no MLX / Metal required).
"""
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Minimal stubs so the module can be imported without MLX
# ---------------------------------------------------------------------------

class _FakeTreeNode:
    def __init__(self):
        self.block_refs: list[int] = []
        self.children: dict = {}

    def touch(self):
        pass


class _FakeRadixTree:
    """Minimal RadixTree stand-in that actually implements find/insert."""

    def __init__(self):
        self._store: dict[tuple, list[int]] = {}  # frozen token_ids → block_refs
        self.prefix_hits: int = 0
        self._lock = threading.Lock()

    def insert_prefix(self, token_ids: list[int], block_refs: list[int]) -> None:
        with self._lock:
            self._store[tuple(token_ids)] = list(block_refs)

    def find_prefix(self, token_ids: list[int]) -> tuple[int, list[int]]:
        """Find the longest stored prefix of token_ids."""
        best_len = 0
        best_refs: list[int] = []
        with self._lock:
            for stored, refs in self._store.items():
                n = min(len(stored), len(token_ids))
                for i in range(n):
                    if stored[i] != token_ids[i]:
                        n = i
                        break
                if n > best_len:
                    best_len = n
                    best_refs = list(refs)
        if best_len > 0:
            self.prefix_hits += 1
        return best_len, best_refs


# ---------------------------------------------------------------------------
# Helper: build a small QuantizedKVCache with fake numpy data
# ---------------------------------------------------------------------------

def _make_cache(n_layers: int = 4, n_tokens: int = 16, n_heads: int = 2, head_dim: int = 8):
    """Build a QuantizedKVCache populated with deterministic random FP16 data."""
    from squish.kv.kv_cache import KVLayerCache, QuantizedKVCache

    cache = QuantizedKVCache(n_layers=n_layers, window=8, mode="int8")
    rng = np.random.default_rng(seed=7)
    for layer in cache._layers:
        layer.n_heads = n_heads
        layer.head_dim = head_dim
        # Simulate old (INT8) tier: (n_heads, n_old, head_dim)
        n_old = max(0, n_tokens - 8)
        if n_old > 0:
            layer.keys_old_q = rng.integers(-127, 127, (n_heads, n_old, head_dim),
                                            dtype=np.int8)
            layer.keys_old_s = rng.random((n_heads, n_old), dtype=np.float32) + 0.1
            layer.values_old_q = rng.integers(-127, 127, (n_heads, n_old, head_dim),
                                              dtype=np.int8)
            layer.values_old_s = rng.random((n_heads, n_old), dtype=np.float32) + 0.1
        # Simulate recent (FP16) window: list of (n_heads, head_dim)
        n_recent = min(n_tokens, 8)
        layer.keys_recent = [
            rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            for _ in range(n_recent)
        ]
        layer.values_recent = [
            rng.standard_normal((n_heads, head_dim)).astype(np.float16)
            for _ in range(n_recent)
        ]
    return cache


# ---------------------------------------------------------------------------
# TestCloneSnapshot — QuantizedKVCache.clone_snapshot()
# ---------------------------------------------------------------------------

class TestCloneSnapshot(unittest.TestCase):

    def test_returns_quantized_kv_cache(self):
        from squish.kv.kv_cache import QuantizedKVCache
        cache = _make_cache()
        snap = cache.clone_snapshot()
        self.assertIsInstance(snap, QuantizedKVCache)

    def test_clone_is_distinct_object(self):
        cache = _make_cache()
        snap = cache.clone_snapshot()
        self.assertIsNot(snap, cache)
        for src_lay, dst_lay in zip(cache._layers, snap._layers):
            self.assertIsNot(src_lay, dst_lay)

    def test_clone_n_layers_matches(self):
        cache = _make_cache(n_layers=6)
        snap = cache.clone_snapshot()
        self.assertEqual(snap.n_layers, 6)

    def test_clone_window_matches(self):
        cache = _make_cache()
        snap = cache.clone_snapshot()
        self.assertEqual(snap.window, cache.window)

    def test_clone_n_tokens_matches(self):
        cache = _make_cache(n_tokens=20)
        snap = cache.clone_snapshot()
        self.assertEqual(snap.n_tokens, cache.n_tokens)

    def test_clone_keys_old_q_same_values(self):
        cache = _make_cache(n_tokens=20)
        snap = cache.clone_snapshot()
        for src_lay, dst_lay in zip(cache._layers, snap._layers):
            if src_lay.keys_old_q is not None:
                np.testing.assert_array_equal(dst_lay.keys_old_q, src_lay.keys_old_q)

    def test_clone_keys_recent_same_values(self):
        cache = _make_cache(n_tokens=10)
        snap = cache.clone_snapshot()
        for src_lay, dst_lay in zip(cache._layers, snap._layers):
            for s, d in zip(src_lay.keys_recent, dst_lay.keys_recent):
                np.testing.assert_array_equal(s, d)

    def test_clone_recent_list_is_copy(self):
        """Mutating source recent list must not affect snapshot."""
        cache = _make_cache(n_tokens=10)
        snap = cache.clone_snapshot()
        # Add a fake array to source's recent list
        cache._layers[0].keys_recent.append(np.zeros((2, 8), dtype=np.float16))
        # Snapshot should be unchanged
        for src_lay, dst_lay in zip(cache._layers, snap._layers):
            break  # compare only layer 0
        self.assertNotEqual(
            len(cache._layers[0].keys_recent),
            len(snap._layers[0].keys_recent),
        )

    def test_reset_source_does_not_affect_snapshot(self):
        """After source.reset(), snapshot data must survive."""
        cache = _make_cache(n_tokens=20)
        snap_tokens_before = snap_after = None
        snap = cache.clone_snapshot()
        snap_tokens_before = snap.n_tokens
        cache.reset()
        self.assertEqual(snap.n_tokens, snap_tokens_before,
                         "Snapshot n_tokens changed after source reset()")

    def test_clone_svd_rank_carried(self):
        cache = _make_cache()
        cache._layers[0]._svd_rank = 16
        cache._layers[0]._svd_Vk = np.eye(8, dtype=np.float16)
        snap = cache.clone_snapshot()
        self.assertEqual(snap._layers[0]._svd_rank, 16)
        np.testing.assert_array_equal(snap._layers[0]._svd_Vk, cache._layers[0]._svd_Vk)

    def test_clone_svd_buf_cleared(self):
        """Calibration buffer should NOT be carried into the snapshot."""
        cache = _make_cache()
        cache._layers[0]._svd_buf_k = [np.zeros((2, 8), dtype=np.float16)]
        snap = cache.clone_snapshot()
        self.assertIsNone(snap._layers[0]._svd_buf_k)
        self.assertIsNone(snap._layers[0]._svd_buf_v)

    def test_snapped_flags_carried(self):
        cache = _make_cache(n_layers=3)
        cache._snapped = [True, False, True]
        snap = cache.clone_snapshot()
        self.assertEqual(snap._snapped, [True, False, True])

    def test_mode_is_int8(self):
        """Snapshot should always use 'int8' mode — no active eviction."""
        cache = _make_cache()
        cache.mode = "snap"
        snap = cache.clone_snapshot()
        self.assertEqual(snap.mode, "int8")


# ---------------------------------------------------------------------------
# TestPrefixKVStore — PrefixKVStore public API
# ---------------------------------------------------------------------------

class TestPrefixKVStoreFind(unittest.TestCase):

    def _make_store(self, max_snapshots=4):
        from squish.kv.prefix_kv_store import PrefixKVStore
        return PrefixKVStore(_FakeRadixTree(), max_snapshots=max_snapshots)

    def test_find_empty_store_returns_zero(self):
        store = self._make_store()
        prefix_len, snap_id = store.find([1, 2, 3, 4, 5, 6, 7, 8, 9])
        self.assertEqual(prefix_len, 0)
        self.assertIsNone(snap_id)

    def test_find_after_store_returns_match(self):
        store = self._make_store()
        cache = _make_cache(n_tokens=10)
        ids = list(range(20))   # 20 tokens → above _MIN_PREFIX_TOKENS
        store.store(ids, cache)
        prefix_len, snap_id = store.find(ids)
        self.assertEqual(prefix_len, len(ids))
        self.assertIsNotNone(snap_id)

    def test_find_prefix_match(self):
        """A new query sharing only the first N tokens of a stored sequence returns N."""
        store = self._make_store()
        cache = _make_cache(n_tokens=10)
        ids = list(range(30))
        store.store(ids, cache)
        # Query shares only the first 20 tokens, then diverges
        new_ids = ids[:20] + [99, 100, 101]
        prefix_len, snap_id = store.find(new_ids)
        # Only 20 tokens overlap between new_ids and the stored 30-token sequence
        self.assertEqual(prefix_len, 20)

    def test_find_short_ids_skipped(self):
        """Token sequences shorter than min_prefix_tokens are never stored/found."""
        store = self._make_store()
        cache = _make_cache(n_tokens=4)
        short_ids = [1, 2, 3]  # below default min 8
        store.store(short_ids, cache)
        prefix_len, snap_id = store.find(short_ids)
        self.assertEqual(prefix_len, 0)
        self.assertIsNone(snap_id)

    def test_find_increments_prefix_hits_on_tree(self):
        store = self._make_store()
        cache = _make_cache(n_tokens=10)
        ids = list(range(20))
        store.store(ids, cache)
        before = store._tree.prefix_hits
        store.find(ids)
        self.assertEqual(store._tree.prefix_hits, before + 1)

    def test_find_evicted_snapshot_returns_zero(self):
        """If snapshot was LRU-evicted, find() must return (0, None)."""
        store = self._make_store(max_snapshots=1)
        cache = _make_cache(n_tokens=10)
        ids_a = list(range(20))
        ids_b = list(range(100, 120))
        store.store(ids_a, cache)
        # Store a second entry — evicts ids_a
        store.store(ids_b, cache)
        # ids_a trie entry exists but snapshot is gone
        prefix_len, snap_id = store.find(ids_a)
        self.assertEqual(prefix_len, 0)
        self.assertIsNone(snap_id)


class TestPrefixKVStoreStore(unittest.TestCase):

    def _make_store(self, max_snapshots=4):
        from squish.kv.prefix_kv_store import PrefixKVStore
        return PrefixKVStore(_FakeRadixTree(), max_snapshots=max_snapshots)

    def test_store_returns_int_snap_id(self):
        store = self._make_store()
        cache = _make_cache()
        snap_id = store.store(list(range(20)), cache)
        self.assertIsInstance(snap_id, int)

    def test_store_increments_snap_id(self):
        store = self._make_store()
        cache = _make_cache()
        id_a = store.store(list(range(20)), cache)
        id_b = store.store(list(range(100, 120)), cache)
        self.assertNotEqual(id_a, id_b)

    def test_store_short_ids_returns_none(self):
        store = self._make_store()
        cache = _make_cache()
        result = store.store([1, 2, 3], cache)
        self.assertIsNone(result)

    def test_store_respects_max_snapshots(self):
        store = self._make_store(max_snapshots=3)
        cache = _make_cache()
        for i in range(10):
            store.store(list(range(i * 20, i * 20 + 20)), cache)
        self.assertLessEqual(len(store), 3)

    def test_store_lru_evicts_oldest(self):
        """After overflow, the oldest snapshot should be evicted."""
        store = self._make_store(max_snapshots=2)
        cache = _make_cache()
        id_a = store.store(list(range(0, 20)), cache)
        id_b = store.store(list(range(20, 40)), cache)  # noqa: F841
        id_c = store.store(list(range(40, 60)), cache)  # noqa: F841
        # id_a should have been evicted
        with store._lock:
            self.assertNotIn(id_a, store._store)

    def test_store_snapshot_is_independent(self):
        """Resetting the source cache must not affect the stored snapshot."""
        store = self._make_store()
        cache = _make_cache(n_tokens=20)
        ids = list(range(20))
        snap_id = store.store(ids, cache)
        tokens_before = cache.n_tokens
        cache.reset()
        with store._lock:
            snap = store._store[snap_id]
        # Snapshot n_tokens should equal what was in the cache at store time
        self.assertEqual(snap.n_tokens, tokens_before)

    def test_store_calls_insert_prefix(self):
        """store() must call insert_prefix on the radix tree."""
        from squish.kv.prefix_kv_store import PrefixKVStore
        tree = _FakeRadixTree()
        store = PrefixKVStore(tree, max_snapshots=4)
        cache = _make_cache()
        ids = list(range(20))
        snap_id = store.store(ids, cache)
        # The radix tree should now have an entry for ids
        p_len, block_refs = tree.find_prefix(ids)
        self.assertGreater(p_len, 0)
        self.assertIn(snap_id, block_refs)


class TestPrefixKVStoreRestore(unittest.TestCase):

    def _make_store(self, max_snapshots=4):
        from squish.kv.prefix_kv_store import PrefixKVStore
        return PrefixKVStore(_FakeRadixTree(), max_snapshots=max_snapshots)

    def test_restore_copies_kv_data(self):
        """restore() must fill target_cache with the snapshot's KV data."""
        from squish.kv.kv_cache import QuantizedKVCache
        store = self._make_store()
        cache = _make_cache(n_tokens=20)
        ids = list(range(20))
        snap_id = store.store(ids, cache)

        # Create an empty target and restore into it
        target = QuantizedKVCache(n_layers=cache.n_layers, window=cache.window)
        result = store.restore(snap_id, target)

        self.assertTrue(result)
        self.assertEqual(target.n_tokens, cache.n_tokens)

    def test_restore_unknown_id_returns_false(self):
        store = self._make_store()
        from squish.kv.kv_cache import QuantizedKVCache
        target = QuantizedKVCache(n_layers=4, window=8)
        result = store.restore(9999, target)
        self.assertFalse(result)

    def test_restore_does_not_alias_old_q(self):
        """After restore, appending to source should not touch the restored target."""
        from squish.kv.kv_cache import QuantizedKVCache
        store = self._make_store()
        cache = _make_cache(n_tokens=16)
        ids = list(range(20))
        snap_id = store.store(ids, cache)

        target = QuantizedKVCache(n_layers=cache.n_layers, window=cache.window)
        store.restore(snap_id, target)
        tokens_after_restore = target.n_tokens

        # Reset source — this should not change target
        cache.reset()
        self.assertEqual(target.n_tokens, tokens_after_restore)

    def test_full_round_trip(self):
        """store → find → restore produces correct n_tokens in target."""
        from squish.kv.kv_cache import QuantizedKVCache
        store = self._make_store()
        cache = _make_cache(n_tokens=18)
        ids = list(range(25))
        store.store(ids, cache)

        prefix_len, snap_id = store.find(ids)
        self.assertGreater(prefix_len, 0)
        self.assertIsNotNone(snap_id)

        target = QuantizedKVCache(n_layers=cache.n_layers, window=cache.window)
        ok = store.restore(snap_id, target)
        self.assertTrue(ok)
        self.assertEqual(target.n_tokens, cache.n_tokens)


# ---------------------------------------------------------------------------
# TestPrefixKVStoreStats
# ---------------------------------------------------------------------------

class TestPrefixKVStoreStats(unittest.TestCase):

    def _make_store(self):
        from squish.kv.prefix_kv_store import PrefixKVStore
        return PrefixKVStore(_FakeRadixTree(), max_snapshots=4)

    def test_stats_keys(self):
        store = self._make_store()
        s = store.stats()
        self.assertIn("snapshots_stored", s)
        self.assertIn("max_snapshots", s)
        self.assertIn("prefix_hits", s)
        self.assertIn("min_prefix_tokens", s)

    def test_stats_snapshots_stored_increments(self):
        store = self._make_store()
        cache = _make_cache()
        self.assertEqual(store.stats()["snapshots_stored"], 0)
        store.store(list(range(20)), cache)
        self.assertEqual(store.stats()["snapshots_stored"], 1)

    def test_len_reflects_count(self):
        store = self._make_store()
        cache = _make_cache()
        self.assertEqual(len(store), 0)
        store.store(list(range(20)), cache)
        self.assertEqual(len(store), 1)


# ---------------------------------------------------------------------------
# TestPrefixKVStoreThreadSafety
# ---------------------------------------------------------------------------

class TestPrefixKVStoreThreadSafety(unittest.TestCase):

    def test_concurrent_store(self):
        """Multiple threads storing snapshots concurrently must not corrupt state."""
        from squish.kv.prefix_kv_store import PrefixKVStore
        store = PrefixKVStore(_FakeRadixTree(), max_snapshots=8)
        errors = []

        def _worker(offset: int):
            try:
                cache = _make_cache(n_tokens=10)
                ids = list(range(offset, offset + 20))
                store.store(ids, cache)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_worker, args=(i * 30,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [], f"Thread errors: {errors}")
        self.assertLessEqual(len(store), 8)

    def test_concurrent_find_and_store(self):
        """find() and store() called concurrently must not raise."""
        from squish.kv.prefix_kv_store import PrefixKVStore
        store = PrefixKVStore(_FakeRadixTree(), max_snapshots=4)
        cache = _make_cache(n_tokens=12)
        ids = list(range(20))
        store.store(ids, cache)
        errors = []

        def _find_worker():
            try:
                for _ in range(50):
                    store.find(ids)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        def _store_worker(offset):
            try:
                for i in range(10):
                    store.store(list(range(offset + i * 20, offset + i * 20 + 20)), cache)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = (
            [threading.Thread(target=_find_worker) for _ in range(3)]
            + [threading.Thread(target=_store_worker, args=(100 + i * 200,)) for i in range(2)]
        )
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], f"Thread errors: {errors}")


if __name__ == "__main__":
    unittest.main()
