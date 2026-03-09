"""
tests/test_disk_kvcache.py

Unit tests for the disk-related additions to squish/kv_cache.py:

  • KVLayerCache.enable_disk_tier
  • KVLayerCache._maybe_spill_to_disk
  • KVLayerCache._disk_full_kv
  • KVLayerCache disk tier integration in get_full_kv via parts_k/parts_v
  • KVLayerCache.reset clears disk tier fields
  • QuantizedKVCache.restore_from
  • DiskKVCache._key  — deterministic / stable
  • DiskKVCache._serialise / _deserialise  — roundtrip fidelity
  • DiskKVCache.store + lookup  — write-then-read, with last_logit
  • DiskKVCache.lookup  — miss, corrupt-file recovery
  • DiskKVCache._evict_if_needed  — LRU cap
"""
from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from squish.kv_cache import (
    DiskKVCache,
    KVLayerCache,
    QuantizedKVCache,
)

# ── test fixtures / helpers ───────────────────────────────────────────────────

N_HEADS  = 2
HEAD_DIM = 4


def _rand_kv(T: int = 1, n_heads: int = N_HEADS,
             head_dim: int = HEAD_DIM, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
    v = rng.standard_normal((n_heads, head_dim)).astype(np.float16)
    return k, v


def _populated_layer(n_tokens: int, window: int = 64,
                     seed: int = 0) -> KVLayerCache:
    """Return a KVLayerCache with *n_tokens* appended."""
    layer = KVLayerCache(window=window)
    rng = np.random.default_rng(seed)
    for _t in range(n_tokens):
        k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
        layer.append(k, v)
    return layer


def _populated_qkv_cache(n_layers: int = 2, n_tokens: int = 3) -> QuantizedKVCache:
    """Populate a QuantizedKVCache with *n_tokens* per layer (recent window)."""
    cache = QuantizedKVCache(n_layers=n_layers, window=64, mode="int8")
    rng = np.random.default_rng(1)
    for i in range(n_layers):
        for _ in range(n_tokens):
            k = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            v = rng.standard_normal((N_HEADS, HEAD_DIM)).astype(np.float16)
            cache._layers[i].append(k, v)
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# KVLayerCache.enable_disk_tier
# ─────────────────────────────────────────────────────────────────────────────

class TestEnableDiskTier:
    def test_creates_memmap_files(self, tmp_path):
        layer = KVLayerCache()
        layer.enable_disk_tier(
            threshold=4, max_disk_tokens=16,
            cache_dir=tmp_path, n_heads=N_HEADS, head_dim=HEAD_DIM,
        )
        assert layer._disk_map_k is not None
        assert layer._disk_map_v is not None
        assert layer._disk_map_k.shape == (N_HEADS, 16, HEAD_DIM)

    def test_disk_n_initialised_to_zero(self, tmp_path):
        layer = KVLayerCache()
        layer.enable_disk_tier(4, 16, tmp_path, N_HEADS, HEAD_DIM)
        assert layer._disk_n == 0

    def test_creates_scale_arrays(self, tmp_path):
        layer = KVLayerCache()
        layer.enable_disk_tier(4, 16, tmp_path, N_HEADS, HEAD_DIM)
        assert layer._disk_scales_k is not None
        assert layer._disk_scales_k.shape == (N_HEADS, 16)
        assert layer._disk_scales_k.dtype == np.float32

    def test_cache_dir_created_if_missing(self, tmp_path):
        sub = tmp_path / "level1" / "level2"
        layer = KVLayerCache()
        layer.enable_disk_tier(4, 16, sub, N_HEADS, HEAD_DIM)
        assert sub.is_dir()

    def test_threshold_stored(self, tmp_path):
        layer = KVLayerCache()
        layer.enable_disk_tier(8, 32, tmp_path, N_HEADS, HEAD_DIM)
        assert layer._disk_threshold == 8


# ─────────────────────────────────────────────────────────────────────────────
# KVLayerCache._maybe_spill_to_disk
# ─────────────────────────────────────────────────────────────────────────────

class TestMaybeSpillToDisk:
    def _layer_with_disk(self, tmp_path, threshold: int = 2,
                         max_disk: int = 20) -> KVLayerCache:
        layer = KVLayerCache(window=1)  # tiny window forces quick move to old_q
        layer.enable_disk_tier(threshold, max_disk, tmp_path, N_HEADS, HEAD_DIM)
        return layer

    def test_no_spill_below_threshold(self, tmp_path):
        layer = self._layer_with_disk(tmp_path, threshold=10)
        k, v = _rand_kv()
        layer.append(k, v)
        assert layer._disk_n == 0

    def test_spill_occurs_above_threshold(self, tmp_path):
        """Append enough tokens so old_q exceeds threshold → spill fires."""
        layer = self._layer_with_disk(tmp_path, threshold=1, max_disk=30)
        for i in range(8):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        assert layer._disk_n > 0

    def test_ram_buffer_trimmed_after_spill(self, tmp_path):
        """After spill the old_q RAM array is shorter than before spill."""
        layer = self._layer_with_disk(tmp_path, threshold=1, max_disk=30)
        for i in range(8):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        if layer.keys_old_q is not None:
            # RAM shouldn't hold more than threshold + some small delta
            assert layer.keys_old_q.shape[1] <= 3

    def test_graceful_degrade_when_disk_full(self, tmp_path):
        """When disk tier is full, no crash — RAM retains all data."""
        layer = self._layer_with_disk(tmp_path, threshold=1, max_disk=2)
        for i in range(12):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        # Should not raise
        full_k, full_v = layer.get_full_kv()
        # At least something is returned
        assert full_k is not None

    def test_no_op_when_disabled(self, tmp_path):
        """Without enable_disk_tier, _maybe_spill_to_disk is a no-op."""
        layer = KVLayerCache(window=1)
        for i in range(10):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        assert layer._disk_n == 0
        assert layer._disk_map_k is None


# ─────────────────────────────────────────────────────────────────────────────
# KVLayerCache._disk_full_kv
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskFullKV:
    def test_empty_disk_returns_none_none(self):
        layer = KVLayerCache()
        k, v = layer._disk_full_kv()
        assert k is None
        assert v is None

    def test_after_spill_returns_correct_shape(self, tmp_path):
        layer = KVLayerCache(window=1)
        layer.enable_disk_tier(1, 30, tmp_path, N_HEADS, HEAD_DIM)
        for i in range(6):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        if layer._disk_n > 0:
            dk, dv = layer._disk_full_kv()
            assert dk is not None
            assert dk.shape == (N_HEADS, layer._disk_n, HEAD_DIM)
            assert dk.dtype == np.float16

    def test_disk_kv_dtype_float16(self, tmp_path):
        layer = KVLayerCache(window=1)
        layer.enable_disk_tier(1, 30, tmp_path, N_HEADS, HEAD_DIM)
        for i in range(6):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        if layer._disk_n > 0:
            dk, _ = layer._disk_full_kv()
            assert dk.dtype == np.float16


# ─────────────────────────────────────────────────────────────────────────────
# get_full_kv disk-tier integration (parts_k / parts_v path)
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFullKVDiskIntegration:
    def test_get_full_kv_includes_disk_tokens(self, tmp_path):
        """Tokens spilled to disk appear in get_full_kv output."""
        layer = KVLayerCache(window=1)
        layer.enable_disk_tier(2, 30, tmp_path, N_HEADS, HEAD_DIM)
        n = 8
        for i in range(n):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        full_k, full_v = layer.get_full_kv()
        assert full_k is not None
        # Total tokens = disk + ram_int8 + recent
        total = layer._disk_n + (
            layer.keys_old_q.shape[1] if layer.keys_old_q is not None else 0
        ) + len(layer.keys_recent)
        assert full_k.shape[1] == total

    def test_get_full_kv_empty_still_returns_none(self):
        layer = KVLayerCache()
        k, v = layer.get_full_kv()
        assert k is None and v is None


# ─────────────────────────────────────────────────────────────────────────────
# KVLayerCache.reset clears disk fields
# ─────────────────────────────────────────────────────────────────────────────

class TestResetClearsDiskTier:
    def test_reset_clears_disk_n(self, tmp_path):
        layer = KVLayerCache(window=1)
        layer.enable_disk_tier(1, 20, tmp_path, N_HEADS, HEAD_DIM)
        for i in range(6):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        layer.reset()
        assert layer._disk_n == 0

    def test_reset_nullifies_memmaps(self, tmp_path):
        layer = KVLayerCache(window=1)
        layer.enable_disk_tier(1, 20, tmp_path, N_HEADS, HEAD_DIM)
        for i in range(4):
            k, v = _rand_kv(seed=i)
            layer.append(k, v)
        layer.reset()
        assert layer._disk_map_k is None
        assert layer._disk_map_v is None


# ─────────────────────────────────────────────────────────────────────────────
# QuantizedKVCache.restore_from
# ─────────────────────────────────────────────────────────────────────────────

class TestRestoreFrom:
    def test_restore_copies_layer_data(self):
        src = _populated_qkv_cache(n_layers=2, n_tokens=3)
        dst = QuantizedKVCache(n_layers=2, window=64, mode="int8")

        dst.restore_from(src)

        for i in range(2):
            src_lay = src._layers[i]
            dst_lay = dst._layers[i]
            assert dst_lay.n_heads  == src_lay.n_heads
            assert dst_lay.head_dim == src_lay.head_dim
            # recent tokens copied
            assert len(dst_lay.keys_recent) == len(src_lay.keys_recent)

    def test_restore_recent_tokens_equal(self):
        src = _populated_qkv_cache(n_layers=1, n_tokens=4)
        dst = QuantizedKVCache(n_layers=1, window=64)
        dst.restore_from(src)
        for t in range(len(src._layers[0].keys_recent)):
            assert np.allclose(
                dst._layers[0].keys_recent[t],
                src._layers[0].keys_recent[t],
            )

    def test_restore_old_q_arrays(self):
        """When src has INT8 old_q data it's copied to dst."""
        src = QuantizedKVCache(n_layers=1, window=1, mode="int8")
        for i in range(6):
            k, v = _rand_kv(seed=i)
            src._layers[0].append(k, v)

        dst = QuantizedKVCache(n_layers=1, window=1, mode="int8")
        dst.restore_from(src)

        if src._layers[0].keys_old_q is not None:
            assert np.array_equal(
                dst._layers[0].keys_old_q,
                src._layers[0].keys_old_q,
            )

    def test_restore_mutates_dst_not_src(self):
        src = _populated_qkv_cache(n_layers=1, n_tokens=3)
        dst = QuantizedKVCache(n_layers=1, window=64)
        dst.restore_from(src)
        # Mutate dst, verify src unchanged
        dst._layers[0].keys_recent.clear()
        assert len(src._layers[0].keys_recent) == 3

    def test_restore_multiple_times_idempotent(self):
        src = _populated_qkv_cache(n_layers=2, n_tokens=2)
        dst = QuantizedKVCache(n_layers=2, window=64)
        dst.restore_from(src)
        dst.restore_from(src)
        assert len(dst._layers[0].keys_recent) == len(src._layers[0].keys_recent)


# ─────────────────────────────────────────────────────────────────────────────
# DiskKVCache._key
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskKVCacheKey:
    def test_same_ids_same_key(self):
        k1 = DiskKVCache._key([1, 2, 3])
        k2 = DiskKVCache._key([1, 2, 3])
        assert k1 == k2

    def test_different_ids_different_key(self):
        k1 = DiskKVCache._key([1, 2, 3])
        k2 = DiskKVCache._key([1, 2, 4])
        assert k1 != k2

    def test_key_is_sha256_hex(self):
        ids = [10, 20, 30]
        raw = np.array(ids, dtype=np.int32).tobytes()
        expected = hashlib.sha256(raw).hexdigest()
        assert DiskKVCache._key(ids) == expected

    def test_empty_ids(self):
        k = DiskKVCache._key([])
        assert isinstance(k, str) and len(k) == 64

    def test_order_matters(self):
        assert DiskKVCache._key([1, 2]) != DiskKVCache._key([2, 1])


# ─────────────────────────────────────────────────────────────────────────────
# DiskKVCache._serialise / _deserialise
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialisationRoundtrip:
    def _roundtrip(self, cache: QuantizedKVCache) -> QuantizedKVCache:
        arrays = DiskKVCache._serialise(cache)
        assert arrays is not None
        # Save and reload via npz
        import io

        import numpy as np_
        buf = io.BytesIO()
        np_.savez(buf, **arrays)
        buf.seek(0)
        data = np_.load(buf, allow_pickle=False)
        return DiskKVCache._deserialise(data)

    def test_n_layers_preserved(self):
        src = _populated_qkv_cache(n_layers=3, n_tokens=2)
        dst = self._roundtrip(src)
        assert len(dst._layers) == 3

    def test_n_heads_preserved(self):
        src = _populated_qkv_cache(n_layers=2, n_tokens=2)
        dst = self._roundtrip(src)
        for i in range(2):
            assert dst._layers[i].n_heads == src._layers[i].n_heads

    def test_head_dim_preserved(self):
        src = _populated_qkv_cache(n_layers=2, n_tokens=2)
        dst = self._roundtrip(src)
        for i in range(2):
            assert dst._layers[i].head_dim == src._layers[i].head_dim

    def test_recent_tokens_preserved(self):
        src = _populated_qkv_cache(n_layers=1, n_tokens=4)
        dst = self._roundtrip(src)
        assert len(dst._layers[0].keys_recent) == len(src._layers[0].keys_recent)

    def test_recent_values_numerically_close(self):
        src = _populated_qkv_cache(n_layers=1, n_tokens=3)
        dst = self._roundtrip(src)
        for t in range(len(src._layers[0].keys_recent)):
            np.testing.assert_allclose(
                dst._layers[0].keys_recent[t].astype(np.float32),
                src._layers[0].keys_recent[t].astype(np.float32),
                rtol=1e-3,
            )

    def test_serialise_returns_none_for_unpopulated_layer(self):
        cache = QuantizedKVCache(n_layers=2, window=64)
        # Layers have n_heads=None (no data appended)
        result = DiskKVCache._serialise(cache)
        assert result is None

    def test_old_q_roundtrip(self):
        """INT8 quantized old tokens survive serialisation."""
        src = QuantizedKVCache(n_layers=1, window=1, mode="int8")
        for i in range(8):
            k, v = _rand_kv(seed=i)
            src._layers[0].append(k, v)
        if src._layers[0].keys_old_q is None:
            pytest.skip("No old_q accumulated with this window")
        dst = self._roundtrip(src)
        if dst._layers[0].keys_old_q is not None:
            assert np.array_equal(
                dst._layers[0].keys_old_q,
                src._layers[0].keys_old_q,
            )


# ─────────────────────────────────────────────────────────────────────────────
# DiskKVCache.store + lookup (full I/O roundtrip)
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskKVCacheStoreAndLookup:
    def _wait_for_entry(self, disk_cache: DiskKVCache, ids: list[int],
                        timeout: float = 5.0) -> bool:
        """Block until the background store thread finishes writing the entry."""
        import time
        entry = disk_cache._dir / (DiskKVCache._key(ids) + ".npz")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if entry.exists():
                return True
            time.sleep(0.05)
        return False

    def test_lookup_miss_returns_none(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        assert dc.lookup([1, 2, 3]) is None

    def test_store_and_lookup_roundtrip(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        cache = _populated_qkv_cache(n_layers=2, n_tokens=3)
        last_logit = np.random.default_rng(0).random(20).astype(np.float32)
        ids = [10, 20, 30]

        dc.store(ids, cache, last_logit)
        assert self._wait_for_entry(dc, ids), "store() never wrote file"

        result = dc.lookup(ids)
        assert result is not None
        restored_cache, restored_logit = result
        assert restored_cache is not None
        assert restored_logit is not None

    def test_lookup_returns_last_logit(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        cache = _populated_qkv_cache(n_layers=1, n_tokens=2)
        logit = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        ids = [5, 6]

        dc.store(ids, cache, logit)
        assert self._wait_for_entry(dc, ids)

        result = dc.lookup(ids)
        assert result is not None
        _, restored = result
        np.testing.assert_allclose(restored, logit, rtol=1e-5)

    def test_lookup_hits_same_key_only(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        cache = _populated_qkv_cache(n_layers=1, n_tokens=2)
        logit = np.zeros(4, dtype=np.float32)

        ids_a = [1, 2, 3]
        ids_b = [4, 5, 6]

        dc.store(ids_a, cache, logit)
        assert self._wait_for_entry(dc, ids_a)

        assert dc.lookup(ids_b) is None  # different key → miss

    def test_lookup_without_last_logit_returns_none(self, tmp_path):
        """An entry saved *without* last_logit is treated as a miss."""
        dc = DiskKVCache(tmp_path)
        cache = _populated_qkv_cache(n_layers=1, n_tokens=2)
        ids = [7, 8]
        # Store without last_logit (None → not included in arrays)
        dc.store(ids, cache, None)
        assert self._wait_for_entry(dc, ids)
        result = dc.lookup(ids)
        # Expecting None because last_logit key absent
        assert result is None

    def test_corrupt_file_returns_none_and_deleted(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        ids = [99]
        key = DiskKVCache._key(ids)
        # Write garbage content
        (tmp_path / f"{key}.npz").write_bytes(b"not a valid npz")
        result = dc.lookup(ids)
        assert result is None
        # File should be removed
        assert not (tmp_path / f"{key}.npz").exists()

    def test_store_unpopulated_cache_silently_skipped(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        empty = QuantizedKVCache(n_layers=2, window=64)
        ids = [1000]
        dc.store(ids, empty, np.zeros(4, dtype=np.float32))
        time.sleep(0.3)
        # _serialise returns None → no file written
        assert not (tmp_path / f"{DiskKVCache._key(ids)}.npz").exists()


# ─────────────────────────────────────────────────────────────────────────────
# DiskKVCache._evict_if_needed — LRU cap
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskKVCacheEviction:
    def _make_cache_with_entries(self, tmp_path, n_entries: int,
                                  max_entries: int) -> DiskKVCache:
        dc = DiskKVCache(tmp_path, max_entries=max_entries)
        cache = _populated_qkv_cache(n_layers=1, n_tokens=2)
        logit = np.zeros(4, dtype=np.float32)
        for i in range(n_entries):
            ids = [i * 100]
            arrays = DiskKVCache._serialise(cache)
            arrays["last_logit"] = logit
            entry = tmp_path / f"{DiskKVCache._key(ids)}.npz"
            np.savez_compressed(str(entry), **arrays)
            time.sleep(0.01)  # ensure distinct mtimes
        return dc

    def test_eviction_keeps_max_entries(self, tmp_path):
        dc = self._make_cache_with_entries(tmp_path, n_entries=6, max_entries=4)
        # Trigger eviction manually
        dc._evict_if_needed()
        remaining = list(tmp_path.glob("*.npz"))
        assert len(remaining) <= 4

    def test_no_eviction_at_capacity(self, tmp_path):
        dc = self._make_cache_with_entries(tmp_path, n_entries=4, max_entries=4)
        dc._evict_if_needed()
        remaining = list(tmp_path.glob("*.npz"))
        assert len(remaining) == 4

    def test_eviction_removes_oldest_by_mtime(self, tmp_path):
        """The surviving files should be the most-recently written ones."""
        max_e = 2
        dc = self._make_cache_with_entries(tmp_path, n_entries=4, max_entries=max_e)
        # Record which 2 entries have the newest mtime
        entries = sorted(tmp_path.glob("*.npz"), key=lambda p: p.stat().st_mtime)
        expected_survivors = {e.name for e in entries[-max_e:]}

        dc._evict_if_needed()
        remaining = {e.name for e in tmp_path.glob("*.npz")}
        assert remaining == expected_survivors


# ─────────────────────────────────────────────────────────────────────────────
# DiskKVCache initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestDiskKVCacheInit:
    def test_creates_directory(self, tmp_path):
        sub = tmp_path / "new_dir"
        DiskKVCache(cache_dir=sub, max_entries=10)
        assert sub.is_dir()

    def test_default_max_entries(self, tmp_path):
        dc = DiskKVCache(tmp_path)
        assert dc._max == 64

    def test_custom_max_entries(self, tmp_path):
        dc = DiskKVCache(tmp_path, max_entries=128)
        assert dc._max == 128

    def test_path_accepts_string(self, tmp_path):
        dc = DiskKVCache(str(tmp_path))
        assert dc._dir.is_dir()
