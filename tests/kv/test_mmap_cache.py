"""Tests for squish.kv.mmap_cache — disk-backed KV storage."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from squish.kv.mmap_cache import MMapKVCache, MMapKVLayer, _META_FILENAME


def _make_token(n_heads=4, head_dim=8, val=1.0):
    return (
        np.full((n_heads, head_dim), val, dtype=np.float16),
        np.full((n_heads, head_dim), -val, dtype=np.float16),
    )


class TestLayerBasics:
    def test_init_creates_files(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=4, n_heads=2, head_dim=8)
        try:
            assert (tmp_path / "L0" / "k.bin").exists()
            assert (tmp_path / "L0" / "v.bin").exists()
            assert (tmp_path / "L0" / _META_FILENAME).exists()
            assert layer.n_tokens == 0
            assert layer.capacity == 4
        finally:
            layer.close()

    def test_append_and_read_roundtrip(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=4, n_heads=2, head_dim=8)
        try:
            k, v = _make_token(n_heads=2, head_dim=8, val=3.0)
            layer.append(k, v)
            assert layer.n_tokens == 1
            k_back, v_back = layer.get(0)
            assert np.array_equal(k_back, k)
            assert np.array_equal(v_back, v)
        finally:
            layer.close()

    def test_get_range(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=8, n_heads=2, head_dim=4)
        try:
            for i in range(5):
                k, v = _make_token(n_heads=2, head_dim=4, val=float(i + 1))
                layer.append(k, v)
            keys, values = layer.get_range(1, 4)
            assert keys.shape == (3, 2, 4)
            assert np.array_equal(keys[0, 0, 0], np.float16(2.0))
            assert np.array_equal(keys[2, 0, 0], np.float16(4.0))
            # values stored as -val
            assert np.array_equal(values[0, 0, 0], np.float16(-2.0))
        finally:
            layer.close()

    def test_overflow_raises(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=2, n_heads=2, head_dim=4)
        try:
            for _ in range(2):
                k, v = _make_token(n_heads=2, head_dim=4)
                layer.append(k, v)
            with pytest.raises(OverflowError):
                layer.append(*_make_token(n_heads=2, head_dim=4))
        finally:
            layer.close()

    def test_evict_oldest_shifts(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=4, n_heads=2, head_dim=4)
        try:
            for i in range(4):
                layer.append(*_make_token(n_heads=2, head_dim=4, val=float(i + 1)))
            dropped = layer.evict_oldest(2)
            assert dropped == 2
            assert layer.n_tokens == 2
            k0, _ = layer.get(0)
            # what was at index 2 (val=3) should now be at index 0
            assert np.array_equal(k0, np.full((2, 4), 3.0, dtype=np.float16))
        finally:
            layer.close()

    def test_shape_validation(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=2, n_heads=4, head_dim=8)
        try:
            bad_k = np.zeros((2, 8), dtype=np.float16)        # n_heads mismatch
            bad_v = np.zeros((4, 8), dtype=np.float16)
            with pytest.raises(ValueError, match="key shape"):
                layer.append(bad_k, bad_v)
        finally:
            layer.close()


class TestLayerPersistence:
    def test_reopen_preserves_data(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=4, n_heads=2, head_dim=4)
        layer.append(*_make_token(n_heads=2, head_dim=4, val=7.0))
        layer.append(*_make_token(n_heads=2, head_dim=4, val=11.0))
        layer.flush()
        layer.close()

        # Reopen with matching shape — n_tokens should restore from meta.
        layer2 = MMapKVLayer(tmp_path / "L0", capacity=4, n_heads=2, head_dim=4)
        try:
            assert layer2.n_tokens == 2
            k0, _ = layer2.get(0)
            assert np.array_equal(k0, np.full((2, 4), 7.0, dtype=np.float16))
        finally:
            layer2.close()

    def test_meta_json_valid(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=4, n_heads=2, head_dim=4)
        try:
            layer.append(*_make_token(n_heads=2, head_dim=4))
            meta = json.loads((tmp_path / "L0" / _META_FILENAME).read_text())
            assert meta["capacity"] == 4
            assert meta["n_heads"] == 2
            assert meta["n_tokens"] == 1
        finally:
            layer.close()

    def test_use_after_close_raises(self, tmp_path: Path):
        layer = MMapKVLayer(tmp_path / "L0", capacity=2, n_heads=2, head_dim=4)
        layer.close()
        with pytest.raises(RuntimeError, match="closed"):
            layer.append(*_make_token(n_heads=2, head_dim=4))


class TestMMapKVCache:
    def test_multi_layer_construction(self, tmp_path: Path):
        cache = MMapKVCache(
            tmp_path / "cache", n_layers=3, capacity=4, n_heads=2, head_dim=4,
        )
        try:
            assert len(cache) == 3
            assert cache.n_tokens == 0
            for i, layer in enumerate(cache):
                assert layer.root.name == f"L{i}"
        finally:
            cache.close()

    def test_append_at_layer(self, tmp_path: Path):
        cache = MMapKVCache(
            tmp_path / "cache", n_layers=2, capacity=4, n_heads=2, head_dim=4,
        )
        try:
            cache.append(0, *_make_token(n_heads=2, head_dim=4, val=5.0))
            cache.append(1, *_make_token(n_heads=2, head_dim=4, val=6.0))
            assert cache[0].n_tokens == 1
            assert cache[1].n_tokens == 1
        finally:
            cache.close()

    def test_disk_bytes_accounting(self, tmp_path: Path):
        cache = MMapKVCache(
            tmp_path / "cache", n_layers=2, capacity=8, n_heads=4, head_dim=16,
            dtype="float16",
        )
        try:
            # per-layer per-buffer: 8 × 4 × 16 × 2 bytes = 1024 → 2048/layer
            expected = 2 * 2 * 8 * 4 * 16 * 2
            assert cache.disk_bytes == expected
        finally:
            cache.close()

    def test_delete_class_method(self, tmp_path: Path):
        root = tmp_path / "cache"
        cache = MMapKVCache(
            root, n_layers=2, capacity=4, n_heads=2, head_dim=4,
        )
        cache.close()
        assert root.exists()
        MMapKVCache.delete(root)
        assert not root.exists()
        # Idempotent: second delete is a no-op
        MMapKVCache.delete(root)

    def test_invalid_n_layers_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            MMapKVCache(tmp_path / "x", n_layers=0, capacity=4,
                        n_heads=2, head_dim=4)

    def test_invalid_capacity_raises(self, tmp_path: Path):
        with pytest.raises(ValueError):
            MMapKVLayer(tmp_path / "L0", capacity=0, n_heads=2, head_dim=4)
