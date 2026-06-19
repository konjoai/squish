"""Behavioral coverage for ``squish.kv.mmap_cache`` — the mmap-backed KV layer
and multi-layer cache: validation, meta reload edge cases, append/get/range/
evict bounds, flush/close lifecycle, and the cache container. Pure-Python
(numpy memmap + filesystem); no MLX.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from squish.kv import mmap_cache as mc
from squish.kv.mmap_cache import MMapKVCache, MMapKVLayer


def _layer(tmp_path, capacity=4, n_heads=2, head_dim=3):
    return MMapKVLayer(tmp_path / "L", capacity=capacity, n_heads=n_heads, head_dim=head_dim)


def _tok(n_heads=2, head_dim=3, fill=1.0):
    return np.full((n_heads, head_dim), fill, dtype=np.float16)


# ── construction validation ─────────────────────────────────────────────────


def test_invalid_capacity_raises(tmp_path):
    with pytest.raises(ValueError, match="capacity must be"):
        MMapKVLayer(tmp_path / "x", capacity=0, n_heads=2, head_dim=2)


def test_invalid_dims_raise(tmp_path):
    with pytest.raises(ValueError, match="n_heads and head_dim"):
        MMapKVLayer(tmp_path / "x", capacity=4, n_heads=0, head_dim=2)


# ── meta reload edge cases (meta present, bins absent) ───────────────────────


def test_meta_shape_mismatch_reinitialises(tmp_path):
    root = tmp_path / "L"
    root.mkdir()
    (root / mc._META_FILENAME).write_text(json.dumps(
        {"capacity": 999, "n_heads": 9, "head_dim": 9, "dtype": "float16", "n_tokens": 7}))
    layer = MMapKVLayer(root, capacity=4, n_heads=2, head_dim=3)  # mismatch → reinit (124-127)
    assert layer.n_tokens == 0


def test_meta_corrupt_is_tolerated(tmp_path):
    root = tmp_path / "L"
    root.mkdir()
    (root / mc._META_FILENAME).write_text("{ not json")
    layer = MMapKVLayer(root, capacity=4, n_heads=2, head_dim=3)  # JSONDecodeError (128-129)
    assert layer.n_tokens == 0


def test_meta_outlives_bins_resets_tokens(tmp_path):
    root = tmp_path / "L"
    root.mkdir()
    (root / mc._META_FILENAME).write_text(json.dumps(
        {"capacity": 4, "n_heads": 2, "head_dim": 3, "dtype": "float16", "n_tokens": 3}))
    layer = MMapKVLayer(root, capacity=4, n_heads=2, head_dim=3)  # bins missing → reset (135-140)
    assert layer.n_tokens == 0


def test_meta_reload_preserves_tokens(tmp_path):
    layer = _layer(tmp_path)
    layer.append(_tok(), _tok())
    layer.flush()
    layer.close()
    reopened = _layer(tmp_path)  # meta + bins present → n_tokens restored
    assert reopened.n_tokens == 1


# ── properties ──────────────────────────────────────────────────────────────


def test_meta_and_properties(tmp_path):
    layer = _layer(tmp_path, capacity=4, n_heads=2, head_dim=3)
    m = layer.meta
    assert m.capacity == 4 and m.n_heads == 2 and m.head_dim == 3
    assert layer.capacity == 4 and layer.root == (tmp_path / "L")
    assert layer.disk_bytes == 2 * (4 * 2 * 3 * np.dtype("float16").itemsize)


# ── append / get / get_range ────────────────────────────────────────────────


def test_append_and_get(tmp_path):
    layer = _layer(tmp_path)
    layer.append(_tok(fill=1.0), _tok(fill=2.0))
    k, v = layer.get(0)
    assert k[0, 0] == 1.0 and v[0, 0] == 2.0


def test_append_bad_key_shape(tmp_path):
    layer = _layer(tmp_path)
    with pytest.raises(ValueError, match="key shape"):
        layer.append(np.ones((5, 5), np.float16), _tok())


def test_append_bad_value_shape(tmp_path):
    layer = _layer(tmp_path)
    with pytest.raises(ValueError, match="value shape"):
        layer.append(_tok(), np.ones((5, 5), np.float16))  # 196


def test_append_overflow(tmp_path):
    layer = _layer(tmp_path, capacity=1)
    layer.append(_tok(), _tok())
    with pytest.raises(OverflowError, match="full at capacity"):
        layer.append(_tok(), _tok())


def test_get_out_of_range(tmp_path):
    layer = _layer(tmp_path)
    with pytest.raises(IndexError):
        layer.get(0)  # empty → 215


def test_get_range(tmp_path):
    layer = _layer(tmp_path)
    for i in range(3):
        layer.append(_tok(fill=i), _tok(fill=i))
    ks, vs = layer.get_range(1, 3)
    assert ks.shape[0] == 2 and ks[0, 0, 0] == 1
    with pytest.raises(IndexError):
        layer.get_range(0, 99)  # 228


# ── evict ───────────────────────────────────────────────────────────────────


def test_evict_oldest_shifts(tmp_path):
    layer = _layer(tmp_path)
    for i in range(3):
        layer.append(_tok(fill=i), _tok(fill=i))
    dropped = layer.evict_oldest(2)
    assert dropped == 2 and layer.n_tokens == 1
    k, _ = layer.get(0)
    assert k[0, 0] == 2  # token index 2 shifted to front


def test_evict_negative_raises(tmp_path):
    with pytest.raises(ValueError, match="n must be"):
        _layer(tmp_path).evict_oldest(-1)  # 246


def test_evict_zero_returns_zero(tmp_path):
    assert _layer(tmp_path).evict_oldest(0) == 0  # 250


# ── flush / close / context manager ──────────────────────────────────────────


def test_flush(tmp_path):
    layer = _layer(tmp_path)
    layer.append(_tok(), _tok())
    layer.flush()  # 257-263 — no raise
    assert layer.n_tokens == 1


def test_close_idempotent_and_requires_open(tmp_path):
    layer = _layer(tmp_path)
    layer.close()
    layer.close()  # already closed → early return (267-268)
    with pytest.raises(RuntimeError, match="closed"):
        layer.append(_tok(), _tok())  # _require_open


def test_close_swallows_flush_error(tmp_path):
    layer = _layer(tmp_path)

    class _BadMap:
        def flush(self):
            raise OSError("flush boom")

    layer._k = _BadMap()
    layer._v = _BadMap()
    layer.close()  # flush errors swallowed (273-274)
    assert layer._closed is True


def test_context_manager(tmp_path):
    with _layer(tmp_path) as layer:  # __enter__ (281)
        layer.append(_tok(), _tok())
    assert layer._closed is True  # __exit__ → close (284)


# ── MMapKVCache (multi-layer) ────────────────────────────────────────────────


def test_cache_invalid_n_layers(tmp_path):
    with pytest.raises(ValueError, match="n_layers must be"):
        MMapKVCache(tmp_path, n_layers=0, capacity=4, n_heads=2, head_dim=3)


def test_cache_container_api(tmp_path):
    cache = MMapKVCache(tmp_path, n_layers=2, capacity=4, n_heads=2, head_dim=3)
    assert len(cache) == 2 and cache.n_layers == 2  # 345
    assert cache.root == tmp_path                    # 341
    assert isinstance(cache[0], MMapKVLayer)
    assert [layer for layer in cache] == cache._layers  # __iter__
    cache.append(0, _tok(), _tok())
    cache.append(1, _tok(), _tok())
    assert cache.n_tokens == 1
    assert cache.disk_bytes == sum(layer.disk_bytes for layer in cache)
    cache.flush()   # 361-363
    cache.close()


def test_cache_n_tokens_empty_layers(tmp_path):
    cache = MMapKVCache(tmp_path, n_layers=1, capacity=4, n_heads=2, head_dim=3)
    cache._layers = []  # representative-layer guard
    assert cache.n_tokens == 0


def test_cache_delete(tmp_path):
    root = tmp_path / "cache"
    cache = MMapKVCache(root, n_layers=1, capacity=4, n_heads=2, head_dim=3)
    cache.close()
    assert root.exists()
    MMapKVCache.delete(root)
    assert not root.exists()
    MMapKVCache.delete(root)  # no-op when absent
