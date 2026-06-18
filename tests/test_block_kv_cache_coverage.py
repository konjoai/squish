"""Behavioral coverage for the error-handling, eviction, idempotency,
cold-tier last-logit round-trip, and mlx↔numpy helper paths of
``squish.kv.block_kv_cache`` left untested by the baseline suite.

MLX-only paths are guarded with ``pytest.importorskip("mlx.core")`` so they run
on the macOS + MLX coverage runner and skip cleanly on Linux.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.kv import block_kv_cache as bkc
from squish.kv.block_kv_cache import (
    BlockEntry,
    BlockKVCache,
    _to_numpy_f16,
    _to_numpy_f32,
    per_block_last_logits_from_full_logits,
    restore_blocks_to_cache,
    slice_cache_into_blocks,
)


@pytest.fixture
def tmp_cache_dir():
    d = Path(tempfile.mkdtemp(prefix="block_kv_cov_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _cache(tmp_cache_dir, **kw):
    kw.setdefault("block_size", 8)
    kw.setdefault("model_key", "cov-model")
    return BlockKVCache(cache_dir=tmp_cache_dir, **kw)


def _blocks(n_blocks, block_size=8, n_layers=2, n_heads=2, head_dim=8):
    per_k, per_v = [], []
    for _ in range(n_blocks):
        per_k.append([np.ones((1, n_heads, block_size, head_dim), np.float16)
                      for _ in range(n_layers)])
        per_v.append([np.ones((1, n_heads, block_size, head_dim), np.float16)
                      for _ in range(n_layers)])
    return per_k, per_v


def test_chmod_failure_is_swallowed(tmp_cache_dir, monkeypatch):
    monkeypatch.setattr(Path, "chmod", lambda self, mode: (_ for _ in ()).throw(OSError("denied")))
    # chmod raising must not prevent construction (line 128 except path).
    cache = _cache(tmp_cache_dir)
    assert cache.block_size() == 8


def test_manifest_corrupt_recreates_cache(tmp_cache_dir):
    cache = _cache(tmp_cache_dir)
    # Corrupt the manifest, then re-instantiate: JSONDecodeError → clear + rewrite.
    (tmp_cache_dir / "manifest.json").write_text("{not valid json")
    cache2 = _cache(tmp_cache_dir)
    m = json.loads((tmp_cache_dir / "manifest.json").read_text())
    assert m["block_size"] == 8 and m["version"] == bkc._CACHE_VERSION
    assert len(cache2._hot) == 0


def test_manifest_write_failure_is_swallowed(tmp_cache_dir, monkeypatch):
    monkeypatch.setattr(
        Path, "write_text",
        lambda self, *a, **k: (_ for _ in ()).throw(OSError("ro fs")),
    )
    # Manifest write failing must not crash construction (lines 444-445).
    cache = _cache(tmp_cache_dir)
    assert isinstance(cache.stats(), dict)


def test_store_is_idempotent_on_known_hashes(tmp_cache_dir):
    cache = _cache(tmp_cache_dir)
    ids = list(range(200, 232))  # 32 tokens → 4 blocks of 8
    pk, pv = _blocks(4)
    cache.store_blocks(ids, pk, pv)
    hot_after_first = len(cache._hot)
    # Second identical store must skip every block (line 275 continue).
    cache.store_blocks(ids, pk, pv)
    assert len(cache._hot) == hot_after_first


def test_store_skips_block_with_empty_keys(tmp_cache_dir):
    cache = _cache(tmp_cache_dir)
    ids = list(range(300, 332))
    pk, pv = _blocks(4)
    pk[1] = []  # block 1 has no layers → must be skipped (line 279)
    pv[1] = []
    cache.store_blocks(ids, pk, pv)
    # Block 0 stored; block 1 skipped, which breaks the contiguous chain at
    # lookup so only the first block matches.
    match = cache.lookup_prefix(ids)
    assert match.matched_tokens == 8


def test_store_with_bad_logit_stores_without_logit(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    ids = list(range(400, 408))  # one block
    pk, pv = _blocks(1)
    # A logit whose conversion raises is caught (lines 286-291) and the block is
    # stored without a logit rather than failing the whole store.
    monkeypatch.setattr(
        bkc, "_to_numpy_f32",
        lambda arr: (_ for _ in ()).throw(ValueError("unconvertible logit")),
    )
    cache.store_blocks(ids, pk, pv, per_block_last_logits=[np.ones(4, np.float32)])
    entry = cache._get_block(cache.chain_hash(ids)[0])
    assert entry is not None
    assert entry.last_logit is None


def test_cold_read_loads_last_logit(tmp_cache_dir):
    cache = _cache(tmp_cache_dir)
    ids = list(range(500, 508))
    pk, pv = _blocks(1)
    logit = np.arange(11, dtype=np.float32)
    cache.store_blocks(ids, pk, pv, per_block_last_logits=[logit])
    # Fresh instance → hot tier empty → lookup reads from cold, exercising the
    # "last_logit" branch (lines 217-219).
    cache2 = _cache(tmp_cache_dir)
    entry = cache2._get_block(cache2.chain_hash(ids)[0])
    assert entry is not None and entry.last_logit is not None
    np.testing.assert_array_equal(entry.last_logit, logit)


def test_cold_read_utime_failure_is_swallowed(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    ids = list(range(600, 608))
    pk, pv = _blocks(1)
    cache.store_blocks(ids, pk, pv)
    h = cache.chain_hash(ids)[0]
    monkeypatch.setattr(os, "utime", lambda *a, **k: (_ for _ in ()).throw(OSError("no utime")))
    cache2 = _cache(tmp_cache_dir)
    # utime failing during cold read must not lose the entry (lines 230-232).
    entry = cache2._get_block(h)
    assert entry is not None and entry.n_tokens == 8


def test_cold_read_corrupt_file_returns_none(tmp_cache_dir):
    cache = _cache(tmp_cache_dir)
    ids = list(range(700, 708))
    pk, pv = _blocks(1)
    cache.store_blocks(ids, pk, pv)
    h = cache.chain_hash(ids)[0]
    # Overwrite the cold .npz with garbage → np.load raises → warning + None
    # (lines 235-237).
    cold_path = cache._cold_path(h)
    cold_path.write_bytes(b"not a real npz file")
    cache2 = _cache(tmp_cache_dir)
    assert cache2._read_cold(h) is None
    # And the missing-file branch (lines 207-208).
    assert cache2._read_cold("0" * 64) is None


def test_hot_eviction_when_over_budget(tmp_cache_dir):
    # Tiny hot budget forces eviction down to one entry (lines 317-319).
    cache = _cache(tmp_cache_dir, hot_max_bytes=1)
    ids = list(range(800, 832))
    pk, pv = _blocks(4)
    cache.store_blocks(ids, pk, pv)
    assert len(cache._hot) == 1  # evicted down to the floor of 1
    assert cache._hot_bytes <= cache._hot[next(reversed(cache._hot))].nbytes


def test_add_to_hot_existing_hash_bumps_lru_without_double_counting(tmp_cache_dir):
    cache = _cache(tmp_cache_dir, hot_max_bytes=10**12)
    entry = BlockEntry(hash="abc", n_layers=1, n_tokens=8,
                       keys=[np.ones((1, 1, 8, 8), np.float16)],
                       values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
    cache._add_to_hot(entry)
    bytes_once = cache._hot_bytes
    cache._add_to_hot(entry)  # same hash → move_to_end, no re-accounting (310-311)
    assert cache._hot_bytes == bytes_once
    assert len(cache._hot) == 1


def test_cold_write_failure_is_swallowed(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    monkeypatch.setattr(bkc.np, "savez",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    ids = list(range(900, 908))
    pk, pv = _blocks(1)
    # Cold write failing (lines 338-344) leaves the block in hot but not cold.
    cache.store_blocks(ids, pk, pv)
    h = cache.chain_hash(ids)[0]
    assert h in cache._hot
    assert not cache._cold_path(h).exists()


def test_evict_cold_lru_removes_oldest_until_under_budget(tmp_cache_dir):
    cache = _cache(tmp_cache_dir)
    # Write three cold files directly with distinct atimes.
    paths = []
    for i in range(3):
        e = BlockEntry(hash=f"{i:064d}", n_layers=1, n_tokens=8,
                       keys=[np.ones((1, 1, 8, 8), np.float16)],
                       values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
        cache._write_cold(e)
        paths.append(cache._cold_path(e.hash))
    one_size = paths[0].stat().st_size
    # Make path[0] the oldest, path[2] newest.
    os.utime(paths[0], (1, 1))
    os.utime(paths[1], (2, 2))
    os.utime(paths[2], (3, 3))
    # Budget that holds ~2 files → the oldest (paths[0]) must be evicted.
    cache._cold_max_bytes = one_size * 2 + 1
    evicted = cache._evict_cold_lru()
    assert evicted == 1
    assert not paths[0].exists()
    assert paths[2].exists()


def test_evict_cold_lru_noop_when_under_budget(tmp_cache_dir):
    cache = _cache(tmp_cache_dir, cold_max_bytes=10**12)
    e = BlockEntry(hash="f" * 64, n_layers=1, n_tokens=8,
                   keys=[np.ones((1, 1, 8, 8), np.float16)],
                   values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
    cache._write_cold(e)
    # Under budget → early return 0 (lines 370-371).
    assert cache._evict_cold_lru() == 0


def test_write_cold_triggers_random_eviction(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    calls = {"n": 0}
    monkeypatch.setattr(cache, "_evict_cold_lru", lambda: calls.__setitem__("n", calls["n"] + 1))
    # _write_cold does a local `import random`; patch the stdlib module so its
    # random() always trips the 1/32 eviction sampler (line 348).
    monkeypatch.setattr("random.random", lambda: 0.0)
    e = BlockEntry(hash="a" * 64, n_layers=1, n_tokens=8,
                   keys=[np.ones((1, 1, 8, 8), np.float16)],
                   values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
    cache._write_cold(e)
    assert calls["n"] == 1


def test_stats_reports_configuration(tmp_cache_dir):
    cache = _cache(tmp_cache_dir, hot_max_bytes=123, cold_max_bytes=456)
    s = cache.stats()
    assert s == {
        "block_size": 8, "hot_entries": 0, "hot_bytes": 0,
        "hot_max_bytes": 123, "cold_max_bytes": 456, "model_key": "cov-model",
    }


def test_clear_swallows_unlink_and_rmdir_errors(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    ids = list(range(1000, 1008))
    pk, pv = _blocks(1)
    cache.store_blocks(ids, pk, pv)  # creates a cold subdir + file
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked")))
    monkeypatch.setattr(Path, "rmdir", lambda self: (_ for _ in ()).throw(OSError("not empty")))
    # Both error paths (lines 406, 410) must be swallowed.
    cache.clear()
    assert len(cache._hot) == 0  # hot is still cleared


def test_to_numpy_f16_and_f32_numpy_passthrough():
    arr = np.ones((2, 3), dtype=np.float32)
    out16 = _to_numpy_f16(arr)
    out32 = _to_numpy_f32(arr.astype(np.float64))
    assert out16.dtype == np.float16
    assert out32.dtype == np.float32


def test_per_block_last_logits_from_full_logits_numpy():
    n_blocks, block_size, vocab = 3, 4, 5
    total = n_blocks * block_size
    # Encode the position index into the array so we can assert the right
    # last-position slice is taken for each block.
    full = np.arange(total, dtype=np.float32).reshape(1, total, 1) * np.ones((1, 1, vocab))
    out = per_block_last_logits_from_full_logits(full, n_blocks, block_size)
    assert len(out) == n_blocks
    for i in range(n_blocks):
        expected_pos = (i + 1) * block_size - 1
        assert float(out[i][0]) == expected_pos


def test_slice_cache_into_blocks_numpy():
    block_size, n_blocks, n_layers, n_heads, head_dim = 4, 2, 2, 2, 3
    total = block_size * n_blocks

    class _Layer:
        def __init__(self):
            self.keys = np.arange(total, dtype=np.float16).reshape(1, 1, total, 1) * np.ones(
                (1, n_heads, 1, head_dim), np.float16)
            self.values = self.keys + 1

    cache = [_Layer() for _ in range(n_layers)]
    pk, pv = slice_cache_into_blocks(cache, block_size, n_blocks, n_layers)
    assert len(pk) == n_blocks and len(pk[0]) == n_layers
    assert pk[0][0].shape == (1, n_heads, block_size, head_dim)
    # Block 1 starts at token index block_size.
    assert float(pk[1][0][0, 0, 0, 0]) == block_size


def test_slice_cache_into_blocks_returns_empty_on_missing_kv():
    class _NoKV:
        keys = None
        values = None

    pk, pv = slice_cache_into_blocks([_NoKV()], block_size=4, n_blocks=1, n_layers=1)
    assert pk == [] and pv == []


def test_restore_blocks_returns_none_on_empty():
    assert restore_blocks_to_cache(cache=[], matched_blocks=[]) is None


def test_to_numpy_f16_f32_mlx_branch():
    mx = pytest.importorskip("mlx.core")
    arr = mx.ones((2, 3), dtype=mx.float32)
    out16 = _to_numpy_f16(arr)
    out32 = _to_numpy_f32(arr)
    assert out16.dtype == np.float16 and out16.shape == (2, 3)
    assert out32.dtype == np.float32 and out32.shape == (2, 3)


def test_to_numpy_f16_bfloat16_fallback():
    mx = pytest.importorskip("mlx.core")
    # numpy has no native bfloat16 → the direct np.array path raises and the
    # function falls back through float32 (lines 463-464 / 484-485).
    arr = mx.ones((2, 2), dtype=mx.bfloat16)
    out = _to_numpy_f16(arr)
    assert out.dtype == np.float16 and out.shape == (2, 2)


def _restore_layer_cls():
    class _Layer:
        def __init__(self):
            self.keys = None
            self.values = None
            self.offset = 0
    return _Layer


def test_restore_blocks_to_cache_success():
    mx = pytest.importorskip("mlx.core")
    n_layers, n_heads, head_dim, block_size = 2, 2, 4, 8
    blocks = []
    for _ in range(2):
        blocks.append(BlockEntry(
            hash="h", n_layers=n_layers, n_tokens=block_size,
            keys=[np.ones((1, n_heads, block_size, head_dim), np.float16) for _ in range(n_layers)],
            values=[np.ones((1, n_heads, block_size, head_dim), np.float16) for _ in range(n_layers)],
            nbytes=0,
        ))
    Layer = _restore_layer_cls()
    cache = [Layer() for _ in range(n_layers)]
    result = restore_blocks_to_cache(cache, blocks, target_dtype=mx.float16)
    assert result == (n_layers, 2 * block_size)
    for layer in cache:
        assert layer.keys.shape[2] == 2 * block_size
        assert layer.keys.dtype == mx.float16
        assert layer.offset == 2 * block_size


def test_restore_blocks_layer_count_mismatch_returns_none():
    pytest.importorskip("mlx.core")
    blocks = [BlockEntry(hash="h", n_layers=2, n_tokens=8,
                         keys=[np.ones((1, 2, 8, 4), np.float16)] * 2,
                         values=[np.ones((1, 2, 8, 4), np.float16)] * 2, nbytes=0)]
    # cache has 1 layer but entry declares 2 → mismatch (lines 582-587).
    assert restore_blocks_to_cache([object()], blocks) is None


def test_restore_blocks_unsupported_layer_returns_none():
    pytest.importorskip("mlx.core")
    blocks = [BlockEntry(hash="h", n_layers=1, n_tokens=8,
                         keys=[np.ones((1, 2, 8, 4), np.float16)],
                         values=[np.ones((1, 2, 8, 4), np.float16)], nbytes=0)]
    # Single layer that lacks .keys/.values attributes → None (lines 599-600).
    assert restore_blocks_to_cache([object()], blocks) is None


def test_restore_blocks_layer_without_offset_attr():
    mx = pytest.importorskip("mlx.core")

    class _LayerNoOffset:
        keys = None
        values = None
        # deliberately no `offset` attribute → the `if hasattr(...)` is False

    block = BlockEntry(hash="h", n_layers=1, n_tokens=8,
                       keys=[np.ones((1, 2, 8, 4), np.float16)],
                       values=[np.ones((1, 2, 8, 4), np.float16)], nbytes=0)
    cache = [_LayerNoOffset()]
    # Restore still succeeds; the offset assignment is simply skipped (603→589).
    assert restore_blocks_to_cache(cache, [block]) == (1, 8)
    assert cache[0].keys.shape[2] == 8


def test_to_numpy_helpers_raise_without_mlx(monkeypatch):
    import sys
    # Force `import mlx.core` to raise ImportError even where MLX is installed.
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    with pytest.raises(TypeError, match="mlx not available"):
        _to_numpy_f16([1, 2, 3])  # non-ndarray → tries mlx → ImportError (465-466)
    with pytest.raises(TypeError, match="mlx not available"):
        _to_numpy_f32([1, 2, 3])  # (486-487)


def test_restore_blocks_returns_none_without_mlx(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mlx.core", None)
    block = BlockEntry(hash="h", n_layers=1, n_tokens=8,
                       keys=[np.ones((1, 2, 8, 4), np.float16)],
                       values=[np.ones((1, 2, 8, 4), np.float16)], nbytes=0)
    # mlx import fails → restore bails out with None (lines 579-580).
    assert restore_blocks_to_cache([object()], [block]) is None


def test_to_numpy_f32_bfloat16_fallback():
    mx = pytest.importorskip("mlx.core")
    out = _to_numpy_f32(mx.ones((2, 2), dtype=mx.bfloat16))
    assert out.dtype == np.float32 and out.shape == (2, 2)


def test_write_cold_tmp_unlink_failure_is_swallowed(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    monkeypatch.setattr(bkc.np, "savez",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("savez boom")))
    monkeypatch.setattr(Path, "unlink",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("unlink boom")))
    e = BlockEntry(hash="b" * 64, n_layers=1, n_tokens=8,
                   keys=[np.ones((1, 1, 8, 8), np.float16)],
                   values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
    # savez fails → handler tries tmp.unlink which also fails → inner except (342).
    cache._write_cold(e)
    assert not cache._cold_path(e.hash).exists()


def test_evict_cold_lru_skips_nondir_and_handles_errors(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    # A stray file directly under the cache dir must be skipped (line 363).
    (tmp_cache_dir / "stray.txt").write_text("x")
    for i in range(3):
        e = BlockEntry(hash=f"{i:064d}", n_layers=1, n_tokens=8,
                       keys=[np.ones((1, 1, 8, 8), np.float16)],
                       values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
        cache._write_cold(e)
    # A non-.npz file *inside* a hash subdir must be skipped (line 363).
    (tmp_cache_dir / "00" / "notes.txt").write_text("x")
    # Force every unlink during eviction to fail → the OSError is swallowed and
    # nothing is removed (lines 374-383 loop body + 381 except).
    cache._cold_max_bytes = 0
    monkeypatch.setattr(Path, "unlink",
                        lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked")))
    evicted = cache._evict_cold_lru()
    assert evicted == 0  # all unlinks failed


def test_evict_cold_lru_skips_unstattable_file(tmp_cache_dir, monkeypatch):
    cache = _cache(tmp_cache_dir)
    e = BlockEntry(hash="c" * 64, n_layers=1, n_tokens=8,
                   keys=[np.ones((1, 1, 8, 8), np.float16)],
                   values=[np.ones((1, 1, 8, 8), np.float16)], nbytes=128)
    cache._write_cold(e)
    real_stat = Path.stat

    def flaky_stat(self, *a, **k):
        if self.suffix == ".npz":
            raise OSError("stat failed")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    # The .npz file can't be stat'd → it is skipped (lines 366-367); with no
    # measurable usage the function returns 0.
    assert cache._evict_cold_lru() == 0
