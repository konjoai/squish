"""Unit tests for squish.kv.block_kv_cache (v5 + v5.1 last-logit field)."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from squish.kv.block_kv_cache import (
    BlockEntry,
    BlockKVCache,
    PrefixMatch,
    _to_numpy_f16,
    _to_numpy_f32,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_cache_dir():
    d = Path(tempfile.mkdtemp(prefix="block_kv_test_"))
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def small_cache(tmp_cache_dir):
    return BlockKVCache(
        cache_dir=tmp_cache_dir,
        block_size=8,
        model_key="test-model-v51",
    )


# ── Hash chain ──────────────────────────────────────────────────────────────────


def test_chain_hash_is_deterministic(small_cache):
    ids = list(range(100, 132))
    h1 = small_cache.chain_hash(ids)
    h2 = small_cache.chain_hash(ids)
    assert h1 == h2
    assert len(h1) == len(ids) // small_cache.block_size()


def test_chain_hash_prefix_property(small_cache):
    ids_a = list(range(100, 132))
    ids_b = list(range(100, 124)) + list(range(900, 908))
    h_a = small_cache.chain_hash(ids_a)
    h_b = small_cache.chain_hash(ids_b)
    # First 3 blocks (24 tokens) match; the 4th differs.
    assert h_a[:3] == h_b[:3]
    assert h_a[3] != h_b[3]


def test_chain_hash_changes_with_model_key(tmp_cache_dir):
    a = BlockKVCache(cache_dir=tmp_cache_dir, block_size=8, model_key="A")
    # Recreate (manifest would mismatch otherwise) with same dir but different
    # model_key — clear forces a fresh manifest with the new key.
    a.clear()
    b = BlockKVCache(cache_dir=tmp_cache_dir, block_size=8, model_key="B")
    ids = list(range(100, 132))
    assert a.chain_hash(ids) != b.chain_hash(ids)


# ── Store / lookup ──────────────────────────────────────────────────────────────


def _fake_block_arrays(block_size, n_layers=2, n_heads=4, head_dim=16):
    keys = [np.random.randn(1, n_heads, block_size, head_dim).astype(np.float16)
            for _ in range(n_layers)]
    vals = [np.random.randn(1, n_heads, block_size, head_dim).astype(np.float16)
            for _ in range(n_layers)]
    return keys, vals


def test_store_then_lookup_full_match(small_cache):
    ids = list(range(100, 132))  # 32 tokens, 4 blocks of 8
    per_b_k = []
    per_b_v = []
    for _ in range(4):
        k, v = _fake_block_arrays(small_cache.block_size())
        per_b_k.append(k)
        per_b_v.append(v)
    small_cache.store_blocks(ids, per_b_k, per_b_v)
    match = small_cache.lookup_prefix(ids)
    assert match.matched_tokens == 32


def test_partial_match(small_cache):
    ids_a = list(range(100, 132))  # 32 tokens
    ids_b = list(range(100, 124)) + list(range(900, 908))  # 32 tokens, last block differs
    per_b_k_a, per_b_v_a = zip(
        *[_fake_block_arrays(small_cache.block_size()) for _ in range(4)],
        strict=True,
    )
    small_cache.store_blocks(ids_a, list(per_b_k_a), list(per_b_v_a))
    match = small_cache.lookup_prefix(ids_b)
    # First 3 blocks (24 tokens) shared with ids_a; 4th block is new.
    assert match.matched_tokens == 24


def test_cold_tier_survives_reinstance(tmp_cache_dir):
    ids = list(range(100, 132))
    per_b_k, per_b_v = zip(
        *[_fake_block_arrays(8) for _ in range(4)], strict=True,
    )
    a = BlockKVCache(cache_dir=tmp_cache_dir, block_size=8, model_key="mk")
    a.store_blocks(ids, list(per_b_k), list(per_b_v))
    # Drop the instance, restart fresh
    del a
    b = BlockKVCache(cache_dir=tmp_cache_dir, block_size=8, model_key="mk")
    match = b.lookup_prefix(ids)
    assert match.matched_tokens == 32


def test_lookup_requires_full_block(small_cache):
    # Prompts shorter than one block always miss.
    short_ids = list(range(5))
    match = small_cache.lookup_prefix(short_ids)
    assert match.matched_tokens == 0


def test_manifest_block_size_mismatch_clears(tmp_cache_dir):
    a = BlockKVCache(cache_dir=tmp_cache_dir, block_size=8, model_key="mk")
    ids = list(range(0, 32))
    per_b_k, per_b_v = zip(
        *[_fake_block_arrays(8) for _ in range(4)], strict=True,
    )
    a.store_blocks(ids, list(per_b_k), list(per_b_v))
    assert a.lookup_prefix(ids).matched_tokens == 32
    # New instance with different block_size: cache should clear.
    b = BlockKVCache(cache_dir=tmp_cache_dir, block_size=16, model_key="mk")
    assert b.lookup_prefix(list(range(0, 64))).matched_tokens == 0


# ── v5.1 per-block last_logit ──────────────────────────────────────────────────


def test_v5_1_store_and_load_last_logit(small_cache):
    ids = list(range(0, 32))
    per_b_k, per_b_v = zip(
        *[_fake_block_arrays(8) for _ in range(4)], strict=True,
    )
    # One distinct logit per block (vocab=10 for simplicity)
    per_b_logits = [
        np.random.randn(10).astype(np.float32) for _ in range(4)
    ]
    small_cache.store_blocks(
        ids, list(per_b_k), list(per_b_v),
        per_block_last_logits=per_b_logits,
    )
    match = small_cache.lookup_prefix(ids)
    assert match.matched_tokens == 32
    # Every matched block must carry the stored logit.
    for stored_logit, blk in zip(per_b_logits, match.matched_blocks, strict=True):
        assert blk.last_logit is not None
        np.testing.assert_allclose(blk.last_logit, stored_logit, atol=1e-5)


def test_v5_1_legacy_entry_without_logit(small_cache):
    # Store without logits, then read; last_logit must be None — not crash.
    ids = list(range(0, 16))
    per_b_k, per_b_v = zip(
        *[_fake_block_arrays(8) for _ in range(2)], strict=True,
    )
    small_cache.store_blocks(ids, list(per_b_k), list(per_b_v))
    match = small_cache.lookup_prefix(ids)
    for blk in match.matched_blocks:
        assert blk.last_logit is None


def test_v5_1_partial_logits(small_cache):
    """If only some blocks have logits, the rest must still load without them."""
    ids = list(range(0, 32))
    per_b_k, per_b_v = zip(
        *[_fake_block_arrays(8) for _ in range(4)], strict=True,
    )
    # First two blocks have logits, last two don't
    per_b_logits = [
        np.random.randn(10).astype(np.float32),
        np.random.randn(10).astype(np.float32),
        None,
        None,
    ]
    small_cache.store_blocks(
        ids, list(per_b_k), list(per_b_v),
        per_block_last_logits=per_b_logits,
    )
    match = small_cache.lookup_prefix(ids)
    assert match.matched_tokens == 32
    assert match.matched_blocks[0].last_logit is not None
    assert match.matched_blocks[1].last_logit is not None
    assert match.matched_blocks[2].last_logit is None
    assert match.matched_blocks[3].last_logit is None


# ── _to_numpy helpers ──────────────────────────────────────────────────────────


def test_to_numpy_f16_passthrough():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float16)
    out = _to_numpy_f16(arr)
    np.testing.assert_array_equal(out, arr)


def test_to_numpy_f32_passthrough():
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = _to_numpy_f32(arr)
    np.testing.assert_array_equal(out, arr)


def test_block_entry_repr():
    entry = BlockEntry(
        hash="abc",
        n_layers=2,
        n_tokens=8,
        keys=[np.zeros((1, 1, 8, 4), dtype=np.float16)],
        values=[np.zeros((1, 1, 8, 4), dtype=np.float16)],
        nbytes=128,
    )
    assert entry.last_logit is None  # default
    assert entry.n_tokens == 8


# ── PrefixMatch dataclass ─────────────────────────────────────────────────────


def test_prefix_match_empty():
    m = PrefixMatch(matched_blocks=[], matched_tokens=0)
    assert m.matched_tokens == 0
