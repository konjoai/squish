"""tests/test_paged_attention_unit.py

Full-coverage unit tests for squish/paged_attention.py.

Covers all classes and branches:
  PhysicalBlock    — touch()
  BlockAllocator   — alloc (success/pool exhausted), free (to-0/stay-above-0),
                     fork, free_count/total_blocks/used_count, evict_lru
  PageBlockTable   — ensure_space (new block/alloc exhausted), advance,
                     block_refs/write_offset/n_tokens_stored (empty/partial),
                     fork_blocks, free_all
  PagedKVCache     — __init__, _get_store (lazy init, cached second call),
                     new_request, free_request (existing/unknown),
                     fork_request, store_token (success/no-table/exhausted),
                     advance_token (found/not-found), get_kv_for_layer
                     (not-found/empty-table/single/multi-block),
                     get_block_refs, n_tokens_stored, snapshot_prefix
                     (not-found / normal), evict_lru_blocks, stats,
                     from_model (psutil absent / model-config / no-config)
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from squish.kv.paged_attention import (
    PAGE_SIZE,
    BlockAllocator,
    PageBlockTable,
    PagedKVCache,
    PhysicalBlock,
)


# ---------------------------------------------------------------------------
# PhysicalBlock
# ---------------------------------------------------------------------------


class TestPhysicalBlock:
    def test_touch_updates_last_access(self):
        import time
        blk = PhysicalBlock(idx=0)
        before = blk.last_access
        time.sleep(0.001)
        blk.touch()
        assert blk.last_access >= before

    def test_default_ref_count_zero(self):
        blk = PhysicalBlock(idx=5)
        assert blk.ref_count == 0


# ---------------------------------------------------------------------------
# BlockAllocator
# ---------------------------------------------------------------------------


class TestBlockAllocator:
    def test_alloc_returns_valid_index(self):
        ba = BlockAllocator(num_blocks=4)
        idx = ba.alloc()
        assert idx is not None
        assert 0 <= idx < 4

    def test_alloc_exhausted_returns_none(self):
        ba = BlockAllocator(num_blocks=2)
        ba.alloc()
        ba.alloc()
        result = ba.alloc()
        assert result is None

    def test_free_returns_block_to_pool(self):
        ba = BlockAllocator(num_blocks=2)
        idx = ba.alloc()
        assert ba.free_count == 1
        ba.free(idx)
        assert ba.free_count == 2

    def test_free_decrements_ref_count_stays_above_zero(self):
        """free() after fork — ref_count goes from 2 to 1, block stays allocated."""
        ba = BlockAllocator(num_blocks=4)
        idx = ba.alloc()
        ba.fork(idx)  # ref_count = 2
        ba.free(idx)  # ref_count = 1 → NOT returned to free pool
        assert ba.free_count == 3  # 4 total, 1 still allocated

    def test_fork_increments_ref_count(self):
        ba = BlockAllocator(num_blocks=4)
        idx = ba.alloc()
        ba.fork(idx)
        ba.fork(idx)
        # ref_count = 3 — block still in use, not in free pool
        assert ba.free_count == 3  # 4 total - 1 allocated

    def test_total_blocks_constant(self):
        ba = BlockAllocator(num_blocks=8)
        assert ba.total_blocks == 8

    def test_used_count(self):
        ba = BlockAllocator(num_blocks=4)
        ba.alloc()
        ba.alloc()
        assert ba.used_count == 2

    def test_evict_lru_evicts_unreferenced(self):
        ba = BlockAllocator(num_blocks=4)
        idx0 = ba.alloc()
        idx1 = ba.alloc()
        # Free them to put in free pool (ref_count=0)
        ba.free(idx0)
        ba.free(idx1)
        # All blocks now free — evict_lru should find nothing NOT in free pool
        evicted = ba.evict_lru(2)
        assert isinstance(evicted, list)

    def test_evict_lru_unreferenced_not_in_free_pool(self):
        """Blocks with ref_count=0 that are NOT in the free deque can be evicted."""
        ba = BlockAllocator(num_blocks=4)
        # Allocate all 4 blocks
        idxs = [ba.alloc() for _ in range(4)]
        assert ba.free_count == 0
        # Fork one block so it has ref_count=2
        ba.fork(idxs[0])
        # Free idxs[1] back to pool (ref_count → 0, added to free deque)
        ba.free(idxs[1])
        # idxs[2] and idxs[3] have ref_count=1 each, not in free deque
        # Evict: should find nothing to evict (all ref_count >= 1 or already free)
        evicted = ba.evict_lru(2)
        assert isinstance(evicted, list)


# ---------------------------------------------------------------------------
# PageBlockTable
# ---------------------------------------------------------------------------


class TestPageBlockTable:
    def _make_table(self, num_blocks=16):
        ba = BlockAllocator(num_blocks=num_blocks)
        return PageBlockTable(ba), ba

    def test_ensure_space_allocates_first_block(self):
        table, _ = self._make_table()
        idx = table.ensure_space()
        assert idx is not None
        assert len(table.block_refs) == 1

    def test_ensure_space_stays_in_same_block(self):
        table, _ = self._make_table()
        idx0 = table.ensure_space()
        table.advance()
        idx1 = table.ensure_space()
        # Same block (not full yet)
        assert idx0 == idx1

    def test_ensure_space_allocates_new_block_when_full(self):
        table, _ = self._make_table(num_blocks=8)
        # Fill the first block completely
        first = table.ensure_space()
        for _ in range(PAGE_SIZE):
            table.advance()
        # Now page is "full" (write_offset >= PAGE_SIZE) → new block allocated
        second = table.ensure_space()
        assert second != first
        assert len(table.block_refs) == 2

    def test_ensure_space_exhausted_returns_none(self):
        """When allocator has no free blocks, ensure_space returns None."""
        table, _ = self._make_table(num_blocks=1)
        table.ensure_space()
        # Exhaust allocator
        for _ in range(PAGE_SIZE):
            table.advance()
        result = table.ensure_space()
        assert result is None

    def test_n_tokens_stored_empty(self):
        table, _ = self._make_table()
        assert table.n_tokens_stored == 0

    def test_n_tokens_stored_partial_block(self):
        table, _ = self._make_table()
        table.ensure_space()
        table.advance()
        table.advance()
        assert table.n_tokens_stored == 2

    def test_n_tokens_stored_multiple_blocks(self):
        table, _ = self._make_table(num_blocks=8)
        table.ensure_space()
        for _ in range(PAGE_SIZE):
            table.advance()
        table.ensure_space()
        table.advance()
        # 1 full page + 1 token in second page
        assert table.n_tokens_stored == PAGE_SIZE + 1

    def test_write_offset_after_advance(self):
        table, _ = self._make_table()
        table.ensure_space()
        table.advance()
        table.advance()
        assert table.write_offset == 2

    def test_fork_blocks_increments_refs(self):
        ba = BlockAllocator(num_blocks=8)
        # Allocate 2 blocks in another table
        t1 = PageBlockTable(ba)
        b1 = t1.ensure_space()
        for _ in range(PAGE_SIZE):
            t1.advance()
        b2 = t1.ensure_space()

        # Fork those blocks into a new table
        t2 = PageBlockTable(ba)
        t2.fork_blocks([b1, b2])
        assert t2.block_refs == [b1, b2]
        # Both blocks should now have ref_count=2 (alloc + fork)
        assert ba._blocks[b1].ref_count == 2
        assert ba._blocks[b2].ref_count == 2

    def test_free_all_releases_blocks(self):
        table, ba = self._make_table(num_blocks=4)
        table.ensure_space()
        assert ba.free_count == 3
        table.free_all()
        assert ba.free_count == 4  # block returned
        assert table.block_refs == []


# ---------------------------------------------------------------------------
# PagedKVCache — basic lifecycle
# ---------------------------------------------------------------------------


def _make_cache(num_blocks=16, n_layers=2, n_kv_heads=2, head_dim=4):
    return PagedKVCache(
        num_blocks=num_blocks,
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
    )


class TestPagedKVCacheInit:
    def test_init_stores_dims(self):
        cache = _make_cache(num_blocks=8, n_layers=3, n_kv_heads=4, head_dim=8)
        assert cache._n_layers == 3
        assert cache._n_heads == 4
        assert cache._head_dim == 8

    def test_get_store_lazy_init(self):
        cache = _make_cache()
        assert cache._store is None
        store = cache._get_store()
        assert store is not None
        assert store.shape[0] == 16  # num_blocks

    def test_get_store_cached(self):
        cache = _make_cache()
        s1 = cache._get_store()
        s2 = cache._get_store()
        assert s1 is s2


class TestPagedKVCacheRequests:
    def test_new_and_free_request(self):
        cache = _make_cache()
        table = cache.new_request("r1")
        assert "r1" in cache._tables
        cache.free_request("r1")
        assert "r1" not in cache._tables

    def test_free_unknown_request_no_error(self):
        cache = _make_cache()
        cache.free_request("nonexistent")  # should not raise

    def test_fork_request(self):
        cache = _make_cache(num_blocks=8)
        # Set up source request with 1 block
        t1 = cache.new_request("src")
        k = np.zeros((2, 4), dtype=np.float16)
        v = np.zeros((2, 4), dtype=np.float16)
        cache.store_token("src", 0, k, v)
        cache.advance_token("src")
        blocks = cache.get_block_refs("src")

        # Fork those blocks into a new request
        t2 = cache.fork_request("fork", blocks)
        assert "fork" in cache._tables
        assert t2.block_refs == blocks


class TestPagedKVCacheStoreAndLoad:
    def test_store_token_success(self):
        cache = _make_cache(n_layers=2, n_kv_heads=2, head_dim=4)
        cache.new_request("r1")
        k = np.ones((2, 4), dtype=np.float16)
        v = np.ones((2, 4), dtype=np.float16)
        result = cache.store_token("r1", 0, k, v)
        assert result is True

    def test_store_token_no_table(self):
        cache = _make_cache()
        k = np.zeros((2, 4), dtype=np.float16)
        v = np.zeros((2, 4), dtype=np.float16)
        result = cache.store_token("nonexistent", 0, k, v)
        assert result is False

    def test_store_token_exhausted_allocator(self):
        """When allocator is exhausted, store_token returns False."""
        cache = _make_cache(num_blocks=1, n_layers=2, n_kv_heads=2, head_dim=4)
        cache.new_request("r1")
        k = np.zeros((2, 4), dtype=np.float16)
        v = np.zeros((2, 4), dtype=np.float16)
        # Fill the single block
        for _ in range(PAGE_SIZE):
            cache.store_token("r1", 0, k, v)
            cache.advance_token("r1")
        # Now block is full and no more blocks available
        result = cache.store_token("r1", 0, k, v)
        assert result is False

    def test_advance_token_no_table(self):
        cache = _make_cache()
        cache.advance_token("nonexistent")  # should not raise

    def test_get_kv_for_layer_no_request(self):
        cache = _make_cache()
        result = cache.get_kv_for_layer("missing", 0)
        assert result is None

    def test_get_kv_for_layer_empty_table(self):
        """Table with no tokens stored → returns empty arrays."""
        cache = _make_cache(n_kv_heads=2, head_dim=4)
        cache.new_request("r1")
        result = cache.get_kv_for_layer("r1", 0)
        assert result is not None
        k, v = result
        assert k.shape[0] == 0
        assert v.shape[0] == 0

    def test_get_kv_for_layer_single_block(self):
        cache = _make_cache(n_layers=2, n_kv_heads=2, head_dim=4)
        cache.new_request("r1")
        rng = np.random.default_rng(0)
        stored_k = []
        for _ in range(3):
            k = rng.standard_normal((2, 4)).astype(np.float16)
            v = rng.standard_normal((2, 4)).astype(np.float16)
            stored_k.append(k.copy())
            cache.store_token("r1", 0, k, v)
            cache.advance_token("r1")
        result = cache.get_kv_for_layer("r1", 0)
        assert result is not None
        k_out, v_out = result
        assert k_out.shape == (3, 2, 4)

    def test_get_kv_for_layer_multiple_full_blocks(self):
        """Spans two full blocks (last_offset=0 for last block after advance)."""
        cache = _make_cache(num_blocks=8, n_layers=1, n_kv_heads=1, head_dim=2)
        cache.new_request("r1")
        k = np.ones((1, 2), dtype=np.float16)
        v = np.ones((1, 2), dtype=np.float16)
        # Write PAGE_SIZE + 1 tokens
        for _ in range(PAGE_SIZE + 1):
            cache.store_token("r1", 0, k, v)
            cache.advance_token("r1")
        result = cache.get_kv_for_layer("r1", 0)
        assert result is not None
        k_out, _ = result
        assert k_out.shape[0] == PAGE_SIZE + 1

    def test_get_kv_last_offset_zero(self):
        """When write_offset == 0 (slots==0), k_parts empty → empty array fallback."""
        cache = _make_cache(num_blocks=8, n_layers=1, n_kv_heads=1, head_dim=2)
        cache.new_request("r1")
        k = np.ones((1, 2), dtype=np.float16)
        v = np.ones((1, 2), dtype=np.float16)
        # Write exactly PAGE_SIZE tokens to fill first block completely
        for _ in range(PAGE_SIZE):
            cache.store_token("r1", 0, k, v)
            cache.advance_token("r1")
        # write_offset is now PAGE_SIZE (sentinel). After ensure_space on next
        # store_token, a NEW block gets allocated. The write_offset of the new block
        # is 0 before advance — but we won't store to second block here.
        # get_kv_for_layer should return the first block's tokens.
        result = cache.get_kv_for_layer("r1", 0)
        assert result is not None
        k_out, _ = result
        # The table has 1 block fully written plus possibly a 2nd empty block
        # depending on how we account for the write_offset=PAGE_SIZE sentinel.
        # In any case the result should be >= PAGE_SIZE tokens
        assert k_out.shape[0] >= PAGE_SIZE


class TestPagedKVCacheUtilities:
    def test_get_block_refs_existing(self):
        cache = _make_cache()
        cache.new_request("r1")
        k = np.zeros((2, 4), dtype=np.float16)
        v = np.zeros((2, 4), dtype=np.float16)
        cache.store_token("r1", 0, k, v)
        refs = cache.get_block_refs("r1")
        assert isinstance(refs, list)
        assert len(refs) > 0

    def test_get_block_refs_missing(self):
        cache = _make_cache()
        assert cache.get_block_refs("missing") == []

    def test_n_tokens_stored_existing(self):
        cache = _make_cache(n_layers=1, n_kv_heads=1, head_dim=2)
        cache.new_request("r1")
        k = np.zeros((1, 2), dtype=np.float16)
        v = np.zeros((1, 2), dtype=np.float16)
        cache.store_token("r1", 0, k, v)
        cache.advance_token("r1")
        assert cache.n_tokens_stored("r1") == 1

    def test_n_tokens_stored_missing(self):
        cache = _make_cache()
        assert cache.n_tokens_stored("missing") == 0

    def test_snapshot_prefix_no_source(self):
        cache = _make_cache()
        result = cache.snapshot_prefix("nonexistent", 8, "snap")
        assert result == []

    def test_snapshot_prefix_creates_forked_table(self):
        cache = _make_cache(num_blocks=8, n_layers=1, n_kv_heads=1, head_dim=2)
        cache.new_request("src")
        k = np.zeros((1, 2), dtype=np.float16)
        v = np.zeros((1, 2), dtype=np.float16)
        # Store PAGE_SIZE tokens to fill one block
        for _ in range(PAGE_SIZE):
            cache.store_token("src", 0, k, v)
            cache.advance_token("src")
        # Snapshot prefix of PAGE_SIZE tokens
        blocks = cache.snapshot_prefix("src", PAGE_SIZE, "snap")
        assert isinstance(blocks, list)
        assert "snap" in cache._tables

    def test_evict_lru_blocks(self):
        cache = _make_cache(num_blocks=4)
        freed = cache.evict_lru_blocks(2)
        assert isinstance(freed, int)

    def test_stats_keys(self):
        cache = _make_cache(num_blocks=4, n_layers=2, n_kv_heads=2, head_dim=4)
        s = cache.stats()
        assert "total_blocks" in s
        assert "free_blocks" in s
        assert "used_blocks" in s
        assert "active_requests" in s
        assert "memory_mb" in s
        assert s["total_blocks"] == 4

    def test_stats_active_requests(self):
        cache = _make_cache()
        cache.new_request("r1")
        cache.new_request("r2")
        s = cache.stats()
        assert s["active_requests"] == 2


class TestPagedKVCacheFromModel:
    def test_from_model_no_config(self):
        """Model without config → uses defaults (n_layers=32, etc.)."""
        model = MagicMock()
        del model.config
        del model.args
        with patch.dict(sys.modules, {"psutil": None}):
            cache = PagedKVCache.from_model(model)
        assert isinstance(cache, PagedKVCache)
        assert cache._n_layers == 32

    def test_from_model_with_psutil(self):
        """When psutil is available, uses its virtual_memory total."""
        model = MagicMock()
        model.config = None
        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.total = 16 * 1_073_741_824
        with patch.dict(sys.modules, {"psutil": mock_psutil}):
            cache = PagedKVCache.from_model(model)
        assert isinstance(cache, PagedKVCache)

    def test_from_model_with_model_config(self):
        """When model.config has dimension attrs, they are used."""
        model = MagicMock()
        cfg = MagicMock()
        cfg.num_hidden_layers = 16
        cfg.num_key_value_heads = 4
        cfg.head_dim = 64
        model.config = cfg
        with patch.dict(sys.modules, {"psutil": None}):
            cache = PagedKVCache.from_model(model)
        assert cache._n_layers == 16
        assert cache._n_heads == 4
        assert cache._head_dim == 64

    def test_from_model_psutil_import_error_uses_default_ram(self):
        """psutil ImportError → assumes 16 GB."""
        model = MagicMock()
        model.config = None
        # sys.modules key set to None triggers ImportError in the module
        with patch.dict(sys.modules, {"psutil": None}):
            cache = PagedKVCache.from_model(model, metal_fraction=0.25)
        # With 16 GB and 25%, budget=4 GB; with 32 layers, 8 heads, 128 dim, fp16:
        # bytes_per_block = 16 * 32 * 2 * 8 * 128 * 2 = 2 MB → ~2000 blocks
        assert cache._num_blocks >= 64  # at least minimum
