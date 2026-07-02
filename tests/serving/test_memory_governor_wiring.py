"""
tests/serving/test_memory_governor_wiring.py

Kill-test evidence for Phase 2 (WARNING half only) of the memory-governor
eviction sprint: proves the ``_on_memory_pressure_change`` callback registered
in squish/server.py Phase 13B actually shrinks live BlockKVCache/PromptKVStore
budgets on a WARNING transition, and restores them on NORMAL.

URGENT and CRITICAL are intentionally NOT covered here — those are later,
separately-approved phases of the same sprint (see squish/server.py's
``_on_memory_pressure_change`` docstring).
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import squish.server as _srv
from squish.kv.block_kv_cache import BlockKVCache
from squish.kv.prompt_kv_cache import PromptKVStore
from squish.serving.memory_governor import LEVEL_CRITICAL, LEVEL_NORMAL, LEVEL_URGENT, LEVEL_WARNING


@pytest.fixture(autouse=True)
def _reset_pressure_state():
    """Every test gets a clean slate: no caches wired, no captured baseline."""
    orig_bkv    = _srv._block_kv_cache
    orig_pkv    = _srv._prompt_kv_store
    orig_hot    = _srv._original_hot_max_bytes
    orig_prompt = _srv._original_prompt_max_bytes

    _srv._block_kv_cache          = None
    _srv._prompt_kv_store         = None
    _srv._original_hot_max_bytes    = None
    _srv._original_prompt_max_bytes = None

    yield

    _srv._block_kv_cache          = orig_bkv
    _srv._prompt_kv_store         = orig_pkv
    _srv._original_hot_max_bytes    = orig_hot
    _srv._original_prompt_max_bytes = orig_prompt


def _fake_block_arrays(block_size, n_layers=2, n_heads=4, head_dim=16):
    keys = [np.random.randn(1, n_heads, block_size, head_dim).astype(np.float16)
            for _ in range(n_layers)]
    vals = [np.random.randn(1, n_heads, block_size, head_dim).astype(np.float16)
            for _ in range(n_layers)]
    return keys, vals


@pytest.fixture
def real_block_cache(tmp_path):
    # hot_max_bytes sized to exactly fit all 8 blocks (2 layers x 4 heads x
    # 8 tokens x 16 head_dim x fp16 x (k+v) = 4096 bytes/block x 8 = 32768) so
    # a 50% WARNING shrink forces real eviction instead of staying far under
    # the default 2 GiB ceiling.
    cache = BlockKVCache(
        cache_dir=tmp_path / "block", block_size=8, model_key="mk",
        hot_max_bytes=32768,
    )
    ids = list(range(0, 64))  # 8 blocks
    per_b_k, per_b_v = zip(*[_fake_block_arrays(8) for _ in range(8)], strict=True)
    cache.store_blocks(ids, list(per_b_k), list(per_b_v))
    return cache


@pytest.fixture
def real_prompt_store(tmp_path):
    store = PromptKVStore(cache_dir=tmp_path / "prompt", max_bytes=1_000_000_000)
    return store


# ── End-to-end: real cache instances ─────────────────────────────────────────


class TestWarningShrinksRealCaches:
    def test_warning_shrinks_hot_tier_cache(self, real_block_cache):
        _srv._block_kv_cache = real_block_cache
        original_max = real_block_cache.stats()["hot_max_bytes"]

        _srv._on_memory_pressure_change(LEVEL_WARNING)

        new_max = real_block_cache.stats()["hot_max_bytes"]
        assert new_max == max(1, int(original_max * 0.5))
        assert new_max < original_max

    def test_warning_shrinks_prompt_cache(self, real_prompt_store):
        _srv._prompt_kv_store = real_prompt_store
        original_max = real_prompt_store.max_bytes

        _srv._on_memory_pressure_change(LEVEL_WARNING)

        assert real_prompt_store.max_bytes == max(1, int(original_max * 0.5))
        assert real_prompt_store.max_bytes < original_max

    def test_warning_then_normal_restores_hot_tier(self, real_block_cache):
        _srv._block_kv_cache = real_block_cache
        original_max = real_block_cache.stats()["hot_max_bytes"]

        _srv._on_memory_pressure_change(LEVEL_WARNING)
        assert real_block_cache.stats()["hot_max_bytes"] < original_max

        _srv._on_memory_pressure_change(LEVEL_NORMAL)
        assert real_block_cache.stats()["hot_max_bytes"] == original_max

    def test_warning_then_normal_restores_prompt_cache(self, real_prompt_store):
        _srv._prompt_kv_store = real_prompt_store
        original_max = real_prompt_store.max_bytes

        _srv._on_memory_pressure_change(LEVEL_WARNING)
        assert real_prompt_store.max_bytes < original_max

        _srv._on_memory_pressure_change(LEVEL_NORMAL)
        assert real_prompt_store.max_bytes == original_max

    def test_warning_actually_evicts_hot_entries(self, real_block_cache):
        """Not just a lowered ceiling — the hot tier must actually shed entries
        that no longer fit, immediately (no waiting for the next store)."""
        _srv._block_kv_cache = real_block_cache
        entries_before = real_block_cache.stats()["hot_entries"]
        assert entries_before == 8

        _srv._on_memory_pressure_change(LEVEL_WARNING)

        assert real_block_cache.stats()["hot_entries"] < entries_before


# ── Precise call assertions via mocks ────────────────────────────────────────


class TestPressureCallbackMocked:
    def test_warning_calls_setters_with_half_of_original(self):
        mock_bkv = MagicMock()
        mock_bkv.stats.return_value = {"hot_max_bytes": 2_000_000_000}
        mock_pkv = MagicMock()
        mock_pkv.max_bytes = 1_000_000_000
        _srv._block_kv_cache  = mock_bkv
        _srv._prompt_kv_store = mock_pkv

        _srv._on_memory_pressure_change(LEVEL_WARNING)

        mock_bkv.set_hot_max_bytes.assert_called_once_with(1_000_000_000)
        mock_pkv.set_max_bytes.assert_called_once_with(500_000_000)

    def test_normal_restores_captured_baseline_exactly(self):
        mock_bkv = MagicMock()
        mock_bkv.stats.return_value = {"hot_max_bytes": 2_000_000_000}
        mock_pkv = MagicMock()
        mock_pkv.max_bytes = 1_000_000_000
        _srv._block_kv_cache  = mock_bkv
        _srv._prompt_kv_store = mock_pkv

        _srv._on_memory_pressure_change(LEVEL_WARNING)
        _srv._on_memory_pressure_change(LEVEL_NORMAL)

        mock_bkv.set_hot_max_bytes.assert_called_with(2_000_000_000)
        mock_pkv.set_max_bytes.assert_called_with(1_000_000_000)

    def test_no_caches_configured_is_a_safe_noop(self):
        # _block_kv_cache / _prompt_kv_store are None (autouse fixture default)
        _srv._on_memory_pressure_change(LEVEL_WARNING)
        _srv._on_memory_pressure_change(LEVEL_NORMAL)
        # No exception == pass

    def test_urgent_and_critical_do_not_shrink_yet(self):
        """URGENT/CRITICAL are unbuilt, separately-gated phases of this sprint —
        the WARNING-only callback must not touch the budgets for them."""
        mock_bkv = MagicMock()
        mock_bkv.stats.return_value = {"hot_max_bytes": 2_000_000_000}
        mock_pkv = MagicMock()
        mock_pkv.max_bytes = 1_000_000_000
        _srv._block_kv_cache  = mock_bkv
        _srv._prompt_kv_store = mock_pkv

        _srv._on_memory_pressure_change(LEVEL_URGENT)
        _srv._on_memory_pressure_change(LEVEL_CRITICAL)

        mock_bkv.set_hot_max_bytes.assert_not_called()
        mock_pkv.set_max_bytes.assert_not_called()

    def test_baseline_captured_only_once(self):
        """A second WARNING event must shrink from the ORIGINAL baseline, not
        re-derive a smaller 'original' from the already-shrunk value."""
        mock_bkv = MagicMock()
        mock_bkv.stats.return_value = {"hot_max_bytes": 2_000_000_000}
        _srv._block_kv_cache = mock_bkv

        _srv._on_memory_pressure_change(LEVEL_WARNING)
        # Simulate the cache now reporting its shrunk size on a second poll.
        mock_bkv.stats.return_value = {"hot_max_bytes": 1_000_000_000}
        _srv._on_memory_pressure_change(LEVEL_WARNING)

        # Both calls should have shrunk from the same 2_000_000_000 baseline.
        assert all(
            call.args == (1_000_000_000,) for call in mock_bkv.set_hot_max_bytes.call_args_list
        )
