"""tests/integration/test_radix_kv_reuse_integration.py

Phase 13 — RadixTree KV prefix reuse integration tests.

Verifies the full flow:
  1. First prompt is prefilled and the KV snapshot is stored.
  2. Second prompt (same prefix + delta tokens) finds the snapshot.
  3. Only delta tokens need prefilling on the second request.
  4. KV state is correctly restored from the snapshot.

Coverage targets
────────────────
PrefixKVStore.store()
  - Snapshot is stored for prompts above min_prefix_tokens
  - Short prompts are ignored
  - LRU eviction respects max_snapshots limit

PrefixKVStore.find()
  - Returns non-zero prefix_len for a stored prefix
  - Returns (0, None) for an unknown prefix
  - Returns (0, None) for inputs shorter than min_prefix_tokens

PrefixKVStore.restore()
  - Returns True and calls target.restore_from() on hit
  - Returns False for an evicted snapshot ID

Integration flow
  - store → find → restore → delta-only prefill
  - Shared prefix between two different prompts
  - Snapshot eviction and miss handling
"""
from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, call

import pytest

from squish.kv.prefix_kv_store import PrefixKVStore
from squish.kv.radix_cache import RadixTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_radix_tree(maxsize: int = 512) -> RadixTree:
    return RadixTree(maxsize=maxsize)


def _make_kv_mock() -> MagicMock:
    """Return a mock QuantizedKVCache with a clone_snapshot() and restore_from()."""
    mock = MagicMock()
    # clone_snapshot() returns a new distinct mock each time (independent snapshot)
    mock.clone_snapshot.side_effect = lambda: MagicMock()
    return mock


def _tokens(n: int, offset: int = 0) -> list[int]:
    """Return a list of n token ids starting at offset."""
    return list(range(offset, offset + n))


# ---------------------------------------------------------------------------
# PrefixKVStore.store — basic behaviour
# ---------------------------------------------------------------------------


class TestPrefixKVStoreStore:
    def test_store_returns_snapshot_id_for_long_prompt(self):
        pks = PrefixKVStore(_make_radix_tree(), max_snapshots=4, min_prefix_tokens=4)
        kv  = _make_kv_mock()
        ids = _tokens(20)
        snap_id = pks.store(ids, kv)
        assert snap_id is not None
        assert isinstance(snap_id, int)

    def test_store_ignores_short_prompt(self):
        pks = PrefixKVStore(_make_radix_tree(), max_snapshots=4, min_prefix_tokens=8)
        kv  = _make_kv_mock()
        ids = _tokens(4)  # below min_prefix_tokens=8
        snap_id = pks.store(ids, kv)
        assert snap_id is None

    def test_store_calls_clone_snapshot(self):
        pks = PrefixKVStore(_make_radix_tree(), max_snapshots=4)
        kv  = _make_kv_mock()
        pks.store(_tokens(20), kv)
        kv.clone_snapshot.assert_called_once()


# ---------------------------------------------------------------------------
# PrefixKVStore.find — unchanged prefix found
# ---------------------------------------------------------------------------


class TestPrefixKVStoreFind:
    def test_find_returns_positive_prefix_len_after_store(self):
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=4, min_prefix_tokens=4)
        kv  = _make_kv_mock()
        ids = _tokens(20)
        pks.store(ids, kv)

        # Same ids → should find full length
        prefix_len, snap_id = pks.find(ids)
        assert prefix_len > 0
        assert snap_id is not None

    def test_find_returns_zero_for_unknown_prefix(self):
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=4, min_prefix_tokens=4)
        kv  = _make_kv_mock()
        pks.store(_tokens(20), kv)

        # Completely different ids → miss
        prefix_len, snap_id = pks.find(_tokens(20, offset=1000))
        assert prefix_len == 0
        assert snap_id is None

    def test_find_returns_zero_for_short_input(self):
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=4, min_prefix_tokens=8)
        kv  = _make_kv_mock()
        pks.store(_tokens(20), kv)

        short = _tokens(4)  # too short to look up
        prefix_len, snap_id = pks.find(short)
        assert prefix_len == 0
        assert snap_id is None

    def test_find_prefix_and_delta_split(self):
        """store(A), then find(A + delta) → prefix_len == len(A)."""
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=4, min_prefix_tokens=4)
        kv  = _make_kv_mock()
        prefix_ids = _tokens(16)
        pks.store(prefix_ids, kv)

        delta = _tokens(8, offset=100)  # 8 new tokens after the prefix
        full_ids = prefix_ids + delta
        prefix_len, snap_id = pks.find(full_ids)
        assert prefix_len == len(prefix_ids)
        assert snap_id is not None


# ---------------------------------------------------------------------------
# PrefixKVStore.restore — KV state recovery
# ---------------------------------------------------------------------------


class TestPrefixKVStoreRestore:
    def test_restore_returns_true_on_hit(self):
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=4, min_prefix_tokens=4)
        kv  = _make_kv_mock()
        pks.store(_tokens(20), kv)
        prefix_len, snap_id = pks.find(_tokens(20))
        assert snap_id is not None

        target = _make_kv_mock()
        result = pks.restore(snap_id, target)
        assert result is True

    def test_restore_calls_restore_from_on_target(self):
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=4, min_prefix_tokens=4)
        kv  = _make_kv_mock()
        pks.store(_tokens(20), kv)
        _, snap_id = pks.find(_tokens(20))

        target = _make_kv_mock()
        pks.restore(snap_id, target)
        target.restore_from.assert_called_once()

    def test_restore_returns_false_for_invalid_snap_id(self):
        pks = PrefixKVStore(_make_radix_tree(), max_snapshots=4)
        target = _make_kv_mock()
        result = pks.restore(99999, target)
        assert result is False
        target.restore_from.assert_not_called()


# ---------------------------------------------------------------------------
# Integration: full store → find → restore → delta-only prefill simulation
# ---------------------------------------------------------------------------


class TestRadixKVReuseIntegration:
    def test_delta_only_prefill_on_second_request(self):
        """
        End-to-end radix KV reuse test.

        Turn 1: full prompt A (20 tokens) → prefill all 20 → store snapshot
        Turn 2: prompt A + delta (8 new tokens) → find prefix → only 8 tokens
                 need prefilling
        """
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=8, min_prefix_tokens=4)
        kv  = _make_kv_mock()  # simulated KV cache

        # ── Turn 1: full prompt ──
        prompt_A = _tokens(20)
        prefill_call_counts_turn1 = len(prompt_A)  # prefill all 20 tokens
        pks.store(prompt_A, kv)

        # ── Turn 2: same prefix + delta ──
        delta    = _tokens(8, offset=100)
        full_ids = prompt_A + delta

        prefix_len, snap_id = pks.find(full_ids)
        assert prefix_len == len(prompt_A), (
            f"Expected prefix_len={len(prompt_A)}, got {prefix_len}"
        )

        # Restore KV from snapshot (simulates setting up KV state from cache)
        target_kv = _make_kv_mock()
        ok = pks.restore(snap_id, target_kv)
        assert ok, "restore() should succeed for a valid snapshot"
        target_kv.restore_from.assert_called_once()

        # The caller would now prefill only `delta` tokens, not the full prompt.
        delta_tokens_to_prefill = full_ids[prefix_len:]
        assert len(delta_tokens_to_prefill) == len(delta)
        assert delta_tokens_to_prefill == delta

    def test_shared_prefix_across_two_sessions(self):
        """Two different request suffixes sharing a system-prompt prefix."""
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=8, min_prefix_tokens=4)
        kv  = _make_kv_mock()

        system_prompt = _tokens(32)
        pks.store(system_prompt, kv)

        # Session A: system + user message A
        session_a = system_prompt + _tokens(10, offset=200)
        prefix_len_a, snap_a = pks.find(session_a)
        assert prefix_len_a == len(system_prompt)
        assert snap_a is not None

        # Session B: system + user message B (different suffix)
        session_b = system_prompt + _tokens(12, offset=300)
        prefix_len_b, snap_b = pks.find(session_b)
        assert prefix_len_b == len(system_prompt)
        assert snap_b is not None
        assert snap_a == snap_b  # same snapshot covers both

    def test_lru_eviction_and_miss(self):
        """Evicted snapshots cause a cache miss on the next find()."""
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=2, min_prefix_tokens=4)  # only 2 slots

        kv = _make_kv_mock()
        ids_a = _tokens(16, offset=0)
        ids_b = _tokens(16, offset=100)
        ids_c = _tokens(16, offset=200)

        pks.store(ids_a, kv)  # snap_id 0 → slot 1/2
        pks.store(ids_b, kv)  # snap_id 1 → slot 2/2  (LRU: oldest=A)
        pks.store(ids_c, kv)  # snap_id 2 → evicts A  (LRU: oldest=B)

        # A was evicted — its trie entry points to a stale snap_id
        prefix_a, _ = pks.find(ids_a)
        # Either the trie returns 0 (stale entry purged) or find() detects eviction
        # and returns (0, None).  Both are valid.
        if prefix_a > 0:
            # Trie still has the entry but snap_id is stale → restore must fail
            _, snap_id_a = pks.find(ids_a)
            result = pks.restore(snap_id_a, _make_kv_mock())
            assert result is False

    def test_thread_safety_concurrent_stores(self):
        """Concurrent store() calls must not corrupt the LRU order."""
        rt  = _make_radix_tree()
        pks = PrefixKVStore(rt, max_snapshots=16, min_prefix_tokens=4)
        kv  = _make_kv_mock()

        errors: list[Exception] = []

        def _store_thread(offset: int) -> None:
            try:
                pks.store(_tokens(16, offset=offset * 100), kv)
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=_store_thread, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
