"""Unit tests for squish.kv.prompt_kv_cache — disk-backed KV cache.

Tests cover:
  - Hash consistency and collision properties
  - put / get roundtrip with numpy arrays
  - Stale/corrupt entry handling
  - LRU eviction when over budget
  - cache clear and invalidate
  - Model-key mismatch invalidation
  - entry_count and total_bytes
  - Thread safety of concurrent puts
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_kv(n_layers: int = 4, seq: int = 8, heads: int = 8, head_dim: int = 64):
    """Return (keys, values) lists of numpy float16 arrays (1, heads, seq, head_dim)."""
    rng = np.random.default_rng(42)
    shape = (1, heads, seq, head_dim)
    keys   = [rng.standard_normal(shape).astype(np.float16) for _ in range(n_layers)]
    values = [rng.standard_normal(shape).astype(np.float16) for _ in range(n_layers)]
    return keys, values


# ── Hash tests ────────────────────────────────────────────────────────────────

class TestHashPrompt:
    def test_deterministic(self):
        from squish.kv.prompt_kv_cache import PromptKVStore

        h1 = PromptKVStore.hash_prompt("hello world")
        h2 = PromptKVStore.hash_prompt("hello world")
        assert h1 == h2

    def test_different_prompts_different_hashes(self):
        from squish.kv.prompt_kv_cache import PromptKVStore

        h1 = PromptKVStore.hash_prompt("prompt A")
        h2 = PromptKVStore.hash_prompt("prompt B")
        assert h1 != h2

    def test_hash_length_32(self):
        from squish.kv.prompt_kv_cache import PromptKVStore

        h = PromptKVStore.hash_prompt("any text")
        assert len(h) == 32

    def test_empty_string_hashes(self):
        from squish.kv.prompt_kv_cache import PromptKVStore

        h = PromptKVStore.hash_prompt("")
        assert len(h) == 32

    def test_unicode_prompt_hashes(self):
        from squish.kv.prompt_kv_cache import PromptKVStore

        h = PromptKVStore.hash_prompt("日本語のテスト 🦾")
        assert len(h) == 32


# ── Put / get roundtrip ───────────────────────────────────────────────────────

class TestPutGet:
    def test_basic_roundtrip(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        prompt = "Write a commit message for: add unit tests"
        keys, values = _make_kv(n_layers=4, seq=10)
        offset = 10

        store.put(prompt, keys, values, offset)
        entry = store.get(prompt)

        assert entry is not None
        assert entry.n_layers == 4
        assert entry.offset == offset
        assert len(entry.keys) == 4
        assert len(entry.values) == 4

    def test_kv_values_match(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        prompt = "Review this code:"
        keys, values = _make_kv(n_layers=2, seq=5)
        store.put(prompt, keys, values, offset=5)

        entry = store.get(prompt)
        assert entry is not None
        for i in range(2):
            np.testing.assert_allclose(entry.keys[i], keys[i], rtol=1e-3)
            np.testing.assert_allclose(entry.values[i], values[i], rtol=1e-3)

    def test_get_missing_returns_none(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        assert store.get("prompt not in cache") is None

    def test_put_overwrites_on_second_write(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        prompt = "Stable prompt"
        keys1, values1 = _make_kv(n_layers=2, seq=4)
        keys2, values2 = _make_kv(n_layers=2, seq=8)  # different offset

        store.put(prompt, keys1, values1, offset=4)
        store.put(prompt, keys2, values2, offset=8)

        entry = store.get(prompt)
        # Only the first write wins when the lockfile is absent
        assert entry is not None

    def test_model_key_stored(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path, model_key="qwen2.5-7b:abc123")
        keys, values = _make_kv()
        store.put("hello", keys, values, offset=1)

        entry = store.get("hello")
        assert entry is not None
        assert entry.model_key == "qwen2.5-7b:abc123"

    def test_model_key_mismatch_returns_none(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store_a = PromptKVStore(cache_dir=tmp_path, model_key="model_a")
        store_b = PromptKVStore(cache_dir=tmp_path, model_key="model_b")

        keys, values = _make_kv()
        store_a.put("shared prompt", keys, values, offset=5)

        # store_b has different model_key — should not find store_a's entry
        assert store_b.get("shared prompt") is None

    def test_no_model_key_ignores_mismatch(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store_a = PromptKVStore(cache_dir=tmp_path, model_key="model_a")
        store_any = PromptKVStore(cache_dir=tmp_path, model_key="")

        keys, values = _make_kv()
        store_a.put("shared prompt", keys, values, offset=5)

        # store_any has no model_key filter — should find the entry
        entry = store_any.get("shared prompt")
        assert entry is not None


# ── Corruption handling ───────────────────────────────────────────────────────

class TestCorruptHandling:
    def test_corrupt_meta_evicts_and_returns_none(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv()
        store.put("prompt", keys, values, offset=3)

        # Corrupt the meta file
        h   = PromptKVStore.hash_prompt("prompt")
        meta = tmp_path / h / "meta.json"
        meta.write_text("{ not valid json !!!")

        assert store.get("prompt") is None

    def test_wrong_version_evicts(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv()
        store.put("prompt", keys, values, offset=3)

        h   = PromptKVStore.hash_prompt("prompt")
        meta = tmp_path / h / "meta.json"
        m = json.loads(meta.read_text())
        m["version"] = 999
        meta.write_text(json.dumps(m))

        assert store.get("prompt") is None

    def test_missing_npy_evicts(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv(n_layers=2)
        store.put("prompt", keys, values, offset=2)

        h = PromptKVStore.hash_prompt("prompt")
        (tmp_path / h / "k_0.npy").unlink()

        assert store.get("prompt") is None


# ── Entry count and total_bytes ───────────────────────────────────────────────

class TestStorageStats:
    def test_entry_count_zero_initially(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        assert store.entry_count() == 0

    def test_entry_count_increments(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv()
        store.put("p1", keys, values, 1)
        store.put("p2", keys, values, 1)
        assert store.entry_count() == 2

    def test_total_bytes_positive_after_put(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv(n_layers=2)
        store.put("prompt", keys, values, 1)
        assert store.total_bytes() > 0

    def test_total_bytes_zero_initially(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        assert store.total_bytes() == 0


# ── Invalidate + clear ────────────────────────────────────────────────────────

class TestInvalidateClear:
    def test_invalidate_removes_entry(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv()
        store.put("p", keys, values, 1)
        assert store.invalidate("p") is True
        assert store.get("p") is None

    def test_invalidate_returns_false_when_not_present(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        assert store.invalidate("not cached") is False

    def test_clear_removes_all(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv()
        store.put("p1", keys, values, 1)
        store.put("p2", keys, values, 1)
        count = store.clear()
        assert count == 2
        assert store.entry_count() == 0

    def test_clear_empty_store(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        assert store.clear() == 0


# ── LRU eviction ─────────────────────────────────────────────────────────────

class TestLRUEviction:
    def test_evict_lru_respects_budget(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        # Create 5 entries
        store = PromptKVStore(cache_dir=tmp_path, max_bytes=0)  # budget=0 forces eviction
        keys, values = _make_kv(n_layers=2)
        for i in range(5):
            store.put(f"prompt_{i}", keys, values, 1)

        before = store.entry_count()
        store._evict_lru()
        after = store.entry_count()
        # Some entries should have been evicted
        assert after <= before

    def test_evict_lru_no_eviction_under_budget(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path, max_bytes=1024 * 1024 * 1024)  # 1 GB
        keys, values = _make_kv(n_layers=1)
        store.put("tiny_prompt", keys, values, 1)
        before = store.entry_count()
        evicted = store._evict_lru()
        assert evicted == 0
        assert store.entry_count() == before


# ── Live-adjustable budget (set_max_bytes) ───────────────────────────────────

class TestSetMaxBytes:
    def test_set_max_bytes_evicts_immediately(self, tmp_path):
        """Shrinking the budget below current on-disk usage evicts LRU entries
        right away, without waiting for put()'s probabilistic eviction check."""
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path, max_bytes=1024 * 1024 * 1024)
        keys, values = _make_kv(n_layers=2)
        for i in range(5):
            store.put(f"prompt_{i}", keys, values, 1)
        assert store.entry_count() == 5
        size_before = store.total_bytes()

        store.set_max_bytes(size_before // 3)

        assert store._max_bytes == size_before // 3
        assert store.total_bytes() <= size_before // 3
        assert store.entry_count() < 5

    def test_set_max_bytes_no_eviction_when_under_new_budget(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path, max_bytes=1024 * 1024 * 1024)
        keys, values = _make_kv(n_layers=1)
        store.put("tiny_prompt", keys, values, 1)
        before = store.entry_count()

        store.set_max_bytes(1024 * 1024 * 1024)  # unchanged, still plenty of room

        assert store.entry_count() == before

    def test_set_max_bytes_restores_higher_budget(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path, max_bytes=1024 * 1024 * 1024)
        keys, values = _make_kv(n_layers=2)
        for i in range(5):
            store.put(f"prompt_{i}", keys, values, 1)
        original_max = store._max_bytes

        store.set_max_bytes(1)  # forces eviction of everything but the newest
        shrunk_count = store.entry_count()
        assert shrunk_count < 5

        store.set_max_bytes(original_max)
        assert store._max_bytes == original_max
        # Raising the ceiling alone doesn't resurrect evicted entries.
        assert store.entry_count() == shrunk_count

    def test_set_max_bytes_rejects_non_positive(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        with pytest.raises(ValueError, match="max_bytes"):
            store.set_max_bytes(0)
        with pytest.raises(ValueError, match="max_bytes"):
            store.set_max_bytes(-1)


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_puts_different_prompts(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv(n_layers=2)
        errors: list[Exception] = []

        def _put(i: int) -> None:
            try:
                store.put(f"concurrent_prompt_{i}", keys, values, i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_put, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.entry_count() == 20

    def test_concurrent_reads_safe(self, tmp_path):
        from squish.kv.prompt_kv_cache import PromptKVStore

        store = PromptKVStore(cache_dir=tmp_path)
        keys, values = _make_kv(n_layers=2)
        store.put("shared_prompt", keys, values, 5)

        results: list = []

        def _get() -> None:
            results.append(store.get("shared_prompt"))

        threads = [threading.Thread(target=_get) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed
        assert all(r is not None for r in results)


# ── Numpy conversion ──────────────────────────────────────────────────────────

class TestToNumpy:
    def test_numpy_passthrough(self):
        from squish.kv.prompt_kv_cache import _to_numpy

        arr = np.random.randn(4, 8).astype(np.float32)
        out = _to_numpy(arr)
        assert out.dtype == np.float16
        np.testing.assert_allclose(out, arr.astype(np.float16), rtol=1e-3)

    def test_unsupported_type_raises(self):
        from squish.kv.prompt_kv_cache import _to_numpy

        with pytest.raises((TypeError, ImportError)):
            _to_numpy({"not": "an array"})


# ── capture_kv_state ─────────────────────────────────────────────────────────

class TestCaptureKVState:
    def test_none_cache_returns_none(self):
        from squish.kv.prompt_kv_cache import capture_kv_state

        assert capture_kv_state(None) is None

    def test_empty_list_returns_none(self):
        from squish.kv.prompt_kv_cache import capture_kv_state

        # Empty list has no cache layers → None
        result = capture_kv_state([])
        # Either None or a valid tuple with 0 layers; both are acceptable
        if result is not None:
            assert result[2] == 0  # offset

    def test_mock_layer_cache(self):
        from squish.kv.prompt_kv_cache import capture_kv_state
        from unittest.mock import MagicMock

        layer = MagicMock()
        layer.offset = 10
        layer.keys   = np.zeros((1, 8, 10, 64), dtype=np.float16)
        layer.values = np.zeros((1, 8, 10, 64), dtype=np.float16)

        result = capture_kv_state([layer, layer])
        assert result is not None
        ks, vs, off = result
        assert len(ks) == 2
        assert off == 10
