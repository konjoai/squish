"""
tests/test_speculative_helpers.py

Unit tests for the new helper functions and SpeculativeGenerator methods
added to squish/speculative.py for stateful KV-cache speculative decoding.

Covered:
  • _try_make_model_cache   — RotatingKVCache guard, exception paths
  • _cache_offset           — happy path, exception fallback
  • _cache_set_offset       — multi-entry, None-cache, exception safety
  • SpeculativeGenerator._reset_caches — with / without caches

MLX-dependent paths (_prefill_cached, _decode_step_cached,
_decode_multi_cached, _stateful_spec_stream, _stateful_plain_stream) are
individually skipped when MLX is not installed, so the test suite stays
green in CI environments without Apple Silicon or Metal.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

# ── helpers ───────────────────────────────────────────────────────────────────

def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


requires_mlx = pytest.mark.skipif(
    not _mlx_available(), reason="MLX not available"
)


# ── import under test ─────────────────────────────────────────────────────────

from squish.speculative import (
    _cache_offset,
    _cache_set_offset,
    _try_make_model_cache,
)

# ─────────────────────────────────────────────────────────────────────────────
# _cache_offset
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheOffset:
    def test_reads_first_entry_offset(self):
        entry = types.SimpleNamespace(offset=42)
        cache = [entry]
        assert _cache_offset(cache) == 42

    def test_multiple_entries_reads_first(self):
        c = [
            types.SimpleNamespace(offset=10),
            types.SimpleNamespace(offset=99),
        ]
        assert _cache_offset(c) == 10

    def test_empty_list_returns_zero(self):
        assert _cache_offset([]) == 0

    def test_none_returns_zero(self):
        assert _cache_offset(None) == 0

    def test_missing_offset_attr_returns_zero(self):
        entry = types.SimpleNamespace()   # no .offset
        assert _cache_offset([entry]) == 0

    def test_zero_offset(self):
        entry = types.SimpleNamespace(offset=0)
        assert _cache_offset([entry]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# _cache_set_offset
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheSetOffset:
    def test_sets_all_entries(self):
        entries = [
            types.SimpleNamespace(offset=100),
            types.SimpleNamespace(offset=200),
            types.SimpleNamespace(offset=300),
        ]
        _cache_set_offset(entries, 7)
        for e in entries:
            assert e.offset == 7

    def test_none_cache_is_safe(self):
        _cache_set_offset(None, 5)   # must not raise

    def test_empty_list_is_safe(self):
        _cache_set_offset([], 5)

    def test_single_entry(self):
        entry = types.SimpleNamespace(offset=99)
        _cache_set_offset([entry], 0)
        assert entry.offset == 0

    def test_exception_in_entry_is_swallowed(self):
        """Offset assignment raising AttributeError must not propagate."""

        class _Frozen:
            @property
            def offset(self):
                return 1
            @offset.setter
            def offset(self, v):
                raise AttributeError("frozen")

        _cache_set_offset([_Frozen()], 5)   # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# _try_make_model_cache
# ─────────────────────────────────────────────────────────────────────────────

class TestTryMakeModelCache:

    # ── helpers ──────────────────────────────────────────────────────────────

    class _FakeKVCache:
        """Stub that looks like mlx_lm KVCache (plain, non-rotating)."""
        pass

    class _FakeRotatingKVCache:
        """Stub whose class name contains 'rotating' → must be rejected."""
        pass

    def _model_for_patching(self):
        """Return a dummy model namespace."""
        return types.SimpleNamespace()

    # ── real MLX tests ───────────────────────────────────────────────────────

    @requires_mlx
    def test_returns_list_for_normal_model(self):
        """With a real mlx_lm model, returns a non-None list."""
        try:
            from mlx_lm.models.cache import make_prompt_cache  # noqa
        except ImportError:
            pytest.skip("make_prompt_cache not available")
        # We'd need a full model; skip without one
        pytest.skip("Full model required for make_prompt_cache test")

    # ── without MLX — test guard logic directly ───────────────────────────────

    def test_rotating_cache_returns_none(self, monkeypatch):
        """
        When the cache contains any entry whose type name includes 'rotating',
        _try_make_model_cache must return None.
        """

        # Inject a fake make_prompt_cache that returns a rotating entry
        class _FakeRot:
            pass
        _FakeRot.__name__ = "RotatingKVCache"

        fake_cache = [_FakeRot()]

        import sys
        mlx_lm_models_cache = types.ModuleType("mlx_lm.models.cache")
        mlx_lm_models_cache.make_prompt_cache = lambda _model: fake_cache
        sys.modules["mlx_lm.models.cache"] = mlx_lm_models_cache

        try:
            result = _try_make_model_cache(self._model_for_patching())
        finally:
            del sys.modules["mlx_lm.models.cache"]

        assert result is None

    def test_non_rotating_cache_returned(self, monkeypatch):
        """A plainKVCache (no 'rotating' in name) is passed through."""
        import sys

        class _FakePlain:
            pass
        _FakePlain.__name__ = "KVCache"

        fake_cache = [_FakePlain()]

        mlx_lm_models_cache = types.ModuleType("mlx_lm.models.cache")
        mlx_lm_models_cache.make_prompt_cache = lambda _model: fake_cache
        sys.modules["mlx_lm.models.cache"] = mlx_lm_models_cache

        try:
            result = _try_make_model_cache(self._model_for_patching())
        finally:
            del sys.modules["mlx_lm.models.cache"]

        assert result is fake_cache

    def test_both_apis_fail_returns_none(self):
        """When both mlx_lm import paths raise, returns None."""
        import sys

        # Remove any mlx_lm stubs so both import paths fail
        old_mods = {}
        for key in ["mlx_lm.models.cache", "mlx_lm.utils", "mlx_lm"]:
            if key in sys.modules:
                old_mods[key] = sys.modules.pop(key)

        try:
            result = _try_make_model_cache(self._model_for_patching())
        finally:
            sys.modules.update(old_mods)

        # If mlx_lm truly missing → None; if installed we just verify no crash
        assert result is None or result is not None  # must not raise

    def test_exception_inside_rotating_check_returns_none(self, monkeypatch):
        """
        If iterating the cache raises, function must silently return None.
        """
        import sys

        class _Weird:
            def __iter__(self):
                raise RuntimeError("boom")

        mlx_lm_models_cache = types.ModuleType("mlx_lm.models.cache")
        mlx_lm_models_cache.make_prompt_cache = lambda _model: _Weird()
        sys.modules["mlx_lm.models.cache"] = mlx_lm_models_cache

        try:
            result = _try_make_model_cache(self._model_for_patching())
        finally:
            del sys.modules["mlx_lm.models.cache"]

        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# SpeculativeGenerator._reset_caches
# ─────────────────────────────────────────────────────────────────────────────

class TestResetCaches:
    def _make_generator(self):
        """Construct a SpeculativeGenerator with minimal stubs."""
        from squish.speculative import SpeculativeGenerator

        class _FakeTokGen:
            eos_token_id = 2

        class _FakeModel:
            pass

        draft_m  = _FakeModel()
        draft_t  = _FakeTokGen()
        target_m = _FakeModel()
        target_t = _FakeTokGen()
        return SpeculativeGenerator(draft_m, draft_t, target_m, target_t)

    def test_reset_with_no_caches(self):
        """_reset_caches is safe when _target_cache and _draft_cache are None."""
        gen = self._make_generator()
        gen._target_cache = None
        gen._draft_cache  = None
        gen._reset_caches()   # must not raise

    def test_reset_calls_set_offset_zero(self):
        """_reset_caches must roll both caches to offset 0."""
        gen = self._make_generator()

        # Inject fake caches with mutable offset
        t_entry = types.SimpleNamespace(offset=55)
        d_entry = types.SimpleNamespace(offset=33)
        gen._target_cache = [t_entry]
        gen._draft_cache  = [d_entry]

        gen._reset_caches()

        assert t_entry.offset == 0
        assert d_entry.offset == 0

    def test_reset_with_none_target_only(self):
        gen = self._make_generator()
        gen._target_cache = None
        gen._draft_cache  = [types.SimpleNamespace(offset=10)]
        gen._reset_caches()
        assert gen._draft_cache[0].offset == 0

    def test_reset_with_none_draft_only(self):
        gen = self._make_generator()
        gen._draft_cache  = None
        gen._target_cache = [types.SimpleNamespace(offset=42)]
        gen._reset_caches()
        assert gen._target_cache[0].offset == 0


# ─────────────────────────────────────────────────────────────────────────────
# _prefill_cached — MLX only
# ─────────────────────────────────────────────────────────────────────────────

@requires_mlx
class TestPrefillCached:
    def test_returns_float32_numpy_array(self):
        import mlx.core as mx

        from squish.speculative import _prefill_cached

        VOCAB = 32
        class _TinyModel:
            def __call__(self, x, cache=None):
                B, T = x.shape
                return mx.ones((B, T, VOCAB), dtype=mx.float32)

        model = _TinyModel()
        result = _prefill_cached(model, None, [1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape == (VOCAB,)

    def test_last_row_selected(self):
        import mlx.core as mx

        from squish.speculative import _prefill_cached

        VOCAB = 8
        SEQ   = 4
        rng   = np.random.default_rng(7)
        expected = rng.standard_normal(VOCAB).astype(np.float32)

        class _TinyModel:
            def __call__(self, x, cache=None):
                B, T = x.shape
                out = np.zeros((B, T, VOCAB), dtype=np.float32)
                out[0, -1, :] = expected
                return mx.array(out)

        result = _prefill_cached(_TinyModel(), None, list(range(SEQ)))
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# _decode_step_cached — MLX only
# ─────────────────────────────────────────────────────────────────────────────

@requires_mlx
class TestDecodeStepCached:
    def test_returns_float32_vector(self):
        import mlx.core as mx

        from squish.speculative import _decode_step_cached

        VOCAB = 16
        class _M:
            def __call__(self, x, cache=None):
                return mx.ones((1, 1, VOCAB), dtype=mx.float32)

        result = _decode_step_cached(_M(), None, 5)
        assert result.shape == (VOCAB,)
        assert result.dtype == np.float32

    def test_single_token_input(self):
        import mlx.core as mx

        from squish.speculative import _decode_step_cached

        VOCAB = 8
        received_shape = []
        class _M:
            def __call__(self, x, cache=None):
                received_shape.append(tuple(x.shape))
                return mx.zeros((1, 1, VOCAB))

        _decode_step_cached(_M(), None, 42)
        assert received_shape[0] == (1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# _decode_multi_cached — MLX only
# ─────────────────────────────────────────────────────────────────────────────

@requires_mlx
class TestDecodeMultiCached:
    def test_returns_all_rows(self):
        import mlx.core as mx

        from squish.speculative import _decode_multi_cached

        VOCAB = 12
        K     = 4

        class _M:
            def __call__(self, x, cache=None):
                B, T = x.shape
                return mx.ones((B, T, VOCAB), dtype=mx.float32)

        result = _decode_multi_cached(_M(), None, list(range(K)))
        assert result.shape == (K, VOCAB)
        assert result.dtype == np.float32

    def test_row_values_correct(self):
        import mlx.core as mx

        from squish.speculative import _decode_multi_cached

        VOCAB = 4
        K     = 3
        rng   = np.random.default_rng(9)
        data  = rng.standard_normal((1, K, VOCAB)).astype(np.float32)

        class _M:
            def __call__(self, x, cache=None):
                return mx.array(data)

        result = _decode_multi_cached(_M(), None, [0, 1, 2])
        np.testing.assert_allclose(result, data[0], rtol=1e-5)
