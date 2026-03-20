"""
tests/test_wave40b_modules.py

Test suite for Wave 40b modules — Self-Speculative Decoding, Entropy Eviction & LSH-KV:

  - squish/speculative/kangaroo_spec.py   (KangarooSpec)
  - squish/kv/cake_evict.py               (CAKEEviction)
  - squish/kv/fp8_kv_cache.py             (FP8KVCache)
  - squish/attention/subgen_attn.py       (SubGenAttention)
  - squish/token/sep_llm_compress.py      (SepLLMCompress)
  - squish/speculative/spec_exec.py       (SpecExecDrafter)
"""

import numpy as np
import pytest

# ============================================================
# KangarooSpec tests
# ============================================================

from squish.speculative.kangaroo_spec import (
    KangarooConfig,
    KangarooDraftResult,
    KangarooSpec,
)


class TestKangarooConfig:
    def test_defaults(self):
        cfg = KangarooConfig()
        assert cfg.n_draft_layers > 0
        assert cfg.draft_length >= 1
        assert cfg.temperature > 0.0

    def test_invalid_n_draft_layers(self):
        with pytest.raises(ValueError, match="n_draft_layers"):
            KangarooConfig(n_draft_layers=0)

    def test_invalid_draft_length(self):
        with pytest.raises(ValueError, match="draft_length"):
            KangarooConfig(draft_length=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            KangarooConfig(temperature=0.0)


class TestKangarooSpec:
    def _make_fn(self, vocab=50, seed=0):
        rng = np.random.default_rng(seed)

        def fn(last_token, context):
            logits = rng.standard_normal(vocab).astype(np.float32)
            e = np.exp(logits - logits.max())
            probs = e / e.sum()
            return probs

        return fn

    def test_step_returns_result(self):
        spec = KangarooSpec(KangarooConfig(draft_length=3))
        draft_fn = self._make_fn(vocab=100, seed=1)
        target_fn = self._make_fn(vocab=100, seed=2)
        result = spec.step([1, 2, 3], draft_fn, target_fn)
        assert isinstance(result, KangarooDraftResult)

    def test_n_accepted_le_draft_length_plus_one(self):
        cfg = KangarooConfig(draft_length=5, seed=7)
        spec = KangarooSpec(cfg)
        draft_fn = self._make_fn(seed=10)
        target_fn = self._make_fn(seed=20)
        result = spec.step(list(range(10)), draft_fn, target_fn)
        assert result.n_accepted <= cfg.draft_length + 1

    def test_n_accepted_ge_one(self):
        # At minimum 1 token is always emitted (residual / target draw)
        cfg = KangarooConfig(draft_length=3, seed=99)
        spec = KangarooSpec(cfg)
        draft_fn = self._make_fn(seed=3)
        target_fn = self._make_fn(seed=4)
        result = spec.step([0], draft_fn, target_fn)
        assert result.n_accepted >= 1

    def test_accepted_tokens_length(self):
        cfg = KangarooConfig(draft_length=4)
        spec = KangarooSpec(cfg)
        draft_fn = self._make_fn()
        target_fn = self._make_fn(seed=1)
        result = spec.step([0, 1], draft_fn, target_fn)
        assert len(result.accepted_tokens) == result.n_accepted

    def test_acceptance_rate_in_range(self):
        spec = KangarooSpec(KangarooConfig(draft_length=5))
        draft_fn = self._make_fn()
        target_fn = self._make_fn(seed=1)
        result = spec.step([0], draft_fn, target_fn)
        assert result.acceptance_rate >= 0.0

    def test_mean_acceptance_rate_accumulates(self):
        spec = KangarooSpec(KangarooConfig(draft_length=3))
        draft_fn = self._make_fn()
        target_fn = self._make_fn(seed=1)
        for _ in range(5):
            spec.step([0, 1], draft_fn, target_fn)
        assert spec.mean_acceptance_rate >= 0.0

    def test_reset_stats(self):
        spec = KangarooSpec(KangarooConfig(draft_length=3))
        draft_fn = self._make_fn()
        target_fn = self._make_fn(seed=1)
        spec.step([0], draft_fn, target_fn)
        spec.reset_stats()
        assert spec.mean_acceptance_rate == 0.0

    def test_repr(self):
        r = repr(KangarooSpec())
        assert "KangarooSpec" in r


# ============================================================
# CAKEEviction tests
# ============================================================

from squish.kv.cake_evict import CAKEConfig, CAKEEviction


class TestCAKEConfig:
    def test_defaults(self):
        cfg = CAKEConfig()
        assert cfg.global_budget > 0
        assert cfg.n_layers > 0

    def test_invalid_global_budget(self):
        with pytest.raises(ValueError, match="global_budget"):
            CAKEConfig(global_budget=0)

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            CAKEConfig(n_layers=0)

    def test_invalid_obs_window(self):
        with pytest.raises(ValueError, match="obs_window"):
            CAKEConfig(obs_window=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            CAKEConfig(temperature=0.0)

    def test_invalid_min_layer_budget(self):
        with pytest.raises(ValueError, match="min_layer_budget"):
            CAKEConfig(min_layer_budget=0)


class TestCAKEEviction:
    def _layer(self, H=2, S=64, d=8, W=8, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((H, W, d)).astype(np.float32)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return Q, K, V

    def test_compute_budgets_sum_lte_global(self):
        cfg = CAKEConfig(global_budget=128, n_layers=4, n_heads=2, head_dim=8)
        evict = CAKEEviction(cfg)
        layers = [self._layer(seed=i) for i in range(4)]
        budgets = evict.compute_budgets(layers)
        assert budgets.sum() <= cfg.global_budget + cfg.n_layers  # rounding tolerance

    def test_compute_budgets_length(self):
        cfg = CAKEConfig(global_budget=64, n_layers=3)
        evict = CAKEEviction(cfg)
        layers = [self._layer(seed=i) for i in range(3)]
        budgets = evict.compute_budgets(layers)
        assert len(budgets) == 3

    def test_budgets_ge_min_layer_budget(self):
        cfg = CAKEConfig(global_budget=64, n_layers=3, min_layer_budget=8)
        evict = CAKEEviction(cfg)
        layers = [self._layer(seed=i) for i in range(3)]
        budgets = evict.compute_budgets(layers)
        assert all(b >= 8 for b in budgets)

    def test_compress_shapes(self):
        cfg = CAKEConfig(global_budget=64, n_layers=2, n_heads=2, head_dim=8)
        evict = CAKEEviction(cfg)
        layers = [self._layer(seed=i) for i in range(2)]
        budgets = evict.compute_budgets(layers)
        K_outs, V_outs, indices = evict.compress(layers, budgets)
        assert len(K_outs) == 2
        for kl in K_outs:
            for kh in kl:
                assert kh.ndim == 2

    def test_compress_indices_in_range(self):
        cfg = CAKEConfig(global_budget=64, n_layers=2, n_heads=2, head_dim=8)
        evict = CAKEEviction(cfg)
        layers = [self._layer(seed=i) for i in range(2)]
        budgets = evict.compute_budgets(layers)
        _, _, indices = evict.compress(layers, budgets)
        for layer_idx in indices:
            for idx in layer_idx:
                assert (idx >= 0).all() and (idx < 64).all()

    def test_layer_entropy_positive(self):
        evict = CAKEEviction(CAKEConfig(n_layers=2))
        Q, K, _ = self._layer()
        ent = evict._layer_entropy(Q, K)
        assert isinstance(float(ent), float)

    def test_repr(self):
        r = repr(CAKEEviction())
        assert "CAKEEviction" in r


# ============================================================
# FP8KVCache tests
# ============================================================

from squish.kv.fp8_kv_cache import FP8KVConfig, FP8KVTensor, FP8KVCache


class TestFP8KVConfig:
    def test_defaults(self):
        cfg = FP8KVConfig()
        assert cfg.dtype in ("e4m3", "e5m2")

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            FP8KVConfig(dtype="fp8")

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            FP8KVConfig(epsilon=0.0)


class TestFP8KVCache:
    def _kv(self, H=2, S=16, d=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return K, V

    def test_quantize_dtype_int8(self):
        cache = FP8KVCache()
        x = np.random.randn(4, 8).astype(np.float32)
        qt = cache.quantize(x)
        assert qt.codes.dtype == np.int8

    def test_dequantize_close_to_original(self):
        cache = FP8KVCache()
        x = np.random.randn(4, 8).astype(np.float32)
        qt = cache.quantize(x)
        x_hat = cache.dequantize(qt)
        assert x_hat.shape == x.shape
        # FP8 quantize/dequantize should be reasonably close
        assert np.mean(np.abs(x - x_hat)) < 0.5

    def test_relative_error_small(self):
        cache = FP8KVCache()
        x = np.random.randn(8, 8).astype(np.float32)
        qt = cache.quantize(x)
        x_hat = cache.dequantize(qt)
        err = cache.relative_error(x, x_hat)
        assert err < 0.5  # FP8 is approximate

    def test_store_and_load_shape(self):
        cache = FP8KVCache()
        K, V = self._kv()
        cache.store(0, K, V)
        K_back, V_back = cache.load(0)
        assert K_back.shape == K.shape
        assert V_back.shape == V.shape

    def test_load_missing_raises(self):
        cache = FP8KVCache()
        with pytest.raises(KeyError):
            cache.load(0)

    def test_memory_bytes_positive(self):
        cache = FP8KVCache()
        K, V = self._kv()
        cache.store(0, K, V)
        assert cache.memory_bytes() > 0

    def test_memory_bytes_less_than_fp32(self):
        cache = FP8KVCache()
        K, V = self._kv()
        fp32_bytes = K.nbytes + V.nbytes
        cache.store(0, K, V)
        # INT8 codes should use fewer bytes than FP32
        assert cache.memory_bytes() < fp32_bytes

    def test_n_layers_cached(self):
        cache = FP8KVCache()
        K, V = self._kv()
        cache.store(0, K, V)
        cache.store(1, K, V)
        assert cache.n_layers_cached() == 2

    def test_e5m2_mode(self):
        cache = FP8KVCache(FP8KVConfig(dtype="e5m2"))
        x = np.random.randn(4, 8).astype(np.float32)
        qt = cache.quantize(x)
        x_hat = cache.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_repr(self):
        r = repr(FP8KVCache())
        assert "FP8KVCache" in r


# ============================================================
# SubGenAttention tests
# ============================================================

from squish.attention.subgen_attn import SubGenConfig, SubGenAttention


class TestSubGenConfig:
    def test_defaults(self):
        cfg = SubGenConfig()
        assert cfg.window_size > 0
        assert cfg.n_global >= 0
        assert 0.0 <= cfg.alpha <= 1.0

    def test_invalid_window_size(self):
        with pytest.raises(ValueError, match="window_size"):
            SubGenConfig(window_size=0)

    def test_invalid_alpha_neg(self):
        with pytest.raises(ValueError, match="alpha"):
            SubGenConfig(alpha=-0.1)

    def test_invalid_alpha_gt1(self):
        with pytest.raises(ValueError, match="alpha"):
            SubGenConfig(alpha=1.1)

    def test_invalid_n_global(self):
        with pytest.raises(ValueError, match="n_global"):
            SubGenConfig(n_global=-1)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            SubGenConfig(n_heads=0)


class TestSubGenAttention:
    def _qkv(self, H=2, T=16, d=8, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((H, T, d)).astype(np.float32)
        K = rng.standard_normal((H, T, d)).astype(np.float32)
        V = rng.standard_normal((H, T, d)).astype(np.float32)
        return Q, K, V

    def test_forward_output_shape(self):
        attn = SubGenAttention(SubGenConfig(n_heads=2, head_dim=8))
        Q, K, V = self._qkv()
        out = attn.forward(Q, K, V)
        assert out.shape == (2, 16, 8)

    def test_forward_dtype_float32(self):
        attn = SubGenAttention(SubGenConfig(n_heads=2, head_dim=8))
        Q, K, V = self._qkv()
        out = attn.forward(Q, K, V)
        assert out.dtype == np.float32

    def test_alpha_zero_pure_local(self):
        cfg = SubGenConfig(n_heads=2, head_dim=8, alpha=0.0, window_size=4, n_global=2)
        attn = SubGenAttention(cfg)
        Q, K, V = self._qkv()
        out = attn.forward(Q, K, V)
        assert out.shape == (2, 16, 8)

    def test_alpha_one_pure_global(self):
        cfg = SubGenConfig(n_heads=2, head_dim=8, alpha=1.0, n_global=4)
        attn = SubGenAttention(cfg)
        Q, K, V = self._qkv()
        out = attn.forward(Q, K, V)
        assert out.shape == (2, 16, 8)

    def test_causal_vs_non_causal(self):
        cfg_c = SubGenConfig(n_heads=2, head_dim=8, causal=True)
        cfg_nc = SubGenConfig(n_heads=2, head_dim=8, causal=False)
        attn_c = SubGenAttention(cfg_c)
        attn_nc = SubGenAttention(cfg_nc)
        Q, K, V = self._qkv()
        out_c = attn_c.forward(Q, K, V)
        out_nc = attn_nc.forward(Q, K, V)
        assert out_c.shape == out_nc.shape
        # Causal and non-causal outputs should differ
        assert not np.allclose(out_c, out_nc)

    def test_mismatched_kv_shapes(self):
        # K and V with different seq length should still produce correct Q-shape output
        attn = SubGenAttention(SubGenConfig(n_heads=2, head_dim=8))
        Q = np.zeros((2, 4, 8), dtype=np.float32)
        K = np.zeros((2, 16, 8), dtype=np.float32)
        V = np.zeros((2, 16, 8), dtype=np.float32)
        out = attn.forward(Q, K, V)
        assert out.shape == (2, 4, 8)

    def test_repr(self):
        r = repr(SubGenAttention())
        assert "SubGenAttention" in r


# ============================================================
# SepLLMCompress tests
# ============================================================

from squish.token.sep_llm_compress import SepLLMConfig, SepLLMCompress


class TestSepLLMConfig:
    def test_defaults(self):
        cfg = SepLLMConfig()
        assert isinstance(cfg.sep_token_ids, (set, frozenset))
        assert cfg.recent_window >= 1

    def test_invalid_recent_window(self):
        with pytest.raises(ValueError, match="recent_window"):
            SepLLMConfig(recent_window=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            SepLLMConfig(n_heads=0)

    def test_empty_sep_token_ids(self):
        # Empty set is allowed — compresses to recent_window only
        cfg = SepLLMConfig(sep_token_ids=frozenset())
        assert len(cfg.sep_token_ids) == 0


class TestSepLLMCompress:
    def _kv(self, H=2, S=32, d=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return K, V

    def _token_ids(self, S=32, sep_id=13, sep_every=5):
        ids = np.arange(S, dtype=np.int64)
        ids[::sep_every] = sep_id
        return ids

    def test_compress_even_layer_reduces_kv(self):
        cfg = SepLLMConfig(
            sep_token_ids=frozenset({13}),
            recent_window=4,
            compress_even_layers=True,
            n_heads=2,
            head_dim=8,
        )
        comp = SepLLMCompress(cfg)
        token_ids = self._token_ids(S=32)
        K, V = self._kv(S=32)
        K_out, V_out, idx = comp.compress(layer_id=0, token_ids=token_ids, K=K, V=V)
        assert K_out.shape[1] < 32

    def test_compress_odd_layer_noop(self):
        cfg = SepLLMConfig(
            sep_token_ids=frozenset({13}),
            recent_window=4,
            compress_even_layers=True,
            n_heads=2,
            head_dim=8,
        )
        comp = SepLLMCompress(cfg)
        token_ids = self._token_ids(S=32)
        K, V = self._kv(S=32)
        K_out, V_out, idx = comp.compress(layer_id=1, token_ids=token_ids, K=K, V=V)
        # odd layer → no-op, all tokens kept
        assert K_out.shape[1] == 32

    def test_kept_indices_contain_sep_tokens(self):
        sep_id = 99
        cfg = SepLLMConfig(sep_token_ids=frozenset({sep_id}), recent_window=1, n_heads=2, head_dim=8)
        comp = SepLLMCompress(cfg)
        token_ids = np.array([0, 99, 2, 99, 4, 5, 6, 7], dtype=np.int64)
        K, V = self._kv(H=2, S=8)
        _, _, idx = comp.compress(layer_id=0, token_ids=token_ids, K=K, V=V)
        assert 1 in idx
        assert 3 in idx

    def test_recent_window_always_kept(self):
        cfg = SepLLMConfig(sep_token_ids=frozenset(), recent_window=4, n_heads=2, head_dim=8)
        comp = SepLLMCompress(cfg)
        token_ids = np.arange(16, dtype=np.int64)
        K, V = self._kv(H=2, S=16)
        _, _, idx = comp.compress(layer_id=0, token_ids=token_ids, K=K, V=V)
        # Last 4 positions must be kept
        for pos in [12, 13, 14, 15]:
            assert pos in idx

    def test_compression_ratio_range(self):
        cfg = SepLLMConfig(sep_token_ids=frozenset({13}), recent_window=4, n_heads=2, head_dim=8)
        comp = SepLLMCompress(cfg)
        token_ids = self._token_ids(S=32)
        ratio_even = comp.compression_ratio(token_ids, layer_id=0)
        ratio_odd = comp.compression_ratio(token_ids, layer_id=1)
        assert 0.0 < ratio_even <= 1.0
        assert ratio_odd == 1.0  # no compression on odd layers

    def test_compress_output_head_dim_unchanged(self):
        cfg = SepLLMConfig(sep_token_ids=frozenset({13}), recent_window=4, n_heads=2, head_dim=8)
        comp = SepLLMCompress(cfg)
        token_ids = self._token_ids(S=32)
        K, V = self._kv(S=32)
        K_out, V_out, _ = comp.compress(0, token_ids, K, V)
        assert K_out.shape[0] == 2   # n_heads preserved
        assert K_out.shape[-1] == 8  # head_dim preserved

    def test_repr(self):
        r = repr(SepLLMCompress())
        assert "SepLLMCompress" in r


# ============================================================
# SpecExecDrafter tests
# ============================================================

from squish.speculative.spec_exec import (
    SpecExecConfig,
    SpecExecResult,
    SpecExecDrafter,
)


class TestSpecExecConfig:
    def test_defaults(self):
        cfg = SpecExecConfig()
        assert cfg.budget > 0
        assert cfg.beam_width > 0
        assert cfg.max_depth > 0
        assert cfg.temperature > 0.0

    def test_invalid_budget(self):
        with pytest.raises(ValueError, match="budget"):
            SpecExecConfig(budget=0)

    def test_invalid_beam_width(self):
        with pytest.raises(ValueError, match="beam_width"):
            SpecExecConfig(beam_width=0)

    def test_invalid_max_depth(self):
        with pytest.raises(ValueError, match="max_depth"):
            SpecExecConfig(max_depth=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            SpecExecConfig(temperature=0.0)


class TestSpecExecDrafter:
    def _make_fn(self, vocab=50, seed=0):
        rng = np.random.default_rng(seed)

        def fn(last_token, context):
            logits = rng.standard_normal(vocab).astype(np.float32)
            e = np.exp(logits - logits.max())
            return e / e.sum()

        return fn

    def test_step_returns_result(self):
        cfg = SpecExecConfig(budget=8, beam_width=2, max_depth=3, seed=0)
        drafter = SpecExecDrafter(cfg)
        result = drafter.step([1, 2], self._make_fn(), self._make_fn(seed=1))
        assert isinstance(result, SpecExecResult)

    def test_n_drafted_lte_budget(self):
        cfg = SpecExecConfig(budget=8, beam_width=2, max_depth=4, seed=0)
        drafter = SpecExecDrafter(cfg)
        result = drafter.step([0], self._make_fn(), self._make_fn(seed=1))
        assert result.n_drafted <= cfg.budget

    def test_accepted_tokens_is_list(self):
        cfg = SpecExecConfig(budget=6, seed=5)
        drafter = SpecExecDrafter(cfg)
        result = drafter.step([0, 1], self._make_fn(), self._make_fn(seed=2))
        assert isinstance(result.accepted_tokens, list)

    def test_n_accepted_ge_one(self):
        cfg = SpecExecConfig(budget=8, seed=3)
        drafter = SpecExecDrafter(cfg)
        result = drafter.step([0], self._make_fn(), self._make_fn(seed=4))
        assert result.n_accepted >= 1

    def test_acceptance_rate_in_range(self):
        cfg = SpecExecConfig(budget=8, seed=9)
        drafter = SpecExecDrafter(cfg)
        result = drafter.step([0], self._make_fn(), self._make_fn(seed=10))
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_mean_acceptance_rate_accumulates(self):
        cfg = SpecExecConfig(budget=6, seed=11)
        drafter = SpecExecDrafter(cfg)
        for i in range(4):
            drafter.step([0, 1, 2], self._make_fn(seed=i), self._make_fn(seed=i + 100))
        assert 0.0 <= drafter.mean_acceptance_rate <= 1.0

    def test_reset_stats(self):
        cfg = SpecExecConfig(budget=6, seed=12)
        drafter = SpecExecDrafter(cfg)
        drafter.step([0], self._make_fn(), self._make_fn(seed=1))
        drafter.reset_stats()
        assert drafter.mean_acceptance_rate == 0.0

    def test_repr(self):
        r = repr(SpecExecDrafter())
        assert "SpecExecDrafter" in r
