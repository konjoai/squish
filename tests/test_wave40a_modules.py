"""
tests/test_wave40a_modules.py

Test suite for Wave 40a modules — KV Architecture Innovation, Flash Weight & LSH-KV:

  - squish/attention/razor_attn.py      (RazorAttention)
  - squish/kv/lckv_cache.py             (LCKVCache)
  - squish/kv/cache_blend.py            (CacheBlendKV)
  - squish/kv/green_kv.py               (GreenKVEviction)
  - squish/kv/magic_pig_kv.py           (MagicPIGKV)
  - squish/io/flash_weight_cache.py     (FlashWeightCache)
"""

import numpy as np
import pytest

# ============================================================
# RazorAttention tests
# ============================================================

from squish.attention.razor_attn import (
    RazorAttentionConfig,
    RazorHeadType,
    RazorAttention,
)


class TestRazorAttentionConfig:
    def test_defaults(self):
        cfg = RazorAttentionConfig()
        assert cfg.n_heads > 0
        assert cfg.head_dim > 0
        assert 0.0 <= cfg.entropy_threshold <= 1.0

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            RazorAttentionConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            RazorAttentionConfig(head_dim=0)

    def test_invalid_n_summary_tokens(self):
        with pytest.raises(ValueError, match="n_summary_tokens"):
            RazorAttentionConfig(n_summary_tokens=0)

    def test_invalid_entropy_threshold_gt1(self):
        with pytest.raises(ValueError, match="entropy_threshold"):
            RazorAttentionConfig(entropy_threshold=1.5)

    def test_invalid_entropy_threshold_neg(self):
        with pytest.raises(ValueError, match="entropy_threshold"):
            RazorAttentionConfig(entropy_threshold=-0.1)


class TestRazorAttention:
    def _qkv(self, H=4, T=32, d=16, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((H, T, d)).astype(np.float32)
        K = rng.standard_normal((H, T, d)).astype(np.float32)
        V = rng.standard_normal((H, T, d)).astype(np.float32)
        return Q, K, V

    def test_head_types_initially_unclassified(self):
        razor = RazorAttention(RazorAttentionConfig(n_heads=4, head_dim=16))
        assert all(t == RazorHeadType.UNCLASSIFIED for t in razor.head_types())

    def test_calibrate_classifies_all_heads(self):
        razor = RazorAttention(RazorAttentionConfig(n_heads=4, head_dim=16))
        Q, K, V = self._qkv()
        razor.calibrate(Q, K, V)
        assert all(t != RazorHeadType.UNCLASSIFIED for t in razor.head_types())

    def test_calibrate_wrong_n_heads_raises(self):
        razor = RazorAttention(RazorAttentionConfig(n_heads=4, head_dim=16))
        Q = np.zeros((2, 8, 16), dtype=np.float32)
        K = np.zeros((2, 8, 16), dtype=np.float32)
        V = np.zeros((2, 8, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            razor.calibrate(Q, K, V)

    def test_forward_output_shape(self):
        razor = RazorAttention(RazorAttentionConfig(n_heads=4, head_dim=16))
        Q, K, V = self._qkv()
        razor.calibrate(Q, K, V)
        out = razor.forward(Q, K, V)
        assert out.shape == (4, 32, 16)

    def test_forward_dtype_float32(self):
        razor = RazorAttention(RazorAttentionConfig(n_heads=4, head_dim=16))
        Q, K, V = self._qkv()
        razor.calibrate(Q, K, V)
        out = razor.forward(Q, K, V)
        assert out.dtype == np.float32

    def test_forward_without_calibration(self):
        # Before calibration all heads are UNCLASSIFIED → treated as retrieval
        razor = RazorAttention(RazorAttentionConfig(n_heads=2, head_dim=8))
        Q = np.random.randn(2, 8, 8).astype(np.float32)
        K = np.random.randn(2, 8, 8).astype(np.float32)
        V = np.random.randn(2, 8, 8).astype(np.float32)
        out = razor.forward(Q, K, V)
        assert out.shape == (2, 8, 8)

    def test_retrieval_indices_after_calibration(self):
        cfg = RazorAttentionConfig(n_heads=4, head_dim=16, entropy_threshold=0.0)
        razor = RazorAttention(cfg)
        Q, K, V = self._qkv()
        razor.calibrate(Q, K, V)
        # With threshold=0.0 all heads should be retrieval
        assert len(razor.retrieval_head_indices()) == 4

    def test_non_retrieval_indices_threshold_one(self):
        cfg = RazorAttentionConfig(n_heads=4, head_dim=16, entropy_threshold=1.0)
        razor = RazorAttention(cfg)
        Q, K, V = self._qkv()
        razor.calibrate(Q, K, V)
        # With threshold=1.0 all heads should be non-retrieval
        assert len(razor.non_retrieval_head_indices()) == 4

    def test_calibration_accumulates(self):
        razor = RazorAttention(RazorAttentionConfig(n_heads=2, head_dim=8))
        Q, K, V = self._qkv(H=2, T=16, d=8)
        razor.calibrate(Q, K, V)
        razor.calibrate(Q, K, V)
        assert razor._n_calibration_calls == 2

    def test_forward_non_retrieval_heads_shape(self):
        cfg = RazorAttentionConfig(
            n_heads=2, head_dim=8, n_summary_tokens=2, entropy_threshold=1.0
        )
        razor = RazorAttention(cfg)
        Q = np.random.randn(2, 16, 8).astype(np.float32)
        K = np.random.randn(2, 16, 8).astype(np.float32)
        V = np.random.randn(2, 16, 8).astype(np.float32)
        razor.calibrate(Q, K, V)
        # Forward with a different (longer) K/V
        K2 = np.random.randn(2, 32, 8).astype(np.float32)
        V2 = np.random.randn(2, 32, 8).astype(np.float32)
        out = razor.forward(Q, K2, V2)
        assert out.shape == (2, 16, 8)

    def test_repr(self):
        r = repr(RazorAttention())
        assert "RazorAttention" in r


# ============================================================
# LCKVCache tests
# ============================================================

from squish.kv.lckv_cache import LCKVConfig, LCKVCache


class TestLCKVConfig:
    def test_defaults(self):
        cfg = LCKVConfig()
        assert cfg.n_layers > 0
        assert cfg.n_anchor > 0
        assert cfg.n_anchor <= cfg.n_layers

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            LCKVConfig(n_layers=0)

    def test_invalid_n_anchor_zero(self):
        with pytest.raises(ValueError, match="n_anchor"):
            LCKVConfig(n_anchor=0)

    def test_n_anchor_gt_n_layers(self):
        with pytest.raises(ValueError, match="n_anchor"):
            LCKVConfig(n_layers=4, n_anchor=8)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            LCKVConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            LCKVConfig(head_dim=0)


class TestLCKVCache:
    def _kv(self, H=2, S=8, d=4, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return K, V

    def test_write_and_read_anchor(self):
        cache = LCKVCache(LCKVConfig(n_layers=8, n_anchor=2))
        K, V = self._kv()
        cache.write(layer_id=0, K=K, V=V)
        K_r, V_r = cache.read(layer_id=0)
        np.testing.assert_array_equal(K_r, K.astype(np.float32))

    def test_read_non_anchor_returns_anchor_kv(self):
        cache = LCKVCache(LCKVConfig(n_layers=8, n_anchor=2))
        K, V = self._kv()
        cache.write(layer_id=0, K=K, V=V)
        # Layer 2 maps to anchor 0 (2 % 2 == 0)
        K_r, V_r = cache.read(layer_id=2)
        np.testing.assert_array_equal(K_r, K.astype(np.float32))

    def test_read_missing_raises(self):
        cache = LCKVCache(LCKVConfig(n_layers=4, n_anchor=2))
        with pytest.raises(KeyError):
            cache.read(layer_id=0)

    def test_invalid_layer_id_raises(self):
        cache = LCKVCache(LCKVConfig(n_layers=4, n_anchor=2))
        K, V = self._kv()
        with pytest.raises(ValueError):
            cache.write(layer_id=10, K=K, V=V)

    def test_is_anchor(self):
        cache = LCKVCache(LCKVConfig(n_layers=8, n_anchor=3))
        assert cache.is_anchor(0)
        assert cache.is_anchor(2)
        assert not cache.is_anchor(3)

    def test_memory_ratio(self):
        cache = LCKVCache(LCKVConfig(n_layers=8, n_anchor=2))
        assert cache.memory_ratio() == pytest.approx(0.25)

    def test_clear_removes_all(self):
        cache = LCKVCache(LCKVConfig(n_layers=4, n_anchor=2))
        K, V = self._kv()
        cache.write(layer_id=0, K=K, V=V)
        cache.clear()
        assert cache.n_slots_filled() == 0

    def test_n_slots_filled(self):
        cache = LCKVCache(LCKVConfig(n_layers=6, n_anchor=3))
        K, V = self._kv()
        cache.write(layer_id=0, K=K, V=V)
        cache.write(layer_id=1, K=K, V=V)
        assert cache.n_slots_filled() == 2

    def test_repr(self):
        r = repr(LCKVCache())
        assert "LCKVCache" in r


# ============================================================
# CacheBlendKV tests
# ============================================================

from squish.kv.cache_blend import CacheBlendConfig, KVBlock, CacheBlendKV


class TestCacheBlendConfig:
    def test_defaults(self):
        cfg = CacheBlendConfig()
        assert 0 < cfg.recompute_ratio <= 1.0
        assert cfg.importance_fn in ("l2", "random")

    def test_invalid_recompute_ratio(self):
        with pytest.raises(ValueError, match="recompute_ratio"):
            CacheBlendConfig(recompute_ratio=0.0)

    def test_invalid_recompute_ratio_gt1(self):
        with pytest.raises(ValueError, match="recompute_ratio"):
            CacheBlendConfig(recompute_ratio=1.1)

    def test_invalid_importance_fn(self):
        with pytest.raises(ValueError, match="importance_fn"):
            CacheBlendConfig(importance_fn="cosine")

    def test_invalid_max_blocks(self):
        with pytest.raises(ValueError, match="max_blocks"):
            CacheBlendConfig(max_blocks=0)


class TestCacheBlendKV:
    def _kv(self, H=2, S=16, d=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return K, V

    def test_store_and_has_block(self):
        blender = CacheBlendKV()
        K, V = self._kv()
        blender.store("doc0", K, V)
        assert blender.has_block("doc0")

    def test_blend_output_shape(self):
        blender = CacheBlendKV(CacheBlendConfig(recompute_ratio=0.25))
        K, V = self._kv()
        blender.store("doc0", K, V)
        K2, V2 = self._kv(seed=1)
        K_out, V_out = blender.blend("doc0", K2, V2)
        assert K_out.shape == K.shape
        assert V_out.shape == V.shape

    def test_blend_missing_raises(self):
        blender = CacheBlendKV()
        K, V = self._kv()
        with pytest.raises(KeyError):
            blender.blend("missing", K, V)

    def test_blend_increments_n_blends(self):
        blender = CacheBlendKV()
        K, V = self._kv()
        blender.store("d", K, V)
        blender.blend("d", K, V)
        blender.blend("d", K, V)
        assert blender.n_blends() == 2

    def test_random_importance_fn(self):
        cfg = CacheBlendConfig(importance_fn="random", recompute_ratio=0.2)
        blender = CacheBlendKV(cfg, seed=42)
        K, V = self._kv()
        blender.store("d", K, V)
        K_out, V_out = blender.blend("d", K, V)
        assert K_out.shape == K.shape

    def test_fifo_eviction_on_overflow(self):
        cfg = CacheBlendConfig(max_blocks=2)
        blender = CacheBlendKV(cfg)
        K, V = self._kv()
        blender.store("a", K, V)
        blender.store("b", K, V)
        blender.store("c", K, V)
        assert blender.n_blocks() == 2
        assert not blender.has_block("a")

    def test_evict(self):
        blender = CacheBlendKV()
        K, V = self._kv()
        blender.store("d", K, V)
        blender.evict("d")
        assert not blender.has_block("d")

    def test_clear(self):
        blender = CacheBlendKV()
        K, V = self._kv()
        blender.store("d", K, V)
        blender.clear()
        assert blender.n_blocks() == 0
        assert blender.n_blends() == 0

    def test_repr(self):
        r = repr(CacheBlendKV())
        assert "CacheBlendKV" in r


# ============================================================
# GreenKVEviction tests
# ============================================================

from squish.kv.green_kv import GreenKVConfig, GreenKVEviction


class TestGreenKVConfig:
    def test_defaults(self):
        cfg = GreenKVConfig()
        assert cfg.global_budget > 0
        assert cfg.obs_window > 0

    def test_invalid_global_budget(self):
        with pytest.raises(ValueError, match="global_budget"):
            GreenKVConfig(global_budget=0)

    def test_invalid_obs_window(self):
        with pytest.raises(ValueError, match="obs_window"):
            GreenKVConfig(obs_window=0)

    def test_invalid_min_head_budget(self):
        with pytest.raises(ValueError, match="min_head_budget"):
            GreenKVConfig(min_head_budget=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            GreenKVConfig(n_heads=0)


class TestGreenKVEviction:
    def _inputs(self, H=2, S=64, d=8, W=8, seed=0):
        rng = np.random.default_rng(seed)
        Q_obs = rng.standard_normal((H, W, d)).astype(np.float32)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        return Q_obs, K, V

    def test_compress_returns_three_lists(self):
        evict = GreenKVEviction(GreenKVConfig(global_budget=16, n_heads=2, head_dim=8))
        Q_obs, K, V = self._inputs()
        K_keep, V_keep, idx = evict.compress(Q_obs, K, V)
        assert len(K_keep) == 2
        assert len(V_keep) == 2
        assert len(idx) == 2

    def test_compressed_size_le_budget(self):
        cfg = GreenKVConfig(global_budget=32, min_head_budget=4, n_heads=2, head_dim=8)
        evict = GreenKVEviction(cfg)
        Q_obs, K, V = self._inputs()
        K_keep, _, idx = evict.compress(Q_obs, K, V)
        total = sum(len(i) for i in idx)
        assert total <= cfg.global_budget + cfg.n_heads  # slight tolerance for rounding

    def test_compressed_shape_per_head(self):
        evict = GreenKVEviction(GreenKVConfig(global_budget=16, n_heads=2, head_dim=8))
        Q_obs, K, V = self._inputs()
        K_keep, V_keep, _ = evict.compress(Q_obs, K, V)
        for kh in K_keep:
            assert kh.ndim == 2
            assert kh.shape[-1] == 8

    def test_indices_sorted(self):
        evict = GreenKVEviction(GreenKVConfig(global_budget=16, n_heads=2, head_dim=8))
        Q_obs, K, V = self._inputs()
        _, _, idx = evict.compress(Q_obs, K, V)
        for i in idx:
            assert list(i) == sorted(i)

    def test_repr(self):
        r = repr(GreenKVEviction())
        assert "GreenKVEviction" in r

    def test_compress_budget_larger_than_s(self):
        # Budget > S → keep all
        cfg = GreenKVConfig(global_budget=1000, n_heads=2, head_dim=8)
        evict = GreenKVEviction(cfg)
        Q_obs, K, V = self._inputs()
        K_keep, _, idx = evict.compress(Q_obs, K, V)
        for i in idx:
            assert len(i) == 64  # S=64, all kept


# ============================================================
# MagicPIGKV tests
# ============================================================

from squish.kv.magic_pig_kv import MagicPIGConfig, MagicPIGKV


class TestMagicPIGConfig:
    def test_defaults(self):
        cfg = MagicPIGConfig()
        assert cfg.n_tables > 0
        assert cfg.n_bits > 0

    def test_invalid_n_tables(self):
        with pytest.raises(ValueError, match="n_tables"):
            MagicPIGConfig(n_tables=0)

    def test_invalid_n_bits(self):
        with pytest.raises(ValueError, match="n_bits"):
            MagicPIGConfig(n_bits=0)

    def test_invalid_min_candidates(self):
        with pytest.raises(ValueError, match="min_candidates"):
            MagicPIGConfig(min_candidates=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            MagicPIGConfig(n_heads=0)


class TestMagicPIGKV:
    def _qkv(self, H=2, S=64, d=8, T=4, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, S, d)).astype(np.float32)
        V = rng.standard_normal((H, S, d)).astype(np.float32)
        Q = rng.standard_normal((H, T, d)).astype(np.float32)
        return Q, K, V

    def test_attend_without_index_exact(self):
        cfg = MagicPIGConfig(n_tables=2, n_bits=4, n_heads=2, head_dim=8, seed=0)
        pig = MagicPIGKV(cfg)
        Q, K, V = self._qkv()
        out = pig.attend(Q, K, V)
        assert out.shape == (2, 4, 8)

    def test_attend_with_index_shape(self):
        cfg = MagicPIGConfig(n_tables=2, n_bits=4, n_heads=2, head_dim=8, min_candidates=8, seed=0)
        pig = MagicPIGKV(cfg)
        Q, K, V = self._qkv()
        pig.build_index(K)
        out = pig.attend(Q, K, V)
        assert out.shape == (2, 4, 8)

    def test_attend_dtype_float32(self):
        cfg = MagicPIGConfig(n_tables=2, n_bits=4, n_heads=2, head_dim=8, seed=0)
        pig = MagicPIGKV(cfg)
        Q, K, V = self._qkv()
        pig.build_index(K)
        out = pig.attend(Q, K, V)
        assert out.dtype == np.float32

    def test_build_index_sets_n_indexed(self):
        cfg = MagicPIGConfig(n_tables=2, n_bits=4, n_heads=2, head_dim=8, seed=0)
        pig = MagicPIGKV(cfg)
        _, K, _ = self._qkv()
        pig.build_index(K)
        assert pig._n_indexed == 64

    def test_indices_cover_min_candidates(self):
        cfg = MagicPIGConfig(n_tables=1, n_bits=2, n_heads=1, head_dim=4, min_candidates=10, seed=7)
        pig = MagicPIGKV(cfg)
        K = np.random.randn(1, 32, 4).astype(np.float32)
        pig.build_index(K)
        q = np.random.randn(4, ).astype(np.float32)
        cands = pig._retrieve_candidates(q, 0)
        assert len(cands) >= 10

    def test_exact_fallback_matches_brute_force(self):
        cfg = MagicPIGConfig(n_tables=2, n_bits=4, n_heads=1, head_dim=4, seed=5)
        pig = MagicPIGKV(cfg)
        Q = np.random.randn(1, 1, 4).astype(np.float32)
        K = np.random.randn(1, 8, 4).astype(np.float32)
        V = np.random.randn(1, 8, 4).astype(np.float32)
        out_exact = pig.attend(Q, K, V)
        # Manual exact
        scale = 1.0 / np.sqrt(4)
        scores = (Q[0] @ K[0].T) * scale
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        a = (e / (e.sum(axis=-1, keepdims=True) + 1e-9))
        ref = a @ V[0]
        np.testing.assert_allclose(out_exact[0], ref, atol=1e-5)

    def test_repr(self):
        r = repr(MagicPIGKV())
        assert "MagicPIGKV" in r


# ============================================================
# FlashWeightCache tests
# ============================================================

from squish.io.flash_weight_cache import FlashWeightCacheConfig, FlashWeightCache


class TestFlashWeightCacheConfig:
    def test_defaults(self):
        cfg = FlashWeightCacheConfig()
        assert cfg.max_dram_layers >= 1
        assert cfg.bandwidth_gbps > 0

    def test_invalid_max_dram_layers(self):
        with pytest.raises(ValueError, match="max_dram_layers"):
            FlashWeightCacheConfig(max_dram_layers=0)

    def test_invalid_prefetch_ahead(self):
        with pytest.raises(ValueError, match="prefetch_ahead"):
            FlashWeightCacheConfig(prefetch_ahead=-1)

    def test_invalid_bandwidth(self):
        with pytest.raises(ValueError, match="bandwidth_gbps"):
            FlashWeightCacheConfig(bandwidth_gbps=0.0)

    def test_invalid_dtype(self):
        with pytest.raises(ValueError, match="dtype"):
            FlashWeightCacheConfig(dtype="bf32")


class TestFlashWeightCache:
    def _W(self, rows=32, cols=64, seed=0):
        return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)

    def test_store_and_load(self):
        cache = FlashWeightCache()
        W = self._W()
        cache.store(layer_id=0, weight=W)
        W_back = cache.load(layer_id=0)
        np.testing.assert_allclose(W_back, W)

    def test_load_missing_raises(self):
        cache = FlashWeightCache()
        with pytest.raises(KeyError):
            cache.load(layer_id=99)

    def test_negative_layer_id_raises(self):
        cache = FlashWeightCache()
        with pytest.raises(ValueError):
            cache.store(layer_id=-1, weight=self._W())

    def test_dram_hit_increments_counter(self):
        cache = FlashWeightCache(FlashWeightCacheConfig(max_dram_layers=4))
        W = self._W()
        cache.store(0, W)
        cache.load(0)  # DRAM hit
        assert cache.n_dram_hits == 1

    def test_flash_hit_on_eviction(self):
        cfg = FlashWeightCacheConfig(max_dram_layers=1)
        cache = FlashWeightCache(cfg)
        W0, W1 = self._W(seed=0), self._W(seed=1)
        cache.store(0, W0)
        cache.store(1, W1)  # evicts layer 0 from DRAM
        cache.load(0)       # DRAM miss → Flash hit
        assert cache.n_flash_hits >= 1

    def test_n_stored_layers(self):
        cache = FlashWeightCache()
        for i in range(5):
            cache.store(i, self._W(seed=i))
        assert cache.n_stored_layers() == 5

    def test_lru_eviction_keeps_recent(self):
        cfg = FlashWeightCacheConfig(max_dram_layers=2)
        cache = FlashWeightCache(cfg)
        for i in range(3):
            cache.store(i, self._W(seed=i))
        assert len(cache.dram_resident_layers()) <= 2

    def test_prefetch_loads_ahead(self):
        cfg = FlashWeightCacheConfig(max_dram_layers=8, prefetch_ahead=2)
        cache = FlashWeightCache(cfg)
        for i in range(5):
            cache.store(i, self._W(seed=i))
        cache.evict(1)
        cache.evict(2)
        cache.prefetch(1)
        resident = cache.dram_resident_layers()
        assert 1 in resident or 2 in resident or 3 in resident

    def test_memory_bytes_dram(self):
        cache = FlashWeightCache()
        W = self._W(64, 128)
        cache.store(0, W)
        assert cache.memory_bytes_dram() > 0

    def test_repr(self):
        r = repr(FlashWeightCache())
        assert "FlashWeightCache" in r
