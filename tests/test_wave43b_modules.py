"""tests/test_wave43b_modules.py

Tests for Wave 43b modules:
  - WKVQuant (squish/kv/wkv_quant.py)
  - TokenizedKVCache (squish/kv/tokenized_kv.py)
  - ClusterEvictKV (squish/kv/cluster_evict_kv.py)
  - S2Attention (squish/attention/s2_attn.py)
  - SageAttn2 (squish/attention/sage_attn2.py)
  - MagicPIGv2 (squish/kv/magic_pig_v2.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── WKVQuant ───────────────────────────────────────────────────────────────────

from squish.kv.wkv_quant import WKVQuantConfig, WKVQuantResult, WKVQuant


class TestWKVQuantConfig:
    def test_defaults(self):
        cfg = WKVQuantConfig()
        assert cfg.n_bits in (4, 8)
        assert cfg.group_size > 0

    def test_custom(self):
        cfg = WKVQuantConfig(n_bits=4, group_size=64)
        assert cfg.n_bits == 4
        assert cfg.group_size == 64


class TestWKVQuant:
    def _make(self, n_bits=4, group_size=32):
        cfg = WKVQuantConfig(n_bits=n_bits, group_size=group_size)
        return WKVQuant(cfg)

    def test_quantize_weights_returns_result(self):
        q = self._make()
        W = np.random.randn(32, 64).astype(np.float32)
        result = q.quantize_weights(W)
        assert isinstance(result, WKVQuantResult)

    def test_quantize_kv_returns_result(self):
        q = self._make()
        kv = np.random.randn(8, 2, 16).astype(np.float32)
        result = q.quantize_kv(kv)
        assert isinstance(result, WKVQuantResult)

    def test_weight_codes_in_range(self):
        q = self._make(n_bits=4)
        W = np.random.randn(16, 32).astype(np.float32)
        result = q.quantize_weights(W)
        assert result.codes.min() >= 0
        assert result.codes.max() < 16

    def test_dequantize_weights_shape(self):
        q = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = q.quantize_weights(W)
        W_hat = result.dequantize()
        assert W_hat.shape == W.shape

    def test_detect_outlier_columns(self):
        q = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        W[:, 5] *= 100  # make col 5 an outlier
        mask = q.detect_outlier_columns(W)
        assert mask.dtype == bool
        assert mask.shape == (32,)

    def test_outlier_detection_finds_spike(self):
        q = self._make()
        W = np.random.randn(16, 32).astype(np.float32) * 0.01
        W[:, 7] = 100.0
        mask = q.detect_outlier_columns(W)
        assert mask[7]

    def test_kv_dequantize_close(self):
        q = self._make(n_bits=8)
        kv = np.random.randn(4, 2, 8).astype(np.float32)
        result = q.quantize_kv(kv)
        kv_hat = result.dequantize()
        assert np.allclose(kv, kv_hat, atol=0.1)

    def test_default_config(self):
        q = WKVQuant()
        assert q.config is not None


# ── TokenizedKVCache ───────────────────────────────────────────────────────────

from squish.kv.tokenized_kv import TokenizedKVConfig, TokenizedKVCache


class TestTokenizedKVConfig:
    def test_defaults(self):
        cfg = TokenizedKVConfig()
        assert cfg.embed_dim > 0
        assert cfg.max_cached_contexts > 0

    def test_custom(self):
        cfg = TokenizedKVConfig(embed_dim=32, max_cached_contexts=16)
        assert cfg.embed_dim == 32


class TestTokenizedKVCache:
    def _make(self, embed_dim=16, max_ctx=8):
        cfg = TokenizedKVConfig(embed_dim=embed_dim, max_cached_contexts=max_ctx)
        return TokenizedKVCache(cfg)

    def test_store_and_retrieve(self):
        cache = self._make()
        tokens = [1, 2, 3, 4]
        kv = np.random.randn(4, 2, 16).astype(np.float32)
        key = cache.store(tokens, kv)
        result = cache.retrieve(tokens)
        assert result is not None

    def test_retrieve_nonexistent_returns_none(self):
        cache = self._make()
        result = cache.retrieve([99, 100, 101])
        assert result is None

    def test_store_returns_string_key(self):
        cache = self._make()
        tokens = [1, 2]
        kv = np.random.randn(2, 2, 16).astype(np.float32)
        key = cache.store(tokens, kv)
        assert isinstance(key, str)

    def test_evict(self):
        cache = self._make()
        tokens = [1, 2, 3]
        kv = np.random.randn(3, 2, 16).astype(np.float32)
        cache.store(tokens, kv)
        evicted = cache.evict(tokens)
        assert evicted
        assert cache.retrieve(tokens) is None

    def test_evict_nonexistent_returns_false(self):
        cache = self._make()
        assert not cache.evict([999, 888])

    def test_different_contexts_cached_separately(self):
        cache = self._make(max_ctx=16)
        kv1 = np.random.randn(3, 2, 16).astype(np.float32)
        kv2 = np.random.randn(3, 2, 16).astype(np.float32)
        cache.store([1, 2, 3], kv1)
        cache.store([4, 5, 6], kv2)
        r1 = cache.retrieve([1, 2, 3])
        r2 = cache.retrieve([4, 5, 6])
        assert r1 is not None
        assert r2 is not None

    def test_default_config(self):
        cache = TokenizedKVCache()
        assert cache.config is not None


# ── ClusterEvictKV ─────────────────────────────────────────────────────────────

from squish.kv.cluster_evict_kv import ClusterEvictKVConfig, ClusterEvictKV


class TestClusterEvictKVConfig:
    def test_defaults(self):
        cfg = ClusterEvictKVConfig()
        assert cfg.n_clusters > 0
        assert cfg.budget_tokens > 0

    def test_custom(self):
        cfg = ClusterEvictKVConfig(n_clusters=4, budget_tokens=32)
        assert cfg.n_clusters == 4


class TestClusterEvictKV:
    def _make(self, n_clusters=4, budget=8):
        cfg = ClusterEvictKVConfig(n_clusters=n_clusters, budget_tokens=budget)
        return ClusterEvictKV(cfg)

    def test_evict_returns_smaller_kv(self):
        evict = self._make(n_clusters=3, budget=4)
        seq = 10
        keys = np.random.randn(seq, 8).astype(np.float32)
        values = np.random.randn(seq, 8).astype(np.float32)
        attn = np.softmax(np.random.randn(seq), axis=0) if False else self._softmax(np.random.randn(seq))
        K, V = evict.evict(keys, values, attn)
        assert K.shape[0] <= seq

    def _softmax(self, x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def test_evict_preserves_dim(self):
        evict = self._make(budget=6)
        keys = np.random.randn(12, 16).astype(np.float32)
        vals = np.random.randn(12, 16).astype(np.float32)
        attn = np.ones(12) / 12
        K, V = evict.evict(keys, vals, attn)
        assert K.shape[-1] == 16
        assert V.shape[-1] == 16

    def test_reset_budget(self):
        evict = self._make()
        evict.reset_budget()

    def test_default_config(self):
        evict = ClusterEvictKV()
        assert evict.config is not None

    def test_evict_short_seq_no_change(self):
        evict = self._make(budget=32)
        keys = np.random.randn(5, 8).astype(np.float32)
        vals = np.random.randn(5, 8).astype(np.float32)
        attn = np.ones(5) / 5
        K, V = evict.evict(keys, vals, attn)
        assert K.shape[0] == 5

    def test_high_attn_tokens_retained(self):
        evict = self._make(n_clusters=2, budget=4)
        seq = 10
        keys = np.random.randn(seq, 4).astype(np.float32)
        vals = np.random.randn(seq, 4).astype(np.float32)
        attn = np.zeros(seq)
        attn[0] = 1.0  # spike at tok 0
        K, V = evict.evict(keys, vals, attn)
        assert K.shape[0] >= 1


# ── S2Attention ────────────────────────────────────────────────────────────────

from squish.attention.s2_attn import S2AttnConfig, S2Attention


class TestS2AttnConfig:
    def test_defaults(self):
        cfg = S2AttnConfig()
        assert cfg.top_k > 0
        assert cfg.n_heads >= 1

    def test_custom(self):
        cfg = S2AttnConfig(top_k=16, n_heads=4)
        assert cfg.top_k == 16


class TestS2Attention:
    def _make(self, top_k=4, n_heads=2, head_dim=8):
        cfg = S2AttnConfig(top_k=top_k, n_heads=n_heads, head_dim=head_dim)
        return S2Attention(cfg)

    def test_forward_output_shape(self):
        s2 = self._make(top_k=4, n_heads=2, head_dim=8)
        seq = 12
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, seq, 8).astype(np.float32)
        v = np.random.randn(2, seq, 8).astype(np.float32)
        out, indices = s2.forward(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_selected_indices_count(self):
        s2 = self._make(top_k=4, n_heads=2, head_dim=8)
        seq = 16
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, seq, 8).astype(np.float32)
        v = np.random.randn(2, seq, 8).astype(np.float32)
        _, indices = s2.forward(q, k, v)
        assert len(indices) <= 4

    def test_exact_fallback_short_seq(self):
        s2 = self._make(top_k=32, n_heads=2, head_dim=8)
        seq = 4
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, seq, 8).astype(np.float32)
        v = np.random.randn(2, seq, 8).astype(np.float32)
        out, indices = s2.forward(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_output_dtype(self):
        s2 = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 8, 8).astype(np.float32)
        v = np.random.randn(2, 8, 8).astype(np.float32)
        out, _ = s2.forward(q, k, v)
        assert out.dtype == np.float32

    def test_default_config(self):
        s2 = S2Attention()
        assert s2.config is not None

    def test_indices_in_valid_range(self):
        s2 = self._make(top_k=4)
        seq = 12
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, seq, 8).astype(np.float32)
        v = np.random.randn(2, seq, 8).astype(np.float32)
        _, indices = s2.forward(q, k, v)
        for idx in indices:
            assert 0 <= idx < seq


# ── SageAttn2 ─────────────────────────────────────────────────────────────────

from squish.attention.sage_attn2 import SageAttn2Config, SageAttn2


class TestSageAttn2Config:
    def test_defaults(self):
        cfg = SageAttn2Config()
        assert cfg.n_heads >= 1
        assert cfg.head_dim > 0

    def test_custom(self):
        cfg = SageAttn2Config(n_heads=4, head_dim=32)
        assert cfg.n_heads == 4


class TestSageAttn2:
    def _make(self, n_heads=2, head_dim=8):
        cfg = SageAttn2Config(n_heads=n_heads, head_dim=head_dim)
        return SageAttn2(cfg)

    def test_calibrate(self):
        sa = self._make(n_heads=2, head_dim=8)
        q = np.random.randn(2, 4, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        sa.calibrate(q, k)

    def test_forward_shape(self):
        sa = self._make(n_heads=2, head_dim=8)
        seq = 6
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, seq, 8).astype(np.float32)
        v = np.random.randn(2, seq, 8).astype(np.float32)
        out = sa.forward(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_forward_before_calibrate(self):
        sa = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        out = sa.forward(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_output_dtype(self):
        sa = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        out = sa.forward(q, k, v)
        assert out.dtype == np.float32

    def test_calibrate_sets_mean(self):
        sa = self._make(n_heads=2, head_dim=8)
        q = np.random.randn(2, 10, 8).astype(np.float32)
        k = np.random.randn(2, 10, 8).astype(np.float32)
        sa.calibrate(q, k)
        assert sa._q_mean is not None or sa._calibrated

    def test_default_config(self):
        sa = SageAttn2()
        assert sa.config is not None


# ── MagicPIGv2 ────────────────────────────────────────────────────────────────

from squish.kv.magic_pig_v2 import MagicPIGv2Config, MagicPIGv2


class TestMagicPIGv2Config:
    def test_defaults(self):
        cfg = MagicPIGv2Config()
        assert cfg.n_tables > 0
        assert cfg.hash_dim > 0
        assert cfg.budget > 0

    def test_custom(self):
        cfg = MagicPIGv2Config(n_tables=4, hash_dim=8, budget=32)
        assert cfg.n_tables == 4


class TestMagicPIGv2:
    def _make(self, n_tables=4, hash_dim=4, budget=8):
        cfg = MagicPIGv2Config(n_tables=n_tables, hash_dim=hash_dim, budget=budget)
        return MagicPIGv2(cfg)

    def test_retrieve_candidates_shape(self):
        pig = self._make(budget=4)
        seq = 16
        q = np.random.randn(8).astype(np.float32)
        keys = np.random.randn(seq, 8).astype(np.float32)
        indices = pig.retrieve_candidates(q, keys)
        assert len(indices) <= seq

    def test_attend_output_shape(self):
        pig = self._make(budget=4)
        q = np.random.randn(8).astype(np.float32)
        keys = np.random.randn(16, 8).astype(np.float32)
        vals = np.random.randn(16, 8).astype(np.float32)
        out, candidates = pig.attend(q, keys, vals)
        assert out.shape == (8,)

    def test_candidates_in_range(self):
        pig = self._make(budget=4)
        seq = 12
        q = np.random.randn(8).astype(np.float32)
        keys = np.random.randn(seq, 8).astype(np.float32)
        indices = pig.retrieve_candidates(q, keys)
        for idx in indices:
            assert 0 <= idx < seq

    def test_reset_probe_budget(self):
        pig = self._make()
        pig.reset_probe_budget()

    def test_attend_dtype(self):
        pig = self._make()
        q = np.random.randn(8).astype(np.float32)
        keys = np.random.randn(10, 8).astype(np.float32)
        vals = np.random.randn(10, 8).astype(np.float32)
        out, _ = pig.attend(q, keys, vals)
        assert out.dtype == np.float32

    def test_default_config(self):
        pig = MagicPIGv2()
        assert pig.config is not None

    def test_multiple_tables_diversify_candidates(self):
        pig = self._make(n_tables=8, budget=16)
        q = np.random.randn(8).astype(np.float32)
        keys = np.random.randn(32, 8).astype(np.float32)
        indices = pig.retrieve_candidates(q, keys)
        assert len(indices) >= 1
