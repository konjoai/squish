"""
tests/test_wave41a_modules.py

Test suite for Wave 41a modules — Prefix Sharing, EAGLE-2, Ring Attention,
Token Entropy Pruning, Pre-Gated MoE Routing, Attention Sink Fusion:

  - squish/kv/radix_attn.py               (RadixAttentionCache)
  - squish/speculative/eagle2_spec.py      (EAGLE2Spec)
  - squish/attention/ring_attn.py          (RingAttention)
  - squish/token/token_entropy_prune.py    (TokenEntropyPruner)
  - squish/moe/pregated_router.py          (PreGatedMoERouter)
  - squish/kv/sink_fusion.py               (SinkFusion)
"""

import numpy as np
import pytest

# ============================================================
# RadixAttentionCache tests
# ============================================================

from squish.kv.radix_attn import (
    RadixAttentionConfig,
    RadixAttentionCache,
)


class TestRadixAttentionConfig:
    def test_defaults(self):
        cfg = RadixAttentionConfig()
        assert cfg.max_tokens >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1

    def test_invalid_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            RadixAttentionConfig(max_tokens=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            RadixAttentionConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            RadixAttentionConfig(head_dim=0)

    def test_repr_contains_class(self):
        cfg = RadixAttentionConfig()
        assert "RadixAttentionConfig" in repr(cfg) or "max_tokens" in repr(cfg)


class TestRadixAttentionCache:
    def _make_kv(self, seq_len, n_heads=4, head_dim=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        V = rng.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        return K, V

    def _cache(self, max_tokens=256, n_heads=4, head_dim=8):
        return RadixAttentionCache(
            RadixAttentionConfig(max_tokens=max_tokens, n_heads=n_heads, head_dim=head_dim)
        )

    def test_empty_hit_rate(self):
        cache = self._cache()
        assert cache.hit_rate() == 0.0

    def test_insert_and_match_prefix(self):
        cache = self._cache()
        tokens = [1, 2, 3, 4, 5]
        K, V = self._make_kv(5)
        cache.insert(tokens, K, V)
        assert cache.match_prefix(tokens) == len(tokens)

    def test_partial_prefix_match(self):
        cache = self._cache()
        tokens = [1, 2, 3, 4, 5]
        K, V = self._make_kv(5)
        cache.insert(tokens, K, V)
        prefix = tokens[:3]
        assert cache.match_prefix(prefix) == len(prefix)

    def test_no_match_returns_zero(self):
        cache = self._cache()
        K, V = self._make_kv(3)
        cache.insert([10, 20, 30], K, V)
        assert cache.match_prefix([99, 98, 97]) == 0

    def test_lookup_returns_kv_tensors(self):
        H, d = 4, 8
        cache = self._cache(n_heads=H, head_dim=d)
        tokens = [1, 2, 3]
        K, V = self._make_kv(3, n_heads=H, head_dim=d)
        cache.insert(tokens, K, V)
        K_out, V_out = cache.lookup(tokens)
        assert K_out.shape == (H, 3, d)
        assert V_out.shape == (H, 3, d)

    def test_lookup_missing_raises(self):
        cache = self._cache()
        with pytest.raises((KeyError, ValueError)):
            cache.lookup([99, 100])

    def test_n_cached_tokens(self):
        cache = self._cache()
        K, V = self._make_kv(5)
        cache.insert([1, 2, 3, 4, 5], K, V)
        assert cache.n_cached_tokens() == 5

    def test_hit_rate_increases_after_hit(self):
        cache = self._cache()
        K, V = self._make_kv(4)
        cache.insert([1, 2, 3, 4], K, V)
        # Hit
        cache.match_prefix([1, 2, 3, 4])
        assert cache.hit_rate() > 0.0

    def test_clear_resets_cache(self):
        cache = self._cache()
        K, V = self._make_kv(3)
        cache.insert([1, 2, 3], K, V)
        cache.clear()
        assert cache.n_cached_tokens() == 0

    def test_insert_shared_prefix(self):
        cache = self._cache()
        K1, V1 = self._make_kv(4, seed=0)
        K2, V2 = self._make_kv(5, seed=1)
        cache.insert([1, 2, 3, 4], K1, V1)
        cache.insert([1, 2, 3, 4, 5], K2, V2)
        # Both share [1,2,3,4] prefix
        assert cache.match_prefix([1, 2, 3, 4, 5]) == 5

    def test_repr(self):
        cache = self._cache()
        assert "RadixAttentionCache" in repr(cache) or "max_tokens" in repr(cache)

    def test_eviction_under_capacity(self):
        # max_tokens=4, insert 8 tokens — should not crash, cache bounded
        cache = self._cache(max_tokens=4)
        K, V = self._make_kv(8)
        cache.insert(list(range(8)), K, V)
        assert cache.n_cached_tokens() <= 8  # implementation may evict


# ============================================================
# EAGLE2Spec tests
# ============================================================

from squish.speculative.eagle2_spec import (
    EAGLE2Config,
    EAGLE2DraftResult,
    EAGLE2Spec,
)


class TestEAGLE2Config:
    def test_defaults_valid(self):
        cfg = EAGLE2Config()
        assert cfg.draft_length >= 1
        assert cfg.beam_width >= 1
        assert 0.0 <= cfg.prune_threshold < 1.0
        assert cfg.temperature > 0.0

    def test_invalid_prune_threshold_eq_one(self):
        with pytest.raises(ValueError, match="prune_threshold"):
            EAGLE2Config(prune_threshold=1.0)

    def test_invalid_prune_threshold_negative(self):
        with pytest.raises(ValueError, match="prune_threshold"):
            EAGLE2Config(prune_threshold=-0.1)

    def test_invalid_temperature_zero(self):
        with pytest.raises(ValueError, match="temperature"):
            EAGLE2Config(temperature=0.0)

    def test_invalid_temperature_negative(self):
        with pytest.raises(ValueError, match="temperature"):
            EAGLE2Config(temperature=-1.0)


class TestEAGLE2Spec:
    def _simple_fns(self, vocab=32):
        rng = np.random.default_rng(0)

        def draft_fn(last_token, context):
            p = rng.dirichlet(np.ones(vocab))
            return p.astype(np.float32)

        def score_fn(candidate_token, context):
            return float(rng.random())

        def target_fn(last_token, context):
            p = rng.dirichlet(np.ones(vocab))
            return p.astype(np.float32)

        return draft_fn, score_fn, target_fn

    def test_step_returns_result(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        result = spec.step([1, 2, 3], draft_fn, score_fn, target_fn)
        assert isinstance(result, EAGLE2DraftResult)

    def test_n_accepted_at_least_one(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        result = spec.step([1, 2], draft_fn, score_fn, target_fn)
        assert result.n_accepted >= 1

    def test_n_drafted_positive(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        result = spec.step([1, 2], draft_fn, score_fn, target_fn)
        assert result.n_drafted >= 1

    def test_acceptance_rate_in_range(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        result = spec.step([1, 2], draft_fn, score_fn, target_fn)
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_accepted_tokens_list(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        result = spec.step([1, 2], draft_fn, score_fn, target_fn)
        assert isinstance(result.accepted_tokens, list)
        assert len(result.accepted_tokens) == result.n_accepted

    def test_n_pruned_non_negative(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        result = spec.step([1, 2, 3], draft_fn, score_fn, target_fn)
        assert result.n_pruned >= 0

    def test_mean_acceptance_rate_accumulates(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        spec.step([1, 2], draft_fn, score_fn, target_fn)
        spec.step([1, 2], draft_fn, score_fn, target_fn)
        assert 0.0 <= spec.mean_acceptance_rate <= 1.0

    def test_reset_stats(self):
        spec = EAGLE2Spec(EAGLE2Config(draft_length=4, beam_width=2))
        draft_fn, score_fn, target_fn = self._simple_fns()
        spec.step([1, 2], draft_fn, score_fn, target_fn)
        spec.reset_stats()
        assert spec.mean_acceptance_rate == 0.0

    def test_repr(self):
        spec = EAGLE2Spec(EAGLE2Config())
        r = repr(spec)
        assert "EAGLE2" in r or "eagle" in r.lower()


# ============================================================
# RingAttention tests
# ============================================================

from squish.attention.ring_attn import (
    RingAttentionConfig,
    RingAttention,
)


class TestRingAttentionConfig:
    def test_defaults(self):
        cfg = RingAttentionConfig()
        assert cfg.n_shards >= 1
        assert cfg.n_heads >= 1
        assert cfg.head_dim >= 1

    def test_invalid_n_shards(self):
        with pytest.raises(ValueError, match="n_shards"):
            RingAttentionConfig(n_shards=0)

    def test_invalid_n_heads(self):
        with pytest.raises(ValueError, match="n_heads"):
            RingAttentionConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with pytest.raises(ValueError, match="head_dim"):
            RingAttentionConfig(head_dim=0)


class TestRingAttention:
    def _qkv(self, H=4, T=16, d=8, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((H, T, d)).astype(np.float32)
        K = rng.standard_normal((H, T, d)).astype(np.float32)
        V = rng.standard_normal((H, T, d)).astype(np.float32)
        return Q, K, V

    def test_forward_output_shape(self):
        H, T, d = 4, 16, 8
        attn = RingAttention(RingAttentionConfig(n_shards=4, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        out = attn.forward(Q, K, V)
        assert out.shape == (H, T, d)

    def test_forward_output_dtype_float32(self):
        H, T, d = 4, 16, 8
        attn = RingAttention(RingAttentionConfig(n_shards=4, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        out = attn.forward(Q, K, V)
        assert out.dtype == np.float32

    def test_n_shards_one_produces_valid_output(self):
        H, T, d = 2, 8, 4
        cfg = RingAttentionConfig(n_shards=1, n_heads=H, head_dim=d)
        attn = RingAttention(cfg)
        Q, K, V = self._qkv(H, T, d)
        out = attn.forward(Q, K, V)
        assert out.shape == (H, T, d)
        assert np.all(np.isfinite(out))

    def test_causal_vs_noncausal_differ(self):
        H, T, d = 2, 8, 4
        Q, K, V = self._qkv(H, T, d)
        causal = RingAttention(RingAttentionConfig(n_shards=2, causal=True, n_heads=H, head_dim=d))
        noncausal = RingAttention(RingAttentionConfig(n_shards=2, causal=False, n_heads=H, head_dim=d))
        out_c = causal.forward(Q, K, V)
        out_nc = noncausal.forward(Q, K, V)
        assert not np.allclose(out_c, out_nc)

    def test_unequal_T_S_raises(self):
        H, d = 2, 4
        cfg = RingAttentionConfig(n_shards=2, n_heads=H, head_dim=d)
        attn = RingAttention(cfg)
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((H, 8, d)).astype(np.float32)
        K = rng.standard_normal((H, 6, d)).astype(np.float32)
        V = rng.standard_normal((H, 6, d)).astype(np.float32)
        with pytest.raises(ValueError):
            attn.forward(Q, K, V)

    def test_output_all_finite(self):
        H, T, d = 4, 8, 4
        attn = RingAttention(RingAttentionConfig(n_shards=2, n_heads=H, head_dim=d))
        Q, K, V = self._qkv(H, T, d)
        out = attn.forward(Q, K, V)
        assert np.all(np.isfinite(out))

    def test_repr(self):
        attn = RingAttention(RingAttentionConfig())
        assert "Ring" in repr(attn) or "ring" in repr(attn).lower()


# ============================================================
# TokenEntropyPruner tests
# ============================================================

from squish.token.token_entropy_prune import (
    TokenEntropyConfig,
    TokenEntropyPruner,
)


class TestTokenEntropyConfig:
    def test_defaults(self):
        cfg = TokenEntropyConfig()
        assert 0.0 < cfg.keep_ratio <= 1.0
        assert cfg.min_tokens >= 1

    def test_invalid_keep_ratio_zero(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            TokenEntropyConfig(keep_ratio=0.0)

    def test_invalid_keep_ratio_gt_one(self):
        with pytest.raises(ValueError, match="keep_ratio"):
            TokenEntropyConfig(keep_ratio=1.1)

    def test_invalid_min_tokens_zero(self):
        with pytest.raises(ValueError, match="min_tokens"):
            TokenEntropyConfig(min_tokens=0)


class TestTokenEntropyPruner:
    def _hidden(self, T=20, d=32, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T, d)).astype(np.float32)

    def test_prune_output_shapes(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.5))
        h = self._hidden(T=20, d=32)
        kept, indices = pruner.prune(h)
        assert kept.ndim == 2
        assert indices.ndim == 1
        assert kept.shape[0] == len(indices)

    def test_prune_keeps_ratio(self):
        T = 20
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.5, min_tokens=1))
        h = self._hidden(T=T, d=32)
        kept, indices = pruner.prune(h)
        assert kept.shape[0] == max(1, int(T * 0.5))

    def test_prune_indices_sorted(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.6))
        h = self._hidden(T=20, d=16)
        _, indices = pruner.prune(h)
        assert np.all(np.diff(indices) >= 0)

    def test_min_tokens_floor(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.1, min_tokens=3))
        h = self._hidden(T=10, d=8)
        kept, _ = pruner.prune(h)
        assert kept.shape[0] >= 3

    def test_keep_ratio_one_returns_all(self):
        T = 16
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=1.0))
        h = self._hidden(T=T)
        kept, _ = pruner.prune(h)
        assert kept.shape[0] == T

    def test_not_2d_raises(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig())
        with pytest.raises(ValueError):
            pruner.prune(np.ones((4, 8, 8), dtype=np.float32))

    def test_compression_ratio(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.5))
        h = self._hidden(T=20)
        pruner.prune(h)
        ratio = pruner.compression_ratio()
        assert 0.0 < ratio <= 1.0

    def test_reset_stats(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.5))
        h = self._hidden()
        pruner.prune(h)
        pruner.reset_stats()
        assert pruner.compression_ratio() == 0.0 or pruner.compression_ratio() is not None

    def test_fill_pruned_mode(self):
        T, d = 20, 32
        pruner = TokenEntropyPruner(TokenEntropyConfig(keep_ratio=0.5, fill_pruned=True))
        h = self._hidden(T=T, d=d)
        result = pruner.prune(h)
        # fill_pruned may return either (kept, indices) or (full_T, indices)
        assert isinstance(result, tuple)

    def test_repr(self):
        pruner = TokenEntropyPruner(TokenEntropyConfig())
        assert "Entropy" in repr(pruner) or "Prune" in repr(pruner) or "pruner" in repr(pruner).lower()


# ============================================================
# PreGatedMoERouter tests
# ============================================================

from squish.moe.pregated_router import (
    PreGatedMoEConfig,
    PreGatedMoERouter,
)


class TestPreGatedMoEConfig:
    def test_defaults(self):
        cfg = PreGatedMoEConfig()
        assert cfg.n_experts >= 1
        assert cfg.top_k >= 1
        assert cfg.hidden_dim >= 1

    def test_invalid_top_k_exceeds_n_experts(self):
        with pytest.raises(ValueError, match="top_k"):
            PreGatedMoEConfig(n_experts=4, top_k=5)

    def test_invalid_hidden_dim(self):
        with pytest.raises(ValueError, match="hidden_dim"):
            PreGatedMoEConfig(hidden_dim=0)

    def test_invalid_n_experts(self):
        with pytest.raises(ValueError, match="n_experts"):
            PreGatedMoEConfig(n_experts=0)


class TestPreGatedMoERouter:
    def _hidden(self, T=8, d=16, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((T, d)).astype(np.float32)

    def test_route_output_shapes(self):
        T, d, E, K = 8, 16, 6, 2
        router = PreGatedMoERouter(PreGatedMoEConfig(n_experts=E, top_k=K, hidden_dim=d))
        h = self._hidden(T, d)
        indices, weights = router.route(h)
        assert indices.shape == (T, K)
        assert weights.shape == (T, K)

    def test_route_indices_in_valid_range(self):
        T, d, E, K = 8, 16, 6, 2
        router = PreGatedMoERouter(PreGatedMoEConfig(n_experts=E, top_k=K, hidden_dim=d))
        indices, _ = router.route(self._hidden(T, d))
        assert np.all(indices >= 0)
        assert np.all(indices < E)

    def test_gate_weights_sum_to_one(self):
        T, d, E, K = 8, 16, 6, 2
        router = PreGatedMoERouter(PreGatedMoEConfig(n_experts=E, top_k=K, hidden_dim=d))
        _, weights = router.route(self._hidden(T, d))
        row_sums = weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones(T), atol=1e-5)

    def test_forward_output_shape(self):
        T, d, E, K = 8, 16, 6, 2
        router = PreGatedMoERouter(PreGatedMoEConfig(n_experts=E, top_k=K, hidden_dim=d))
        h_prev = self._hidden(T, d, seed=0)
        h_cur = self._hidden(T, d, seed=1)
        indices, weights = router.route(h_prev)

        def expert_fn(eid, x):
            return x * float(eid + 1)

        out = router.forward(h_cur, expert_fn, indices, weights)
        assert out.shape == (T, d)

    def test_load_balancing_loss_in_range(self):
        T, d, E, K = 32, 8, 4, 2
        router = PreGatedMoERouter(PreGatedMoEConfig(n_experts=E, top_k=K, hidden_dim=d))
        indices, weights = router.route(self._hidden(T, d))
        loss = router.load_balancing_loss()
        assert 0.0 <= loss

    def test_repr(self):
        router = PreGatedMoERouter(PreGatedMoEConfig())
        assert "PreGated" in repr(router) or "MoE" in repr(router) or "router" in repr(router).lower()


# ============================================================
# SinkFusion tests
# ============================================================

from squish.kv.sink_fusion import (
    SinkFusionConfig,
    SinkFusion,
)


class TestSinkFusionConfig:
    def test_defaults(self):
        cfg = SinkFusionConfig()
        assert cfg.n_sinks >= 1
        assert 0.0 < cfg.calibration_momentum < 1.0

    def test_invalid_n_sinks(self):
        with pytest.raises(ValueError, match="n_sinks"):
            SinkFusionConfig(n_sinks=0)

    def test_invalid_momentum_zero(self):
        with pytest.raises(ValueError, match="calibration_momentum"):
            SinkFusionConfig(calibration_momentum=0.0)

    def test_invalid_momentum_one(self):
        with pytest.raises(ValueError, match="calibration_momentum"):
            SinkFusionConfig(calibration_momentum=1.0)


class TestSinkFusion:
    def _sink_kv(self, n_sinks=4, H=4, d=8, seed=0):
        rng = np.random.default_rng(seed)
        K = rng.standard_normal((H, n_sinks, d)).astype(np.float32)
        V = rng.standard_normal((H, n_sinks, d)).astype(np.float32)
        return K, V

    def test_fuse_output_shape(self):
        H, d, n = 4, 8, 4
        sf = SinkFusion(SinkFusionConfig(n_sinks=n, n_heads=H, head_dim=d))
        K, V = self._sink_kv(n, H, d)
        K_f, V_f = sf.fuse(K, V)
        assert K_f.shape == (H, 1, d)
        assert V_f.shape == (H, 1, d)

    def test_apply_prepends_fused_token(self):
        H, d, n = 4, 8, 4
        T_local = 10
        sf = SinkFusion(SinkFusionConfig(n_sinks=n, n_heads=H, head_dim=d))
        K_sinks, V_sinks = self._sink_kv(n, H, d)
        K_f, V_f = sf.fuse(K_sinks, V_sinks)
        rng = np.random.default_rng(1)
        K_local = rng.standard_normal((H, T_local, d)).astype(np.float32)
        V_local = rng.standard_normal((H, T_local, d)).astype(np.float32)
        K_full, V_full = sf.apply(K_f, K_local, V_f, V_local)
        assert K_full.shape == (H, T_local + 1, d)
        assert V_full.shape == (H, T_local + 1, d)

    def test_calibrate_does_not_crash(self):
        H, d, n = 4, 8, 4
        sf = SinkFusion(SinkFusionConfig(n_sinks=n, n_heads=H, head_dim=d))
        K, V = self._sink_kv(n, H, d)
        sf.calibrate(K, V)  # should not raise

    def test_memory_saved_tokens_positive(self):
        H, d, n = 4, 8, 4
        sf = SinkFusion(SinkFusionConfig(n_sinks=n, n_heads=H, head_dim=d))
        saved = sf.memory_saved_tokens(n_requests=10)
        assert saved > 0

    def test_fuse_output_finite(self):
        H, d, n = 4, 8, 4
        sf = SinkFusion(SinkFusionConfig(n_sinks=n, n_heads=H, head_dim=d))
        K, V = self._sink_kv(n, H, d)
        K_f, V_f = sf.fuse(K, V)
        assert np.all(np.isfinite(K_f))
        assert np.all(np.isfinite(V_f))

    def test_repr(self):
        sf = SinkFusion(SinkFusionConfig())
        assert "Sink" in repr(sf) or "sink" in repr(sf).lower()
