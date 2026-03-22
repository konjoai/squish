"""tests/test_wave46b_modules.py

Wave 46b test suite — MinP · ContrastiveSearch · RazorAttention · CacheBlend ·
GreenKV · PrebeleRouter

Tests cover:
- Config defaults and validation
- Core algorithm shapes / dtypes
- Probability / distribution properties
- Edge cases and integration sequences
"""

from __future__ import annotations

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# MinPSampler
# ---------------------------------------------------------------------------

class TestMinPSampler(unittest.TestCase):
    def _make(self, vocab=128, min_p_factor=0.1, temperature=1.0, top_k=0):
        from squish.sampling.minp_sampler import MinPSampler, MinPConfig
        return MinPSampler(MinPConfig(
            min_p_factor=min_p_factor,
            temperature=temperature,
            top_k=top_k,
        )), vocab

    def test_config_defaults(self):
        from squish.sampling.minp_sampler import MinPConfig
        cfg = MinPConfig()
        self.assertAlmostEqual(cfg.min_p_factor, 0.1)
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.top_k, 0)

    def test_config_invalid_min_p(self):
        from squish.sampling.minp_sampler import MinPConfig
        with self.assertRaises(ValueError):
            MinPConfig(min_p_factor=-0.1)
        with self.assertRaises(ValueError):
            MinPConfig(min_p_factor=1.0)  # must be < 1.0

    def test_config_invalid_temperature(self):
        from squish.sampling.minp_sampler import MinPConfig
        with self.assertRaises(ValueError):
            MinPConfig(temperature=0.0)

    def test_config_invalid_top_k(self):
        from squish.sampling.minp_sampler import MinPConfig
        with self.assertRaises(ValueError):
            MinPConfig(top_k=-1)

    def test_config_property(self):
        sampler, _ = self._make()
        from squish.sampling.minp_sampler import MinPConfig
        self.assertIsInstance(sampler.config, MinPConfig)

    def test_sample_returns_valid_token(self):
        sampler, vocab = self._make(vocab=256)
        logits = np.random.randn(vocab).astype(np.float32)
        result = sampler.sample(logits)
        self.assertGreaterEqual(result.token_id, 0)
        self.assertLess(result.token_id, vocab)

    def test_sample_probability_nonneg(self):
        sampler, vocab = self._make(vocab=64)
        logits = np.random.randn(vocab).astype(np.float32)
        result = sampler.sample(logits)
        self.assertGreater(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    def test_sample_at_least_one_candidate(self):
        sampler, vocab = self._make(vocab=64)
        logits = np.random.randn(vocab).astype(np.float32)
        result = sampler.sample(logits)
        self.assertGreaterEqual(result.n_candidates, 1)

    def test_threshold_matches_min_p_factor(self):
        min_p = 0.05
        sampler, vocab = self._make(vocab=128, min_p_factor=min_p)
        logits = np.random.randn(vocab).astype(np.float32)
        result = sampler.sample(logits)
        # Threshold should be min_p * max_probability
        p = np.exp(logits) / np.exp(logits).sum()
        expected_threshold = min_p * p.max()
        self.assertAlmostEqual(result.threshold, expected_threshold, places=5)

    def test_sample_batch_shape(self):
        sampler, vocab = self._make(vocab=64)
        logits = np.random.randn(8, vocab).astype(np.float32)
        ids = sampler.sample_batch(logits)
        self.assertEqual(ids.shape, (8,))
        self.assertEqual(ids.dtype, np.int32)

    def test_sample_batch_valid_range(self):
        sampler, vocab = self._make(vocab=64)
        logits = np.random.randn(16, vocab).astype(np.float32)
        ids = sampler.sample_batch(logits)
        self.assertTrue(np.all(ids >= 0))
        self.assertTrue(np.all(ids < vocab))

    def test_filter_logits_shape(self):
        sampler, vocab = self._make(vocab=64)
        logits = np.random.randn(vocab).astype(np.float32)
        filtered = sampler.filter_logits(logits)
        self.assertEqual(filtered.shape, (vocab,))

    def test_filter_logits_has_neg_inf(self):
        sampler, vocab = self._make(vocab=64, min_p_factor=0.5)
        logits = np.random.randn(vocab).astype(np.float32)
        filtered = sampler.filter_logits(logits)
        # With min_p=0.5, many tokens should be pruned
        self.assertTrue(np.any(np.isneginf(filtered)))

    def test_filter_logits_min_p_zero_no_pruning(self):
        sampler, vocab = self._make(vocab=32, min_p_factor=0.0)
        logits = np.random.randn(vocab).astype(np.float32)
        filtered = sampler.filter_logits(logits)
        # p=0 → threshold=0 → all tokens kept
        self.assertFalse(np.any(np.isneginf(filtered)))

    def test_top_k_restricts_candidates(self):
        from squish.sampling.minp_sampler import MinPSampler, MinPConfig
        sampler = MinPSampler(MinPConfig(top_k=5, min_p_factor=0.0))
        logits = np.random.randn(128).astype(np.float32)
        result = sampler.sample(logits)
        self.assertLessEqual(result.n_candidates, 5)

    def test_temperature_scaling(self):
        from squish.sampling.minp_sampler import MinPSampler, MinPConfig
        # Very high temperature → near-uniform; very low → peaked
        logits = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high_t = MinPSampler(MinPConfig(temperature=100.0, min_p_factor=0.0))
        low_t = MinPSampler(MinPConfig(temperature=0.01, min_p_factor=0.0))
        high_result = high_t.sample(logits)
        low_result = low_t.sample(logits)
        # Low temperature should almost always pick token 0
        self.assertEqual(low_result.token_id, 0)


# ---------------------------------------------------------------------------
# ContrastiveSearch
# ---------------------------------------------------------------------------

class TestContrastiveSearch(unittest.TestCase):
    def _make(self, vocab=64, top_k=4, embed_dim=16):
        from squish.sampling.contrastive_search import ContrastiveSearch, ContrastiveSearchConfig
        cfg = ContrastiveSearchConfig(
            top_k=top_k,
            vocab_size=vocab,
            embed_dim=embed_dim,
            context_window=8,
        )
        emb = np.random.randn(vocab, embed_dim).astype(np.float32)
        return ContrastiveSearch(cfg, embeddings=emb)

    def test_config_defaults(self):
        from squish.sampling.contrastive_search import ContrastiveSearchConfig
        cfg = ContrastiveSearchConfig()
        self.assertEqual(cfg.top_k, 5)
        self.assertAlmostEqual(cfg.alpha, 0.6)

    def test_config_invalid_alpha(self):
        from squish.sampling.contrastive_search import ContrastiveSearchConfig
        with self.assertRaises(ValueError):
            ContrastiveSearchConfig(alpha=-0.1)
        with self.assertRaises(ValueError):
            ContrastiveSearchConfig(alpha=1.1)

    def test_config_invalid_top_k(self):
        from squish.sampling.contrastive_search import ContrastiveSearchConfig
        with self.assertRaises(ValueError):
            ContrastiveSearchConfig(top_k=0)

    def test_initial_context_len(self):
        cs = self._make()
        self.assertEqual(cs.context_len, 0)

    def test_step_valid_token_id(self):
        vocab = 64
        cs = self._make(vocab=vocab)
        logits = np.random.randn(vocab).astype(np.float32)
        result = cs.step(logits)
        self.assertGreaterEqual(result.token_id, 0)
        self.assertLess(result.token_id, vocab)

    def test_step_increments_context_len(self):
        cs = self._make()
        logits = np.random.randn(64).astype(np.float32)
        cs.step(logits)
        self.assertEqual(cs.context_len, 1)

    def test_step_increments_multiple_times(self):
        cs = self._make()
        for _ in range(5):
            logits = np.random.randn(64).astype(np.float32)
            cs.step(logits)
        self.assertEqual(cs.context_len, 5)

    def test_step_candidates_shape(self):
        top_k = 5
        cs = self._make(top_k=top_k)
        logits = np.random.randn(64).astype(np.float32)
        result = cs.step(logits)
        self.assertEqual(len(result.candidates), top_k)

    def test_step_scores_finite(self):
        cs = self._make()
        logits = np.random.randn(64).astype(np.float32)
        result = cs.step(logits)
        self.assertTrue(np.isfinite(result.model_score))
        self.assertTrue(np.isfinite(result.contrastive_score))

    def test_step_degeneration_penalty_nonneg(self):
        cs = self._make()
        logits = np.random.randn(64).astype(np.float32)
        result = cs.step(logits)
        self.assertGreaterEqual(result.degeneration_penalty, 0.0)

    def test_reset_context_clears(self):
        cs = self._make()
        for _ in range(3):
            cs.step(np.random.randn(64).astype(np.float32))
        cs.reset_context()
        self.assertEqual(cs.context_len, 0)

    def test_generate_length(self):
        cs = self._make(vocab=64, top_k=3)
        n = 5
        logits_seq = [np.random.randn(64).astype(np.float32) for _ in range(n)]
        tokens = cs.generate(logits_seq)
        self.assertEqual(len(tokens), n)

    def test_generate_valid_range(self):
        vocab = 64
        cs = self._make(vocab=vocab, top_k=3)
        logits_seq = [np.random.randn(vocab).astype(np.float32) for _ in range(4)]
        tokens = cs.generate(logits_seq)
        for t in tokens:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, vocab)

    def test_no_embeddings_provided(self):
        from squish.sampling.contrastive_search import ContrastiveSearch, ContrastiveSearchConfig
        cfg = ContrastiveSearchConfig(top_k=3, vocab_size=32, embed_dim=8)
        cs = ContrastiveSearch(cfg)
        logits = np.random.randn(32).astype(np.float32)
        result = cs.step(logits)
        self.assertGreaterEqual(result.token_id, 0)

    def test_alpha_zero_uses_only_model_score(self):
        from squish.sampling.contrastive_search import ContrastiveSearch, ContrastiveSearchConfig
        cfg = ContrastiveSearchConfig(top_k=4, vocab_size=32, embed_dim=8, alpha=0.0)
        emb = np.random.randn(32, 8).astype(np.float32)
        cs = ContrastiveSearch(cfg, embeddings=emb)
        logits = np.random.randn(32).astype(np.float32)
        result = cs.step(logits)
        self.assertAlmostEqual(result.degeneration_penalty, 0.0, places=4)


# ---------------------------------------------------------------------------
# RazorAttention
# ---------------------------------------------------------------------------

class TestRazorAttention(unittest.TestCase):
    def _make(self, n_heads=4, head_dim=16, entropy_threshold=0.5):
        from squish.attention.razor_attn import RazorAttention, RazorAttentionConfig
        return RazorAttention(RazorAttentionConfig(
            n_heads=n_heads,
            head_dim=head_dim,
            entropy_threshold=entropy_threshold,
        ))

    def test_config_defaults(self):
        from squish.attention.razor_attn import RazorAttentionConfig
        cfg = RazorAttentionConfig()
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.head_dim, 64)
        self.assertAlmostEqual(cfg.entropy_threshold, 0.5)

    def test_config_invalid_n_heads(self):
        from squish.attention.razor_attn import RazorAttentionConfig
        with self.assertRaises(ValueError):
            RazorAttentionConfig(n_heads=0)

    def test_config_invalid_entropy_threshold(self):
        from squish.attention.razor_attn import RazorAttentionConfig
        with self.assertRaises(ValueError):
            RazorAttentionConfig(entropy_threshold=1.5)

    def test_initial_head_types_unclassified(self):
        from squish.attention.razor_attn import RazorHeadType
        razor = self._make(n_heads=4)
        types = razor.head_types()
        self.assertTrue(all(t == RazorHeadType.UNCLASSIFIED for t in types))

    def test_calibrate_classifies_heads(self):
        from squish.attention.razor_attn import RazorHeadType
        razor = self._make(n_heads=4)
        T = 32
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        types = razor.head_types()
        for t in types:
            self.assertIn(t, [RazorHeadType.RETRIEVAL, RazorHeadType.NON_RETRIEVAL])

    def test_calibrate_head_types_exhaustive(self):
        razor = self._make(n_heads=4)
        for _ in range(3):
            T = 48
            Q = np.random.randn(4, T, 16).astype(np.float32)
            K = np.random.randn(4, T, 16).astype(np.float32)
            V = np.random.randn(4, T, 16).astype(np.float32)
            razor.calibrate(Q, K, V)
        n_heads = 4
        r = len(razor.retrieval_head_indices())
        nr = len(razor.non_retrieval_head_indices())
        self.assertEqual(r + nr, n_heads)

    def test_retrieval_indices_unique(self):
        razor = self._make(n_heads=4)
        T = 24
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        indices = razor.retrieval_head_indices()
        self.assertEqual(len(set(indices)), len(indices))

    def test_non_retrieval_indices_unique(self):
        razor = self._make(n_heads=4)
        T = 24
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        indices = razor.non_retrieval_head_indices()
        self.assertEqual(len(set(indices)), len(indices))

    def test_high_entropy_threshold_all_non_retrieval(self):
        razor = self._make(n_heads=4, entropy_threshold=1.0)
        T = 32
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        self.assertEqual(len(razor.retrieval_head_indices()), 0)

    def test_low_entropy_threshold_all_retrieval(self):
        razor = self._make(n_heads=4, entropy_threshold=0.0)
        T = 32
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        self.assertEqual(len(razor.retrieval_head_indices()), 4)

    def test_forward_output_shape(self):
        razor = self._make(n_heads=4, head_dim=16)
        T = 32
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        out = razor.forward(Q, K, V)
        self.assertEqual(out.shape, (4, T, 16))

    def test_forward_dtype_float32(self):
        razor = self._make(n_heads=4, head_dim=16)
        T = 16
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        out = razor.forward(Q, K, V)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_output_finite(self):
        razor = self._make(n_heads=4, head_dim=16)
        T = 16
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        razor.calibrate(Q, K, V)
        out = razor.forward(Q, K, V)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_forward_without_calibration(self):
        razor = self._make(n_heads=4, head_dim=16)
        T = 8
        Q = np.random.randn(4, T, 16).astype(np.float32)
        K = np.random.randn(4, T, 16).astype(np.float32)
        V = np.random.randn(4, T, 16).astype(np.float32)
        # Should not raise — uses all-non-retrieval default
        out = razor.forward(Q, K, V)
        self.assertEqual(out.shape, (4, T, 16))

    def test_calibrate_wrong_head_count(self):
        razor = self._make(n_heads=4)
        T = 16
        wrong_H = 8
        Q = np.random.randn(wrong_H, T, 16).astype(np.float32)
        K = np.random.randn(wrong_H, T, 16).astype(np.float32)
        V = np.random.randn(wrong_H, T, 16).astype(np.float32)
        with self.assertRaises(ValueError):
            razor.calibrate(Q, K, V)


# ---------------------------------------------------------------------------
# CacheBlend
# ---------------------------------------------------------------------------

class TestCacheBlend(unittest.TestCase):
    def _make(self, n_heads=2, head_dim=8, max_cached_seqs=16, blend_overlap=2):
        from squish.kv.cacheblend import CacheBlend, CacheBlendConfig
        return CacheBlend(CacheBlendConfig(
            n_heads=n_heads,
            head_dim=head_dim,
            max_cached_seqs=max_cached_seqs,
            blend_overlap=blend_overlap,
        ))

    def test_config_defaults(self):
        from squish.kv.cacheblend import CacheBlendConfig
        cfg = CacheBlendConfig()
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.max_cached_seqs, 256)

    def test_config_invalid_n_heads(self):
        from squish.kv.cacheblend import CacheBlendConfig
        with self.assertRaises(ValueError):
            CacheBlendConfig(n_heads=0)

    def test_config_invalid_blend_overlap(self):
        from squish.kv.cacheblend import CacheBlendConfig
        with self.assertRaises(ValueError):
            CacheBlendConfig(blend_overlap=0)

    def test_initial_n_cached_zero(self):
        cb = self._make()
        self.assertEqual(cb.n_cached, 0)

    def test_store_kv_increments_n_cached(self):
        cb = self._make()
        ids = np.array([1, 2, 3], dtype=np.int32)
        K = np.random.randn(3, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        V = np.random.randn(3, 2, 8).astype(np.float32)
        cb.store_kv(ids, K, V)
        self.assertEqual(cb.n_cached, 1)

    def test_store_kv_multiple(self):
        cb = self._make()
        for i in range(5):
            ids = np.array([i * 10, i * 10 + 1], dtype=np.int32)
            K = np.random.randn(2, 2, 8).astype(np.float32)  # (seq_len=2, n_heads=2, head_dim=8)
            V = np.random.randn(2, 2, 8).astype(np.float32)
            cb.store_kv(ids, K, V)
        self.assertEqual(cb.n_cached, 5)

    def test_blend_no_match_zero_cache_hit(self):
        cb = self._make(n_heads=2, head_dim=8, blend_overlap=2)
        fresh_ids = np.array([100, 101, 102, 103], dtype=np.int32)
        T = 4
        fresh_K = np.random.randn(T, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        fresh_V = np.random.randn(T, 2, 8).astype(np.float32)
        result = cb.blend(fresh_ids, fresh_K, fresh_V)
        self.assertEqual(result.cache_hit_ratio, 0.0)

    def test_blend_k_shape(self):
        cb = self._make(n_heads=2, head_dim=8)
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        K = np.random.randn(4, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        V = np.random.randn(4, 2, 8).astype(np.float32)
        result = cb.blend(ids, K, V)
        self.assertEqual(result.K.shape[1], 2)   # n_heads
        self.assertEqual(result.K.shape[2], 8)   # head_dim

    def test_blend_v_shape(self):
        cb = self._make(n_heads=2, head_dim=8)
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        K = np.random.randn(4, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        V = np.random.randn(4, 2, 8).astype(np.float32)
        result = cb.blend(ids, K, V)
        self.assertEqual(result.V.shape[1], 2)   # n_heads
        self.assertEqual(result.V.shape[2], 8)   # head_dim

    def test_blend_with_matching_prefix(self):
        cb = self._make(n_heads=2, head_dim=8, blend_overlap=2)
        ids = np.array([10, 20, 30, 40], dtype=np.int32)
        K = np.random.randn(4, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        V = np.random.randn(4, 2, 8).astype(np.float32)
        cb.store_kv(ids, K, V)
        # New request shares the same prefix
        new_ids = np.array([10, 20, 30, 40, 50, 60], dtype=np.int32)
        new_K = np.random.randn(6, 2, 8).astype(np.float32)
        new_V = np.random.randn(6, 2, 8).astype(np.float32)
        result = cb.blend(new_ids, new_K, new_V)
        self.assertGreater(result.cached_tokens, 0)
        self.assertGreater(result.cache_hit_ratio, 0.0)

    def test_blend_cached_plus_recomputed(self):
        cb = self._make(n_heads=2, head_dim=8, blend_overlap=2)
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        K = np.random.randn(4, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        V = np.random.randn(4, 2, 8).astype(np.float32)
        result = cb.blend(ids, K, V)
        total = result.cached_tokens + result.recomputed_tokens
        self.assertEqual(total, len(ids))

    def test_blend_result_k_dtype(self):
        cb = self._make(n_heads=2, head_dim=8)
        ids = np.array([1, 2, 3], dtype=np.int32)
        K = np.random.randn(3, 2, 8).astype(np.float32)  # (seq_len, n_heads, head_dim)
        V = np.random.randn(3, 2, 8).astype(np.float32)
        result = cb.blend(ids, K, V)
        self.assertEqual(result.K.dtype, np.float32)
        self.assertEqual(result.V.dtype, np.float32)

    def test_cache_eviction_respects_max(self):
        cb = self._make(max_cached_seqs=3)
        for i in range(5):
            ids = np.array([i * 5, i * 5 + 1], dtype=np.int32)
            K = np.random.randn(2, 2, 8).astype(np.float32)  # (seq_len=2, n_heads=2, head_dim=8)
            V = np.random.randn(2, 2, 8).astype(np.float32)
            cb.store_kv(ids, K, V)
        self.assertLessEqual(cb.n_cached, 3)


# ---------------------------------------------------------------------------
# GreenKVEviction
# ---------------------------------------------------------------------------

class TestGreenKV(unittest.TestCase):
    def _make(self, n_heads=4, head_dim=8, global_budget=32, obs_window=4, min_head_budget=2):
        from squish.kv.green_kv import GreenKVEviction, GreenKVConfig
        return GreenKVEviction(GreenKVConfig(
            global_budget=global_budget,
            obs_window=obs_window,
            min_head_budget=min_head_budget,
            n_heads=n_heads,
            head_dim=head_dim,
        ))

    def test_config_defaults(self):
        from squish.kv.green_kv import GreenKVConfig
        cfg = GreenKVConfig()
        self.assertEqual(cfg.global_budget, 512)
        self.assertEqual(cfg.obs_window, 32)
        self.assertEqual(cfg.min_head_budget, 16)

    def test_config_invalid_global_budget(self):
        from squish.kv.green_kv import GreenKVConfig
        with self.assertRaises(ValueError):
            GreenKVConfig(global_budget=0)

    def test_config_invalid_obs_window(self):
        from squish.kv.green_kv import GreenKVConfig
        with self.assertRaises(ValueError):
            GreenKVConfig(obs_window=0)

    def test_config_property(self):
        evict = self._make()
        from squish.kv.green_kv import GreenKVConfig
        self.assertIsInstance(evict.config, GreenKVConfig)

    def test_compress_returns_three_items(self):
        evict = self._make(n_heads=4, head_dim=8, global_budget=16, obs_window=4)
        seq_len = 64
        K = np.random.randn(4, seq_len, 8).astype(np.float32)
        V = np.random.randn(4, seq_len, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        result = evict.compress(Q_obs, K, V)
        self.assertEqual(len(result), 3)

    def test_compress_k_keep_is_list(self):
        evict = self._make(n_heads=4, head_dim=8, global_budget=16, obs_window=4)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        K_keep, V_keep, kept_idx = evict.compress(Q_obs, K, V)
        self.assertEqual(len(K_keep), 4)

    def test_compress_k_shape_reduced(self):
        evict = self._make(n_heads=4, head_dim=8, global_budget=16, obs_window=4)
        seq_len = 64
        K = np.random.randn(4, seq_len, 8).astype(np.float32)
        V = np.random.randn(4, seq_len, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        K_keep, V_keep, kept_idx = evict.compress(Q_obs, K, V)
        for h in range(4):
            self.assertLessEqual(K_keep[h].shape[0], seq_len)

    def test_compress_k_v_shape_match(self):
        evict = self._make(n_heads=4, head_dim=8, global_budget=16, obs_window=4)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        K_keep, V_keep, kept_idx = evict.compress(Q_obs, K, V)
        for h in range(4):
            self.assertEqual(K_keep[h].shape[0], V_keep[h].shape[0])

    def test_compress_head_dim_preserved(self):
        evict = self._make(n_heads=4, head_dim=8, global_budget=16, obs_window=4)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        K_keep, V_keep, _ = evict.compress(Q_obs, K, V)
        for h in range(4):
            self.assertEqual(K_keep[h].shape[1], 8)
            self.assertEqual(V_keep[h].shape[1], 8)

    def test_compress_kept_idx_in_range(self):
        evict = self._make(n_heads=2, head_dim=8, global_budget=8, obs_window=4)
        seq_len = 32
        K = np.random.randn(2, seq_len, 8).astype(np.float32)
        V = np.random.randn(2, seq_len, 8).astype(np.float32)
        Q_obs = np.random.randn(2, 4, 8).astype(np.float32)
        _, _, kept_idx = evict.compress(Q_obs, K, V)
        for h in range(2):
            self.assertTrue(np.all(kept_idx[h] >= 0))
            self.assertTrue(np.all(kept_idx[h] < seq_len))

    def test_compress_total_retained_lte_global_budget(self):
        evict = self._make(n_heads=4, head_dim=8, global_budget=20, obs_window=4, min_head_budget=2)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        K_keep, _, _ = evict.compress(Q_obs, K, V)
        total = sum(K_keep[h].shape[0] for h in range(4))
        self.assertLessEqual(total, 20)

    def test_compress_min_head_budget_respected(self):
        min_b = 3
        evict = self._make(n_heads=4, head_dim=8, global_budget=40, obs_window=4, min_head_budget=min_b)
        K = np.random.randn(4, 64, 8).astype(np.float32)
        V = np.random.randn(4, 64, 8).astype(np.float32)
        Q_obs = np.random.randn(4, 4, 8).astype(np.float32)
        K_keep, _, _ = evict.compress(Q_obs, K, V)
        for h in range(4):
            self.assertGreaterEqual(K_keep[h].shape[0], min_b)

    def test_compress_short_seq_no_error(self):
        """Seq shorter than global_budget — keep everything."""
        evict = self._make(n_heads=2, head_dim=8, global_budget=64, obs_window=4, min_head_budget=1)
        seq_len = 10
        K = np.random.randn(2, seq_len, 8).astype(np.float32)
        V = np.random.randn(2, seq_len, 8).astype(np.float32)
        Q_obs = np.random.randn(2, 4, 8).astype(np.float32)
        K_keep, _, _ = evict.compress(Q_obs, K, V)
        for h in range(2):
            self.assertLessEqual(K_keep[h].shape[0], seq_len)

    def test_compress_dtype_preserved(self):
        evict = self._make(n_heads=2, head_dim=8, global_budget=8, obs_window=4)
        K = np.random.randn(2, 32, 8).astype(np.float32)
        V = np.random.randn(2, 32, 8).astype(np.float32)
        Q_obs = np.random.randn(2, 4, 8).astype(np.float32)
        K_keep, V_keep, _ = evict.compress(Q_obs, K, V)
        for h in range(2):
            self.assertEqual(K_keep[h].dtype, np.float32)
            self.assertEqual(V_keep[h].dtype, np.float32)


# ---------------------------------------------------------------------------
# PrebeleRouter
# ---------------------------------------------------------------------------

class TestPrebeleRouter(unittest.TestCase):
    def _make(self, n_servers=4, chunk_size=16, load_weight=0.1):
        from squish.serving.preble_router import PrebeleRouter, PrebeleConfig
        return PrebeleRouter(PrebeleConfig(
            n_servers=n_servers,
            chunk_size=chunk_size,
            load_weight=load_weight,
        ))

    def test_config_defaults(self):
        from squish.serving.preble_router import PrebeleConfig
        cfg = PrebeleConfig()
        self.assertEqual(cfg.n_servers, 4)
        self.assertEqual(cfg.chunk_size, 64)
        self.assertAlmostEqual(cfg.load_weight, 0.1)

    def test_config_invalid_n_servers(self):
        from squish.serving.preble_router import PrebeleConfig
        with self.assertRaises(ValueError):
            PrebeleConfig(n_servers=0)

    def test_config_invalid_chunk_size(self):
        from squish.serving.preble_router import PrebeleConfig
        with self.assertRaises(ValueError):
            PrebeleConfig(chunk_size=0)

    def test_config_invalid_load_weight(self):
        from squish.serving.preble_router import PrebeleConfig
        with self.assertRaises(ValueError):
            PrebeleConfig(load_weight=-0.1)

    def test_config_property(self):
        router = self._make()
        from squish.serving.preble_router import PrebeleConfig
        self.assertIsInstance(router.config, PrebeleConfig)

    def test_initial_server_loads_zero(self):
        router = self._make(n_servers=4)
        loads = router.server_loads
        self.assertEqual(len(loads), 4)
        self.assertTrue(all(l == 0 for l in loads))

    def test_route_returns_valid_server(self):
        router = self._make(n_servers=4)
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        result = router.route(ids)
        self.assertGreaterEqual(result.server_id, 0)
        self.assertLess(result.server_id, 4)

    def test_route_increments_load(self):
        router = self._make(n_servers=4)
        ids = np.array([10, 20, 30], dtype=np.int32)
        result = router.route(ids)
        self.assertEqual(router.server_loads[result.server_id], 1)

    def test_route_multiple_increments(self):
        router = self._make(n_servers=4)
        for _ in range(5):
            ids = np.array([1, 2, 3], dtype=np.int32)
            router.route(ids)
        total_load = sum(router.server_loads)
        self.assertEqual(total_load, 5)

    def test_complete_request_decrements_load(self):
        router = self._make(n_servers=4)
        ids = np.array([1, 2, 3], dtype=np.int32)
        result = router.route(ids)
        sid = result.server_id
        before = router.server_loads[sid]
        router.complete_request(sid)
        self.assertEqual(router.server_loads[sid], before - 1)

    def test_load_never_negative(self):
        router = self._make(n_servers=4)
        router.complete_request(0)  # No-op on zero load
        self.assertGreaterEqual(router.server_loads[0], 0)

    def test_warm_cache_then_route_higher_overlap(self):
        router = self._make(n_servers=4)
        ids = np.array([5, 6, 7, 8, 9, 10], dtype=np.int32)
        router.warm_cache(0, ids)
        result = router.route(ids)
        # Server 0 has the warm prefix → should be selected
        self.assertEqual(result.server_id, 0)
        self.assertGreater(result.overlap_score, 0)

    def test_cache_stats_length(self):
        router = self._make(n_servers=4)
        stats = router.cache_stats()
        self.assertEqual(len(stats), 4)

    def test_cache_stats_increases_after_warm(self):
        router = self._make(n_servers=4, chunk_size=2)
        ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
        router.warm_cache(0, ids)
        stats = router.cache_stats()
        self.assertGreater(stats[0], 0)

    def test_route_overlap_score_nonneg(self):
        router = self._make(n_servers=3)
        ids = np.array([1, 2, 3, 4], dtype=np.int32)
        result = router.route(ids)
        self.assertGreaterEqual(result.overlap_score, 0)

    def test_route_current_load_nonneg(self):
        router = self._make(n_servers=3)
        ids = np.array([10, 20], dtype=np.int32)
        result = router.route(ids)
        self.assertGreaterEqual(result.current_load, 0)

    def test_route_short_ids(self):
        router = self._make(n_servers=2, chunk_size=16)
        ids = np.array([42], dtype=np.int32)
        result = router.route(ids)
        self.assertGreaterEqual(result.server_id, 0)
        self.assertLess(result.server_id, 2)


if __name__ == "__main__":
    unittest.main()
