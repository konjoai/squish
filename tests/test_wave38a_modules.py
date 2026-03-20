"""
tests/test_wave38a_modules.py

Wave 38a integration tests:
  - QuestAttention          (squish/attention/quest_attn.py)
  - SnapKV                  (squish/kv/snap_kv.py)
  - MagicDecAttention       (squish/attention/magic_dec.py)
  - InfiniGenKVManager      (squish/kv/infinite_gen.py)
  - RetrievalAttention      (squish/attention/retrieval_attn.py)
  - OuroborosDrafter        (squish/speculative/ouroboros_draft.py)

Coverage: config validation, happy-path core operations, edge cases,
stats tracking, error paths.  >= 72 tests, all deterministic.
"""
import math
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# QuestAttention
# ---------------------------------------------------------------------------
from squish.attention.quest_attn import QuestAttention, QuestConfig, QuestStats


class TestQuestConfig(unittest.TestCase):
    def test_default_config_valid(self):
        cfg = QuestConfig()
        self.assertGreater(cfg.budget_ratio, 0)
        self.assertLessEqual(cfg.budget_ratio, 1.0)
        self.assertGreater(cfg.page_size, 0)

    def test_custom_values(self):
        cfg = QuestConfig(budget_ratio=0.5, page_size=8)
        self.assertEqual(cfg.budget_ratio, 0.5)
        self.assertEqual(cfg.page_size, 8)

    def test_invalid_budget_ratio_zero(self):
        with self.assertRaises(ValueError):
            QuestConfig(budget_ratio=0.0)

    def test_invalid_budget_ratio_over_one(self):
        with self.assertRaises(ValueError):
            QuestConfig(budget_ratio=1.1)

    def test_invalid_page_size(self):
        with self.assertRaises(ValueError):
            QuestConfig(page_size=0)

    def test_invalid_min_length(self):
        with self.assertRaises(ValueError):
            QuestConfig(min_length=-1)


class TestQuestAttention(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.hd = 32

    def _make_kv(self, seq_len):
        k = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        v = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        q = self.rng.standard_normal(self.hd).astype(np.float32)
        return q, k, v

    def test_exact_path_short_context(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=100)
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(32)
        out = qa.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))
        self.assertEqual(qa.stats.exact_calls, 1)
        self.assertEqual(qa.stats.sparse_calls, 0)

    def test_sparse_path_long_context(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4)
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(128)
        out = qa.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))
        self.assertEqual(qa.stats.sparse_calls, 1)

    def test_output_finite(self):
        qa = QuestAttention(QuestConfig(head_dim=self.hd, min_length=8, page_size=4))
        q, k, v = self._make_kv(64)
        out = qa.attend(q, k, v)
        self.assertTrue(np.isfinite(out).all())

    def test_page_score_fn_max(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4,
                          page_score_fn="max")
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(64)
        out = qa.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))

    def test_page_score_fn_first(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4,
                          page_score_fn="first")
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(64)
        out = qa.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))

    def test_stats_counting(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4)
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(64)
        for _ in range(5):
            qa.attend(q, k, v)
        self.assertEqual(qa.stats.attn_calls, 5)

    def test_reset_stats(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4)
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(64)
        qa.attend(q, k, v)
        qa.reset_stats()
        self.assertEqual(qa.stats.attn_calls, 0)
        self.assertEqual(qa.stats.sparse_calls, 0)

    def test_sparsity_stat_positive(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4,
                          budget_ratio=0.25)
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(128)
        qa.attend(q, k, v)
        self.assertGreater(qa.stats.sparsity, 0.0)

    def test_budget_ratio_1_selects_all_pages(self):
        cfg = QuestConfig(head_dim=self.hd, min_length=8, page_size=4,
                          budget_ratio=1.0)
        qa = QuestAttention(cfg)
        q, k, v = self._make_kv(64)
        qa.attend(q, k, v)
        self.assertEqual(qa.stats.total_pages_skipped, 0)


# ---------------------------------------------------------------------------
# SnapKV
# ---------------------------------------------------------------------------
from squish.kv.snap_kv import SnapKV, SnapKVConfig, SnapKVStats


class TestSnapKVConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = SnapKVConfig()
        self.assertGreater(cfg.obs_window, 0)
        self.assertGreater(cfg.budget, 0)
        self.assertGreater(cfg.pool_kernel, 0)

    def test_invalid_obs_window(self):
        with self.assertRaises(ValueError):
            SnapKVConfig(obs_window=0)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            SnapKVConfig(budget=0)

    def test_invalid_pool_kernel(self):
        with self.assertRaises(ValueError):
            SnapKVConfig(pool_kernel=0)


class TestSnapKV(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.hd = 32

    def _make_kv(self, seq_len):
        k = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        v = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        return k, v

    def test_no_compression_short(self):
        cfg = SnapKVConfig(budget=512)
        snap = SnapKV(cfg)
        k, v = self._make_kv(32)
        k_c, v_c = snap.compress(k, v)
        np.testing.assert_array_equal(k_c, k)
        np.testing.assert_array_equal(v_c, v)

    def test_compresses_long_sequence(self):
        cfg = SnapKVConfig(budget=64, obs_window=8)
        snap = SnapKV(cfg)
        k, v = self._make_kv(256)
        k_c, v_c = snap.compress(k, v)
        self.assertLessEqual(k_c.shape[0], cfg.budget)
        self.assertEqual(k_c.shape[1], self.hd)

    def test_output_shape_matches(self):
        cfg = SnapKVConfig(budget=64, obs_window=8)
        snap = SnapKV(cfg)
        k, v = self._make_kv(256)
        k_c, v_c = snap.compress(k, v)
        self.assertEqual(k_c.shape, v_c.shape)

    def test_stats_updated(self):
        cfg = SnapKVConfig(budget=64, obs_window=8)
        snap = SnapKV(cfg)
        k, v = self._make_kv(256)
        snap.compress(k, v)
        self.assertEqual(snap.stats.compress_calls, 1)
        self.assertEqual(snap.stats.total_tokens_in, 256)

    def test_compression_ratio_positive(self):
        cfg = SnapKVConfig(budget=64, obs_window=8)
        snap = SnapKV(cfg)
        k, v = self._make_kv(256)
        snap.compress(k, v)
        self.assertGreater(snap.stats.mean_compression_ratio, 1.0)

    def test_reset_stats(self):
        snap = SnapKV()
        k, v = self._make_kv(64)
        snap.compress(k, v)
        snap.reset_stats()
        self.assertEqual(snap.stats.compress_calls, 0)

    def test_observation_window_always_kept(self):
        obs = 8
        budget = 32
        cfg = SnapKVConfig(budget=budget, obs_window=obs)
        snap = SnapKV(cfg)
        k, v = self._make_kv(200)
        k_c, v_c = snap.compress(k, v)
        # The compressed output should include the obs window tokens
        # (last obs positions in original should appear in output)
        self.assertGreaterEqual(k_c.shape[0], min(obs, budget))


# ---------------------------------------------------------------------------
# MagicDecAttention
# ---------------------------------------------------------------------------
from squish.attention.magic_dec import MagicDecAttention, MagicDecConfig, MagicDecStats


class TestMagicDecConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = MagicDecConfig()
        self.assertGreaterEqual(cfg.n_sinks, 0)
        self.assertGreater(cfg.n_recent, 0)
        self.assertGreater(cfg.landmark_stride, 0)

    def test_invalid_n_sinks(self):
        with self.assertRaises(ValueError):
            MagicDecConfig(n_sinks=-1)

    def test_invalid_n_recent(self):
        with self.assertRaises(ValueError):
            MagicDecConfig(n_recent=0)

    def test_invalid_landmark_stride(self):
        with self.assertRaises(ValueError):
            MagicDecConfig(landmark_stride=0)


class TestMagicDecAttention(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.hd = 32

    def _make_qkv(self, seq_len):
        q = self.rng.standard_normal(self.hd).astype(np.float32)
        k = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        v = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        return q, k, v

    def test_exact_path_short(self):
        cfg = MagicDecConfig(head_dim=self.hd, min_length=128)
        md = MagicDecAttention(cfg)
        q, k, v = self._make_qkv(32)
        out = md.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))
        self.assertEqual(md.stats.exact_calls, 1)

    def test_sparse_path_long(self):
        cfg = MagicDecConfig(head_dim=self.hd, min_length=32, n_sinks=2,
                             n_recent=16, landmark_stride=8)
        md = MagicDecAttention(cfg)
        q, k, v = self._make_qkv(128)
        out = md.attend(q, k, v)
        self.assertEqual(md.stats.sparse_calls, 1)
        self.assertEqual(out.shape, (self.hd,))

    def test_output_finite(self):
        cfg = MagicDecConfig(head_dim=self.hd, min_length=32)
        md = MagicDecAttention(cfg)
        for seq_len in [16, 64, 256]:
            q, k, v = self._make_qkv(seq_len)
            out = md.attend(q, k, v)
            self.assertTrue(np.isfinite(out).all(), f"non-finite at seq_len={seq_len}")

    def test_sparsity_positive(self):
        cfg = MagicDecConfig(head_dim=self.hd, min_length=32, n_sinks=2,
                             n_recent=16, landmark_stride=8)
        md = MagicDecAttention(cfg)
        q, k, v = self._make_qkv(512)
        md.attend(q, k, v)
        self.assertGreater(md.stats.mean_sparsity, 0.0)

    def test_stats_context_tokens_tracked(self):
        cfg = MagicDecConfig(head_dim=self.hd, min_length=32)
        md = MagicDecAttention(cfg)
        q, k, v = self._make_qkv(64)
        md.attend(q, k, v)
        self.assertEqual(md.stats.total_context_tokens, 64)

    def test_reset_stats(self):
        md = MagicDecAttention(MagicDecConfig(head_dim=self.hd))
        q, k, v = self._make_qkv(64)
        md.attend(q, k, v)
        md.reset_stats()
        self.assertEqual(md.stats.attn_calls, 0)

    def test_zero_sinks(self):
        cfg = MagicDecConfig(head_dim=self.hd, n_sinks=0, min_length=32)
        md = MagicDecAttention(cfg)
        q, k, v = self._make_qkv(128)
        out = md.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))


# ---------------------------------------------------------------------------
# InfiniGenKVManager
# ---------------------------------------------------------------------------
from squish.kv.infinite_gen import InfiniGenKVManager, InfiniGenConfig, InfiniGenStats


class TestInfiniGenConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = InfiniGenConfig()
        self.assertGreater(cfg.hot_capacity, 0)
        self.assertGreaterEqual(cfg.prefetch_k, 0)

    def test_invalid_hot_capacity(self):
        with self.assertRaises(ValueError):
            InfiniGenConfig(hot_capacity=0)

    def test_invalid_prefetch_k(self):
        with self.assertRaises(ValueError):
            InfiniGenConfig(prefetch_k=-1)

    def test_invalid_importance_decay(self):
        with self.assertRaises(ValueError):
            InfiniGenConfig(importance_decay=0.0)


class TestInfiniGenKVManager(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.hd = 16

    def _kv(self):
        k = self.rng.standard_normal(self.hd).astype(np.float32)
        v = self.rng.standard_normal(self.hd).astype(np.float32)
        return k, v

    def test_put_and_get_hot(self):
        mgr = InfiniGenKVManager(InfiniGenConfig(hot_capacity=16, head_dim=self.hd))
        k, v = self._kv()
        mgr.put(0, k, v)
        pos, ks, vs = mgr.get(np.array([0]))
        self.assertEqual(pos[0], 0)
        np.testing.assert_array_almost_equal(ks[0], k)
        self.assertEqual(mgr.stats.hot_hits, 1)

    def test_missing_position_skipped(self):
        mgr = InfiniGenKVManager(InfiniGenConfig(hot_capacity=16, head_dim=self.hd))
        pos, ks, vs = mgr.get(np.array([99]))
        self.assertEqual(len(pos), 0)

    def test_eviction_to_cold(self):
        cap = 4
        mgr = InfiniGenKVManager(InfiniGenConfig(hot_capacity=cap, head_dim=self.hd))
        for i in range(cap + 2):
            k, v = self._kv()
            mgr.put(i, k, v)
        hot_size, cold_size = mgr.size()
        self.assertLessEqual(hot_size, cap)
        self.assertGreater(cold_size, 0)

    def test_cold_to_hot_fetch(self):
        cap = 2
        mgr = InfiniGenKVManager(InfiniGenConfig(hot_capacity=cap, head_dim=self.hd))
        for i in range(cap + 1):
            k, v = self._kv()
            mgr.put(i, k, v)
        # Position 0 should be in cold; fetch it
        mgr.get(np.array([0]))
        self.assertGreater(mgr.stats.cold_fetches, 0)

    def test_update_scores(self):
        mgr = InfiniGenKVManager(InfiniGenConfig(head_dim=self.hd))
        k, v = self._kv()
        mgr.put(0, k, v)
        mgr.update_scores(np.array([0]), np.array([0.8]))
        self.assertIn(0, mgr._scores)

    def test_reset_clears_state(self):
        mgr = InfiniGenKVManager(InfiniGenConfig(head_dim=self.hd))
        k, v = self._kv()
        mgr.put(0, k, v)
        mgr.reset()
        self.assertEqual(mgr.size(), (0, 0))
        self.assertEqual(mgr.stats.hot_hits, 0)

    def test_hit_rate_perfect(self):
        mgr = InfiniGenKVManager(InfiniGenConfig(hot_capacity=100, head_dim=self.hd))
        k, v = self._kv()
        mgr.put(0, k, v)
        mgr.get(np.array([0]))
        self.assertAlmostEqual(mgr.stats.hit_rate, 1.0)


# ---------------------------------------------------------------------------
# RetrievalAttention
# ---------------------------------------------------------------------------
from squish.attention.retrieval_attn import RetrievalAttention, RetrievalAttnConfig, RetrievalAttnStats


class TestRetrievalAttnConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = RetrievalAttnConfig()
        self.assertGreater(cfg.n_neighbors, 0)
        self.assertGreater(cfg.ef_construction, 0)

    def test_invalid_n_neighbors(self):
        with self.assertRaises(ValueError):
            RetrievalAttnConfig(n_neighbors=0)

    def test_invalid_ef_construction(self):
        with self.assertRaises(ValueError):
            RetrievalAttnConfig(ef_construction=0)

    def test_invalid_ef_search(self):
        with self.assertRaises(ValueError):
            RetrievalAttnConfig(ef_search=0)


class TestRetrievalAttention(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.hd = 32

    def _make_qkv(self, seq_len):
        q = self.rng.standard_normal(self.hd).astype(np.float32)
        k = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        v = self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)
        return q, k, v

    def test_exact_path_short(self):
        cfg = RetrievalAttnConfig(head_dim=self.hd, min_length=512)
        ra = RetrievalAttention(cfg)
        q, k, v = self._make_qkv(32)
        out = ra.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))
        self.assertEqual(ra.stats.exact_calls, 1)

    def test_approx_path_long(self):
        cfg = RetrievalAttnConfig(head_dim=self.hd, min_length=32, n_neighbors=8)
        ra = RetrievalAttention(cfg)
        q, k, v = self._make_qkv(128)
        out = ra.attend(q, k, v)
        self.assertEqual(out.shape, (self.hd,))
        self.assertEqual(ra.stats.approx_calls, 1)

    def test_output_finite(self):
        cfg = RetrievalAttnConfig(head_dim=self.hd, min_length=32, n_neighbors=8)
        ra = RetrievalAttention(cfg)
        q, k, v = self._make_qkv(128)
        out = ra.attend(q, k, v)
        self.assertTrue(np.isfinite(out).all())

    def test_build_index_increments_counter(self):
        cfg = RetrievalAttnConfig(head_dim=self.hd)
        ra = RetrievalAttention(cfg)
        k = self.rng.standard_normal((64, self.hd)).astype(np.float32)
        ra.build_index(k)
        self.assertEqual(ra.stats.index_builds, 1)

    def test_backend_is_string(self):
        ra = RetrievalAttention()
        self.assertIsInstance(ra.backend, str)
        self.assertIn(ra.backend, ("numpy", "hnswlib"))

    def test_reset_stats(self):
        cfg = RetrievalAttnConfig(head_dim=self.hd, min_length=32, n_neighbors=8)
        ra = RetrievalAttention(cfg)
        q, k, v = self._make_qkv(128)
        ra.attend(q, k, v)
        ra.reset_stats()
        self.assertEqual(ra.stats.attn_calls, 0)


# ---------------------------------------------------------------------------
# OuroborosDrafter
# ---------------------------------------------------------------------------
from squish.speculative.ouroboros_draft import OuroborosDrafter, OuroborosConfig, OuroborosStats


class TestOuroborosConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = OuroborosConfig()
        self.assertGreater(cfg.depth, 0)
        self.assertGreaterEqual(cfg.feedback_window, 0)

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            OuroborosConfig(depth=0)

    def test_invalid_feedback_window(self):
        with self.assertRaises(ValueError):
            OuroborosConfig(feedback_window=-1)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            OuroborosConfig(temperature=-0.1)


class TestOuroborosDrafter(unittest.TestCase):
    def setUp(self):
        self.cfg = OuroborosConfig(depth=4, vocab_size=100, use_ngram=True)

    def test_draft_length(self):
        drafter = OuroborosDrafter(self.cfg)
        tokens = drafter.draft([1, 2, 3])
        self.assertEqual(len(tokens), self.cfg.depth)

    def test_draft_tokens_in_vocab(self):
        drafter = OuroborosDrafter(self.cfg)
        tokens = drafter.draft([1, 2, 3])
        for tok in tokens:
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, self.cfg.vocab_size)

    def test_accept_feedback_updates_stats(self):
        drafter = OuroborosDrafter(self.cfg)
        drafter.draft([1, 2, 3])
        drafter.accept_feedback([5, 6, 7])
        self.assertEqual(drafter.stats.total_accepted_tokens, 3)

    def test_ngram_table_built_from_feedback(self):
        drafter = OuroborosDrafter(self.cfg)
        drafter.accept_feedback([10, 11, 12, 13, 14])
        self.assertGreater(len(drafter._ngram_counts), 0)

    def test_feedback_influences_later_drafts(self):
        drafter = OuroborosDrafter(self.cfg)
        drafter.accept_feedback([1, 2, 3, 4, 5])
        # After feedback, n-gram lookup should sometimes succeed → feedback_uses > 0
        # (draft with context that matches an n-gram)
        for _ in range(10):
            drafter.draft([1, 2])
        self.assertGreater(drafter.stats.feedback_uses, 0)

    def test_reset(self):
        drafter = OuroborosDrafter(self.cfg)
        drafter.accept_feedback([1, 2, 3])
        drafter.reset()
        self.assertEqual(len(drafter._verified_context), 0)
        self.assertEqual(drafter.stats.total_accepted_tokens, 0)

    def test_stats_draft_steps(self):
        drafter = OuroborosDrafter(self.cfg)
        for _ in range(5):
            drafter.draft([1])
        self.assertEqual(drafter.stats.draft_steps, 5)

    def test_no_ngram_path(self):
        cfg = OuroborosConfig(depth=3, vocab_size=50, use_ngram=False)
        drafter = OuroborosDrafter(cfg)
        tokens = drafter.draft([1, 2, 3])
        self.assertEqual(len(tokens), 3)

    def test_temperature_sampling(self):
        cfg = OuroborosConfig(depth=3, vocab_size=50, use_ngram=False,
                              temperature=1.0)
        drafter = OuroborosDrafter(cfg)
        tokens = drafter.draft([1, 2, 3])
        self.assertEqual(len(tokens), 3)

    def test_mean_acceptance_rate_zero_before_accept(self):
        drafter = OuroborosDrafter(self.cfg)
        drafter.draft([1, 2])
        self.assertAlmostEqual(drafter.stats.mean_acceptance_rate, 0.0)

    def test_mean_acceptance_rate_nonzero_after_accept(self):
        drafter = OuroborosDrafter(self.cfg)
        drafter.draft([1, 2])
        drafter.accept_feedback([5])
        self.assertGreater(drafter.stats.mean_acceptance_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
