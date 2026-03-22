"""Tests for Wave 51a modules: BudgetForcingDecoder, TestTimeComputeRouter,
DVTSSearch, ChainOfDraftSampler, CoconutDecoder, PRMBeamSearch.
"""

import unittest
from typing import List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# BudgetForcingDecoder tests
# ---------------------------------------------------------------------------
from squish.serving.budget_forcing import (
    BudgetForcingConfig,
    BudgetForcingState,
    BudgetForcingDecoder,
)


class TestBudgetForcingConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = BudgetForcingConfig()
        self.assertEqual(cfg.max_thinking_tokens, 512)
        self.assertEqual(cfg.wait_token, "Wait")

    def test_invalid_max_thinking_tokens(self):
        with self.assertRaises((ValueError, Exception)):
            BudgetForcingConfig(max_thinking_tokens=0)

    def test_invalid_soft_ramp_start_low(self):
        with self.assertRaises((ValueError, Exception)):
            BudgetForcingConfig(soft_ramp_start=0.0)

    def test_invalid_soft_ramp_start_high(self):
        with self.assertRaises((ValueError, Exception)):
            BudgetForcingConfig(soft_ramp_start=1.0)

    def test_invalid_soft_ramp_max_temp(self):
        with self.assertRaises((ValueError, Exception)):
            BudgetForcingConfig(soft_ramp_max_temp=1.0)

    def test_custom_tokens(self):
        cfg = BudgetForcingConfig(wait_token="Hold", commit_token="Answer:")
        self.assertEqual(cfg.wait_token, "Hold")

    def test_max_thinking_tokens_one(self):
        cfg = BudgetForcingConfig(max_thinking_tokens=1)
        self.assertEqual(cfg.max_thinking_tokens, 1)


class TestBudgetForcingDecoder(unittest.TestCase):
    def setUp(self):
        self.cfg = BudgetForcingConfig(max_thinking_tokens=4)
        self.dec = BudgetForcingDecoder(self.cfg)

    def test_new_state_not_committed(self):
        state = self.dec.new_state()
        self.assertFalse(state.committed)

    def test_new_state_not_budget_exhausted(self):
        state = self.dec.new_state()
        self.assertFalse(state.budget_exhausted)

    def test_step_increments_counter(self):
        state = self.dec.new_state()
        self.dec.step(self.cfg.think_open_token, state)  # enter thinking segment
        self.dec.step("some_token", state)
        self.assertEqual(state.thinking_tokens_used, 2)

    def test_budget_fraction_zero_at_start(self):
        state = self.dec.new_state()
        self.assertAlmostEqual(self.dec.budget_fraction(state), 0.0)

    def test_budget_fraction_increases(self):
        state = self.dec.new_state()
        self.dec.step(self.cfg.think_open_token, state)  # enter thinking segment
        self.dec.step("tok", state)
        self.assertGreater(self.dec.budget_fraction(state), 0.0)

    def test_budget_exhausted_after_max_tokens(self):
        state = self.dec.new_state()
        self.dec.step(self.cfg.think_open_token, state)  # enter thinking segment
        # max_thinking_tokens=4; already used 1 above; 3 more to exhaust
        for _ in range(3):
            self.dec.step("tok", state)
        self.assertTrue(state.budget_exhausted)

    def test_inject_wait(self):
        state = self.dec.new_state()
        self.dec.inject_wait(state)
        self.assertIn("Wait", state.injections)

    def test_reset_clears_state(self):
        state = self.dec.new_state()
        for _ in range(4):
            self.dec.step("tok", state)
        self.dec.reset(state)
        self.assertEqual(state.thinking_tokens_used, 0)

    def test_should_extend_returns_bool(self):
        state = self.dec.new_state()
        result = self.dec.should_extend(state)
        self.assertIsInstance(result, bool)

    def test_temperature_multiplier_near_budget(self):
        state = self.dec.new_state()
        for _ in range(3):
            self.dec.step("tok", state)
        mult = self.dec._temperature_multiplier(state)
        self.assertGreaterEqual(mult, 1.0)


# ---------------------------------------------------------------------------
# TestTimeComputeRouter tests
# ---------------------------------------------------------------------------
from squish.sampling.test_time_scale import (
    ComputeStrategy,
    TestTimeScaleConfig,
    TestTimeScaleResult,
    TestTimeComputeRouter,
)


class TestComputeStrategy(unittest.TestCase):
    def test_greedy_value(self):
        self.assertEqual(ComputeStrategy.GREEDY.value, "greedy")

    def test_four_strategies(self):
        self.assertEqual(len(ComputeStrategy), 4)


class TestTestTimeScaleConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TestTimeScaleConfig()
        self.assertGreater(cfg.hard_threshold, cfg.easy_threshold)

    def test_invalid_thresholds(self):
        with self.assertRaises((ValueError, Exception)):
            TestTimeScaleConfig(easy_threshold=5.0, hard_threshold=1.0)

    def test_best_of_n_n_positive(self):
        cfg = TestTimeScaleConfig(best_of_n_n=16)
        self.assertEqual(cfg.best_of_n_n, 16)

    def test_top_p_range(self):
        cfg = TestTimeScaleConfig(top_p=0.95)
        self.assertAlmostEqual(cfg.top_p, 0.95)


class TestTestTimeComputeRouter(unittest.TestCase):
    def setUp(self):
        self.router = TestTimeComputeRouter(TestTimeScaleConfig())

    def test_route_returns_result(self):
        logits = np.random.randn(100).astype(np.float32)
        result = self.router.route(logits)
        self.assertIsInstance(result, TestTimeScaleResult)

    def test_route_strategy_valid(self):
        logits = np.random.randn(100).astype(np.float32)
        result = self.router.route(logits)
        self.assertIsInstance(result.strategy, ComputeStrategy)

    def test_route_entropy_nonnegative(self):
        logits = np.random.randn(50).astype(np.float32)
        result = self.router.route(logits)
        self.assertGreaterEqual(result.entropy, 0.0)

    def test_route_from_probs(self):
        probs = np.array([0.9, 0.05, 0.05])
        result = self.router.route_from_probs(probs)
        self.assertIsInstance(result, TestTimeScaleResult)

    def test_reset_stats(self):
        logits = np.random.randn(50).astype(np.float32)
        self.router.route(logits)
        self.router.reset_stats()
        stats = self.router.routing_stats()
        self.assertEqual(sum(stats.values()), 0)

    def test_routing_stats_keys(self):
        stats = self.router.routing_stats()
        # routing_stats returns string keys matching ComputeStrategy values
        for s in ComputeStrategy:
            self.assertIn(s.value, stats)

    def test_low_entropy_picks_greedy(self):
        # Peaked distribution → low entropy → GREEDY
        probs = np.zeros(100)
        probs[0] = 0.999
        probs[1:] = 0.001 / 99
        result = self.router.route_from_probs(probs)
        self.assertEqual(result.strategy, ComputeStrategy.GREEDY)


# ---------------------------------------------------------------------------
# DVTSSearch tests
# ---------------------------------------------------------------------------
from squish.sampling.dvts_search import (
    DVTSConfig,
    DVTSNode,
    DVTSResult,
    DVTSSearch,
)


def _dummy_prm_scorer(tokens, step=0):
    return float(np.sum(tokens) % 5) / 5.0


def _dummy_expand_fn(tokens):
    rng = np.random.default_rng(sum(tokens) % 100)
    extras = [list(rng.integers(1, 10, size=3)) for _ in range(2)]
    return [(e, float(-rng.uniform(0.1, 0.5))) for e in extras]


def _dummy_extract_answer(tokens):
    return "ans" if len(tokens) > 8 else None


class TestDVTSConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DVTSConfig()
        self.assertEqual(cfg.n_subtrees, 4)
        self.assertGreater(cfg.diversity_temperature, 0)

    def test_invalid_n_subtrees(self):
        with self.assertRaises((ValueError, Exception)):
            DVTSConfig(n_subtrees=0)


class TestDVTSNode(unittest.TestCase):
    def test_combined_score(self):
        node = DVTSNode(prefix=[1, 2], token_score=0.5, prm_score=0.8, depth=1)
        self.assertIsInstance(node.combined_score, float)

    def test_is_leaf_with_no_children(self):
        node = DVTSNode(prefix=[1], token_score=0.0, prm_score=0.0, depth=0)
        self.assertTrue(node.is_leaf)


class TestDVTSSearch(unittest.TestCase):
    def setUp(self):
        self.cfg = DVTSConfig(n_subtrees=2, expand_depth=3, seed=42)
        self.search = DVTSSearch(self.cfg)

    def test_run_returns_result(self):
        result = self.search.run(
            seed_tokens=[1, 2, 3],
            prm_scorer=_dummy_prm_scorer,
            expand_fn=_dummy_expand_fn,
            extract_answer=_dummy_extract_answer,
        )
        self.assertIsInstance(result, DVTSResult)

    def test_run_best_answer_nonempty(self):
        result = self.search.run([1, 2], _dummy_prm_scorer, _dummy_expand_fn, _dummy_extract_answer)
        self.assertIsNotNone(result.best_answer)

    def test_nodes_expanded_positive(self):
        result = self.search.run([1], _dummy_prm_scorer, _dummy_expand_fn, _dummy_extract_answer)
        self.assertGreater(result.n_nodes_expanded, 0)

    def test_diverse_seeds(self):
        seeds = self.search.make_diverse_seeds([1, 2, 3], vocab_size=50)
        self.assertEqual(len(seeds), self.cfg.n_subtrees)

    def test_subtree_roots_count(self):
        result = self.search.run([1, 2], _dummy_prm_scorer, _dummy_expand_fn, _dummy_extract_answer)
        self.assertLessEqual(len(result.subtree_roots), self.cfg.n_subtrees)


# ---------------------------------------------------------------------------
# ChainOfDraftSampler tests
# ---------------------------------------------------------------------------
from squish.sampling.chain_of_draft import (
    ChainOfDraftConfig,
    ChainOfDraftState,
    ChainOfDraftSampler,
)


class TestChainOfDraftConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ChainOfDraftConfig()
        self.assertGreater(cfg.max_step_tokens, 0)
        self.assertGreater(cfg.length_penalty, 0)

    def test_invalid_max_step_tokens(self):
        with self.assertRaises((ValueError, Exception)):
            ChainOfDraftConfig(max_step_tokens=0)

    def test_invalid_length_penalty(self):
        with self.assertRaises((ValueError, Exception)):
            ChainOfDraftConfig(length_penalty=-1.0)


class TestChainOfDraftSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = ChainOfDraftConfig(max_step_tokens=3)
        self.sampler = ChainOfDraftSampler(self.cfg)

    def test_new_state_zero_counts(self):
        state = self.sampler.new_state()
        self.assertEqual(state.current_step_tokens, 0)
        self.assertEqual(state.steps_completed, 0)

    def test_step_returns_pair(self):
        state = self.sampler.new_state()
        result = self.sampler.step("tok", state)
        self.assertEqual(len(result), 2)

    def test_step_increments_token_count(self):
        state = self.sampler.new_state()
        self.sampler.step("tok", state)
        self.assertEqual(state.current_step_tokens, 1)

    def test_apply_penalty_reduces_logits(self):
        state = self.sampler.new_state()
        logits = np.zeros(10, dtype=np.float32)
        penalised = self.sampler.apply_penalty(logits, 2.0)
        self.assertTrue(np.all(penalised <= 0.0))

    def test_compression_ratio_zero_initial(self):
        state = self.sampler.new_state()
        ratio = self.sampler.compression_ratio(state)
        self.assertIsInstance(ratio, float)

    def test_boundary_token_resets_step_count(self):
        state = self.sampler.new_state()
        self.sampler.step("tok1", state)
        self.sampler.step("\n\n", state)
        self.assertGreaterEqual(state.steps_completed, 1)


# ---------------------------------------------------------------------------
# CoconutDecoder tests
# ---------------------------------------------------------------------------
from squish.reasoning.coconut import (
    CoconutConfig,
    LatentThoughtState,
    CoconutResult,
    CoconutDecoder,
)


class TestCoconutConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CoconutConfig()
        self.assertGreaterEqual(cfg.max_latent_steps, 1)
        self.assertGreaterEqual(cfg.beam_width, 1)

    def test_invalid_max_latent_steps(self):
        with self.assertRaises((ValueError, Exception)):
            CoconutConfig(max_latent_steps=0)

    def test_invalid_beam_width(self):
        with self.assertRaises((ValueError, Exception)):
            CoconutConfig(beam_width=0)

    def test_invalid_latent_dim(self):
        with self.assertRaises((ValueError, Exception)):
            CoconutConfig(latent_dim=0)

    def test_custom_latent_dim(self):
        cfg = CoconutConfig(latent_dim=128)
        self.assertEqual(cfg.latent_dim, 128)


class TestCoconutFallback(unittest.TestCase):
    def setUp(self):
        self.cfg = CoconutConfig(max_latent_steps=4, fallback_to_token_decode=True)
        self.decoder = CoconutDecoder(self.cfg)

    def test_fallback_decode(self):
        result = self.decoder.decode("test prompt")
        self.assertIsInstance(result, CoconutResult)
        self.assertTrue(result.used_fallback)

    def test_fallback_n_latent_steps_positive(self):
        result = self.decoder.decode("test problem")
        self.assertGreater(result.n_latent_steps, 0)

    def test_fallback_answer_nonempty(self):
        result = self.decoder.decode("test")
        self.assertTrue(len(result.answer) > 0)


class TestCoconutWithProjection(unittest.TestCase):
    def setUp(self):
        self.cfg = CoconutConfig(latent_dim=16, max_latent_steps=3, beam_width=2)
        rng = np.random.default_rng(0)
        W = rng.standard_normal((16, 16)).astype(np.float32) * 0.01
        self.decoder = CoconutDecoder(
            self.cfg,
            projection_head=lambda h, z: np.tanh(W @ z),
            answer_decoder=lambda z: f"ans:{z[0]:.2f}",
        )

    def test_decode_with_projection(self):
        result = self.decoder.decode("problem", hidden_state=np.ones(16))
        self.assertFalse(result.used_fallback)

    def test_decode_n_steps_equals_max(self):
        result = self.decoder.decode("problem", hidden_state=np.ones(16))
        self.assertEqual(result.n_latent_steps, self.cfg.max_latent_steps)

    def test_token_reduction_ratio(self):
        result = self.decoder.decode("x", hidden_state=np.ones(16))
        self.assertGreater(result.token_reduction_ratio, 0.0)

    def test_install_answer_decoder(self):
        decoder = CoconutDecoder(self.cfg, projection_head=lambda h, z: z)
        decoder.install_answer_decoder(lambda z: "custom")
        result = decoder.decode("x", hidden_state=np.zeros(16))
        self.assertEqual(result.answer, "custom")

    def test_decode_no_context(self):
        # self.decoder has a projection_head installed → should NOT use fallback
        result = self.decoder.decode("no context")
        self.assertFalse(result.used_fallback)
        # A fresh decoder without projection_head → fallback
        dec_no_head = CoconutDecoder(self.cfg)
        result2 = dec_no_head.decode("prompt")
        self.assertTrue(result2.used_fallback)


# ---------------------------------------------------------------------------
# PRMBeamSearch tests
# ---------------------------------------------------------------------------
from squish.sampling.prm_beam_search import (
    PRMBeamConfig,
    PRMBeamCandidate,
    PRMBeamResult,
    PRMBeamSearch,
)


def _prm_scorer(tokens, step):
    return 0.5 + 0.1 * (step % 3)


def _expand_fn(tokens):
    rng = np.random.default_rng(sum(tokens) % 7)
    return [(list(rng.integers(1, 20, size=2)), float(-rng.uniform(0.1, 0.6)))]


def _extract_answer(tokens):
    return f"answer_{len(tokens)}" if len(tokens) >= 8 else None


class TestPRMBeamConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PRMBeamConfig()
        self.assertEqual(cfg.beam_width, 8)
        self.assertEqual(cfg.max_steps, 32)

    def test_invalid_beam_width(self):
        with self.assertRaises((ValueError, Exception)):
            PRMBeamConfig(beam_width=0)

    def test_invalid_weight_sum(self):
        with self.assertRaises((ValueError, Exception)):
            PRMBeamConfig(prm_weight=0.5, token_prob_weight=0.6)

    def test_valid_weight_sum(self):
        cfg = PRMBeamConfig(prm_weight=0.6, token_prob_weight=0.4)
        self.assertAlmostEqual(cfg.prm_weight + cfg.token_prob_weight, 1.0)


class TestPRMBeamCandidate(unittest.TestCase):
    def test_mean_prm_empty(self):
        c = PRMBeamCandidate()
        self.assertEqual(c.mean_prm_score, 0.0)

    def test_mean_prm_with_scores(self):
        c = PRMBeamCandidate(prm_scores=[0.4, 0.8])
        self.assertAlmostEqual(c.mean_prm_score, 0.6)

    def test_combined_score_positive(self):
        c = PRMBeamCandidate(prm_scores=[0.5, 0.7], log_prob=-1.0)
        score = c.combined_score(0.7, 0.3)
        self.assertIsInstance(score, float)


class TestPRMBeamSearch(unittest.TestCase):
    def setUp(self):
        self.cfg = PRMBeamConfig(beam_width=2, max_steps=4, seed=0)
        self.search = PRMBeamSearch(self.cfg)

    def test_search_returns_result(self):
        result = self.search.search([1, 2], _prm_scorer, _expand_fn, _extract_answer)
        self.assertIsInstance(result, PRMBeamResult)

    def test_best_answer_set(self):
        result = self.search.search([1, 2, 3, 4], _prm_scorer, _expand_fn, _extract_answer)
        self.assertIsNotNone(result.best_answer)

    def test_all_candidates_sorted(self):
        result = self.search.search([1, 2], _prm_scorer, _expand_fn, _extract_answer)
        scores = self.search._score_candidates(result.all_candidates)
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_n_steps_positive(self):
        result = self.search.search([1], _prm_scorer, _expand_fn, _extract_answer)
        self.assertGreater(result.n_steps_taken, 0)

    def test_beam_width_respected(self):
        result = self.search.search([1, 2], _prm_scorer, _expand_fn, _extract_answer)
        self.assertLessEqual(len(result.all_candidates), self.cfg.beam_width)


if __name__ == "__main__":
    unittest.main()
