"""Tests for Wave 51b modules: BestOfNSampler, SelfConsistencyVoter,
ThoughtBudgetGate, ReasoningKVManager, DraftReasoningVerifier,
ParallelReasoningScheduler.
"""

import unittest
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# BestOfNSampler tests
# ---------------------------------------------------------------------------
from squish.sampling.best_of_n import (
    BestOfNConfig,
    BestOfNResult,
    BestOfNSampler,
)


class TestBestOfNConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = BestOfNConfig()
        self.assertEqual(cfg.n, 8)
        self.assertAlmostEqual(cfg.temperature, 0.8)

    def test_invalid_n(self):
        with self.assertRaises((ValueError, Exception)):
            BestOfNConfig(n=0)

    def test_invalid_temperature(self):
        with self.assertRaises((ValueError, Exception)):
            BestOfNConfig(temperature=0.0)

    def test_invalid_aggregation(self):
        with self.assertRaises((ValueError, Exception)):
            BestOfNConfig(reward_aggregation="unknown")

    def test_mean_aggregation(self):
        cfg = BestOfNConfig(reward_aggregation="mean")
        self.assertEqual(cfg.reward_aggregation, "mean")


class TestBestOfNResult(unittest.TestCase):
    def test_best_score(self):
        r = BestOfNResult(best_completion="A", all_scores=[0.3, 0.9, 0.5], best_index=1)
        self.assertAlmostEqual(r.best_score, 0.9)

    def test_mean_score(self):
        r = BestOfNResult(best_completion="B", all_scores=[0.2, 0.4, 0.6], best_index=2)
        self.assertAlmostEqual(r.mean_score, 0.4)

    def test_empty_scores(self):
        r = BestOfNResult(best_completion="", all_scores=[], best_index=0)
        self.assertEqual(r.mean_score, 0.0)


class TestBestOfNSampler(unittest.TestCase):
    def setUp(self):
        self.cfg = BestOfNConfig(n=4, seed=99)
        self.sampler = BestOfNSampler(self.cfg)

    def test_sample_max_picks_highest(self):
        completions = ["A", "B", "C"]
        scores = {"A": 0.1, "B": 0.9, "C": 0.5}
        result = self.sampler.sample(completions, lambda c: scores[c])
        self.assertEqual(result.best_completion, "B")

    def test_sample_scores_length(self):
        completions = ["X", "Y", "Z"]
        result = self.sampler.sample(completions, lambda c: len(c) * 0.1)
        self.assertEqual(len(result.all_scores), 3)

    def test_sample_empty_raises(self):
        with self.assertRaises((ValueError, Exception)):
            self.sampler.sample([], lambda c: 0.0)

    def test_simulate_returns_result(self):
        result = self.sampler.simulate(n=4)
        self.assertIsInstance(result, BestOfNResult)

    def test_simulate_scores_clipped(self):
        result = self.sampler.simulate(n=8)
        for s in result.all_scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)

    def test_simulate_majority_mode(self):
        cfg = BestOfNConfig(n=8, reward_aggregation="mean", seed=7)
        sampler = BestOfNSampler(cfg)
        result = sampler.simulate(n=8)
        self.assertIsInstance(result.best_completion, str)

    def test_best_index_valid_range(self):
        completions = ["a", "b", "c", "d"]
        result = self.sampler.sample(completions, lambda c: ord(c[0]) / 100)
        self.assertGreaterEqual(result.best_index, 0)
        self.assertLess(result.best_index, len(completions))


# ---------------------------------------------------------------------------
# SelfConsistencyVoter tests
# ---------------------------------------------------------------------------
from squish.reasoning.self_consistency import (
    SelfConsistencyConfig,
    SelfConsistencyResult,
    SelfConsistencyVoter,
)


class TestSelfConsistencyConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SelfConsistencyConfig()
        self.assertEqual(cfg.k, 8)
        self.assertAlmostEqual(cfg.temperature, 0.9)

    def test_invalid_k(self):
        with self.assertRaises((ValueError, Exception)):
            SelfConsistencyConfig(k=0)

    def test_invalid_temperature(self):
        with self.assertRaises((ValueError, Exception)):
            SelfConsistencyConfig(temperature=0.0)

    def test_with_pattern(self):
        cfg = SelfConsistencyConfig(answer_pattern=r"Answer: (.+)")
        self.assertIsNotNone(cfg.answer_pattern)


class TestSelfConsistencyVoter(unittest.TestCase):
    def setUp(self):
        self.voter = SelfConsistencyVoter(SelfConsistencyConfig())

    def test_vote_majority(self):
        chains = ["The answer is 4\n4", "4 is correct\n4", "I think 5\n5"]
        result = self.voter.vote(chains)
        self.assertEqual(result.winner, "4")

    def test_vote_counts_sum(self):
        chains = ["a\na", "b\nb", "a\na"]
        result = self.voter.vote(chains)
        self.assertEqual(sum(result.vote_counts.values()), 3)

    def test_vote_empty_raises(self):
        with self.assertRaises((ValueError, Exception)):
            self.voter.vote([])

    def test_extract_answer_last_line(self):
        chain = "Step 1: ...\nStep 2: ...\n42"
        ans = self.voter.extract_answer(chain)
        self.assertEqual(ans, "42")

    def test_extract_answer_with_pattern(self):
        voter = SelfConsistencyVoter(
            SelfConsistencyConfig(answer_pattern=r"Answer: (\d+)")
        )
        chain = "Thinking...\nAnswer: 7"
        ans = voter.extract_answer(chain)
        self.assertEqual(ans, "7")

    def test_winner_vote_share_range(self):
        chains = ["ans\n1", "ans\n1", "ans\n2"]
        result = self.voter.vote(chains)
        self.assertGreaterEqual(result.winner_vote_share, 0.0)
        self.assertLessEqual(result.winner_vote_share, 1.0)

    def test_n_chains(self):
        chains = ["a\na", "b\nb", "a\na", "c\nc"]
        result = self.voter.vote(chains)
        self.assertEqual(result.n_chains, 4)

    def test_majority_vote_tie_broken_alphabetically(self):
        counts = {"b": 2, "a": 2, "c": 1}
        winner = self.voter.majority_vote(counts)
        self.assertEqual(winner, "b")  # max by (count, key) → ("b", 2) > ("a", 2)

    def test_majority_vote_empty_raises(self):
        with self.assertRaises((ValueError, Exception)):
            self.voter.majority_vote({})


# ---------------------------------------------------------------------------
# ThoughtBudgetGate tests
# ---------------------------------------------------------------------------
from squish.token.thought_budget_gate import (
    ThoughtBudgetConfig,
    ThoughtBudgetState,
    ThoughtBudgetGate,
)


class TestThoughtBudgetConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ThoughtBudgetConfig()
        self.assertGreater(cfg.max_thinking_tokens, 0)

    def test_invalid_max_thinking_tokens(self):
        with self.assertRaises((ValueError, Exception)):
            ThoughtBudgetConfig(max_thinking_tokens=0)

    def test_invalid_soft_budget_fraction(self):
        with self.assertRaises((ValueError, Exception)):
            ThoughtBudgetConfig(soft_budget_fraction=0.0)

    def test_boundary_tokens_default(self):
        cfg = ThoughtBudgetConfig()
        self.assertIsInstance(cfg.boundary_tokens, list)
        self.assertGreater(len(cfg.boundary_tokens), 0)


class TestThoughtBudgetGate(unittest.TestCase):
    def setUp(self):
        self.cfg = ThoughtBudgetConfig(max_thinking_tokens=5)
        self.gate = ThoughtBudgetGate(self.cfg)

    def test_new_state_in_thinking(self):
        state = self.gate.new_state()
        self.assertTrue(state.in_thinking)

    def test_step_increments_total(self):
        state = self.gate.new_state()
        self.gate.step("tok", state)
        self.assertEqual(state.total_tokens, 1)

    def test_boundary_token_transitions_segment(self):
        state = self.gate.new_state()
        for tok in ["a", "b", "</think>"]:
            self.gate.step(tok, state)
        self.assertTrue(state.in_answer)

    def test_force_commit_on_budget_exhaustion(self):
        state = self.gate.new_state()
        inject_found = False
        for _ in range(6):
            _, inject = self.gate.step("tok", state)
            if inject:
                inject_found = True
        self.assertTrue(inject_found)

    def test_budget_fraction_zero_start(self):
        state = self.gate.new_state()
        self.assertAlmostEqual(self.gate.budget_fraction(state), 0.0)

    def test_budget_fraction_one_in_answer(self):
        state = self.gate.new_state()
        self.gate.step("</think>", state)
        self.assertAlmostEqual(self.gate.budget_fraction(state), 1.0)

    def test_near_soft_budget(self):
        cfg = ThoughtBudgetConfig(max_thinking_tokens=10, soft_budget_fraction=0.8)
        gate = ThoughtBudgetGate(cfg)
        state = gate.new_state()
        for _ in range(9):
            gate.step("tok", state)
        self.assertTrue(gate.near_soft_budget(state))

    def test_reset_restores_thinking_segment(self):
        state = self.gate.new_state()
        self.gate.step("</think>", state)
        self.gate.reset(state)
        self.assertTrue(state.in_thinking)

    def test_segment_of(self):
        state = self.gate.new_state()
        self.assertEqual(self.gate.segment_of(state), "thinking")


# ---------------------------------------------------------------------------
# ReasoningKVManager tests
# ---------------------------------------------------------------------------
from squish.kv.reasoning_kv import (
    ReasoningKVConfig,
    ReasoningKVSegment,
    ReasoningKVState,
    ReasoningKVManager,
)


class TestReasoningKVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ReasoningKVConfig()
        self.assertEqual(cfg.thinking_bits, 2)
        self.assertEqual(cfg.answer_bits, 16)

    def test_invalid_thinking_bits(self):
        with self.assertRaises((ValueError, Exception)):
            ReasoningKVConfig(thinking_bits=3)

    def test_invalid_answer_bits(self):
        with self.assertRaises((ValueError, Exception)):
            ReasoningKVConfig(answer_bits=7)

    def test_invalid_group_size(self):
        with self.assertRaises((ValueError, Exception)):
            ReasoningKVConfig(group_size=0)


class TestReasoningKVSegment(unittest.TestCase):
    def test_values(self):
        self.assertEqual(ReasoningKVSegment.THINKING.value, "thinking")
        self.assertEqual(ReasoningKVSegment.ANSWER.value, "answer")


class TestReasoningKVState(unittest.TestCase):
    def test_initial_segment_thinking(self):
        state = ReasoningKVState()
        self.assertEqual(state.segment, ReasoningKVSegment.THINKING)

    def test_total_tokens_zero(self):
        state = ReasoningKVState()
        self.assertEqual(state.total_tokens, 0)

    def test_compression_ratio_zero(self):
        state = ReasoningKVState()
        self.assertEqual(state.compression_ratio, 0.0)


class TestReasoningKVManager(unittest.TestCase):
    def setUp(self):
        self.cfg = ReasoningKVConfig(thinking_bits=2, group_size=4)
        self.mgr = ReasoningKVManager(self.cfg)

    def test_new_state(self):
        state = self.mgr.new_state()
        self.assertIsInstance(state, ReasoningKVState)

    def test_update_thinking_tokens(self):
        state = self.mgr.new_state()
        k = np.random.randn(8).astype(np.float32)
        v = np.random.randn(8).astype(np.float32)
        self.mgr.update(k, v, "tok", state)
        self.assertEqual(state.n_thinking_tokens, 1)

    def test_update_transitions_on_boundary(self):
        state = self.mgr.new_state()
        k = np.random.randn(8).astype(np.float32)
        v = np.random.randn(8).astype(np.float32)
        self.mgr.update(k, v, "</think>", state)
        self.assertEqual(state.segment, ReasoningKVSegment.ANSWER)

    def test_update_answer_tokens(self):
        state = self.mgr.new_state()
        k = np.random.randn(8).astype(np.float32)
        v = np.random.randn(8).astype(np.float32)
        self.mgr.update(k, v, "</think>", state)
        self.mgr.update(k, v, "answer_tok", state)
        self.assertEqual(state.n_answer_tokens, 1)

    def test_get_kv_shape(self):
        state = self.mgr.new_state()
        for tok in ["a", "b", "</think>", "c"]:
            k = np.random.randn(8).astype(np.float32)
            v = np.random.randn(8).astype(np.float32)
            self.mgr.update(k, v, tok, state)
        k_out, v_out = self.mgr.get_kv(state)
        self.assertEqual(k_out.shape[0], state.total_tokens)

    def test_get_kv_empty_state(self):
        state = self.mgr.new_state()
        k_out, v_out = self.mgr.get_kv(state)
        self.assertEqual(k_out.size, 0)

    def test_memory_summary_keys(self):
        state = self.mgr.new_state()
        summary = self.mgr.memory_summary(state)
        self.assertIn("segment", summary)
        self.assertIn("compression_ratio", summary)

    def test_boundary_position_set(self):
        state = self.mgr.new_state()
        k = np.ones(8, dtype=np.float32)
        v = np.ones(8, dtype=np.float32)
        self.mgr.update(k, v, "tok", state)
        self.mgr.update(k, v, "</think>", state)
        self.assertIsNotNone(state.boundary_position)


# ---------------------------------------------------------------------------
# DraftReasoningVerifier tests
# ---------------------------------------------------------------------------
from squish.speculative.draft_reasoning import (
    DraftReasoningConfig,
    DraftReasoningState,
    DraftReasoningVerifier,
)


class TestDraftReasoningConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DraftReasoningConfig()
        self.assertGreater(cfg.token_prob_threshold, 0.0)
        self.assertGreater(cfg.cosine_threshold, 0.0)

    def test_invalid_token_prob_threshold(self):
        with self.assertRaises((ValueError, Exception)):
            DraftReasoningConfig(token_prob_threshold=0.0)

    def test_invalid_cosine_threshold(self):
        with self.assertRaises((ValueError, Exception)):
            DraftReasoningConfig(cosine_threshold=1.1)

    def test_invalid_context_window(self):
        with self.assertRaises((ValueError, Exception)):
            DraftReasoningConfig(context_window=0)


class TestDraftReasoningVerifier(unittest.TestCase):
    def setUp(self):
        self.cfg = DraftReasoningConfig(
            token_prob_threshold=0.5,
            cosine_threshold=0.7,
            context_window=4,
        )
        self.verifier = DraftReasoningVerifier(self.cfg)

    def test_new_state_zero_counts(self):
        state = self.verifier.new_state()
        self.assertEqual(state.n_accepted, 0)
        self.assertEqual(state.n_rejected, 0)

    def test_high_prob_high_cosine_accepted(self):
        state = self.verifier.new_state()
        h = np.ones(8, dtype=np.float32)
        context = [np.ones(8, dtype=np.float32)] * 3
        accepted = self.verifier.verify(0.9, h, context, state)
        self.assertTrue(accepted)

    def test_low_prob_rejected(self):
        state = self.verifier.new_state()
        h = np.ones(8)
        context = [np.ones(8)] * 3
        accepted = self.verifier.verify(0.1, h, context, state)
        self.assertFalse(accepted)

    def test_low_cosine_rejected(self):
        state = self.verifier.new_state()
        h = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        context = [np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)] * 3
        accepted = self.verifier.verify(0.9, h, context, state)
        self.assertFalse(accepted)

    def test_acceptance_rate_starts_zero(self):
        state = self.verifier.new_state()
        self.assertEqual(self.verifier.acceptance_rate(state), 0.0)

    def test_acceptance_rate_after_decisions(self):
        state = self.verifier.new_state()
        h = np.ones(4)
        ctx = [np.ones(4)] * 2
        self.verifier.verify(0.9, h, ctx, state)
        self.verifier.verify(0.1, h, ctx, state)
        rate = self.verifier.acceptance_rate(state)
        self.assertAlmostEqual(rate, 0.5)

    def test_reset_clears_state(self):
        state = self.verifier.new_state()
        h = np.ones(4)
        ctx = [np.ones(4)]
        self.verifier.verify(0.9, h, ctx, state)
        self.verifier.reset(state)
        self.assertEqual(state.n_accepted, 0)
        self.assertEqual(len(state.acceptance_history), 0)

    def test_no_context_relies_on_prob(self):
        state = self.verifier.new_state()
        h = np.zeros(8)
        accepted = self.verifier.verify(0.9, h, [], state)
        self.assertTrue(accepted)  # no context → cosine_ok=True; prob_ok→True

    def test_cosine_sim_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        self.assertAlmostEqual(self.verifier._cosine_sim(v, v), 1.0, places=5)

    def test_cosine_sim_zero_vector(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.ones(3, dtype=np.float32)
        self.assertEqual(self.verifier._cosine_sim(a, b), 0.0)

    def test_calibrate_threshold(self):
        samples = [
            (0.8, np.ones(4), [np.ones(4)], True),
            (0.2, np.ones(4), [np.ones(4)], False),
        ] * 4
        thresh = self.verifier.calibrate_threshold(samples, target_rate=0.5)
        self.assertIsInstance(thresh, float)
        self.assertGreater(thresh, 0.0)


# ---------------------------------------------------------------------------
# ParallelReasoningScheduler tests
# ---------------------------------------------------------------------------
from squish.serving.parallel_reasoning import (
    ParallelReasoningConfig,
    ParallelReasoningRequest,
    ParallelReasoningResult,
    ParallelReasoningScheduler,
)


class TestParallelReasoningConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ParallelReasoningConfig()
        self.assertEqual(cfg.min_chains, 1)
        self.assertGreater(cfg.max_chains, 0)

    def test_invalid_max_chains(self):
        with self.assertRaises((ValueError, Exception)):
            ParallelReasoningConfig(max_chains=0)

    def test_min_gt_max_raises(self):
        with self.assertRaises((ValueError, Exception)):
            ParallelReasoningConfig(min_chains=5, max_chains=3)

    def test_invalid_aggregation(self):
        with self.assertRaises((ValueError, Exception)):
            ParallelReasoningConfig(aggregation="unknown")

    def test_hard_le_easy_raises(self):
        with self.assertRaises((ValueError, Exception)):
            ParallelReasoningConfig(easy_threshold=5.0, hard_threshold=2.0)


class TestParallelReasoningRequest(unittest.TestCase):
    def test_auto_request_id(self):
        req = ParallelReasoningRequest(prompt="test")
        self.assertTrue(len(req.request_id) > 0)

    def test_custom_n_chains(self):
        req = ParallelReasoningRequest(prompt="x", n_chains=4)
        self.assertEqual(req.n_chains, 4)


class TestParallelReasoningScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = ParallelReasoningConfig(
            min_chains=1, max_chains=4,
            easy_threshold=1.0, hard_threshold=3.0,
            seed=0,
        )
        self.sched = ParallelReasoningScheduler(self.cfg)

    def _gen_fn(self, prompt, n):
        return [f"chain_{i}: {prompt}\n{prompt.split()[-1] if prompt.split() else 'y'}" for i in range(n)]

    def test_dispatch_easy(self):
        n = self.sched.dispatch(0.0)
        self.assertEqual(n, self.cfg.min_chains)

    def test_dispatch_hard(self):
        n = self.sched.dispatch(10.0)
        self.assertEqual(n, self.cfg.max_chains)

    def test_dispatch_mid(self):
        n = self.sched.dispatch(2.0)
        self.assertGreaterEqual(n, self.cfg.min_chains)
        self.assertLessEqual(n, self.cfg.max_chains)

    def test_aggregate_self_consistency(self):
        chains = ["Think...\n42", "Think...\n42", "Think...\n43"]
        winner, counts = self.sched.aggregate(chains, method="self_consistency")
        self.assertEqual(winner, "42")

    def test_aggregate_empty_raises(self):
        with self.assertRaises((ValueError, Exception)):
            self.sched.aggregate([])

    def test_schedule_returns_result(self):
        req = ParallelReasoningRequest(prompt="2+2", n_chains=3)
        result = self.sched.schedule(req, generate_fn=self._gen_fn)
        self.assertIsInstance(result, ParallelReasoningResult)

    def test_schedule_n_chains(self):
        req = ParallelReasoningRequest(prompt="easy problem", n_chains=2)
        result = self.sched.schedule(req, generate_fn=self._gen_fn)
        self.assertEqual(result.n_chains, 2)

    def test_schedule_wall_seconds_nonnegative(self):
        req = ParallelReasoningRequest(prompt="test")
        result = self.sched.schedule(req, generate_fn=self._gen_fn, difficulty_score=1.5)
        self.assertGreaterEqual(result.wall_seconds, 0.0)

    def test_schedule_request_id_preserved(self):
        req = ParallelReasoningRequest(prompt="x", request_id="abc-123")
        result = self.sched.schedule(req, generate_fn=self._gen_fn)
        self.assertEqual(result.request_id, "abc-123")

    def test_vote_counts_sum_to_n_chains(self):
        req = ParallelReasoningRequest(prompt="What is 2+2?", n_chains=4)
        result = self.sched.schedule(req, generate_fn=self._gen_fn)
        self.assertEqual(sum(result.vote_counts.values()), result.n_chains)


if __name__ == "__main__":
    unittest.main()
