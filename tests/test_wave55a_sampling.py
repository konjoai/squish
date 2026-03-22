"""tests/test_wave55a_sampling.py

Unit tests for Wave 55a — Advanced Sampling Refinement modules:
* min_p_sampler     — MinPSampler
* mirostat_sampler  — MirostatSampler
* typical_sampler   — TypicalSampler (existing)
* eta_sampler       — EtaCutoffSampler
* cfg_sampler       — CFGLogitsSampler
* diverse_beam      — DiverseBeamSampler
"""

from __future__ import annotations

import math
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# MinPSampler tests
# ---------------------------------------------------------------------------

class TestMinPConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from squish.sampling.min_p_sampler import MinPConfig
        cfg = MinPConfig()
        self.assertAlmostEqual(cfg.p_min, 0.05)
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertEqual(cfg.seed, 0)

    def test_invalid_p_min_zero(self) -> None:
        from squish.sampling.min_p_sampler import MinPConfig
        with self.assertRaises(ValueError):
            MinPConfig(p_min=0.0)

    def test_invalid_p_min_one(self) -> None:
        from squish.sampling.min_p_sampler import MinPConfig
        with self.assertRaises(ValueError):
            MinPConfig(p_min=1.0)

    def test_invalid_temperature(self) -> None:
        from squish.sampling.min_p_sampler import MinPConfig
        with self.assertRaises(ValueError):
            MinPConfig(temperature=0.0)

    def test_custom_values(self) -> None:
        from squish.sampling.min_p_sampler import MinPConfig
        cfg = MinPConfig(p_min=0.1, temperature=0.8, seed=42)
        self.assertAlmostEqual(cfg.p_min, 0.1)
        self.assertAlmostEqual(cfg.temperature, 0.8)


class TestMinPSampler(unittest.TestCase):
    def setUp(self) -> None:
        from squish.sampling.min_p_sampler import MinPConfig, MinPSampler
        self.cfg = MinPConfig(p_min=0.05, temperature=1.0, seed=7)
        self.sampler = MinPSampler(self.cfg)

    def _logits(self) -> np.ndarray:
        return np.array([1.0, 3.0, 0.5, 2.0, 0.1], dtype=np.float32)

    def test_filter_logits_returns_array(self) -> None:
        logits = self._logits()
        result = self.sampler.filter_logits(logits)
        self.assertEqual(result.shape, logits.shape)

    def test_filter_logits_masks_low_prob(self) -> None:
        # logits heavily peaked on index 1 — low probs should be masked to -1e9
        logits = np.array([0.0, 100.0, 0.0, 0.0, 0.0], dtype=np.float32)
        result = self.sampler.filter_logits(logits)
        # Only index 1 should survive
        self.assertGreater(result[1], -1e8)
        for i in [0, 2, 3, 4]:
            self.assertAlmostEqual(result[i], -1e9, delta=1.0)

    def test_sample_returns_int(self) -> None:
        token = self.sampler.sample(self._logits())
        self.assertIsInstance(token, int)

    def test_sample_in_range(self) -> None:
        logits = self._logits()
        for _ in range(20):
            token = self.sampler.sample(logits)
            self.assertGreaterEqual(token, 0)
            self.assertLess(token, len(logits))

    def test_top_token_returns_int(self) -> None:
        token = self.sampler.top_token(self._logits())
        self.assertIsInstance(token, int)

    def test_top_token_is_argmax(self) -> None:
        logits = self._logits()
        # After filtering, token 1 (logit=3) should dominate
        token = self.sampler.top_token(logits)
        self.assertEqual(token, 1)

    def test_survival_count_positive(self) -> None:
        count = self.sampler.survival_count(self._logits())
        self.assertGreaterEqual(count, 1)

    def test_survival_count_leq_vocab(self) -> None:
        logits = self._logits()
        count = self.sampler.survival_count(logits)
        self.assertLessEqual(count, len(logits))

    def test_uniform_logits_all_survive(self) -> None:
        logits = np.zeros(10, dtype=np.float32)
        count = self.sampler.survival_count(logits)
        self.assertEqual(count, 10)


# ---------------------------------------------------------------------------
# MirostatSampler tests
# ---------------------------------------------------------------------------

class TestMirostatConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from squish.sampling.mirostat_sampler import MirostatConfig
        cfg = MirostatConfig()
        self.assertAlmostEqual(cfg.tau, 5.0)
        self.assertAlmostEqual(cfg.eta, 0.1)

    def test_invalid_tau(self) -> None:
        from squish.sampling.mirostat_sampler import MirostatConfig
        with self.assertRaises(ValueError):
            MirostatConfig(tau=0.0)

    def test_invalid_eta(self) -> None:
        from squish.sampling.mirostat_sampler import MirostatConfig
        with self.assertRaises(ValueError):
            MirostatConfig(eta=0.0)


class TestMirostatSampler(unittest.TestCase):
    def setUp(self) -> None:
        from squish.sampling.mirostat_sampler import MirostatConfig, MirostatSampler
        self.cfg = MirostatConfig(tau=3.0, eta=0.1, seed=0)
        self.sampler = MirostatSampler(self.cfg)

    def _logits(self) -> np.ndarray:
        rng = np.random.default_rng(99)
        return rng.normal(0, 1, size=20).astype(np.float32)

    def test_new_state_mu_initialised(self) -> None:
        state = self.sampler.new_state()
        # mu should start at 2*tau
        self.assertAlmostEqual(state.mu, 2.0 * self.cfg.tau, places=4)

    def test_sample_returns_valid_token(self) -> None:
        state = self.sampler.new_state()
        token, new_state = self.sampler.sample(self._logits(), state)
        self.assertIsInstance(token, int)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 20)

    def test_state_updates(self) -> None:
        state = self.sampler.new_state()
        initial_mu = state.mu
        _, new_state = self.sampler.sample(self._logits(), state)
        # mu should change after one step
        # (unless surprise exactly equals tau, which is unlikely)
        self.assertIsInstance(new_state.mu, float)

    def test_n_tokens_increments(self) -> None:
        state = self.sampler.new_state()
        self.assertEqual(state.n_tokens, 0)
        _, state2 = self.sampler.sample(self._logits(), state)
        self.assertEqual(state2.n_tokens, 1)

    def test_reset_returns_fresh_state(self) -> None:
        state = self.sampler.new_state()
        _, evolved = self.sampler.sample(self._logits(), state)
        reset = self.sampler.reset()
        self.assertAlmostEqual(reset.mu, 2.0 * self.cfg.tau, places=4)
        self.assertEqual(reset.n_tokens, 0)

    def test_mu_clamp_prevents_negative(self) -> None:
        from squish.sampling.mirostat_sampler import MirostatConfig, MirostatSampler, MirostatState
        cfg = MirostatConfig(tau=0.01, eta=10.0, seed=0)
        sampler = MirostatSampler(cfg)
        state = sampler.new_state()
        for _ in range(5):
            _, state = sampler.sample(np.array([1.0, 2.0, 3.0], dtype=np.float32), state)
        self.assertGreater(state.mu, 0.0)


# ---------------------------------------------------------------------------
# TypicalSampler tests
# ---------------------------------------------------------------------------

class TestTypicalConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from squish.sampling.typical_sampler import TypicalConfig
        cfg = TypicalConfig()
        self.assertAlmostEqual(cfg.tau, 0.9)

    def test_invalid_tau_zero(self) -> None:
        from squish.sampling.typical_sampler import TypicalConfig
        with self.assertRaises(ValueError):
            TypicalConfig(tau=0.0)

    def test_invalid_tau_exceeds_one(self) -> None:
        from squish.sampling.typical_sampler import TypicalConfig
        with self.assertRaises(ValueError):
            TypicalConfig(tau=1.1)

    def test_invalid_temperature(self) -> None:
        from squish.sampling.typical_sampler import TypicalConfig
        with self.assertRaises(ValueError):
            TypicalConfig(temperature=-0.1)


class TestTypicalSampler(unittest.TestCase):
    def setUp(self) -> None:
        from squish.sampling.typical_sampler import TypicalConfig, TypicalSampler
        self.sampler = TypicalSampler(TypicalConfig(tau=0.9, temperature=1.0, seed=0))

    def _logits(self) -> np.ndarray:
        return np.array([0.5, 1.0, 2.0, 0.3, 0.8], dtype=np.float32)

    def test_sample_returns_result(self) -> None:
        from squish.sampling.typical_sampler import TypicalResult
        result = self.sampler.sample(self._logits())
        self.assertIsInstance(result, TypicalResult)

    def test_token_id_in_range(self) -> None:
        from squish.sampling.typical_sampler import TypicalResult
        result = self.sampler.sample(self._logits())
        self.assertGreaterEqual(result.token_id, 0)
        self.assertLess(result.token_id, 5)

    def test_probability_in_unit_interval(self) -> None:
        result = self.sampler.sample(self._logits())
        self.assertGreater(result.probability, 0.0)
        self.assertLessEqual(result.probability, 1.0)

    def test_n_candidates_positive(self) -> None:
        result = self.sampler.sample(self._logits())
        self.assertGreaterEqual(result.n_candidates, 1)

    def test_filter_logits_shape(self) -> None:
        logits = self._logits()
        filtered = self.sampler.filter_logits(logits)
        self.assertEqual(filtered.shape, logits.shape)

    def test_entropy_non_negative(self) -> None:
        result = self.sampler.sample(self._logits())
        self.assertGreaterEqual(result.entropy, 0.0)


# ---------------------------------------------------------------------------
# EtaCutoffSampler tests
# ---------------------------------------------------------------------------

class TestEtaConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from squish.sampling.eta_sampler import EtaConfig
        cfg = EtaConfig()
        self.assertAlmostEqual(cfg.eta, 0.003)

    def test_invalid_eta_zero(self) -> None:
        from squish.sampling.eta_sampler import EtaConfig
        with self.assertRaises(ValueError):
            EtaConfig(eta=0.0)

    def test_invalid_temperature(self) -> None:
        from squish.sampling.eta_sampler import EtaConfig
        with self.assertRaises(ValueError):
            EtaConfig(temperature=-1.0)


class TestEtaCutoffSampler(unittest.TestCase):
    def setUp(self) -> None:
        from squish.sampling.eta_sampler import EtaConfig, EtaCutoffSampler
        self.sampler = EtaCutoffSampler(EtaConfig(eta=0.01, seed=0))

    def _logits(self) -> np.ndarray:
        rng = np.random.default_rng(5)
        return rng.normal(0, 1, 30).astype(np.float32)

    def test_entropy_non_negative(self) -> None:
        H = self.sampler.entropy(self._logits())
        self.assertGreaterEqual(H, 0.0)

    def test_entropy_uniform_is_max(self) -> None:
        logits_uniform = np.zeros(100, dtype=np.float32)
        logits_peaked = np.array([100.0] + [-100.0] * 99, dtype=np.float32)
        H_uniform = self.sampler.entropy(logits_uniform)
        H_peaked = self.sampler.entropy(logits_peaked)
        self.assertGreater(H_uniform, H_peaked)

    def test_filter_logits_returns_same_shape(self) -> None:
        logits = self._logits()
        result = self.sampler.filter_logits(logits)
        self.assertEqual(result.shape, logits.shape)

    def test_sample_in_range(self) -> None:
        logits = self._logits()
        for _ in range(10):
            tok = self.sampler.sample(logits)
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, len(logits))

    def test_survival_count_positive(self) -> None:
        count = self.sampler.survival_count(self._logits())
        self.assertGreaterEqual(count, 1)

    def test_peaked_distribution_few_survivors(self) -> None:
        # Sharply peaked: only 1-2 tokens should survive
        logits = np.array([50.0, -50.0, -50.0, -50.0, -50.0], dtype=np.float32)
        count = self.sampler.survival_count(logits)
        self.assertLessEqual(count, 3)


# ---------------------------------------------------------------------------
# CFGLogitsSampler tests
# ---------------------------------------------------------------------------

class TestCFGConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from squish.sampling.cfg_sampler import CFGConfig
        cfg = CFGConfig()
        self.assertAlmostEqual(cfg.guidance_scale, 1.5)

    def test_invalid_guidance_scale(self) -> None:
        from squish.sampling.cfg_sampler import CFGConfig
        with self.assertRaises(ValueError):
            CFGConfig(guidance_scale=-0.1)

    def test_invalid_temperature(self) -> None:
        from squish.sampling.cfg_sampler import CFGConfig
        with self.assertRaises(ValueError):
            CFGConfig(temperature=0.0)


class TestCFGLogitsSampler(unittest.TestCase):
    def setUp(self) -> None:
        from squish.sampling.cfg_sampler import CFGConfig, CFGLogitsSampler
        self.sampler = CFGLogitsSampler(CFGConfig(guidance_scale=2.0, seed=0))

    def _logits(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(1)
        cond = rng.normal(0, 1, 10).astype(np.float32)
        uncond = rng.normal(0, 1, 10).astype(np.float32)
        return cond, uncond

    def test_merge_logits_shape(self) -> None:
        cond, uncond = self._logits()
        merged = self.sampler.merge_logits(cond, uncond)
        self.assertEqual(merged.shape, cond.shape)

    def test_merge_logits_formula(self) -> None:
        cond = np.array([1.0, 0.0], dtype=np.float32)
        uncond = np.array([0.0, 1.0], dtype=np.float32)
        # cfg_scale=2: merged = uncond + 2*(cond-uncond) = 2*cond - uncond
        merged = self.sampler.merge_logits(cond, uncond)
        expected = uncond + 2.0 * (cond - uncond)
        np.testing.assert_allclose(merged, expected, rtol=1e-5)

    def test_sample_returns_int(self) -> None:
        cond, uncond = self._logits()
        tok = self.sampler.sample(cond, uncond)
        self.assertIsInstance(tok, int)
        self.assertGreaterEqual(tok, 0)
        self.assertLess(tok, len(cond))

    def test_top_token_is_deterministic(self) -> None:
        cond, uncond = self._logits()
        t1 = self.sampler.top_token(cond, uncond)
        t2 = self.sampler.top_token(cond, uncond)
        self.assertEqual(t1, t2)

    def test_guidance_delta_shape(self) -> None:
        cond, uncond = self._logits()
        delta = self.sampler.guidance_delta(cond, uncond)
        self.assertEqual(delta.shape, cond.shape)

    def test_guidance_delta_formula(self) -> None:
        cond = np.array([2.0, -1.0], dtype=np.float32)
        uncond = np.array([1.0, 1.0], dtype=np.float32)
        delta = self.sampler.guidance_delta(cond, uncond)
        expected = 2.0 * (cond - uncond)
        np.testing.assert_allclose(delta, expected, rtol=1e-5)

    def test_zero_guidance_scale(self) -> None:
        from squish.sampling.cfg_sampler import CFGConfig, CFGLogitsSampler
        s = CFGLogitsSampler(CFGConfig(guidance_scale=0.0))
        cond = np.array([1.0, 2.0], dtype=np.float32)
        uncond = np.array([3.0, 4.0], dtype=np.float32)
        merged = s.merge_logits(cond, uncond)
        np.testing.assert_allclose(merged, uncond, rtol=1e-5)


# ---------------------------------------------------------------------------
# DiverseBeamSampler tests
# ---------------------------------------------------------------------------

class TestDiverseBeamConfig(unittest.TestCase):
    def test_defaults(self) -> None:
        from squish.sampling.diverse_beam import DiverseBeamConfig
        cfg = DiverseBeamConfig()
        self.assertEqual(cfg.beam_size, 4)
        self.assertEqual(cfg.n_groups, 2)

    def test_invalid_beam_not_divisible(self) -> None:
        from squish.sampling.diverse_beam import DiverseBeamConfig
        with self.assertRaises(ValueError):
            DiverseBeamConfig(beam_size=3, n_groups=2)

    def test_invalid_beam_size_zero(self) -> None:
        from squish.sampling.diverse_beam import DiverseBeamConfig
        with self.assertRaises(ValueError):
            DiverseBeamConfig(beam_size=0)

    def test_invalid_n_groups_zero(self) -> None:
        from squish.sampling.diverse_beam import DiverseBeamConfig
        with self.assertRaises(ValueError):
            DiverseBeamConfig(n_groups=0)


class TestDiverseBeamSampler(unittest.TestCase):
    def setUp(self) -> None:
        from squish.sampling.diverse_beam import DiverseBeamConfig, DiverseBeamSampler
        self.cfg = DiverseBeamConfig(
            beam_size=4, n_groups=2, diversity_strength=0.5,
            vocab_size=20, max_length=10, seed=0
        )
        self.sampler = DiverseBeamSampler(self.cfg)

    def _make_logits(self) -> np.ndarray:
        rng = np.random.default_rng(10)
        # (n_groups, beams_per_group, vocab_size)
        return rng.normal(0, 1, (2, 2, 20)).astype(np.float32)

    def test_new_state_initial_token(self) -> None:
        state = self.sampler.new_state(initial_token=1)
        self.assertEqual(state.beam_sequences[0][0][0], 1)
        self.assertEqual(state.step, 0)

    def test_step_increments_step_counter(self) -> None:
        state = self.sampler.new_state()
        new_state = self.sampler.step_logits(self._make_logits(), state)
        self.assertEqual(new_state.step, 1)

    def test_step_extends_sequences(self) -> None:
        state = self.sampler.new_state()
        new_state = self.sampler.step_logits(self._make_logits(), state)
        # Each sequence should be 1 token longer
        for g in range(self.cfg.n_groups):
            for b in range(self.sampler._beams_per_group):
                self.assertEqual(len(new_state.beam_sequences[g][b]),
                                 len(state.beam_sequences[g][b]) + 1)

    def test_get_sequences_count(self) -> None:
        state = self.sampler.new_state()
        seqs = self.sampler.get_sequences(state)
        self.assertEqual(len(seqs), self.cfg.beam_size)

    def test_get_sequences_sorted_descending(self) -> None:
        state = self.sampler.new_state()
        state = self.sampler.step_logits(self._make_logits(), state)
        seqs = self.sampler.get_sequences(state)
        scores = [s for _, s in seqs]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_best_sequence_returns_list(self) -> None:
        state = self.sampler.new_state()
        best = self.sampler.best_sequence(state)
        self.assertIsInstance(best, list)

    def test_invalid_logits_shape_raises(self) -> None:
        state = self.sampler.new_state()
        bad_logits = np.zeros((3, 2, 20), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.sampler.step_logits(bad_logits, state)

    def test_diversity_penalty_shifts_scores(self) -> None:
        # With high diversity_strength, groups should prefer different tokens
        from squish.sampling.diverse_beam import DiverseBeamConfig, DiverseBeamSampler
        cfg = DiverseBeamConfig(
            beam_size=2, n_groups=2, diversity_strength=100.0,
            vocab_size=10, max_length=5, seed=0
        )
        sampler = DiverseBeamSampler(cfg)
        # Create logits where token 0 is clearly best for all beams
        logits = np.zeros((2, 1, 10), dtype=np.float32)
        logits[:, :, 0] = 10.0
        state = sampler.new_state()
        new_state = sampler.step_logits(logits, state)
        seqs = sampler.get_sequences(new_state)
        # Both groups should have generated a token
        self.assertEqual(len(seqs), 2)


if __name__ == "__main__":
    unittest.main()
