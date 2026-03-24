"""tests/test_wave68_eagle_head.py

Unit tests for Wave 68: SQUIZD Trained EAGLE Draft Head.

Modules under test
──────────────────
* squish.compress.distill_eagle — EAGLEConfig, EAGLELayerWeights,
                                   EAGLEHeadWeights, EAGLEDistiller,
                                   save_eagle_head, load_eagle_head,
                                   _eagle_forward, SQUIZD_EAGLE_TAG
* squish.speculative.eagle_head — EAGLERunnerConfig, DraftToken,
                                   EAGLEHeadRunner, eagle_decode_step,
                                   ROLLING_WINDOW_SIZE
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from squish.compress.distill_eagle import (
    EAGLE_FORMAT_VERSION,
    SQUIZD_EAGLE_TAG,
    EAGLEConfig,
    EAGLEDistiller,
    EAGLEHeadWeights,
    EAGLELayerWeights,
    _eagle_forward,
    load_eagle_head,
    save_eagle_head,
)
from squish.speculative.eagle_head import (
    ROLLING_WINDOW_SIZE,
    DraftToken,
    EAGLEHeadRunner,
    EAGLERunnerConfig,
    _sample_top_k,
    eagle_decode_step,
)


# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

def _make_config(n_model_layers: int = 8, d_model: int = 64) -> EAGLEConfig:
    return EAGLEConfig(n_model_layers=n_model_layers, d_model=d_model)


def _make_weights(
    cfg: EAGLEConfig | None = None,
    vocab_size: int = 32,
    rng_seed: int = 0,
) -> EAGLEHeadWeights:
    if cfg is None:
        cfg = _make_config()
    distiller = EAGLEDistiller(cfg, rng_seed=rng_seed)
    return distiller._init_weights(vocab_size)


def _make_hidden(d_model: int = 64, *, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng(42)
    return rng.standard_normal(d_model).astype(np.float32)


# ---------------------------------------------------------------------------
# TestEAGLEConfig
# ---------------------------------------------------------------------------

class TestEAGLEConfig(unittest.TestCase):

    def test_basic_init(self):
        cfg = EAGLEConfig(n_model_layers=16, d_model=512)
        self.assertEqual(cfg.n_model_layers, 16)
        self.assertEqual(cfg.d_model, 512)

    def test_d_head_default_ratio(self):
        cfg = EAGLEConfig(n_model_layers=8, d_model=128)
        self.assertEqual(cfg.d_head, 32)  # 128 // 4

    def test_d_head_custom_ratio(self):
        cfg = EAGLEConfig(n_model_layers=8, d_model=128, d_hidden_ratio=8)
        self.assertEqual(cfg.d_head, 16)

    def test_source_layer_indices_small(self):
        cfg = EAGLEConfig(n_model_layers=4, d_model=32)
        p50, p75 = cfg.source_layer_indices
        self.assertGreaterEqual(p50, 0)
        self.assertGreaterEqual(p75, 0)
        self.assertLess(p50, cfg.n_model_layers)
        self.assertLess(p75, cfg.n_model_layers)

    def test_source_layer_indices_large(self):
        cfg = EAGLEConfig(n_model_layers=32, d_model=4096)
        p50, p75 = cfg.source_layer_indices
        # 50th percentile: round(32 * 0.50) - 1 = 15
        # 75th percentile: round(32 * 0.75) - 1 = 23
        self.assertEqual(p50, 15)
        self.assertEqual(p75, 23)

    def test_source_layer_indices_minimum(self):
        cfg = EAGLEConfig(n_model_layers=1, d_model=32)
        p50, p75 = cfg.source_layer_indices
        self.assertGreaterEqual(p50, 0)
        self.assertGreaterEqual(p75, 0)

    def test_invalid_n_model_layers(self):
        with self.assertRaises(ValueError):
            EAGLEConfig(n_model_layers=0, d_model=64)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            EAGLEConfig(n_model_layers=8, d_model=0)

    def test_invalid_d_hidden_ratio(self):
        with self.assertRaises(ValueError):
            EAGLEConfig(n_model_layers=8, d_model=64, d_hidden_ratio=0)

    def test_invalid_acceptance_threshold_above_one(self):
        with self.assertRaises(ValueError):
            EAGLEConfig(n_model_layers=8, d_model=64,
                        acceptance_fallback_threshold=1.1)

    def test_invalid_acceptance_threshold_zero(self):
        with self.assertRaises(ValueError):
            EAGLEConfig(n_model_layers=8, d_model=64,
                        acceptance_fallback_threshold=0.0)

    def test_default_values(self):
        cfg = EAGLEConfig(n_model_layers=8, d_model=64)
        self.assertEqual(cfg.n_draft_layers, 3)
        self.assertEqual(cfg.d_hidden_ratio, 4)
        self.assertEqual(cfg.n_samples, 2000)
        self.assertEqual(cfg.n_epochs, 3)
        self.assertAlmostEqual(cfg.lr, 3e-4, places=7)
        self.assertEqual(cfg.n_draft_tokens, 5)
        self.assertAlmostEqual(cfg.acceptance_fallback_threshold, 0.55, places=7)

    def test_source_layers_non_negative(self):
        for n in [1, 2, 5, 10, 32, 80]:
            cfg = EAGLEConfig(n_model_layers=n, d_model=32)
            p50, p75 = cfg.source_layer_indices
            self.assertGreaterEqual(p50, 0, msg=f"p50 negative for n_layers={n}")
            self.assertGreaterEqual(p75, 0, msg=f"p75 negative for n_layers={n}")


# ---------------------------------------------------------------------------
# TestEAGLELayerWeights
# ---------------------------------------------------------------------------

class TestEAGLELayerWeights(unittest.TestCase):

    def _make_layer(self, d: int = 16) -> EAGLELayerWeights:
        rng = np.random.default_rng(1)
        w = lambda r, c: rng.standard_normal((r, c)).astype(np.float32)
        return EAGLELayerWeights(
            W_q=w(d, d), W_k=w(d, d), W_v=w(d, d), W_o=w(d, d),
            W_ff1=w(d * 4, d), W_ff2=w(d, d * 4),
            ln_attn_gamma=np.ones(d, dtype=np.float32),
            ln_ff_gamma=np.ones(d, dtype=np.float32),
        )

    def test_field_access(self):
        lw = self._make_layer(16)
        for attr in ("W_q", "W_k", "W_v", "W_o", "W_ff1", "W_ff2",
                     "ln_attn_gamma", "ln_ff_gamma"):
            self.assertIsInstance(getattr(lw, attr), np.ndarray)

    def test_field_shapes(self):
        d = 16
        lw = self._make_layer(d)
        self.assertEqual(lw.W_q.shape, (d, d))
        self.assertEqual(lw.W_ff1.shape, (d * 4, d))
        self.assertEqual(lw.W_ff2.shape, (d, d * 4))
        self.assertEqual(lw.ln_attn_gamma.shape, (d,))

    def test_mutable(self):
        lw = self._make_layer(16)
        orig = lw.W_q.copy()
        lw.W_q = lw.W_q * 2
        self.assertFalse(np.array_equal(lw.W_q, orig))


# ---------------------------------------------------------------------------
# TestEAGLEHeadWeights
# ---------------------------------------------------------------------------

class TestEAGLEHeadWeights(unittest.TestCase):

    def test_fields(self):
        weights = _make_weights(vocab_size=64)
        self.assertIsInstance(weights.config, EAGLEConfig)
        self.assertIsInstance(weights.layers, list)
        self.assertIsInstance(weights.input_proj, np.ndarray)
        self.assertIsInstance(weights.output_proj, np.ndarray)
        self.assertIsInstance(weights.vocab_size, int)

    def test_calibration_hash_default(self):
        weights = _make_weights()
        self.assertEqual(weights.calibration_hash, "")

    def test_vocab_size_stored(self):
        for vs in [32, 64, 128]:
            weights = _make_weights(vocab_size=vs)
            self.assertEqual(weights.vocab_size, vs)

    def test_n_layers(self):
        cfg = EAGLEConfig(n_model_layers=8, d_model=64, n_draft_layers=2)
        weights = _make_weights(cfg=cfg)
        self.assertEqual(len(weights.layers), 2)


# ---------------------------------------------------------------------------
# TestEagleForward
# ---------------------------------------------------------------------------

class TestEagleForward(unittest.TestCase):

    def setUp(self):
        self.cfg = _make_config(d_model=64)
        self.weights = _make_weights(self.cfg, vocab_size=32)
        self.rng = np.random.default_rng(7)

    def _make_h_in(self):
        h50 = self.rng.standard_normal(self.cfg.d_model).astype(np.float32)
        h75 = self.rng.standard_normal(self.cfg.d_model).astype(np.float32)
        return np.concatenate([h50, h75])[np.newaxis, :]

    def test_output_shape(self):
        log_probs = _eagle_forward(self._make_h_in(), self.weights)
        self.assertEqual(log_probs.shape, (32,))

    def test_log_softmax_all_negative_or_zero(self):
        log_probs = _eagle_forward(self._make_h_in(), self.weights)
        self.assertTrue(np.all(log_probs <= 0.0))

    def test_sum_exp_approx_one(self):
        log_probs = _eagle_forward(self._make_h_in(), self.weights)
        total = np.sum(np.exp(log_probs))
        self.assertAlmostEqual(float(total), 1.0, places=4)

    def test_deterministic(self):
        h_in = self._make_h_in()
        lp1 = _eagle_forward(h_in, self.weights)
        lp2 = _eagle_forward(h_in, self.weights)
        np.testing.assert_array_equal(lp1, lp2)

    def test_different_d_model(self):
        cfg = EAGLEConfig(n_model_layers=8, d_model=128)
        weights = _make_weights(cfg, vocab_size=16)
        h_in = np.zeros((1, 256), dtype=np.float32)
        lp = _eagle_forward(h_in, weights)
        self.assertEqual(lp.shape, (16,))

    def test_different_batch_shape(self):
        # h_in should be (1, 2*d_model)
        h_in = self._make_h_in()
        # Shape (1, 128) should work fine for d_model=64
        self.assertEqual(h_in.shape, (1, 128))
        out = _eagle_forward(h_in, self.weights)
        self.assertIsNotNone(out)


# ---------------------------------------------------------------------------
# TestEAGLEDistiller
# ---------------------------------------------------------------------------

class TestEAGLEDistiller(unittest.TestCase):

    def setUp(self):
        self.cfg = _make_config(n_model_layers=8, d_model=64)
        self.distiller = EAGLEDistiller(self.cfg, rng_seed=0)

    def test_init(self):
        self.assertIsInstance(self.distiller, EAGLEDistiller)
        self.assertIs(self.distiller.config, self.cfg)

    def test_init_weights_input_proj_shape(self):
        w = self.distiller._init_weights(32)
        d = self.cfg.d_head  # 64 // 4 = 16
        self.assertEqual(w.input_proj.shape, (d, self.cfg.d_model * 2))

    def test_init_weights_output_proj_shape(self):
        vocab = 64
        w = self.distiller._init_weights(vocab)
        d = self.cfg.d_head
        self.assertEqual(w.output_proj.shape, (vocab, d))

    def test_init_layer_count(self):
        w = self.distiller._init_weights(32)
        self.assertEqual(len(w.layers), self.cfg.n_draft_layers)

    def test_init_layer_w_q_shape(self):
        w = self.distiller._init_weights(32)
        d = self.cfg.d_head
        self.assertEqual(w.layers[0].W_q.shape, (d, d))

    def test_distill_returns_head_weights(self):
        prompts = ["hello"] * 3
        rng = np.random.default_rng(0)

        def hidden_states_fn(prompt):
            return [rng.standard_normal((4, 64)).astype(np.float32)
                    for _ in range(8)]

        result = self.distiller.distill(hidden_states_fn, prompts, vocab_size=32)
        self.assertIsInstance(result, EAGLEHeadWeights)

    def test_distill_calibration_hash_length(self):
        prompts = ["abc", "def"]
        rng = np.random.default_rng(1)

        def hidden_states_fn(prompt):
            return [rng.standard_normal((3, 64)).astype(np.float32)
                    for _ in range(8)]

        result = self.distiller.distill(hidden_states_fn, prompts, vocab_size=32)
        self.assertEqual(len(result.calibration_hash), 16)

    def test_distill_vocab_size_preserved(self):
        prompts = ["x"] * 2
        rng = np.random.default_rng(2)

        def hsfn(prompt):
            return [rng.standard_normal((2, 64)).astype(np.float32)
                    for _ in range(8)]

        result = self.distiller.distill(hsfn, prompts, vocab_size=48)
        self.assertEqual(result.vocab_size, 48)

    def test_distill_skips_short_hidden_states(self):
        """distill() should not crash when hidden_states_fn returns too few layers."""
        prompts = ["short"] * 3

        def hsfn_short(prompt):
            return [np.zeros((2, 64), dtype=np.float32)]  # only 1 layer

        # Should complete without error; calibration_hash still set
        result = self.distiller.distill(hsfn_short, prompts, vocab_size=16)
        self.assertIsInstance(result, EAGLEHeadWeights)

    def test_distill_with_empty_prompts(self):
        result = self.distiller.distill(lambda p: [], [], vocab_size=16)
        self.assertIsInstance(result, EAGLEHeadWeights)


# ---------------------------------------------------------------------------
# TestSaveLoadRoundtrip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundtrip(unittest.TestCase):

    def setUp(self):
        cfg = _make_config(n_model_layers=8, d_model=64)
        self.weights = _make_weights(cfg, vocab_size=32)
        self.weights.calibration_hash = "abcdef1234567890"

    def _roundtrip(self) -> tuple[Path, EAGLEHeadWeights]:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.squizd-eagle"
            save_eagle_head(self.weights, path)
            loaded = load_eagle_head(path)
            return path, loaded

    def test_save_returns_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "head.squizd-eagle"
            result = save_eagle_head(self.weights, path)
            self.assertIsInstance(result, Path)

    def test_roundtrip_d_model(self):
        _, loaded = self._roundtrip()
        self.assertEqual(loaded.config.d_model, self.weights.config.d_model)

    def test_roundtrip_n_draft_layers(self):
        _, loaded = self._roundtrip()
        self.assertEqual(loaded.config.n_draft_layers,
                         self.weights.config.n_draft_layers)

    def test_roundtrip_input_proj(self):
        _, loaded = self._roundtrip()
        np.testing.assert_array_almost_equal(
            loaded.input_proj, self.weights.input_proj)

    def test_roundtrip_output_proj(self):
        _, loaded = self._roundtrip()
        np.testing.assert_array_almost_equal(
            loaded.output_proj, self.weights.output_proj)

    def test_roundtrip_layer_w_q(self):
        _, loaded = self._roundtrip()
        np.testing.assert_array_almost_equal(
            loaded.layers[0].W_q, self.weights.layers[0].W_q)

    def test_roundtrip_vocab_size(self):
        _, loaded = self._roundtrip()
        self.assertEqual(loaded.vocab_size, self.weights.vocab_size)

    def test_roundtrip_calibration_hash(self):
        _, loaded = self._roundtrip()
        self.assertEqual(loaded.calibration_hash, "abcdef1234567890")

    def test_roundtrip_layer_count(self):
        _, loaded = self._roundtrip()
        self.assertEqual(len(loaded.layers), len(self.weights.layers))

    def test_invalid_tag_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.squizd-eagle"
            # Write garbage header
            path.write_bytes(b"BAAD" + b"\x00" * 64)
            with self.assertRaises(ValueError):
                load_eagle_head(path)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_eagle_head("/non/existent/path.squizd-eagle")


# ---------------------------------------------------------------------------
# TestEagleDecodeStep
# ---------------------------------------------------------------------------

class TestEagleDecodeStep(unittest.TestCase):

    def setUp(self):
        cfg = _make_config(n_model_layers=8, d_model=64)
        self.weights = _make_weights(cfg, vocab_size=32)
        self.rng = np.random.default_rng(42)

    def _hidden(self):
        return self.rng.standard_normal(64).astype(np.float32)

    def test_returns_list(self):
        drafts = eagle_decode_step(self._hidden(), self._hidden(), self.weights,
                                   n_draft=3, rng=np.random.default_rng(0))
        self.assertIsInstance(drafts, list)

    def test_length_n_draft(self):
        for n in [1, 3, 5]:
            drafts = eagle_decode_step(self._hidden(), self._hidden(), self.weights,
                                       n_draft=n, rng=np.random.default_rng(0))
            self.assertLessEqual(len(drafts), n)
            self.assertGreater(len(drafts), 0)

    def test_descending_log_probs(self):
        drafts = eagle_decode_step(self._hidden(), self._hidden(), self.weights,
                                   n_draft=5, rng=np.random.default_rng(0))
        lps = [d.log_prob for d in drafts]
        self.assertEqual(lps, sorted(lps, reverse=True))

    def test_token_ids_in_vocab(self):
        drafts = eagle_decode_step(self._hidden(), self._hidden(), self.weights,
                                   n_draft=5, rng=np.random.default_rng(0))
        for d in drafts:
            self.assertGreaterEqual(d.token_id, 0)
            self.assertLess(d.token_id, self.weights.vocab_size)

    def test_deterministic_with_seed(self):
        h50, h75 = self._hidden(), self._hidden()
        d1 = eagle_decode_step(h50, h75, self.weights, n_draft=3,
                               rng=np.random.default_rng(99))
        d2 = eagle_decode_step(h50, h75, self.weights, n_draft=3,
                               rng=np.random.default_rng(99))
        self.assertEqual([t.token_id for t in d1],
                         [t.token_id for t in d2])

    def test_top_k_limits_unique_tokens(self):
        # top_k=1 → should always return the single highest-prob token
        drafts = eagle_decode_step(self._hidden(), self._hidden(), self.weights,
                                   n_draft=3, top_k=1,
                                   rng=np.random.default_rng(0))
        self.assertLessEqual(len(drafts), 1)

    def test_1d_input_accepted(self):
        h50 = self._hidden()  # (64,) — 1D
        h75 = self._hidden()
        # Should not raise — reshape is handled internally
        drafts = eagle_decode_step(h50, h75, self.weights, n_draft=2,
                                   rng=np.random.default_rng(0))
        self.assertIsInstance(drafts, list)

    def test_2d_input_accepted(self):
        h50 = self._hidden().reshape(1, 64)  # (1, 64)
        h75 = self._hidden().reshape(1, 64)
        drafts = eagle_decode_step(h50, h75, self.weights, n_draft=2,
                                   rng=np.random.default_rng(0))
        self.assertIsInstance(drafts, list)


# ---------------------------------------------------------------------------
# TestSampleTopK
# ---------------------------------------------------------------------------

class TestSampleTopK(unittest.TestCase):

    def test_returns_draft_tokens(self):
        vocab = 32
        lp = np.log(np.ones(vocab) / vocab)
        tokens = _sample_top_k(lp, 5, rng=np.random.default_rng(0))
        self.assertIsInstance(tokens, list)
        for t in tokens:
            self.assertIsInstance(t, DraftToken)

    def test_length(self):
        vocab = 32
        lp = np.zeros(vocab)
        tokens = _sample_top_k(lp, 5, rng=np.random.default_rng(0))
        self.assertEqual(len(tokens), 5)

    def test_top_k_constrains_candidates(self):
        vocab = 32
        lp = np.zeros(vocab)
        lp[0] = 100.0  # overwhelmingly highest
        # top_k=1 → only token 0 is in candidate set
        tokens = _sample_top_k(lp, 3, top_k=1, rng=np.random.default_rng(0))
        self.assertEqual(len(tokens), 1)
        self.assertEqual(tokens[0].token_id, 0)


# ---------------------------------------------------------------------------
# TestEAGLERunnerConfig
# ---------------------------------------------------------------------------

class TestEAGLERunnerConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = EAGLERunnerConfig()
        self.assertEqual(cfg.n_draft, 5)
        self.assertAlmostEqual(cfg.temperature, 1.0, places=7)
        self.assertEqual(cfg.top_k, 50)

    def test_invalid_n_draft(self):
        with self.assertRaises(ValueError):
            EAGLERunnerConfig(n_draft=0)

    def test_invalid_temperature_zero(self):
        with self.assertRaises(ValueError):
            EAGLERunnerConfig(temperature=0.0)

    def test_invalid_temperature_negative(self):
        with self.assertRaises(ValueError):
            EAGLERunnerConfig(temperature=-0.5)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            EAGLERunnerConfig(top_k=-1)

    def test_top_k_zero_is_valid(self):
        cfg = EAGLERunnerConfig(top_k=0)
        self.assertEqual(cfg.top_k, 0)

    def test_custom_values(self):
        cfg = EAGLERunnerConfig(n_draft=3, temperature=0.7, top_k=20)
        self.assertEqual(cfg.n_draft, 3)
        self.assertAlmostEqual(cfg.temperature, 0.7, places=7)
        self.assertEqual(cfg.top_k, 20)


# ---------------------------------------------------------------------------
# TestDraftToken
# ---------------------------------------------------------------------------

class TestDraftToken(unittest.TestCase):

    def test_fields(self):
        t = DraftToken(token_id=42, log_prob=-1.5)
        self.assertEqual(t.token_id, 42)
        self.assertAlmostEqual(t.log_prob, -1.5, places=7)

    def test_list_sort_by_log_prob(self):
        tokens = [DraftToken(0, -2.0), DraftToken(1, -0.5), DraftToken(2, -1.0)]
        tokens.sort(key=lambda t: t.log_prob, reverse=True)
        self.assertEqual(tokens[0].token_id, 1)

    def test_token_id_zero(self):
        t = DraftToken(token_id=0, log_prob=0.0)
        self.assertEqual(t.token_id, 0)


# ---------------------------------------------------------------------------
# TestEAGLEHeadRunner
# ---------------------------------------------------------------------------

class TestEAGLEHeadRunner(unittest.TestCase):

    def setUp(self):
        cfg = _make_config(n_model_layers=8, d_model=64)
        self.weights = _make_weights(cfg, vocab_size=32)
        self.runner = EAGLEHeadRunner(
            self.weights, EAGLERunnerConfig(n_draft=3), rng_seed=42
        )

    def _hidden(self):
        return np.random.default_rng(7).standard_normal(64).astype(np.float32)

    def test_init(self):
        self.assertIsInstance(self.runner, EAGLEHeadRunner)

    def test_generate_drafts_returns_list(self):
        drafts = self.runner.generate_drafts(self._hidden(), self._hidden())
        self.assertIsInstance(drafts, list)

    def test_generate_drafts_length(self):
        drafts = self.runner.generate_drafts(self._hidden(), self._hidden())
        self.assertGreater(len(drafts), 0)
        self.assertLessEqual(len(drafts), 3)

    def test_generate_drafts_n_draft_override(self):
        drafts = self.runner.generate_drafts(self._hidden(), self._hidden(),
                                              n_draft=1)
        self.assertLessEqual(len(drafts), 1)

    def test_record_acceptance_updates_total(self):
        self.runner.record_acceptance(n_accepted=3, n_proposed=5)
        self.assertEqual(self.runner._total_proposed, 5)
        self.assertEqual(self.runner._total_accepted, 3)

    def test_rolling_acceptance_rate_empty(self):
        # Before any records, should return 1.0 (optimistic warm-up)
        self.assertAlmostEqual(self.runner.rolling_acceptance_rate, 1.0, places=7)

    def test_rolling_acceptance_rate_after_recording(self):
        self.runner.record_acceptance(n_accepted=4, n_proposed=5)
        rate = self.runner.rolling_acceptance_rate
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)
        self.assertAlmostEqual(rate, 4.0 / 5.0, places=5)

    def test_lifetime_acceptance_rate_empty(self):
        self.assertAlmostEqual(self.runner.lifetime_acceptance_rate, 1.0, places=7)

    def test_lifetime_acceptance_rate_after_recording(self):
        self.runner.record_acceptance(n_accepted=2, n_proposed=10)
        self.assertAlmostEqual(
            self.runner.lifetime_acceptance_rate, 0.2, places=7)

    def test_should_fallback_warm_up_guard(self):
        # Only 5 tokens recorded — window < ROLLING_WINDOW_SIZE // 4 = 16
        self.runner.record_acceptance(0, 5)
        self.assertFalse(self.runner.should_fallback())

    def test_should_fallback_high_rate(self):
        # 20 tokens, all accepted → high rate → no fallback
        for _ in range(4):
            self.runner.record_acceptance(n_accepted=5, n_proposed=5)
        self.assertFalse(self.runner.should_fallback())

    def test_should_fallback_low_rate(self):
        # 20 tokens, all rejected → rate 0.0 < 0.55 → fallback
        for _ in range(4):
            self.runner.record_acceptance(n_accepted=0, n_proposed=5)
        self.assertTrue(self.runner.should_fallback())

    def test_reset_stats_clears_window(self):
        self.runner.record_acceptance(n_accepted=0, n_proposed=20)
        self.runner.reset_stats()
        self.assertEqual(len(self.runner._window), 0)
        self.assertEqual(self.runner._total_proposed, 0)
        self.assertEqual(self.runner._total_accepted, 0)
        self.assertAlmostEqual(self.runner.rolling_acceptance_rate, 1.0, places=7)

    def test_record_acceptance_invalid_proposed_negative(self):
        with self.assertRaises(ValueError):
            self.runner.record_acceptance(n_accepted=0, n_proposed=-1)

    def test_record_acceptance_invalid_accepted_negative(self):
        with self.assertRaises(ValueError):
            self.runner.record_acceptance(n_accepted=-1, n_proposed=5)

    def test_record_acceptance_invalid_accepted_gt_proposed(self):
        with self.assertRaises(ValueError):
            self.runner.record_acceptance(n_accepted=6, n_proposed=5)

    def test_window_trimmed_to_rolling_window_size(self):
        # Submit more than ROLLING_WINDOW_SIZE (64) tokens
        for _ in range(20):
            self.runner.record_acceptance(n_accepted=5, n_proposed=5)
        self.assertLessEqual(len(self.runner._window), ROLLING_WINDOW_SIZE)

    def test_rolling_window_size_constant(self):
        self.assertEqual(ROLLING_WINDOW_SIZE, 64)

    def test_default_config_used_when_none(self):
        runner = EAGLEHeadRunner(self.weights)
        self.assertIsInstance(runner.config, EAGLERunnerConfig)
        self.assertEqual(runner.config.n_draft, 5)


if __name__ == "__main__":
    unittest.main()
