"""tests/test_wave61a_rust_kernels.py — Wave 61a Rust kernel tests.

Tests for:
  - RustWandaNM        (rs_wanda_nm.py)
  - RustFluteLUT       (rs_flute_lut.py)
  - RustDeltaNet       (rs_delta_net.py)
  - RustGreenKVScore   (rs_green_kv_score.py)
  - RustJacobiConv     (rs_jacobi_conv.py)
  - RustTreeVerify     (rs_tree_verify.py)

All tests exercise both the Python wrapper logic and the Rust kernel
(when available) or NumPy fallback.  75 tests total.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from squish.kernels.rs_wanda_nm import WandaNMConfig, RustWandaNM
from squish.kernels.rs_flute_lut import FluteLUTConfig, RustFluteLUT
from squish.kernels.rs_delta_net import DeltaNetConfig, RustDeltaNet
from squish.kernels.rs_green_kv_score import GreenKVConfig, RustGreenKVScore
from squish.kernels.rs_jacobi_conv import JacobiConvConfig, RustJacobiConv
from squish.kernels.rs_tree_verify import TreeVerifyConfig, RustTreeVerify


# ── WandaNMConfig ─────────────────────────────────────────────────────────────


class TestWandaNMConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = WandaNMConfig()
        self.assertEqual(cfg.n, 2)
        self.assertEqual(cfg.m, 4)

    def test_custom(self):
        cfg = WandaNMConfig(n=1, m=8)
        self.assertEqual(cfg.n, 1)
        self.assertEqual(cfg.m, 8)


# ── RustWandaNM ───────────────────────────────────────────────────────────────


class TestRustWandaNM(unittest.TestCase):
    def setUp(self):
        self.wanda = RustWandaNM()
        self.rng = np.random.default_rng(42)
        self.w = self.rng.standard_normal((8, 16)).astype(np.float32)
        self.rms = np.ones(16, dtype=np.float32)

    def test_backend_string(self):
        self.assertIn(self.wanda.backend(), ("rust", "numpy"))

    def test_importance_shape(self):
        imp = self.wanda.importance(self.w, self.rms)
        self.assertEqual(imp.shape, self.w.shape)

    def test_importance_dtype(self):
        imp = self.wanda.importance(self.w, self.rms)
        self.assertEqual(imp.dtype, np.float32)

    def test_importance_non_negative(self):
        imp = self.wanda.importance(self.w, self.rms)
        self.assertTrue((imp >= 0).all())

    def test_importance_equals_abs_w_when_rms_ones(self):
        imp = self.wanda.importance(self.w, self.rms)
        np.testing.assert_allclose(imp, np.abs(self.w), rtol=1e-5)

    def test_importance_scales_with_rms(self):
        rms2 = np.full(16, 2.0, dtype=np.float32)
        imp1 = self.wanda.importance(self.w, self.rms)
        imp2 = self.wanda.importance(self.w, rms2)
        np.testing.assert_allclose(imp2, imp1 * 2.0, rtol=1e-5)

    def test_importance_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.wanda.importance(self.w, np.ones(12, dtype=np.float32))

    def test_nm_mask_shape(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        self.assertEqual(mask.shape, self.w.shape)

    def test_nm_mask_dtype(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        self.assertEqual(mask.dtype, np.uint8)

    def test_nm_mask_values_binary(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

    def test_nm_mask_2in4_count(self):
        """Exactly n ones per m-block per row."""
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        rows, cols = mask.shape
        for row in range(rows):
            for bi in range(cols // 4):
                block_sum = mask[row, bi * 4:(bi + 1) * 4].sum()
                self.assertEqual(block_sum, 2)

    def test_nm_mask_n_gt_m_raises(self):
        imp = np.ones_like(self.w)
        with self.assertRaises(ValueError):
            self.wanda.nm_mask(imp, n=5, m=4)

    def test_prune_zeros_pruned_entries(self):
        pruned = self.wanda.prune(self.w, self.rms, n=2, m=4)
        mask = self.wanda.nm_mask(self.wanda.importance(self.w, self.rms), n=2, m=4)
        zero_positions = mask == 0
        np.testing.assert_array_equal(pruned[zero_positions], 0.0)


# ── FluteLUTConfig ────────────────────────────────────────────────────────────


class TestFluteLUTConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = FluteLUTConfig()
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.cb_size, 16)

    def test_custom(self):
        cfg = FluteLUTConfig(group_size=64, cb_size=8)
        self.assertEqual(cfg.group_size, 64)


# ── RustFluteLUT ──────────────────────────────────────────────────────────────


class TestRustFluteLUT(unittest.TestCase):
    def setUp(self):
        self.flute = RustFluteLUT(FluteLUTConfig(group_size=4, cb_size=8))
        self.rng = np.random.default_rng(0)
        self.rows, self.cols = 6, 8
        self.w = self.rng.standard_normal((self.rows, self.cols)).astype(np.float32)
        self.cb = self.rng.standard_normal((2, 8)).astype(np.float32)  # 8 cols / gs=4 = 2 groups

    def test_backend_string(self):
        self.assertIn(self.flute.backend(), ("rust", "numpy"))

    def test_encode_shape(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        self.assertEqual(codes.shape, self.w.shape)

    def test_encode_dtype(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        self.assertEqual(codes.dtype, np.uint8)

    def test_encode_indices_in_range(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        self.assertTrue((codes < 8).all())

    def test_decode_shape(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        dec = self.flute.decode(codes, self.cb, group_size=4)
        self.assertEqual(dec.shape, self.w.shape)

    def test_decode_dtype(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        dec = self.flute.decode(codes, self.cb, group_size=4)
        self.assertEqual(dec.dtype, np.float32)

    def test_encode_decode_values_from_codebook(self):
        """Decoded values must be exact codebook entries."""
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        dec = self.flute.decode(codes, self.cb, group_size=4)
        for g in range(2):
            cs = g * 4
            ce = cs + 4
            for row in range(self.rows):
                for c in range(cs, ce):
                    idx = int(codes[row, c])
                    self.assertAlmostEqual(float(dec[row, c]), float(self.cb[g, idx]), places=5)

    def test_roundtrip_error_finite(self):
        err = self.flute.roundtrip_error(self.w, self.cb, group_size=4)
        self.assertTrue(math.isfinite(err))
        self.assertGreaterEqual(err, 0.0)

    def test_group_mismatch_raises(self):
        bad_cb = np.ones((5, 8), dtype=np.float32)  # wrong number of groups
        with self.assertRaises(ValueError):
            self.flute.encode(self.w, bad_cb, group_size=4)


# ── DeltaNetConfig ────────────────────────────────────────────────────────────


class TestDeltaNetConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DeltaNetConfig()
        self.assertEqual(cfg.eps, 1e-8)

    def test_custom(self):
        cfg = DeltaNetConfig(eps=1e-6)
        self.assertAlmostEqual(cfg.eps, 1e-6)


# ── RustDeltaNet ──────────────────────────────────────────────────────────────


class TestRustDeltaNet(unittest.TestCase):
    def setUp(self):
        self.delta = RustDeltaNet()
        self.rng = np.random.default_rng(7)
        self.T, self.H, self.D = 5, 2, 4

    def _make_qkvb(self):
        q = self.rng.standard_normal((self.T, self.H, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.T, self.H, self.D)).astype(np.float32)
        v = self.rng.standard_normal((self.T, self.H, self.D)).astype(np.float32)
        beta = np.full((self.T, self.H), 0.05, dtype=np.float32)
        return q, k, v, beta

    def test_backend_string(self):
        self.assertIn(self.delta.backend(), ("rust", "numpy"))

    def test_scan_shape(self):
        q, k, v, beta = self._make_qkvb()
        out = self.delta.scan(q, k, v, beta)
        self.assertEqual(out.shape, (self.T, self.H, self.D))

    def test_scan_dtype(self):
        q, k, v, beta = self._make_qkvb()
        out = self.delta.scan(q, k, v, beta)
        self.assertEqual(out.dtype, np.float32)

    def test_scan_output_finite(self):
        q, k, v, beta = self._make_qkvb()
        out = self.delta.scan(q, k, v, beta)
        self.assertTrue(np.isfinite(out).all())

    def test_scan_zero_beta_output_is_zero(self):
        """If beta=0 the state never updates so W@q = 0 (zero init)."""
        q, k, v, _ = self._make_qkvb()
        beta_zero = np.zeros((self.T, self.H), dtype=np.float32)
        out = self.delta.scan(q, k, v, beta_zero)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_scan_shape_mismatch_raises(self):
        q, k, v, beta = self._make_qkvb()
        with self.assertRaises(ValueError):
            self.delta.scan(q, k[:, :1, :], v, beta)

    def test_scan_beta_mismatch_raises(self):
        q, k, v, _ = self._make_qkvb()
        bad_beta = np.ones((self.T, self.H + 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.delta.scan(q, k, v, bad_beta)


# ── GreenKVConfig ─────────────────────────────────────────────────────────────


class TestGreenKVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GreenKVConfig()
        self.assertEqual(cfg.obs_window, 32)

    def test_custom(self):
        cfg = GreenKVConfig(obs_window=8)
        self.assertEqual(cfg.obs_window, 8)


# ── RustGreenKVScore ──────────────────────────────────────────────────────────


class TestRustGreenKVScore(unittest.TestCase):
    def setUp(self):
        self.scorer = RustGreenKVScore()
        self.rng = np.random.default_rng(11)
        self.H, self.obs, self.seq, self.D = 2, 3, 8, 4

    def test_backend_string(self):
        self.assertIn(self.scorer.backend(), ("rust", "numpy"))

    def test_score_shape(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.H, self.seq, self.D)).astype(np.float32)
        s = self.scorer.score(q, k)
        self.assertEqual(s.shape, (self.H, self.seq))

    def test_score_dtype(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.H, self.seq, self.D)).astype(np.float32)
        s = self.scorer.score(q, k)
        self.assertEqual(s.dtype, np.float32)

    def test_score_non_negative(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.H, self.seq, self.D)).astype(np.float32)
        s = self.scorer.score(q, k)
        self.assertTrue((s >= 0).all())

    def test_score_sums_approximately_one_per_head(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.H, self.seq, self.D)).astype(np.float32)
        s = self.scorer.score(q, k)
        for h in range(self.H):
            self.assertAlmostEqual(float(s[h].sum()), 1.0, places=4)

    def test_score_n_heads_mismatch_raises(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k_bad = self.rng.standard_normal((self.H + 1, self.seq, self.D)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.scorer.score(q, k_bad)

    def test_score_head_dim_mismatch_raises(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k_bad = self.rng.standard_normal((self.H, self.seq, self.D + 1)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.scorer.score(q, k_bad)

    def test_top_k_mask_shape(self):
        s = np.random.default_rng(0).random((self.H, self.seq)).astype(np.float32)
        mask = self.scorer.top_k_mask(s, budget=4)
        self.assertEqual(mask.shape, (self.H, self.seq))

    def test_top_k_mask_count(self):
        s = np.random.default_rng(1).random((self.H, self.seq)).astype(np.float32)
        mask = self.scorer.top_k_mask(s, budget=3)
        for h in range(self.H):
            self.assertEqual(mask[h].sum(), 3)


# ── JacobiConvConfig ──────────────────────────────────────────────────────────


class TestJacobiConvConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = JacobiConvConfig()
        self.assertEqual(cfg.temperature, 0.0)

    def test_custom(self):
        cfg = JacobiConvConfig(temperature=1.0)
        self.assertAlmostEqual(cfg.temperature, 1.0)


# ── RustJacobiConv ────────────────────────────────────────────────────────────


class TestRustJacobiConv(unittest.TestCase):
    def setUp(self):
        self.jconv = RustJacobiConv()
        self.rng = np.random.default_rng(5)
        self.N, self.vocab = 6, 20

    def test_backend_string(self):
        self.assertIn(self.jconv.backend(), ("rust", "numpy"))

    def test_check_shape_greedy(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, nf = self.jconv.check(logits, guesses, temperature=0.0)
        self.assertEqual(ng.shape, (self.N,))

    def test_check_dtype(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses, temperature=0.0)
        self.assertEqual(ng.dtype, np.int32)

    def test_check_indices_in_range(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses, temperature=0.0)
        self.assertTrue((ng >= 0).all())
        self.assertTrue((ng < self.vocab).all())

    def test_check_greedy_matches_argmax(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses, temperature=0.0)
        expected = logits.argmax(axis=1).astype(np.int32)
        np.testing.assert_array_equal(ng, expected)

    def test_check_all_fixed_when_guesses_match_argmax(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = logits.argmax(axis=1).astype(np.int32)
        _, nf = self.jconv.check(logits, guesses, temperature=0.0)
        self.assertEqual(nf, self.N)

    def test_check_n_fixed_is_int(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        _, nf = self.jconv.check(logits, guesses)
        self.assertIsInstance(nf, int)

    def test_check_stochastic_shape(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses, temperature=1.0, seed=42)
        self.assertEqual(ng.shape, (self.N,))

    def test_check_n_mismatch_raises(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        bad_g = np.zeros(self.N + 2, dtype=np.int32)
        with self.assertRaises(ValueError):
            self.jconv.check(logits, bad_g)

    def test_converged_true_when_all_match(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = logits.argmax(axis=1).astype(np.int32)
        self.assertTrue(self.jconv.converged(logits, guesses, temperature=0.0))

    def test_converged_false_when_none_match(self):
        logits = np.zeros((self.N, self.vocab), dtype=np.float32)
        logits[:, 0] = 10.0  # all argmax → 0
        guesses = np.ones(self.N, dtype=np.int32)  # all 1
        self.assertFalse(self.jconv.converged(logits, guesses, temperature=0.0))


# ── TreeVerifyConfig ──────────────────────────────────────────────────────────


class TestTreeVerifyConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TreeVerifyConfig()
        self.assertAlmostEqual(cfg.temperature, 1.0)

    def test_custom(self):
        cfg = TreeVerifyConfig(temperature=0.7)
        self.assertAlmostEqual(cfg.temperature, 0.7)


# ── RustTreeVerify ────────────────────────────────────────────────────────────


class TestRustTreeVerify(unittest.TestCase):
    def setUp(self):
        self.verifier = RustTreeVerify()
        self.rng = np.random.default_rng(99)
        self.B, self.D, self.V = 3, 5, 20
        self.draft_tokens = np.zeros((self.B, self.D), dtype=np.int32)
        self.draft_logits = self.rng.standard_normal((self.B, self.D, self.V)).astype(np.float32)
        self.target_logits = self.rng.standard_normal((self.B, self.D, self.V)).astype(np.float32)

    def test_backend_string(self):
        self.assertIn(self.verifier.backend(), ("rust", "numpy"))

    def test_verify_accepted_is_array(self):
        accepted, bl = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        self.assertIsInstance(accepted, np.ndarray)

    def test_verify_best_len_is_int(self):
        _, bl = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        self.assertIsInstance(bl, int)

    def test_verify_best_len_leq_n_draft(self):
        _, bl = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        self.assertLessEqual(bl, self.D)

    def test_verify_best_len_geq_1(self):
        """At least one token accepted (correction token on first rejection)."""
        _, bl = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        self.assertGreaterEqual(bl, 1)

    def test_verify_perfect_draft_accepts_all(self):
        """When draft == target distributions, all tokens should be accepted."""
        logits = self.rng.standard_normal((self.B, self.D, self.V)).astype(np.float32)
        accepted, bl = self.verifier.verify(
            self.draft_tokens, logits, logits, temperature=1.0, seed=0
        )
        self.assertEqual(bl, self.D)

    def test_verify_accepted_dtype(self):
        accepted, _ = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits
        )
        self.assertEqual(accepted.dtype, np.int32)

    def test_verify_b_mismatch_raises(self):
        bad_tl = self.rng.standard_normal((self.B + 1, self.D, self.V)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.verifier.verify(self.draft_tokens, self.draft_logits, bad_tl)

    def test_verify_vocab_mismatch_raises(self):
        bad_tl = self.rng.standard_normal((self.B, self.D, self.V + 5)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.verifier.verify(self.draft_tokens, self.draft_logits, bad_tl)

    def test_acceptance_rate_in_range(self):
        rate = self.verifier.acceptance_rate(
            self.draft_tokens, self.draft_logits, self.target_logits, n_trials=10
        )
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)


if __name__ == "__main__":
    unittest.main()
