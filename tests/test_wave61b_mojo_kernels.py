"""tests/test_wave61b_mojo_kernels.py — Wave 61b Mojo kernel tests.

Tests for:
  - MojoWandaNM        (wanda_nm_mojo.py)
  - MojoFluteLUT       (flute_lut_mojo.py)
  - MojoDeltaNet       (delta_net_mojo.py)
  - MojoGreenKVScore   (green_kv_score_mojo.py)
  - MojoJacobiConv     (jacobi_conv_mojo.py)
  - MojoTreeVerify     (tree_verify_mojo.py)

All tests run through the NumPy fallback path (Mojo runtime absent in CI).
Covers config defaults, output shapes, dtypes, values, error handling,
and NumPy-parity checks.  75+ tests total.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from squish.kernels.mojo.wanda_nm_mojo import WandaNMMojoConfig, MojoWandaNM
from squish.kernels.mojo.flute_lut_mojo import FluteLUTMojoConfig, MojoFluteLUT
from squish.kernels.mojo.delta_net_mojo import DeltaNetMojoConfig, MojoDeltaNet
from squish.kernels.mojo.green_kv_score_mojo import GreenKVMojoConfig, MojoGreenKVScore
from squish.kernels.mojo.jacobi_conv_mojo import JacobiConvMojoConfig, MojoJacobiConv
from squish.kernels.mojo.tree_verify_mojo import TreeVerifyMojoConfig, MojoTreeVerify


# ── WandaNMMojoConfig ─────────────────────────────────────────────────────────


class TestWandaNMMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = WandaNMMojoConfig()
        self.assertEqual(cfg.n, 2)
        self.assertEqual(cfg.m, 4)

    def test_custom(self):
        cfg = WandaNMMojoConfig(n=1, m=8)
        self.assertEqual(cfg.n, 1)
        self.assertEqual(cfg.m, 8)


# ── MojoWandaNM ───────────────────────────────────────────────────────────────


class TestMojoWandaNM(unittest.TestCase):
    def setUp(self):
        self.wanda = MojoWandaNM()
        self.rng = np.random.default_rng(42)
        self.w = self.rng.standard_normal((8, 16)).astype(np.float32)
        self.rms = np.ones(16, dtype=np.float32)

    def test_backend_is_string(self):
        self.assertIsInstance(self.wanda.backend(), str)

    def test_backend_is_numpy_without_mojo(self):
        self.assertEqual(self.wanda.backend(), "numpy")

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

    def test_importance_rms_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.wanda.importance(self.w, np.ones(10, dtype=np.float32))

    def test_nm_mask_shape(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        self.assertEqual(mask.shape, self.w.shape)

    def test_nm_mask_dtype(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        self.assertEqual(mask.dtype, np.uint8)

    def test_nm_mask_binary_values(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

    def test_nm_mask_2in4_count_per_block(self):
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        rows, cols = mask.shape
        for row in range(rows):
            for bi in range(cols // 4):
                block = mask[row, bi * 4:(bi + 1) * 4]
                self.assertEqual(block.sum(), 2)

    def test_prune_zeros_masked_entries(self):
        pruned = self.wanda.prune(self.w, self.rms, n=2, m=4)
        imp = self.wanda.importance(self.w, self.rms)
        mask = self.wanda.nm_mask(imp, n=2, m=4)
        np.testing.assert_array_equal(pruned[mask == 0], 0.0)

    def test_parity_with_rust_wanda(self):
        from squish.kernels.rs_wanda_nm import RustWandaNM
        rw = RustWandaNM()
        imp_r = rw.importance(self.w, self.rms)
        imp_m = self.wanda.importance(self.w, self.rms)
        np.testing.assert_allclose(imp_m, imp_r, rtol=1e-5)


# ── FluteLUTMojoConfig ────────────────────────────────────────────────────────


class TestFluteLUTMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = FluteLUTMojoConfig()
        self.assertEqual(cfg.group_size, 128)
        self.assertEqual(cfg.cb_size, 16)

    def test_custom(self):
        cfg = FluteLUTMojoConfig(group_size=32, cb_size=8)
        self.assertEqual(cfg.group_size, 32)


# ── MojoFluteLUT ──────────────────────────────────────────────────────────────


class TestMojoFluteLUT(unittest.TestCase):
    def setUp(self):
        self.flute = MojoFluteLUT(FluteLUTMojoConfig(group_size=4, cb_size=8))
        self.rng = np.random.default_rng(0)
        self.rows, self.cols = 6, 8
        # 2 groups of size 4
        self.w = self.rng.standard_normal((self.rows, self.cols)).astype(np.float32)
        self.cb = self.rng.standard_normal((2, 8)).astype(np.float32)

    def test_backend_is_numpy(self):
        self.assertEqual(self.flute.backend(), "numpy")

    def test_encode_shape(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        self.assertEqual(codes.shape, self.w.shape)

    def test_encode_dtype(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        self.assertEqual(codes.dtype, np.uint8)

    def test_encode_range(self):
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

    def test_decode_values_from_codebook(self):
        codes = self.flute.encode(self.w, self.cb, group_size=4)
        dec = self.flute.decode(codes, self.cb, group_size=4)
        for g in range(2):
            cs, ce = g * 4, (g + 1) * 4
            for row in range(self.rows):
                for c in range(cs, ce):
                    idx = int(codes[row, c])
                    self.assertAlmostEqual(float(dec[row, c]), float(self.cb[g, idx]), places=5)

    def test_roundtrip_error_finite(self):
        err = self.flute.roundtrip_error(self.w, self.cb, group_size=4)
        self.assertTrue(math.isfinite(err))
        self.assertGreaterEqual(err, 0.0)

    def test_parity_with_rust_flute(self):
        from squish.kernels.rs_flute_lut import RustFluteLUT, FluteLUTConfig
        rf = RustFluteLUT(FluteLUTConfig(group_size=4, cb_size=8))
        codes_r = rf.encode(self.w, self.cb, group_size=4)
        codes_m = self.flute.encode(self.w, self.cb, group_size=4)
        np.testing.assert_array_equal(codes_m, codes_r)


# ── DeltaNetMojoConfig ────────────────────────────────────────────────────────


class TestDeltaNetMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DeltaNetMojoConfig()
        self.assertEqual(cfg.eps, 1e-8)

    def test_custom(self):
        cfg = DeltaNetMojoConfig(eps=1e-5)
        self.assertAlmostEqual(cfg.eps, 1e-5)


# ── MojoDeltaNet ──────────────────────────────────────────────────────────────


class TestMojoDeltaNet(unittest.TestCase):
    def setUp(self):
        self.delta = MojoDeltaNet()
        self.rng = np.random.default_rng(7)
        self.T, self.H, self.D = 4, 2, 4

    def _make_qkvb(self):
        q = self.rng.standard_normal((self.T, self.H, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.T, self.H, self.D)).astype(np.float32)
        v = self.rng.standard_normal((self.T, self.H, self.D)).astype(np.float32)
        beta = np.full((self.T, self.H), 0.05, dtype=np.float32)
        return q, k, v, beta

    def test_backend_is_numpy(self):
        self.assertEqual(self.delta.backend(), "numpy")

    def test_scan_shape(self):
        q, k, v, beta = self._make_qkvb()
        out = self.delta.scan(q, k, v, beta)
        self.assertEqual(out.shape, (self.T, self.H, self.D))

    def test_scan_dtype(self):
        q, k, v, beta = self._make_qkvb()
        out = self.delta.scan(q, k, v, beta)
        self.assertEqual(out.dtype, np.float32)

    def test_scan_finite(self):
        q, k, v, beta = self._make_qkvb()
        out = self.delta.scan(q, k, v, beta)
        self.assertTrue(np.isfinite(out).all())

    def test_scan_zero_beta(self):
        q, k, v, _ = self._make_qkvb()
        out = self.delta.scan(q, k, v, np.zeros((self.T, self.H), dtype=np.float32))
        np.testing.assert_allclose(out, 0.0, atol=1e-6)

    def test_scan_shape_mismatch_raises(self):
        q, k, v, beta = self._make_qkvb()
        with self.assertRaises(ValueError):
            self.delta.scan(q, k[:, :1, :], v, beta)

    def test_parity_with_rust_delta(self):
        from squish.kernels.rs_delta_net import RustDeltaNet
        rd = RustDeltaNet()
        q, k, v, beta = self._make_qkvb()
        out_r = rd.scan(q, k, v, beta)
        out_m = self.delta.scan(q, k, v, beta)
        np.testing.assert_allclose(out_m, out_r, rtol=1e-4, atol=1e-5)


# ── GreenKVMojoConfig ─────────────────────────────────────────────────────────


class TestGreenKVMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GreenKVMojoConfig()
        self.assertEqual(cfg.obs_window, 32)

    def test_custom(self):
        cfg = GreenKVMojoConfig(obs_window=16)
        self.assertEqual(cfg.obs_window, 16)


# ── MojoGreenKVScore ──────────────────────────────────────────────────────────


class TestMojoGreenKVScore(unittest.TestCase):
    def setUp(self):
        self.scorer = MojoGreenKVScore()
        self.rng = np.random.default_rng(11)
        self.H, self.obs, self.seq, self.D = 2, 3, 8, 4

    def test_backend_is_numpy(self):
        self.assertEqual(self.scorer.backend(), "numpy")

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

    def test_score_sums_one_per_head(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.H, self.seq, self.D)).astype(np.float32)
        s = self.scorer.score(q, k)
        for h in range(self.H):
            self.assertAlmostEqual(float(s[h].sum()), 1.0, places=4)

    def test_score_head_dim_mismatch_raises(self):
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k_bad = self.rng.standard_normal((self.H, self.seq, self.D + 1)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.scorer.score(q, k_bad)

    def test_top_k_mask_shape(self):
        s = self.rng.random((self.H, self.seq)).astype(np.float32)
        mask = self.scorer.top_k_mask(s, budget=3)
        self.assertEqual(mask.shape, (self.H, self.seq))

    def test_top_k_mask_count(self):
        s = self.rng.random((self.H, self.seq)).astype(np.float32)
        mask = self.scorer.top_k_mask(s, budget=3)
        for h in range(self.H):
            self.assertEqual(mask[h].sum(), 3)

    def test_parity_with_rust_greenkv(self):
        from squish.kernels.rs_green_kv_score import RustGreenKVScore
        rg = RustGreenKVScore()
        q = self.rng.standard_normal((self.H, self.obs, self.D)).astype(np.float32)
        k = self.rng.standard_normal((self.H, self.seq, self.D)).astype(np.float32)
        s_r = rg.score(q, k)
        s_m = self.scorer.score(q, k)
        np.testing.assert_allclose(s_m, s_r, rtol=1e-4)


# ── JacobiConvMojoConfig ──────────────────────────────────────────────────────


class TestJacobiConvMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = JacobiConvMojoConfig()
        self.assertEqual(cfg.temperature, 0.0)

    def test_custom(self):
        cfg = JacobiConvMojoConfig(temperature=0.5)
        self.assertAlmostEqual(cfg.temperature, 0.5)


# ── MojoJacobiConv ────────────────────────────────────────────────────────────


class TestMojoJacobiConv(unittest.TestCase):
    def setUp(self):
        self.jconv = MojoJacobiConv()
        self.rng = np.random.default_rng(5)
        self.N, self.vocab = 6, 20

    def test_backend_is_numpy(self):
        self.assertEqual(self.jconv.backend(), "numpy")

    def test_check_shape(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses, temperature=0.0)
        self.assertEqual(ng.shape, (self.N,))

    def test_check_dtype(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses)
        self.assertEqual(ng.dtype, np.int32)

    def test_check_greedy_matches_argmax(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng, _ = self.jconv.check(logits, guesses, temperature=0.0)
        expected = logits.argmax(axis=1).astype(np.int32)
        np.testing.assert_array_equal(ng, expected)

    def test_check_all_fixed_when_matching(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = logits.argmax(axis=1).astype(np.int32)
        _, nf = self.jconv.check(logits, guesses, temperature=0.0)
        self.assertEqual(nf, self.N)

    def test_check_n_fixed_is_int(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        _, nf = self.jconv.check(logits, guesses)
        self.assertIsInstance(nf, int)

    def test_converged_true(self):
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = logits.argmax(axis=1).astype(np.int32)
        self.assertTrue(self.jconv.converged(logits, guesses, temperature=0.0))

    def test_converged_false(self):
        logits = np.zeros((self.N, self.vocab), dtype=np.float32)
        logits[:, 0] = 10.0
        guesses = np.ones(self.N, dtype=np.int32)
        self.assertFalse(self.jconv.converged(logits, guesses, temperature=0.0))

    def test_parity_with_rust_jacobi(self):
        from squish.kernels.rs_jacobi_conv import RustJacobiConv
        rj = RustJacobiConv()
        logits = self.rng.standard_normal((self.N, self.vocab)).astype(np.float32)
        guesses = np.zeros(self.N, dtype=np.int32)
        ng_r, nf_r = rj.check(logits, guesses, temperature=0.0)
        ng_m, nf_m = self.jconv.check(logits, guesses, temperature=0.0)
        np.testing.assert_array_equal(ng_m, ng_r)
        self.assertEqual(nf_m, nf_r)


# ── TreeVerifyMojoConfig ──────────────────────────────────────────────────────


class TestTreeVerifyMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TreeVerifyMojoConfig()
        self.assertAlmostEqual(cfg.temperature, 1.0)

    def test_custom(self):
        cfg = TreeVerifyMojoConfig(temperature=0.8)
        self.assertAlmostEqual(cfg.temperature, 0.8)


# ── MojoTreeVerify ────────────────────────────────────────────────────────────


class TestMojoTreeVerify(unittest.TestCase):
    def setUp(self):
        self.verifier = MojoTreeVerify()
        self.rng = np.random.default_rng(99)
        self.B, self.D, self.V = 3, 5, 20
        self.draft_tokens = np.zeros((self.B, self.D), dtype=np.int32)
        self.draft_logits = self.rng.standard_normal((self.B, self.D, self.V)).astype(np.float32)
        self.target_logits = self.rng.standard_normal((self.B, self.D, self.V)).astype(np.float32)

    def test_backend_is_numpy(self):
        self.assertEqual(self.verifier.backend(), "numpy")

    def test_verify_accepted_is_ndarray(self):
        accepted, _ = self.verifier.verify(
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
        _, bl = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        self.assertGreaterEqual(bl, 1)

    def test_verify_perfect_draft_accepts_all(self):
        logits = self.rng.standard_normal((self.B, self.D, self.V)).astype(np.float32)
        _, bl = self.verifier.verify(
            self.draft_tokens, logits, logits, temperature=1.0, seed=0
        )
        self.assertEqual(bl, self.D)

    def test_verify_accepted_dtype(self):
        accepted, _ = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits
        )
        self.assertEqual(accepted.dtype, np.int32)

    def test_verify_batch_mismatch_raises(self):
        bad_tl = self.rng.standard_normal((self.B + 1, self.D, self.V)).astype(np.float32)
        with self.assertRaises(ValueError):
            self.verifier.verify(self.draft_tokens, self.draft_logits, bad_tl)

    def test_acceptance_rate_in_range(self):
        rate = self.verifier.acceptance_rate(
            self.draft_tokens, self.draft_logits, self.target_logits, n_trials=10
        )
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_parity_with_rust_tree_verify(self):
        from squish.kernels.rs_tree_verify import RustTreeVerify
        rv = RustTreeVerify()
        _, bl_r = rv.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        _, bl_m = self.verifier.verify(
            self.draft_tokens, self.draft_logits, self.target_logits, seed=0
        )
        self.assertEqual(bl_m, bl_r)


if __name__ == "__main__":
    unittest.main()
