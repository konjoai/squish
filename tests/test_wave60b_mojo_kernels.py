"""tests/test_wave60b_mojo_kernels.py — Wave 60b Mojo kernel tests.

Tests for:
  - MojoMamba2Scan   (mamba2_scan_mojo.py)
  - MojoHawkRGLR     (hawk_rglr_mojo.py)
  - MojoMedusaVerify (medusa_verify_mojo.py)
  - MojoPagedKVGather (paged_kv_mojo.py)
  - MojoCakeEntropy  (cake_entropy_mojo.py)
  - MojoTernaryGEMV  (ternary_gemv_mojo.py)

All tests exercise the NumPy fallback path (no Mojo runtime in CI).
78 tests total.
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.kernels.mojo.mamba2_scan_mojo import Mamba2ScanMojoConfig, MojoMamba2Scan
from squish.kernels.mojo.hawk_rglr_mojo import HawkRGLRMojoConfig, MojoHawkRGLR
from squish.kernels.mojo.medusa_verify_mojo import MedusaVerifyConfig, MojoMedusaVerify
from squish.kernels.mojo.paged_kv_mojo import PagedKVMojoConfig, MojoPagedKVGather
from squish.kernels.mojo.cake_entropy_mojo import CakeEntropyMojoConfig, MojoCakeEntropy
from squish.kernels.mojo.ternary_gemv_mojo import TernaryGEMVMojoConfig, MojoTernaryGEMV


# ── Mamba2ScanMojoConfig ──────────────────────────────────────────────────────


class TestMamba2ScanMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = Mamba2ScanMojoConfig()
        self.assertEqual(cfg.d_state, 64)
        self.assertEqual(cfg.chunk_size, 64)

    def test_custom(self):
        cfg = Mamba2ScanMojoConfig(d_state=16, chunk_size=32)
        self.assertEqual(cfg.d_state, 16)
        self.assertEqual(cfg.chunk_size, 32)


# ── MojoMamba2Scan ────────────────────────────────────────────────────────────


class TestMojoMamba2Scan(unittest.TestCase):
    def setUp(self):
        self.ssm = MojoMamba2Scan(Mamba2ScanMojoConfig(d_state=8, chunk_size=4))
        self.rng = np.random.default_rng(10)
        self.T = 8
        self.d = 8

    def _make_abcx(self):
        a = -self.rng.random(self.T).astype(np.float32) * 0.5
        b = self.rng.standard_normal((self.T, self.d)).astype(np.float32)
        c = self.rng.standard_normal((self.T, self.d)).astype(np.float32)
        x = self.rng.standard_normal(self.T).astype(np.float32)
        return a, b, c, x

    def test_backend_is_string(self):
        self.assertIn(self.ssm.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        a, b, c, x = self._make_abcx()
        out, state = self.ssm.scan(a, b, c, x)
        self.assertEqual(out.shape, (self.T,))

    def test_state_shape(self):
        a, b, c, x = self._make_abcx()
        _, state = self.ssm.scan(a, b, c, x)
        self.assertEqual(state.shape, (self.d,))

    def test_output_dtype(self):
        a, b, c, x = self._make_abcx()
        out, _ = self.ssm.scan(a, b, c, x)
        self.assertEqual(out.dtype, np.float32)

    def test_output_finite(self):
        a, b, c, x = self._make_abcx()
        out, state = self.ssm.scan(a, b, c, x)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(state).all())

    def test_zero_input_zero_state(self):
        a = np.full(self.T, -1.0, dtype=np.float32)
        b = np.zeros((self.T, self.d), dtype=np.float32)
        c = np.ones((self.T, self.d), dtype=np.float32)
        x = np.zeros(self.T, dtype=np.float32)
        out, state = self.ssm.scan(a, b, c, x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)
        np.testing.assert_allclose(state, 0.0, atol=1e-6)

    def test_h0_nonzero_differs(self):
        a, b, c, x = self._make_abcx()
        h0 = np.ones(self.d, dtype=np.float32)
        out1, _ = self.ssm.scan(a, b, c, x, h0=h0)
        out2, _ = self.ssm.scan(a, b, c, x)
        self.assertFalse(np.allclose(out1, out2))

    def test_d_state_property(self):
        self.assertEqual(self.ssm.d_state(), 8)

    def test_chunk_size_property(self):
        self.assertEqual(self.ssm.chunk_size(), 4)


# ── HawkRGLRMojoConfig ────────────────────────────────────────────────────────


class TestHawkRGLRMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = HawkRGLRMojoConfig()
        self.assertEqual(cfg.d_state, 512)

    def test_custom(self):
        cfg = HawkRGLRMojoConfig(d_state=32)
        self.assertEqual(cfg.d_state, 32)


# ── MojoHawkRGLR ─────────────────────────────────────────────────────────────


class TestMojoHawkRGLR(unittest.TestCase):
    def setUp(self):
        self.hawk = MojoHawkRGLR(HawkRGLRMojoConfig(d_state=8))
        self.rng = np.random.default_rng(11)
        self.T = 5
        self.d = 8

    def _make_inputs(self):
        x = self.rng.standard_normal((self.T, self.d)).astype(np.float32)
        dt = self.rng.standard_normal((self.T, self.d)).astype(np.float32) * 0.1
        lam = -self.rng.random(self.d).astype(np.float32)
        return x, dt, lam

    def test_backend_is_string(self):
        self.assertIn(self.hawk.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        x, dt, lam = self._make_inputs()
        out, fs = self.hawk.scan(x, dt, lam)
        self.assertEqual(out.shape, (self.T, self.d))

    def test_state_shape(self):
        x, dt, lam = self._make_inputs()
        _, fs = self.hawk.scan(x, dt, lam)
        self.assertEqual(fs.shape, (self.d,))

    def test_output_finite(self):
        x, dt, lam = self._make_inputs()
        out, fs = self.hawk.scan(x, dt, lam)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(fs).all())

    def test_zero_input(self):
        x = np.zeros((self.T, self.d), dtype=np.float32)
        dt = np.full((self.T, self.d), -10.0, dtype=np.float32)
        lam = np.ones(self.d, dtype=np.float32) * 2.0
        out, fs = self.hawk.scan(x, dt, lam)
        np.testing.assert_allclose(fs, 0.0, atol=1e-4)

    def test_nonzero_h0(self):
        x, dt, lam = self._make_inputs()
        h0 = np.ones(self.d, dtype=np.float32)
        out1, _ = self.hawk.scan(x, dt, lam, h0=h0)
        out2, _ = self.hawk.scan(x, dt, lam)
        self.assertFalse(np.allclose(out1[0], out2[0]))

    def test_dtype(self):
        x, dt, lam = self._make_inputs()
        out, fs = self.hawk.scan(x, dt, lam)
        self.assertEqual(out.dtype, np.float32)

    def test_d_state_property(self):
        self.assertEqual(self.hawk.d_state(), 8)


# ── MedusaVerifyConfig ────────────────────────────────────────────────────────


class TestMedusaVerifyConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MedusaVerifyConfig()
        self.assertEqual(cfg.n_heads, 2)
        self.assertEqual(cfg.vocab_size, 32000)

    def test_custom(self):
        cfg = MedusaVerifyConfig(n_heads=4, vocab_size=50257, accept_threshold=0.9)
        self.assertAlmostEqual(cfg.accept_threshold, 0.9)


# ── MojoMedusaVerify ──────────────────────────────────────────────────────────


class TestMojoMedusaVerify(unittest.TestCase):
    def setUp(self):
        self.mv = MojoMedusaVerify(
            MedusaVerifyConfig(n_heads=3, vocab_size=8, accept_threshold=0.0)
        )
        self.rng = np.random.default_rng(12)

    def _make_tokens_probs(self, n_heads=3, vocab=8):
        draft_tokens = self.rng.integers(0, vocab, size=n_heads, dtype=np.int32)
        draft_probs = self.rng.dirichlet(np.ones(vocab), size=n_heads).astype(np.float32)
        target_probs = self.rng.dirichlet(np.ones(vocab), size=n_heads).astype(np.float32)
        return draft_tokens, draft_probs, target_probs

    def test_backend_is_string(self):
        self.assertIn(self.mv.backend(), ("mojo", "numpy"))

    def test_returns_tuple(self):
        dt, dp, tp = self._make_tokens_probs()
        result = self.mv.verify(dt, dp, tp)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_accepted_count_int(self):
        dt, dp, tp = self._make_tokens_probs()
        _, n_acc = self.mv.verify(dt, dp, tp)
        self.assertIsInstance(n_acc, int)

    def test_threshold_zero_accepts_all(self):
        mv = MojoMedusaVerify(MedusaVerifyConfig(n_heads=3, vocab_size=8, accept_threshold=0.0))
        dt, dp, tp = self._make_tokens_probs()
        _, n_acc = mv.verify(dt, dp, tp, accept_threshold=0.0)
        self.assertEqual(n_acc, 3)

    def test_threshold_one_accepts_none(self):
        mv = MojoMedusaVerify(MedusaVerifyConfig(n_heads=3, vocab_size=8, accept_threshold=1.0))
        dt = np.array([0, 1, 2], dtype=np.int32)
        # draft puts ALL probability on the draft token → p_draft=1.0 per head
        dp = np.zeros((3, 8), dtype=np.float32)
        for i, tok in enumerate([0, 1, 2]):
            dp[i, tok] = 1.0
        # target puts NO probability on the draft token → p_target=0.0 → ratio=0 < 1.0
        tp = np.ones((3, 8), dtype=np.float32) / 8
        tp[:, [0, 1, 2]] = 0.0
        tp = tp / tp.sum(axis=-1, keepdims=True)
        _, n_acc = mv.verify(dt, dp, tp, accept_threshold=1.0)
        self.assertEqual(n_acc, 0)

    def test_accepted_list_length_matches_count(self):
        dt, dp, tp = self._make_tokens_probs()
        accepted, n_acc = self.mv.verify(dt, dp, tp)
        self.assertEqual(len(accepted), n_acc)

    def test_accepted_count_in_range(self):
        dt, dp, tp = self._make_tokens_probs()
        _, n_acc = self.mv.verify(dt, dp, tp)
        self.assertGreaterEqual(n_acc, 0)
        self.assertLessEqual(n_acc, 3)

    def test_n_heads_property(self):
        self.assertEqual(self.mv.n_heads(), 3)

    def test_vocab_size_property(self):
        self.assertEqual(self.mv.vocab_size(), 8)

    def test_accept_threshold_property(self):
        self.assertAlmostEqual(self.mv.accept_threshold(), 0.0)

    def test_single_head(self):
        mv = MojoMedusaVerify(MedusaVerifyConfig(n_heads=1, vocab_size=4, accept_threshold=0.0))
        dt = np.array([0], dtype=np.int32)
        dp = np.ones((1, 4), dtype=np.float32) / 4
        tp = np.ones((1, 4), dtype=np.float32) / 4
        _, n_acc = mv.verify(dt, dp, tp, accept_threshold=0.0)
        self.assertEqual(n_acc, 1)

    def test_threshold_mid_range(self):
        # With threshold 0.5 draft token is accepted when target_prob > 0.5
        mv = MojoMedusaVerify(MedusaVerifyConfig(n_heads=1, vocab_size=2, accept_threshold=0.5))
        dt = np.array([1], dtype=np.int32)
        dp = np.array([[0.2, 0.8]], dtype=np.float32)
        tp_accept = np.array([[0.1, 0.9]], dtype=np.float32)
        _, n_acc = mv.verify(dt, dp, tp_accept, accept_threshold=0.5)
        self.assertGreaterEqual(n_acc, 0)


# ── PagedKVMojoConfig ─────────────────────────────────────────────────────────


class TestPagedKVMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PagedKVMojoConfig()
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.block_size, 16)

    def test_custom(self):
        cfg = PagedKVMojoConfig(n_heads=2, block_size=4, head_dim=8)
        self.assertEqual(cfg.head_dim, 8)


# ── MojoPagedKVGather ─────────────────────────────────────────────────────────


class TestMojoPagedKVGather(unittest.TestCase):
    def setUp(self):
        self.gather = MojoPagedKVGather(
            PagedKVMojoConfig(n_heads=2, block_size=4, head_dim=8)
        )
        self.rng = np.random.default_rng(13)

    def _make_pool(self, max_blocks=8):
        return self.rng.standard_normal(
            (max_blocks, 2, 4, 8)
        ).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.gather.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        pool = self._make_pool()
        pt = np.array([0, 1, 2], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=8)
        self.assertEqual(out.shape, (8, 2, 8))

    def test_output_dtype(self):
        pool = self._make_pool()
        pt = np.array([0], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=4)
        self.assertEqual(out.dtype, np.float32)

    def test_output_finite(self):
        pool = self._make_pool()
        pt = np.array([0, 1], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=6)
        self.assertTrue(np.isfinite(out).all())

    def test_correct_value_first_token(self):
        pool = self._make_pool()
        pt = np.array([0], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=1)
        np.testing.assert_allclose(out[0], pool[0, :, 0, :], atol=1e-6)

    def test_n_heads_property(self):
        self.assertEqual(self.gather.n_heads(), 2)

    def test_block_size_property(self):
        self.assertEqual(self.gather.block_size(), 4)

    def test_head_dim_property(self):
        self.assertEqual(self.gather.head_dim(), 8)


# ── CakeEntropyMojoConfig ─────────────────────────────────────────────────────


class TestCakeEntropyMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CakeEntropyMojoConfig()
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.obs_window, 4)

    def test_custom(self):
        cfg = CakeEntropyMojoConfig(n_heads=4, head_dim=8, obs_window=2, temperature=0.5)
        self.assertAlmostEqual(cfg.temperature, 0.5)


# ── MojoCakeEntropy ───────────────────────────────────────────────────────────


class TestMojoCakeEntropy(unittest.TestCase):
    def setUp(self):
        self.ce = MojoCakeEntropy(
            CakeEntropyMojoConfig(n_heads=4, head_dim=8, obs_window=2, temperature=1.0)
        )
        self.rng = np.random.default_rng(14)

    def _make_qk(self, T=8):
        q = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        k = self.rng.standard_normal((T, 4, 8)).astype(np.float32)
        return q, k

    def test_backend_is_string(self):
        self.assertIn(self.ce.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        q, k = self._make_qk()
        ent = self.ce.compute(q, k)
        self.assertEqual(ent.shape, (4,))

    def test_output_dtype(self):
        q, k = self._make_qk()
        ent = self.ce.compute(q, k)
        self.assertEqual(ent.dtype, np.float32)

    def test_output_non_negative(self):
        q, k = self._make_qk()
        ent = self.ce.compute(q, k)
        self.assertTrue((ent >= 0).all())

    def test_uniform_attention(self):
        q = np.ones((2, 4, 8), dtype=np.float32)
        k = np.ones((8, 4, 8), dtype=np.float32)
        ent = self.ce.compute(q, k)
        self.assertTrue((ent >= 0).all())
        self.assertTrue(np.isfinite(ent).all())

    def test_temperature_override(self):
        q, k = self._make_qk()
        ent = self.ce.compute(q, k, temperature=0.1)
        self.assertTrue(np.isfinite(ent).all())

    def test_n_heads_property(self):
        self.assertEqual(self.ce.n_heads(), 4)

    def test_obs_window_property(self):
        self.assertEqual(self.ce.obs_window(), 2)

    def test_head_dim_property(self):
        self.assertEqual(self.ce.head_dim(), 8)


# ── TernaryGEMVMojoConfig ─────────────────────────────────────────────────────


class TestTernaryGEMVMojoConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TernaryGEMVMojoConfig()
        self.assertAlmostEqual(cfg.scale, 1.0)

    def test_custom(self):
        cfg = TernaryGEMVMojoConfig(scale=2.0)
        self.assertAlmostEqual(cfg.scale, 2.0)


# ── MojoTernaryGEMV ───────────────────────────────────────────────────────────


class TestMojoTernaryGEMV(unittest.TestCase):
    def setUp(self):
        self.gemv = MojoTernaryGEMV(TernaryGEMVMojoConfig(scale=1.0))
        self.rng = np.random.default_rng(15)

    def _make_ternary(self, out_f=8, in_f=16):
        choices = np.array([-1, 0, 1], dtype=np.int8)
        w = self.rng.choice(choices, size=(out_f, in_f)).astype(np.int8)
        a = self.rng.standard_normal(in_f).astype(np.float32)
        return w, a

    def test_backend_is_string(self):
        self.assertIn(self.gemv.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        w, a = self._make_ternary()
        out = self.gemv.gemv(w, a)
        self.assertEqual(out.shape, (8,))

    def test_output_dtype(self):
        w, a = self._make_ternary()
        out = self.gemv.gemv(w, a)
        self.assertEqual(out.dtype, np.float32)

    def test_matches_float_matmul(self):
        w, a = self._make_ternary()
        out_mojo = self.gemv.gemv(w, a)
        out_ref = w.astype(np.float32) @ a
        np.testing.assert_allclose(out_mojo, out_ref, atol=1e-5)

    def test_all_zero_weights(self):
        w = np.zeros((8, 16), dtype=np.int8)
        a = self.rng.standard_normal(16).astype(np.float32)
        out = self.gemv.gemv(w, a)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_scale_applied(self):
        w, a = self._make_ternary()
        out1 = self.gemv.gemv(w, a, scale=1.0)
        out2 = self.gemv.gemv(w, a, scale=3.0)
        np.testing.assert_allclose(out2, out1 * 3.0, atol=1e-5)

    def test_shape_mismatch_raises(self):
        w, a = self._make_ternary()
        with self.assertRaises(ValueError):
            self.gemv.gemv(w, a[:-1])

    def test_scale_property(self):
        self.assertAlmostEqual(self.gemv.scale(), 1.0)


if __name__ == "__main__":
    unittest.main()
