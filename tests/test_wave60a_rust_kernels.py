"""tests/test_wave60a_rust_kernels.py — Wave 60a Rust kernel tests.

Tests for:
  - RustMamba2SSM     (rs_mamba2_ssm.py)
  - RustAdaRound      (rs_adaround.py)
  - RustPagedKVGather (rs_paged_kv.py)
  - RustHawkRGLR      (rs_hawk_rglr.py)
  - RustCakeEntropy   (rs_cake_entropy.py)
  - RustTernaryGEMV   (rs_ternary_gemv.py)

All tests use NumPy fallback path (Rust crate not compiled in CI).
77 tests total.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from squish.kernels.rs_mamba2_ssm import Mamba2ScanConfig, RustMamba2SSM
from squish.kernels.rs_adaround import AdaRoundConfig, RustAdaRound
from squish.kernels.rs_paged_kv import PagedKVConfig, RustPagedKVGather
from squish.kernels.rs_hawk_rglr import HawkRGLRConfig, RustHawkRGLR
from squish.kernels.rs_cake_entropy import CakeEntropyConfig, RustCakeEntropy
from squish.kernels.rs_ternary_gemv import TernaryGEMVConfig, RustTernaryGEMV


# ── Mamba2ScanConfig ──────────────────────────────────────────────────────────


class TestMamba2ScanConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = Mamba2ScanConfig()
        self.assertEqual(cfg.d_state, 64)
        self.assertEqual(cfg.d_model, 512)

    def test_custom(self):
        cfg = Mamba2ScanConfig(d_state=32, d_model=256)
        self.assertEqual(cfg.d_state, 32)


# ── RustMamba2SSM ─────────────────────────────────────────────────────────────


class TestRustMamba2SSM(unittest.TestCase):
    def setUp(self):
        self.ssm = RustMamba2SSM(Mamba2ScanConfig(d_state=8, d_model=16))
        self.rng = np.random.default_rng(0)
        self.T = 10
        self.d = 8

    def _make_abcx(self):
        a = -self.rng.random(self.T).astype(np.float32) * 0.1  # slightly neg log-A
        b = self.rng.standard_normal((self.T, self.d)).astype(np.float32)
        c = self.rng.standard_normal((self.T, self.d)).astype(np.float32)
        x = self.rng.standard_normal(self.T).astype(np.float32)
        return a, b, c, x

    def test_backend_is_string(self):
        self.assertIn(self.ssm.backend(), ("rust", "numpy"))

    def test_scan_output_shape(self):
        a, b, c, x = self._make_abcx()
        out, fs = self.ssm.scan(a, b, c, x)
        self.assertEqual(out.shape, (self.T,))

    def test_scan_state_shape(self):
        a, b, c, x = self._make_abcx()
        _, fs = self.ssm.scan(a, b, c, x)
        self.assertEqual(fs.shape, (self.d,))

    def test_scan_output_finite(self):
        a, b, c, x = self._make_abcx()
        out, _ = self.ssm.scan(a, b, c, x)
        self.assertTrue(np.isfinite(out).all())

    def test_scan_zero_input_zero_output_from_zero_state(self):
        a = np.full(self.T, -1.0, dtype=np.float32)
        b = np.zeros((self.T, self.d), dtype=np.float32)
        c = np.ones((self.T, self.d), dtype=np.float32)
        x = np.zeros(self.T, dtype=np.float32)
        out, fs = self.ssm.scan(a, b, c, x)
        np.testing.assert_allclose(out, 0.0, atol=1e-6)
        np.testing.assert_allclose(fs, 0.0, atol=1e-6)

    def test_scan_with_nonzero_h0(self):
        a, b, c, x = self._make_abcx()
        h0 = np.ones(self.d, dtype=np.float32)
        out1, _ = self.ssm.scan(a, b, c, x, h0=h0)
        out2, _ = self.ssm.scan(a, b, c, x)
        # Different h0 → different output
        self.assertFalse(np.allclose(out1, out2))

    def test_decode_step_output_scalar(self):
        b_vec = self.rng.standard_normal(self.d).astype(np.float32)
        c_vec = self.rng.standard_normal(self.d).astype(np.float32)
        state = np.zeros(self.d, dtype=np.float32)
        y, ns = self.ssm.decode_step(1.0, b_vec, c_vec, 0.5, state)
        self.assertIsInstance(float(y), float)

    def test_decode_step_state_shape(self):
        b_vec = self.rng.standard_normal(self.d).astype(np.float32)
        c_vec = self.rng.standard_normal(self.d).astype(np.float32)
        state = np.zeros(self.d, dtype=np.float32)
        _, ns = self.ssm.decode_step(1.0, b_vec, c_vec, 0.5, state)
        self.assertEqual(ns.shape, (self.d,))

    def test_decode_matches_scan_first_step(self):
        a, b, c, x = self._make_abcx()
        out_scan, _ = self.ssm.scan(a, b, c, x)
        a0 = float(np.exp(a[0]))
        y_dec, _ = self.ssm.decode_step(a0, b[0], c[0], float(x[0]), np.zeros(self.d))
        self.assertAlmostEqual(float(out_scan[0]), y_dec, places=5)

    def test_d_state_property(self):
        self.assertEqual(self.ssm.d_state(), 8)

    def test_scan_dtype(self):
        a, b, c, x = self._make_abcx()
        out, fs = self.ssm.scan(a, b, c, x)
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(fs.dtype, np.float32)


# ── AdaRoundConfig ────────────────────────────────────────────────────────────


class TestAdaRoundConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AdaRoundConfig()
        self.assertAlmostEqual(cfg.lr, 1e-3)
        self.assertAlmostEqual(cfg.lambda_reg, 0.01)

    def test_custom(self):
        cfg = AdaRoundConfig(lr=1e-4, lambda_reg=0.1)
        self.assertAlmostEqual(cfg.lr, 1e-4)


# ── RustAdaRound ──────────────────────────────────────────────────────────────


class TestRustAdaRound(unittest.TestCase):
    def setUp(self):
        self.ar = RustAdaRound(AdaRoundConfig(lr=0.01))
        self.rng = np.random.default_rng(1)
        self.N = 64

    def _make_inputs(self):
        v = self.rng.standard_normal(self.N).astype(np.float32)
        w = self.rng.standard_normal(self.N).astype(np.float32)
        wf = np.floor(w).astype(np.float32)
        qs = np.full(self.N, 0.1, dtype=np.float32)
        return v, w, wf, qs

    def test_backend_is_string(self):
        self.assertIn(self.ar.backend(), ("rust", "numpy"))

    def test_output_shape(self):
        v, w, wf, qs = self._make_inputs()
        v_new = self.ar.step(v, w, wf, qs, beta=1.0)
        self.assertEqual(v_new.shape, (self.N,))

    def test_output_dtype(self):
        v, w, wf, qs = self._make_inputs()
        v_new = self.ar.step(v, w, wf, qs, beta=1.0)
        self.assertEqual(v_new.dtype, np.float32)

    def test_output_finite(self):
        v, w, wf, qs = self._make_inputs()
        v_new = self.ar.step(v, w, wf, qs, beta=1.0)
        self.assertTrue(np.isfinite(v_new).all())

    def test_lr_override(self):
        v, w, wf, qs = self._make_inputs()
        v1 = self.ar.step(v, w, wf, qs, beta=1.0, lr=0.0)
        # lr=0 → V unchanged
        np.testing.assert_allclose(v1, v, atol=1e-5)

    def test_high_beta_converges_to_binary(self):
        # With very high beta, h(V) saturates to 0 or 1
        v = np.array([5.0, -5.0, 0.0], dtype=np.float32)
        w = np.zeros(3, dtype=np.float32)
        wf = np.zeros(3, dtype=np.float32)
        qs = np.ones(3, dtype=np.float32)
        for _ in range(10):
            v = self.ar.step(v, w, wf, qs, beta=100.0, lr=0.0001)
        self.assertTrue(np.isfinite(v).all())

    def test_shape_mismatch_raises(self):
        v, w, wf, qs = self._make_inputs()
        with self.assertRaises(ValueError):
            self.ar.step(v[:-1], w, wf, qs, beta=1.0)

    def test_lr_property(self):
        self.assertAlmostEqual(self.ar.lr(), 0.01)

    def test_lambda_reg_property(self):
        self.assertAlmostEqual(self.ar.lambda_reg(), 0.01)


# ── PagedKVConfig ─────────────────────────────────────────────────────────────


class TestPagedKVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = PagedKVConfig()
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.block_size, 16)

    def test_custom(self):
        cfg = PagedKVConfig(n_heads=4, block_size=8, head_dim=64)
        self.assertEqual(cfg.head_dim, 64)


# ── RustPagedKVGather ─────────────────────────────────────────────────────────


class TestRustPagedKVGather(unittest.TestCase):
    def setUp(self):
        self.gather = RustPagedKVGather(PagedKVConfig(n_heads=2, block_size=4, head_dim=8))
        self.rng = np.random.default_rng(2)

    def _make_pool(self, max_blocks=8):
        return self.rng.standard_normal(
            (max_blocks, 2, 4, 8)
        ).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.gather.backend(), ("rust", "numpy"))

    def test_gather_shape(self):
        pool = self._make_pool()
        page_table = np.array([0, 1, 2], dtype=np.int32)
        out = self.gather.gather(pool, page_table, n_valid_tokens=8)
        self.assertEqual(out.shape, (8, 2, 8))

    def test_gather_dtype(self):
        pool = self._make_pool()
        pt = np.array([0], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=4)
        self.assertEqual(out.dtype, np.float32)

    def test_gather_correct_values(self):
        # Page 0, block_size=4, first token → pool_4d[0, :, 0, :]
        pool = self._make_pool()
        pt = np.array([0], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=1)
        np.testing.assert_allclose(out[0], pool[0, :, 0, :], atol=1e-6)

    def test_gather_second_page(self):
        pool = self._make_pool()
        pt = np.array([3, 5], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=5)
        # token 4 is in page 5 (4//4=1), position 0
        np.testing.assert_allclose(out[4], pool[5, :, 0, :], atol=1e-6)

    def test_block_size_property(self):
        self.assertEqual(self.gather.block_size(), 4)

    def test_n_heads_property(self):
        self.assertEqual(self.gather.n_heads(), 2)

    def test_single_token(self):
        pool = self._make_pool()
        pt = np.array([2], dtype=np.int32)
        out = self.gather.gather(pool, pt, n_valid_tokens=1)
        self.assertEqual(out.shape, (1, 2, 8))


# ── HawkRGLRConfig ────────────────────────────────────────────────────────────


class TestHawkRGLRConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = HawkRGLRConfig()
        self.assertEqual(cfg.d_state, 512)

    def test_custom(self):
        cfg = HawkRGLRConfig(d_state=16, d_model=64)
        self.assertEqual(cfg.d_state, 16)


# ── RustHawkRGLR ──────────────────────────────────────────────────────────────


class TestRustHawkRGLR(unittest.TestCase):
    def setUp(self):
        self.hawk = RustHawkRGLR(HawkRGLRConfig(d_state=8, d_model=8))
        self.rng = np.random.default_rng(3)
        self.T = 6
        self.d = 8

    def _make_inputs(self):
        x = self.rng.standard_normal((self.T, self.d)).astype(np.float32)
        dt = self.rng.standard_normal((self.T, self.d)).astype(np.float32) * 0.1
        lam = -self.rng.random(self.d).astype(np.float32)  # negative log-eig
        return x, dt, lam

    def test_backend_is_string(self):
        self.assertIn(self.hawk.backend(), ("rust", "numpy"))

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

    def test_zero_input_decays_to_zero(self):
        x = np.zeros((self.T, self.d), dtype=np.float32)
        dt = np.full((self.T, self.d), -5.0, dtype=np.float32)
        lam = np.ones(self.d, dtype=np.float32) * 2.0  # strong decay
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
        self.assertEqual(fs.dtype, np.float32)

    def test_d_state_property(self):
        self.assertEqual(self.hawk.d_state(), 8)


# ── CakeEntropyConfig ─────────────────────────────────────────────────────────


class TestCakeEntropyConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CakeEntropyConfig()
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.obs_window, 4)

    def test_custom(self):
        cfg = CakeEntropyConfig(n_heads=4, obs_window=2, temperature=0.5)
        self.assertAlmostEqual(cfg.temperature, 0.5)


# ── RustCakeEntropy ───────────────────────────────────────────────────────────


class TestRustCakeEntropy(unittest.TestCase):
    def setUp(self):
        self.ce = RustCakeEntropy(
            CakeEntropyConfig(n_heads=4, head_dim=8, obs_window=2, temperature=1.0)
        )
        self.rng = np.random.default_rng(4)

    def _make_qk(self, T=8):
        q = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        k = self.rng.standard_normal((T, 4, 8)).astype(np.float32)
        return q, k

    def test_backend_is_string(self):
        self.assertIn(self.ce.backend(), ("rust", "numpy"))

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

    def test_uniform_attention_high_entropy(self):
        # Uniform K → roughly equal attention weights → high entropy
        q = np.ones((2, 4, 8), dtype=np.float32)
        k = np.ones((8, 4, 8), dtype=np.float32)
        ent = self.ce.compute(q, k)
        self.assertTrue((ent > 0).all())

    def test_temperature_override(self):
        q, k = self._make_qk()
        ent_low = self.ce.compute(q, k, temperature=0.1)
        ent_high = self.ce.compute(q, k, temperature=10.0)
        # Higher temperature → more uniform → higher entropy
        self.assertTrue(float(ent_high.mean()) >= float(ent_low.mean()) - 1e-3)

    def test_n_heads_property(self):
        self.assertEqual(self.ce.n_heads(), 4)

    def test_obs_window_property(self):
        self.assertEqual(self.ce.obs_window(), 2)


# ── TernaryGEMVConfig ─────────────────────────────────────────────────────────


class TestTernaryGEMVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TernaryGEMVConfig()
        self.assertAlmostEqual(cfg.scale, 1.0)

    def test_custom(self):
        cfg = TernaryGEMVConfig(scale=0.5)
        self.assertAlmostEqual(cfg.scale, 0.5)


# ── RustTernaryGEMV ───────────────────────────────────────────────────────────


class TestRustTernaryGEMV(unittest.TestCase):
    def setUp(self):
        self.gemv = RustTernaryGEMV(TernaryGEMVConfig(scale=1.0))
        self.rng = np.random.default_rng(5)

    def _make_ternary(self, out_f=8, in_f=16):
        choices = np.array([-1, 0, 1], dtype=np.int8)
        w = self.rng.choice(choices, size=(out_f, in_f)).astype(np.int8)
        a = self.rng.standard_normal(in_f).astype(np.float32)
        return w, a

    def test_backend_is_string(self):
        self.assertIn(self.gemv.backend(), ("rust", "numpy"))

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
        out_ternary = self.gemv.gemv(w, a)
        out_dense = (w.astype(np.float32) @ a)
        np.testing.assert_allclose(out_ternary, out_dense, atol=1e-5)

    def test_all_zero_weights_zero_output(self):
        w = np.zeros((8, 16), dtype=np.int8)
        a = self.rng.standard_normal(16).astype(np.float32)
        out = self.gemv.gemv(w, a)
        np.testing.assert_allclose(out, 0.0, atol=1e-7)

    def test_scale_applied(self):
        w, a = self._make_ternary()
        out1 = self.gemv.gemv(w, a, scale=1.0)
        out2 = self.gemv.gemv(w, a, scale=2.0)
        np.testing.assert_allclose(out2, out1 * 2, atol=1e-5)

    def test_shape_mismatch_raises(self):
        w, a = self._make_ternary()
        with self.assertRaises(ValueError):
            self.gemv.gemv(w, a[:-1])

    def test_sparsity_all_zero(self):
        w = np.zeros((8, 16), dtype=np.int8)
        sp = self.gemv.sparsity(w)
        self.assertAlmostEqual(sp, 1.0, places=6)

    def test_sparsity_no_zero(self):
        w = np.ones((8, 16), dtype=np.int8)
        sp = self.gemv.sparsity(w)
        self.assertAlmostEqual(sp, 0.0, places=6)

    def test_scale_property(self):
        self.assertAlmostEqual(self.gemv.scale(), 1.0)


if __name__ == "__main__":
    unittest.main()
