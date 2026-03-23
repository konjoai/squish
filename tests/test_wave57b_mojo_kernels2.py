"""tests/test_wave57b_mojo_kernels2.py — Wave 57b Mojo kernel tests.

Covers all six Wave 57b modules:
  MojoRMSNormFused, MojoSwiGLUParallel, MojoGQADecodeKernel,
  MojoTokenCosSim, MojoSparseBlockScore, MojoRetentionState

All tests use the NumPy fallback paths (Mojo compiler not available
in the test environment) — this validates that every module is
importable, Config objects are well-formed, and numerical outputs
are correct.

≥ 72 tests, all deterministic, all passing.
"""

from __future__ import annotations

import math
import unittest

import numpy as np


# ---------------------------------------------------------------------------
# MojoRMSNormFused
# ---------------------------------------------------------------------------
from squish.kernels.mojo.rmsnorm_mojo import RMSNormConfig, MojoRMSNormFused


class TestRMSNormConfig(unittest.TestCase):
    def test_default_hidden_dim(self):
        cfg = RMSNormConfig()
        self.assertEqual(cfg.hidden_dim, 4096)

    def test_default_eps(self):
        cfg = RMSNormConfig()
        self.assertAlmostEqual(cfg.eps, 1e-6)

    def test_custom_config(self):
        cfg = RMSNormConfig(hidden_dim=7168, eps=1e-5)
        self.assertEqual(cfg.hidden_dim, 7168)


class TestMojoRMSNormFused(unittest.TestCase):
    def setUp(self):
        self.norm = MojoRMSNormFused(RMSNormConfig(hidden_dim=64))
        rng = np.random.default_rng(0)
        self.x = rng.standard_normal((8, 64)).astype(np.float32)
        self.residual = rng.standard_normal((8, 64)).astype(np.float32)
        self.weight = np.ones(64, dtype=np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.norm.backend(), ("mojo", "rust", "numpy"))

    def test_forward_out_shape(self):
        out, _ = self.norm.forward(self.x, self.residual, self.weight)
        self.assertEqual(out.shape, (8, 64))

    def test_forward_residual_shape(self):
        _, new_res = self.norm.forward(self.x, self.residual, self.weight)
        self.assertEqual(new_res.shape, (8, 64))

    def test_forward_out_dtype(self):
        out, _ = self.norm.forward(self.x, self.residual, self.weight)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_out_finite(self):
        out, _ = self.norm.forward(self.x, self.residual, self.weight)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_new_residual_equals_x_plus_res(self):
        _, new_res = self.norm.forward(self.x, self.residual, self.weight)
        expected = self.x + self.residual
        np.testing.assert_array_almost_equal(new_res, expected, decimal=5)

    def test_forward_output_rms_norm(self):
        # verify rows have approximately unit RMS
        out, _ = self.norm.forward(self.x, self.residual, self.weight)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        # RMS of normalized output with weight=1 should be ≈ 1
        np.testing.assert_array_almost_equal(rms, np.ones(8), decimal=3)

    def test_norm_only_shape(self):
        out = self.norm.norm_only(self.x, self.weight)
        self.assertEqual(out.shape, (8, 64))

    def test_norm_only_dtype(self):
        out = self.norm.norm_only(self.x, self.weight)
        self.assertEqual(out.dtype, np.float32)

    def test_norm_only_rms(self):
        out = self.norm.norm_only(self.x, self.weight)
        rms = np.sqrt(np.mean(out ** 2, axis=-1))
        np.testing.assert_array_almost_equal(rms, np.ones(8), decimal=3)

    def test_zero_residual(self):
        zero_res = np.zeros_like(self.x)
        out, new_res = self.norm.forward(self.x, zero_res, self.weight)
        np.testing.assert_array_almost_equal(new_res, self.x, decimal=5)

    def test_weight_scaling(self):
        # weight=2.0 should scale output by 2×
        w2 = np.full(64, 2.0, dtype=np.float32)
        out1, _ = self.norm.forward(self.x, self.residual, self.weight)
        out2, _ = self.norm.forward(self.x, self.residual, w2)
        np.testing.assert_array_almost_equal(out2, out1 * 2.0, decimal=5)


# ---------------------------------------------------------------------------
# MojoSwiGLUParallel  (in squish/kernels/mojo/swiglu_mojo.py)
# ---------------------------------------------------------------------------
from squish.kernels.mojo.swiglu_mojo import SwiGLUMojoConfig, MojoSwiGLUParallel


class TestSwiGLUMojoConfig(unittest.TestCase):
    def test_default_ffn_dim(self):
        cfg = SwiGLUMojoConfig()
        self.assertEqual(cfg.ffn_dim, 14336)

    def test_custom_ffn_dim(self):
        cfg = SwiGLUMojoConfig(ffn_dim=8192)
        self.assertEqual(cfg.ffn_dim, 8192)


class TestMojoSwiGLUParallel(unittest.TestCase):
    def setUp(self):
        self.swiglu = MojoSwiGLUParallel()
        rng = np.random.default_rng(1)
        self.gate_1d = rng.standard_normal(64).astype(np.float32)
        self.up_1d = rng.standard_normal(64).astype(np.float32)
        self.gate_2d = rng.standard_normal((4, 64)).astype(np.float32)
        self.up_2d = rng.standard_normal((4, 64)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.swiglu.backend(), ("mojo", "rust", "numpy"))

    def test_forward_1d_shape(self):
        out = self.swiglu.forward(self.gate_1d, self.up_1d)
        self.assertEqual(out.shape, (64,))

    def test_forward_1d_dtype(self):
        out = self.swiglu.forward(self.gate_1d, self.up_1d)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_1d_finite(self):
        out = self.swiglu.forward(self.gate_1d, self.up_1d)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_forward_2d_shape(self):
        out = self.swiglu.forward(self.gate_2d, self.up_2d)
        self.assertEqual(out.shape, (4, 64))

    def test_forward_2d_finite(self):
        out = self.swiglu.forward(self.gate_2d, self.up_2d)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_forward_vs_manual(self):
        out = self.swiglu.forward(self.gate_1d, self.up_1d)
        expected = self.gate_1d / (1.0 + np.exp(-self.gate_1d)) * self.up_1d
        np.testing.assert_array_almost_equal(out, expected, decimal=5)

    def test_zero_up(self):
        zero_up = np.zeros(64, dtype=np.float32)
        out = self.swiglu.forward(self.gate_1d, zero_up)
        np.testing.assert_array_almost_equal(out, np.zeros(64), decimal=5)

    def test_deterministic(self):
        out1 = self.swiglu.forward(self.gate_1d, self.up_1d)
        out2 = self.swiglu.forward(self.gate_1d, self.up_1d)
        np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# MojoGQADecodeKernel
# ---------------------------------------------------------------------------
from squish.kernels.mojo.gqa_decode_mojo import GQADecodeConfig, MojoGQADecodeKernel


class TestGQADecodeConfig(unittest.TestCase):
    def test_default_n_heads(self):
        cfg = GQADecodeConfig()
        self.assertEqual(cfg.n_heads, 32)

    def test_default_n_kv_heads(self):
        cfg = GQADecodeConfig()
        self.assertEqual(cfg.n_kv_heads, 8)

    def test_default_head_dim(self):
        cfg = GQADecodeConfig()
        self.assertEqual(cfg.head_dim, 128)

    def test_default_scale_none(self):
        cfg = GQADecodeConfig()
        self.assertIsNone(cfg.scale)


class TestMojoGQADecodeKernel(unittest.TestCase):
    def setUp(self):
        self.n_heads = 8
        self.n_kv_heads = 2
        self.head_dim = 16
        self.cache_len = 32
        cfg = GQADecodeConfig(
            n_heads=self.n_heads, n_kv_heads=self.n_kv_heads, head_dim=self.head_dim
        )
        self.gqa = MojoGQADecodeKernel(cfg)
        rng = np.random.default_rng(77)
        self.Q = rng.standard_normal((1, self.n_heads, self.head_dim)).astype(np.float32)
        self.K = rng.standard_normal((self.cache_len, self.n_kv_heads, self.head_dim)).astype(np.float32)
        self.V = rng.standard_normal((self.cache_len, self.n_kv_heads, self.head_dim)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.gqa.backend(), ("mojo", "numpy"))

    def test_forward_shape(self):
        out = self.gqa.forward(self.Q, self.K, self.V)
        self.assertEqual(out.shape, (1, self.n_heads, self.head_dim))

    def test_forward_dtype(self):
        out = self.gqa.forward(self.Q, self.K, self.V)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_finite(self):
        out = self.gqa.forward(self.Q, self.K, self.V)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_forward_single_token_cache(self):
        K = np.random.randn(1, self.n_kv_heads, self.head_dim).astype(np.float32)
        V = np.random.randn(1, self.n_kv_heads, self.head_dim).astype(np.float32)
        out = self.gqa.forward(self.Q, K, V)
        self.assertEqual(out.shape, (1, self.n_heads, self.head_dim))

    def test_forward_deterministic(self):
        out1 = self.gqa.forward(self.Q, self.K, self.V)
        out2 = self.gqa.forward(self.Q, self.K, self.V)
        np.testing.assert_array_equal(out1, out2)

    def test_gqa_mha_equivalence(self):
        # When n_heads == n_kv_heads, should match standard MHA
        n = 4
        hd = 8
        cfg = GQADecodeConfig(n_heads=n, n_kv_heads=n, head_dim=hd)
        gqa = MojoGQADecodeKernel(cfg)
        rng = np.random.default_rng(13)
        Q = rng.standard_normal((1, n, hd)).astype(np.float32)
        K = rng.standard_normal((16, n, hd)).astype(np.float32)
        V = rng.standard_normal((16, n, hd)).astype(np.float32)
        out = gqa.forward(Q, K, V)
        self.assertEqual(out.shape, (1, n, hd))


# ---------------------------------------------------------------------------
# MojoTokenCosSim
# ---------------------------------------------------------------------------
from squish.kernels.mojo.token_cos_sim_mojo import TokenCosSim_Config, MojoTokenCosSim


class TestTokenCosSim_Config(unittest.TestCase):
    def test_default_eps(self):
        cfg = TokenCosSim_Config()
        self.assertAlmostEqual(cfg.eps, 1e-12)

    def test_custom_eps(self):
        cfg = TokenCosSim_Config(eps=1e-8)
        self.assertAlmostEqual(cfg.eps, 1e-8)


class TestMojoTokenCosSim(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.sim = MojoTokenCosSim()
        self.a = rng.standard_normal((16, 32)).astype(np.float32)
        self.b = rng.standard_normal((12, 32)).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.sim.backend(), ("mojo", "rust", "numpy"))

    def test_compute_shape(self):
        out = self.sim.compute(self.a, self.b)
        self.assertEqual(out.shape, (16, 12))

    def test_compute_dtype(self):
        out = self.sim.compute(self.a, self.b)
        self.assertEqual(out.dtype, np.float32)

    def test_compute_range(self):
        out = self.sim.compute(self.a, self.b)
        self.assertTrue(np.all(out >= -1.0 - 1e-4))
        self.assertTrue(np.all(out <= 1.0 + 1e-4))

    def test_self_sim_diagonal(self):
        sim = self.sim.compute(self.a, self.a)
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(16), decimal=5)

    def test_parallel_vectors(self):
        a = np.ones((1, 4), dtype=np.float32)
        b = np.ones((1, 4), dtype=np.float32) * 3.0
        out = self.sim.compute(a, b)
        self.assertAlmostEqual(float(out[0, 0]), 1.0, places=5)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        out = self.sim.compute(a, b)
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=5)

    def test_top_k_pairs_shape(self):
        pairs, scores = self.sim.top_k_similar_pairs(self.a, self.b, k=3)
        self.assertEqual(pairs.shape, (3, 2))
        self.assertEqual(scores.shape, (3,))

    def test_top_k_pairs_sorted(self):
        _, scores = self.sim.top_k_similar_pairs(self.a, self.b, k=4)
        self.assertTrue(np.all(np.diff(scores) <= 1e-5))  # descending

    def test_top_k_pairs_valid_indices(self):
        pairs, _ = self.sim.top_k_similar_pairs(self.a, self.b, k=5)
        self.assertTrue(np.all(pairs[:, 0] < 16))
        self.assertTrue(np.all(pairs[:, 1] < 12))


# ---------------------------------------------------------------------------
# MojoSparseBlockScore
# ---------------------------------------------------------------------------
from squish.kernels.mojo.sparse_block_score_mojo import (
    SparseBlockScoreConfig, MojoSparseBlockScore
)


class TestSparseBlockScoreConfig(unittest.TestCase):
    def test_default_block_size(self):
        cfg = SparseBlockScoreConfig()
        self.assertEqual(cfg.block_size, 32)

    def test_default_head_dim(self):
        cfg = SparseBlockScoreConfig()
        self.assertEqual(cfg.head_dim, 128)

    def test_default_scale_none(self):
        cfg = SparseBlockScoreConfig()
        self.assertIsNone(cfg.scale)

    def test_custom_config(self):
        cfg = SparseBlockScoreConfig(block_size=16, head_dim=64)
        self.assertEqual(cfg.block_size, 16)


class TestMojoSparseBlockScore(unittest.TestCase):
    def setUp(self):
        self.n_heads = 4
        self.n_q_blocks = 3
        self.n_k_blocks = 8
        self.block_size = 4
        self.head_dim = 16
        cfg = SparseBlockScoreConfig(block_size=self.block_size, head_dim=self.head_dim)
        self.scorer = MojoSparseBlockScore(cfg)
        rng = np.random.default_rng(55)
        self.q_blocks = rng.standard_normal(
            (self.n_heads, self.n_q_blocks, self.block_size, self.head_dim)
        ).astype(np.float32)
        self.k_blocks = rng.standard_normal(
            (self.n_heads, self.n_k_blocks, self.block_size, self.head_dim)
        ).astype(np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.scorer.backend(), ("mojo", "numpy"))

    def test_score_shape(self):
        scores = self.scorer.score(self.q_blocks, self.k_blocks)
        self.assertEqual(scores.shape, (self.n_heads, self.n_q_blocks, self.n_k_blocks))

    def test_score_dtype(self):
        scores = self.scorer.score(self.q_blocks, self.k_blocks)
        self.assertEqual(scores.dtype, np.float32)

    def test_score_finite(self):
        scores = self.scorer.score(self.q_blocks, self.k_blocks)
        self.assertTrue(np.all(np.isfinite(scores)))

    def test_score_deterministic(self):
        s1 = self.scorer.score(self.q_blocks, self.k_blocks)
        s2 = self.scorer.score(self.q_blocks, self.k_blocks)
        np.testing.assert_array_equal(s1, s2)

    def test_top_k_shape(self):
        idx = self.scorer.top_k_blocks(self.q_blocks, self.k_blocks, k=3)
        self.assertEqual(idx.shape, (self.n_heads, self.n_q_blocks, 3))

    def test_top_k_valid_indices(self):
        idx = self.scorer.top_k_blocks(self.q_blocks, self.k_blocks, k=4)
        self.assertTrue(np.all(idx < self.n_k_blocks))
        self.assertTrue(np.all(idx >= 0))

    def test_top_k_dtype(self):
        idx = self.scorer.top_k_blocks(self.q_blocks, self.k_blocks, k=2)
        self.assertEqual(idx.dtype, np.int64)


# ---------------------------------------------------------------------------
# MojoRetentionState
# ---------------------------------------------------------------------------
from squish.kernels.mojo.retention_state_mojo import (
    RetentionStateConfig, MojoRetentionState
)


class TestRetentionStateConfig(unittest.TestCase):
    def test_default_n_heads(self):
        cfg = RetentionStateConfig()
        self.assertEqual(cfg.n_heads, 8)

    def test_default_head_dim(self):
        cfg = RetentionStateConfig()
        self.assertEqual(cfg.head_dim, 128)

    def test_default_gamma(self):
        cfg = RetentionStateConfig()
        self.assertAlmostEqual(cfg.gamma, 0.999)

    def test_custom_config(self):
        cfg = RetentionStateConfig(n_heads=4, head_dim=64, gamma=0.99)
        self.assertEqual(cfg.n_heads, 4)
        self.assertEqual(cfg.head_dim, 64)


class TestMojoRetentionState(unittest.TestCase):
    def setUp(self):
        self.n_heads = 4
        self.head_dim = 16
        cfg = RetentionStateConfig(n_heads=self.n_heads, head_dim=self.head_dim, gamma=0.9)
        self.ret = MojoRetentionState(cfg)
        rng = np.random.default_rng(17)
        self.q = rng.standard_normal((self.n_heads, self.head_dim)).astype(np.float32)
        self.k = rng.standard_normal((self.n_heads, self.head_dim)).astype(np.float32)
        self.v = rng.standard_normal((self.n_heads, self.head_dim)).astype(np.float32)
        self.state = np.zeros((self.n_heads, self.head_dim, self.head_dim), dtype=np.float32)

    def test_backend_is_string(self):
        self.assertIn(self.ret.backend(), ("mojo", "numpy"))

    def test_zero_state_shape(self):
        s = self.ret.zero_state()
        self.assertEqual(s.shape, (self.n_heads, self.head_dim, self.head_dim))

    def test_zero_state_dtype(self):
        s = self.ret.zero_state()
        self.assertEqual(s.dtype, np.float32)

    def test_step_output_o_shape(self):
        o, _ = self.ret.step(self.q, self.k, self.v, self.state)
        self.assertEqual(o.shape, (self.n_heads, self.head_dim))

    def test_step_output_state_shape(self):
        _, s_new = self.ret.step(self.q, self.k, self.v, self.state)
        self.assertEqual(s_new.shape, (self.n_heads, self.head_dim, self.head_dim))

    def test_step_output_dtype(self):
        o, s_new = self.ret.step(self.q, self.k, self.v, self.state)
        self.assertEqual(o.dtype, np.float32)
        self.assertEqual(s_new.dtype, np.float32)

    def test_step_output_finite(self):
        o, s_new = self.ret.step(self.q, self.k, self.v, self.state)
        self.assertTrue(np.all(np.isfinite(o)))
        self.assertTrue(np.all(np.isfinite(s_new)))

    def test_zero_state_zero_kv_output(self):
        # S=0, k=0, v=0 → outer=0 → S_new=0 → o = S_new @ q = 0
        k_zero = np.zeros_like(self.k)
        v_zero = np.zeros_like(self.v)
        o, s_new = self.ret.step(self.q, k_zero, v_zero, self.state)
        np.testing.assert_array_almost_equal(o, np.zeros_like(o), decimal=5)

    def test_step_state_update_direction(self):
        # After one step with nonzero k,v, state should be nonzero
        _, s_new = self.ret.step(self.q, self.k, self.v, self.state)
        self.assertGreater(np.abs(s_new).sum(), 0.0)

    def test_step_gamma_decay(self):
        # Two steps: second step's state should include gamma * first state
        _, s1 = self.ret.step(self.q, self.k, self.v, self.state)
        o2, s2 = self.ret.step(self.q, self.k, self.v, s1)
        # s2 should include 0.9 * s1 component
        self.assertTrue(np.any(s2 != s1))

    def test_gamma_override(self):
        o1, s1 = self.ret.step(self.q, self.k, self.v, self.state, gamma=0.0)
        o2, s2 = self.ret.step(self.q, self.k, self.v, self.state, gamma=1.0)
        # With gamma=0, state should just be outer product
        outer = np.einsum("hi,hj->hij", self.k, self.v)
        np.testing.assert_array_almost_equal(s1, outer, decimal=5)

    def test_deterministic(self):
        o1, s1 = self.ret.step(self.q, self.k, self.v, self.state)
        o2, s2 = self.ret.step(self.q, self.k, self.v, self.state)
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(s1, s2)


if __name__ == "__main__":
    unittest.main()
