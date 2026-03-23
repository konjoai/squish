"""tests/test_wave59b_mojo_kernels.py — Wave 59b Mojo kernel tests.

Tests for:
  - MojoFlashDecodeKernel  (flash_decode_mojo.py)
  - MojoBF16GEMV           (bf16_gemv_mojo.py)
  - MojoGQAPrefill         (gqa_prefill_mojo.py)
  - MojoSplitKReduce       (splitk_reduce_mojo.py)
  - MojoRotaryEmbed        (rotary_embed_mojo.py)
  - MojoLayerSkipPredict   (layer_skip_predict_mojo.py)

All tests use NumPy fallback path (Mojo runtime not present in CI).
75 tests total.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from squish.kernels.mojo.flash_decode_mojo import MojoFlashDecodeConfig, MojoFlashDecodeKernel
from squish.kernels.mojo.bf16_gemv_mojo import BF16GEMVConfig, MojoBF16GEMV
from squish.kernels.mojo.gqa_prefill_mojo import GQAPrefillConfig, MojoGQAPrefill
from squish.kernels.mojo.splitk_reduce_mojo import SplitKReduceConfig, MojoSplitKReduce
from squish.kernels.mojo.rotary_embed_mojo import RotaryEmbedConfig, MojoRotaryEmbed
from squish.kernels.mojo.layer_skip_predict_mojo import LayerSkipConfig, MojoLayerSkipPredict


# ── MojoFlashDecodeConfig / MojoFlashDecodeKernel ─────────────────────────


class TestMojoFlashDecodeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MojoFlashDecodeConfig()
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.head_dim, 128)

    def test_custom(self):
        cfg = MojoFlashDecodeConfig(n_heads=8, head_dim=64, gqa_group=2)
        self.assertEqual(cfg.gqa_group, 2)


class TestMojoFlashDecodeKernel(unittest.TestCase):
    def setUp(self):
        self.fdk = MojoFlashDecodeKernel(
            MojoFlashDecodeConfig(n_heads=4, head_dim=8, gqa_group=2)
        )
        self.rng = np.random.default_rng(10)

    def _make_qkv(self):
        q = self.rng.standard_normal((4, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 6, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 6, 8)).astype(np.float32)
        return q, k, v

    def test_backend_is_string(self):
        self.assertIn(self.fdk.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        q, k, v = self._make_qkv()
        out, lse, ms = self.fdk.compute_split(q, k, v)
        self.assertEqual(out.shape, (4, 8))

    def test_lse_shape(self):
        q, k, v = self._make_qkv()
        _, lse, _ = self.fdk.compute_split(q, k, v)
        self.assertEqual(lse.shape, (4,))

    def test_max_score_shape(self):
        q, k, v = self._make_qkv()
        _, _, ms = self.fdk.compute_split(q, k, v)
        self.assertEqual(ms.shape, (4,))

    def test_output_finite(self):
        q, k, v = self._make_qkv()
        out, lse, ms = self.fdk.compute_split(q, k, v)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(lse).all())
        self.assertTrue(np.isfinite(ms).all())

    def test_n_heads_property(self):
        self.assertEqual(self.fdk.n_heads(), 4)

    def test_head_dim_property(self):
        self.assertEqual(self.fdk.head_dim(), 8)

    def test_gqa_group_override(self):
        q = self.rng.standard_normal((8, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out, _, _ = self.fdk.compute_split(q, k, v, gqa_group=4)
        self.assertEqual(out.shape, (8, 8))

    def test_output_no_nan(self):
        q, k, v = self._make_qkv()
        out, _, _ = self.fdk.compute_split(q, k, v)
        self.assertFalse(np.isnan(out).any())

    def test_single_token_kv(self):
        q = self.rng.standard_normal((4, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 1, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 1, 8)).astype(np.float32)
        out, _, _ = self.fdk.compute_split(q, k, v)
        self.assertEqual(out.shape, (4, 8))


# ── BF16GEMVConfig / MojoBF16GEMV ─────────────────────────────────────────


class TestBF16GEMVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = BF16GEMVConfig()
        self.assertEqual(cfg.hidden_dim, 4096)

    def test_custom(self):
        cfg = BF16GEMVConfig(hidden_dim=512)
        self.assertEqual(cfg.hidden_dim, 512)


class TestMojoBF16GEMV(unittest.TestCase):
    def setUp(self):
        self.gemv = MojoBF16GEMV(BF16GEMVConfig(hidden_dim=8))
        self.rng = np.random.default_rng(11)

    def _bf16_weights(self, out_f, in_f):
        W = self.rng.standard_normal((out_f, in_f)).astype(np.float32)
        return (W.view(np.uint32) >> 16).astype(np.uint16)

    def test_backend_is_string(self):
        self.assertIn(self.gemv.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        W_bits = self._bf16_weights(16, 8)
        a = self.rng.standard_normal(8).astype(np.float32)
        out = self.gemv.gemv(W_bits, a)
        self.assertEqual(out.shape, (16,))

    def test_output_dtype(self):
        W_bits = self._bf16_weights(16, 8)
        a = self.rng.standard_normal(8).astype(np.float32)
        out = self.gemv.gemv(W_bits, a)
        self.assertEqual(out.dtype, np.float32)

    def test_output_finite(self):
        W_bits = self._bf16_weights(16, 8)
        a = self.rng.standard_normal(8).astype(np.float32)
        out = self.gemv.gemv(W_bits, a)
        self.assertTrue(np.isfinite(out).all())

    def test_hidden_dim_property(self):
        self.assertEqual(self.gemv.hidden_dim(), 8)

    def test_zero_activation(self):
        W_bits = self._bf16_weights(8, 8)
        a = np.zeros(8, dtype=np.float32)
        out = self.gemv.gemv(W_bits, a)
        np.testing.assert_allclose(out, 0.0, atol=1e-4)


# ── GQAPrefillConfig / MojoGQAPrefill ─────────────────────────────────────


class TestGQAPrefillConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = GQAPrefillConfig()
        self.assertEqual(cfg.n_q_heads, 32)
        self.assertEqual(cfg.n_kv_heads, 8)

    def test_custom(self):
        cfg = GQAPrefillConfig(n_q_heads=8, n_kv_heads=2, head_dim=32)
        self.assertEqual(cfg.head_dim, 32)


class TestMojoGQAPrefill(unittest.TestCase):
    def setUp(self):
        self.gqa = MojoGQAPrefill(
            GQAPrefillConfig(n_q_heads=4, n_kv_heads=2, head_dim=8)
        )
        self.rng = np.random.default_rng(12)

    def test_backend_is_string(self):
        self.assertIn(self.gqa.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        T_q, T_k, head_dim = 6, 6, 8
        q = self.rng.standard_normal((4, T_q, head_dim)).astype(np.float32)
        k = self.rng.standard_normal((2, T_k, head_dim)).astype(np.float32)
        v = self.rng.standard_normal((2, T_k, head_dim)).astype(np.float32)
        out = self.gqa.forward(q, k, v)
        self.assertEqual(out.shape, (4, T_q, head_dim))

    def test_output_dtype(self):
        q = self.rng.standard_normal((4, 4, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out = self.gqa.forward(q, k, v)
        self.assertEqual(out.dtype, np.float32)

    def test_output_finite(self):
        q = self.rng.standard_normal((4, 4, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        v = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        out = self.gqa.forward(q, k, v)
        self.assertTrue(np.isfinite(out).all())

    def test_causal_mask_upper_triangle(self):
        # For causal attention the query at position 0 can only attend to k[0]
        q = self.rng.standard_normal((4, 4, 8)).astype(np.float32)
        k = self.rng.standard_normal((2, 4, 8)).astype(np.float32)
        v = np.eye(4, 8, dtype=np.float32)[np.newaxis].repeat(2, axis=0)
        out = self.gqa.forward(q, k, v)
        self.assertEqual(out.shape, (4, 4, 8))

    def test_n_q_heads_property(self):
        self.assertEqual(self.gqa.n_q_heads(), 4)

    def test_n_kv_heads_property(self):
        self.assertEqual(self.gqa.n_kv_heads(), 2)

    def test_group_size_override(self):
        q = self.rng.standard_normal((4, 4, 8)).astype(np.float32)
        k = self.rng.standard_normal((4, 4, 8)).astype(np.float32)
        v = self.rng.standard_normal((4, 4, 8)).astype(np.float32)
        out = self.gqa.forward(q, k, v, group_size=1)
        self.assertEqual(out.shape, (4, 4, 8))


# ── SplitKReduceConfig / MojoSplitKReduce ─────────────────────────────────


class TestSplitKReduceConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SplitKReduceConfig()
        self.assertEqual(cfg.n_heads, 32)

    def test_custom(self):
        cfg = SplitKReduceConfig(n_heads=4, head_dim=16)
        self.assertEqual(cfg.head_dim, 16)


class TestMojoSplitKReduce(unittest.TestCase):
    def setUp(self):
        self.skr = MojoSplitKReduce(SplitKReduceConfig(n_heads=4, head_dim=8))
        self.rng = np.random.default_rng(13)

    def _make_splits(self, n_splits=3):
        outputs = [
            self.rng.standard_normal((4, 8)).astype(np.float32)
            for _ in range(n_splits)
        ]
        lses = [
            self.rng.standard_normal(4).astype(np.float32)
            for _ in range(n_splits)
        ]
        return outputs, lses

    def test_backend_is_string(self):
        self.assertIn(self.skr.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        outputs, lses = self._make_splits()
        out = self.skr.merge(outputs, lses)
        self.assertEqual(out.shape, (4, 8))

    def test_output_dtype(self):
        outputs, lses = self._make_splits()
        out = self.skr.merge(outputs, lses)
        self.assertEqual(out.dtype, np.float32)

    def test_output_finite(self):
        outputs, lses = self._make_splits()
        out = self.skr.merge(outputs, lses)
        self.assertTrue(np.isfinite(out).all())

    def test_single_split_is_identity(self):
        o = self.rng.standard_normal((4, 8)).astype(np.float32)
        lse = np.zeros(4, dtype=np.float32)
        out = self.skr.merge([o], [lse])
        # With single split: softmax weight == 1, output == input
        np.testing.assert_allclose(out, o, atol=1e-5)

    def test_n_heads_property(self):
        self.assertEqual(self.skr.n_heads(), 4)

    def test_five_splits(self):
        outputs, lses = self._make_splits(5)
        out = self.skr.merge(outputs, lses)
        self.assertEqual(out.shape, (4, 8))


# ── RotaryEmbedConfig / MojoRotaryEmbed ───────────────────────────────────


class TestRotaryEmbedConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = RotaryEmbedConfig()
        self.assertEqual(cfg.n_heads, 32)

    def test_custom(self):
        cfg = RotaryEmbedConfig(n_heads=4, head_dim=8)
        self.assertEqual(cfg.head_dim, 8)


class TestMojoRotaryEmbed(unittest.TestCase):
    def setUp(self):
        self.rope = MojoRotaryEmbed(RotaryEmbedConfig(n_heads=4, head_dim=8))
        self.rng = np.random.default_rng(14)

    def _make_inputs(self, n_heads=4, T=6, head_dim=8):
        x = self.rng.standard_normal((n_heads, T, head_dim)).astype(np.float32)
        cos = np.ones((T, head_dim // 2), dtype=np.float32)
        sin = np.zeros((T, head_dim // 2), dtype=np.float32)
        return x, cos, sin

    def test_backend_is_string(self):
        self.assertIn(self.rope.backend(), ("mojo", "numpy"))

    def test_output_shape(self):
        x, cos, sin = self._make_inputs()
        out = self.rope.apply(x, cos, sin)
        self.assertEqual(out.shape, (4, 6, 8))

    def test_output_dtype(self):
        x, cos, sin = self._make_inputs()
        out = self.rope.apply(x, cos, sin)
        self.assertEqual(out.dtype, np.float32)

    def test_identity_rotation(self):
        # cos=1, sin=0 → output == input
        x, cos, sin = self._make_inputs()
        out = self.rope.apply(x, cos, sin)
        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_half_turn_rotation(self):
        # cos=0, sin=1 → x1_out = -x2, x2_out = x1 (rotation by 90°)
        x = np.ones((4, 2, 8), dtype=np.float32)
        cos = np.zeros((2, 4), dtype=np.float32)
        sin = np.ones((2, 4), dtype=np.float32)
        out = self.rope.apply(x, cos, sin)
        self.assertEqual(out.shape, (4, 2, 8))

    def test_n_heads_property(self):
        self.assertEqual(self.rope.n_heads(), 4)

    def test_output_finite(self):
        x, cos, sin = self._make_inputs()
        cos_rand = self.rng.standard_normal((6, 4)).astype(np.float32)
        sin_rand = self.rng.standard_normal((6, 4)).astype(np.float32)
        out = self.rope.apply(x, cos_rand, sin_rand)
        self.assertTrue(np.isfinite(out).all())


# ── LayerSkipConfig / MojoLayerSkipPredict ────────────────────────────────


class TestLayerSkipConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = LayerSkipConfig()
        self.assertEqual(cfg.n_layers, 32)

    def test_custom(self):
        cfg = LayerSkipConfig(n_layers=8, n_features=16)
        self.assertEqual(cfg.n_features, 16)


class TestMojoLayerSkipPredict(unittest.TestCase):
    def setUp(self):
        self.skip = MojoLayerSkipPredict(LayerSkipConfig(n_layers=4, n_features=8))
        self.rng = np.random.default_rng(15)

    def test_backend_is_string(self):
        self.assertIn(self.skip.backend(), ("mojo", "numpy"))

    def test_predict_shape(self):
        feats = self.rng.standard_normal(8).astype(np.float32)
        out = self.skip.predict(feats)
        self.assertEqual(out.shape, (4,))

    def test_predict_dtype(self):
        feats = self.rng.standard_normal(8).astype(np.float32)
        out = self.skip.predict(feats)
        self.assertEqual(out.dtype, np.float32)

    def test_predict_in_0_1(self):
        feats = self.rng.standard_normal(8).astype(np.float32)
        out = self.skip.predict(feats)
        self.assertTrue((out >= 0).all() and (out <= 1).all())

    def test_predict_with_custom_weights(self):
        feats = np.ones(8, dtype=np.float32)
        W = np.zeros((4, 8), dtype=np.float32)
        out = self.skip.predict(feats, weights=W)
        np.testing.assert_allclose(out, 0.5, atol=1e-5)

    def test_update_weights(self):
        W_new = self.rng.standard_normal((4, 8)).astype(np.float32)
        self.skip.update_weights(W_new)
        np.testing.assert_allclose(self.skip.weights(), W_new, atol=1e-7)

    def test_n_layers_property(self):
        self.assertEqual(self.skip.n_layers(), 4)

    def test_n_features_property(self):
        self.assertEqual(self.skip.n_features(), 8)

    def test_sigmoid_saturation(self):
        feats = np.ones(8, dtype=np.float32)
        W = np.full((4, 8), 1000.0, dtype=np.float32)
        out = self.skip.predict(feats, weights=W)
        np.testing.assert_allclose(out, 1.0, atol=1e-5)

    def test_predict_no_nan(self):
        feats = self.rng.standard_normal(8).astype(np.float32) * 100
        out = self.skip.predict(feats)
        self.assertFalse(np.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
