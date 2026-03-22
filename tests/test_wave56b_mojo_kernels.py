"""Tests for Wave 56b Mojo kernel Python wrappers.

All tests exercise the NumPy fallback path — they run in any environment
whether or not the Mojo toolchain (``magic``) is installed.  Covers:
- MojoBridge      (mojo_bridge.py)
- MojoSoftmax     (softmax_mojo.py)
- MojoRoPE        (rope_mojo.py)
- MojoNF4Dequant  (nf4_dequant_mojo.py)
- MojoINT4GEMM    (int4_gemm_mojo.py)
- MojoFlashPrefill (flash_prefill_mojo.py)
"""

import unittest

import numpy as np

from squish.kernels.mojo.mojo_bridge import MojoBridge, MojoBridgeConfig
from squish.kernels.mojo.softmax_mojo import MojoSoftmax, MojoSoftmaxConfig
from squish.kernels.mojo.rope_mojo import MojoRoPE, MojoRoPEConfig
from squish.kernels.mojo.nf4_dequant_mojo import MojoNF4Dequant, MojoNF4DequantConfig
from squish.kernels.mojo.int4_gemm_mojo import MojoINT4GEMM, MojoINT4GEMMConfig
from squish.kernels.mojo.flash_prefill_mojo import MojoFlashPrefill, MojoFlashPrefillConfig
from squish.kernels.rs_nf4 import RustNF4Kernel, NF4KernelConfig

RNG = np.random.default_rng(99)


# ═══════════════════════════════════════════════════════════════════════════════
# MojoBridge
# ═══════════════════════════════════════════════════════════════════════════════

class TestMojoBridgeConfig(unittest.TestCase):
    def test_default_lib_path_none(self):
        cfg = MojoBridgeConfig()
        self.assertIsNone(cfg.lib_path)

    def test_default_rust_fallback_true(self):
        cfg = MojoBridgeConfig()
        self.assertTrue(cfg.rust_fallback)

    def test_custom_lib_path(self):
        cfg = MojoBridgeConfig(lib_path="/tmp/libsquish.so")
        self.assertEqual(cfg.lib_path, "/tmp/libsquish.so")


class TestMojoBridge(unittest.TestCase):
    def test_instantiation(self):
        bridge = MojoBridge()
        self.assertIsInstance(bridge, MojoBridge)

    def test_is_available_returns_bool(self):
        bridge = MojoBridge()
        self.assertIsInstance(bridge.is_available(), bool)

    def test_backend_returns_string(self):
        bridge = MojoBridge()
        self.assertIsInstance(bridge.backend(), str)

    def test_backend_valid_value(self):
        bridge = MojoBridge()
        self.assertIn(bridge.backend(), ("mojo", "rust", "numpy"))

    def test_load_kernel_unknown_returns_none(self):
        bridge = MojoBridge()
        # In CI without compiled libraries, any kernel lookup returns None
        result = bridge.load_kernel("nonexistent_kernel_xyz")
        self.assertIsNone(result)

    def test_no_mojo_library_fallback(self):
        # With no library compiled, backend() falls back gracefully
        bridge = MojoBridge(MojoBridgeConfig(lib_path="/nonexistent/path.so"))
        backend = bridge.backend()
        self.assertIn(backend, ("rust", "numpy"))

    def test_rust_fallback_false(self):
        bridge = MojoBridge(MojoBridgeConfig(rust_fallback=False))
        # When both Mojo and Rust are unavailable, backend is "numpy"
        if not bridge.is_available():
            self.assertIn(bridge.backend(), ("numpy", "rust"))


# ═══════════════════════════════════════════════════════════════════════════════
# MojoSoftmax
# ═══════════════════════════════════════════════════════════════════════════════

class TestMojoSoftmaxConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MojoSoftmaxConfig()
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertAlmostEqual(cfg.top_p, 0.9)
        self.assertEqual(cfg.seed, 0)


class TestMojoSoftmax(unittest.TestCase):
    def test_forward_sums_to_one(self):
        ks = MojoSoftmax()
        logits = RNG.standard_normal(50).astype(np.float32)
        probs = ks.forward(logits)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_forward_all_non_negative(self):
        ks = MojoSoftmax()
        logits = RNG.standard_normal(50).astype(np.float32)
        probs = ks.forward(logits)
        self.assertTrue(np.all(probs >= 0))

    def test_forward_output_dtype(self):
        ks = MojoSoftmax()
        logits = np.zeros(10, dtype=np.float32)
        probs = ks.forward(logits)
        self.assertEqual(probs.dtype, np.float32)

    def test_forward_output_shape(self):
        ks = MojoSoftmax()
        logits = RNG.standard_normal(20).astype(np.float32)
        probs = ks.forward(logits)
        self.assertEqual(probs.shape, logits.shape)

    def test_forward_uniform_uniform_probs(self):
        ks = MojoSoftmax()
        logits = np.zeros(8, dtype=np.float32)
        probs = ks.forward(logits)
        np.testing.assert_allclose(probs, np.full(8, 1.0 / 8), atol=1e-6)

    def test_forward_single_argmax(self):
        ks = MojoSoftmax()
        logits = np.array([-1000.0, 1000.0, -1000.0], dtype=np.float32)
        probs = ks.forward(logits)
        self.assertGreater(probs[1], 0.99)

    def test_temperature_affects_distribution(self):
        ks_hot  = MojoSoftmax(MojoSoftmaxConfig(temperature=2.0))
        ks_cold = MojoSoftmax(MojoSoftmaxConfig(temperature=0.1))
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        p_hot  = ks_hot.forward(logits)
        p_cold = ks_cold.forward(logits)
        # Higher temperature → flatter distribution
        self.assertGreater(p_hot.min(), p_cold.min())

    def test_fused_top_p_sums_to_one(self):
        ks = MojoSoftmax(MojoSoftmaxConfig(top_p=0.8))
        logits = RNG.standard_normal(50).astype(np.float32)
        filtered = ks.fused_top_p(logits)
        self.assertAlmostEqual(float(filtered.sum()), 1.0, places=5)

    def test_fused_top_p_zeros_some(self):
        ks = MojoSoftmax()
        logits = np.array([10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        filtered = ks.fused_top_p(logits, p=0.9)
        self.assertGreater(np.sum(filtered == 0), 0)

    def test_fused_top_p_custom_p(self):
        ks = MojoSoftmax()
        logits = RNG.standard_normal(100).astype(np.float32)
        f1 = ks.fused_top_p(logits, p=0.5)
        f2 = ks.fused_top_p(logits, p=0.95)
        # Smaller p → fewer tokens kept
        self.assertLessEqual(np.sum(f1 > 0), np.sum(f2 > 0))

    def test_backend_string(self):
        ks = MojoSoftmax()
        self.assertIn(ks.backend(), ("mojo", "rust", "numpy"))


# ═══════════════════════════════════════════════════════════════════════════════
# MojoRoPE
# ═══════════════════════════════════════════════════════════════════════════════

class TestMojoRoPEConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MojoRoPEConfig()
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.max_seq_len, 4096)
        self.assertAlmostEqual(cfg.base, 10000.0)


class TestMojoRoPE(unittest.TestCase):
    def test_build_freqs_shape(self):
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8, max_seq_len=16))
        freqs = kr.build_freqs(8)
        self.assertEqual(freqs.shape, (8, 4))  # (seq_len, head_dim//2)

    def test_build_freqs_complex_dtype(self):
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8))
        freqs = kr.build_freqs(4)
        self.assertTrue(np.issubdtype(freqs.dtype, np.complexfloating))

    def test_build_freqs_magnitude_one(self):
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8))
        freqs = kr.build_freqs(4)
        # All complex exponentials should have |z| == 1
        np.testing.assert_allclose(np.abs(freqs), 1.0, atol=1e-6)

    def test_apply_output_shape(self):
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8, max_seq_len=16))
        x = RNG.standard_normal((2, 4, 8)).astype(np.float32)
        positions = np.arange(4, dtype=np.int32)
        out = kr.apply(x, positions)
        self.assertEqual(out.shape, x.shape)

    def test_apply_output_dtype(self):
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8))
        x = RNG.standard_normal((2, 4, 8)).astype(np.float32)
        positions = np.arange(4, dtype=np.int32)
        out = kr.apply(x, positions)
        self.assertEqual(out.dtype, np.float32)

    def test_apply_position_zero_unchanged(self):
        # At position 0, cos(0)=1 and sin(0)=0 — RoPE should be identity
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8))
        x = RNG.standard_normal((1, 1, 8)).astype(np.float32)
        positions = np.array([0], dtype=np.int32)
        out = kr.apply(x, positions)
        np.testing.assert_allclose(x, out, atol=1e-6)

    def test_apply_preserves_norm(self):
        # RoPE is an isometry — output norms should equal input norms
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8))
        x = RNG.standard_normal((2, 4, 8)).astype(np.float32)
        positions = np.arange(4, dtype=np.int32)
        out = kr.apply(x, positions)
        in_norm  = np.linalg.norm(x,   axis=-1)
        out_norm = np.linalg.norm(out, axis=-1)
        np.testing.assert_allclose(in_norm, out_norm, atol=1e-5)

    def test_apply_wrong_shape_raises(self):
        kr = MojoRoPE(MojoRoPEConfig(head_dim=8))
        with self.assertRaises(ValueError):
            kr.apply(np.ones((2, 4, 16), dtype=np.float32), np.arange(4, dtype=np.int32))

    def test_backend_string(self):
        kr = MojoRoPE()
        self.assertIn(kr.backend(), ("mojo", "rust", "numpy"))


# ═══════════════════════════════════════════════════════════════════════════════
# MojoNF4Dequant
# ═══════════════════════════════════════════════════════════════════════════════

class TestMojoNF4DequantConfig(unittest.TestCase):
    def test_default_group_size(self):
        cfg = MojoNF4DequantConfig()
        self.assertEqual(cfg.group_size, 64)


class TestMojoNF4Dequant(unittest.TestCase):
    def _make_packed(self, n_rows=4, n_cols=8, gs=4):
        """Create consistent packed NF4 data via RustNF4Kernel."""
        k = RustNF4Kernel(NF4KernelConfig(group_size=gs))
        W = RNG.standard_normal((n_rows, n_cols)).astype(np.float32)
        packed, scales = k.quantize(W)
        return packed, scales, W

    def test_dequantize_output_shape(self):
        kd = MojoNF4Dequant(MojoNF4DequantConfig(group_size=4))
        packed, scales, W = self._make_packed(gs=4)
        out = kd.dequantize(packed, scales)
        self.assertEqual(out.shape, W.shape)

    def test_dequantize_output_dtype(self):
        kd = MojoNF4Dequant(MojoNF4DequantConfig(group_size=4))
        packed, scales, _ = self._make_packed(gs=4)
        out = kd.dequantize(packed, scales)
        self.assertEqual(out.dtype, np.float32)

    def test_dequantize_quality(self):
        kd = MojoNF4Dequant(MojoNF4DequantConfig(group_size=4))
        packed, scales, W = self._make_packed(gs=4)
        out = kd.dequantize(packed, scales)
        np.testing.assert_allclose(W, out, atol=0.5)

    def test_dequantize_zeros(self):
        kd = MojoNF4Dequant(MojoNF4DequantConfig(group_size=4))
        packed, scales, _ = self._make_packed(gs=4)
        packed_zero = np.zeros_like(packed)  # all zero nibbles → NF4_LUT[0]
        scales_one = np.ones_like(scales)
        out = kd.dequantize(packed_zero, scales_one)
        np.testing.assert_allclose(out, -1.0, atol=1e-5)

    def test_backend_string(self):
        kd = MojoNF4Dequant()
        self.assertIn(kd.backend(), ("mojo", "rust", "numpy"))

    def test_consistency_with_rs_nf4_kernel(self):
        """MojoNF4Dequant NumPy path should match RustNF4Kernel NumPy path."""
        gs = 4
        packed, scales, W = self._make_packed(gs=gs)
        rs_k  = RustNF4Kernel(NF4KernelConfig(group_size=gs))
        mojo_k = MojoNF4Dequant(MojoNF4DequantConfig(group_size=gs))
        out_rs   = rs_k.dequantize(packed, scales)
        out_mojo = mojo_k.dequantize(packed, scales)
        np.testing.assert_allclose(out_rs, out_mojo, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════════
# MojoINT4GEMM
# ═══════════════════════════════════════════════════════════════════════════════

class TestMojoINT4GEMMConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MojoINT4GEMMConfig()
        self.assertEqual(cfg.group_size, 128)
        self.assertTrue(cfg.fuse_dequant)


class TestMojoINT4GEMM(unittest.TestCase):
    def _make_weights(self, n=8, k=16, gs=4):
        """Generate nibble-packed asymmetric INT4 weights."""
        from squish.kernels.mojo.int4_gemm_mojo import MojoINT4GEMM
        W = RNG.standard_normal((n, k)).astype(np.float32)
        n_groups = k // gs
        # Compute per-group asymmetric quantization manually
        W_r = W.reshape(n, n_groups, gs)
        gmin = W_r.min(axis=-1)
        gmax = W_r.max(axis=-1)
        scales  = np.where(gmax == gmin, 1.0, (gmax - gmin) / 15.0)
        offsets = gmin
        scale_full  = np.repeat(scales,  gs, axis=1)
        offset_full = np.repeat(offsets, gs, axis=1)
        W_q = np.clip(np.round((W - offset_full) / scale_full), 0, 15).astype(np.uint8)
        lo = W_q[:, 0::2] & 0x0F
        hi = (W_q[:, 1::2] & 0x0F) << 4
        packed = (lo | hi).astype(np.uint8)
        return packed, scales, offsets, W

    def test_matmul_output_shape(self):
        kg = MojoINT4GEMM(MojoINT4GEMMConfig(group_size=4))
        m, k, n = 4, 16, 8
        x = RNG.standard_normal((m, k)).astype(np.float32)
        packed, scales, offsets, _ = self._make_weights(n=n, k=k, gs=4)
        out = kg.matmul(x, packed, scales, offsets)
        self.assertEqual(out.shape, (m, n))

    def test_matmul_output_dtype(self):
        kg = MojoINT4GEMM(MojoINT4GEMMConfig(group_size=4))
        x = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, scales, offsets, _ = self._make_weights(n=8, k=16, gs=4)
        out = kg.matmul(x, packed, scales, offsets)
        self.assertEqual(out.dtype, np.float32)

    def test_matmul_zero_input(self):
        kg = MojoINT4GEMM(MojoINT4GEMMConfig(group_size=4))
        x = np.zeros((2, 16), dtype=np.float32)
        packed, scales, offsets, _ = self._make_weights(n=4, k=16, gs=4)
        out = kg.matmul(x, packed, scales, offsets)
        self.assertEqual(out.shape, (2, 4))

    def test_matmul_requires_2d_x(self):
        kg = MojoINT4GEMM()
        with self.assertRaises(ValueError):
            packed = np.zeros((4, 8), dtype=np.uint8)
            scales = np.ones((4, 4), dtype=np.float32)
            offsets = np.zeros((4, 4), dtype=np.float32)
            kg.matmul(np.ones(16, dtype=np.float32), packed, scales, offsets)

    def test_backend_string(self):
        kg = MojoINT4GEMM()
        self.assertIn(kg.backend(), ("mojo", "rust", "numpy"))

    def test_matmul_approximate_correctness(self):
        gs = 4
        kg = MojoINT4GEMM(MojoINT4GEMMConfig(group_size=gs))
        n, k = 8, 16
        packed, scales, offsets, W_f32 = self._make_weights(n=n, k=k, gs=gs)
        x = RNG.standard_normal((4, k)).astype(np.float32)
        out = kg.matmul(x, packed, scales, offsets)
        ref = (x @ W_f32.T).astype(np.float32)
        # Allow generous tolerance due to quantization noise
        np.testing.assert_allclose(out, ref, atol=3.0, rtol=0.5)

    def test_matmul_large_batch(self):
        kg = MojoINT4GEMM(MojoINT4GEMMConfig(group_size=4))
        x = RNG.standard_normal((32, 16)).astype(np.float32)
        packed, scales, offsets, _ = self._make_weights(n=8, k=16, gs=4)
        out = kg.matmul(x, packed, scales, offsets)
        self.assertEqual(out.shape, (32, 8))


# ═══════════════════════════════════════════════════════════════════════════════
# MojoFlashPrefill
# ═══════════════════════════════════════════════════════════════════════════════

class TestMojoFlashPrefillConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MojoFlashPrefillConfig()
        self.assertEqual(cfg.block_size, 16)
        self.assertEqual(cfg.n_heads, 32)
        self.assertEqual(cfg.head_dim, 128)
        self.assertTrue(cfg.causal)


class TestMojoFlashPrefill(unittest.TestCase):
    def _make_qkv(self, n_heads=2, seq_len=8, head_dim=8):
        Q = RNG.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        K = RNG.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        V = RNG.standard_normal((n_heads, seq_len, head_dim)).astype(np.float32)
        return Q, K, V

    def test_forward_output_shape(self):
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=4))
        Q, K, V = self._make_qkv()
        out = kf.forward(Q, K, V)
        self.assertEqual(out.shape, Q.shape)

    def test_forward_output_dtype(self):
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=4))
        Q, K, V = self._make_qkv()
        out = kf.forward(Q, K, V)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_finite_output(self):
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=4))
        Q, K, V = self._make_qkv()
        out = kf.forward(Q, K, V)
        self.assertTrue(np.all(np.isfinite(out)))

    def test_forward_requires_3d_qkv(self):
        kf = MojoFlashPrefill()
        with self.assertRaises(ValueError):
            kf.forward(np.ones((2, 8)), np.ones((2, 8)), np.ones((2, 8)))

    def test_forward_mismatched_shapes_raises(self):
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=4))
        Q, K, V = self._make_qkv()
        with self.assertRaises(ValueError):
            kf.forward(Q, K[..., :4], V)

    def test_causal_mask_blocks_future(self):
        kf = MojoFlashPrefill(
            MojoFlashPrefillConfig(block_size=4, causal=True)
        )
        kf_nc = MojoFlashPrefill(
            MojoFlashPrefillConfig(block_size=4, causal=False)
        )
        Q, K, V = self._make_qkv(seq_len=8)
        # Causal and non-causal outputs should differ
        out_causal = kf.forward(Q, K, V)
        out_noncausal = kf_nc.forward(Q, K, V)
        self.assertFalse(np.allclose(out_causal, out_noncausal, atol=1e-4))

    def test_single_token(self):
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=4))
        Q, K, V = self._make_qkv(seq_len=1, head_dim=8)
        out = kf.forward(Q, K, V)
        self.assertEqual(out.shape, Q.shape)

    def test_attn_output_in_v_range(self):
        # Flash attention output = weighted average of V — should be in V range
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=4))
        Q, K = self._make_qkv()[:2]
        V = np.ones_like(Q)  # constant V → all outputs should be ~1
        out = kf.forward(Q, K, V)
        np.testing.assert_allclose(out, 1.0, atol=1e-4)

    def test_block_size_one(self):
        kf = MojoFlashPrefill(MojoFlashPrefillConfig(block_size=1))
        Q, K, V = self._make_qkv(seq_len=4, head_dim=8)
        out = kf.forward(Q, K, V)
        self.assertEqual(out.shape, Q.shape)

    def test_matches_standard_sdpa(self):
        """Flash prefill should match standard scaled dot-product attention."""
        kf = MojoFlashPrefill(
            MojoFlashPrefillConfig(block_size=4, causal=False)
        )
        n_heads, seq_len, head_dim = 2, 6, 8
        Q, K, V = self._make_qkv(n_heads=n_heads, seq_len=seq_len, head_dim=head_dim)

        # Reference: standard SDPA
        scale = 1.0 / np.sqrt(head_dim)
        ref = np.zeros_like(Q)
        for h in range(n_heads):
            scores = (Q[h] @ K[h].T) * scale  # (seq, seq)
            probs = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            ref[h] = probs @ V[h]

        out = kf.forward(Q, K, V)
        np.testing.assert_allclose(out, ref, atol=1e-4)

    def test_backend_string(self):
        kf = MojoFlashPrefill()
        self.assertIn(kf.backend(), ("mojo", "rust", "numpy"))


if __name__ == "__main__":
    unittest.main()
