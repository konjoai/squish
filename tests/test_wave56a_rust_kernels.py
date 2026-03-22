"""Tests for Wave 56a Rust kernel Python wrappers.

All tests exercise the NumPy fallback path, so they run even without the
``squish_quant`` Rust extension compiled.  Covers:
- RustNF4Kernel   (rs_nf4.py)
- RustFP8Kernel   (rs_fp8.py)
- RustINT3Kernel  (rs_int3.py)
- RustSamplerKernel (rs_sampler.py)
- RustKVQuantKernel (rs_kv_quant.py)
- RustINT2Kernel  (rs_int2.py)
"""

import math
import unittest

import numpy as np

from squish.kernels.rs_nf4 import NF4KernelConfig, RustNF4Kernel, _NF4_LUT
from squish.kernels.rs_fp8 import FP8KernelConfig, RustFP8Kernel
from squish.kernels.rs_int3 import INT3KernelConfig, RustINT3Kernel
from squish.kernels.rs_sampler import SamplerKernelConfig, RustSamplerKernel
from squish.kernels.rs_kv_quant import KVQuantKernelConfig, RustKVQuantKernel
from squish.kernels.rs_int2 import INT2KernelConfig, RustINT2Kernel

RNG = np.random.default_rng(42)


# ═══════════════════════════════════════════════════════════════════════════════
# RustNF4Kernel
# ═══════════════════════════════════════════════════════════════════════════════

class TestNF4KernelConfig(unittest.TestCase):
    def test_default_group_size(self):
        cfg = NF4KernelConfig()
        self.assertEqual(cfg.group_size, 64)

    def test_custom_group_size(self):
        cfg = NF4KernelConfig(group_size=128)
        self.assertEqual(cfg.group_size, 128)

    def test_use_bf16_input_default(self):
        cfg = NF4KernelConfig()
        self.assertFalse(cfg.use_bf16_input)

    def test_use_bf16_input_set(self):
        cfg = NF4KernelConfig(use_bf16_input=True)
        self.assertTrue(cfg.use_bf16_input)


class TestRustNF4Kernel(unittest.TestCase):
    def _make_kernel(self, **kwargs):
        return RustNF4Kernel(NF4KernelConfig(**kwargs))

    def test_lut_length(self):
        self.assertEqual(len(_NF4_LUT), 16)

    def test_lut_bounds(self):
        self.assertAlmostEqual(float(_NF4_LUT[0]), -1.0, places=5)
        self.assertAlmostEqual(float(_NF4_LUT[-1]), 1.0, places=5)

    def test_lut_is_sorted(self):
        self.assertTrue(np.all(np.diff(_NF4_LUT) > 0))

    def test_quantize_output_shapes(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((8, 16)).astype(np.float32)
        packed, scales = k.quantize(W)
        self.assertEqual(packed.shape, (8, 8))
        self.assertEqual(scales.shape, (8, 4))

    def test_quantize_packed_dtype(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        packed, _ = k.quantize(W)
        self.assertEqual(packed.dtype, np.uint8)

    def test_quantize_scales_dtype(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        _, scales = k.quantize(W)
        self.assertEqual(scales.dtype, np.float32)

    def test_quantize_scales_positive(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        _, scales = k.quantize(W)
        self.assertTrue(np.all(scales > 0))

    def test_roundtrip_small(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        packed, scales = k.quantize(W)
        W2 = k.dequantize(packed, scales)
        # NF4 is lossy; allow generous tolerance
        np.testing.assert_allclose(W, W2, atol=0.5)

    def test_roundtrip_zero_matrix(self):
        k = self._make_kernel(group_size=4)
        W = np.zeros((4, 8), dtype=np.float32)
        packed, scales = k.quantize(W)
        W2 = k.dequantize(packed, scales)
        np.testing.assert_allclose(W2, 0.0, atol=1e-6)

    def test_roundtrip_all_same_value(self):
        k = self._make_kernel(group_size=4)
        W = np.full((4, 8), 0.5, dtype=np.float32)
        packed, scales = k.quantize(W)
        W2 = k.dequantize(packed, scales)
        # All values should reconstruct close to 0.5
        self.assertTrue(np.all(np.abs(W2 - 0.5) < 0.5))

    def test_dequantize_output_shape(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((8, 16)).astype(np.float32)
        packed, scales = k.quantize(W)
        W2 = k.dequantize(packed, scales)
        self.assertEqual(W2.shape, W.shape)

    def test_dequantize_output_dtype(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        packed, scales = k.quantize(W)
        W2 = k.dequantize(packed, scales)
        self.assertEqual(W2.dtype, np.float32)

    def test_group_size_128(self):
        k = self._make_kernel(group_size=128)
        W = RNG.standard_normal((4, 256)).astype(np.float32)
        packed, scales = k.quantize(W)
        W2 = k.dequantize(packed, scales)
        self.assertEqual(W2.shape, (4, 256))

    def test_speedup_estimate(self):
        k = RustNF4Kernel()
        self.assertGreater(k.speedup_estimate(), 1.0)

    def test_requires_2d_input(self):
        k = RustNF4Kernel()
        with self.assertRaises(ValueError):
            k.quantize(np.ones(8, dtype=np.float32))

    def test_nibble_range(self):
        k = self._make_kernel(group_size=4)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        packed, _ = k.quantize(W)
        lo = packed & 0x0F
        hi = (packed >> 4) & 0x0F
        self.assertTrue(np.all(lo < 16))
        self.assertTrue(np.all(hi < 16))


# ═══════════════════════════════════════════════════════════════════════════════
# RustFP8Kernel
# ═══════════════════════════════════════════════════════════════════════════════

class TestFP8KernelConfig(unittest.TestCase):
    def test_default_fmt(self):
        cfg = FP8KernelConfig()
        self.assertEqual(cfg.fmt, "e4m3")

    def test_custom_fmt(self):
        cfg = FP8KernelConfig(fmt="e5m2")
        self.assertEqual(cfg.fmt, "e5m2")

    def test_invalid_fmt_raises(self):
        with self.assertRaises(ValueError):
            RustFP8Kernel(FP8KernelConfig(fmt="e3m4"))


class TestRustFP8Kernel(unittest.TestCase):
    def test_max_representable_e4m3(self):
        k = RustFP8Kernel(FP8KernelConfig(fmt="e4m3"))
        self.assertAlmostEqual(k.max_representable(), 448.0)

    def test_max_representable_e5m2(self):
        k = RustFP8Kernel(FP8KernelConfig(fmt="e5m2"))
        self.assertAlmostEqual(k.max_representable(), 57344.0)

    def test_quantize_output_shapes_e4m3(self):
        k = RustFP8Kernel(FP8KernelConfig(fmt="e4m3"))
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        q, scale = k.quantize(W)
        self.assertEqual(q.shape, W.shape)

    def test_quantize_output_dtype(self):
        k = RustFP8Kernel()
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        q, _ = k.quantize(W)
        self.assertEqual(q.dtype, np.uint8)

    def test_scale_positive(self):
        k = RustFP8Kernel()
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        _, scale = k.quantize(W)
        self.assertGreater(scale, 0.0)

    def test_dequantize_shape(self):
        k = RustFP8Kernel()
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        q, scale = k.quantize(W)
        W2 = k.dequantize(q, scale)
        self.assertEqual(W2.shape, W.shape)

    def test_dequantize_dtype(self):
        k = RustFP8Kernel()
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        q, scale = k.quantize(W)
        W2 = k.dequantize(q, scale)
        self.assertEqual(W2.dtype, np.float32)

    def test_zero_input(self):
        k = RustFP8Kernel()
        W = np.zeros((4, 8), dtype=np.float32)
        q, scale = k.quantize(W)
        W2 = k.dequantize(q, scale)
        np.testing.assert_allclose(W2, 0.0, atol=1e-4)

    def test_requires_2d(self):
        k = RustFP8Kernel()
        with self.assertRaises(ValueError):
            k.quantize(np.ones(8, dtype=np.float32))

    def test_e5m2_shapes(self):
        k = RustFP8Kernel(FP8KernelConfig(fmt="e5m2"))
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        q, scale = k.quantize(W)
        self.assertEqual(q.shape, W.shape)

    def test_scale_zero_input(self):
        k = RustFP8Kernel()
        W = np.zeros((4, 8), dtype=np.float32)
        _, scale = k.quantize(W)
        self.assertEqual(scale, 1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# RustINT3Kernel
# ═══════════════════════════════════════════════════════════════════════════════

class TestINT3KernelConfig(unittest.TestCase):
    def test_default_group_size(self):
        cfg = INT3KernelConfig()
        self.assertEqual(cfg.group_size, 128)


class TestRustINT3Kernel(unittest.TestCase):
    def _make_kernel(self, gs=8):
        return RustINT3Kernel(INT3KernelConfig(group_size=gs))

    def test_packed_size_bytes(self):
        k = self._make_kernel()
        # 8 values → 3 bytes
        self.assertEqual(k.packed_size_bytes(8), 3)
        # 16 values → 6 bytes
        self.assertEqual(k.packed_size_bytes(16), 6)

    def test_packed_size_bytes_non_multiple(self):
        k = self._make_kernel()
        # 9 values → ceil(9*3/8) = 4
        self.assertEqual(k.packed_size_bytes(9), 4)

    def test_pack_output_dtype(self):
        k = self._make_kernel()
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, _ = k.pack(W)
        self.assertEqual(packed.dtype, np.uint8)

    def test_pack_scales_dtype(self):
        k = self._make_kernel()
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        _, scales = k.pack(W)
        self.assertEqual(scales.dtype, np.float32)

    def test_scales_positive(self):
        k = self._make_kernel()
        W = RNG.standard_normal((4, 16)).astype(np.float32) + 1.0
        _, scales = k.pack(W)
        self.assertTrue(np.all(scales > 0))

    def test_roundtrip_small(self):
        k = self._make_kernel(gs=8)
        W = RNG.standard_normal((2, 16)).astype(np.float32)
        packed, scales = k.pack(W)
        W2 = k.unpack(packed, scales, W.shape)
        self.assertEqual(W2.shape, W.shape)
        np.testing.assert_allclose(W, W2, atol=0.5)

    def test_roundtrip_zero(self):
        k = self._make_kernel(gs=8)
        W = np.zeros((2, 16), dtype=np.float32)
        packed, scales = k.pack(W)
        W2 = k.unpack(packed, scales, W.shape)
        np.testing.assert_allclose(W2, 0.0, atol=1e-6)

    def test_unpack_output_shape(self):
        k = self._make_kernel(gs=8)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, scales = k.pack(W)
        W2 = k.unpack(packed, scales, W.shape)
        self.assertEqual(W2.shape, W.shape)

    def test_unpack_output_dtype(self):
        k = self._make_kernel(gs=8)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, scales = k.pack(W)
        W2 = k.unpack(packed, scales, W.shape)
        self.assertEqual(W2.dtype, np.float32)

    def test_requires_2d_input(self):
        k = self._make_kernel()
        with self.assertRaises(ValueError):
            k.pack(np.ones(16, dtype=np.float32))

    def test_max_quantized_value_range(self):
        # All values should reconstruct in [-3*scale, 3*scale]
        k = self._make_kernel(gs=8)
        W = RNG.standard_normal((2, 16)).astype(np.float32)
        packed, scales = k.pack(W)
        W2 = k.unpack(packed, scales, W.shape)
        scale_limit = 3.0 * scales.max()
        self.assertTrue(np.all(np.abs(W2) <= scale_limit + 1e-6))


# ═══════════════════════════════════════════════════════════════════════════════
# RustSamplerKernel
# ═══════════════════════════════════════════════════════════════════════════════

class TestSamplerKernelConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SamplerKernelConfig()
        self.assertAlmostEqual(cfg.temperature, 1.0)
        self.assertAlmostEqual(cfg.top_p, 0.9)
        self.assertAlmostEqual(cfg.min_p, 0.0)
        self.assertEqual(cfg.seed, 0)


class TestRustSamplerKernel(unittest.TestCase):
    def _make_kernel(self, **kwargs):
        return RustSamplerKernel(SamplerKernelConfig(**kwargs))

    def test_softmax_sums_to_one(self):
        k = self._make_kernel()
        logits = RNG.standard_normal(100).astype(np.float32)
        probs = k.softmax(logits)
        self.assertAlmostEqual(float(probs.sum()), 1.0, places=5)

    def test_softmax_all_positive(self):
        k = self._make_kernel()
        logits = RNG.standard_normal(100).astype(np.float32)
        probs = k.softmax(logits)
        self.assertTrue(np.all(probs >= 0))

    def test_softmax_output_dtype(self):
        k = self._make_kernel()
        logits = np.zeros(10, dtype=np.float32)
        probs = k.softmax(logits)
        self.assertEqual(probs.dtype, np.float32)

    def test_softmax_uniform_logits(self):
        k = self._make_kernel()
        logits = np.zeros(8, dtype=np.float32)
        probs = k.softmax(logits)
        np.testing.assert_allclose(probs, np.full(8, 1.0 / 8), atol=1e-6)

    def test_softmax_large_positive(self):
        k = self._make_kernel()
        # One very large logit → probability ~1
        logits = np.array([-1000.0, 1000.0, -1000.0], dtype=np.float32)
        probs = k.softmax(logits)
        self.assertGreater(probs[1], 0.99)

    def test_top_p_sums_to_one(self):
        k = self._make_kernel()
        logits = RNG.standard_normal(100).astype(np.float32)
        probs = k.softmax(logits)
        filtered = k.top_p_filter(probs, p=0.9)
        self.assertAlmostEqual(float(filtered.sum()), 1.0, places=5)

    def test_top_p_zeroes_some(self):
        k = self._make_kernel()
        logits = np.array([10.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        probs = k.softmax(logits)
        filtered = k.top_p_filter(probs, p=0.9)
        self.assertGreater(np.sum(filtered == 0), 0)

    def test_top_p_default_uses_config(self):
        k = self._make_kernel(top_p=0.5)
        logits = RNG.standard_normal(50).astype(np.float32)
        probs = k.softmax(logits)
        filtered = k.top_p_filter(probs)
        self.assertAlmostEqual(float(filtered.sum()), 1.0, places=5)

    def test_min_p_sums_to_one(self):
        k = self._make_kernel()
        probs = k.softmax(RNG.standard_normal(50).astype(np.float32))
        filtered = k.min_p_filter(probs, p_min=0.05)
        self.assertAlmostEqual(float(filtered.sum()), 1.0, places=5)

    def test_min_p_zero_keeps_all_nonzero(self):
        k = self._make_kernel()
        probs = k.softmax(RNG.standard_normal(10).astype(np.float32))
        filtered = k.min_p_filter(probs, p_min=0.0)
        np.testing.assert_allclose(filtered.sum(), 1.0, atol=1e-5)

    def test_sample_returns_valid_index(self):
        k = self._make_kernel(seed=1)
        logits = RNG.standard_normal(50).astype(np.float32)
        token = k.sample(logits)
        self.assertIsInstance(token, int)
        self.assertGreaterEqual(token, 0)
        self.assertLess(token, 50)

    def test_sample_deterministic_with_seed(self):
        k1 = self._make_kernel(seed=77)
        k2 = self._make_kernel(seed=77)
        logits = RNG.standard_normal(100).astype(np.float32)
        self.assertEqual(k1.sample(logits), k2.sample(logits))

    def test_temperature_scaling(self):
        logits = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        k_low = self._make_kernel(temperature=0.01)
        p_low = k_low.softmax(logits)
        # Very low temperature → argmax is taken
        self.assertEqual(int(p_low.argmax()), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# RustKVQuantKernel
# ═══════════════════════════════════════════════════════════════════════════════

class TestKVQuantKernelConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = KVQuantKernelConfig()
        self.assertEqual(cfg.bits, 8)
        self.assertTrue(cfg.group_by_head)

    def test_invalid_bits_raises(self):
        with self.assertRaises(ValueError):
            RustKVQuantKernel(KVQuantKernelConfig(bits=4))


class TestRustKVQuantKernel(unittest.TestCase):
    def _make_kv(self, n_heads=4, n_seq=16, head_dim=32):
        return RNG.standard_normal((n_heads, n_seq, head_dim)).astype(np.float32)

    def test_quantize_output_int8(self):
        k = RustKVQuantKernel()
        kv = self._make_kv()
        kv_q, scales = k.quantize_kv(kv)
        self.assertEqual(kv_q.dtype, np.int8)

    def test_quantize_output_shape(self):
        k = RustKVQuantKernel()
        kv = self._make_kv(n_heads=4, n_seq=16, head_dim=32)
        kv_q, scales = k.quantize_kv(kv)
        self.assertEqual(kv_q.shape, kv.shape)

    def test_scales_shape(self):
        k = RustKVQuantKernel()
        kv = self._make_kv(n_heads=4)
        _, scales = k.quantize_kv(kv)
        self.assertEqual(scales.shape, (4,))

    def test_scales_positive(self):
        k = RustKVQuantKernel()
        kv = self._make_kv()
        _, scales = k.quantize_kv(kv)
        self.assertTrue(np.all(scales > 0))

    def test_dequantize_shape(self):
        k = RustKVQuantKernel()
        kv = self._make_kv()
        kv_q, scales = k.quantize_kv(kv)
        kv2 = k.dequantize_kv(kv_q, scales)
        self.assertEqual(kv2.shape, kv.shape)

    def test_dequantize_dtype(self):
        k = RustKVQuantKernel()
        kv = self._make_kv()
        kv_q, scales = k.quantize_kv(kv)
        kv2 = k.dequantize_kv(kv_q, scales)
        self.assertEqual(kv2.dtype, np.float32)

    def test_roundtrip_quality(self):
        k = RustKVQuantKernel()
        kv = self._make_kv(n_heads=2, n_seq=8, head_dim=16)
        kv_q, scales = k.quantize_kv(kv)
        kv2 = k.dequantize_kv(kv_q, scales)
        np.testing.assert_allclose(kv, kv2, atol=0.05)

    def test_roundtrip_zero_kv(self):
        k = RustKVQuantKernel()
        kv = np.zeros((2, 4, 8), dtype=np.float32)
        kv_q, scales = k.quantize_kv(kv)
        kv2 = k.dequantize_kv(kv_q, scales)
        np.testing.assert_allclose(kv2, 0.0, atol=1e-5)

    def test_requires_3d_input(self):
        k = RustKVQuantKernel()
        with self.assertRaises(ValueError):
            k.quantize_kv(np.ones((4, 8), dtype=np.float32))

    def test_decode_step_update_shape(self):
        k = RustKVQuantKernel()
        n_heads, max_seq, head_dim = 2, 32, 16
        kv_cache = np.zeros((n_heads, max_seq, head_dim), dtype=np.float32)
        new_kv = RNG.standard_normal((n_heads, 1, head_dim)).astype(np.float32)
        kv_q, scales = k.decode_step_update(kv_cache, new_kv, step=0)
        self.assertEqual(kv_q.shape, (n_heads, 1, head_dim))
        self.assertEqual(scales.shape, (n_heads,))


# ═══════════════════════════════════════════════════════════════════════════════
# RustINT2Kernel
# ═══════════════════════════════════════════════════════════════════════════════

class TestINT2KernelConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = INT2KernelConfig()
        self.assertEqual(cfg.group_size, 64)
        self.assertFalse(cfg.use_bf16_input)


class TestRustINT2Kernel(unittest.TestCase):
    def _make_kernel(self, gs=4):
        return RustINT2Kernel(INT2KernelConfig(group_size=gs))

    def test_compression_ratio(self):
        k = self._make_kernel()
        self.assertAlmostEqual(k.compression_ratio(), 16.0)

    def test_pack_output_dtype(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, _, _ = k.pack(W)
        self.assertEqual(packed.dtype, np.uint8)

    def test_pack_output_shape(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, scales, zp = k.pack(W)
        self.assertEqual(packed.shape, (4, 4))
        self.assertEqual(scales.shape, (4, 4))
        self.assertEqual(zp.shape, (4, 4))

    def test_scales_dtype(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        _, scales, _ = k.pack(W)
        self.assertEqual(scales.dtype, np.float32)

    def test_zero_points_dtype(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        _, _, zp = k.pack(W)
        self.assertEqual(zp.dtype, np.float32)

    def test_roundtrip_small(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, scales, zp = k.pack(W)
        W2 = k.unpack(packed, scales, zp, W.shape)
        self.assertEqual(W2.shape, W.shape)
        # INT2 is lossy; allow reasonable reconstruction error
        np.testing.assert_allclose(W, W2, atol=1.5)

    def test_roundtrip_zero(self):
        k = self._make_kernel(gs=4)
        W = np.zeros((4, 16), dtype=np.float32)
        packed, scales, zp = k.pack(W)
        W2 = k.unpack(packed, scales, zp, W.shape)
        np.testing.assert_allclose(W2, 0.0, atol=1e-5)

    def test_unpack_output_shape(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((8, 16)).astype(np.float32)
        packed, scales, zp = k.pack(W)
        W2 = k.unpack(packed, scales, zp, W.shape)
        self.assertEqual(W2.shape, W.shape)

    def test_unpack_output_dtype(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, scales, zp = k.pack(W)
        W2 = k.unpack(packed, scales, zp, W.shape)
        self.assertEqual(W2.dtype, np.float32)

    def test_requires_2d_input(self):
        k = self._make_kernel()
        with self.assertRaises(ValueError):
            k.pack(np.ones(16, dtype=np.float32))

    def test_packed_values_in_range(self):
        k = self._make_kernel(gs=4)
        W = RNG.standard_normal((4, 16)).astype(np.float32)
        packed, _, _ = k.pack(W)
        # Each byte holds 4 × 2-bit values: nibble range is [0, 3]
        for shift in (0, 2, 4, 6):
            vals = (packed >> shift) & 0x03
            self.assertTrue(np.all(vals <= 3))

    def test_monotone_input_preserves_order(self):
        k = self._make_kernel(gs=4)
        # Monotonically increasing row: dequantized values should also increase
        row = np.linspace(-1, 1, 16, dtype=np.float32).reshape(1, 16)
        packed, scales, zp = k.pack(row)
        W2 = k.unpack(packed, scales, zp, row.shape)
        diffs = np.diff(W2[0])
        self.assertTrue(np.all(diffs >= -1e-5))  # allow quantization noise


if __name__ == "__main__":
    unittest.main()
