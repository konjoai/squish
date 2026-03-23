"""tests/test_wave63b_mojo_kernels.py — Test suite for Wave 63b Mojo kernel wrappers.

Covers:
- MojoAQLMEncode  (aqlm_encode_kernel, aqlm_kmeans_kernel stubs, NumPy fallback)
- MojoBitDistiller (bit_distiller_quant_kernel stub, NumPy fallback)
- MojoGGUFMixed   (gguf_mixed_quant_kernel stub, NumPy fallback)
- MojoPQCacheFit  (pq_cache_fit_kernel stub, NumPy fallback)
- MojoMagicPIG    (magic_pig_score_kernel stub, NumPy fallback)
- MojoMiloINT3    (milo_int3_pack_kernel + milo_quant_kernel stubs, NumPy fallback)

All Mojo kernels are stubs (not yet compiled) so all tests run the NumPy
fallback path.  Tests verify shapes, dtypes, value correctness, error handling,
and that the wrapper classes are importable and functional.
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.kernels.mojo.aqlm_encode_mojo import AQLMEncodeMojoConfig, MojoAQLMEncode
from squish.kernels.mojo.bit_distiller_mojo import BitDistillerMojoConfig, MojoBitDistiller
from squish.kernels.mojo.gguf_mixed_mojo import GGUFMixedMojoConfig, MojoGGUFMixed
from squish.kernels.mojo.pq_cache_fit_mojo import PQCacheFitMojoConfig, MojoPQCacheFit
from squish.kernels.mojo.magic_pig_mojo import MagicPIGMojoConfig, MojoMagicPIG
from squish.kernels.mojo.milo_int3_mojo import MiloINT3MojoConfig, MojoMiloINT3


# ── Helpers ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0xC0FFEE)


def rand_f32(*shape) -> np.ndarray:
    return RNG.standard_normal(shape).astype(np.float32)


# ── MojoAQLMEncode ────────────────────────────────────────────────────────────


class TestMojoAQLMEncodeInit(unittest.TestCase):
    def test_default_config(self):
        enc = MojoAQLMEncode()
        self.assertIsInstance(enc._cfg, AQLMEncodeMojoConfig)

    def test_custom_config(self):
        enc = MojoAQLMEncode(AQLMEncodeMojoConfig(n_iter=3, seed=9))
        self.assertEqual(enc._cfg.n_iter, 3)

    def test_backend_fallback(self):
        enc = MojoAQLMEncode()
        self.assertIn(enc.backend(), {"mojo", "numpy"})


class TestMojoAQLMEncodeFitCodebook(unittest.TestCase):
    def setUp(self):
        self.enc = MojoAQLMEncode(AQLMEncodeMojoConfig(n_iter=5, seed=1))

    def test_output_shape(self):
        vecs = rand_f32(32, 4)
        cb = self.enc.fit_codebook(vecs, k=8)
        self.assertEqual(cb.dtype, np.float32)
        self.assertEqual(cb.shape[1], 4)

    def test_k_clamped(self):
        cb = self.enc.fit_codebook(rand_f32(4, 4), k=256)
        self.assertLessEqual(cb.shape[0], 4)

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.enc.fit_codebook(rand_f32(16), k=4)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            self.enc.fit_codebook(rand_f32(16, 4), k=0)


class TestMojoAQLMEncodeEncode(unittest.TestCase):
    def setUp(self):
        self.enc = MojoAQLMEncode(AQLMEncodeMojoConfig(n_iter=5, seed=1))
        self.cb = self.enc.fit_codebook(rand_f32(32, 4), k=8)

    def test_encode_indices_shape(self):
        res = rand_f32(8, 4, 4)
        idx, upd = self.enc.encode(res, self.cb)
        self.assertEqual(idx.shape, (8, 4))

    def test_encode_residuals_shape(self):
        res = rand_f32(8, 4, 4)
        _, upd = self.enc.encode(res, self.cb)
        self.assertEqual(upd.shape, (8, 4, 4))

    def test_encode_indices_in_range(self):
        res = rand_f32(4, 2, 4)
        idx, _ = self.enc.encode(res, self.cb)
        self.assertTrue((idx < len(self.cb)).all())

    def test_decode_shape(self):
        res = rand_f32(4, 2, 4)
        idx, _ = self.enc.encode(res, self.cb)
        rec = self.enc.decode(idx, self.cb)
        self.assertEqual(rec.shape, (4, 2, 4))


# ── MojoBitDistiller ──────────────────────────────────────────────────────────


class TestMojoBitDistillerInit(unittest.TestCase):
    def test_default_config(self):
        bd = MojoBitDistiller()
        self.assertIsInstance(bd._cfg, BitDistillerMojoConfig)

    def test_backend_fallback(self):
        self.assertIn(MojoBitDistiller().backend(), {"mojo", "numpy"})


class TestMojoBitDistillerQuantize(unittest.TestCase):
    def setUp(self):
        self.bd = MojoBitDistiller(BitDistillerMojoConfig(bits=4, group_size=32))

    def test_output_shapes(self):
        w = rand_f32(8, 64)
        q, s, z = self.bd.quantize(w)
        self.assertEqual(q.shape, (8, 64))
        self.assertEqual(len(s), 8 * 2)

    def test_quantized_dtype(self):
        q, _, _ = self.bd.quantize(rand_f32(4, 32))
        self.assertEqual(q.dtype, np.int8)

    def test_scales_positive(self):
        _, s, _ = self.bd.quantize(rand_f32(4, 32))
        self.assertTrue((s > 0).all())

    def test_quantized_range_bits4(self):
        q, _, _ = self.bd.quantize(rand_f32(4, 32), bits=4)
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= 15).all())

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.bd.quantize(rand_f32(32))

    def test_dequantize_shape(self):
        w = rand_f32(4, 64)
        q, s, z = self.bd.quantize(w)
        w_hat = self.bd.dequantize(q, s, z)
        self.assertEqual(w_hat.shape, w.shape)

    def test_dequantize_dtype(self):
        w = rand_f32(4, 32)
        q, s, z = self.bd.quantize(w)
        self.assertEqual(self.bd.dequantize(q, s, z).dtype, np.float32)

    def test_bits2(self):
        w = rand_f32(4, 32)
        q, _, _ = self.bd.quantize(w, bits=2)
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= 3).all())


# ── MojoGGUFMixed ─────────────────────────────────────────────────────────────


class TestMojoGGUFMixed(unittest.TestCase):
    def setUp(self):
        self.gguf = MojoGGUFMixed(GGUFMixedMojoConfig(bits=4, group_size=32))

    def test_backend(self):
        self.assertIn(self.gguf.backend(), {"mojo", "numpy"})

    def test_output_shapes(self):
        w = rand_f32(8, 64)
        q, s, m, ss = self.gguf.quantize(w)
        self.assertEqual(q.shape, (8, 64))
        gpr = (64 + 31) // 32
        self.assertEqual(len(s), 8 * gpr)
        self.assertEqual(len(m), 8 * gpr)
        self.assertGreater(len(ss), 0)

    def test_quantized_dtype(self):
        q, _, _, _ = self.gguf.quantize(rand_f32(4, 32))
        self.assertEqual(q.dtype, np.int8)

    def test_super_scales_positive(self):
        _, _, _, ss = self.gguf.quantize(rand_f32(8, 64))
        self.assertTrue((ss > 0).all())

    def test_quantized_4bit_range(self):
        q, _, _, _ = self.gguf.quantize(rand_f32(4, 32), bits=4)
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= 15).all())

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.gguf.quantize(rand_f32(32))

    def test_dequantize_shape(self):
        w = rand_f32(4, 64)
        q, s, m, _ = self.gguf.quantize(w)
        w_hat = self.gguf.dequantize(q, s, m)
        self.assertEqual(w_hat.shape, w.shape)

    def test_dequantize_dtype(self):
        w = rand_f32(4, 32)
        q, s, m, _ = self.gguf.quantize(w)
        self.assertEqual(self.gguf.dequantize(q, s, m).dtype, np.float32)


# ── MojoPQCacheFit ────────────────────────────────────────────────────────────


class TestMojoPQCacheFit(unittest.TestCase):
    def setUp(self):
        self.pq = MojoPQCacheFit(PQCacheFitMojoConfig(n_iters=5, seed=42))

    def test_backend(self):
        self.assertIn(self.pq.backend(), {"mojo", "numpy"})

    def test_fit_shape(self):
        sv = rand_f32(64, 8)
        cb = self.pq.fit(sv, k=16)
        self.assertEqual(cb.shape[1], 8)
        self.assertLessEqual(cb.shape[0], 16)

    def test_fit_dtype(self):
        cb = self.pq.fit(rand_f32(32, 4), k=8)
        self.assertEqual(cb.dtype, np.float32)

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.pq.fit(rand_f32(16), k=4)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            self.pq.fit(rand_f32(16, 4), k=0)

    def test_encode_shape(self):
        sv = rand_f32(32, 4)
        cb = self.pq.fit(sv, k=8)
        idx = self.pq.encode(sv, cb)
        self.assertEqual(idx.shape, (32,))

    def test_encode_in_range(self):
        sv = rand_f32(32, 4)
        cb = self.pq.fit(sv, k=8)
        idx = self.pq.encode(sv, cb)
        self.assertTrue((idx >= 0).all())
        self.assertTrue((idx < 8).all())

    def test_decode_shape(self):
        sv = rand_f32(16, 4)
        cb = self.pq.fit(sv, k=4)
        idx = self.pq.encode(sv, cb)
        rec = self.pq.decode(idx, cb)
        self.assertEqual(rec.shape, (16, 4))


# ── MojoMagicPIG ─────────────────────────────────────────────────────────────


class TestMojoMagicPIG(unittest.TestCase):
    def setUp(self):
        self.pig = MojoMagicPIG(MagicPIGMojoConfig())

    def test_backend(self):
        self.assertIn(self.pig.backend(), {"mojo", "numpy"})

    def test_score_shape(self):
        Q = rand_f32(4, 2, 8)
        K = rand_f32(4, 16, 8)
        V = rand_f32(4, 16, 8)
        out = self.pig.score(Q, K, V)
        self.assertEqual(out.shape, (4, 2, 8))

    def test_score_dtype(self):
        out = self.pig.score(rand_f32(2, 1, 4), rand_f32(2, 8, 4), rand_f32(2, 8, 4))
        self.assertEqual(out.dtype, np.float32)

    def test_score_finite(self):
        out = self.pig.score(rand_f32(2, 3, 4), rand_f32(2, 8, 4), rand_f32(2, 8, 4))
        self.assertTrue(np.isfinite(out).all())

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.pig.score(rand_f32(4, 8), rand_f32(4, 8), rand_f32(4, 8))

    def test_head_mismatch(self):
        with self.assertRaises(ValueError):
            self.pig.score(rand_f32(2, 1, 4), rand_f32(3, 8, 4), rand_f32(3, 8, 4))

    def test_kv_len_mismatch(self):
        with self.assertRaises(ValueError):
            self.pig.score(rand_f32(2, 1, 4), rand_f32(2, 8, 4), rand_f32(2, 6, 4))

    def test_attention_weights_shape(self):
        Q = rand_f32(2, 3, 4)
        K = rand_f32(2, 8, 4)
        w = self.pig.attention_weights(Q, K)
        self.assertEqual(w.shape, (2, 3, 8))

    def test_attention_weights_sum_to_one(self):
        Q = rand_f32(1, 2, 8)
        K = rand_f32(1, 16, 8)
        w = self.pig.attention_weights(Q, K)
        np.testing.assert_allclose(w.sum(axis=-1), np.ones((1, 2)), atol=1e-5)

    def test_attention_weights_non_negative(self):
        w = self.pig.attention_weights(rand_f32(2, 4, 8), rand_f32(2, 16, 8))
        self.assertTrue((w >= 0).all())

    def test_uniform_values_output(self):
        # If V is uniform, output should match V regardless of weights
        Q = rand_f32(1, 1, 4)
        K = rand_f32(1, 8, 4)
        V = np.ones((1, 8, 4), dtype=np.float32)
        out = self.pig.score(Q, K, V)
        np.testing.assert_allclose(out, np.ones((1, 1, 4)), atol=1e-5)


# ── MojoMiloINT3 ─────────────────────────────────────────────────────────────


class TestMojoMiloINT3(unittest.TestCase):
    def setUp(self):
        self.milo = MojoMiloINT3(MiloINT3MojoConfig(group_size=64))

    def test_backend(self):
        self.assertIn(self.milo.backend(), {"mojo", "numpy"})

    def test_quantize_shapes(self):
        w = rand_f32(8, 128)
        q, s, z = self.milo.quantize(w)
        self.assertEqual(q.shape, (8, 128))
        self.assertEqual(len(s), 8 * 2)

    def test_quantized_dtype(self):
        q, _, _ = self.milo.quantize(rand_f32(4, 64))
        self.assertEqual(q.dtype, np.int8)

    def test_scales_positive(self):
        _, s, _ = self.milo.quantize(rand_f32(4, 64))
        self.assertTrue((s > 0).all())

    def test_quantized_range(self):
        q, _, _ = self.milo.quantize(rand_f32(4, 64))
        self.assertTrue((q >= -3).all())
        self.assertTrue((q <= 3).all())

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.milo.quantize(rand_f32(64))

    def test_pack_output_length(self):
        vals = np.zeros(8, dtype=np.int8)
        packed = self.milo.pack(vals)
        self.assertEqual(len(packed), 3)

    def test_pack_dtype(self):
        packed = self.milo.pack(np.zeros(8, dtype=np.int8))
        self.assertEqual(packed.dtype, np.uint8)

    def test_pack_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.milo.pack(np.zeros((4, 4), dtype=np.int8))

    def test_pack_unpack_roundtrip_small(self):
        vals = np.array([0, 1, 2, 3, -1, -2, -3, 0], dtype=np.int8)
        packed = self.milo.pack(vals)
        recovered = self.milo.unpack(packed, len(vals))
        np.testing.assert_array_equal(vals, recovered)

    def test_pack_unpack_roundtrip_16(self):
        vals = np.array([1, -1, 2, -2, 3, -3, 0, 1, -1, 2, -2, 3, -3, 0, 1, -1], dtype=np.int8)
        packed = self.milo.pack(vals)
        recovered = self.milo.unpack(packed, 16)
        np.testing.assert_array_equal(vals, recovered)

    def test_pack_all_zeros(self):
        packed = self.milo.pack(np.zeros(8, dtype=np.int8))
        np.testing.assert_array_equal(packed, np.zeros(3, dtype=np.uint8))

    def test_dequantize_shape(self):
        w = rand_f32(4, 128)
        q, s, _ = self.milo.quantize(w)
        w_hat = self.milo.dequantize(q, s)
        self.assertEqual(w_hat.shape, w.shape)

    def test_dequantize_dtype(self):
        w = rand_f32(4, 64)
        q, s, _ = self.milo.quantize(w)
        self.assertEqual(self.milo.dequantize(q, s).dtype, np.float32)

    def test_roundtrip_error_bounded(self):
        w = (RNG.standard_normal((8, 64)) * 0.5).astype(np.float32)
        q, s, _ = self.milo.quantize(w)
        w_hat = self.milo.dequantize(q, s)
        rel = np.abs(w_hat - w).mean() / (np.abs(w).mean() + 1e-6)
        self.assertLess(rel, 0.5)

    def test_uniform_quant_max(self):
        w = np.ones((4, 64), dtype=np.float32) * 2.0
        q, _, _ = self.milo.quantize(w)
        self.assertTrue((q == 3).all())

    def test_negative_uniform_quant_min(self):
        w = np.ones((4, 64), dtype=np.float32) * -2.0
        q, _, _ = self.milo.quantize(w)
        self.assertTrue((q == -3).all())


if __name__ == "__main__":
    unittest.main()
