"""tests/test_wave63a_rust_kernels.py — Test suite for Wave 63a Rust kernel wrappers.

Covers:
- RustAQLMEncode  (aqlm_encode_f32, aqlm_kmeans_f32)
- RustBitDistiller (bit_distiller_quant_f32, bit_distiller_refine_f32)
- RustGGUFMixed   (gguf_mixed_quant_f32)
- RustPQCacheFit  (pq_cache_fit_f32)
- RustMagicPIG    (magic_pig_score_f32)
- RustMiloINT3    (milo_pack_int3_u8, milo_quant_f32)
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.kernels.rs_aqlm_encode import AQLMEncodeConfig, RustAQLMEncode
from squish.kernels.rs_bit_distiller import BitDistillerConfig, RustBitDistiller
from squish.kernels.rs_gguf_mixed import GGUFMixedConfig, RustGGUFMixed
from squish.kernels.rs_pq_cache_fit import PQCacheFitConfig, RustPQCacheFit
from squish.kernels.rs_magic_pig import MagicPIGConfig, RustMagicPIG
from squish.kernels.rs_milo_int3 import MiloINT3Config, RustMiloINT3


# ── Helpers ───────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(0xDEADBEEF)


def rand_f32(*shape) -> np.ndarray:
    return RNG.standard_normal(shape).astype(np.float32)


# ── RustAQLMEncode ────────────────────────────────────────────────────────────


class TestRustAQLMEncodeInit(unittest.TestCase):
    def setUp(self):
        self.enc = RustAQLMEncode()

    def test_default_config(self):
        self.assertIsInstance(self.enc._cfg, AQLMEncodeConfig)

    def test_custom_config(self):
        cfg = AQLMEncodeConfig(n_iter=10, seed=7)
        enc = RustAQLMEncode(cfg)
        self.assertEqual(enc._cfg.n_iter, 10)

    def test_backend_string(self):
        be = self.enc.backend()
        self.assertIn(be, {"rust", "numpy"})


class TestRustAQLMEncodeFitCodebook(unittest.TestCase):
    def setUp(self):
        self.enc = RustAQLMEncode(AQLMEncodeConfig(n_iter=5, seed=1))

    def test_output_shape(self):
        vecs = rand_f32(64, 8)
        cb = self.enc.fit_codebook(vecs, k=16)
        self.assertEqual(cb.shape, (16, 8))

    def test_output_dtype(self):
        cb = self.enc.fit_codebook(rand_f32(32, 4), k=8)
        self.assertEqual(cb.dtype, np.float32)

    def test_k_clamped_to_n(self):
        cb = self.enc.fit_codebook(rand_f32(4, 8), k=256)
        self.assertLessEqual(cb.shape[0], 4)

    def test_k_equals_1(self):
        cb = self.enc.fit_codebook(rand_f32(10, 4), k=1)
        self.assertEqual(cb.shape[0], 1)

    def test_deterministic(self):
        vecs = rand_f32(32, 4)
        cb1 = self.enc.fit_codebook(vecs, k=8, seed=42)
        cb2 = self.enc.fit_codebook(vecs, k=8, seed=42)
        np.testing.assert_array_equal(cb1, cb2)

    def test_different_seeds_differ(self):
        vecs = rand_f32(32, 4)
        cb1 = self.enc.fit_codebook(vecs, k=8, seed=1)
        cb2 = self.enc.fit_codebook(vecs, k=8, seed=99)
        self.assertFalse(np.allclose(cb1, cb2))

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.enc.fit_codebook(rand_f32(16), k=4)

    def test_invalid_k(self):
        with self.assertRaises(ValueError):
            self.enc.fit_codebook(rand_f32(16, 4), k=0)


class TestRustAQLMEncodeEncode(unittest.TestCase):
    def setUp(self):
        self.enc = RustAQLMEncode(AQLMEncodeConfig(n_iter=5, seed=1))
        self.cb = self.enc.fit_codebook(rand_f32(32, 4), k=8)

    def test_indices_shape(self):
        res = rand_f32(16, 8, 4)
        idx, upd = self.enc.encode(res, self.cb)
        self.assertEqual(idx.shape, (16, 8))

    def test_residuals_shape(self):
        res = rand_f32(16, 8, 4)
        idx, upd = self.enc.encode(res, self.cb)
        self.assertEqual(upd.shape, (16, 8, 4))

    def test_indices_dtype(self):
        res = rand_f32(4, 2, 4)
        idx, _ = self.enc.encode(res, self.cb)
        self.assertEqual(idx.dtype, np.uint16)

    def test_residuals_dtype(self):
        res = rand_f32(4, 2, 4)
        _, upd = self.enc.encode(res, self.cb)
        self.assertEqual(upd.dtype, np.float32)

    def test_indices_in_range(self):
        res = rand_f32(8, 4, 4)
        idx, _ = self.enc.encode(res, self.cb)
        self.assertTrue((idx < len(self.cb)).all())

    def test_invalid_residuals_ndim(self):
        with self.assertRaises(ValueError):
            self.enc.encode(rand_f32(16, 4), self.cb)

    def test_gs_mismatch(self):
        bad_cb = rand_f32(8, 7)  # gs=7 but residuals gs=4
        with self.assertRaises(ValueError):
            self.enc.encode(rand_f32(4, 2, 4), bad_cb)

    def test_decode_roundtrip_shape(self):
        res = rand_f32(4, 2, 4)
        idx, _ = self.enc.encode(res, self.cb)
        rec = self.enc.decode(idx, self.cb)
        self.assertEqual(rec.shape, (4, 2, 4))


# ── RustBitDistiller ──────────────────────────────────────────────────────────


class TestRustBitDistillerInit(unittest.TestCase):
    def test_default_config(self):
        bd = RustBitDistiller()
        self.assertIsInstance(bd._cfg, BitDistillerConfig)

    def test_backend_string(self):
        bd = RustBitDistiller()
        self.assertIn(bd.backend(), {"rust", "numpy"})


class TestRustBitDistillerQuantize(unittest.TestCase):
    def setUp(self):
        self.bd = RustBitDistiller(BitDistillerConfig(bits=4, group_size=32))

    def test_output_shapes(self):
        w = rand_f32(8, 64)
        q, s, z = self.bd.quantize(w)
        self.assertEqual(q.shape, (8, 64))
        self.assertEqual(len(s), 8 * 2)  # 64/32 = 2 groups per row
        self.assertEqual(len(z), 8 * 2)

    def test_quantized_dtype(self):
        q, _, _ = self.bd.quantize(rand_f32(4, 32))
        self.assertEqual(q.dtype, np.int8)

    def test_scales_positive(self):
        _, s, _ = self.bd.quantize(rand_f32(4, 32))
        self.assertTrue((s > 0).all())

    def test_quantized_range(self):
        q, _, _ = self.bd.quantize(rand_f32(4, 32), bits=4)
        levels = (1 << 4) - 1
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= levels).all())

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.bd.quantize(rand_f32(16))

    def test_uniform_input(self):
        w = np.ones((4, 32), dtype=np.float32)
        q, s, z = self.bd.quantize(w)
        self.assertEqual(q.shape, w.shape)


class TestRustBitDistillerRefine(unittest.TestCase):
    def setUp(self):
        self.bd = RustBitDistiller(BitDistillerConfig(bits=4, n_steps=2))

    def test_output_shapes(self):
        w = rand_f32(4, 128)
        teacher = rand_f32(4, 128)
        s, z = self.bd.refine(w, teacher)
        self.assertGreater(len(s), 0)
        self.assertEqual(len(s), len(z))

    def test_scales_positive(self):
        w = rand_f32(4, 128)
        s, _ = self.bd.refine(w, w.copy())
        self.assertTrue((s > 0).all())

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            self.bd.refine(rand_f32(4, 64), rand_f32(4, 32))

    def test_dequantize_shape(self):
        w = rand_f32(4, 64)
        q, s, z = self.bd.quantize(w, group_size=32)
        w_hat = self.bd.dequantize(q, s, z, group_size=32)
        self.assertEqual(w_hat.shape, w.shape)


# ── RustGGUFMixed ─────────────────────────────────────────────────────────────


class TestRustGGUFMixed(unittest.TestCase):
    def setUp(self):
        self.gguf = RustGGUFMixed(GGUFMixedConfig(bits=4, group_size=32))

    def test_backend(self):
        self.assertIn(self.gguf.backend(), {"rust", "numpy"})

    def test_output_shapes(self):
        w = rand_f32(8, 64)
        q, s, m, ss = self.gguf.quantize(w)
        self.assertEqual(q.shape, (8, 64))
        gpr = (64 + 31) // 32  # 2
        n_blocks = 8 * gpr
        self.assertEqual(len(s), n_blocks)
        self.assertEqual(len(m), n_blocks)
        self.assertGreater(len(ss), 0)

    def test_super_scales_positive(self):
        _, _, _, ss = self.gguf.quantize(rand_f32(8, 64))
        self.assertTrue((ss > 0).all())

    def test_quantized_dtype(self):
        q, _, _, _ = self.gguf.quantize(rand_f32(4, 32))
        self.assertEqual(q.dtype, np.int8)

    def test_quantized_range_bits4(self):
        q, _, _, _ = self.gguf.quantize(rand_f32(4, 32), bits=4)
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= 15).all())

    def test_dequantize_shape(self):
        w = rand_f32(4, 64)
        q, s, m, _ = self.gguf.quantize(w)
        w_hat = self.gguf.dequantize(q, s, m)
        self.assertEqual(w_hat.shape, w.shape)

    def test_dequantize_dtype(self):
        w = rand_f32(4, 32)
        q, s, m, _ = self.gguf.quantize(w)
        w_hat = self.gguf.dequantize(q, s, m)
        self.assertEqual(w_hat.dtype, np.float32)

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.gguf.quantize(rand_f32(16))

    def test_bits2(self):
        w = rand_f32(4, 32)
        q, s, m, ss = self.gguf.quantize(w, bits=2)
        self.assertTrue((q >= 0).all())
        self.assertTrue((q <= 3).all())

    def test_custom_group_size(self):
        w = rand_f32(4, 64)
        q, s, _, _ = self.gguf.quantize(w, group_size=16)
        self.assertEqual(len(s), 4 * (64 // 16))


# ── RustPQCacheFit ────────────────────────────────────────────────────────────


class TestRustPQCacheFit(unittest.TestCase):
    def setUp(self):
        self.pq = RustPQCacheFit(PQCacheFitConfig(n_iters=5, seed=42))

    def test_backend(self):
        self.assertIn(self.pq.backend(), {"rust", "numpy"})

    def test_fit_output_shape(self):
        sv = rand_f32(64, 8)
        cb = self.pq.fit(sv, k=16)
        self.assertEqual(cb.shape, (16, 8))

    def test_fit_dtype(self):
        cb = self.pq.fit(rand_f32(32, 4), k=8)
        self.assertEqual(cb.dtype, np.float32)

    def test_k_clamped(self):
        cb = self.pq.fit(rand_f32(4, 8), k=256)
        self.assertLessEqual(cb.shape[0], 4)

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

    def test_encode_dtype(self):
        sv = rand_f32(16, 4)
        cb = self.pq.fit(sv, k=4)
        idx = self.pq.encode(sv, cb)
        self.assertEqual(idx.dtype, np.int32)

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

    def test_sub_dim_mismatch(self):
        sv = rand_f32(16, 4)
        cb = rand_f32(8, 8)  # wrong sub_dim
        with self.assertRaises(ValueError):
            self.pq.encode(sv, cb)


# ── RustMagicPIG ─────────────────────────────────────────────────────────────


class TestRustMagicPIG(unittest.TestCase):
    def setUp(self):
        self.pig = RustMagicPIG(MagicPIGConfig())

    def test_backend(self):
        self.assertIn(self.pig.backend(), {"rust", "numpy"})

    def test_score_output_shape(self):
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

    def test_single_head_single_query(self):
        Q = rand_f32(1, 1, 4)
        K = rand_f32(1, 4, 4)
        V = rand_f32(1, 4, 4)
        out = self.pig.score(Q, K, V)
        self.assertEqual(out.shape, (1, 1, 4))

    def test_score_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.pig.score(rand_f32(4, 8), rand_f32(4, 8), rand_f32(4, 8))

    def test_score_head_mismatch(self):
        with self.assertRaises(ValueError):
            self.pig.score(rand_f32(2, 1, 4), rand_f32(3, 8, 4), rand_f32(3, 8, 4))

    def test_score_kv_len_mismatch(self):
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

    def test_output_norm_bounded(self):
        # Attention output should have bounded norm (entries weighted sum of V)
        V = np.ones((2, 8, 4), dtype=np.float32)
        out = self.pig.score(rand_f32(2, 2, 4), rand_f32(2, 8, 4), V)
        # Output = weighted avg of V rows → should be close to 1.0 per entry
        self.assertTrue(np.abs(out).max() < 10.0)


# ── RustMiloINT3 ─────────────────────────────────────────────────────────────


class TestRustMiloINT3(unittest.TestCase):
    def setUp(self):
        self.milo = RustMiloINT3(MiloINT3Config(group_size=64))

    def test_backend(self):
        self.assertIn(self.milo.backend(), {"rust", "numpy"})

    def test_quantize_shapes(self):
        w = rand_f32(8, 128)
        q, s, z = self.milo.quantize(w)
        self.assertEqual(q.shape, (8, 128))
        self.assertEqual(len(s), 8 * 2)  # 128/64 = 2 groups per row
        self.assertEqual(len(z), 8 * 2)

    def test_quantized_dtype(self):
        q, _, _ = self.milo.quantize(rand_f32(4, 64))
        self.assertEqual(q.dtype, np.int8)

    def test_scales_positive(self):
        _, s, _ = self.milo.quantize(rand_f32(4, 64))
        self.assertTrue((s > 0).all())

    def test_zeros_all_zero(self):
        _, _, z = self.milo.quantize(rand_f32(4, 64))
        np.testing.assert_array_equal(z, np.zeros_like(z))

    def test_quantized_range(self):
        q, _, _ = self.milo.quantize(rand_f32(4, 64))
        self.assertTrue((q >= -3).all())
        self.assertTrue((q <= 3).all())

    def test_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.milo.quantize(rand_f32(16))

    def test_uniform_weights(self):
        w = np.ones((4, 64), dtype=np.float32) * 2.0
        q, s, _ = self.milo.quantize(w)
        # All values should quantise to 3 (max INT3 positive)
        self.assertTrue((q == 3).all())

    def test_pack_output_length(self):
        vals = np.zeros(8, dtype=np.int8)
        packed = self.milo.pack(vals)
        self.assertEqual(len(packed), 3)  # 8 * 3 / 8 = 3

    def test_pack_non_multiple_length(self):
        vals = np.zeros(5, dtype=np.int8)
        packed = self.milo.pack(vals)
        expected = (5 * 3 + 7) // 8
        self.assertEqual(len(packed), expected)

    def test_pack_dtype(self):
        packed = self.milo.pack(np.zeros(8, dtype=np.int8))
        self.assertEqual(packed.dtype, np.uint8)

    def test_pack_invalid_ndim(self):
        with self.assertRaises(ValueError):
            self.milo.pack(np.zeros((4, 4), dtype=np.int8))

    def test_pack_unpack_roundtrip(self):
        vals = np.array([1, -1, 2, -2, 3, -3, 0, 1], dtype=np.int8)
        packed = self.milo.pack(vals)
        recovered = self.milo.unpack(packed, len(vals))
        np.testing.assert_array_equal(vals, recovered)

    def test_unpack_length(self):
        n = 16
        vals = np.zeros(n, dtype=np.int8)
        packed = self.milo.pack(vals)
        recovered = self.milo.unpack(packed, n)
        self.assertEqual(len(recovered), n)

    def test_dequantize_shape(self):
        w = rand_f32(4, 128)
        q, s, _ = self.milo.quantize(w)
        w_hat = self.milo.dequantize(q, s)
        self.assertEqual(w_hat.shape, w.shape)

    def test_dequantize_dtype(self):
        w = rand_f32(4, 64)
        q, s, _ = self.milo.quantize(w)
        w_hat = self.milo.dequantize(q, s)
        self.assertEqual(w_hat.dtype, np.float32)

    def test_roundtrip_error_bounded(self):
        w = (RNG.standard_normal((8, 64)) * 0.5).astype(np.float32)
        q, s, _ = self.milo.quantize(w)
        w_hat = self.milo.dequantize(q, s)
        # INT3 has low precision; allow up to 50% relative error
        rel_err = np.abs(w_hat - w).mean() / (np.abs(w).mean() + 1e-6)
        self.assertLess(rel_err, 0.5)

    def test_pack_all_zeros(self):
        vals = np.zeros(24, dtype=np.int8)
        packed = self.milo.pack(vals)
        np.testing.assert_array_equal(packed, np.zeros(9, dtype=np.uint8))

    def test_quantize_float_zero(self):
        w = np.zeros((4, 64), dtype=np.float32)
        q, s, _ = self.milo.quantize(w)
        np.testing.assert_array_equal(q, np.zeros_like(q))


if __name__ == "__main__":
    unittest.main()
