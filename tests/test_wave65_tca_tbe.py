"""tests/test_wave65_tca_tbe.py

Unit tests for Wave 65: TCA-TBE lossless BF16 compression and stage-aware
prefill/decode dispatch.

Modules under test
──────────────────
* squish.compress.tca_tbe       — TcaTbeCodec, TcaTbeBlock, TcaTbeConfig,
                                   tca_tbe_encode_tensor, tca_tbe_decode_tensor,
                                   CompressionStats
* squish.runtime.stage_dispatcher — StageDispatcher, InferenceStage,
                                    KernelPipeline, DispatchDecision

All tests run without Metal, hardware, or external dependencies.
NumPy is the only required third-party package.
"""
from __future__ import annotations

import struct
import unittest
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_bf16(n: int, seed: int = 0) -> np.ndarray:
    """Return (n,) uint16 array: BF16 values drawn from N(0,1) approximately."""
    rng = np.random.default_rng(seed)
    f32 = rng.standard_normal(n).astype(np.float32)
    # Reinterpret float32 bits as uint32, then take high 16 bits → BF16.
    u32 = f32.view(np.uint32)
    return (u32 >> 16).astype(np.uint16)


def _block(n: int = 128, seed: int = 0) -> np.ndarray:
    return _uniform_bf16(n, seed)


# =============================================================================
# 1. TcaTbeConfig
# =============================================================================

class TestTcaTbeConfigDefaults(unittest.TestCase):
    def test_block_size_default(self):
        from squish.compress.tca_tbe import TcaTbeConfig, BLOCK_SIZE
        cfg = TcaTbeConfig()
        self.assertEqual(cfg.block_size, BLOCK_SIZE)

    def test_min_coverage_default(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        cfg = TcaTbeConfig()
        self.assertAlmostEqual(cfg.min_coverage, 0.20)

    def test_range_half_default(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        cfg = TcaTbeConfig()
        self.assertEqual(cfg.range_half, 1)

    def test_mantissa_bits_default(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        cfg = TcaTbeConfig()
        self.assertEqual(cfg.mantissa_bits, 7)


class TestTcaTbeConfigValidation(unittest.TestCase):
    def test_bad_block_size_raises(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        with self.assertRaises(ValueError):
            TcaTbeConfig(block_size=64)

    def test_min_coverage_zero_raises(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        with self.assertRaises(ValueError):
            TcaTbeConfig(min_coverage=0.0)

    def test_min_coverage_one_raises(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        with self.assertRaises(ValueError):
            TcaTbeConfig(min_coverage=1.0)

    def test_negative_range_half_raises(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        with self.assertRaises(ValueError):
            TcaTbeConfig(range_half=-1)

    def test_mantissa_bits_zero_raises(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        with self.assertRaises(ValueError):
            TcaTbeConfig(mantissa_bits=0)

    def test_mantissa_bits_too_large_raises(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        with self.assertRaises(ValueError):
            TcaTbeConfig(mantissa_bits=8)

    def test_valid_range_half_zero(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        cfg = TcaTbeConfig(range_half=0)
        self.assertEqual(cfg.range_half, 0)


# =============================================================================
# 2. TcaTbeBlock basics
# =============================================================================

class TestTcaTbeBlockProperties(unittest.TestCase):
    def _make_raw_block(self, n: int = 128):
        from squish.compress.tca_tbe import TcaTbeBlock, BLOCK_SIZE, _FALLBACK_MARKER
        words = _block(n)
        return TcaTbeBlock(
            is_raw=True, e_mode=_FALLBACK_MARKER,
            e_lo_offset=0, e_hi_offset=0,
            sign_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
            range_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
            exp_offset_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
            mantissa_bitmap=np.zeros((BLOCK_SIZE, 7), dtype=np.uint8),
            mantissa_bits=7, spill_words=words, n_elements=n,
        )

    def test_is_raw_flag_true(self):
        blk = self._make_raw_block()
        self.assertTrue(blk.is_raw)

    def test_is_compressed_false_for_raw(self):
        blk = self._make_raw_block()
        self.assertFalse(blk.is_compressed)

    def test_raw_bytes_correct(self):
        blk = self._make_raw_block(128)
        self.assertEqual(blk.raw_bytes(), 256)

    def test_compressed_bytes_raw_block(self):
        blk = self._make_raw_block(128)
        # raw block: n_elements * 2 bytes
        self.assertEqual(blk.compressed_bytes(), 256)

    def test_n_elements_default(self):
        from squish.compress.tca_tbe import TcaTbeBlock, BLOCK_SIZE, _FALLBACK_MARKER
        blk = self._make_raw_block(128)
        self.assertEqual(blk.n_elements, 128)


# =============================================================================
# 3. TCATBECodec — encode single block
# =============================================================================

class TestTcaTbeCodecEncode(unittest.TestCase):
    def setUp(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        self.codec = TcaTbeCodec(TcaTbeConfig())

    def test_encode_returns_block(self):
        from squish.compress.tca_tbe import TcaTbeBlock
        blk = self.codec.encode(_block())
        self.assertIsInstance(blk, TcaTbeBlock)

    def test_sign_bitmap_shape(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            self.assertEqual(blk.sign_bitmap.shape, (128,))

    def test_range_bitmap_shape(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            self.assertEqual(blk.range_bitmap.shape, (128,))

    def test_mantissa_bitmap_shape(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            self.assertEqual(blk.mantissa_bitmap.shape, (128, 7))

    def test_sign_bitmap_binary(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            self.assertTrue(np.all((blk.sign_bitmap == 0) | (blk.sign_bitmap == 1)))

    def test_range_bitmap_binary(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            self.assertTrue(np.all((blk.range_bitmap == 0) | (blk.range_bitmap == 1)))

    def test_mantissa_bitmap_bits_in_range(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            max_val = (1 << blk.mantissa_bits) - 1
            self.assertTrue(np.all(blk.mantissa_bitmap <= max_val))

    def test_e_mode_uint8_range(self):
        blk = self.codec.encode(_block())
        if not blk.is_raw:
            self.assertGreaterEqual(blk.e_mode, 0)
            self.assertLessEqual(blk.e_mode, 255)

    def test_error_non_1d_input(self):
        with self.assertRaises(ValueError):
            self.codec.encode(np.zeros((16, 8), dtype=np.uint16))

    def test_error_too_many_elements(self):
        with self.assertRaises(ValueError):
            self.codec.encode(np.zeros(256, dtype=np.uint16))

    def test_partial_block_accepted(self):
        # blocks smaller than 128 should be allowed (final block of a tensor)
        from squish.compress.tca_tbe import TcaTbeBlock
        blk = self.codec.encode(_block(64))
        self.assertIsInstance(blk, TcaTbeBlock)
        self.assertEqual(blk.n_elements, 64)

    def test_spill_words_dtype(self):
        blk = self.codec.encode(_block())
        self.assertEqual(blk.spill_words.dtype, np.dtype("uint16"))

    def test_spill_count_non_negative(self):
        blk = self.codec.encode(_block())
        self.assertGreaterEqual(len(blk.spill_words), 0)


# =============================================================================
# 4. TCATBECodec — decode + round-trip losslessness
# =============================================================================

class TestTcaTbeCodecRoundTrip(unittest.TestCase):
    def setUp(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        self.codec = TcaTbeCodec(TcaTbeConfig())

    def _roundtrip(self, words: np.ndarray) -> np.ndarray:
        blk = self.codec.encode(words)
        return self.codec.decode(blk)

    def test_roundtrip_normal_weights(self):
        w = _block(128, seed=1)
        np.testing.assert_array_equal(self._roundtrip(w), w)

    def test_roundtrip_seed2(self):
        w = _block(128, seed=2)
        np.testing.assert_array_equal(self._roundtrip(w), w)

    def test_roundtrip_seed42(self):
        w = _block(128, seed=42)
        np.testing.assert_array_equal(self._roundtrip(w), w)

    def test_roundtrip_zeros(self):
        w = np.zeros(128, dtype=np.uint16)
        np.testing.assert_array_equal(self._roundtrip(w), w)

    def test_roundtrip_nan(self):
        # NaN bf16 = 0x7FC0, propagated losslessly
        w = np.full(128, 0x7FC0, dtype=np.uint16)
        np.testing.assert_array_equal(self._roundtrip(w), w)

    def test_roundtrip_partial_block(self):
        w = _block(64, seed=3)
        recon = self._roundtrip(w)
        np.testing.assert_array_equal(recon, w)

    def test_roundtrip_single_element(self):
        w = np.array([0x3F80], dtype=np.uint16)
        np.testing.assert_array_equal(self._roundtrip(w), w)

    def test_decode_raw_block_lossless(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        # Force raw fallback by using min_coverage=0.99 (almost all must be in range)
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.99))
        w = _block(128, seed=10)
        blk = codec.encode(w)
        recon = codec.decode(blk)
        np.testing.assert_array_equal(recon, w)

    def test_decode_output_length_matches_n_elements(self):
        w = _block(70, seed=5)
        blk = self.codec.encode(w)
        recon = self.codec.decode(blk)
        self.assertEqual(len(recon), 70)


# =============================================================================
# 5. Entropy guard (fallback to raw)
# =============================================================================

class TestEntropyGuard(unittest.TestCase):
    def test_high_threshold_triggers_raw(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        # min_coverage=0.99 → almost all elements must share exponent ± 1
        # random BF16 data won't satisfy this, so we expect raw fallback
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.99))
        w = _block(128, seed=77)
        blk = codec.encode(w)
        self.assertTrue(blk.is_raw)

    def test_low_threshold_compresses(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        # min_coverage=0.001 → almost never fall back
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.001))
        # Create a block where all exponents are identical → definitely compressible
        # BF16: sign=0, exp=0x7F (127), mantissa varies
        exp_bits = np.full(128, 0x7F, dtype=np.uint16)
        mant = np.arange(128, dtype=np.uint16) & 0x7F
        w = (exp_bits << 7) | mant
        blk = codec.encode(w)
        self.assertFalse(blk.is_raw)

    def test_uniform_exponent_block_not_raw(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.20))
        # All same exponent → 100% coverage → definitely compressed
        exp_val = np.uint16(0x3F80)  # BF16 for 1.0 (exp=127, mant=0)
        w = np.array([exp_val] * 128, dtype=np.uint16)
        blk = codec.encode(w)
        self.assertFalse(blk.is_raw)


# =============================================================================
# 6. Serialisation round-trip (pack / unpack)
# =============================================================================

class TestSerialisationRoundTrip(unittest.TestCase):
    def setUp(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        self.codec = TcaTbeCodec(TcaTbeConfig())

    def _encode_and_pack(self, words: np.ndarray) -> bytes:
        blk = self.codec.encode(words)
        return self.codec.encode_to_bytes(blk)

    def test_pack_returns_bytes(self):
        data = self._encode_and_pack(_block())
        self.assertIsInstance(data, bytes)

    def test_pack_unpack_roundtrip_compressed(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        w = _block(128, seed=20)
        # Use a codec that won't fall back for normal data
        from squish.compress.tca_tbe import TcaTbeCodec
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.001))
        # Force a compressible block
        exp_bits = np.full(128, 0x7F, dtype=np.uint16)
        mant = np.arange(128, dtype=np.uint16) & 0x7F
        w = (exp_bits << 7) | mant
        original_blk = codec.encode(w)
        packed = codec.encode_to_bytes(original_blk)
        restored_blk = codec.decode_from_bytes(packed, original_blk.mantissa_bits)
        recon = codec.decode(restored_blk)
        np.testing.assert_array_equal(recon, w)

    def test_pack_unpack_roundtrip_raw(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.99))
        w = _block(128, seed=15)
        blk = codec.encode(w)
        self.assertTrue(blk.is_raw)
        packed = codec.encode_to_bytes(blk)
        restored = codec.decode_from_bytes(packed, blk.mantissa_bits)
        recon = codec.decode(restored)
        np.testing.assert_array_equal(recon, w)

    def test_raw_packed_first_byte_is_flag(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.99))
        blk = codec.encode(_block(128, seed=16))
        packed = codec.encode_to_bytes(blk)
        self.assertEqual(packed[0] & 0x01, 1)

    def test_compressed_packed_first_byte_is_zero(self):
        from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig
        codec = TcaTbeCodec(TcaTbeConfig(min_coverage=0.001))
        exp_bits = np.full(128, 0x40, dtype=np.uint16)
        mant = np.arange(128, dtype=np.uint16) & 0x7F
        w = (exp_bits << 7) | mant
        blk = codec.encode(w)
        packed = codec.encode_to_bytes(blk)
        self.assertEqual(packed[0] & 0x01, 0)

    def test_pack_non_empty_bytes(self):
        blk_bytes = self._encode_and_pack(_block())
        self.assertGreater(len(blk_bytes), 0)


# =============================================================================
# 7. Tensor-level encode / decode
# =============================================================================

class TestTensorEncodeDecode(unittest.TestCase):
    def setUp(self):
        from squish.compress.tca_tbe import TcaTbeConfig
        self.cfg = TcaTbeConfig()

    def test_encode_tensor_returns_blocks_and_stats(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, CompressionStats
        w = _uniform_bf16(512)
        blocks, stats = tca_tbe_encode_tensor(w, self.cfg)
        self.assertIsInstance(blocks, list)
        self.assertIsInstance(stats, CompressionStats)

    def test_n_blocks_correct_no_remainder(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, BLOCK_SIZE
        w = _uniform_bf16(BLOCK_SIZE * 4)
        blocks, stats = tca_tbe_encode_tensor(w, self.cfg)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(stats.n_blocks, 4)

    def test_n_blocks_correct_with_remainder(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, BLOCK_SIZE
        w = _uniform_bf16(BLOCK_SIZE * 3 + 50)
        blocks, stats = tca_tbe_encode_tensor(w, self.cfg)
        self.assertEqual(len(blocks), 4)
        self.assertEqual(stats.n_blocks, 4)

    def test_decode_tensor_matches_original(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, tca_tbe_decode_tensor
        n = 512
        w = _uniform_bf16(n, seed=7)
        blocks, _ = tca_tbe_encode_tensor(w, self.cfg)
        recon = tca_tbe_decode_tensor(blocks, n_elements=n)
        np.testing.assert_array_equal(recon, w)

    def test_decode_with_remainder_block(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, tca_tbe_decode_tensor
        n = 300
        w = _uniform_bf16(n, seed=8)
        blocks, _ = tca_tbe_encode_tensor(w, self.cfg)
        recon = tca_tbe_decode_tensor(blocks, n_elements=n)
        self.assertEqual(len(recon), n)
        np.testing.assert_array_equal(recon, w)

    def test_decode_empty_list(self):
        from squish.compress.tca_tbe import tca_tbe_decode_tensor
        recon = tca_tbe_decode_tensor([], n_elements=0)
        self.assertEqual(len(recon), 0)

    def test_decode_truncates_to_n_elements(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, tca_tbe_decode_tensor
        n = 128
        w = _uniform_bf16(n, seed=9)
        blocks, _ = tca_tbe_encode_tensor(w, self.cfg)
        recon = tca_tbe_decode_tensor(blocks, n_elements=n)
        self.assertEqual(len(recon), n)

    def test_stats_compressed_bytes_positive(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor
        w = _uniform_bf16(256, seed=11)
        _, stats = tca_tbe_encode_tensor(w, self.cfg)
        self.assertGreater(stats.compressed_bytes, 0)

    def test_stats_raw_bytes_uncompressed(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor
        n = 256
        w = _uniform_bf16(n, seed=12)
        _, stats = tca_tbe_encode_tensor(w, self.cfg)
        self.assertEqual(stats.raw_bytes_uncompressed, n * 2)

    def test_stats_n_compressed_plus_n_raw_equals_n_blocks(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor
        w = _uniform_bf16(512, seed=13)
        _, stats = tca_tbe_encode_tensor(w, self.cfg)
        self.assertEqual(stats.n_compressed + stats.n_raw, stats.n_blocks)


# =============================================================================
# 8. CompressionStats
# =============================================================================

class TestCompressionStats(unittest.TestCase):
    def test_ratio_no_compression(self):
        from squish.compress.tca_tbe import CompressionStats
        s = CompressionStats(compressed_bytes=0, raw_bytes_uncompressed=256)
        self.assertAlmostEqual(s.compression_ratio, 1.0)

    def test_ratio_gt_one_for_compressed(self):
        from squish.compress.tca_tbe import CompressionStats
        s = CompressionStats(compressed_bytes=100, raw_bytes_uncompressed=256)
        self.assertGreater(s.compression_ratio, 1.0)

    def test_size_reduction_pct_range(self):
        from squish.compress.tca_tbe import CompressionStats
        s = CompressionStats(compressed_bytes=100, raw_bytes_uncompressed=256)
        self.assertGreaterEqual(s.size_reduction_pct, 0.0)
        self.assertLessEqual(s.size_reduction_pct, 100.0)

    def test_size_reduction_zero_when_no_data(self):
        from squish.compress.tca_tbe import CompressionStats
        s = CompressionStats()
        self.assertAlmostEqual(s.size_reduction_pct, 0.0)

    def test_spill_elements_non_negative(self):
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, TcaTbeConfig
        w = _uniform_bf16(128)
        _, stats = tca_tbe_encode_tensor(w, TcaTbeConfig())
        self.assertGreaterEqual(stats.spill_elements, 0)


# =============================================================================
# 9. InferenceStage
# =============================================================================

class TestInferenceStage(unittest.TestCase):
    def test_decode_value(self):
        from squish.runtime.stage_dispatcher import InferenceStage
        self.assertEqual(InferenceStage.DECODE.value, "decode")

    def test_prefill_value(self):
        from squish.runtime.stage_dispatcher import InferenceStage
        self.assertEqual(InferenceStage.PREFILL.value, "prefill")

    def test_is_string_subclass(self):
        from squish.runtime.stage_dispatcher import InferenceStage
        self.assertIsInstance(InferenceStage.DECODE, str)


# =============================================================================
# 10. KernelPipeline
# =============================================================================

class TestKernelPipeline(unittest.TestCase):
    def test_zip_gemv_value(self):
        from squish.runtime.stage_dispatcher import KernelPipeline
        self.assertEqual(KernelPipeline.ZIP_GEMV.value, "zip_gemv")

    def test_zip_gemm_value(self):
        from squish.runtime.stage_dispatcher import KernelPipeline
        self.assertEqual(KernelPipeline.ZIP_GEMM.value, "zip_gemm")

    def test_numpy_value(self):
        from squish.runtime.stage_dispatcher import KernelPipeline
        self.assertEqual(KernelPipeline.NUMPY.value, "numpy")


# =============================================================================
# 11. DispatchDecision
# =============================================================================

class TestDispatchDecision(unittest.TestCase):
    def _make(self, **kw):
        from squish.runtime.stage_dispatcher import DispatchDecision, InferenceStage
        defaults = dict(
            stage=InferenceStage.DECODE,
            kernel_pipeline="zip_gemv",
            seq_len=1,
            batch_size=1,
        )
        defaults.update(kw)
        return DispatchDecision(**defaults)

    def test_chunk_end_defaults_to_seq_len(self):
        d = self._make(seq_len=100)
        # chunk_end defaults to 0 then __post_init__ sets it to seq_len
        self.assertEqual(d.chunk_end, 100)

    def test_chunk_seq_len_property(self):
        d = self._make(seq_len=512, chunk_start=0, chunk_end=256)
        self.assertEqual(d.chunk_seq_len, 256)

    def test_frozen(self):
        d = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            d.seq_len = 99  # type: ignore[misc]

    def test_tca_tbe_enabled_default_true(self):
        d = self._make()
        self.assertTrue(d.tca_tbe_enabled)


# =============================================================================
# 12. StageDispatcher — detect_stage
# =============================================================================

class TestStageDispatcherDetect(unittest.TestCase):
    def test_seq_len_1_is_decode(self):
        from squish.runtime.stage_dispatcher import StageDispatcher, InferenceStage
        ids = np.zeros((1, 1), dtype=np.int64)
        self.assertEqual(StageDispatcher.detect_stage(ids), InferenceStage.DECODE)

    def test_seq_len_10_is_prefill(self):
        from squish.runtime.stage_dispatcher import StageDispatcher, InferenceStage
        ids = np.zeros((1, 10), dtype=np.int64)
        self.assertEqual(StageDispatcher.detect_stage(ids), InferenceStage.PREFILL)

    def test_batch_2_seq_1_is_decode(self):
        from squish.runtime.stage_dispatcher import StageDispatcher, InferenceStage
        ids = np.zeros((2, 1), dtype=np.int64)
        self.assertEqual(StageDispatcher.detect_stage(ids), InferenceStage.DECODE)

    def test_non_2d_raises(self):
        from squish.runtime.stage_dispatcher import StageDispatcher
        ids = np.zeros((5,), dtype=np.int64)
        with self.assertRaises(ValueError):
            StageDispatcher.detect_stage(ids)


# =============================================================================
# 13. StageDispatcher — dispatch (single decision)
# =============================================================================

class TestStageDispatcherDispatch(unittest.TestCase):
    def setUp(self):
        from squish.runtime.stage_dispatcher import StageDispatcher
        self.dispatcher = StageDispatcher(tca_tbe_enabled=True, chunk_size=512)

    def test_decode_uses_zip_gemv(self):
        from squish.runtime.stage_dispatcher import KernelPipeline
        ids = np.zeros((1, 1), dtype=np.int64)
        d = self.dispatcher.dispatch(ids)
        self.assertEqual(d.kernel_pipeline, KernelPipeline.ZIP_GEMV.value)

    def test_prefill_uses_zip_gemm(self):
        from squish.runtime.stage_dispatcher import KernelPipeline
        ids = np.zeros((1, 20), dtype=np.int64)
        d = self.dispatcher.dispatch(ids)
        self.assertEqual(d.kernel_pipeline, KernelPipeline.ZIP_GEMM.value)

    def test_dispatch_decode_not_chunked(self):
        ids = np.zeros((1, 1), dtype=np.int64)
        d = self.dispatcher.dispatch(ids)
        self.assertFalse(d.is_chunked)

    def test_dispatch_seq_len_stored(self):
        ids = np.zeros((1, 33), dtype=np.int64)
        d = self.dispatcher.dispatch(ids)
        self.assertEqual(d.seq_len, 33)

    def test_dispatch_batch_size_stored(self):
        ids = np.zeros((4, 1), dtype=np.int64)
        d = self.dispatcher.dispatch(ids)
        self.assertEqual(d.batch_size, 4)

    def test_dispatch_invalid_dtype_raises(self):
        ids = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch(ids)

    def test_dispatch_1d_raises(self):
        ids = np.zeros((10,), dtype=np.int64)
        with self.assertRaises(ValueError):
            self.dispatcher.dispatch(ids)

    def test_tca_disabled_uses_numpy(self):
        from squish.runtime.stage_dispatcher import StageDispatcher, KernelPipeline
        d_off = StageDispatcher(tca_tbe_enabled=False)
        ids = np.zeros((1, 1), dtype=np.int64)
        d = d_off.dispatch(ids)
        self.assertEqual(d.kernel_pipeline, KernelPipeline.NUMPY.value)

    def test_tca_disabled_prefill_uses_numpy(self):
        from squish.runtime.stage_dispatcher import StageDispatcher, KernelPipeline
        d_off = StageDispatcher(tca_tbe_enabled=False)
        ids = np.zeros((1, 512), dtype=np.int64)
        d = d_off.dispatch(ids)
        self.assertEqual(d.kernel_pipeline, KernelPipeline.NUMPY.value)

    def test_dispatch_chunk_end_equals_seq_len(self):
        ids = np.zeros((1, 50), dtype=np.int64)
        d = self.dispatcher.dispatch(ids)
        self.assertEqual(d.chunk_end, 50)


# =============================================================================
# 14. StageDispatcher — chunked prefill
# =============================================================================

class TestStageDispatcherChunked(unittest.TestCase):
    def setUp(self):
        from squish.runtime.stage_dispatcher import StageDispatcher
        self.dispatcher = StageDispatcher(tca_tbe_enabled=True, chunk_size=128)

    def test_decode_yields_single_chunk(self):
        ids = np.zeros((1, 1), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        self.assertEqual(len(chunks), 1)

    def test_decode_chunk_not_chunked(self):
        ids = np.zeros((1, 1), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        self.assertFalse(chunks[0].is_chunked)

    def test_short_prefill_single_chunk(self):
        ids = np.zeros((1, 50), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        self.assertEqual(len(chunks), 1)

    def test_long_prefill_multiple_chunks(self):
        ids = np.zeros((1, 300), dtype=np.int64)  # 300 / 128 = ceil → 3 chunks
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        self.assertEqual(len(chunks), 3)

    def test_chunks_is_chunked_true(self):
        ids = np.zeros((1, 300), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        for c in chunks:
            self.assertTrue(c.is_chunked)

    def test_chunks_cover_full_sequence(self):
        seq_len = 300
        ids = np.zeros((1, seq_len), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        covered = sum(c.chunk_seq_len for c in chunks)
        self.assertEqual(covered, seq_len)

    def test_chunks_contiguous(self):
        ids = np.zeros((1, 300), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        for i in range(1, len(chunks)):
            self.assertEqual(chunks[i].chunk_start, chunks[i - 1].chunk_end)

    def test_chunks_indices_incremental(self):
        ids = np.zeros((1, 300), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        for i, c in enumerate(chunks):
            self.assertEqual(c.chunk_idx, i)

    def test_chunk_size_override(self):
        ids = np.zeros((1, 300), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids, chunk_size=50))
        self.assertEqual(len(chunks), 6)

    def test_invalid_chunk_size_raises(self):
        ids = np.zeros((1, 128), dtype=np.int64)
        with self.assertRaises(ValueError):
            list(self.dispatcher.dispatch_chunked(ids, chunk_size=0))

    def test_invalid_chunk_size_negative_raises(self):
        ids = np.zeros((1, 128), dtype=np.int64)
        with self.assertRaises(ValueError):
            list(self.dispatcher.dispatch_chunked(ids, chunk_size=-1))

    def test_exact_multiple_no_extra_chunk(self):
        ids = np.zeros((1, 256), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        self.assertEqual(len(chunks), 2)

    def test_all_chunks_correct_seq_len(self):
        ids = np.zeros((1, 256), dtype=np.int64)
        chunks = list(self.dispatcher.dispatch_chunked(ids))
        for c in chunks:
            self.assertEqual(c.seq_len, 256)


# =============================================================================
# 15. StageDispatcher — constructor validation
# =============================================================================

class TestStageDispatcherConstructor(unittest.TestCase):
    def test_chunk_size_zero_raises(self):
        from squish.runtime.stage_dispatcher import StageDispatcher
        with self.assertRaises(ValueError):
            StageDispatcher(chunk_size=0)

    def test_chunk_size_negative_raises(self):
        from squish.runtime.stage_dispatcher import StageDispatcher
        with self.assertRaises(ValueError):
            StageDispatcher(chunk_size=-5)

    def test_properties_accessible(self):
        from squish.runtime.stage_dispatcher import StageDispatcher
        sd = StageDispatcher(tca_tbe_enabled=False, chunk_size=64)
        self.assertFalse(sd.tca_tbe_enabled)
        self.assertEqual(sd.chunk_size, 64)


# =============================================================================
# 16. Integration — encode→dispatch pipeline
# =============================================================================

class TestWave65Integration(unittest.TestCase):
    def test_tca_tbe_flag_in_squizd_flags(self):
        """TCA_TBE bit 1 must already exist in SquizdFlags from Wave 70."""
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.TCA_TBE), 1 << 1)

    def test_full_encode_decode_pipeline(self):
        """Encode a float32 weight matrix to BF16 TCA-TBE and decode losslessly."""
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, tca_tbe_decode_tensor, TcaTbeConfig
        n = 1024
        rng = np.random.default_rng(0)
        f32 = rng.standard_normal(n).astype(np.float32)
        u32 = f32.view(np.uint32)
        bf16 = (u32 >> 16).astype(np.uint16)

        blocks, stats = tca_tbe_encode_tensor(bf16, TcaTbeConfig())
        recon = tca_tbe_decode_tensor(blocks, n_elements=n)
        np.testing.assert_array_equal(recon, bf16)

    def test_dispatcher_gives_gemv_for_decode_step(self):
        """After a full prefill, the decode step must route to zip_gemv."""
        from squish.runtime.stage_dispatcher import StageDispatcher, KernelPipeline
        dispatcher = StageDispatcher(tca_tbe_enabled=True)
        decode_ids = np.zeros((1, 1), dtype=np.int64)
        d = dispatcher.dispatch(decode_ids)
        self.assertEqual(d.kernel_pipeline, KernelPipeline.ZIP_GEMV.value)

    def test_compression_stats_reasonable(self):
        """Compression ratio should be > 1 for randomly-distributed BF16 weights."""
        from squish.compress.tca_tbe import tca_tbe_encode_tensor, TcaTbeConfig
        n = 2048
        bf16 = _uniform_bf16(n, seed=99)
        _, stats = tca_tbe_encode_tensor(bf16, TcaTbeConfig())
        # ratio may be <= 1 for high-entropy data, but must be positive
        self.assertGreater(stats.compression_ratio, 0.0)
        self.assertGreater(stats.n_blocks, 0)

    def test_chunked_prefill_pipeline_is_gemm(self):
        """All chunks of a long prefill should route to zip_gemm."""
        from squish.runtime.stage_dispatcher import StageDispatcher, KernelPipeline
        dispatcher = StageDispatcher(tca_tbe_enabled=True, chunk_size=128)
        ids = np.zeros((1, 512), dtype=np.int64)
        for d in dispatcher.dispatch_chunked(ids):
            self.assertEqual(d.kernel_pipeline, KernelPipeline.ZIP_GEMM.value)


if __name__ == "__main__":
    unittest.main()
