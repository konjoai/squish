"""tests/test_dfloat11_unit.py — 100% coverage for squish/dfloat11.py"""
import struct

import numpy as np
import pytest

from squish.dfloat11 import (
    CompressedBlock,
    CompressedModel,
    DFloat11Compressor,
    DFloat11Config,
    HuffmanCodec,
    compress_model,
)

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------------------
# DFloat11Config
# ---------------------------------------------------------------------------

class TestDFloat11Config:
    def test_defaults(self):
        cfg = DFloat11Config()
        assert cfg.block_size == 1024
        assert cfg.min_symbol_freq == 1

    def test_custom(self):
        cfg = DFloat11Config(block_size=512, min_symbol_freq=2)
        assert cfg.block_size == 512
        assert cfg.min_symbol_freq == 2

    def test_invalid_block_size(self):
        with pytest.raises(ValueError, match="block_size"):
            DFloat11Config(block_size=0)

    def test_invalid_min_symbol_freq(self):
        with pytest.raises(ValueError, match="min_symbol_freq"):
            DFloat11Config(min_symbol_freq=0)


# ---------------------------------------------------------------------------
# HuffmanCodec
# ---------------------------------------------------------------------------

class TestHuffmanCodec:
    def _uniform_freq(self, n: int) -> dict:
        return {i: 10 for i in range(n)}

    def test_empty_freq_raises(self):
        with pytest.raises(ValueError, match="empty"):
            HuffmanCodec({})

    def test_all_zero_freq_raises(self):
        with pytest.raises(ValueError, match="empty"):
            HuffmanCodec({0: 0, 1: 0})

    def test_single_symbol(self):
        codec = HuffmanCodec({42: 100})
        data = codec.encode(np.array([42, 42, 42], dtype=np.uint8))
        decoded = codec.decode(data, 3)
        assert list(decoded) == [42, 42, 42]

    def test_code_dict_round_trip(self):
        freq = {0: 50, 1: 25, 2: 15, 3: 10}
        codec = HuffmanCodec(freq)
        symbols = np.array([0, 1, 2, 3, 0, 0, 1], dtype=np.uint8)
        data    = codec.encode(symbols)
        decoded = codec.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_encode_decode_all_256_symbols(self):
        freq  = {i: max(1, 256 - i) for i in range(256)}
        codec = HuffmanCodec(freq)
        symbols = np.arange(256, dtype=np.uint8)
        data    = codec.encode(symbols)
        decoded = codec.decode(data, 256)
        np.testing.assert_array_equal(decoded, symbols)

    def test_encode_has_header(self):
        codec  = HuffmanCodec({0: 10, 1: 5})
        data   = codec.encode(np.array([0, 1, 0], dtype=np.uint8))
        assert len(data) >= 4  # 4-byte header

    def test_compressed_smaller_than_raw(self):
        # Highly skewed distribution → good compression
        freq   = {0: 10000, 1: 1, 2: 1}
        codec  = HuffmanCodec(freq)
        symbols = np.zeros(10000, dtype=np.uint8)
        data    = codec.encode(symbols)
        # Compressed should be much smaller than 10000 raw bytes
        assert len(data) < 2000

    def test_to_dict_from_code_dict(self):
        freq  = {10: 3, 20: 7, 30: 1}
        codec = HuffmanCodec(freq)
        codes = codec.to_dict()
        assert isinstance(codes, dict)
        assert set(codes.keys()) == {10, 20, 30}

        codec2   = HuffmanCodec.from_code_dict(codes)
        symbols  = np.array([10, 20, 30, 10, 20], dtype=np.uint8)
        data     = codec.encode(symbols)
        decoded  = codec2.decode(data, len(symbols))
        np.testing.assert_array_equal(decoded, symbols)

    def test_from_code_dict_decode_table(self):
        # Verify decode table is populated by from_code_dict
        codes  = {5: "0", 6: "10", 7: "11"}
        codec  = HuffmanCodec.from_code_dict(codes)
        assert codec._decode_table["0"] == 5
        assert codec._decode_table["10"] == 6
        assert codec._decode_table["11"] == 7

    def test_two_symbol_codec(self):
        freq   = {0: 50, 255: 50}
        codec  = HuffmanCodec(freq)
        sym    = np.array([0, 255, 0, 255], dtype=np.uint8)
        data   = codec.encode(sym)
        decoded = codec.decode(data, 4)
        np.testing.assert_array_equal(decoded, sym)


# ---------------------------------------------------------------------------
# CompressedBlock
# ---------------------------------------------------------------------------

class TestCompressedBlock:
    def _make_block(self, n=128):
        codes        = {0: "0", 1: "10", 2: "11"}
        exp_data     = b"\x00\x00\x00\x08" + b"\xAA" * 2
        sign_mant    = b"\x00" * n
        return CompressedBlock(
            exponent_data=exp_data,
            sign_mantissa=sign_mant,
            codes=codes,
            n_values=n,
        )

    def test_n_values(self):
        blk = self._make_block(n=64)
        assert blk.n_values == 64

    def test_original_size(self):
        blk = self._make_block(n=100)
        assert blk.original_size == 200  # 2 bytes per f16

    def test_compressed_size(self):
        blk = self._make_block(n=100)
        # sign_mant = 100 bytes + exponent_data
        assert blk.compressed_size == len(blk.exponent_data) + 100

    def test_compression_ratio_positive(self):
        blk = self._make_block(n=256)
        assert blk.compression_ratio > 0.0

    def test_dtype_str_default(self):
        blk = self._make_block()
        assert blk.dtype_str == "float16"


# ---------------------------------------------------------------------------
# DFloat11Compressor — compress_block / decompress_block
# ---------------------------------------------------------------------------

class TestDFloat11CompressorBlock:
    def _comp(self):
        return DFloat11Compressor(DFloat11Config(block_size=512))

    def test_lossless_roundtrip_small(self):
        arr  = np.array([1.0, -0.5, 0.25, 3.14159, 0.0, -1e-3], dtype=np.float16)
        comp = self._comp()
        blk  = comp.compress_block(arr)
        restored = comp.decompress_block(blk)
        np.testing.assert_array_equal(restored, arr)

    def test_lossless_roundtrip_random(self):
        arr  = RNG.standard_normal(512).astype(np.float16) * 0.1
        comp = self._comp()
        blk  = comp.compress_block(arr)
        restored = comp.decompress_block(blk)
        np.testing.assert_array_equal(restored, arr)

    def test_lossless_roundtrip_float32_input(self):
        arr  = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        comp = self._comp()
        blk  = comp.compress_block(arr)
        restored = comp.decompress_block(blk)
        expected = arr.astype(np.float16)
        np.testing.assert_array_equal(restored, expected)

    def test_n_values_stored(self):
        arr = np.ones(200, dtype=np.float16)
        blk = DFloat11Compressor().compress_block(arr)
        assert blk.n_values == 200

    def test_single_value(self):
        arr  = np.array([0.5], dtype=np.float16)
        comp = self._comp()
        blk  = comp.compress_block(arr)
        restored = comp.decompress_block(blk)
        np.testing.assert_array_equal(restored, arr)

    def test_compression_ratio_ge_1_for_typical_weights(self):
        # Typical LLM weights: small normal distribution → biased exponents
        weights = (RNG.standard_normal(4096) * 0.02).astype(np.float16)
        comp    = DFloat11Compressor()
        blk     = comp.compress_block(weights)
        assert blk.compression_ratio >= 1.0

    def test_multidimensional_input_flattened(self):
        arr  = np.ones((4, 8), dtype=np.float16)
        comp = self._comp()
        blk  = comp.compress_block(arr)
        # Should work on flattened view
        assert blk.n_values == 32
        restored = comp.decompress_block(blk)
        np.testing.assert_array_equal(restored, arr.ravel())


# ---------------------------------------------------------------------------
# DFloat11Compressor — compress_array / decompress_array
# ---------------------------------------------------------------------------

class TestDFloat11CompressorArray:
    def test_roundtrip_larger_than_block(self):
        weights = (RNG.standard_normal(3000) * 0.05).astype(np.float16)
        comp    = DFloat11Compressor(DFloat11Config(block_size=512))
        blocks  = comp.compress_array(weights)
        assert len(blocks) == 6  # ceil(3000 / 512) = 6
        restored = comp.decompress_array(blocks)
        np.testing.assert_array_equal(restored, weights)

    def test_roundtrip_exact_block_size(self):
        weights  = np.arange(1024, dtype=np.float16)
        comp     = DFloat11Compressor(DFloat11Config(block_size=1024))
        blocks   = comp.compress_array(weights)
        assert len(blocks) == 1
        restored = comp.decompress_array(blocks)
        np.testing.assert_array_equal(restored, weights)

    def test_single_block_array(self):
        weights  = np.zeros(10, dtype=np.float16)
        comp     = DFloat11Compressor(DFloat11Config(block_size=1024))
        blocks   = comp.compress_array(weights)
        assert len(blocks) == 1
        restored = comp.decompress_array(blocks)
        np.testing.assert_array_equal(restored, weights)


# ---------------------------------------------------------------------------
# CompressedModel
# ---------------------------------------------------------------------------

class TestCompressedModel:
    def _state_dict(self):
        return {
            "layer0.weight": (RNG.standard_normal((32, 128)) * 0.02).astype(np.float16),
            "layer0.bias":   np.zeros(32, dtype=np.float16),
            "layer1.weight": (RNG.standard_normal((64, 32)) * 0.02).astype(np.float16),
        }

    def test_compress_model_sizes(self):
        sd    = self._state_dict()
        model = compress_model(sd, DFloat11Config(block_size=512))
        assert set(model.layer_names()) == set(sd.keys())
        assert model.original_size  > 0
        assert model.compressed_size > 0
        assert model.compression_ratio > 0.0

    def test_compress_model_original_size_matches(self):
        sd    = self._state_dict()
        model = compress_model(sd)
        expected = sum(a.astype(np.float16).nbytes for a in sd.values())
        assert model.original_size == expected

    def test_decompressed_layer_shape(self):
        sd    = self._state_dict()
        comp  = DFloat11Compressor()
        model = compress_model(sd)
        w16   = model.decompressed_layer("layer0.weight", comp)
        assert w16.shape == sd["layer0.weight"].shape
        assert w16.dtype == np.float16

    def test_decompressed_layer_lossless(self):
        sd    = self._state_dict()
        comp  = DFloat11Compressor()
        model = compress_model(sd)
        for name, orig in sd.items():
            restored = model.decompressed_layer(name, comp)
            np.testing.assert_array_equal(restored.ravel(), orig.ravel().astype(np.float16))

    def test_empty_state_dict(self):
        model = compress_model({})
        assert model.layer_names() == []
        assert model.original_size  == 0
        assert model.compressed_size == 0

    def test_compression_ratio_zero_compressed_safe(self):
        model = CompressedModel()
        assert model.compression_ratio == 0.0 / max(0, 1)


# ---------------------------------------------------------------------------
# Coverage gap — decode empty bits (for loop never entered, line 229->237)
# ---------------------------------------------------------------------------

class TestHuffmanCodecCoverageGaps:
    def test_decode_empty_bits_skips_loop(self):
        """When n_symbols=0 the for loop body is never entered (229->237 branch)."""
        codec = HuffmanCodec({0: 10, 1: 5})
        # encode an empty array — produces a 4-byte header with bit_count=0
        data    = codec.encode(np.array([], dtype=np.uint8))
        decoded = codec.decode(data, 0)
        assert decoded.shape == (0,)
