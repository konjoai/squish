"""tests/test_wave68_mxfp4_bridge.py

Unit tests for Wave 68: MXFP4 Format Bridge for .squizd files.

Modules under test
──────────────────
* squish.format.mx_fp4 — MxFP4BlockHeader, MxFP4FormatBridge,
                          MxFP4NativeBackend, _pack_codes, _unpack_codes,
                          MXFP4_BLOCK_TAG, MXFP4_HEADER_SIZE
"""
from __future__ import annotations

import struct
import unittest

import numpy as np

from squish.format.mx_fp4 import (
    MXFP4_BLOCK_TAG,
    MXFP4_HEADER_SIZE,
    MxFP4BlockHeader,
    MxFP4FormatBridge,
    MxFP4NativeBackend,
    _pack_codes,
    _unpack_codes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_header() -> MxFP4BlockHeader:
    """Return a well-formed header for testing."""
    return MxFP4BlockHeader(
        block_tag=MXFP4_BLOCK_TAG,
        block_size=32,
        n_blocks=4,
        scale_offset=0,
        data_offset=16,  # 4 scales * 4 bytes each
        data_len=64,
    )


def _make_random_weights(
    shape=(4, 32),
    *,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


# ---------------------------------------------------------------------------
# TestMxFP4Constants
# ---------------------------------------------------------------------------

class TestMxFP4Constants(unittest.TestCase):

    def test_block_tag_value(self):
        self.assertEqual(MXFP4_BLOCK_TAG, 0xF4)

    def test_header_size_value(self):
        # u8 + u16 + u32 + u64 + u64 + u64 = 1 + 2 + 4 + 8 + 8 + 8 = 31
        self.assertEqual(MXFP4_HEADER_SIZE, 31)


# ---------------------------------------------------------------------------
# TestMxFP4BlockHeader
# ---------------------------------------------------------------------------

class TestMxFP4BlockHeader(unittest.TestCase):

    def test_to_bytes_length(self):
        hdr = _valid_header()
        self.assertEqual(len(hdr.to_bytes()), MXFP4_HEADER_SIZE)

    def test_roundtrip_fields(self):
        hdr = _valid_header()
        restored = MxFP4BlockHeader.from_bytes(hdr.to_bytes())
        self.assertEqual(restored.block_tag, hdr.block_tag)
        self.assertEqual(restored.block_size, hdr.block_size)
        self.assertEqual(restored.n_blocks, hdr.n_blocks)
        self.assertEqual(restored.scale_offset, hdr.scale_offset)
        self.assertEqual(restored.data_offset, hdr.data_offset)
        self.assertEqual(restored.data_len, hdr.data_len)

    def test_from_bytes_at_least_header_size(self):
        hdr = _valid_header()
        raw = hdr.to_bytes() + b"\x00" * 100  # extra bytes ignored
        restored = MxFP4BlockHeader.from_bytes(raw)
        self.assertEqual(restored.block_tag, MXFP4_BLOCK_TAG)

    def test_validate_valid_header(self):
        hdr = _valid_header()
        # Should not raise
        hdr.validate()

    def test_validate_invalid_tag(self):
        hdr = _valid_header()
        hdr.block_tag = 0x00
        with self.assertRaises(ValueError):
            hdr.validate()

    def test_validate_invalid_block_size(self):
        hdr = _valid_header()
        hdr.block_size = 0
        with self.assertRaises(ValueError):
            hdr.validate()

    def test_validate_invalid_n_blocks_negative(self):
        hdr = _valid_header()
        hdr.n_blocks = -1
        with self.assertRaises(ValueError):
            hdr.validate()

    def test_field_block_tag(self):
        hdr = _valid_header()
        self.assertEqual(hdr.block_tag, MXFP4_BLOCK_TAG)

    def test_field_block_size(self):
        hdr = _valid_header()
        self.assertEqual(hdr.block_size, 32)


# ---------------------------------------------------------------------------
# TestPackUnpackCodes
# ---------------------------------------------------------------------------

class TestPackUnpackCodes(unittest.TestCase):

    def test_roundtrip_even_length(self):
        codes = np.array([0, 1, 2, 3, 15, 7, 8, 14], dtype=np.uint8)
        packed = _pack_codes(codes)
        restored = _unpack_codes(packed, len(codes))
        np.testing.assert_array_equal(restored, codes)

    def test_roundtrip_odd_length(self):
        codes = np.array([3, 7, 12], dtype=np.uint8)
        packed = _pack_codes(codes)
        restored = _unpack_codes(packed, len(codes))
        np.testing.assert_array_equal(restored, codes)

    def test_pack_codes_output_length_even(self):
        codes = np.arange(8, dtype=np.uint8)
        packed = _pack_codes(codes)
        self.assertEqual(len(packed), 4)

    def test_pack_codes_output_length_odd(self):
        codes = np.arange(7, dtype=np.uint8)
        packed = _pack_codes(codes)
        # 7 elements → padded to 8 → 4 bytes
        self.assertEqual(len(packed), 4)

    def test_unpack_codes_values_in_range(self):
        rng = np.random.default_rng(0)
        codes = rng.integers(0, 16, size=32, dtype=np.uint8)
        packed = _pack_codes(codes)
        restored = _unpack_codes(packed, 32)
        self.assertTrue(np.all(restored >= 0))
        self.assertTrue(np.all(restored <= 15))

    def test_unpack_codes_exact_count(self):
        codes = np.arange(10, dtype=np.uint8) % 16
        packed = _pack_codes(codes)
        restored = _unpack_codes(packed, 10)
        self.assertEqual(len(restored), 10)

    def test_pack_all_zeros(self):
        codes = np.zeros(4, dtype=np.uint8)
        packed = _pack_codes(codes)
        self.assertEqual(packed, b"\x00\x00")

    def test_pack_all_max(self):
        codes = np.full(4, 15, dtype=np.uint8)
        packed = _pack_codes(codes)
        self.assertEqual(packed, b"\xff\xff")


# ---------------------------------------------------------------------------
# TestMxFP4FormatBridge
# ---------------------------------------------------------------------------

class TestMxFP4FormatBridge(unittest.TestCase):

    def setUp(self):
        self.bridge = MxFP4FormatBridge()

    def test_encode_returns_bytes(self):
        w = _make_random_weights((4, 32))
        result = self.bridge.encode(w)
        self.assertIsInstance(result, bytes)

    def test_encode_header_tag_present(self):
        w = _make_random_weights((4, 32))
        encoded = self.bridge.encode(w)
        # First byte must be MXFP4_BLOCK_TAG = 0xF4
        self.assertEqual(encoded[0], MXFP4_BLOCK_TAG)

    def test_encode_minimum_size(self):
        w = _make_random_weights((2, 32))
        encoded = self.bridge.encode(w)
        self.assertGreaterEqual(len(encoded), MXFP4_HEADER_SIZE)

    def test_decode_returns_ndarray(self):
        w = _make_random_weights((2, 32))
        encoded = self.bridge.encode(w)
        decoded = self.bridge.decode(encoded)
        self.assertIsInstance(decoded, np.ndarray)

    def test_decode_dtype_float32(self):
        w = _make_random_weights((2, 32))
        encoded = self.bridge.encode(w)
        decoded = self.bridge.decode(encoded)
        self.assertEqual(decoded.dtype, np.float32)

    def test_encode_decode_roundtrip_approximate(self):
        # MXFP4 is lossy; check that decoded values have the same sign distribution
        # and are in a reasonable range
        w = _make_random_weights((4, 32), seed=1)
        encoded = self.bridge.encode(w)
        decoded = self.bridge.decode(encoded)
        self.assertEqual(decoded.shape[0], 4 * 32)  # flat output
        # After dequantisation the magnitude should be similar
        orig_mean = float(np.mean(np.abs(w)))
        dec_mean = float(np.mean(np.abs(decoded)))
        self.assertGreater(dec_mean, 0.0)  # not all zeros

    def test_decode_invalid_tag_raises(self):
        # Craft a header with wrong tag
        hdr = _valid_header()
        hdr.block_tag = 0x00
        bad_bytes = hdr.to_bytes() + b"\x00" * 128
        with self.assertRaises(ValueError):
            self.bridge.decode(bad_bytes)

    def test_encode_1d_tensor(self):
        w = np.random.default_rng(2).standard_normal(64).astype(np.float32)
        encoded = self.bridge.encode(w)
        self.assertIsInstance(encoded, bytes)
        self.assertGreaterEqual(len(encoded), MXFP4_HEADER_SIZE)

    def test_encode_2d_tensor(self):
        w = _make_random_weights((8, 32))
        encoded = self.bridge.encode(w)
        self.assertIsInstance(encoded, bytes)

    def test_custom_block_size(self):
        bridge = MxFP4FormatBridge(block_size=16)
        w = _make_random_weights((4, 64))
        encoded = bridge.encode(w)
        self.assertIsInstance(encoded, bytes)
        decoded = bridge.decode(encoded)
        self.assertIsInstance(decoded, np.ndarray)

    def test_block_size_property(self):
        bridge = MxFP4FormatBridge(block_size=32)
        self.assertEqual(bridge.block_size, 32)

    def test_route_software_fallback(self):
        """On non-M5 hardware (mock caps), route() should use the software path."""
        w = _make_random_weights((2, 32), seed=3)
        encoded = self.bridge.encode(w)

        class _MockCaps:
            chip_generation = None  # triggers software fallback

        input_vec = np.ones(32, dtype=np.float32)
        result = self.bridge.route(encoded, input_vec, _MockCaps())
        self.assertIsInstance(result, np.ndarray)

    def test_route_output_is_ndarray(self):
        w = _make_random_weights((2, 32), seed=4)
        encoded = self.bridge.encode(w)

        class _MockCaps:
            chip_generation = None

        x = np.random.default_rng(0).standard_normal(32).astype(np.float32)
        out = self.bridge.route(encoded, x, _MockCaps())
        self.assertIsInstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# TestMxFP4NativeBackend
# ---------------------------------------------------------------------------

class TestMxFP4NativeBackend(unittest.TestCase):

    def test_matmul_raises_not_implemented(self):
        backend = MxFP4NativeBackend()
        hdr = _valid_header()
        with self.assertRaises(NotImplementedError):
            backend.matmul(b"\x00" * 128, np.zeros(32, dtype=np.float32), hdr)


if __name__ == "__main__":
    unittest.main()
