"""tests/test_wave64_astc_compression.py

Unit tests for Wave 64: SQUIZD ASTC Compression Pipeline.

Modules under test
──────────────────
* squish.compress.astc_encoder   — ASTC encoder + NumPy fallback
* squish.format.squish_header    — 256-byte .squizd binary header
* squish.loaders.astc_loader     — MTLTexture registration (simulation mode)

All tests run in simulation mode (no libastcenc, no Metal hardware required).
Hardware-specific code paths are exercised via environment variable overrides
and ``force_numpy_fallback=True``.
"""
from __future__ import annotations

import math
import os
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_weights(rows: int = 16, cols: int = 16, seed: int = 42) -> np.ndarray:
    """Return a reproducible float32 weight matrix."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


# ===========================================================================
# 1. ASTCEncoderConfig
# ===========================================================================

class TestASTCEncoderConfig(unittest.TestCase):
    """Tests for ASTCEncoderConfig validation and defaults."""

    def test_default_block_size(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig, ASTC_BLOCK_X, ASTC_BLOCK_Y
        cfg = ASTCEncoderConfig()
        self.assertEqual(cfg.block_x, ASTC_BLOCK_X)
        self.assertEqual(cfg.block_y, ASTC_BLOCK_Y)

    def test_default_quality(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        cfg = ASTCEncoderConfig()
        self.assertEqual(cfg.quality, 100.0)

    def test_default_float_range_false(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        cfg = ASTCEncoderConfig()
        self.assertFalse(cfg.is_float_range)

    def test_texels_per_block(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        cfg = ASTCEncoderConfig(block_x=6, block_y=6)
        self.assertEqual(cfg.texels_per_block, 36)

    def test_custom_block_size(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        cfg = ASTCEncoderConfig(block_x=8, block_y=8)
        self.assertEqual(cfg.texels_per_block, 64)

    def test_quality_range_valid_min(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        cfg = ASTCEncoderConfig(quality=0.0)
        self.assertEqual(cfg.quality, 0.0)

    def test_quality_range_valid_max(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        cfg = ASTCEncoderConfig(quality=100.0)
        self.assertEqual(cfg.quality, 100.0)

    def test_block_too_small_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        with self.assertRaises(ValueError):
            ASTCEncoderConfig(block_x=2, block_y=6)

    def test_block_too_small_y_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        with self.assertRaises(ValueError):
            ASTCEncoderConfig(block_x=6, block_y=3)

    def test_quality_out_of_range_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        with self.assertRaises(ValueError):
            ASTCEncoderConfig(quality=101.0)

    def test_quality_negative_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoderConfig
        with self.assertRaises(ValueError):
            ASTCEncoderConfig(quality=-1.0)


# ===========================================================================
# 2. ASTCEncoder — padding logic
# ===========================================================================

class TestASTCEncoderPadding(unittest.TestCase):
    """Tests for block-boundary padding in ASTCEncoder._pad_to_blocks."""

    def _pad(self, rows: int, cols: int) -> tuple:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        w = _make_weights(rows, cols)
        _, shape = enc._pad_to_blocks(w)  # type: ignore[protected-access]
        return shape

    def test_exact_block_boundary_unchanged(self) -> None:
        shape = self._pad(12, 12)
        self.assertEqual(shape, (12, 12))

    def test_rows_rounded_up(self) -> None:
        shape = self._pad(7, 6)
        self.assertEqual(shape[0], 12)  # ceil(7/6)*6 = 12

    def test_cols_rounded_up(self) -> None:
        shape = self._pad(6, 7)
        self.assertEqual(shape[1], 12)  # ceil(7/6)*6 = 12

    def test_single_row(self) -> None:
        shape = self._pad(1, 1)
        self.assertEqual(shape, (6, 6))

    def test_large_matrix(self) -> None:
        shape = self._pad(512, 256)
        self.assertEqual(shape[0] % 6, 0)
        self.assertEqual(shape[1] % 6, 0)

    def test_padded_shape_x_multiple_of_block_x(self) -> None:
        for rows in [5, 11, 100, 513]:
            for cols in [3, 13, 37, 300]:
                shape = self._pad(rows, cols)
                self.assertEqual(shape[0] % 6, 0, f"rows={rows}")
                self.assertEqual(shape[1] % 6, 0, f"cols={cols}")


# ===========================================================================
# 3. ASTCEncoder NumPy encode/decode path
# ===========================================================================

class TestASTCEncodeNumpyPath(unittest.TestCase):
    """Tests for the NumPy fallback encode/decode round-trip."""

    def _encode(self, rows: int = 12, cols: int = 12, seed: int = 7) -> Any:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        cfg = ASTCEncoderConfig()
        enc = ASTCEncoder(config=cfg, force_numpy_fallback=True)
        return enc, enc.encode(_make_weights(rows, cols, seed=seed))

    def test_native_encoding_used_is_false_for_fallback(self) -> None:
        _, result = self._encode()
        self.assertFalse(result.native_encoding_used)

    def test_block_bytes_length_correct(self) -> None:
        from squish.compress.astc_encoder import ASTC_BLOCK_BYTES
        _, result = self._encode(12, 12)
        self.assertEqual(len(result.block_bytes), result.n_blocks * ASTC_BLOCK_BYTES)

    def test_scale_table_shape(self) -> None:
        _, result = self._encode(12, 12)
        self.assertEqual(result.scale_table.shape, (result.n_blocks,))

    def test_scale_table_dtype(self) -> None:
        _, result = self._encode()
        self.assertEqual(result.scale_table.dtype, np.float32)

    def test_scale_table_positive(self) -> None:
        _, result = self._encode()
        self.assertTrue(np.all(result.scale_table > 0))

    def test_original_shape_preserved(self) -> None:
        _, result = self._encode(8, 24)
        self.assertEqual(result.original_shape, (8, 24))

    def test_padded_shape_multiples_of_6(self) -> None:
        _, result = self._encode(7, 13)
        self.assertEqual(result.padded_shape[0] % 6, 0)
        self.assertEqual(result.padded_shape[1] % 6, 0)

    def test_n_blocks_correct(self) -> None:
        _, result = self._encode(12, 12)
        expected = (12 // 6) * (12 // 6)  # 4
        self.assertEqual(result.n_blocks, expected)

    def test_1d_vector_input(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        w = np.ones(12, dtype=np.float32)
        result = enc.encode(w)
        self.assertEqual(result.original_shape, (1, 12))

    def test_3d_tensor_flattened(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        w = np.random.randn(4, 3, 6).astype(np.float32)
        result = enc.encode(w)
        self.assertEqual(result.original_shape, (12, 6))

    def test_decode_returns_float32(self) -> None:
        enc, result = self._encode()
        decoded = enc.decode(result)
        self.assertEqual(decoded.dtype, np.float32)

    def test_decode_shape_matches_original(self) -> None:
        enc, result = self._encode(8, 24)
        decoded = enc.decode(result)
        self.assertEqual(decoded.shape, (8, 24))

    def test_zero_weight_matrix(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        w = np.zeros((12, 12), dtype=np.float32)
        result = enc.encode(w)
        self.assertEqual(result.original_shape, (12, 12))


# ===========================================================================
# 4. ASTCEncodeResult — properties and serialisation
# ===========================================================================

class TestASTCEncodeResult(unittest.TestCase):
    """Tests for ASTCEncodeResult properties and wire-format serialisation."""

    def _result(self, rows: int = 12, cols: int = 12) -> Any:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        return enc.encode(_make_weights(rows, cols))

    def test_bpw_for_exact_blocks(self) -> None:
        result = self._result(12, 12)
        # 4 blocks × 16 bytes × 8 bits / (12*12 = 144) = 512/144 ≈ 3.56
        expected = (result.n_blocks * 16 * 8) / (12 * 12)
        self.assertAlmostEqual(result.bpw, expected, places=5)

    def test_bpw_positive(self) -> None:
        result = self._result()
        self.assertGreater(result.bpw, 0)

    def test_total_bytes_includes_scale_table(self) -> None:
        result = self._result()
        expected = len(result.block_bytes) + result.n_blocks * 4
        self.assertEqual(result.total_bytes, expected)

    def test_serialise_starts_with_magic(self) -> None:
        result = self._result()
        data = result.serialise()
        self.assertEqual(data[:8], b"ASTCBLK1")

    def test_serialise_length(self) -> None:
        from squish.compress.astc_encoder import ASTC_BLOCK_BYTES
        result = self._result(12, 12)
        expected_len = 8 + 5 * 4 + result.n_blocks * ASTC_BLOCK_BYTES + result.n_blocks * 4
        self.assertEqual(len(result.serialise()), expected_len)

    def test_deserialise_roundtrip_n_blocks(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult
        result = self._result()
        data = result.serialise()
        restored = ASTCEncodeResult.deserialise(data)
        self.assertEqual(restored.n_blocks, result.n_blocks)

    def test_deserialise_roundtrip_original_shape(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult
        result = self._result(8, 18)
        data = result.serialise()
        restored = ASTCEncodeResult.deserialise(data)
        self.assertEqual(restored.original_shape, (8, 18))

    def test_deserialise_roundtrip_block_bytes(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult
        result = self._result()
        data = result.serialise()
        restored = ASTCEncodeResult.deserialise(data)
        self.assertEqual(restored.block_bytes, result.block_bytes)

    def test_deserialise_roundtrip_scale_table(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult
        result = self._result()
        data = result.serialise()
        restored = ASTCEncodeResult.deserialise(data)
        np.testing.assert_array_almost_equal(restored.scale_table, result.scale_table)

    def test_deserialise_bad_magic_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult
        with self.assertRaises(ValueError):
            ASTCEncodeResult.deserialise(b"BADMAGIC" + b"\x00" * 100)

    def test_invalid_block_bytes_length_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult
        with self.assertRaises(ValueError):
            ASTCEncodeResult(
                block_bytes=b"\x00" * 10,  # wrong length
                scale_table=np.array([1.0], dtype=np.float32),
                original_shape=(6, 6),
                padded_shape=(6, 6),
                n_blocks=1,
                native_encoding_used=False,
            )

    def test_invalid_scale_table_shape_raises(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult, ASTC_BLOCK_BYTES
        with self.assertRaises(ValueError):
            ASTCEncodeResult(
                block_bytes=b"\x00" * ASTC_BLOCK_BYTES,
                scale_table=np.array([1.0, 2.0], dtype=np.float32),  # wrong shape
                original_shape=(6, 6),
                padded_shape=(6, 6),
                n_blocks=1,
                native_encoding_used=False,
            )


# ===========================================================================
# 5. SquizdHeader — basic fields and serialisation
# ===========================================================================

class TestSquizdHeaderBasic(unittest.TestCase):
    """Tests for SquizdHeader default values and round-trip serialisation."""

    def test_default_magic_on_serialise(self) -> None:
        from squish.format.squish_header import SquizdHeader, SQUIZD_MAGIC
        hdr = SquizdHeader()
        raw = hdr.serialise()
        self.assertEqual(raw[:4], SQUIZD_MAGIC)

    def test_serialise_length_is_256(self) -> None:
        from squish.format.squish_header import SquizdHeader, SQUIZD_HEADER_SIZE
        hdr = SquizdHeader()
        self.assertEqual(len(hdr.serialise()), SQUIZD_HEADER_SIZE)

    def test_version_byte_in_serialised_data(self) -> None:
        from squish.format.squish_header import SquizdHeader, SQUIZD_VERSION
        hdr = SquizdHeader()
        raw = hdr.serialise()
        version = struct.unpack_from("<H", raw, 4)[0]
        self.assertEqual(version, SQUIZD_VERSION)

    def test_default_flags_zero(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdFlag
        hdr = SquizdHeader()
        self.assertEqual(hdr.flags, SquizdFlag.NONE)

    def test_default_arch_unknown(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdArch
        hdr = SquizdHeader()
        self.assertEqual(hdr.arch_id, SquizdArch.UNKNOWN)

    def test_num_layers_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(num_layers=32)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.num_layers, 32)

    def test_hidden_dim_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(hidden_dim=4096)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.hidden_dim, 4096)

    def test_num_heads_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(num_heads=32)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.num_heads, 32)

    def test_vocab_size_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(vocab_size=151936)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.vocab_size, 151936)

    def test_compression_bpw_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(compression_bpw=3.5625)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertAlmostEqual(restored.compression_bpw, 3.5625, places=4)

    def test_sparsity_ratio_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(sparsity_ratio=0.5)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertAlmostEqual(restored.sparsity_ratio, 0.5, places=6)

    def test_calibration_hash_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        cal_hash = 0xDEADBEEFCAFEBABE
        hdr = SquizdHeader(calibration_hash=cal_hash)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.calibration_hash, cal_hash)

    def test_spare_crc_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(spare_crc=0xABCD1234)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.spare_crc, 0xABCD1234)

    def test_draft_hash_stored_correctly(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(draft_hash=0x0102030405060708)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.draft_hash, 0x0102030405060708)


# ===========================================================================
# 6. SquizdFlag — flag operations
# ===========================================================================

class TestSquizdHeaderFlags(unittest.TestCase):
    """Tests for SquizdFlag enumeration and bitwise operations."""

    def test_astc_flag_value(self) -> None:
        from squish.format.squish_header import SquizdFlag
        self.assertEqual(int(SquizdFlag.ASTC), 1)

    def test_tca_tbe_flag_value(self) -> None:
        from squish.format.squish_header import SquizdFlag
        self.assertEqual(int(SquizdFlag.TCA_TBE), 2)

    def test_int4_flag_value(self) -> None:
        from squish.format.squish_header import SquizdFlag
        self.assertEqual(int(SquizdFlag.INT4), 4)

    def test_flag_combination(self) -> None:
        from squish.format.squish_header import SquizdFlag
        combined = SquizdFlag.ASTC | SquizdFlag.INT4
        self.assertEqual(int(combined), 5)

    def test_has_method_present(self) -> None:
        from squish.format.squish_header import SquizdFlag
        flags = SquizdFlag.ASTC | SquizdFlag.TCA_TBE
        self.assertTrue(flags.has(SquizdFlag.ASTC))
        self.assertFalse(flags.has(SquizdFlag.INT4))

    def test_from_uint32_known_bits(self) -> None:
        from squish.format.squish_header import SquizdFlag
        f = SquizdFlag.from_uint32(0b101)
        self.assertTrue(f.has(SquizdFlag.ASTC))
        self.assertFalse(f.has(SquizdFlag.TCA_TBE))
        self.assertTrue(f.has(SquizdFlag.INT4))

    def test_from_uint32_unknown_bits_dropped(self) -> None:
        from squish.format.squish_header import SquizdFlag
        # Bit 31 is not defined — should be silently dropped
        f = SquizdFlag.from_uint32(0x80000001)
        self.assertTrue(f.has(SquizdFlag.ASTC))
        self.assertFalse(bool(f & 0x80000000))

    def test_flags_roundtrip_through_header(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdFlag
        flags = SquizdFlag.ASTC | SquizdFlag.SPARSE | SquizdFlag.ANE_COREML
        hdr = SquizdHeader(flags=flags)
        restored = SquizdHeader.from_bytes(hdr.serialise())
        self.assertEqual(restored.flags, flags)

    def test_flags_coerced_from_int(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdFlag
        hdr = SquizdHeader(flags=5)  # type: ignore[arg-type]
        self.assertTrue(hdr.flags.has(SquizdFlag.ASTC))
        self.assertTrue(hdr.flags.has(SquizdFlag.INT4))

    def test_none_flag_is_zero(self) -> None:
        from squish.format.squish_header import SquizdFlag
        self.assertEqual(int(SquizdFlag.NONE), 0)

    def test_flags_stored_at_offset_6(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdFlag
        hdr = SquizdHeader(flags=SquizdFlag.ASTC)
        raw = hdr.serialise()
        flags_val = struct.unpack_from("<I", raw, 6)[0]
        self.assertEqual(flags_val, int(SquizdFlag.ASTC))


# ===========================================================================
# 7. SquizdArch enumeration
# ===========================================================================

class TestSquizdHeaderArch(unittest.TestCase):
    """Tests for SquizdArch enumeration."""

    def test_all_known_arch_values(self) -> None:
        from squish.format.squish_header import SquizdArch
        for arch, val in [
            (SquizdArch.UNKNOWN, 0),
            (SquizdArch.LLAMA, 1),
            (SquizdArch.MISTRAL, 2),
            (SquizdArch.QWEN, 3),
            (SquizdArch.GEMMA, 4),
            (SquizdArch.DEEPSEEK, 5),
            (SquizdArch.PHI, 6),
        ]:
            self.assertEqual(int(arch), val)

    def test_unknown_arch_value_maps_to_unknown(self) -> None:
        from squish.format.squish_header import SquizdArch
        self.assertEqual(SquizdArch(99), SquizdArch.UNKNOWN)

    def test_arch_roundtrip_through_header(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdArch
        for arch in [SquizdArch.LLAMA, SquizdArch.QWEN, SquizdArch.PHI]:
            hdr = SquizdHeader(arch_id=arch)
            restored = SquizdHeader.from_bytes(hdr.serialise())
            self.assertEqual(restored.arch_id, arch)

    def test_arch_coerced_from_int(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdArch
        hdr = SquizdHeader(arch_id=3)  # type: ignore[arg-type]
        self.assertEqual(hdr.arch_id, SquizdArch.QWEN)

    def test_arch_id_offset_in_serialised_data(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdArch
        hdr = SquizdHeader(arch_id=SquizdArch.MISTRAL)
        raw = hdr.serialise()
        arch_val = struct.unpack_from("<H", raw, 12)[0]
        self.assertEqual(arch_val, int(SquizdArch.MISTRAL))


# ===========================================================================
# 8. SquizdHeader full round-trip
# ===========================================================================

class TestSquizdHeaderRoundtrip(unittest.TestCase):
    """Full round-trip serialise/from_bytes tests."""

    def _full_header(self):
        from squish.format.squish_header import SquizdHeader, SquizdFlag, SquizdArch
        return SquizdHeader(
            flags=SquizdFlag.ASTC | SquizdFlag.INT4 | SquizdFlag.SPARSE,
            num_layers=32,
            arch_id=SquizdArch.QWEN,
            hidden_dim=4096,
            num_heads=32,
            vocab_size=151936,
            compression_bpw=3.5625,
            sparsity_ratio=0.25,
            calibration_hash=0xCAFEBABE00112233,
            spare_crc=0xDEAD,
            draft_hash=0xBEEF12345678,
        )

    def test_full_roundtrip_flags(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(r.flags, hdr.flags)

    def test_full_roundtrip_num_layers(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(r.num_layers, 32)

    def test_full_roundtrip_arch(self) -> None:
        from squish.format.squish_header import SquizdArch
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(r.arch_id, SquizdArch.QWEN)

    def test_full_roundtrip_hidden_dim(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(r.hidden_dim, 4096)

    def test_full_roundtrip_vocab_size(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(r.vocab_size, 151936)

    def test_full_roundtrip_bpw(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertAlmostEqual(r.compression_bpw, 3.5625, places=4)

    def test_full_roundtrip_sparsity(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertAlmostEqual(r.sparsity_ratio, 0.25, places=6)

    def test_full_roundtrip_calibration_hash(self) -> None:
        hdr = self._full_header()
        r = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(r.calibration_hash, 0xCAFEBABE00112233)

    def test_from_file_roundtrip(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = self._full_header()
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(hdr.serialise())
            tmp = f.name
        try:
            restored = SquizdHeader.from_file(tmp)
            self.assertEqual(restored.num_layers, 32)
            self.assertEqual(restored.hidden_dim, 4096)
        finally:
            os.unlink(tmp)

    def test_equality(self) -> None:
        hdr = self._full_header()
        hdr2 = type(hdr).from_bytes(hdr.serialise())
        self.assertEqual(hdr, hdr2)


# ===========================================================================
# 9. SquizdHeader edge cases
# ===========================================================================

class TestSquizdHeaderEdgeCases(unittest.TestCase):
    """Edge cases: short data, bad magic, future version."""

    def test_data_too_short_raises(self) -> None:
        from squish.format.squish_header import SquizdHeader
        with self.assertRaises(ValueError):
            SquizdHeader.from_bytes(b"\x00" * 10)

    def test_bad_magic_raises(self) -> None:
        from squish.format.squish_header import SquizdHeader
        data = b"BADM" + b"\x00" * 252
        with self.assertRaises(ValueError):
            SquizdHeader.from_bytes(data)

    def test_future_version_raises(self) -> None:
        from squish.format.squish_header import SquizdHeader, SQUIZD_VERSION
        hdr = SquizdHeader()
        raw = bytearray(hdr.serialise())
        # Bump version to SQUIZD_VERSION + 1
        struct.pack_into("<H", raw, 4, SQUIZD_VERSION + 1)
        with self.assertRaises(ValueError):
            SquizdHeader.from_bytes(bytes(raw))

    def test_empty_bytes_raises(self) -> None:
        from squish.format.squish_header import SquizdHeader
        with self.assertRaises(ValueError):
            SquizdHeader.from_bytes(b"")

    def test_exactly_256_bytes_parses(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(num_layers=1)
        raw = hdr.serialise()
        self.assertEqual(len(raw), 256)
        restored = SquizdHeader.from_bytes(raw)
        self.assertEqual(restored.num_layers, 1)

    def test_backward_compat_with_runtime_format(self) -> None:
        """Bytes 0-13 must match squish_runtime.py compact header layout."""
        from squish.format.squish_header import SquizdHeader, SquizdFlag, SquizdArch
        hdr = SquizdHeader(
            flags=SquizdFlag.ASTC,
            num_layers=7,
            arch_id=SquizdArch.LLAMA,
        )
        raw = hdr.serialise()
        # squish_runtime.py reads: magic(4) + "<HI HH" = version, flags, layer_count, arch_id
        magic = raw[:4]
        version, flags_val, layer_count, arch_val = struct.unpack_from("<HIHH", raw, 4)
        self.assertEqual(magic, b"SQZD")
        self.assertEqual(flags_val, int(SquizdFlag.ASTC))
        self.assertEqual(layer_count, 7)
        self.assertEqual(arch_val, int(SquizdArch.LLAMA))

    def test_is_valid_zero_layers_false(self) -> None:
        from squish.format.squish_header import SquizdHeader
        hdr = SquizdHeader(num_layers=0)
        self.assertFalse(hdr.is_valid())

    def test_is_valid_astc_with_bpw(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdFlag
        hdr = SquizdHeader(flags=SquizdFlag.ASTC, num_layers=32, compression_bpw=3.56)
        self.assertTrue(hdr.is_valid())

    def test_summary_contains_arch(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdArch
        hdr = SquizdHeader(arch_id=SquizdArch.QWEN, num_layers=32)
        self.assertIn("QWEN", hdr.summary())


# ===========================================================================
# 10. ASTCLoader simulation mode
# ===========================================================================

class TestASTCLoader(unittest.TestCase):
    """Tests for ASTCLoader in simulation mode (no Metal required)."""

    def setUp(self) -> None:
        os.environ["SQUISH_FORCE_METAL_SIMULATION"] = "1"

    def tearDown(self) -> None:
        os.environ.pop("SQUISH_FORCE_METAL_SIMULATION", None)
        # Reset cached state
        import squish.loaders.astc_loader as mod
        mod._METAL_AVAILABLE = None

    def _make_texture(self, rows: int = 12, cols: int = 12) -> Any:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        from squish.loaders.astc_loader import ASTCLoader, ASTCLoaderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        result = enc.encode(_make_weights(rows, cols))
        loader = ASTCLoader(config=ASTCLoaderConfig(allow_simulation=True))
        return loader.create_texture(result)

    def test_backend_is_simulation(self) -> None:
        tex = self._make_texture()
        self.assertEqual(tex.backend, "simulation")

    def test_mtl_texture_is_none_in_simulation(self) -> None:
        tex = self._make_texture()
        self.assertIsNone(tex.mtl_texture)

    def test_original_shape_matches_input(self) -> None:
        tex = self._make_texture(8, 24)
        self.assertEqual(tex.original_shape, (8, 24))

    def test_decode_returns_float32(self) -> None:
        tex = self._make_texture()
        decoded = tex.decode()
        self.assertEqual(decoded.dtype, np.float32)

    def test_decode_shape_matches_original(self) -> None:
        tex = self._make_texture(8, 24)
        decoded = tex.decode()
        self.assertEqual(decoded.shape, (8, 24))

    def test_texture_descriptor_dict_has_format(self) -> None:
        from squish.loaders.astc_loader import METAL_FORMAT_ASTC_6x6_HDR
        tex = self._make_texture()
        d = tex.texture_descriptor_dict()
        self.assertEqual(d["pixelFormat"], METAL_FORMAT_ASTC_6x6_HDR)

    def test_texture_descriptor_dict_dimensions(self) -> None:
        tex = self._make_texture(12, 12)
        d = tex.texture_descriptor_dict()
        self.assertEqual(d["width"], tex.padded_shape[1])
        self.assertEqual(d["height"], tex.padded_shape[0])

    def test_allow_simulation_false_raises_without_metal(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        from squish.loaders.astc_loader import ASTCLoader, ASTCLoaderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        result = enc.encode(_make_weights())
        loader = ASTCLoader(config=ASTCLoaderConfig(allow_simulation=False))
        with self.assertRaises(RuntimeError):
            loader.create_texture(result)

    def test_n_blocks_property(self) -> None:
        tex = self._make_texture(12, 12)
        expected = (12 // 6) * (12 // 6)
        self.assertEqual(tex.n_blocks, expected)

    def test_scale_table_shape(self) -> None:
        tex = self._make_texture(12, 12)
        self.assertEqual(tex.scale_table.shape, (tex.n_blocks,))

    def test_layer_name_stored(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        from squish.loaders.astc_loader import ASTCLoader, ASTCLoaderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        result = enc.encode(_make_weights())
        loader = ASTCLoader(config=ASTCLoaderConfig(allow_simulation=True))
        tex = loader.create_texture(result, layer_name="q_proj")
        self.assertEqual(tex.layer_name, "q_proj")

    def test_load_from_file(self) -> None:
        from squish.compress.astc_encoder import ASTCEncoder, ASTCEncoderConfig
        from squish.loaders.astc_loader import ASTCLoader, ASTCLoaderConfig
        enc = ASTCEncoder(config=ASTCEncoderConfig(), force_numpy_fallback=True)
        result = enc.encode(_make_weights(12, 12))
        payload = result.serialise()
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(payload)
            tmp = f.name
        try:
            loader = ASTCLoader(config=ASTCLoaderConfig(allow_simulation=True))
            tex = loader.load_from_file(tmp, layer_name="k_proj")
            self.assertEqual(tex.original_shape, (12, 12))
            self.assertEqual(tex.layer_name, "k_proj")
        finally:
            os.unlink(tmp)

    def test_is_metal_available_false_in_simulation_env(self) -> None:
        from squish.loaders.astc_loader import is_metal_available
        self.assertFalse(is_metal_available())

    def test_config_is_accessible(self) -> None:
        from squish.loaders.astc_loader import ASTCLoader, ASTCLoaderConfig
        cfg = ASTCLoaderConfig(allow_simulation=True, verify_on_load=True)
        loader = ASTCLoader(config=cfg)
        self.assertTrue(loader.config.verify_on_load)


# ===========================================================================
# 11. encode_weight_tensor convenience function
# ===========================================================================

class TestEncodeWeightTensorConvenience(unittest.TestCase):
    """Tests for the module-level encode_weight_tensor helper."""

    def test_returns_astc_encode_result(self) -> None:
        from squish.compress.astc_encoder import ASTCEncodeResult, encode_weight_tensor
        result = encode_weight_tensor(_make_weights(), force_numpy_fallback=True)
        self.assertIsInstance(result, ASTCEncodeResult)

    def test_original_shape_matches(self) -> None:
        from squish.compress.astc_encoder import encode_weight_tensor
        w = _make_weights(16, 32)
        result = encode_weight_tensor(w, force_numpy_fallback=True)
        self.assertEqual(result.original_shape, (16, 32))

    def test_custom_block_size(self) -> None:
        from squish.compress.astc_encoder import encode_weight_tensor
        w = _make_weights(8, 8)
        result = encode_weight_tensor(w, block_x=4, block_y=4, force_numpy_fallback=True)
        self.assertEqual(result.padded_shape[0] % 4, 0)
        self.assertEqual(result.padded_shape[1] % 4, 0)

    def test_quality_param_accepted(self) -> None:
        from squish.compress.astc_encoder import encode_weight_tensor
        # Should not raise
        result = encode_weight_tensor(_make_weights(), quality=50.0, force_numpy_fallback=True)
        self.assertGreater(result.n_blocks, 0)

    def test_bpw_near_3_56_for_6x6(self) -> None:
        from squish.compress.astc_encoder import encode_weight_tensor
        # For an exact 6×6 block (36 weights → 16 bytes = 128 bits)
        # BPW = 128/36 ≈ 3.556
        w = np.ones((6, 6), dtype=np.float32)
        result = encode_weight_tensor(w, force_numpy_fallback=True)
        self.assertAlmostEqual(result.bpw, 128 / 36, places=2)


# ===========================================================================
# 12. build_minimal_header — backward compat helper
# ===========================================================================

class TestBuildMinimalHeader(unittest.TestCase):
    """Tests for build_minimal_header and compatibility with squish_runtime."""

    def test_output_is_256_bytes(self) -> None:
        from squish.format.squish_header import build_minimal_header, SquizdFlag, SquizdArch
        raw = build_minimal_header(SquizdFlag.ASTC, num_layers=32, arch_id=SquizdArch.QWEN)
        self.assertEqual(len(raw), 256)

    def test_magic_correct(self) -> None:
        from squish.format.squish_header import build_minimal_header, SquizdFlag, SquizdArch
        raw = build_minimal_header(SquizdFlag.NONE, num_layers=1, arch_id=SquizdArch.UNKNOWN)
        self.assertEqual(raw[:4], b"SQZD")

    def test_flags_at_offset_6(self) -> None:
        from squish.format.squish_header import build_minimal_header, SquizdFlag, SquizdArch
        raw = build_minimal_header(SquizdFlag.ASTC | SquizdFlag.INT4, num_layers=8, arch_id=SquizdArch.LLAMA)
        flags_val = struct.unpack_from("<I", raw, 6)[0]
        self.assertEqual(flags_val, int(SquizdFlag.ASTC | SquizdFlag.INT4))

    def test_num_layers_at_offset_10(self) -> None:
        from squish.format.squish_header import build_minimal_header, SquizdFlag, SquizdArch
        raw = build_minimal_header(SquizdFlag.NONE, num_layers=42, arch_id=SquizdArch.UNKNOWN)
        layers = struct.unpack_from("<H", raw, 10)[0]
        self.assertEqual(layers, 42)

    def test_arch_id_at_offset_12(self) -> None:
        from squish.format.squish_header import build_minimal_header, SquizdFlag, SquizdArch
        raw = build_minimal_header(SquizdFlag.NONE, num_layers=1, arch_id=SquizdArch.DEEPSEEK)
        arch = struct.unpack_from("<H", raw, 12)[0]
        self.assertEqual(arch, int(SquizdArch.DEEPSEEK))

    def test_parseable_by_from_bytes(self) -> None:
        from squish.format.squish_header import (
            SquizdHeader, SquizdFlag, SquizdArch, build_minimal_header,
        )
        raw = build_minimal_header(SquizdFlag.ASTC, num_layers=16, arch_id=SquizdArch.GEMMA)
        hdr = SquizdHeader.from_bytes(raw)
        self.assertTrue(hdr.flags.has(SquizdFlag.ASTC))
        self.assertEqual(hdr.num_layers, 16)
        self.assertEqual(hdr.arch_id, SquizdArch.GEMMA)


# ===========================================================================
# 13. read_header helper
# ===========================================================================

class TestReadHeaderHelper(unittest.TestCase):
    """Tests for the read_header convenience function."""

    def test_returns_none_for_nonexistent_file(self) -> None:
        from squish.format.squish_header import read_header
        result = read_header("/tmp/__squish_nonexistent_file__.squizd")
        self.assertIsNone(result)

    def test_returns_none_for_short_file(self) -> None:
        from squish.format.squish_header import read_header
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(b"\x00" * 10)
            tmp = f.name
        try:
            result = read_header(tmp)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp)

    def test_returns_none_for_bad_magic(self) -> None:
        from squish.format.squish_header import read_header
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(b"NOPE" + b"\x00" * 252)
            tmp = f.name
        try:
            result = read_header(tmp)
            self.assertIsNone(result)
        finally:
            os.unlink(tmp)

    def test_returns_header_for_valid_file(self) -> None:
        from squish.format.squish_header import SquizdHeader, SquizdFlag, SquizdArch, read_header
        hdr = SquizdHeader(
            flags=SquizdFlag.ASTC,
            num_layers=8,
            arch_id=SquizdArch.PHI,
        )
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(hdr.serialise())
            tmp = f.name
        try:
            result = read_header(tmp)
            self.assertIsNotNone(result)
            assert result is not None
            self.assertEqual(result.num_layers, 8)
            self.assertEqual(result.arch_id, SquizdArch.PHI)
        finally:
            os.unlink(tmp)

    def test_read_header_accepts_path_object(self) -> None:
        from squish.format.squish_header import SquizdHeader, read_header
        hdr = SquizdHeader(num_layers=4)
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(hdr.serialise())
            tmp = f.name
        try:
            result = read_header(Path(tmp))
            self.assertIsNotNone(result)
        finally:
            os.unlink(tmp)


# ===========================================================================
# 14. is_astcenc_available
# ===========================================================================

class TestIsAstcencAvailable(unittest.TestCase):
    """Tests for the libastcenc availability probe."""

    def test_returns_bool(self) -> None:
        from squish.compress.astc_encoder import is_astcenc_available
        result = is_astcenc_available()
        self.assertIsInstance(result, bool)

    def test_force_lib_env_var(self) -> None:
        """SQUISH_ASTCENC_LIB pointing to a non-existent path → False."""
        import squish.compress.astc_encoder as mod
        old_available = mod._LIB_AVAILABLE
        old_lib = mod._LIB_ASTCENC
        try:
            mod._LIB_AVAILABLE = None
            mod._LIB_ASTCENC = None
            with patch.dict(os.environ, {"SQUISH_ASTCENC_LIB": "/nonexistent/libastcenc.so"}):
                result = mod._probe_libastcenc()
            self.assertFalse(result)
        finally:
            mod._LIB_AVAILABLE = old_available
            mod._LIB_ASTCENC = old_lib


# ===========================================================================
# 15. ASTC constants
# ===========================================================================

class TestASTCConstants(unittest.TestCase):
    """Validate module-level constants."""

    def test_block_bytes_is_16(self) -> None:
        from squish.compress.astc_encoder import ASTC_BLOCK_BYTES
        self.assertEqual(ASTC_BLOCK_BYTES, 16)

    def test_block_x_is_6(self) -> None:
        from squish.compress.astc_encoder import ASTC_BLOCK_X
        self.assertEqual(ASTC_BLOCK_X, 6)

    def test_block_y_is_6(self) -> None:
        from squish.compress.astc_encoder import ASTC_BLOCK_Y
        self.assertEqual(ASTC_BLOCK_Y, 6)

    def test_metal_format_constant(self) -> None:
        from squish.loaders.astc_loader import METAL_FORMAT_ASTC_6x6_HDR
        self.assertEqual(METAL_FORMAT_ASTC_6x6_HDR, 124)

    def test_squizd_header_size_is_256(self) -> None:
        from squish.format.squish_header import SQUIZD_HEADER_SIZE
        self.assertEqual(SQUIZD_HEADER_SIZE, 256)

    def test_squizd_magic(self) -> None:
        from squish.format.squish_header import SQUIZD_MAGIC
        self.assertEqual(SQUIZD_MAGIC, b"SQZD")

    def test_squizd_version_is_1(self) -> None:
        from squish.format.squish_header import SQUIZD_VERSION
        self.assertEqual(SQUIZD_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
