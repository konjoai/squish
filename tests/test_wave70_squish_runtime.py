"""tests/test_wave70_squish_runtime.py

Unit tests for Wave 70: SquishRuntime — unified SQUIZD dispatch path.

Modules under test
──────────────────
* squish.runtime.squish_runtime — SquishRuntime, SquizdFlags, SquizdHeader
* squish.runtime.format_validator — SquizdFormatValidator, SquizdFormatError,
  ValidationResult

All tests run without hardware, Metal, or coremltools.  File I/O uses
Python's ``tempfile`` module so tests are portable across macOS / Linux / CI.
"""
from __future__ import annotations

import binascii
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants kept in sync with the modules under test
# ---------------------------------------------------------------------------
MAGIC = b"SQZD"
HEADER_SIZE = 256
CURRENT_VERSION = 1


def _build_header(
    flags: int = 0,
    layer_count: int = 32,
    arch_id: int = 0,
    version: int = CURRENT_VERSION,
    override_magic: Optional[bytes] = None,
    sparsity_crc: int = 0,
    eagle_hash: int = 0,
) -> bytes:
    """Build a valid 256-byte SQUIZD header."""
    magic = override_magic if override_magic is not None else MAGIC
    raw = (
        magic
        + struct.pack("<HI HH I Q", version, flags, layer_count, arch_id, sparsity_crc, eagle_hash)
    )
    return raw.ljust(HEADER_SIZE, b"\x00")


# =============================================================================
# 1. SquizdFlags
# =============================================================================

class TestSquizdFlagsBasic(unittest.TestCase):
    def test_none_value(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.NONE), 0)

    def test_astc_bit_zero(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.ASTC), 1)

    def test_tca_tbe_bit_one(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.TCA_TBE), 2)

    def test_int4_bit_two(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.INT4), 4)

    def test_sparse_bit_three(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.SPARSE), 8)

    def test_eagle_bit_four(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.EAGLE), 16)

    def test_int2_bit_five(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.INT2), 32)

    def test_ane_coreml_bit_six(self):
        from squish.runtime.squish_runtime import SquizdFlags
        self.assertEqual(int(SquizdFlags.ANE_COREML), 64)


class TestSquizdFlagsOps(unittest.TestCase):
    def test_combination(self):
        from squish.runtime.squish_runtime import SquizdFlags
        combined = SquizdFlags.ASTC | SquizdFlags.INT4
        self.assertTrue(combined.has(SquizdFlags.ASTC))
        self.assertTrue(combined.has(SquizdFlags.INT4))
        self.assertFalse(combined.has(SquizdFlags.SPARSE))

    def test_from_uint32_zero(self):
        from squish.runtime.squish_runtime import SquizdFlags
        f = SquizdFlags.from_uint32(0)
        self.assertEqual(f, SquizdFlags.NONE)

    def test_from_uint32_astc(self):
        from squish.runtime.squish_runtime import SquizdFlags
        f = SquizdFlags.from_uint32(1)
        self.assertTrue(f.has(SquizdFlags.ASTC))

    def test_from_uint32_all_bits(self):
        from squish.runtime.squish_runtime import SquizdFlags
        val = (1 << 7) - 1  # bits 0–6 set
        f = SquizdFlags.from_uint32(val)
        for flag in [
            SquizdFlags.ASTC, SquizdFlags.TCA_TBE, SquizdFlags.INT4,
            SquizdFlags.SPARSE, SquizdFlags.EAGLE, SquizdFlags.INT2,
            SquizdFlags.ANE_COREML,
        ]:
            self.assertTrue(f.has(flag))

    def test_has_false_for_unset(self):
        from squish.runtime.squish_runtime import SquizdFlags
        f = SquizdFlags.INT4
        self.assertFalse(f.has(SquizdFlags.ASTC))

    def test_from_uint32_truncates_to_32_bits(self):
        from squish.runtime.squish_runtime import SquizdFlags
        big = (1 << 32) | 1
        f = SquizdFlags.from_uint32(big)
        self.assertTrue(f.has(SquizdFlags.ASTC))


# =============================================================================
# 2. SquizdHeader
# =============================================================================

class TestSquizdHeader(unittest.TestCase):
    def _make(self, magic=MAGIC, version=1, flags=0, layers=32, arch_id=0):
        from squish.runtime.squish_runtime import SquizdFlags, SquizdHeader
        raw = _build_header(flags=flags, layer_count=layers, arch_id=arch_id,
                            version=version, override_magic=magic)
        return SquizdHeader(
            magic=magic,
            version=version,
            flags=SquizdFlags.from_uint32(flags),
            layer_count=layers,
            arch_id=arch_id,
            raw_bytes=raw,
        )

    def test_is_valid_good(self):
        h = self._make()
        self.assertTrue(h.is_valid)

    def test_is_valid_bad_magic(self):
        h = self._make(magic=b"XXXX")
        self.assertFalse(h.is_valid)

    def test_is_valid_bad_version(self):
        h = self._make(version=99)
        self.assertFalse(h.is_valid)

    def test_summary_no_flags(self):
        h = self._make(flags=0)
        s = h.summary()
        self.assertIn("SQUIZD", s)
        self.assertIn("NONE", s)

    def test_summary_astc_flag(self):
        h = self._make(flags=1)
        self.assertIn("ASTC", h.summary())

    def test_layer_count_preserved(self):
        h = self._make(layers=64)
        self.assertEqual(h.layer_count, 64)


# =============================================================================
# 3. SquishRuntime.from_flags constructor
# =============================================================================

class TestSquishRuntimeFromFlags(unittest.TestCase):
    def test_from_flags_none(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.NONE, layer_count=8)
        self.assertEqual(rt.layer_count, 8)
        self.assertEqual(rt.active_flags, SquizdFlags.NONE)

    def test_from_flags_astc(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.ASTC, layer_count=4)
        self.assertTrue(rt.active_flags.has(SquizdFlags.ASTC))

    def test_from_flags_int4(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4)
        self.assertTrue(rt.active_flags.has(SquizdFlags.INT4))

    def test_from_flags_multiple(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        flags = SquizdFlags.INT4 | SquizdFlags.SPARSE
        rt = SquishRuntime.from_flags(flags, layer_count=16)
        self.assertTrue(rt.active_flags.has(SquizdFlags.INT4))
        self.assertTrue(rt.active_flags.has(SquizdFlags.SPARSE))

    def test_layer_count_propagated(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.NONE, layer_count=24)
        self.assertEqual(rt.layer_count, 24)

    def test_header_is_valid(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4)
        self.assertTrue(rt.header.is_valid)


# =============================================================================
# 4. SquishRuntime.from_file constructor
# =============================================================================

class TestSquishRuntimeFromFile(unittest.TestCase):
    def _write_squizd(self, tmp: Path, flags: int = 0, layers: int = 32) -> Path:
        path = tmp / "model.squizd"
        path.write_bytes(_build_header(flags=flags, layer_count=layers))
        return path

    def test_from_file_valid(self):
        from squish.runtime.squish_runtime import SquishRuntime
        with tempfile.TemporaryDirectory() as td:
            path = self._write_squizd(Path(td), flags=4, layers=16)
            rt = SquishRuntime.from_file(path)
            self.assertTrue(rt.header.is_valid)
            self.assertEqual(rt.layer_count, 16)

    def test_from_file_missing_path(self):
        from squish.runtime.squish_runtime import SquishRuntime
        rt = SquishRuntime.from_file("/nonexistent/file.squizd")
        self.assertFalse(rt.header.is_valid)
        self.assertEqual(rt.layer_count, 0)

    def test_from_file_truncated(self):
        from squish.runtime.squish_runtime import SquishRuntime
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "short.squizd"
            path.write_bytes(MAGIC + b"\x00" * 10)
            rt = SquishRuntime.from_file(path)
            self.assertFalse(rt.header.is_valid)

    def test_from_file_wrong_magic(self):
        from squish.runtime.squish_runtime import SquishRuntime
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bad.squizd"
            path.write_bytes(_build_header(override_magic=b"XXXX"))
            rt = SquishRuntime.from_file(path)
            self.assertFalse(rt.header.is_valid)

    def test_from_file_flags_propagated(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        with tempfile.TemporaryDirectory() as td:
            path = self._write_squizd(Path(td), flags=int(SquizdFlags.INT4))
            rt = SquishRuntime.from_file(path)
            self.assertTrue(rt.active_flags.has(SquizdFlags.INT4))


# =============================================================================
# 5. Dispatch table
# =============================================================================

class TestDispatchTable(unittest.TestCase):
    def test_dispatch_table_length(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4, layer_count=12)
        self.assertEqual(len(rt.dispatch_table), 12)

    def test_dispatch_astc_selects_astc(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.ASTC, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.ASTC)

    def test_dispatch_int4_selects_int4(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.INT4)

    def test_dispatch_int2_selects_int2(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT2, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.INT2)

    def test_dispatch_tca_tbe_selects_tca(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.TCA_TBE, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.TCA_TBE)

    def test_dispatch_none_selects_numpy(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.NONE, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.NUMPY)

    def test_dispatch_ane_coreml_selects_coreml(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.ANE_COREML, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.COREML)

    def test_dispatch_ane_beats_astc(self):
        from squish.runtime.squish_runtime import KernelStack, SquishRuntime, SquizdFlags
        flags = SquizdFlags.ANE_COREML | SquizdFlags.ASTC
        rt = SquishRuntime.from_flags(flags, layer_count=2)
        for rec in rt.dispatch_table:
            self.assertEqual(rec.kernel_stack, KernelStack.COREML)

    def test_sparse_flag_sets_sparse_enabled(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4 | SquizdFlags.SPARSE, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertTrue(rec.sparse_enabled)

    def test_no_sparse_flag_unsets_sparse_enabled(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertFalse(rec.sparse_enabled)

    def test_eagle_flag_sets_draft_enabled(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.EAGLE, layer_count=4)
        for rec in rt.dispatch_table:
            self.assertTrue(rec.draft_enabled)

    def test_dispatch_table_layer_indices(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        n = 8
        rt = SquishRuntime.from_flags(SquizdFlags.INT4, layer_count=n)
        for i, rec in enumerate(rt.dispatch_table):
            self.assertEqual(rec.layer_idx, i)

    def test_dispatch_table_is_copy(self):
        """dispatch_table property returns a copy, not the internal list."""
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        rt = SquishRuntime.from_flags(SquizdFlags.INT4, layer_count=4)
        t1 = rt.dispatch_table
        t2 = rt.dispatch_table
        self.assertIsNot(t1, t2)


# =============================================================================
# 6. generate() and generate_stream()
# =============================================================================

class TestGenerate(unittest.TestCase):
    def _rt(self, flags=0, layers=4):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        return SquishRuntime.from_flags(SquizdFlags.from_uint32(flags), layer_count=layers)

    def test_generate_returns_string(self):
        rt = self._rt()
        result = rt.generate("hello", max_new_tokens=5, seed=0)
        self.assertIsInstance(result, str)

    def test_generate_deterministic_with_seed(self):
        rt = self._rt()
        r1 = rt.generate("hello", max_new_tokens=10, seed=42)
        r2 = rt.generate("hello", max_new_tokens=10, seed=42)
        self.assertEqual(r1, r2)

    def test_generate_different_seeds_differ(self):
        rt = self._rt()
        r1 = rt.generate("hello", max_new_tokens=20, seed=1)
        r2 = rt.generate("hello", max_new_tokens=20, seed=999)
        # Statistically extremely unlikely to be equal.
        self.assertNotEqual(r1, r2)

    def test_generate_stream_yields_pairs(self):
        rt = self._rt()
        for token, reason in rt.generate_stream("test", max_new_tokens=3, seed=0):
            self.assertIsInstance(token, str)
            self.assertIn(reason, (None, "stop", "length"))

    def test_generate_stream_last_has_finish_reason(self):
        rt = self._rt()
        pairs = list(rt.generate_stream("hello", max_new_tokens=5, seed=0))
        self.assertIsNotNone(pairs[-1][1])

    def test_generate_max_new_tokens_observed(self):
        rt = self._rt()
        pairs = list(rt.generate_stream("hi", max_new_tokens=4, seed=0))
        self.assertLessEqual(len(pairs), 4)

    def test_generate_empty_prompt(self):
        rt = self._rt()
        result = rt.generate("", max_new_tokens=3, seed=0)
        self.assertIsInstance(result, str)

    def test_generate_with_sparse_flags(self):
        from squish.runtime.squish_runtime import SquizdFlags
        rt = self._rt(flags=int(SquizdFlags.INT4 | SquizdFlags.SPARSE), layers=4)
        result = rt.generate("sparse test", max_new_tokens=4, seed=5)
        self.assertIsInstance(result, str)

    def test_generate_with_eagle_flags(self):
        from squish.runtime.squish_runtime import SquizdFlags
        rt = self._rt(flags=int(SquizdFlags.EAGLE), layers=4)
        result = rt.generate("eagle test", max_new_tokens=4, seed=5)
        self.assertIsInstance(result, str)


# =============================================================================
# 7. build_squizd_header helper
# =============================================================================

class TestBuildSquizdHeader(unittest.TestCase):
    def test_header_size(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        h = SquishRuntime.build_squizd_header(SquizdFlags.INT4)
        self.assertEqual(len(h), HEADER_SIZE)

    def test_magic_present(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        h = SquishRuntime.build_squizd_header(SquizdFlags.INT4)
        self.assertEqual(h[:4], MAGIC)

    def test_version_encoded(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        h = SquishRuntime.build_squizd_header(SquizdFlags.INT4)
        version, = struct.unpack_from("<H", h, 4)
        self.assertEqual(version, CURRENT_VERSION)

    def test_flags_encoded(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        h = SquishRuntime.build_squizd_header(SquizdFlags.ASTC | SquizdFlags.INT4)
        flags, = struct.unpack_from("<I", h, 6)
        self.assertEqual(flags & 0x5, 0x5)

    def test_layer_count_encoded(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        h = SquishRuntime.build_squizd_header(SquizdFlags.INT4, layer_count=48)
        lc, = struct.unpack_from("<H", h, 10)
        self.assertEqual(lc, 48)

    def test_arch_id_encoded(self):
        from squish.runtime.squish_runtime import SquishRuntime, SquizdFlags
        h = SquishRuntime.build_squizd_header(SquizdFlags.INT4, arch_id=7)
        aid, = struct.unpack_from("<H", h, 12)
        self.assertEqual(aid, 7)


# =============================================================================
# 8. SquizdFormatValidator
# =============================================================================

class TestFormatValidatorGoodFile(unittest.TestCase):
    def _good_header(self, flags=4, layers=32):
        return _build_header(flags=flags, layer_count=layers)

    def _write(self, tmp, data):
        p = Path(tmp) / "model.squizd"
        p.write_bytes(data)
        return p

    def test_valid_file_passes(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, self._good_header())
            result = SquizdFormatValidator().validate(path)
            self.assertTrue(result.valid)
            self.assertTrue(result.magic_ok)
            self.assertTrue(result.version_ok)
            self.assertTrue(result.layer_count_ok)

    def test_valid_bytes_passes(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        result = SquizdFormatValidator().validate_bytes(self._good_header())
        self.assertTrue(result.valid)

    def test_no_errors_on_success(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        result = SquizdFormatValidator().validate_bytes(self._good_header())
        self.assertEqual(result.errors, [])

    def test_success_classmethod(self):
        from squish.runtime.format_validator import ValidationResult
        r = ValidationResult.success()
        self.assertTrue(r.valid)

    def test_to_dict_has_all_keys(self):
        from squish.runtime.format_validator import ValidationResult
        r = ValidationResult.success()
        d = r.to_dict()
        for key in ["valid", "magic_ok", "version_ok", "layer_count_ok",
                    "sparsity_crc_ok", "eagle_hash_ok", "errors"]:
            self.assertIn(key, d)


class TestFormatValidatorBadMagic(unittest.TestCase):
    def test_wrong_magic_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(override_magic=b"XXXX")
        result = SquizdFormatValidator().validate_bytes(data)
        self.assertFalse(result.magic_ok)
        self.assertFalse(result.valid)
        self.assertTrue(len(result.errors) > 0)

    def test_missing_file_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        result = SquizdFormatValidator().validate("/no/such/file.squizd")
        self.assertFalse(result.valid)
        self.assertTrue(any("not found" in e.lower() for e in result.errors))

    def test_truncated_data_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        result = SquizdFormatValidator().validate_bytes(MAGIC + b"\x00" * 10)
        self.assertFalse(result.valid)


class TestFormatValidatorVersion(unittest.TestCase):
    def test_version_zero_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(version=0)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertFalse(r.version_ok)
        self.assertFalse(r.valid)

    def test_version_99_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(version=99)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertFalse(r.version_ok)

    def test_version_one_passes(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(version=1)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertTrue(r.version_ok)


class TestFormatValidatorLayerCount(unittest.TestCase):
    def test_zero_layers_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(layer_count=0)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertFalse(r.layer_count_ok)

    def test_512_layers_ok(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(layer_count=512)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertTrue(r.layer_count_ok)

    def test_513_layers_fails(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        data = _build_header(layer_count=513)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertFalse(r.layer_count_ok)


class TestFormatValidatorAssertValid(unittest.TestCase):
    def test_assert_valid_good_file_no_raise(self):
        from squish.runtime.format_validator import SquizdFormatValidator
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "good.squizd"
            p.write_bytes(_build_header())
            SquizdFormatValidator().assert_valid(p)  # must not raise

    def test_assert_valid_bad_file_raises(self):
        from squish.runtime.format_validator import SquizdFormatError, SquizdFormatValidator
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "bad.squizd"
            p.write_bytes(_build_header(override_magic=b"XXXX"))
            with self.assertRaises(SquizdFormatError):
                SquizdFormatValidator().assert_valid(p)

    def test_squizd_format_error_has_errors_list(self):
        from squish.runtime.format_validator import SquizdFormatError
        err = SquizdFormatError("oops", errors=["bad magic"])
        self.assertEqual(len(err.errors), 1)

    def test_squizd_format_error_str(self):
        from squish.runtime.format_validator import SquizdFormatError
        err = SquizdFormatError("oops", errors=["bad magic"])
        self.assertIn("bad magic", str(err))


class TestFormatValidatorSparisityCRC(unittest.TestCase):
    def test_sparsity_crc_zero_with_sparse_flag_passes(self):
        """CRC = 0 over all-zero sparsity block is treated as 'not populated' → pass."""
        from squish.runtime.format_validator import SquizdFormatValidator
        zero_block_crc = binascii.crc32(b"\x00" * 32) & 0xFFFFFFFF
        data = _build_header(flags=8, sparsity_crc=zero_block_crc)
        r = SquizdFormatValidator().validate_bytes(data)
        self.assertTrue(r.sparsity_crc_ok)

    def test_sparsity_crc_bad_fails(self):
        """Non-matching stored CRC should fail validation."""
        from squish.runtime.format_validator import SquizdFormatValidator
        data = bytearray(_build_header(flags=8, sparsity_crc=0xDEADBEEF))
        r = SquizdFormatValidator().validate_bytes(bytes(data))
        self.assertFalse(r.sparsity_crc_ok)


# =============================================================================
# 9. SQUIZD_MAGIC and SQUIZD_VERSION constants
# =============================================================================

class TestModuleConstants(unittest.TestCase):
    def test_magic_bytes(self):
        from squish.runtime.squish_runtime import SQUIZD_MAGIC
        self.assertEqual(SQUIZD_MAGIC, b"SQZD")

    def test_version_is_one(self):
        from squish.runtime.squish_runtime import SQUIZD_VERSION
        self.assertEqual(SQUIZD_VERSION, 1)

    def test_validator_magic_constant(self):
        from squish.runtime.format_validator import SQUIZD_MAGIC
        self.assertEqual(SQUIZD_MAGIC, b"SQZD")

    def test_validator_version_constants(self):
        from squish.runtime.format_validator import SQUIZD_CURRENT_VERSION, SQUIZD_MIN_VERSION
        self.assertEqual(SQUIZD_CURRENT_VERSION, 1)
        self.assertEqual(SQUIZD_MIN_VERSION, 1)


if __name__ == "__main__":
    unittest.main()
