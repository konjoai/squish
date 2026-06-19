"""Behavioral coverage for the SQUIZD format validator branches left untested
by the baseline suite: the multi-error string form, the EAGLE draft-head hash
and sparsity-CRC checks, strict reserved-byte enforcement, and the FNV-1a-64
helper.

Pure-Python — builds in-memory 256-byte headers; no MLX, no disk needed.
"""
from __future__ import annotations

import binascii
import struct

from squish.runtime.format_validator import (
    SQUIZD_MAGIC,
    SquizdFormatError,
    SquizdFormatValidator,
    _fnv1a_64,
)


def _header(*, version=1, flags=0, layer_count=2, arch_id=0,
            spare_crc=0, draft_hash=0, size=256) -> bytes:
    buf = bytearray(b"\x00" * size)
    buf[0:4] = SQUIZD_MAGIC
    struct.pack_into("<H", buf, 4, version)
    struct.pack_into("<I", buf, 6, flags)
    struct.pack_into("<H", buf, 10, layer_count)
    struct.pack_into("<H", buf, 12, arch_id)
    struct.pack_into("<I", buf, 14, spare_crc)
    struct.pack_into("<Q", buf, 18, draft_hash)
    return bytes(buf)


_FLAG_SPARSE = 1 << 3
_FLAG_EAGLE = 1 << 4


# ── SquizdFormatError.__str__ ───────────────────────────────────────────────


def test_error_str_single_vs_multi():
    assert str(SquizdFormatError("only one")) == "only one"
    multi = SquizdFormatError("m", errors=["first", "second"])
    assert str(multi) == "  [1] first\n  [2] second"  # multi-error branch (82)


# ── happy path ──────────────────────────────────────────────────────────────


def test_valid_header_passes():
    r = SquizdFormatValidator().validate_bytes(_header())
    assert r.valid is True and r.errors == []
    assert r.to_dict()["valid"] is True


# ── _fnv1a_64 ───────────────────────────────────────────────────────────────


def test_fnv1a_64_known_vector():
    # FNV-1a 64-bit of empty input is the offset basis.
    assert _fnv1a_64(b"") == 0xCBF29CE484222325
    # Non-empty input differs and stays within 64 bits.
    h = _fnv1a_64(b"squish")
    assert 0 <= h < 2 ** 64 and h != 0xCBF29CE484222325


# ── EAGLE draft-head hash check ─────────────────────────────────────────────


def test_eagle_flag_zero_hash_passes():
    # EAGLE flag set but draft_hash == 0 → "not set" → passes (349-350).
    r = SquizdFormatValidator().validate_bytes(_header(flags=_FLAG_EAGLE, draft_hash=0))
    assert r.eagle_hash_ok is True and r.valid is True


def test_eagle_hash_match_passes():
    buf = bytearray(_header(flags=_FLAG_EAGLE))
    buf[160:192] = bytes(range(32))  # populate the draft-head block
    correct = _fnv1a_64(bytes(buf[160:192]))
    struct.pack_into("<Q", buf, 18, correct)
    r = SquizdFormatValidator().validate_bytes(bytes(buf))
    assert r.eagle_hash_ok is True and r.valid is True


def test_eagle_hash_mismatch_fails():
    buf = bytearray(_header(flags=_FLAG_EAGLE))
    buf[160:192] = bytes(range(32))
    struct.pack_into("<Q", buf, 18, 0xDEADBEEFCAFEBABE)  # wrong hash (353-358)
    r = SquizdFormatValidator().validate_bytes(bytes(buf))
    assert r.eagle_hash_ok is False and r.valid is False
    assert any("draft-head hash mismatch" in e for e in r.errors)


# ── sparsity CRC check ──────────────────────────────────────────────────────


def test_sparsity_zero_block_and_crc_passes():
    # SPARSE flag, zero block, stored CRC 0 → not-yet-populated → passes (330-331).
    r = SquizdFormatValidator().validate_bytes(_header(flags=_FLAG_SPARSE, spare_crc=0))
    assert r.sparsity_crc_ok is True and r.valid is True


def test_sparsity_crc_match_passes():
    buf = bytearray(_header(flags=_FLAG_SPARSE))
    buf[128:160] = bytes(range(1, 33))
    crc = binascii.crc32(bytes(buf[128:160])) & 0xFFFFFFFF
    struct.pack_into("<I", buf, 14, crc)
    r = SquizdFormatValidator().validate_bytes(bytes(buf))
    assert r.sparsity_crc_ok is True and r.valid is True


def test_sparsity_crc_mismatch_fails():
    buf = bytearray(_header(flags=_FLAG_SPARSE))
    buf[128:160] = bytes(range(1, 33))
    struct.pack_into("<I", buf, 14, 0x12345678)  # wrong CRC (332-337)
    r = SquizdFormatValidator().validate_bytes(bytes(buf))
    assert r.sparsity_crc_ok is False and r.valid is False
    assert any("sparsity CRC32 mismatch" in e for e in r.errors)


# ── strict reserved-byte enforcement ────────────────────────────────────────


def test_strict_reserved_bytes_must_be_zero():
    buf = bytearray(_header())
    buf[200] = 0xFF  # a reserved byte (>= _OFF_RESERVED) is non-zero
    r = SquizdFormatValidator(strict=True).validate_bytes(bytes(buf))
    # All structural checks pass, but strict mode flags the reserved byte and
    # marks the result invalid (269-273, 282-283).
    assert r.magic_ok and r.version_ok and r.layer_count_ok
    assert r.valid is False
    assert any("reserved header bytes" in e for e in r.errors)


def test_non_strict_ignores_reserved_bytes():
    buf = bytearray(_header())
    buf[200] = 0xFF
    r = SquizdFormatValidator(strict=False).validate_bytes(bytes(buf))
    assert r.valid is True  # reserved bytes tolerated in non-strict mode


def test_strict_valid_header_with_zero_reserved_passes():
    # strict mode + all reserved bytes zero → the any(non-zero) check is False
    # and validation passes (270→275).
    r = SquizdFormatValidator(strict=True).validate_bytes(_header())
    assert r.valid is True and r.errors == []
