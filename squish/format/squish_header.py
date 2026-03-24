# [Experimental] This module is part of Squish v38+ (Wave 64).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""SquizdHeader — ``.squizd`` binary format header v0.1 definition.

This module is the canonical definition of the SQUIZD binary header.  All
components of the SQUIZD stack (Waves 64–70) read and write this structure.

Header layout (256 bytes, little-endian)
─────────────────────────────────────────
Offset  Size  Type    Field
──────  ────  ──────  ────────────────────────────────────────────────────
 0       4    bytes   magic — always b"SQZD"
 4       2    u16     version — format semantic version (currently 1)
 6       4    u32     flags — compression feature bitfield (see :class:`SquizdFlag`)
10       2    u16     num_layers — number of transformer layers
12       2    u16     arch_id — model architecture (see :class:`SquizdArch`)
14       4    u32     spare_crc — CRC32 of sparsity metadata block (0 if absent)
18       8    u64     draft_hash — FNV-64 hash of EAGLE draft-head appendix (0 if absent)
26       4    u32     hidden_dim — model hidden dimension
30       2    u16     num_heads — number of attention heads
32       4    u32     vocab_size — vocabulary size
36       4    f32     compression_bpw — actual bits per weight for this file
40       4    f32     sparsity_ratio — fraction of weights masked by structured sparsity
44       8    u64     calibration_hash — FNV-64 hash of calibration data used
52     204    bytes   reserved — zero-padded; reserved for future format versions
────    ────  ──────  ────────────────────────────────────────────────────

Total: 256 bytes.  All integer fields are little-endian.

Flag bits (``flags`` field)
────────────────────────────
Bit 0 — ASTC      : ASTC 6×6 hardware-texture compressed weights (Wave 64)
Bit 1 — TCA_TBE   : TCA-TBE lossless bitmap encoding (Wave 65)
Bit 2 — INT4      : INT4-quantised weight blocks
Bit 3 — SPARSE    : structured FFN sparsity masks (Wave 66)
Bit 4 — EAGLE     : trained EAGLE-3 draft head appendix (Wave 68)
Bit 5 — INT2      : INT2 sub-4-bit weight blocks (hybrid path)
Bit 6 — ANE_COREML: ANE CoreML appendix present (Wave 69)
Bit 7 — MXFP4    : OCP MX FP4 per-block weights (Wave 68, M5+)
Bit 8 — INT3      : INT3-MiLo 3-bit + LRC compensator

Architecture enum (``arch_id`` field)
──────────────────────────────────────
0  UNKNOWN
1  LLAMA    — LLaMA / LLaMA-2 / LLaMA-3 family
2  MISTRAL  — Mistral / Mixtral
3  QWEN     — Qwen2.5 / Qwen3 family
4  GEMMA    — Gemma2 / Gemma3
5  DEEPSEEK — DeepSeek-V2 / DeepSeek-R1 distillations
6  PHI      — Phi-3 / Phi-4 family

Usage::

    from squish.format.squish_header import SquizdHeader, SquizdFlag, SquizdArch

    hdr = SquizdHeader(
        flags=SquizdFlag.ASTC | SquizdFlag.INT4,
        num_layers=32,
        arch_id=SquizdArch.QWEN,
        hidden_dim=4096,
        num_heads=32,
        vocab_size=151936,
        compression_bpw=3.56,
    )
    raw = hdr.serialise()           # 256-byte bytes object
    hdr2 = SquizdHeader.from_bytes(raw)
    assert hdr2 == hdr
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Optional, Union

__all__ = [
    "SQUIZD_MAGIC",
    "SQUIZD_VERSION",
    "SQUIZD_HEADER_SIZE",
    "SquizdFlag",
    "SquizdArch",
    "SquizdHeader",
    "build_minimal_header",
    "read_header",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQUIZD_MAGIC: bytes = b"SQZD"
SQUIZD_VERSION: int = 1
SQUIZD_HEADER_SIZE: int = 256

# Struct format for the 256-byte header (little-endian)
# Field summary (left to right):
#   magic(4), version(H), flags(I), num_layers(H), arch_id(H),
#   spare_crc(I), draft_hash(Q), hidden_dim(I), num_heads(H),
#   vocab_size(I), compression_bpw(f), sparsity_ratio(f),
#   calibration_hash(Q), reserved(204s)
# Size check: 4+2+4+2+2+4+8+4+2+4+4+4+8+204 = 256 ✓
_HEADER_STRUCT: struct.Struct = struct.Struct(
    "<4s H I H H I Q I H I f f Q 204s"
)
assert _HEADER_STRUCT.size == SQUIZD_HEADER_SIZE, (
    f"Header struct size mismatch: {_HEADER_STRUCT.size} != {SQUIZD_HEADER_SIZE}"
)

_RESERVED_BYTES: bytes = b"\x00" * 204


# ---------------------------------------------------------------------------
# Flag and arch enumerations
# ---------------------------------------------------------------------------

class SquizdFlag(IntFlag):
    """Compression and feature flags stored in the header ``flags`` field."""

    NONE       = 0
    ASTC       = 1 << 0   # Wave 64 — ASTC 6×6 hardware texture
    TCA_TBE    = 1 << 1   # Wave 65 — TCA-TBE lossless bitmap
    INT4       = 1 << 2   # INT4 quantised blocks
    SPARSE     = 1 << 3   # Wave 66 — structured FFN sparsity
    EAGLE      = 1 << 4   # Wave 68 — EAGLE draft head appendix
    INT2       = 1 << 5   # INT2 sub-4-bit hybrid blocks
    ANE_COREML = 1 << 6   # Wave 69 — ANE CoreML appendix
    MXFP4     = 1 << 7   # Wave 68 — OCP MX FP4 (M5+)
    INT3       = 1 << 8   # INT3-MiLo 3-bit + LRC

    @classmethod
    def from_uint32(cls, value: int) -> "SquizdFlag":
        """Construct from a raw uint32 (unknown bits are silently dropped)."""
        known = int(cls.ASTC | cls.TCA_TBE | cls.INT4 | cls.SPARSE
                    | cls.EAGLE | cls.INT2 | cls.ANE_COREML | cls.MXFP4 | cls.INT3)
        return cls(value & known)

    def has(self, flag: "SquizdFlag") -> bool:
        """Return True if *flag* is set."""
        return bool(self & flag)


class SquizdArch(IntEnum):
    """Model architecture identifiers stored in the header ``arch_id`` field."""

    UNKNOWN  = 0
    LLAMA    = 1
    MISTRAL  = 2
    QWEN     = 3
    GEMMA    = 4
    DEEPSEEK = 5
    PHI      = 6

    @classmethod
    def _missing_(cls, value: object) -> "SquizdArch":
        return cls.UNKNOWN


# ---------------------------------------------------------------------------
# Header dataclass
# ---------------------------------------------------------------------------

@dataclass(eq=True)
class SquizdHeader:
    """SQUIZD binary format header v0.1.

    All fields have sensible defaults so a minimal header can be created with
    just the compression flags and layer count.

    Attributes:
        flags:            Feature flags bitfield (:class:`SquizdFlag`).
        num_layers:       Number of transformer layers.
        arch_id:          Architecture identifier (:class:`SquizdArch`).
        hidden_dim:       Model hidden dimension.
        num_heads:        Number of attention heads.
        vocab_size:       Vocabulary size.
        compression_bpw:  Actual bits per weight (informational).
        sparsity_ratio:   Fraction of weights masked by structured sparsity.
        calibration_hash: FNV-64 hash of calibration data (informational).
        spare_crc:        CRC32 of sparsity metadata block (0 if absent).
        draft_hash:       FNV-64 hash of EAGLE draft-head appendix (0 if absent).
    """

    flags: SquizdFlag = SquizdFlag.NONE
    num_layers: int = 0
    arch_id: SquizdArch = SquizdArch.UNKNOWN
    hidden_dim: int = 0
    num_heads: int = 0
    vocab_size: int = 0
    compression_bpw: float = 0.0
    sparsity_ratio: float = 0.0
    calibration_hash: int = 0
    spare_crc: int = 0
    draft_hash: int = 0

    def __post_init__(self) -> None:
        # Coerce flags to SquizdFlag
        if not isinstance(self.flags, SquizdFlag):
            self.flags = SquizdFlag.from_uint32(int(self.flags))
        # Coerce arch_id to SquizdArch
        if not isinstance(self.arch_id, SquizdArch):
            try:
                self.arch_id = SquizdArch(int(self.arch_id))
            except ValueError:
                self.arch_id = SquizdArch.UNKNOWN

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def serialise(self) -> bytes:
        """Serialise to a 256-byte ``bytes`` object (little-endian)."""
        return _HEADER_STRUCT.pack(
            SQUIZD_MAGIC,
            SQUIZD_VERSION,
            int(self.flags),
            self.num_layers & 0xFFFF,
            int(self.arch_id) & 0xFFFF,
            self.spare_crc & 0xFFFFFFFF,
            self.draft_hash & 0xFFFFFFFFFFFFFFFF,
            self.hidden_dim & 0xFFFFFFFF,
            self.num_heads & 0xFFFF,
            self.vocab_size & 0xFFFFFFFF,
            float(self.compression_bpw),
            float(self.sparsity_ratio),
            self.calibration_hash & 0xFFFFFFFFFFFFFFFF,
            _RESERVED_BYTES,
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SquizdHeader":
        """Parse a :class:`SquizdHeader` from *data* (must be ≥ 256 bytes).

        Raises
        ------
        ValueError
            If *data* is too short, the magic bytes are wrong, or the format
            version is higher than :data:`SQUIZD_VERSION`.
        """
        if len(data) < SQUIZD_HEADER_SIZE:
            raise ValueError(
                f"Data too short for SQUIZD header: "
                f"{len(data)} < {SQUIZD_HEADER_SIZE}"
            )
        raw = data[:SQUIZD_HEADER_SIZE]
        (
            magic,
            version,
            flags_raw,
            num_layers,
            arch_id_raw,
            spare_crc,
            draft_hash,
            hidden_dim,
            num_heads,
            vocab_size,
            bpw,
            sparsity,
            cal_hash,
            _reserved,
        ) = _HEADER_STRUCT.unpack(raw)

        if magic != SQUIZD_MAGIC:
            raise ValueError(
                f"Invalid SQUIZD magic: expected {SQUIZD_MAGIC!r}, got {magic!r}"
            )
        if version > SQUIZD_VERSION:
            raise ValueError(
                f"Unsupported SQUIZD format version {version}; "
                f"this build supports up to {SQUIZD_VERSION}"
            )

        return cls(
            flags=SquizdFlag.from_uint32(flags_raw),
            num_layers=int(num_layers),
            arch_id=SquizdArch(arch_id_raw),
            hidden_dim=int(hidden_dim),
            num_heads=int(num_heads),
            vocab_size=int(vocab_size),
            compression_bpw=float(bpw),
            sparsity_ratio=float(sparsity),
            calibration_hash=int(cal_hash),
            spare_crc=int(spare_crc),
            draft_hash=int(draft_hash),
        )

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "SquizdHeader":
        """Read the header from a ``.squizd`` file at *path*.

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        ValueError
            If the header is invalid.
        """
        p = Path(path)
        data = p.read_bytes()
        return cls.from_bytes(data)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def is_valid(self) -> bool:
        """Return True if the header flags and dimensions are self-consistent."""
        if self.num_layers == 0:
            return False
        if self.flags.has(SquizdFlag.ASTC) and self.compression_bpw <= 0:
            return False
        return True

    def summary(self) -> str:
        """Return a human-readable single-line summary of the header."""
        flag_names = [f.name for f in SquizdFlag if f != SquizdFlag.NONE and self.flags.has(f)]
        flags_str = "+".join(flag_names) if flag_names else "NONE"
        return (
            f"SQUIZD v{SQUIZD_VERSION} | {self.arch_id.name} "
            f"| layers={self.num_layers} | flags={flags_str} "
            f"| {self.compression_bpw:.2f} bpw"
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict representation."""
        return {
            "magic": SQUIZD_MAGIC.decode("ascii"),
            "version": SQUIZD_VERSION,
            "flags": int(self.flags),
            "flag_names": [f.name for f in SquizdFlag if f and self.flags.has(f)],
            "num_layers": self.num_layers,
            "arch_id": int(self.arch_id),
            "arch_name": self.arch_id.name,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "vocab_size": self.vocab_size,
            "compression_bpw": float(self.compression_bpw),
            "sparsity_ratio": float(self.sparsity_ratio),
            "calibration_hash": self.calibration_hash,
            "spare_crc": self.spare_crc,
            "draft_hash": self.draft_hash,
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def build_minimal_header(
    flags: Union[SquizdFlag, int] = SquizdFlag.NONE,
    *,
    num_layers: int = 32,
    arch_id: Union[SquizdArch, int] = SquizdArch.UNKNOWN,
) -> bytes:
    """Build a minimal 256-byte header blob for testing and bootstrapping.

    Parameters
    ----------
    flags:
        Feature flags.
    num_layers:
        Number of transformer layers.
    arch_id:
        Architecture identifier.

    Returns
    -------
    bytes
        256-byte serialised header.
    """
    hdr = SquizdHeader(
        flags=SquizdFlag.from_uint32(int(flags)),
        num_layers=num_layers,
        arch_id=SquizdArch(int(arch_id)) if not isinstance(arch_id, SquizdArch) else arch_id,
    )
    return hdr.serialise()


def read_header(path: Union[str, Path]) -> Optional[SquizdHeader]:
    """Read the SQUIZD header from *path*, returning None on any error.

    Unlike :meth:`SquizdHeader.from_file` this function never raises; it
    returns None if the file is missing, too short, or has invalid magic.
    """
    try:
        return SquizdHeader.from_file(path)
    except (FileNotFoundError, ValueError, struct.error):
        return None
