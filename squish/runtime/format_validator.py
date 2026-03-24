# [Experimental] This module is part of Squish v44+ (Wave 70).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""SquizdFormatValidator — validates ``.squizd`` files before loading.

Checks the magic bytes, format version, layer count, sparsity metadata CRC,
and draft-head hash (when present).  All violations are reported via a
:class:`ValidationResult` rather than failing fast, so callers can surface
every issue to users at once.

Usage::

    from squish.runtime.format_validator import SquizdFormatValidator

    result = SquizdFormatValidator().validate("model.squizd")
    if not result.valid:
        print(result.errors)
"""

from __future__ import annotations

import binascii
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

__all__ = [
    "SQUIZD_MAGIC",
    "SQUIZD_CURRENT_VERSION",
    "SQUIZD_MIN_VERSION",
    "SQUIZD_HEADER_SIZE",
    "SquizdFormatError",
    "ValidationResult",
    "SquizdFormatValidator",
]

# ---------------------------------------------------------------------------
# Constants — keep in sync with squish_runtime.py
# ---------------------------------------------------------------------------

SQUIZD_MAGIC: bytes = b"SQZD"
SQUIZD_CURRENT_VERSION: int = 1
SQUIZD_MIN_VERSION: int = 1
SQUIZD_HEADER_SIZE: int = 256

# Byte offsets within the 256-byte header
_OFF_MAGIC      = 0
_OFF_VERSION    = 4       # uint16 LE — format version
_OFF_FLAGS      = 6       # uint32 LE — feature flags bitfield
_OFF_LAYERS     = 10      # uint16 LE — number of transformer layers
_OFF_ARCH_ID    = 12      # uint16 LE — architecture identifier
_OFF_SPARE_CRC  = 14      # uint32 LE — CRC32 of sparsity metadata block
_OFF_DRAFT_HASH = 18      # uint64 LE — FNV-64-like hash of draft-head appendix
_OFF_RESERVED   = 26      # remaining bytes reserved; must be zero for v1

# Flag bit that signals EAGLE draft head presence
_FLAG_EAGLE  = 1 << 4
_FLAG_SPARSE = 1 << 3


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class SquizdFormatError(ValueError):
    """Raised when a SQUIZD file fails validation and cannot be used safely.

    Attributes:
        path: Source file path if known.
        errors: List of human-readable error strings.
    """

    def __init__(self, message: str, *, errors: Optional[List[str]] = None, path: Optional[Path] = None) -> None:
        super().__init__(message)
        self.errors: List[str] = errors or [message]
        self.path: Optional[Path] = path

    def __str__(self) -> str:  # noqa: D105
        if len(self.errors) == 1:
            return self.errors[0]
        return "\n".join(f"  [{i + 1}] {e}" for i, e in enumerate(self.errors))


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Collected result of a SQUIZD file validation run.

    Attributes:
        valid: ``True`` if *all* mandatory checks passed.
        magic_ok: Magic bytes ``b"SQZD"`` matched.
        version_ok: Header version is within ``[SQUIZD_MIN_VERSION, SQUIZD_CURRENT_VERSION]``.
        layer_count_ok: Layer count is non-zero and ≤ 512.
        sparsity_crc_ok: Sparsity-metadata CRC32 matches stored value
            (only meaningful when the SPARSE flag is set).
        eagle_hash_ok: Draft-head hash matches stored value
            (only meaningful when the EAGLE flag is set).
        errors: Non-empty list of human-readable error strings when not valid.
    """

    valid: bool
    magic_ok: bool
    version_ok: bool
    layer_count_ok: bool
    sparsity_crc_ok: bool
    eagle_hash_ok: bool
    errors: List[str] = field(default_factory=list)

    @classmethod
    def success(cls) -> "ValidationResult":
        """Return a fully-passing :class:`ValidationResult`."""
        return cls(
            valid=True,
            magic_ok=True,
            version_ok=True,
            layer_count_ok=True,
            sparsity_crc_ok=True,
            eagle_hash_ok=True,
        )

    def to_dict(self) -> dict:
        """Return a plain-dict representation for serialisation."""
        return {
            "valid": self.valid,
            "magic_ok": self.magic_ok,
            "version_ok": self.version_ok,
            "layer_count_ok": self.layer_count_ok,
            "sparsity_crc_ok": self.sparsity_crc_ok,
            "eagle_hash_ok": self.eagle_hash_ok,
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class SquizdFormatValidator:
    """Validates ``.squizd`` files or raw header bytes.

    All validation is non-destructive; the file is read but not modified.

    Parameters:
        strict: If ``True``, treat reserved-bytes being non-zero as an error
            (disabled by default to allow future minor extensions).
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, path: Union[str, Path]) -> ValidationResult:
        """Validate *path* on disk.

        Args:
            path: Path to the ``.squizd`` file.

        Returns:
            :class:`ValidationResult`.  Never raises; all errors are
            collected into the ``errors`` field.
        """
        p = Path(path)
        if not p.exists():
            return self._fail(
                magic_ok=False,
                version_ok=False,
                layer_count_ok=False,
                sparsity_crc_ok=False,
                eagle_hash_ok=False,
                errors=[f"File not found: {p}"],
            )
        data = p.read_bytes()
        return self.validate_bytes(data, _source=str(p))

    def validate_bytes(
        self,
        data: bytes,
        *,
        _source: str = "<bytes>",
    ) -> ValidationResult:
        """Validate a raw bytes object that represents a SQUIZD file.

        Args:
            data: Raw SQUIZD file bytes (only the header is inspected).
            _source: Informational source label used in error messages.

        Returns:
            :class:`ValidationResult`.
        """
        errors: List[str] = []
        flags = 0
        version = 0
        layer_count = 0
        stored_spare_crc = 0
        stored_draft_hash = 0

        # ---- Magic ----------------------------------------------------------
        magic_ok = data[:4] == SQUIZD_MAGIC
        if not magic_ok:
            got = data[:4] if len(data) >= 4 else data
            errors.append(
                f"{_source}: bad magic bytes {got!r}; expected {SQUIZD_MAGIC!r}"
            )

        # Stop early if the buffer is too short to parse further.
        if len(data) < SQUIZD_HEADER_SIZE:
            errors.append(
                f"{_source}: header too short ({len(data)} bytes; need {SQUIZD_HEADER_SIZE})"
            )
            return self._fail(
                magic_ok=magic_ok,
                version_ok=False,
                layer_count_ok=False,
                sparsity_crc_ok=False,
                eagle_hash_ok=False,
                errors=errors,
            )

        # ---- Unpack fixed fields -------------------------------------------
        try:
            version, flags, layer_count, arch_id = struct.unpack_from(
                "<HI HH", data, _OFF_VERSION
            )
            stored_spare_crc, = struct.unpack_from("<I", data, _OFF_SPARE_CRC)
            stored_draft_hash, = struct.unpack_from("<Q", data, _OFF_DRAFT_HASH)
        except struct.error as exc:
            errors.append(f"{_source}: header unpack failed: {exc}")
            return self._fail(
                magic_ok=magic_ok, version_ok=False,
                layer_count_ok=False, sparsity_crc_ok=False,
                eagle_hash_ok=False, errors=errors,
            )

        # ---- Version --------------------------------------------------------
        version_ok = SQUIZD_MIN_VERSION <= version <= SQUIZD_CURRENT_VERSION
        if not version_ok:
            errors.append(
                f"{_source}: unsupported version {version}; "
                f"supported range [{SQUIZD_MIN_VERSION}, {SQUIZD_CURRENT_VERSION}]"
            )

        # ---- Layer count ----------------------------------------------------
        MAX_LAYERS = 512
        layer_count_ok = 1 <= layer_count <= MAX_LAYERS
        if not layer_count_ok:
            errors.append(
                f"{_source}: invalid layer_count {layer_count}; must be 1–{MAX_LAYERS}"
            )

        # ---- Sparsity CRC ---------------------------------------------------
        sparsity_crc_ok = True
        if flags & _FLAG_SPARSE:
            sparsity_crc_ok = self._check_sparsity_crc(data, stored_spare_crc, _source, errors)

        # ---- Draft-head hash ------------------------------------------------
        eagle_hash_ok = True
        if flags & _FLAG_EAGLE:
            eagle_hash_ok = self._check_eagle_hash(data, stored_draft_hash, _source, errors)

        # ---- Reserved bytes -------------------------------------------------
        if self.strict:
            reserved = data[_OFF_RESERVED:SQUIZD_HEADER_SIZE]
            if any(b != 0 for b in reserved):
                errors.append(
                    f"{_source}: strict mode: reserved header bytes ({_OFF_RESERVED}–{SQUIZD_HEADER_SIZE - 1}) must be zero"
                )

        valid = magic_ok and version_ok and layer_count_ok and sparsity_crc_ok and eagle_hash_ok and not (
            self.strict and bool(errors)
        )
        # Re-check validity: if we accumulated new strict errors, mark invalid.
        if errors and not any(
            not x for x in [magic_ok, version_ok, layer_count_ok, sparsity_crc_ok, eagle_hash_ok]
        ):
            if self.strict and len(errors) > 0:
                valid = False

        return ValidationResult(
            valid=valid and len(errors) == 0,
            magic_ok=magic_ok,
            version_ok=version_ok,
            layer_count_ok=layer_count_ok,
            sparsity_crc_ok=sparsity_crc_ok,
            eagle_hash_ok=eagle_hash_ok,
            errors=errors,
        )

    def assert_valid(self, path: Union[str, Path]) -> None:
        """Validate *path* and raise :class:`SquizdFormatError` on failure.

        Args:
            path: Path to the ``.squizd`` file.

        Raises:
            SquizdFormatError: If the file fails any validation check.
        """
        result = self.validate(path)
        if not result.valid:
            raise SquizdFormatError(
                f"SQUIZD validation failed for {path}",
                errors=result.errors,
                path=Path(path),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sparsity_crc(
        data: bytes, stored: int, source: str, errors: List[str]
    ) -> bool:
        """Verify CRC32 of the sparsity metadata block.

        The sparsity metadata is defined as the 32 bytes starting at offset
        128 in the header.  If the block is absent (all zeros) and the
        stored CRC is 0, the check passes (the file was written before
        sparsity data was populated).
        """
        block = data[128:160]
        computed = binascii.crc32(block) & 0xFFFFFFFF
        # Treat both-zero as not-yet-populated: pass.
        if stored == 0 and computed == binascii.crc32(b"\x00" * 32) & 0xFFFFFFFF:
            return True
        if computed != stored:
            errors.append(
                f"{source}: sparsity CRC32 mismatch "
                f"(stored=0x{stored:08x}, computed=0x{computed:08x})"
            )
            return False
        return True

    @staticmethod
    def _check_eagle_hash(
        data: bytes, stored: int, source: str, errors: List[str]
    ) -> bool:
        """Verify the FNV-1a-64 hash of the draft-head block.

        The draft-head metadata is the 32 bytes starting at offset 160.
        A stored hash of zero means "not set", which passes validation.
        """
        if stored == 0:
            return True
        block = data[160:192]
        computed = _fnv1a_64(block)
        if computed != stored:
            errors.append(
                f"{source}: draft-head hash mismatch "
                f"(stored=0x{stored:016x}, computed=0x{computed:016x})"
            )
            return False
        return True

    @staticmethod
    def _fail(
        *,
        magic_ok: bool,
        version_ok: bool,
        layer_count_ok: bool,
        sparsity_crc_ok: bool,
        eagle_hash_ok: bool,
        errors: List[str],
    ) -> ValidationResult:
        return ValidationResult(
            valid=False,
            magic_ok=magic_ok,
            version_ok=version_ok,
            layer_count_ok=layer_count_ok,
            sparsity_crc_ok=sparsity_crc_ok,
            eagle_hash_ok=eagle_hash_ok,
            errors=errors,
        )


# ---------------------------------------------------------------------------
# Helper: FNV-1a 64-bit hash
# ---------------------------------------------------------------------------

def _fnv1a_64(data: bytes) -> int:
    """Compute FNV-1a 64-bit hash of *data*."""
    FNV_PRIME  = 0x00000100000001B3
    FNV_OFFSET = 0xCBF29CE484222325
    h = FNV_OFFSET
    for byte in data:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h
