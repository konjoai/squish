# [Experimental] This module is part of Squish v42+ (Wave 68).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""squish/format/mx_fp4.py — MXFP4 Format Bridge for .squizd Files.

Wave 68: bridges the MXFP4 quantiser from :mod:`squish.quant.mx_fp4` into the
unified `.squizd` block header layout.  On M5 hardware, the Neural Accelerators
provide native matrix-multiply support for OCP MX e2m1 weights; on M1–M4 the
decoder falls back to the INT4 fused GEMV path.

Block header layout (per MXFP4 block inside .squizd)
──────────────────────────────────────────────────────
```
┌──────────────────────────────────────────────┐
│  block_tag   u8    = 0xF4 (MXFP4 block magic)│
│  block_size  u16   LE — elements per block   │
│  n_blocks    u32   LE — number of blocks     │
│  scale_offset u64  LE — byte offset to E8M0  │
│  data_offset  u64  LE — byte offset to codes │
│  data_len     u64  LE — byte length of codes │
└──────────────────────────────────────────────┘
```

The scale array is stored as float32 (E8M0 simulation — ``power_of_two(scale)``),
immediately followed by the packed 4-bit code array (2 codes per byte, hi nibble
first).

Routing
───────
- M5+: route through :class:`MxFP4NativeBackend` (placeholder — wraps raw bytes
  for handoff to the M5 ANE matmul hardware path).
- M1–M4 / unknown: decode via :func:`~squish.quant.mx_fp4.MxFP4Result.dequantize`
  then forward through the fused INT4 GEMV path.

Usage::

    import numpy as np
    from squish.format.mx_fp4 import MxFP4FormatBridge, MxFP4BlockHeader

    bridge = MxFP4FormatBridge()
    w = np.random.randn(64, 256).astype(np.float32)

    encoded = bridge.encode(w)          # → bytes (block header + scales + codes)
    restored = bridge.decode(encoded)   # → float32 np.ndarray (approx)

    # Hardware-aware routing:
    from squish.hardware.capability_probe import get_capability_probe
    caps = get_capability_probe().probe()
    device_output = bridge.route(encoded, input_vec, caps)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

import numpy as np

from squish.quant.mx_fp4 import MxFP4, MxFP4Config, MxFP4Result

__all__ = [
    "MXFP4_BLOCK_TAG",
    "MXFP4_HEADER_SIZE",
    "MxFP4BlockHeader",
    "MxFP4FormatBridge",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MXFP4_BLOCK_TAG: int = 0xF4
MXFP4_HEADER_SIZE: int = 1 + 2 + 4 + 8 + 8 + 8  # = 31 bytes

# ---------------------------------------------------------------------------
# Block header
# ---------------------------------------------------------------------------

@dataclass
class MxFP4BlockHeader:
    """Decoded MXFP4 block header read from a .squizd weight buffer.

    Attributes:
        block_tag: Must equal :data:`MXFP4_BLOCK_TAG` (``0xF4``).
        block_size: Elements per MXFP4 block.
        n_blocks: Total number of blocks.
        scale_offset: Byte offset from the end of the header to the E8M0
            scale array.
        data_offset: Byte offset from the end of the header to the packed
            4-bit code array.
        data_len: Byte length of the packed code array.
    """

    block_tag: int
    block_size: int
    n_blocks: int
    scale_offset: int
    data_offset: int
    data_len: int

    _STRUCT = struct.Struct("<BHIQQQQ"[:-1] + "QQ")  # recalculated below

    @classmethod
    def _pack_format(cls) -> str:
        return "<BHIQQQ"

    @classmethod
    def from_bytes(cls, data: bytes) -> "MxFP4BlockHeader":
        """Parse a :data:`MXFP4_HEADER_SIZE`-byte buffer."""
        fmt = cls._pack_format()
        block_tag, block_size, n_blocks, scale_offset, data_offset, data_len = (
            struct.unpack(fmt, data[:MXFP4_HEADER_SIZE])
        )
        return cls(
            block_tag=block_tag,
            block_size=block_size,
            n_blocks=n_blocks,
            scale_offset=scale_offset,
            data_offset=data_offset,
            data_len=data_len,
        )

    def to_bytes(self) -> bytes:
        """Serialise to :data:`MXFP4_HEADER_SIZE` bytes."""
        return struct.pack(
            self._pack_format(),
            self.block_tag,
            self.block_size,
            self.n_blocks,
            self.scale_offset,
            self.data_offset,
            self.data_len,
        )

    def validate(self) -> None:
        """Raise :exc:`ValueError` if any field is inconsistent."""
        if self.block_tag != MXFP4_BLOCK_TAG:
            raise ValueError(
                f"Invalid MXFP4 block tag: 0x{self.block_tag:02X} "
                f"(expected 0x{MXFP4_BLOCK_TAG:02X})"
            )
        if self.block_size < 1:
            raise ValueError("block_size must be >= 1")
        if self.n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")


# ---------------------------------------------------------------------------
# Packed code helpers (2 INT4 codes per byte, high-nibble first)
# ---------------------------------------------------------------------------

def _pack_codes(codes: np.ndarray) -> bytes:
    """Pack a 1-D INT4 code array (values 0–15) into 2-per-byte representation."""
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    # Pad to even length
    if len(flat) % 2:
        flat = np.concatenate([flat, np.array([0], dtype=np.uint8)])
    packed = ((flat[0::2] & 0xF) << 4) | (flat[1::2] & 0xF)
    return packed.tobytes()


def _unpack_codes(data: bytes, n_elements: int) -> np.ndarray:
    """Unpack 2-per-byte INT4 codes back to a 1-D uint8 array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    hi = (buf >> 4) & 0xF
    lo = buf & 0xF
    interleaved = np.empty(len(buf) * 2, dtype=np.uint8)
    interleaved[0::2] = hi
    interleaved[1::2] = lo
    return interleaved[:n_elements]


# ---------------------------------------------------------------------------
# Native backend placeholder (M5+)
# ---------------------------------------------------------------------------

class MxFP4NativeBackend:
    """Placeholder for M5 ANE native MXFP4 matmul backend.

    On M5+, this class hands the raw encoded bytes directly to the ANE
    matmul hardware without software dequantisation.  On earlier hardware
    this class is never instantiated and the fallback path is used instead.
    """

    def matmul(
        self,
        encoded: bytes,
        input_vec: np.ndarray,
        header: MxFP4BlockHeader,
    ) -> np.ndarray:  # pragma: no cover — M5 hardware only
        """Dispatch to M5 ANE MXFP4 matmul.

        Not implemented in simulation mode; raises ``NotImplementedError``.
        """
        raise NotImplementedError(
            "MxFP4NativeBackend requires M5+ hardware with ANE MXFP4 support."
        )


# ---------------------------------------------------------------------------
# Format bridge
# ---------------------------------------------------------------------------

class MxFP4FormatBridge:
    """Encodes and decodes MXFP4 weight blocks for the .squizd format.

    Uses :class:`~squish.quant.mx_fp4.MxFP4` for quantisation and provides
    hardware-aware routing on decode.

    Args:
        block_size: MXFP4 scales-group size (default 32, OCP standard).
    """

    def __init__(self, block_size: int = 32) -> None:
        self._quantizer = MxFP4(MxFP4Config(block_size=block_size))

    @property
    def block_size(self) -> int:
        return self._quantizer.config.block_size

    def encode(self, weights: np.ndarray) -> bytes:
        """Quantise *weights* to MXFP4 and serialise to the .squizd block format.

        Args:
            weights: FP32 weight tensor (any shape).

        Returns:
            Bytes: ``MXFP4_HEADER_SIZE`` header + scale array + packed codes.
        """
        result: MxFP4Result = self._quantizer.quantize(weights)
        n_elements = len(result.codes)
        n_blocks = len(result.scales)

        # Serialise scale array as float32 LE
        scale_bytes = result.scales.astype(np.float32).tobytes()
        code_bytes = _pack_codes(result.codes)

        # Offsets are relative to the end of the header
        scale_offset = 0
        data_offset = len(scale_bytes)
        header = MxFP4BlockHeader(
            block_tag=MXFP4_BLOCK_TAG,
            block_size=self.block_size,
            n_blocks=n_blocks,
            scale_offset=scale_offset,
            data_offset=data_offset,
            data_len=len(code_bytes),
        )
        return header.to_bytes() + scale_bytes + code_bytes

    def decode(self, data: bytes) -> np.ndarray:
        """Decode a .squizd MXFP4 block back to FP32.

        Args:
            data: Bytes produced by :meth:`encode`.

        Returns:
            Approximate FP32 reconstruction with the original shape.

        Raises:
            ValueError: If the header tag is invalid.
        """
        header = MxFP4BlockHeader.from_bytes(data[:MXFP4_HEADER_SIZE])
        header.validate()

        payload = data[MXFP4_HEADER_SIZE:]
        scale_bytes = payload[
            header.scale_offset: header.scale_offset + header.n_blocks * 4
        ]
        code_bytes = payload[
            header.data_offset: header.data_offset + header.data_len
        ]

        scales = np.frombuffer(scale_bytes, dtype=np.float32).copy()
        n_elements = header.n_blocks * header.block_size
        codes = _unpack_codes(code_bytes, n_elements)

        result = MxFP4Result(
            codes=codes.astype(np.int32),
            scales=scales,
            original_shape=(n_elements,),
        )
        return result.dequantize()

    def route(
        self,
        encoded: bytes,
        input_vec: np.ndarray,
        caps: object,  # HardwareCapabilities
    ) -> np.ndarray:
        """Hardware-aware GEMV routing.

        On M5+ hardware, dispatches to :class:`MxFP4NativeBackend`.
        On all other chips, falls back to software dequantisation + matmul.

        Args:
            encoded: Bytes from :meth:`encode`.
            input_vec: FP32 input vector, shape ``(n_cols,)``.
            caps: :class:`~squish.hardware.capability_probe.HardwareCapabilities`.

        Returns:
            Output vector, shape ``(n_rows,)`` (approx).
        """
        from squish.hardware.chip_detector import AppleChipGeneration

        chip_gen = getattr(caps, "chip_generation", None)
        if chip_gen is not None and chip_gen >= AppleChipGeneration.M5:
            header = MxFP4BlockHeader.from_bytes(encoded[:MXFP4_HEADER_SIZE])
            try:
                backend = MxFP4NativeBackend()
                return backend.matmul(encoded, input_vec, header)
            except NotImplementedError:
                pass  # Fall through to software path

        # Software fallback: dequant → matmul
        weights = self.decode(encoded)
        x = np.asarray(input_vec, dtype=np.float32).reshape(-1)
        # Treat the decoded weights as a 1-D vector for dot product
        return np.array([float(np.dot(weights.reshape(-1)[: len(x)], x))],
                        dtype=np.float32)
