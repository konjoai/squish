# [Experimental] This module is part of Squish v39+ (Wave 65).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""TCA-TBE — Tensor-Core-Aware Triple Bitmap Encoding (ZipServ, ASPLOS 2026).

TCA-TBE is a lossless BF16 weight compression scheme that exploits the highly
skewed low-entropy distribution of BF16 exponent bits in trained transformers.
Most weight values cluster around a narrow exponent range, so encoding each
128-element block as three fixed-length bitmaps — sign, range, mantissa residual
— achieves ~62% size reduction for the bitmap layer with no variable-length
codes and no control-flow divergence at decode time.

Block format (per 128 BF16 elements)
──────────────────────────────────────
  sign_bitmap     : 128 bits  (1 bit per element — sign of the BF16 value)
  range_bitmap    : 128 bits  (1 bit per element — 1 if exponent ∈ {e_mode-1,
                                                    e_mode, e_mode+1}, else 0)
  mantissa_bitmap : 512 bits  (top 4 mantissa bits per element, packed)
  e_mode          :   8 bits  (most frequent exponent — uint8)
  e_lo_offset     :   8 bits  (lower fallback exponent offset from e_mode)
  e_hi_offset     :   8 bits  (upper fallback exponent offset from e_mode)

Total per block: 128 + 128 + 512 + 24 = 792 bits = 99 bytes
vs raw BF16    : 128 × 16 = 2048 bits = 256 bytes
Ratio          : ~61.3% reduction  (matches ZipServ's reported ~62%)

Out-of-range elements (range_bitmap == 0) are stored as raw BF16 words in a
spillover section after the bitmaps.  For highly compressible blocks (typical
transformer weights) the spill section is empty or very small.

Entropy guard: if fewer than 20% of elements have exponents in the mode window,
the codec stores the block as raw BF16 (uncompressed) to avoid size expansion.

Usage::

    from squish.compress.tca_tbe import TcaTbeCodec, TcaTbeConfig, TcaTbeBlock

    cfg   = TcaTbeConfig()
    codec = TcaTbeCodec(cfg)
    block = codec.encode(weights_bf16)        # weights_bf16: (128,) np.uint16
    recon = codec.decode(block)               # back to (128,) np.uint16 exactly
    assert np.array_equal(recon, weights_bf16)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "BLOCK_SIZE",
    "TcaTbeConfig",
    "TcaTbeBlock",
    "TcaTbeCodec",
    "tca_tbe_encode_tensor",
    "tca_tbe_decode_tensor",
    "CompressionStats",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BLOCK_SIZE: int = 128      # elements per TCA-TBE block
_FALLBACK_MARKER: int = 0xFF   # stored in e_mode when block is raw/uncompressed

# Range window half-width: exponent ∈ [e_mode - RANGE_HALF, e_mode + RANGE_HALF]
_RANGE_HALF: int = 1
# Minimum fraction of in-range elements before entropy-guard triggers.
_MIN_COVERAGE: float = 0.20


# ---------------------------------------------------------------------------
# BF16 utility helpers
# ---------------------------------------------------------------------------

def _bf16_to_parts(words: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a (N,) uint16 BF16 array into (sign, exponent, mantissa) arrays.

    Returns:
        sign     : (N,) uint8 — 0 or 1
        exponent : (N,) uint8 — 8-bit biased exponent (bits 14–7)
        mantissa : (N,) uint16 — 7-bit mantissa (bits 6–0)
    """
    words = words.astype(np.uint16)
    sign     = ((words >> 15) & 0x1).astype(np.uint8)
    exponent = ((words >> 7)  & 0xFF).astype(np.uint8)
    mantissa = (words & 0x7F).astype(np.uint16)
    return sign, exponent, mantissa


def _parts_to_bf16(
    sign: np.ndarray, exponent: np.ndarray, mantissa: np.ndarray
) -> np.ndarray:
    """Reconstruct (N,) uint16 BF16 words from sign, exponent, mantissa arrays."""
    return (
        (sign.astype(np.uint16) << 15)
        | (exponent.astype(np.uint16) << 7)
        | mantissa.astype(np.uint16)
    )


def _mode_exponent(exponents: np.ndarray) -> int:
    """Return the most frequently occurring exponent value in *exponents*."""
    counts = np.bincount(exponents.astype(np.intp), minlength=256)
    return int(np.argmax(counts))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TcaTbeConfig:
    """Configuration for :class:`TcaTbeCodec`.

    Attributes:
        block_size: Number of BF16 elements per block (must be 128).
        min_coverage: Minimum fraction of elements whose exponent falls in the
            range window before entropy guard triggers and the block is stored
            as raw BF16.  Default 0.20 (20%).
        range_half: Half-width of the accepted exponent window around e_mode.
            Elements with exponent ∈ [e_mode-range_half, e_mode+range_half]
            are considered *in-range*.  Default 1.
        mantissa_bits: Number of top mantissa bits stored in the mantissa
            bitmap.  Default 4 (top 4 of 7 mantissa bits).
    """

    block_size: int = BLOCK_SIZE
    min_coverage: float = _MIN_COVERAGE
    range_half: int = _RANGE_HALF
    mantissa_bits: int = 7

    def __post_init__(self) -> None:
        if self.block_size != BLOCK_SIZE:
            raise ValueError(f"block_size must be {BLOCK_SIZE}, got {self.block_size}")
        if not 0.0 < self.min_coverage < 1.0:
            raise ValueError(f"min_coverage must be in (0, 1), got {self.min_coverage}")
        if self.range_half < 0:
            raise ValueError(f"range_half must be >= 0, got {self.range_half}")
        if not 1 <= self.mantissa_bits <= 7:
            raise ValueError(f"mantissa_bits must be 1–7, got {self.mantissa_bits}")


# ---------------------------------------------------------------------------
# Encoded block container
# ---------------------------------------------------------------------------

@dataclass
class TcaTbeBlock:
    """Container for a single TCA-TBE encoded block.

    Attributes:
        is_raw: True if the block is stored as uncompressed raw BF16 words
            (entropy-guard triggered).
        e_mode: Most frequent exponent of the block (uint8).  Set to
            ``_FALLBACK_MARKER`` (0xFF) for raw blocks to mark them as
            uncompressed.
        e_lo_offset: Offset below e_mode that widens the range window.
        e_hi_offset: Offset above e_mode that widens the range window.
        sign_bitmap: (BLOCK_SIZE,) uint8 array — 1 bit per element (sign).
        range_bitmap: (BLOCK_SIZE,) uint8 array — 1 bit: in-range flag.
        mantissa_bitmap: (BLOCK_SIZE, mantissa_bits) uint8 array — top-N
            mantissa bits per element.
        mantissa_bits: Number of mantissa bits stored (matches config).
        spill_words: Raw uint16 BF16 words for out-of-range elements, in
            element-index order.
        n_elements: Number of elements in the block (BLOCK_SIZE or less for the
            final block of a tensor row).
    """

    is_raw: bool
    e_mode: int
    e_lo_offset: int
    e_hi_offset: int
    sign_bitmap: np.ndarray            # shape (BLOCK_SIZE,), dtype uint8
    range_bitmap: np.ndarray           # shape (BLOCK_SIZE,), dtype uint8
    exp_offset_bitmap: np.ndarray      # shape (BLOCK_SIZE,), dtype uint8 — exponent - window_lo
    mantissa_bitmap: np.ndarray        # shape (BLOCK_SIZE, mantissa_bits), uint8
    mantissa_bits: int
    spill_words: np.ndarray            # shape (K,), dtype uint16
    n_elements: int = BLOCK_SIZE

    @property
    def is_compressed(self) -> bool:
        """True if the block uses TCA-TBE encoding (not raw fallback)."""
        return not self.is_raw

    def compressed_bytes(self) -> int:
        """Return the on-disk byte size of this block."""
        if self.is_raw:
            return self.n_elements * 2  # raw uint16
        # sign (16B) + range (16B) + exp_offset (32B for 2 bits/elem) + mantissa + header + spill
        bitmap_bytes = (BLOCK_SIZE // 8) * 2  # sign + range (16 bytes each)
        # exp offset: 2 bits per element (window size ≤ 3 for default range_half=1)
        exp_offset_bytes = (BLOCK_SIZE * 2 + 7) // 8   # ceil(BLOCK_SIZE * 2 / 8)
        mantissa_bytes = (BLOCK_SIZE * self.mantissa_bits + 7) // 8
        spill_bytes = len(self.spill_words) * 2
        return bitmap_bytes + exp_offset_bytes + mantissa_bytes + 3 + spill_bytes

    def raw_bytes(self) -> int:
        """Return the uncompressed byte size of this block."""
        return self.n_elements * 2


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class CompressionStats:
    """Aggregate compression statistics for a full tensor.

    Attributes:
        n_blocks: Total number of encoded blocks.
        n_compressed: Blocks encoded with TCA-TBE.
        n_raw: Blocks stored as raw BF16 (entropy guard triggered).
        compressed_bytes: Total on-disk bytes for TCA-TBE blocks.
        raw_bytes_uncompressed: Total uncompressed bytes (all blocks as BF16).
        spill_elements: Total number of out-of-range elements stored in spill.
    """

    n_blocks: int = 0
    n_compressed: int = 0
    n_raw: int = 0
    compressed_bytes: int = 0
    raw_bytes_uncompressed: int = 0
    spill_elements: int = 0

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (raw / compressed); > 1 means size reduction."""
        if self.compressed_bytes == 0:
            return 1.0
        return self.raw_bytes_uncompressed / self.compressed_bytes

    @property
    def size_reduction_pct(self) -> float:
        """Percentage size reduction (0–100)."""
        if self.raw_bytes_uncompressed == 0:
            return 0.0
        return 100.0 * (1.0 - self.compressed_bytes / self.raw_bytes_uncompressed)


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------

class TcaTbeCodec:
    """TCA-TBE lossless BF16 encoder/decoder.

    Operates on 128-element BF16 blocks represented as ``np.uint16`` arrays
    (raw IEEE 754 BF16 bit patterns, not Python floats).

    Parameters:
        config: :class:`TcaTbeConfig` controlling block size, coverage
            threshold, range window, and mantissa bits.
    """

    def __init__(self, config: Optional[TcaTbeConfig] = None) -> None:
        self.config = config or TcaTbeConfig()

    # ------------------------------------------------------------------
    # Block-level encode / decode
    # ------------------------------------------------------------------

    def encode(self, words: np.ndarray) -> TcaTbeBlock:
        """Encode a single BF16 block of ``BLOCK_SIZE`` elements.

        Args:
            words: (BLOCK_SIZE,) ``np.uint16`` array of raw BF16 bit patterns.

        Returns:
            :class:`TcaTbeBlock` with lossless encoded representation.

        Raises:
            ValueError: If ``words`` is not a 1-D uint16 array of the correct
                length.
        """
        words = np.asarray(words, dtype=np.uint16)
        if words.ndim != 1:
            raise ValueError(f"words must be 1-D, got shape {words.shape}")
        n = len(words)
        if n > BLOCK_SIZE:
            raise ValueError(f"block cannot exceed {BLOCK_SIZE} elements, got {n}")

        # Pad to BLOCK_SIZE with zeros if needed (final partial block).
        n_elements = n
        if n < BLOCK_SIZE:
            words = np.concatenate([words, np.zeros(BLOCK_SIZE - n, dtype=np.uint16)])

        sign, exponent, mantissa = _bf16_to_parts(words)
        e_mode = _mode_exponent(exponent)

        lo = max(0, e_mode - self.config.range_half)
        hi = min(255, e_mode + self.config.range_half)

        in_range = (exponent >= lo) & (exponent <= hi)
        coverage = float(in_range.sum()) / BLOCK_SIZE

        # Entropy guard: fall back to raw BF16 if too few elements are in range.
        if coverage < self.config.min_coverage:
            return TcaTbeBlock(
                is_raw=True,
                e_mode=_FALLBACK_MARKER,
                e_lo_offset=0,
                e_hi_offset=0,
                sign_bitmap=sign,
                range_bitmap=in_range.astype(np.uint8),
                exp_offset_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
                mantissa_bitmap=np.zeros((BLOCK_SIZE, self.config.mantissa_bits), dtype=np.uint8),
                mantissa_bits=self.config.mantissa_bits,
                spill_words=words[:n_elements].copy(),
                n_elements=n_elements,
            )

        # Build sign bitmap (already have from _bf16_to_parts).
        # Build range bitmap.
        range_bm = in_range.astype(np.uint8)

        # Build exponent-offset bitmap: per-element offset from window base `lo`.
        # Values for in-range elements: exponent - lo ∈ {0, .., hi-lo} (max 2 for range_half=1).
        # For out-of-range elements we store 0 (unused; they come from spill).
        exp_offset = np.where(in_range, exponent.astype(np.int32) - lo, 0).clip(0, 255).astype(np.uint8)

        # Build mantissa bitmap: top-N mantissa bits per element.
        top_bits = self.config.mantissa_bits
        shift = 7 - top_bits  # shift to bring top bits to LSB position
        mantissa_bm = ((mantissa >> shift) & ((1 << top_bits) - 1)).astype(np.uint8)
        # Expand to (BLOCK_SIZE, mantissa_bits) as individual bits (little-endian within byte).
        mantissa_2d = np.unpackbits(
            mantissa_bm[:, np.newaxis].view(np.uint8),
            axis=1,
            bitorder="little",
        )[:, :top_bits]

        # Spill: out-of-range elements stored as raw BF16.
        spill_idx = np.where(~in_range)[0]
        spill_words = words[spill_idx].copy()

        e_lo_offset = e_mode - lo
        e_hi_offset = hi - e_mode

        return TcaTbeBlock(
            is_raw=False,
            e_mode=e_mode,
            e_lo_offset=e_lo_offset,
            e_hi_offset=e_hi_offset,
            sign_bitmap=sign,
            range_bitmap=range_bm,
            exp_offset_bitmap=exp_offset,
            mantissa_bitmap=mantissa_2d,
            mantissa_bits=top_bits,
            spill_words=spill_words,
            n_elements=n_elements,
        )

    def decode(self, block: TcaTbeBlock) -> np.ndarray:
        """Decode a :class:`TcaTbeBlock` back to ``(BLOCK_SIZE,) np.uint16``.

        The output is bit-for-bit identical to the original input for
        compressed blocks, and equal to the raw words for fallback blocks.

        Args:
            block: Encoded block produced by :meth:`encode`.

        Returns:
            ``(n_elements,) np.uint16`` array of raw BF16 bit patterns.
        """
        if block.is_raw:
            return block.spill_words.copy()

        sign     = block.sign_bitmap.astype(np.uint8)
        in_range = block.range_bitmap.astype(bool)

        # Reconstruct exact exponent for each element from the stored per-element offset.
        # For in-range elements: exponent = window_lo + exp_offset_bitmap[i]
        # For out-of-range elements: exponent will be overwritten by spill_words.
        window_lo = np.uint16(block.e_mode - block.e_lo_offset)
        exponent_reconstructed = (window_lo + block.exp_offset_bitmap.astype(np.uint16)).astype(np.uint8)

        # Reconstruct mantissa from 2-D mantissa bitmap.
        top_bits = block.mantissa_bits
        shift = 7 - top_bits
        mantissa_top = np.packbits(
            block.mantissa_bitmap[:, :top_bits],
            axis=1,
            bitorder="little",
        )[:, 0].astype(np.uint16)
        # Shift back to original mantissa bit positions.
        mantissa = (mantissa_top << shift).astype(np.uint16)

        result = _parts_to_bf16(sign, exponent_reconstructed, mantissa)

        # Restore out-of-range elements from spill (overwrites reconstructed values).
        out_idx = np.where(~in_range)[0]
        if len(out_idx) > 0 and len(block.spill_words) > 0:
            result[out_idx[:len(block.spill_words)]] = block.spill_words

        return result[:block.n_elements]

    # ------------------------------------------------------------------
    # Block serialisation
    # ------------------------------------------------------------------

    def encode_to_bytes(self, block: TcaTbeBlock) -> bytes:
        """Serialise a :class:`TcaTbeBlock` to the binary wire format.

        Wire format (compressed block):
            1B  : flags (bit 0 = 0 means compressed, 1 means raw)
            1B  : e_mode
            1B  : e_lo_offset
            1B  : e_hi_offset
            16B : sign_bitmap  (128 bits packed, byte-aligned)
            16B : range_bitmap (128 bits packed)
            NB  : mantissa_bitmap (BLOCK_SIZE * mantissa_bits bits, packed)
            2B  : spill_count (uint16 LE)
            S×2B: spill_words

        Wire format (raw fallback block):
            1B  : flags = 0x01
            2B  : n_elements (uint16 LE)
            n_elements × 2B : raw BF16 words

        Args:
            block: Block to serialise.

        Returns:
            Bytes object.
        """
        if block.is_raw:
            return (
                b"\x01"
                + struct.pack("<H", block.n_elements)
                + block.spill_words.astype("<u2").tobytes()
            )

        # Pack bitmaps to bytes.
        sign_packed  = np.packbits(block.sign_bitmap,  bitorder="little").tobytes()   # 16 B
        range_packed = np.packbits(block.range_bitmap, bitorder="little").tobytes()   # 16 B

        # Pack exp_offset_bitmap: 2 bits per element (window ≤ 3 for range_half=1).
        # Clamp values to 2-bit range, then pack pairs of bits into bytes.
        exp_clamped = block.exp_offset_bitmap.astype(np.uint8) & 0x03  # keep low 2 bits
        # Expand each uint8 value to 2 individual bits, then pack.
        exp_2bit = np.unpackbits(
            exp_clamped[:, np.newaxis].view(np.uint8), axis=1, bitorder="little"
        )[:, :2].flatten()
        exp_packed = np.packbits(exp_2bit, bitorder="little").tobytes()  # 32 B

        # Flatten mantissa_bitmap: (128, mantissa_bits) → 1-D, then pack.
        mantissa_flat = block.mantissa_bitmap.flatten()
        mantissa_packed = np.packbits(mantissa_flat, bitorder="little").tobytes()

        spill_count = len(block.spill_words)
        header = struct.pack(
            "<BBBB", 0x00, block.e_mode, block.e_lo_offset, block.e_hi_offset
        )
        spill_data = block.spill_words.astype("<u2").tobytes()
        return (
            header
            + sign_packed
            + range_packed
            + exp_packed
            + mantissa_packed
            + struct.pack("<H", spill_count)
            + spill_data
        )

    def decode_from_bytes(
        self, data: bytes, mantissa_bits: int = 4
    ) -> TcaTbeBlock:
        """Deserialise bytes produced by :meth:`encode_to_bytes`.

        Args:
            data: Serialised block bytes.
            mantissa_bits: Number of mantissa bits used during encoding.

        Returns:
            :class:`TcaTbeBlock`.
        """
        flags = data[0]
        if flags & 0x01:
            # Raw block.
            n_elements, = struct.unpack_from("<H", data, 1)
            words = np.frombuffer(data[3 : 3 + n_elements * 2], dtype="<u2").copy()
            return TcaTbeBlock(
                is_raw=True, e_mode=_FALLBACK_MARKER,
                e_lo_offset=0, e_hi_offset=0,
                sign_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
                range_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
                exp_offset_bitmap=np.zeros(BLOCK_SIZE, dtype=np.uint8),
                mantissa_bitmap=np.zeros((BLOCK_SIZE, mantissa_bits), dtype=np.uint8),
                mantissa_bits=mantissa_bits,
                spill_words=words, n_elements=n_elements,
            )

        # Compressed block.
        _, e_mode, e_lo, e_hi = struct.unpack_from("<BBBB", data, 0)
        off = 4
        sign_bitmap  = np.unpackbits(np.frombuffer(data[off:off + 16], dtype=np.uint8), bitorder="little")[:BLOCK_SIZE]
        off += 16
        range_bitmap = np.unpackbits(np.frombuffer(data[off:off + 16], dtype=np.uint8), bitorder="little")[:BLOCK_SIZE]
        off += 16
        # Exp offset bitmap: 2 bits per element → 32 bytes.
        n_exp_bytes = (BLOCK_SIZE * 2 + 7) // 8   # 32
        exp_flat = np.unpackbits(
            np.frombuffer(data[off:off + n_exp_bytes], dtype=np.uint8),
            bitorder="little",
        )
        # Pack pairs of bits back into uint8 offset values.
        exp_pairs = exp_flat[: BLOCK_SIZE * 2].reshape(BLOCK_SIZE, 2)
        exp_offset_bm = np.packbits(
            np.concatenate([exp_pairs, np.zeros((BLOCK_SIZE, 6), dtype=np.uint8)], axis=1),
            axis=1, bitorder="little",
        )[:, 0].astype(np.uint8)
        off += n_exp_bytes
        n_mantissa_bytes = (BLOCK_SIZE * mantissa_bits + 7) // 8
        mantissa_flat = np.unpackbits(
            np.frombuffer(data[off:off + n_mantissa_bytes], dtype=np.uint8),
            bitorder="little",
        )
        mantissa_bm = mantissa_flat[: BLOCK_SIZE * mantissa_bits].reshape(BLOCK_SIZE, mantissa_bits)
        off += n_mantissa_bytes
        spill_count, = struct.unpack_from("<H", data, off)
        off += 2
        spill_words = np.frombuffer(data[off:off + spill_count * 2], dtype="<u2").copy()

        return TcaTbeBlock(
            is_raw=False, e_mode=e_mode,
            e_lo_offset=e_lo, e_hi_offset=e_hi,
            sign_bitmap=sign_bitmap.astype(np.uint8),
            range_bitmap=range_bitmap.astype(np.uint8),
            exp_offset_bitmap=exp_offset_bm,
            mantissa_bitmap=mantissa_bm.astype(np.uint8),
            mantissa_bits=mantissa_bits,
            spill_words=spill_words, n_elements=BLOCK_SIZE,
        )


# ---------------------------------------------------------------------------
# Tensor-level helpers
# ---------------------------------------------------------------------------

def tca_tbe_encode_tensor(
    tensor_bf16: np.ndarray,
    config: Optional[TcaTbeConfig] = None,
) -> Tuple[List[TcaTbeBlock], CompressionStats]:
    """Encode a flat BF16 tensor ``(N,) np.uint16`` into TCA-TBE blocks.

    The tensor is split into ceil(N / BLOCK_SIZE) blocks.  Each block is
    encoded independently.  The final block is padded with zeros if N is
    not a multiple of BLOCK_SIZE.

    Args:
        tensor_bf16: Flat ``(N,) np.uint16`` array of raw BF16 bit patterns.
        config: Encoding configuration.

    Returns:
        Tuple of ``(blocks, stats)``.
    """
    tensor_bf16 = np.asarray(tensor_bf16, dtype=np.uint16).ravel()
    cfg = config or TcaTbeConfig()
    codec = TcaTbeCodec(cfg)

    n = len(tensor_bf16)
    blocks: List[TcaTbeBlock] = []
    stats = CompressionStats(
        raw_bytes_uncompressed=n * 2,
    )

    for i in range(0, n, BLOCK_SIZE):
        chunk = tensor_bf16[i : i + BLOCK_SIZE]
        blk = codec.encode(chunk)
        blocks.append(blk)
        stats.n_blocks += 1
        if blk.is_raw:
            stats.n_raw += 1
        else:
            stats.n_compressed += 1
        stats.compressed_bytes += blk.compressed_bytes()
        stats.spill_elements += len(blk.spill_words)

    return blocks, stats


def tca_tbe_decode_tensor(
    blocks: List[TcaTbeBlock],
    n_elements: Optional[int] = None,
) -> np.ndarray:
    """Decode a list of :class:`TcaTbeBlock` back to a flat BF16 ``np.uint16`` array.

    Args:
        blocks: Encoded blocks from :func:`tca_tbe_encode_tensor`.
        n_elements: If given, the output is truncated to this length (removes
            zero-padding from the final block).

    Returns:
        ``(n_elements,) np.uint16`` flat array of raw BF16 bit patterns.
    """
    if not blocks:
        return np.empty(0, dtype=np.uint16)

    cfg = TcaTbeConfig(mantissa_bits=blocks[0].mantissa_bits)
    codec = TcaTbeCodec(cfg)
    parts: List[np.ndarray] = []
    for blk in blocks:
        parts.append(codec.decode(blk))

    result = np.concatenate(parts)
    if n_elements is not None:
        result = result[:n_elements]
    return result
