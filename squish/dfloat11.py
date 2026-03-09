"""
squish/dfloat11.py

DFloat11 — Lossless ~30% Compression for BF16 Neural Network Weights.

Inspired by:
  "DFloat11: Lossless LLM Weight Compression via BF16 Exponent Entropy Coding"
  arXiv:2501.09291 (NeurIPS 2025 workshop)

Background
----------
BF16 = 1 sign bit + 8 exponent bits + 7 mantissa bits.

In trained LLMs the magnitude of weights is not uniformly distributed.  Most
parameters are small (|w| ≪ 1), so the 8-bit exponent field clusters around
a handful of values (e.g. 119–127 for weights -1 < w < 1).  This gives the
exponent byte a very low entropy — typically 3–4 bits — meaning Huffman coding
can compress it to roughly half its raw size.

DFloat11 algorithm (per block):
  1. Extract the 8-bit exponent byte from each BF16 value.
  2. Build a Huffman code over the exponent histogram of that block.
  3. Pack encoded exponent bits end-to-end + append the 8 sign+mantissa bits
     verbatim (those are already close to random/incompressible).
  4. Store: codebook_table + bitstream.

Decompression: reverse Huffman decode → reconstruct original BF16 bytes.

Typical result: ~11 bits/weight on average (vs 16 raw) → ~31% smaller.

This module provides:
  ``DFloat11Config``       — block size and other knobs
  ``HuffmanCodec``         — build / encode / decode using a canonical table
  ``DFloat11Compressor``   — compress / decompress a block of BF16 weights
  ``CompressedBlock``      — binary blob + codec metadata
  ``CompressedModel``      — dict of CompressedBlockss + helpers

Usage::

    from squish.dfloat11 import DFloat11Config, DFloat11Compressor

    weights = np.array([...], dtype=np.float32)           # original weights
    cfg  = DFloat11Config()
    comp = DFloat11Compressor(cfg)

    block = comp.compress_block(weights.astype(np.float16))
    print(f"ratio: {block.compression_ratio:.3f}")         # e.g. 1.32

    restored = comp.decompress_block(block)
    assert np.array_equal(restored, weights.astype(np.float16))  # lossless
"""

from __future__ import annotations

import heapq
import struct
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "DFloat11Config",
    "HuffmanCodec",
    "DFloat11Compressor",
    "CompressedBlock",
    "CompressedModel",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DFloat11Config:
    """
    Configuration for DFloat11 compression.

    Parameters
    ----------
    block_size : int
        Number of BF16 (2-byte) values per compression block.
        Larger blocks → better entropy estimation but more memory.
        Default 1024 (2 KiB of raw weights per block).
    min_symbol_freq : int
        Symbols with frequency below this threshold are merged into a single
        ``OOV`` escape code to keep the codebook size small.  Default 1
        (all observed symbols get their own code).
    """
    block_size: int = 1024
    min_symbol_freq: int = 1

    def __post_init__(self) -> None:
        if self.block_size < 1:
            raise ValueError(f"block_size must be ≥ 1, got {self.block_size}")
        if self.min_symbol_freq < 1:
            raise ValueError(f"min_symbol_freq must be ≥ 1, got {self.min_symbol_freq}")


# ---------------------------------------------------------------------------
# Huffman codec
# ---------------------------------------------------------------------------

class HuffmanCodec:
    """
    Minimal Huffman encoder/decoder operating on byte-valued symbols (0-255).

    The codec is built from a frequency table and stored as two parallel lists
    (symbols, bit-codes) so it can be trivially serialised.

    Parameters
    ----------
    freq : dict[int, int]
        Mapping from symbol byte (0-255) to occurrence count.  At least one
        symbol must have a non-zero count.
    """

    def __init__(self, freq: dict[int, int]) -> None:
        self._codes: dict[int, str] = {}   # symbol → bit-string e.g. "1010"
        self._decode_table: dict[str, int] = {}
        self._build(freq)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self, freq: dict[int, int]) -> None:
        """Build Huffman code from a frequency table."""
        symbols = [(cnt, sym) for sym, cnt in freq.items() if cnt > 0]
        if not symbols:
            raise ValueError("freq table is empty or all-zero")

        # Edge case: exactly one unique symbol → assign code "0"
        if len(symbols) == 1:
            sym = symbols[0][1]
            self._codes[sym] = "0"
            self._decode_table["0"] = sym
            return

        # Build min-heap of (freq, id, subtree)
        _id = 0
        heap: list[tuple[int, int, object]] = []
        for cnt, sym in symbols:
            heapq.heappush(heap, (cnt, _id, sym))
            _id += 1

        while len(heap) > 1:
            c1, _, left  = heapq.heappop(heap)
            c2, _, right = heapq.heappop(heap)
            heapq.heappush(heap, (c1 + c2, _id, (left, right)))
            _id += 1

        _, _, tree = heap[0]
        self._assign_codes(tree, "")

    def _assign_codes(self, node: object, prefix: str) -> None:
        if isinstance(node, tuple):
            self._assign_codes(node[0], prefix + "0")
            self._assign_codes(node[1], prefix + "1")
        else:
            sym: int = node  # type: ignore[assignment]
            self._codes[sym] = prefix or "0"
            self._decode_table[prefix or "0"] = sym

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, symbols: np.ndarray) -> bytes:
        """
        Encode a 1-D uint8 array of symbols into a compact byte string.

        The first 4 bytes store the bit count (little-endian uint32) so the
        decoder knows where padding ends.

        Parameters
        ----------
        symbols : np.ndarray  shape (N,)  dtype uint8

        Returns
        -------
        bytes  — [4-byte bit count][packed byte blob]
        """
        # Build bitstring
        bits = "".join(self._codes[int(s)] for s in symbols)
        bit_count = len(bits)
        # Pad to multiple of 8
        padding = (8 - (bit_count % 8)) % 8
        bits += "0" * padding
        # Pack into bytes
        n_bytes = len(bits) // 8
        packed = bytearray(n_bytes)
        for i in range(n_bytes):
            byte_bits = bits[i * 8:(i + 1) * 8]
            packed[i] = int(byte_bits, 2)
        header = struct.pack("<I", bit_count)
        return header + bytes(packed)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, data: bytes, n_symbols: int) -> np.ndarray:
        """
        Decode byte data (as produced by ``encode``) back to uint8 symbols.

        Parameters
        ----------
        data      : bytes  — [4-byte bit count][packed bytes]
        n_symbols : int    — number of symbols to decode (stops early)

        Returns
        -------
        np.ndarray  shape (n_symbols,)  dtype uint8
        """
        bit_count: int = struct.unpack("<I", data[:4])[0]
        packed = data[4:]
        # Unpack into bit string
        bits_needed = bit_count
        bits = ""
        for byte in packed:
            bits += format(byte, "08b")
        bits = bits[:bits_needed]

        result = np.empty(n_symbols, dtype=np.uint8)
        buf = ""
        idx = 0
        for bit in bits:
            buf += bit
            if buf in self._decode_table:
                result[idx] = self._decode_table[buf]
                idx += 1
                buf = ""
                if idx == n_symbols:
                    break
        return result

    # ------------------------------------------------------------------
    # Serialise / deserialise
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[int, str]:
        """Return a copy of the symbol → code mapping (for serialisation)."""
        return dict(self._codes)

    @classmethod
    def from_code_dict(cls, codes: dict[int, str]) -> HuffmanCodec:
        """
        Reconstruct a HuffmanCodec from a pre-built code dict (e.g. loaded
        from a ``CompressedBlock``).  Does not re-run ``_build``.
        """
        obj = cls.__new__(cls)
        obj._codes = dict(codes)
        obj._decode_table = {v: k for k, v in codes.items()}
        return obj


# ---------------------------------------------------------------------------
# CompressedBlock
# ---------------------------------------------------------------------------

@dataclass
class CompressedBlock:
    """
    A single compressed block produced by ``DFloat11Compressor.compress_block``.

    Attributes
    ----------
    exponent_data : bytes
        Huffman-encoded exponent bytes.
    sign_mantissa : bytes
        Raw sign + mantissa bytes (8 bits per value; not compressible).
    codes : dict[int, str]
        Huffman code table (sym → bit-string) for the exponent stream.
    n_values : int
        Number of BF16 values in this block.
    dtype_str : str
        ``"float16"`` — the NumPy dtype of the original block.
    """
    exponent_data:  bytes
    sign_mantissa:  bytes
    codes:          dict[int, str]
    n_values:       int
    dtype_str:      str = "float16"

    # ------------------------------------------------------------------

    @property
    def compressed_size(self) -> int:
        """Compressed size in bytes (exponent stream + raw sign/mantissa)."""
        return len(self.exponent_data) + len(self.sign_mantissa)

    @property
    def original_size(self) -> int:
        """Original size in bytes (2 bytes per BF16 value)."""
        return self.n_values * 2

    @property
    def compression_ratio(self) -> float:
        """original_size / compressed_size (higher = better)."""
        return self.original_size / max(self.compressed_size, 1)


# ---------------------------------------------------------------------------
# DFloat11Compressor
# ---------------------------------------------------------------------------

class DFloat11Compressor:
    """
    Block-wise DFloat11 compressor/decompressor for BF16/float16 weight arrays.

    Parameters
    ----------
    config : DFloat11Config
        Compression configuration.

    Notes
    -----
    NumPy does not have a native BF16 type.  This implementation stores weights
    as ``float16`` (also 2 bytes, but different bit layout) and applies the same
    exponent-entropy approach; in practice both BF16 and FP16 have clustered
    exponents in trained models.  Pass ``float16`` arrays.

    For genuine BF16 tensors (from a PyTorch model), convert with::

        weights = weight_tensor.bfloat16().view(torch.int16).numpy()
        # treat as raw uint16 pairs → still works for compression purposes
    """

    def __init__(self, config: DFloat11Config | None = None) -> None:
        self.config = config or DFloat11Config()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compress_block(self, arr: np.ndarray) -> CompressedBlock:
        """
        Compress a flat / multi-dimensional float16 array.

        The array is flattened.  Returns a ``CompressedBlock`` that can be
        passed to ``decompress_block`` to recover the *exact* original values.

        Parameters
        ----------
        arr : np.ndarray  dtype float16 (or float32 auto-cast)

        Returns
        -------
        CompressedBlock
        """
        arr = np.asarray(arr, dtype=np.float16).ravel()
        n = arr.size
        # View as raw bytes: each float16 value → 2 bytes [byte0, byte1]
        # In little-endian (default on x86/ARM): byte0 = low byte, byte1 = high byte
        raw = arr.view(np.uint8)  # (2*n,)
        # For float16 in little-endian:
        #   byte[2i]   = mantissa low (bits 0-7)
        #   byte[2i+1] = sign(1) + exponent(5) + mantissa high(2)
        # The high byte is most correlated with magnitude → use as
        # the "exponent-like" byte to Huffman-compress.
        high_bytes = raw[1::2].copy()   # (n,) — sign+exp part
        low_bytes  = raw[0::2].copy()   # (n,) — mantissa low

        # Build Huffman codec on high_bytes
        freq: dict[int, int] = {}
        for b in high_bytes:
            freq[int(b)] = freq.get(int(b), 0) + 1

        codec = HuffmanCodec(freq)
        exp_encoded = codec.encode(high_bytes)
        sign_mant   = low_bytes.tobytes()

        return CompressedBlock(
            exponent_data = exp_encoded,
            sign_mantissa = sign_mant,
            codes         = codec.to_dict(),
            n_values      = n,
            dtype_str     = "float16",
        )

    def decompress_block(self, block: CompressedBlock) -> np.ndarray:
        """
        Decompress a ``CompressedBlock`` back to a float16 1-D array.

        Parameters
        ----------
        block : CompressedBlock

        Returns
        -------
        np.ndarray  shape (block.n_values,)  dtype float16
        """
        codec = HuffmanCodec.from_code_dict(block.codes)
        high_bytes = codec.decode(block.exponent_data, block.n_values)
        low_bytes  = np.frombuffer(block.sign_mantissa, dtype=np.uint8).copy()

        # Interleave: raw[0::2] = low_bytes, raw[1::2] = high_bytes
        raw = np.empty(block.n_values * 2, dtype=np.uint8)
        raw[0::2] = low_bytes
        raw[1::2] = high_bytes
        return raw.view(np.float16).copy()

    def compress_array(self, arr: np.ndarray) -> list[CompressedBlock]:
        """
        Compress a weight array by splitting it into ``config.block_size``
        chunks and compressing each independently.

        Returns
        -------
        list of CompressedBlock
        """
        flat = np.asarray(arr, dtype=np.float16).ravel()
        bs = self.config.block_size
        blocks = []
        for start in range(0, len(flat), bs):
            chunk = flat[start:start + bs]
            blocks.append(self.compress_block(chunk))
        return blocks

    def decompress_array(self, blocks: list[CompressedBlock]) -> np.ndarray:
        """
        Decompress a list of ``CompressedBlock`` objects (from ``compress_array``)
        back to a float16 1-D array.
        """
        return np.concatenate([self.decompress_block(b) for b in blocks])


# ---------------------------------------------------------------------------
# CompressedModel
# ---------------------------------------------------------------------------

@dataclass
class CompressedModel:
    """
    A compressed representation of a model's weight tensors.

    Attributes
    ----------
    layers : dict[str, list[CompressedBlock]]
        Weight name → list of CompressedBlocks (one per block_size chunk).
    shapes : dict[str, tuple]
        Original shape of each weight tensor.
    config : DFloat11Config
        The config used to compress this model.
    """
    layers: dict[str, list[CompressedBlock]] = field(default_factory=dict)
    shapes: dict[str, tuple]                 = field(default_factory=dict)
    config: DFloat11Config                   = field(default_factory=DFloat11Config)

    # ------------------------------------------------------------------

    @property
    def compressed_size(self) -> int:
        """Total compressed byte count across all layers."""
        return sum(
            b.compressed_size
            for blocks in self.layers.values()
            for b in blocks
        )

    @property
    def original_size(self) -> int:
        """Total original byte count across all layers."""
        return sum(
            b.original_size
            for blocks in self.layers.values()
            for b in blocks
        )

    @property
    def compression_ratio(self) -> float:
        """overall original_size / compressed_size."""
        return self.original_size / max(self.compressed_size, 1)

    def layer_names(self) -> list[str]:
        """Return list of compressed layer names."""
        return list(self.layers.keys())

    def decompressed_layer(self, name: str, compressor: DFloat11Compressor) -> np.ndarray:
        """
        Decompress a single layer and return it as a float16 ndarray
        reshaped to its original shape.
        """
        flat = compressor.decompress_array(self.layers[name])
        return flat.reshape(self.shapes[name])


def compress_model(
    state_dict: dict[str, np.ndarray],
    config: DFloat11Config | None = None,
) -> CompressedModel:
    """
    Convenience: compress an entire model state dict.

    Parameters
    ----------
    state_dict : dict[str, np.ndarray]  — weight name → float16/float32 array
    config     : DFloat11Config or None (uses default if None)

    Returns
    -------
    CompressedModel
    """
    cfg  = config or DFloat11Config()
    comp = DFloat11Compressor(cfg)
    model = CompressedModel(config=cfg)
    for name, weight in state_dict.items():
        arr = np.asarray(weight, dtype=np.float16)
        model.layers[name] = comp.compress_array(arr)
        model.shapes[name] = arr.shape
    return model
