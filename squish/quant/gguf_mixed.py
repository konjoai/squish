"""GGUF mixed-precision block quantization (llama.cpp v2, 2023).

Q2_K / Q3_K / Q4_K / Q5_K / Q8_0 block-quantization families with
portable byte-level checkpoint encode / decode.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import numpy as np

# Supported GGUF quantization types and their bit widths
_QUANT_BITS: dict[str, int] = {
    "Q2_K": 2,
    "Q3_K": 3,
    "Q4_K": 4,
    "Q5_K": 5,
    "Q8_0": 8,
}

# Magic bytes for the portable checkpoint header
_MAGIC = b"GGUF"
_VERSION = 2


@dataclass
class GGUFConfig:
    """Configuration for GGUF block quantization."""

    quant_type: str = "Q3_K"
    group_size: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if self.quant_type not in _QUANT_BITS:
            raise ValueError(
                f"quant_type must be one of {set(_QUANT_BITS)}, got {self.quant_type!r}"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")


@dataclass
class GGUFTensor:
    """Quantized tensor in GGUF block format."""

    quant_type: str
    quantized: np.ndarray    # (rows, padded_cols) int8
    scales: np.ndarray       # (n_blocks,) float32 per-block scale
    mins: np.ndarray         # (n_blocks,) float32 per-block minimum
    super_scales: np.ndarray # (n_super_blocks,) float32 super-block meta-scale
    shape: Tuple[int, int]   # original (rows, cols) before padding

    @property
    def quant_bits(self) -> int:
        """Number of bits used for this quantization type."""
        return _QUANT_BITS[self.quant_type]


class GGUFMixedQuantizer:
    """GGUF Q_K block quantizer with portable checkpoint support.

    Implements:
    * per-block {scale, min} quantization à la Q_K format
    * super-block meta-scale (one float32 per 8 blocks)
    * byte-level checkpoint encode/decode for portability
    """

    def __init__(self, config: Optional[GGUFConfig] = None) -> None:
        self._config = config or GGUFConfig()

    @property
    def config(self) -> GGUFConfig:
        return self._config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quant_bits_for_type(qt: str) -> int:
        """Return the integer bit width for a given GGUF quant type."""
        return _QUANT_BITS[qt]

    def _quantize_block(
        self,
        block: np.ndarray,
    ) -> Tuple[np.ndarray, float, float]:
        """Quantize a single 1-D block of floats.

        Values are quantized to unsigned range ``[0, q_levels]`` then
        stored in ``int8`` by subtracting an offset of ``1 << (bits-1)``
        so that the full unsigned range maps cleanly into the signed
        ``int8`` domain without overflow.

        Returns
        -------
        q: ``(group_size,)`` int8 quantized values
        scale: float32 block scale
        min_val: float32 block minimum
        """
        bits = _QUANT_BITS[self._config.quant_type]
        q_levels = (1 << bits) - 1  # e.g. 255 for 8-bit, 7 for 3-bit
        q_offset = 1 << (bits - 1)   # e.g. 128 for 8-bit (unsigned → signed)

        b_min = float(block.min())
        b_max = float(block.max())

        scale = (b_max - b_min) / q_levels if b_max != b_min else 1.0
        q_unsigned = np.round((block - b_min) / scale).astype(np.float32)
        q_unsigned = np.clip(q_unsigned, 0, q_levels)
        # Shift into signed int8 range: [0, q_levels] → [-q_offset, q_levels-q_offset]
        q = (q_unsigned - q_offset).astype(np.int8)
        return q, scale, b_min

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, W: np.ndarray) -> GGUFTensor:
        """Block-quantize weight matrix W.

        Parameters
        ----------
        W:
            ``(rows, cols)`` float32 weight matrix.

        Returns
        -------
        GGUFTensor
        """
        cfg = self._config
        W = np.asarray(W, dtype=np.float32)
        rows, cols = W.shape

        # Pad cols to multiple of group_size
        pad = (-cols) % cfg.group_size
        padded_cols = cols + pad
        if pad:
            W_padded = np.pad(W, ((0, 0), (0, pad)))
        else:
            W_padded = W

        n_col_blocks = padded_cols // cfg.group_size
        n_blocks = rows * n_col_blocks
        n_super_blocks = max(1, (n_blocks + 7) // 8)

        quantized = np.empty((rows, padded_cols), dtype=np.int8)
        scales = np.empty(n_blocks, dtype=np.float32)
        mins = np.empty(n_blocks, dtype=np.float32)

        block_idx = 0
        for r in range(rows):
            for bc in range(n_col_blocks):
                col_start = bc * cfg.group_size
                col_end = col_start + cfg.group_size
                q, scale, min_val = self._quantize_block(W_padded[r, col_start:col_end])
                quantized[r, col_start:col_end] = q
                scales[block_idx] = scale
                mins[block_idx] = min_val
                block_idx += 1

        # Super-block meta-scale: mean of each group of 8 scales
        super_scales = np.array(
            [
                float(scales[i * 8 : min((i + 1) * 8, n_blocks)].mean())
                for i in range(n_super_blocks)
            ],
            dtype=np.float32,
        )

        return GGUFTensor(
            quant_type=cfg.quant_type,
            quantized=quantized,
            scales=scales,
            mins=mins,
            super_scales=super_scales,
            shape=(rows, cols),
        )

    def dequantize(self, tensor: GGUFTensor) -> np.ndarray:
        """Reconstruct float32 weight matrix from a GGUFTensor.

        Parameters
        ----------
        tensor:
            Quantized tensor produced by :meth:`quantize`.

        Returns
        -------
        np.ndarray
            ``(rows, cols)`` float32 weight.
        """
        cfg = self._config
        rows, cols = tensor.shape
        padded_cols = tensor.quantized.shape[1]
        n_col_blocks = padded_cols // cfg.group_size

        W_hat = np.empty((rows, padded_cols), dtype=np.float32)
        bits = _QUANT_BITS[tensor.quant_type]
        q_offset = 1 << (bits - 1)  # must mirror the offset used in _quantize_block
        block_idx = 0
        for r in range(rows):
            for bc in range(n_col_blocks):
                col_start = bc * cfg.group_size
                col_end = col_start + cfg.group_size
                scale = tensor.scales[block_idx]
                min_val = tensor.mins[block_idx]
                # Reverse the signed-int8 shift back to unsigned, then dequantize
                q_unsigned = tensor.quantized[r, col_start:col_end].astype(np.float32) + q_offset
                W_hat[r, col_start:col_end] = q_unsigned * scale + min_val
                block_idx += 1

        return W_hat[:, :cols]

    def forward(self, x: np.ndarray, tensor: GGUFTensor) -> np.ndarray:
        """Matrix-multiply input ``x`` by the dequantized weight.

        Parameters
        ----------
        x:
            ``(*, rows)`` float32 input.
        tensor:
            Quantized tensor produced by :meth:`quantize`.

        Returns
        -------
        np.ndarray
            ``(*, cols)`` output.
        """
        W_hat = self.dequantize(tensor)
        return np.tensordot(x, W_hat, axes=([-1], [0]))

    def encode_to_bytes(self, tensor: GGUFTensor) -> bytes:
        """Serialize a GGUFTensor to a portable byte string.

        Format (little-endian)
        ----------------------
        Header (20 bytes):
            magic           4s   "GGUF"
            version         I    uint32
            rows            I    uint32
            cols            I    uint32
            quant_type_len  I    uint32 (length of quant_type string)
        quant_type string:  <quant_type_len> bytes UTF-8
        scales:             float32 × n_blocks
        mins:               float32 × n_blocks
        super_scales:       float32 × n_super_blocks
        quantized:          int8 × (rows × padded_cols)
        """
        rows, cols = tensor.shape
        qt_bytes = tensor.quant_type.encode("utf-8")
        header = struct.pack(
            "<4sIIII",
            _MAGIC,
            _VERSION,
            rows,
            cols,
            len(qt_bytes),
        )
        body = (
            qt_bytes
            + tensor.scales.astype(np.float32).tobytes()
            + tensor.mins.astype(np.float32).tobytes()
            + tensor.super_scales.astype(np.float32).tobytes()
            + tensor.quantized.astype(np.int8).tobytes()
        )
        return header + body

    def decode_from_bytes(self, data: bytes, shape: Tuple[int, int]) -> GGUFTensor:
        """Deserialize a GGUFTensor from bytes produced by :meth:`encode_to_bytes`.

        Parameters
        ----------
        data:
            Raw bytes as returned by :meth:`encode_to_bytes`.
        shape:
            Original ``(rows, cols)`` of the weight matrix.

        Returns
        -------
        GGUFTensor
        """
        cfg = self._config
        rows, cols = shape

        # Parse header
        header_size = struct.calcsize("<4sIIII")
        magic, version, enc_rows, enc_cols, qt_len = struct.unpack_from(
            "<4sIIII", data, 0
        )
        if magic != _MAGIC:
            raise ValueError(f"Invalid GGUF magic bytes: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported GGUF version: {version}")
        if enc_rows != rows or enc_cols != cols:
            raise ValueError(
                f"Shape mismatch: header ({enc_rows}, {enc_cols}) != provided {shape}"
            )

        offset = header_size
        quant_type = data[offset : offset + qt_len].decode("utf-8")
        offset += qt_len

        pad = (-cols) % cfg.group_size
        padded_cols = cols + pad
        n_col_blocks = padded_cols // cfg.group_size
        n_blocks = rows * n_col_blocks
        n_super_blocks = max(1, (n_blocks + 7) // 8)

        scales_bytes = n_blocks * 4
        scales = np.frombuffer(data[offset : offset + scales_bytes], dtype=np.float32).copy()
        offset += scales_bytes

        mins = np.frombuffer(data[offset : offset + scales_bytes], dtype=np.float32).copy()
        offset += scales_bytes

        super_scales_bytes = n_super_blocks * 4
        super_scales = np.frombuffer(
            data[offset : offset + super_scales_bytes], dtype=np.float32
        ).copy()
        offset += super_scales_bytes

        q_bytes = rows * padded_cols
        quantized_flat = np.frombuffer(data[offset : offset + q_bytes], dtype=np.int8).copy()
        quantized = quantized_flat.reshape(rows, padded_cols)

        return GGUFTensor(
            quant_type=quant_type,
            quantized=quantized,
            scales=scales,
            mins=mins,
            super_scales=super_scales,
            shape=(rows, cols),
        )
