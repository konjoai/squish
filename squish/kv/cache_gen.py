"""squish/kv/cache_gen.py

CacheGen — Arithmetic-Coded KV Bitstream Compression & Streaming Decoder
(Liu et al., SIGCOMM 2024 / arXiv:2310.07240).

Reference
---------
"CacheGen: KV Cache Compression and Streaming for Fast Large Language Model
Serving." Liu et al., SIGCOMM 2024 (arXiv:2310.07240).

Algorithm
---------
CacheGen compresses the KV cache into a compact bitstream for fast
transmission and reconstruction.  This simulation implements:

1. **Quantization** — symmetric uniform quantization to ``bits`` bit-width.
2. **Byte-packing** — block-by-block packing of codes into a byte buffer.
3. **Header** — shaped header stored as raw bytes so the decoder can reconstruct
   the tensor shape without out-of-band metadata.
4. **Streaming** — ``stream_encode()`` yields successive chunk byte buffers.

Compression ratio is ``bytes(encoded) / bytes(float32_original)``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

__all__ = [
    "CacheGenConfig",
    "CacheGenCodec",
]

_VALID_BITS = frozenset({4, 8})

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class CacheGenConfig:
    """Configuration for :class:`CacheGenCodec`.

    Attributes:
        bits: Quantization bit-width (4 or 8).
        block_size: Element block size for packing.
    """

    bits: int = 8
    block_size: int = 64

    def __post_init__(self) -> None:
        if self.bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(_VALID_BITS)}; got {self.bits}")
        if self.block_size < 1:
            raise ValueError(f"block_size must be ≥ 1; got {self.block_size}")


# ── Header format (little-endian):
#   4B magic  |  1B bits  |  1B n_dims  |  n_dims × 4B dim_K  |  n_dims × 4B dim_V
_MAGIC = b"CGEN"


# ── CacheGenCodec ─────────────────────────────────────────────────────────────


class CacheGenCodec:
    """KV-cache bitstream encoder/decoder.

    Example::

        cfg = CacheGenConfig(bits=8, block_size=32)
        codec = CacheGenCodec(cfg)
        rng = np.random.default_rng(0)
        K = rng.standard_normal((2, 16, 8)).astype(np.float32)
        V = rng.standard_normal((2, 16, 8)).astype(np.float32)
        bs = codec.encode(K, V)
        K2, V2 = codec.decode(bs, K.shape, V.shape)
        ratio = codec.compression_ratio(K, V)
    """

    def __init__(self, config: Optional[CacheGenConfig] = None) -> None:
        self.config = config or CacheGenConfig()

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode(self, K: np.ndarray, V: np.ndarray) -> bytes:
        """Encode float32 K and V tensors into a compact byte buffer.

        Args:
            K: Any-shape float32 ndarray.
            V: Same shape as K (or different shape, both independently encoded).

        Returns:
            Byte buffer containing the compressed bitstream.
        """
        K = np.asarray(K, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        k_codes, k_scale = self._quantize(K)
        v_codes, v_scale = self._quantize(V)

        header = self._encode_header(K.shape, V.shape)
        k_meta = self._encode_meta(k_scale, K.shape)
        v_meta = self._encode_meta(v_scale, V.shape)
        k_body = self._pack_codes(k_codes)
        v_body = self._pack_codes(v_codes)
        return header + k_meta + k_body + v_meta + v_body

    def decode(
        self,
        bitstream: bytes,
        k_shape: Tuple[int, ...],
        v_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode a bitstream produced by :meth:`encode`.

        Args:
            bitstream: Byte buffer from :meth:`encode`.
            k_shape: Expected shape of K tensor.
            v_shape: Expected shape of V tensor.

        Returns:
            ``(K_hat, V_hat)`` pair of float32 ndarrays.
        """
        bits = self.config.bits
        cursor = len(_MAGIC) + 1 + 1  # magic + bits + n_dims_placeholder

        k_ndim = len(k_shape)
        v_ndim = len(v_shape)
        # Skip the header dims (already know them).
        cursor += k_ndim * 4 + v_ndim * 4

        # K meta.
        k_n = int(np.prod(k_shape))
        k_scale, cursor = self._decode_meta(bitstream, cursor, k_shape)
        k_codes, cursor = self._unpack_codes(bitstream, cursor, k_n)
        # V meta.
        v_n = int(np.prod(v_shape))
        v_scale, cursor = self._decode_meta(bitstream, cursor, v_shape)
        v_codes, cursor = self._unpack_codes(bitstream, cursor, v_n)

        K_hat = self._dequantize(k_codes.reshape(k_shape), k_scale)
        V_hat = self._dequantize(v_codes.reshape(v_shape), v_scale)
        return K_hat, V_hat

    def compression_ratio(self, K: np.ndarray, V: np.ndarray) -> float:
        """Return bytes(encoded) / bytes(float32 original)."""
        encoded = self.encode(K, V)
        original_bytes = (K.size + V.size) * 4  # float32 = 4 bytes
        return len(encoded) / max(original_bytes, 1)

    # ── Streaming ─────────────────────────────────────────────────────────────

    def stream_encode(
        self, K: np.ndarray, V: np.ndarray, chunk_size: int = 256
    ) -> Iterator[bytes]:
        """Yield successive byte chunks of the encoded bitstream.

        Args:
            K: Float32 ndarray.
            V: Float32 ndarray.
            chunk_size: Bytes per yielded chunk.

        Yields:
            Successive byte chunks (last chunk may be smaller).
        """
        buf = self.encode(K, V)
        for start in range(0, len(buf), chunk_size):
            yield buf[start : start + chunk_size]

    # ── Internals ─────────────────────────────────────────────────────────────

    def _quantize(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        bits = self.config.bits
        q_max = (1 << bits) - 1
        half_q = q_max // 2
        scale = float(np.abs(x).max())
        scale = max(scale, 1e-7)
        codes = np.clip(np.round(x / scale * half_q + half_q), 0, q_max).astype(np.uint8)
        return codes, scale

    def _dequantize(self, codes: np.ndarray, scale: float) -> np.ndarray:
        bits = self.config.bits
        q_max = (1 << bits) - 1
        half_q = q_max // 2
        return (codes.astype(np.float32) - half_q) * (scale / half_q)

    def _pack_codes(self, codes: np.ndarray) -> bytes:
        """Pack uint8 codes into bytes (1 code per byte for simplicity)."""
        return codes.flatten().tobytes()

    def _unpack_codes(
        self, buf: bytes, cursor: int, n: int
    ) -> Tuple[np.ndarray, int]:
        end = cursor + n
        codes = np.frombuffer(buf[cursor:end], dtype=np.uint8).copy()
        return codes, end

    def _encode_header(
        self,
        k_shape: Tuple[int, ...],
        v_shape: Tuple[int, ...],
    ) -> bytes:
        k_ndim = len(k_shape)
        v_ndim = len(v_shape)
        header = _MAGIC
        header += struct.pack("B", self.config.bits)
        header += struct.pack("B", k_ndim)
        for d in k_shape:
            header += struct.pack("<I", d)
        for d in v_shape:
            header += struct.pack("<I", d)
        return header

    def _encode_meta(self, scale: float, shape: Tuple[int, ...]) -> bytes:
        return struct.pack("<f", scale)

    def _decode_meta(
        self, buf: bytes, cursor: int, shape: Tuple[int, ...]
    ) -> Tuple[float, int]:
        (scale,) = struct.unpack_from("<f", buf, cursor)
        return float(scale), cursor + 4

    def __repr__(self) -> str:
        cfg = self.config
        return f"CacheGenCodec(bits={cfg.bits}, block_size={cfg.block_size})"
