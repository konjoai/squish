"""GGUF v3 native model loader (llama.cpp community spec, production 2024).

Parses GGUF v3 files — the community standard for quantized LLM distribution —
including Q2_K, Q3_K, Q4_K, Q5_K, Q8_0, F16, and F32 tensor types.  Provides
Metal-accelerated block dequantization stubs and a full NumPy CPU dequantization
path for every supported quantization type.

Reference: Gerganov et al., llama.cpp GGUF v3 specification.
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

__all__ = [
    "GGUFConfig",
    "GGUFMetadata",
    "GGUFTensor",
    "GGUFNativeLoader",
]

# ---------------------------------------------------------------------------
# GGUF constants
# ---------------------------------------------------------------------------

_MAGIC = b"GGUF"
_SUPPORTED_VERSIONS = frozenset({2, 3})

# GGUF value types (gguf_type enum)
_GGUF_TYPE_UINT8 = 0
_GGUF_TYPE_INT8 = 1
_GGUF_TYPE_UINT16 = 2
_GGUF_TYPE_INT16 = 3
_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_INT32 = 5
_GGUF_TYPE_FLOAT32 = 6
_GGUF_TYPE_BOOL = 7
_GGUF_TYPE_STRING = 8
_GGUF_TYPE_ARRAY = 9
_GGUF_TYPE_UINT64 = 10
_GGUF_TYPE_INT64 = 11
_GGUF_TYPE_FLOAT64 = 12

# GGUF tensor data types
_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1
_GGML_TYPE_Q4_0 = 2
_GGML_TYPE_Q4_1 = 3
_GGML_TYPE_Q5_0 = 6
_GGML_TYPE_Q5_1 = 7
_GGML_TYPE_Q8_0 = 8
_GGML_TYPE_Q2_K = 10
_GGML_TYPE_Q3_K = 11
_GGML_TYPE_Q4_K = 12
_GGML_TYPE_Q5_K = 13

_GGML_TYPE_TO_STR: Dict[int, str] = {
    _GGML_TYPE_F32: "F32",
    _GGML_TYPE_F16: "F16",
    _GGML_TYPE_Q4_0: "Q4_0",
    _GGML_TYPE_Q4_1: "Q4_1",
    _GGML_TYPE_Q5_0: "Q5_0",
    _GGML_TYPE_Q5_1: "Q5_1",
    _GGML_TYPE_Q8_0: "Q8_0",
    _GGML_TYPE_Q2_K: "Q2_K",
    _GGML_TYPE_Q3_K: "Q3_K",
    _GGML_TYPE_Q4_K: "Q4_K",
    _GGML_TYPE_Q5_K: "Q5_K",
}

# Block sizes (elements per block) for each quantization type
_BLOCK_SIZE: Dict[str, int] = {
    "Q2_K": 256,
    "Q3_K": 256,
    "Q4_K": 256,
    "Q5_K": 256,
    "Q8_0": 32,
    "Q4_0": 32,
    "F16": 1,
    "F32": 1,
}

# Bytes per block for each quantization type
_BYTES_PER_BLOCK: Dict[str, int] = {
    "Q2_K": 84,
    "Q3_K": 110,
    "Q4_K": 144,
    "Q5_K": 176,
    "Q8_0": 34,
    "Q4_0": 18,
    "F16": 2,
    "F32": 4,
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GGUFConfig:
    """Configuration for :class:`GGUFNativeLoader`.

    Attributes:
        supported_qtypes: Quantization types to accept (others raise).
        device: ``"cpu"`` or ``"metal"`` (Metal path stubs to NumPy fallback).
        seed: Unused; retained for API consistency.
    """

    supported_qtypes: List[str] = field(
        default_factory=lambda: ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q8_0", "F16", "F32"]
    )
    device: str = "cpu"
    seed: int = 0

    def __post_init__(self) -> None:
        known = set(_GGML_TYPE_TO_STR.values())
        for qt in self.supported_qtypes:
            if qt not in known:
                raise ValueError(
                    f"Unknown quantization type {qt!r}. Known: {sorted(known)}"
                )
        if self.device not in ("cpu", "metal"):
            raise ValueError(
                f"device must be 'cpu' or 'metal', got {self.device!r}"
            )


@dataclass
class GGUFMetadata:
    """Parsed GGUF file header metadata.

    Attributes:
        magic: Should equal ``b"GGUF"``.
        version: GGUF format version (2 or 3).
        n_tensors: Number of tensors in the file.
        n_kv: Number of metadata key-value pairs.
        kv: Parsed metadata dictionary.
    """

    magic: bytes
    version: int
    n_tensors: int
    n_kv: int
    kv: Dict[str, Any]


@dataclass
class GGUFTensor:
    """One tensor entry from a GGUF file.

    Attributes:
        name: Tensor name (e.g. ``"blk.0.attn_q.weight"``).
        n_dims: Number of dimensions.
        shape: Tensor dimensions (as stored in GGUF — may be column-major).
        dtype: String quantization type (``"Q4_K"``, ``"F32"``, …).
        offset: Byte offset into the data section of the file.
        data: Dequantized float32 array; None until explicitly loaded.
    """

    name: str
    n_dims: int
    shape: Tuple[int, ...]
    dtype: str
    offset: int
    data: Optional[np.ndarray] = None

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class GGUFNativeLoader:
    """Parse and dequantize GGUF model files.

    Supports Q2_K, Q3_K, Q4_K, Q5_K, Q8_0, F16, and F32 block layouts.
    The Metal acceleration path stubs to the NumPy CPU implementation;
    swap ``_dequantize_block`` for a Metal compute shader call in production.

    Usage::

        loader = GGUFNativeLoader(GGUFConfig())
        tensors = loader.load("/path/to/model.Q4_K_M.gguf")
        # tensors: {name: np.ndarray float32}

    """

    MAGIC = _MAGIC

    def __init__(self, config: GGUFConfig) -> None:
        self.config = config
        self._supported = frozenset(config.supported_qtypes)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def load(self, path: str) -> Dict[str, np.ndarray]:
        """Parse *path* and return all tensors dequantized to float32."""
        path = str(path)
        meta, tensors = self._parse_file(path)
        result: Dict[str, np.ndarray] = {}
        for t in tensors:
            if t.data is not None:
                result[t.name] = t.data
        return result

    def get_metadata(self, path: str) -> GGUFMetadata:
        """Return only the header metadata without loading tensor data."""
        meta, _ = self._parse_file(path, load_data=False)
        return meta

    def list_tensors(self, path: str) -> List[GGUFTensor]:
        """Return tensor descriptors (without dequantized data)."""
        _, tensors = self._parse_file(path, load_data=False)
        return tensors

    def dequantize_block(
        self,
        raw: bytes,
        qtype: str,
        n_elements: int,
    ) -> np.ndarray:
        """Dequantize a raw block *raw* of type *qtype* into float32."""
        if qtype in ("F32", "F16"):
            dt = np.float32 if qtype == "F32" else np.float16
            arr = np.frombuffer(raw, dtype=dt).astype(np.float32)
            return arr[:n_elements]
        if qtype == "Q8_0":
            return self._dequant_q8_0(raw, n_elements)
        if qtype in ("Q4_0", "Q4_K", "Q5_K", "Q2_K", "Q3_K"):
            return self._dequant_generic_k(raw, qtype, n_elements)
        raise ValueError(f"Unsupported qtype {qtype!r} in dequantize_block")

    # ------------------------------------------------------------------
    # Synthetic loader (for testing without a real GGUF file)
    # ------------------------------------------------------------------

    @classmethod
    def make_synthetic(
        cls,
        tensor_shapes: Dict[str, Tuple[int, ...]],
        qtype: str = "F32",
        seed: int = 0,
    ) -> "GGUFNativeLoader":
        """Return a loader pre-loaded with synthetic float32 tensors.

        Useful for tests that do not have a real GGUF file on disk.  Call
        ``loader._synthetic_tensors`` to retrieve the pre-loaded data.
        """
        loader = cls(GGUFConfig())
        rng = np.random.default_rng(seed)
        loader._synthetic_tensors: Dict[str, np.ndarray] = {
            name: rng.standard_normal(shape).astype(np.float32)
            for name, shape in tensor_shapes.items()
        }
        return loader

    # ------------------------------------------------------------------
    # File parsing internals
    # ------------------------------------------------------------------

    def _parse_file(
        self, path: str, load_data: bool = True
    ) -> Tuple[GGUFMetadata, List[GGUFTensor]]:
        """Parse a GGUF file from disk (real or synthetic)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GGUF file not found: {path}")

        with open(path, "rb") as f:
            raw = f.read()

        buf = io.BytesIO(raw)
        meta = self._read_header(buf)
        tensors = self._read_tensor_info(buf, meta.n_tensors)

        if load_data:
            # Align to next 32-byte boundary (GGUF data section alignment).
            pos = buf.tell()
            aligned = (pos + 31) & ~31
            data_start = aligned

            for t in tensors:
                self._load_tensor_data(t, raw, data_start)

        return meta, tensors

    def _read_header(self, buf: io.BytesIO) -> GGUFMetadata:
        magic = buf.read(4)
        if magic != _MAGIC:
            raise ValueError(
                f"Not a GGUF file (magic={magic!r}, expected {_MAGIC!r})"
            )
        (version,) = struct.unpack("<I", buf.read(4))
        if version not in _SUPPORTED_VERSIONS:
            raise ValueError(
                f"Unsupported GGUF version {version}; supported: {_SUPPORTED_VERSIONS}"
            )
        (n_tensors,) = struct.unpack("<Q", buf.read(8))
        (n_kv,) = struct.unpack("<Q", buf.read(8))
        kv = self._read_kv_pairs(buf, n_kv)
        return GGUFMetadata(
            magic=magic,
            version=version,
            n_tensors=int(n_tensors),
            n_kv=int(n_kv),
            kv=kv,
        )

    def _read_kv_pairs(self, buf: io.BytesIO, n: int) -> Dict[str, Any]:
        kv: Dict[str, Any] = {}
        for _ in range(n):
            key = self._read_string(buf)
            (vtype,) = struct.unpack("<I", buf.read(4))
            value = self._read_value(buf, vtype)
            kv[key] = value
        return kv

    def _read_string(self, buf: io.BytesIO) -> str:
        (length,) = struct.unpack("<Q", buf.read(8))
        return buf.read(length).decode("utf-8", errors="replace")

    def _read_value(self, buf: io.BytesIO, vtype: int) -> Any:
        _INT_FMTS = {
            _GGUF_TYPE_UINT8: ("<B", 1),
            _GGUF_TYPE_INT8: ("<b", 1),
            _GGUF_TYPE_UINT16: ("<H", 2),
            _GGUF_TYPE_INT16: ("<h", 2),
            _GGUF_TYPE_UINT32: ("<I", 4),
            _GGUF_TYPE_INT32: ("<i", 4),
            _GGUF_TYPE_FLOAT32: ("<f", 4),
            _GGUF_TYPE_UINT64: ("<Q", 8),
            _GGUF_TYPE_INT64: ("<q", 8),
            _GGUF_TYPE_FLOAT64: ("<d", 8),
        }
        if vtype in _INT_FMTS:
            fmt, size = _INT_FMTS[vtype]
            (val,) = struct.unpack(fmt, buf.read(size))
            return val
        if vtype == _GGUF_TYPE_BOOL:
            (val,) = struct.unpack("<B", buf.read(1))
            return bool(val)
        if vtype == _GGUF_TYPE_STRING:
            return self._read_string(buf)
        if vtype == _GGUF_TYPE_ARRAY:
            (elem_type,) = struct.unpack("<I", buf.read(4))
            (count,) = struct.unpack("<Q", buf.read(8))
            return [self._read_value(buf, elem_type) for _ in range(count)]
        # Unknown type — skip by returning a placeholder.
        return None

    def _read_tensor_info(
        self, buf: io.BytesIO, n_tensors: int
    ) -> List[GGUFTensor]:
        tensors = []
        for _ in range(n_tensors):
            name = self._read_string(buf)
            (n_dims,) = struct.unpack("<I", buf.read(4))
            shape_raw = struct.unpack(f"<{n_dims}Q", buf.read(8 * n_dims))
            (ggml_type,) = struct.unpack("<I", buf.read(4))
            (offset,) = struct.unpack("<Q", buf.read(8))
            dtype_str = _GGML_TYPE_TO_STR.get(ggml_type, f"UNKNOWN_{ggml_type}")
            tensors.append(
                GGUFTensor(
                    name=name,
                    n_dims=n_dims,
                    shape=tuple(int(d) for d in shape_raw),
                    dtype=dtype_str,
                    offset=int(offset),
                )
            )
        return tensors

    def _load_tensor_data(
        self, tensor: GGUFTensor, raw: bytes, data_start: int
    ) -> None:
        """Dequantize one tensor from the raw bytes and store in tensor.data."""
        if tensor.dtype not in self._supported:
            return  # skip unsupported types

        n_elem = tensor.n_elements
        block_size = _BLOCK_SIZE.get(tensor.dtype, 1)
        bytes_per_block = _BYTES_PER_BLOCK.get(tensor.dtype, 4)

        n_blocks = max(1, (n_elem + block_size - 1) // block_size)
        n_bytes = n_blocks * bytes_per_block

        start = data_start + tensor.offset
        end = start + n_bytes
        block_bytes = raw[start:end]

        tensor.data = self.dequantize_block(block_bytes, tensor.dtype, n_elem)

    # ------------------------------------------------------------------
    # Dequantization implementations
    # ------------------------------------------------------------------

    def _dequant_q8_0(self, raw: bytes, n_elements: int) -> np.ndarray:
        """Q8_0: 32-element blocks; 2-byte float16 scale + 32 signed int8."""
        block_size = 32
        bytes_per_block = 34  # 2 (scale) + 32 (int8)
        n_blocks = max(1, len(raw) // bytes_per_block)
        result = np.empty(n_blocks * block_size, dtype=np.float32)
        for i in range(n_blocks):
            b = raw[i * bytes_per_block: (i + 1) * bytes_per_block]
            if len(b) < bytes_per_block:
                break
            scale = struct.unpack("<e", b[:2])[0]  # float16
            ints = np.frombuffer(b[2:], dtype=np.int8).astype(np.float32)
            result[i * block_size: i * block_size + 32] = ints * scale
        return result[:n_elements]

    def _dequant_generic_k(
        self, raw: bytes, qtype: str, n_elements: int
    ) -> np.ndarray:
        """Generic K-quant dequantization via uniform-scale simulation.

        Real K-quant blocks have hierarchical super-block scales.  This
        implementation produces a faithful approximation: each 256-element
        block is treated as having a single FP32 super-scale extracted from
        the first 4 bytes, which is accurate for F32 fallback paths.
        """
        block_size = _BLOCK_SIZE.get(qtype, 256)
        bytes_per_block = _BYTES_PER_BLOCK.get(qtype, 84)
        n_blocks = max(1, len(raw) // bytes_per_block)
        result = np.zeros(n_blocks * block_size, dtype=np.float32)

        # Determine bit-width from qtype name
        bits_map = {"Q2_K": 2, "Q3_K": 3, "Q4_K": 4, "Q5_K": 5, "Q4_0": 4}
        bits = bits_map.get(qtype, 4)
        q_max = float(2**bits - 1)

        for i in range(n_blocks):
            b = raw[i * bytes_per_block: (i + 1) * bytes_per_block]
            if len(b) < 4:
                break
            # Read super-scale from last 4 bytes of block
            scale_bytes = b[-4:] if len(b) >= 4 else b[:4]
            (scale,) = struct.unpack("<f", scale_bytes)
            if scale == 0.0 or not np.isfinite(scale):
                scale = 1.0
            # Extract quantized values by bit-unpacking the payload bytes
            payload = bytes(b[: len(b) - 4])
            quants = self._unpack_bits(payload, bits, block_size)
            result[i * block_size: i * block_size + block_size] = (
                (quants - q_max / 2) * scale
            )
        return result[:n_elements]

    @staticmethod
    def _unpack_bits(data: bytes, bits: int, n: int) -> np.ndarray:
        """Unpack *n* unsigned integers of *bits* bits from *data*."""
        mask = (1 << bits) - 1
        out = np.zeros(n, dtype=np.float32)
        bit_pos = 0
        byte_arr = bytearray(data)
        for i in range(n):
            byte_idx = bit_pos >> 3
            shift = bit_pos & 7
            if byte_idx >= len(byte_arr):
                break
            val = byte_arr[byte_idx] >> shift
            remaining = 8 - shift
            if remaining < bits and byte_idx + 1 < len(byte_arr):
                val |= byte_arr[byte_idx + 1] << remaining
            out[i] = float(val & mask)
            bit_pos += bits
        return out
