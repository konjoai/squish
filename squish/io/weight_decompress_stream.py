"""Overlapped double-buffer weight decompression streaming pipeline.

Implements the CPU-dequantize ↔ GPU-compute overlap pattern described in
*LLM in a Flash* (Apple, 2024) and *FlexGen* (ICML 2023).  A background
thread pool decompresses quantized weight blocks while the compute kernel
consumes the previous buffer, hiding decompression latency.

Reference:
  - Alizadeh et al., "LLM in a Flash" (Apple 2024).
  - Sheng et al., "FlexGen: High-Throughput Generative Inference of LLMs"
    (ICML 2023).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np

__all__ = [
    "WeightStreamConfig",
    "WeightStreamHandle",
    "WeightDecompressStream",
]

# Supported bit-widths for compress/decompress
_VALID_BITS = frozenset({2, 3, 4, 8, 16})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class WeightStreamConfig:
    """Configuration for :class:`WeightDecompressStream`.

    Attributes:
        n_layers: Number of transformer layers in the model.
        bits: Quantization bit-width (2, 3, 4, 8, or 16).
        chunk_size: Number of layers compressed into one buffer slot.
        n_threads: Worker threads for background decompression.
        lookahead: Number of layers to prefetch ahead of current position.
        seed: RNG seed (retained for API uniformity).
    """

    n_layers: int = 32
    bits: int = 4
    chunk_size: int = 1
    n_threads: int = 2
    lookahead: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be ≥ 1, got {self.n_layers}")
        if self.bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(_VALID_BITS)}, got {self.bits}")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be ≥ 1, got {self.chunk_size}")
        if self.n_threads < 1:
            raise ValueError(f"n_threads must be ≥ 1, got {self.n_threads}")
        if self.lookahead < 0:
            raise ValueError(f"lookahead must be ≥ 0, got {self.lookahead}")


@dataclass
class WeightStreamHandle:
    """Opaque handle returned by :meth:`WeightDecompressStream.submit`.

    Attributes:
        layer_idx: Layer index this handle refers to.
        status: One of ``"pending"``, ``"ready"``, or ``"consumed"``.
    """

    layer_idx: int
    status: str = "pending"

    _future: Optional["Future[np.ndarray]"] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.status not in ("pending", "ready", "consumed"):
            raise ValueError(
                f"status must be 'pending', 'ready', or 'consumed', got {self.status!r}"
            )


# ---------------------------------------------------------------------------
# Stream processor
# ---------------------------------------------------------------------------


class WeightDecompressStream:
    """Asynchronous double-buffer weight decompression pipeline.

    Weights are submitted as quantized arrays and decompressed in a
    background :class:`~concurrent.futures.ThreadPoolExecutor`.  The
    caller fetches the decompressed float32 tensor via
    :meth:`fetch`, which blocks only if decompression has not yet
    completed.

    Example::

        cfg = WeightStreamConfig(n_layers=32, bits=4, lookahead=2)
        stream = WeightDecompressStream(cfg)

        compressed = {i: WeightDecompressStream.compress_weight(W[i], bits=4)
                      for i in range(32)}
        handles = stream.prefetch_range(list(range(32)), compressed)
        for h in handles:
            w = stream.fetch(h)   # decompressed float32
            # ... run attention with w ...

    """

    def __init__(self, config: WeightStreamConfig) -> None:
        self.config = config
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=config.n_threads,
            thread_name_prefix="wds_decomp",
        )
        self._lock = threading.Lock()
        self._n_submitted = 0
        self._n_fetched = 0
        self._total_bytes_in = 0
        self._total_bytes_out = 0

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def submit(
        self,
        layer_idx: int,
        compressed: np.ndarray,
    ) -> WeightStreamHandle:
        """Submit *compressed* for background decompression.

        Returns a :class:`WeightStreamHandle` immediately.  The
        decompressed tensor is available once :meth:`is_ready` returns
        ``True`` or after :meth:`fetch` (which blocks until ready).
        """
        bits = self.config.bits
        shape = self._recover_shape(compressed)

        with self._lock:
            self._n_submitted += 1
            self._total_bytes_in += compressed.nbytes

        future: Future[np.ndarray] = self._executor.submit(
            self.__class__.decompress_weight,
            compressed,
            bits,
            shape,
        )
        handle = WeightStreamHandle(layer_idx=layer_idx, status="pending")
        handle._future = future
        return handle

    def fetch(self, handle: WeightStreamHandle) -> np.ndarray:
        """Block until *handle* is ready, then return the float32 tensor."""
        if handle.status == "consumed":
            raise RuntimeError(
                f"Handle for layer {handle.layer_idx} already consumed."
            )
        if handle._future is None:
            raise RuntimeError(
                f"Handle for layer {handle.layer_idx} has no associated future."
            )
        result: np.ndarray = handle._future.result()  # blocks if not done
        handle.status = "consumed"
        handle._future = None

        with self._lock:
            self._n_fetched += 1
            self._total_bytes_out += result.nbytes

        return result

    def is_ready(self, handle: WeightStreamHandle) -> bool:
        """Return ``True`` if the decompressed tensor is available."""
        if handle.status == "consumed":
            return False
        if handle._future is None:
            return False
        return handle._future.done()

    def prefetch_range(
        self,
        layer_indices: List[int],
        compressed_layers: Dict[int, np.ndarray],
    ) -> List[WeightStreamHandle]:
        """Submit a batch of layers and return handles in order."""
        handles: List[WeightStreamHandle] = []
        for idx in layer_indices:
            if idx not in compressed_layers:
                raise KeyError(
                    f"Layer {idx} not in compressed_layers (available: {sorted(compressed_layers.keys())})"
                )
            handles.append(self.submit(idx, compressed_layers[idx]))
        return handles

    def stats(self) -> Dict:
        """Return runtime statistics."""
        with self._lock:
            return {
                "n_submitted": self._n_submitted,
                "n_fetched": self._n_fetched,
                "n_pending": self._n_submitted - self._n_fetched,
                "total_bytes_in": self._total_bytes_in,
                "total_bytes_out": self._total_bytes_out,
                "compression_ratio": (
                    self._total_bytes_out / max(1, self._total_bytes_in)
                ),
                "bits": self.config.bits,
                "n_threads": self.config.n_threads,
                "lookahead": self.config.lookahead,
            }

    def reset(self) -> None:
        """Reset statistics and cancel all pending work."""
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.n_threads,
            thread_name_prefix="wds_decomp",
        )
        with self._lock:
            self._n_submitted = 0
            self._n_fetched = 0
            self._total_bytes_in = 0
            self._total_bytes_out = 0

    def __del__(self) -> None:
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Static compress / decompress
    # ------------------------------------------------------------------

    @staticmethod
    def compress_weight(W: np.ndarray, bits: int) -> np.ndarray:
        """Quantize float32 weight *W* to *bits*-bit integers.

        Uses symmetric uniform quantization per-tensor.  The output is a
        1-D uint8 array whose first 4 bytes encode the float32 scale and
        whose first 4 bytes of the second word encode the original shape
        length and shape values.  The remainder is packed bit data.

        Returns a 1-D ``uint8`` array ready for :meth:`decompress_weight`.
        """
        if bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(_VALID_BITS)}, got {bits}")
        W = np.asarray(W, dtype=np.float32)
        original_shape = W.shape
        flat = W.ravel()

        if bits == 16:
            quantized = flat.astype(np.float16)
            payload = quantized.view(np.uint8)
        else:
            q_max = float((1 << bits) - 1)
            abs_max = float(np.max(np.abs(flat)))
            if abs_max == 0.0:
                abs_max = 1.0
            scale = abs_max / (q_max / 2.0)
            shifted = flat / scale + q_max / 2.0
            quantized_int = np.clip(np.round(shifted), 0, q_max).astype(np.uint8)
            payload = quantized_int.view(np.uint8)

        # --- Header encoding ---
        # Bytes 0..3:  float32 scale (0.0 for fp16 path)
        # Bytes 4..7:  uint32 n_dims
        # Bytes 8..8+4*n_dims: int32 shape dimensions
        # Remaining: payload bytes
        n_dims = len(original_shape)
        if bits == 16:
            scale_val = 0.0
        else:
            scale_val = scale  # type: ignore[assignment]

        header = np.zeros(4 + 4 + 4 * n_dims, dtype=np.uint8)
        header[:4] = np.array([scale_val], dtype=np.float32).view(np.uint8)
        header[4:8] = np.array([n_dims], dtype=np.uint32).view(np.uint8)
        for i, d in enumerate(original_shape):
            header[8 + 4 * i: 8 + 4 * (i + 1)] = np.array(
                [d], dtype=np.int32
            ).view(np.uint8)

        return np.concatenate([header, payload])

    @staticmethod
    def decompress_weight(
        data: np.ndarray,
        bits: int,
        shape: Optional[Tuple[int, ...]] = None,
    ) -> np.ndarray:
        """Dequantize a compressed weight produced by :meth:`compress_weight`.

        *shape* is inferred from the embedded header when ``None``.
        """
        if bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {sorted(_VALID_BITS)}, got {bits}")
        data = np.asarray(data, dtype=np.uint8)

        # Decode header
        (scale_val,) = np.frombuffer(data[:4], dtype=np.float32)
        (n_dims,) = np.frombuffer(data[4:8], dtype=np.uint32)
        header_size = 4 + 4 + 4 * int(n_dims)
        shape_arr = np.frombuffer(
            data[8: 8 + 4 * int(n_dims)], dtype=np.int32
        )
        inferred_shape = tuple(int(x) for x in shape_arr)
        if shape is None:
            shape = inferred_shape

        payload = data[header_size:]

        if bits == 16:
            flat = payload.view(np.float16).astype(np.float32)
        else:
            q_max = float((1 << bits) - 1)
            quantized_int = payload.astype(np.float32)
            flat = (quantized_int - q_max / 2.0) * float(scale_val)

        n_elem = 1
        for d in shape:
            n_elem *= d
        flat = flat[:n_elem]
        return flat.reshape(shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _recover_shape(compressed: np.ndarray) -> Tuple[int, ...]:
        """Decode the shape stored in the compressed array header."""
        data = np.asarray(compressed, dtype=np.uint8)
        (n_dims,) = np.frombuffer(data[4:8], dtype=np.uint32)
        shape_arr = np.frombuffer(
            data[8: 8 + 4 * int(n_dims)], dtype=np.int32
        )
        return tuple(int(x) for x in shape_arr)
