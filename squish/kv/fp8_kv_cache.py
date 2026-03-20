"""squish/kv/fp8_kv_cache.py

FP8KVCache — Per-Tensor FP8 Quantized Key/Value Cache.

Reference
---------
TRT-LLM / FlashInfer production FP8 KV cache (2024).

Algorithm
---------
Storing K and V tensors in FP8 instead of FP16/BF16 halves KV memory:

* **FP8 e4m3** (4-bit exponent, 3-bit mantissa): max abs value ≈ 448.
  Better accuracy; used for weights and KV by TRT-LLM.
* **FP8 e5m2** (5-bit exponent, 2-bit mantissa): max abs value ≈ 57344.
  Larger dynamic range; occasionally preferred for KV caches.

This module simulates FP8 via per-tensor quantization: each K or V tensor is
scaled to fit within the FP8 representable range, stored as INT8, and
dequantized on-the-fly before attention.

Key properties
--------------
* NumPy-only simulation (INT8 storage + float32 scale).
* ``dtype`` — ``"e4m3"`` or ``"e5m2"`` (determines max representable value).
* ``per_tensor`` — one scale per tensor (True) or per-head (False).
* Zero calibration required; scale computed dynamically at store time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "FP8KVConfig",
    "FP8KVTensor",
    "FP8KVCache",
]

# ── Config ────────────────────────────────────────────────────────────────────

_FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


@dataclass
class FP8KVConfig:
    """Configuration for :class:`FP8KVCache`.

    Attributes:
        dtype: FP8 format — ``"e4m3"`` or ``"e5m2"``.
        per_tensor: If True, one scale per tensor; if False, one scale per head.
        epsilon: Small value to avoid division by zero in scale computation.
    """

    dtype: str = "e4m3"
    per_tensor: bool = True
    epsilon: float = 1e-7

    def __post_init__(self) -> None:
        if self.dtype not in ("e4m3", "e5m2"):
            raise ValueError(
                f"dtype must be 'e4m3' or 'e5m2'; got '{self.dtype}'"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be > 0; got {self.epsilon}")

    @property
    def fp8_max(self) -> float:
        return _FP8_MAX[self.dtype]


# ── Data container ─────────────────────────────────────────────────────────────


@dataclass
class FP8KVTensor:
    """A quantized FP8 K or V tensor.

    Attributes:
        codes: INT8 quantized codes, same shape as original tensor.
        scale: Per-tensor or per-head float32 scale(s).
        shape: Original tensor shape.
        config: The FP8KVConfig used for quantization.
    """

    codes: np.ndarray     # int8, shape = original
    scale: np.ndarray     # float32, shape (1,) or (n_heads,)
    shape: Tuple[int, ...]
    config: FP8KVConfig


# ── Core class ─────────────────────────────────────────────────────────────────


class FP8KVCache:
    """FP8-quantized key/value cache with dynamic per-tensor scales.

    Example::

        cfg   = FP8KVConfig(dtype="e4m3")
        cache = FP8KVCache(cfg)

        K = np.random.randn(4, 128, 16).astype(np.float32)
        V = np.random.randn(4, 128, 16).astype(np.float32)

        qK = cache.quantize(K)
        qV = cache.quantize(V)

        K_out = cache.dequantize(qK)   # ≈ K
        V_out = cache.dequantize(qV)   # ≈ V
    """

    def __init__(self, config: Optional[FP8KVConfig] = None) -> None:
        self.config = config or FP8KVConfig()
        self._store: Dict[str, Tuple[FP8KVTensor, FP8KVTensor]] = {}

    # ── Quantize ──────────────────────────────────────────────────────────────

    def quantize(self, x: np.ndarray) -> FP8KVTensor:
        """Quantize a float32 tensor to FP8.

        Args:
            x: Float32 tensor of any shape.  Convention: ``(n_heads, S, head_dim)``.

        Returns:
            :class:`FP8KVTensor` with INT8 codes and float32 scale(s).
        """
        x = np.asarray(x, dtype=np.float32)
        cfg = self.config
        fp8_max = cfg.fp8_max
        eps = cfg.epsilon

        if cfg.per_tensor:
            amax = float(np.abs(x).max()) + eps
            scale = np.float32(fp8_max / amax)
            codes = np.clip(np.round(x * scale), -127, 127).astype(np.int8)
            scale_arr = np.array([scale], dtype=np.float32)
        else:
            # Per-head: x is (H, S, d) — scale over last two dims
            H = x.shape[0]
            scale_arr = np.zeros(H, dtype=np.float32)
            codes = np.zeros_like(x, dtype=np.int8)
            for h in range(H):
                amax_h = float(np.abs(x[h]).max()) + eps
                s = np.float32(fp8_max / amax_h)
                scale_arr[h] = s
                codes[h] = np.clip(np.round(x[h] * s), -127, 127).astype(np.int8)

        return FP8KVTensor(codes=codes, scale=scale_arr, shape=x.shape, config=cfg)

    # ── Dequantize ────────────────────────────────────────────────────────────

    def dequantize(self, qt: FP8KVTensor) -> np.ndarray:
        """Dequantize an FP8 tensor back to float32.

        Args:
            qt: :class:`FP8KVTensor` produced by :meth:`quantize`.

        Returns:
            Float32 tensor of the original shape.
        """
        codes = qt.codes.astype(np.float32)
        if qt.config.per_tensor:
            scale = float(qt.scale[0])
            return codes / scale
        else:
            # Per-head
            out = np.zeros_like(codes)
            for h in range(codes.shape[0]):
                out[h] = codes[h] / float(qt.scale[h])
            return out

    # ── Store / load ──────────────────────────────────────────────────────────

    def store(self, layer_id: int, K: np.ndarray, V: np.ndarray) -> None:
        """Quantize and cache K/V for a layer.

        Args:
            layer_id: Integer layer index.
            K: ``(n_heads, S, head_dim)`` key tensor.
            V: ``(n_heads, S, head_dim)`` value tensor.
        """
        self._store[str(layer_id)] = (self.quantize(K), self.quantize(V))

    def load(self, layer_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Dequantize and return K/V for a layer.

        Args:
            layer_id: Integer layer index.

        Returns:
            ``(K, V)`` as float32 tensors.

        Raises:
            KeyError: If the layer has not been stored.
        """
        key = str(layer_id)
        if key not in self._store:
            raise KeyError(f"Layer {layer_id} not in FP8KVCache")
        qK, qV = self._store[key]
        return self.dequantize(qK), self.dequantize(qV)

    def relative_error(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute relative reconstruction error."""
        denom = float(np.abs(original).mean()) + 1e-9
        return float(np.abs(original - reconstructed).mean()) / denom

    def memory_bytes(self) -> int:
        """Total bytes used by stored FP8 codes."""
        total = 0
        for qK, qV in self._store.values():
            total += qK.codes.nbytes + qV.codes.nbytes
        return total

    def n_layers_cached(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:
        return (
            f"FP8KVCache(dtype={self.config.dtype}, "
            f"per_tensor={self.config.per_tensor}, "
            f"n_layers={self.n_layers_cached()})"
        )
