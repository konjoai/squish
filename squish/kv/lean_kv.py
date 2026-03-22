"""LeanKV: asymmetric K/V cache quantization (arXiv 2407.07805, 2024).

Kang et al., 2024.  Empirical finding: the key cache tolerates lower
precision than the value cache because QK dot-products are more robust to
quantization noise than the V-weighted output sum.  K is quantized to INT4
and V to INT6–8, delivering 3× KV compression vs FP16 at < 0.3 PPL
degradation — better quality-per-byte than uniform INT4 for both tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "LeanKVConfig",
    "LeanKVState",
    "LeanKVQuant",
]


@dataclass
class LeanKVConfig:
    """Configuration for :class:`LeanKVQuant`.

    Attributes:
        k_bits: Quantization bits for the key cache (default 4, range 2–8).
        v_bits: Quantization bits for the value cache (default 8, range 2–8).
        group_size: Elements per quantization group (default 32).
            Smaller groups improve accuracy; 0 means per-tensor quantization.
        per_tensor: If True, use a single scale/zero per tensor instead of
            per-group (equivalent to group_size == tensor size).
        symmetric: If True, use symmetric quantization (zero-point = 0).
        seed: Unused; retained for API consistency.
    """

    k_bits: int = 4
    v_bits: int = 8
    group_size: int = 32
    per_tensor: bool = False
    symmetric: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        for name, bits in (("k_bits", self.k_bits), ("v_bits", self.v_bits)):
            if bits not in (2, 3, 4, 5, 6, 7, 8):
                raise ValueError(f"{name} must be in 2–8, got {bits}")
        if self.group_size < 0:
            raise ValueError(
                f"group_size must be >= 0, got {self.group_size}"
            )


@dataclass
class LeanKVState:
    """Quantized K and V tensors plus de-quantization metadata.

    Attributes:
        k_quantized: Integer K tensor (uint8 storage, logical k_bits).
        v_quantized: Integer V tensor (uint8 storage, logical v_bits).
        k_scale: Per-group (or per-tensor) scale for K.
        k_zero: Per-group (or per-tensor) zero-point for K.
        v_scale: Per-group (or per-tensor) scale for V.
        v_zero: Per-group (or per-tensor) zero-point for V.
        k_bits: Bit width used for K.
        v_bits: Bit width used for V.
        original_shape: Shape of the original K (and V) tensor.
    """

    k_quantized: np.ndarray
    v_quantized: np.ndarray
    k_scale: np.ndarray
    k_zero: np.ndarray
    v_scale: np.ndarray
    v_zero: np.ndarray
    k_bits: int
    v_bits: int
    original_shape: tuple

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def k_bytes(self) -> int:
        """Approximate storage bytes for K (logical, packed)."""
        return int(np.ceil(self.k_quantized.size * self.k_bits / 8))

    @property
    def v_bytes(self) -> int:
        """Approximate storage bytes for V (logical, packed)."""
        return int(np.ceil(self.v_quantized.size * self.v_bits / 8))

    @property
    def fp16_bytes(self) -> int:
        """Reference FP16 storage for one (K or V) tensor."""
        return self.k_quantized.size * 2

    @property
    def compression_ratio(self) -> float:
        """Total K+V bytes compared to two fp16 tensors."""
        return (2 * self.fp16_bytes) / max(self.k_bytes + self.v_bytes, 1)


class LeanKVQuant:
    """Asymmetric K/V cache quantizer.

    Usage::

        cfg = LeanKVConfig(k_bits=4, v_bits=8, group_size=32)
        lkv = LeanKVQuant(cfg)
        state = lkv.quantize_kv(k_tensor, v_tensor)
        k_rec, v_rec = lkv.dequantize_kv(state)

    """

    def __init__(self, config: LeanKVConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize_kv(
        self,
        k: np.ndarray,
        v: np.ndarray,
    ) -> LeanKVState:
        """Quantize ``k`` to ``k_bits`` and ``v`` to ``v_bits``.

        Parameters
        ----------
        k, v:
            FP32/FP16 tensors of the same shape
            (e.g. ``(n_heads, seq_len, head_dim)``).
        """
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have the same shape; got {k.shape} vs {v.shape}"
            )
        shape = k.shape
        k_q, k_sc, k_zp = self._quantize(k.reshape(-1), self.config.k_bits)
        v_q, v_sc, v_zp = self._quantize(v.reshape(-1), self.config.v_bits)
        return LeanKVState(
            k_quantized=k_q,
            v_quantized=v_q,
            k_scale=k_sc,
            k_zero=k_zp,
            v_scale=v_sc,
            v_zero=v_zp,
            k_bits=self.config.k_bits,
            v_bits=self.config.v_bits,
            original_shape=shape,
        )

    def dequantize_kv(
        self, state: LeanKVState
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct FP32 ``k`` and ``v`` from *state*."""
        k = self._dequantize(
            state.k_quantized, state.k_scale, state.k_zero, state.k_bits
        ).reshape(state.original_shape)
        v = self._dequantize(
            state.v_quantized, state.v_scale, state.v_zero, state.v_bits
        ).reshape(state.original_shape)
        return k, v

    def quantize_k(
        self, k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize only the K tensor.  Returns (k_q, scale, zero)."""
        return self._quantize(k.reshape(-1), self.config.k_bits)

    def quantize_v(
        self, v: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize only the V tensor.  Returns (v_q, scale, zero)."""
        return self._quantize(v.reshape(-1), self.config.v_bits)

    def dequantize_k(
        self,
        k_q: np.ndarray,
        scale: np.ndarray,
        zero: np.ndarray,
    ) -> np.ndarray:
        """Dequantize a flat K tensor."""
        return self._dequantize(k_q, scale, zero, self.config.k_bits)

    def dequantize_v(
        self,
        v_q: np.ndarray,
        scale: np.ndarray,
        zero: np.ndarray,
    ) -> np.ndarray:
        """Dequantize a flat V tensor."""
        return self._dequantize(v_q, scale, zero, self.config.v_bits)

    def memory_bytes(
        self,
        n_heads: int,
        seq_len: int,
        head_dim: int,
    ) -> dict:
        """Return estimated K, V, and total bytes for the given KV shapes."""
        n_elems = n_heads * seq_len * head_dim
        k_bytes = int(np.ceil(n_elems * self.config.k_bits / 8))
        v_bytes = int(np.ceil(n_elems * self.config.v_bits / 8))
        fp16_bytes = n_elems * 2
        return {
            "k_bytes": k_bytes,
            "v_bytes": v_bytes,
            "total_bytes": k_bytes + v_bytes,
            "fp16_each_bytes": fp16_bytes,
            "fp16_total_bytes": 2 * fp16_bytes,
            "compression_ratio": (2 * fp16_bytes) / max(k_bytes + v_bytes, 1),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _quantize(
        self, x: np.ndarray, bits: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize flat array *x* to *bits* bits.

        Returns ``(x_int, scale, zero_point)`` where
        ``x ≈ scale * x_int + zero_point``.
        """
        x = x.astype(np.float32)
        n = x.size
        q_max = float(2**bits - 1)

        if self.config.per_tensor or self.config.group_size == 0:
            groups = [x]
        else:
            g = self.config.group_size
            pad = (g - n % g) % g
            x_padded = np.pad(x, (0, pad), constant_values=0.0)
            groups = x_padded.reshape(-1, g)

        quant_list, scale_list, zero_list = [], [], []
        for group in groups:
            x_min = float(np.min(group))
            x_max = float(np.max(group))
            if self.config.symmetric:
                abs_max = max(abs(x_min), abs(x_max), 1e-8)
                scale = abs_max / (q_max / 2.0)
                zero = 0.0
            else:
                scale = max((x_max - x_min) / q_max, 1e-8)
                zero = x_min
            x_q = np.clip(
                np.round((np.asarray(group) - zero) / scale), 0.0, q_max
            ).astype(np.uint8)
            quant_list.append(x_q)
            scale_list.append(np.float32(scale))
            zero_list.append(np.float32(zero))

        x_q_all = np.concatenate(quant_list)[:n]
        scales = np.array(scale_list, dtype=np.float32)
        zeros = np.array(zero_list, dtype=np.float32)
        return x_q_all, scales, zeros

    def _dequantize(
        self,
        x_q: np.ndarray,
        scale: np.ndarray,
        zero: np.ndarray,
        bits: int,
    ) -> np.ndarray:
        """Reconstruct float32 values from quantized *x_q*."""
        n = x_q.size
        if scale.size == 1:
            return (x_q.astype(np.float32) * float(scale[0]) + float(zero[0]))

        g = self.config.group_size if (not self.config.per_tensor and self.config.group_size > 0) else n
        pad = (g - n % g) % g
        x_padded = np.pad(x_q.astype(np.float32), (0, pad))
        groups_q = x_padded.reshape(-1, g)
        out_groups = []
        for i, (gq, sc, zp) in enumerate(zip(groups_q, scale, zero)):
            out_groups.append(gq * float(sc) + float(zp))
        return np.concatenate(out_groups)[:n].astype(np.float32)
