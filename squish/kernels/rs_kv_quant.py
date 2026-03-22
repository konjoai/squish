"""Rust-backed KV-cache INT8 quantization kernel.

Wraps ``squish_quant.{quantize,dequantize}_kv_heads_int8`` from the maturin-
compiled Rust extension.  Falls back to a pure-NumPy implementation when the
extension is unavailable.

KV cache arrays have layout ``(n_heads, n_seq, head_dim)``.  Scales are
computed per-head to preserve relative magnitudes across the sequence dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

__all__ = ["KVQuantKernelConfig", "RustKVQuantKernel"]

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class KVQuantKernelConfig:
    """Configuration for :class:`RustKVQuantKernel`.

    Attributes
    ----------
    bits:
        Quantization bit-width.  Only 8 is currently supported.
    group_by_head:
        When ``True`` (default), one scale per head is used.  Per-head
        quantization preserves relative token importances within each head.
    """

    bits: int = 8
    group_by_head: bool = True


class RustKVQuantKernel:
    """KV-cache quantization kernel (INT8, per-head scale) backed by Rust.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`KVQuantKernelConfig`.
    """

    def __init__(self, config: KVQuantKernelConfig | None = None) -> None:
        self.config = config or KVQuantKernelConfig()
        if self.config.bits != 8:
            raise ValueError("Only bits=8 is supported for KV quantization.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def quantize_kv(
        self, kv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize a KV cache tensor to INT8.

        Parameters
        ----------
        kv:
            float32 array of shape ``(n_heads, n_seq, head_dim)``.

        Returns
        -------
        kv_q:
            int8 array of shape ``(n_heads, n_seq, head_dim)``.
        scales:
            float32 array of shape ``(n_heads,)`` — per-head scale factors.
        """
        if kv.ndim != 3:
            raise ValueError(f"kv must be 3-D (n_heads, n_seq, head_dim), got {kv.ndim}-D")

        kv32 = kv.astype(np.float32, copy=False)

        if _RUST_AVAILABLE:
            try:
                q, s = _sq.quantize_kv_heads_int8(kv32)
                return q, s
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_quantize(kv32)

    def dequantize_kv(
        self, kv_q: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        """Dequantize INT8 KV cache back to float32.

        Parameters
        ----------
        kv_q:
            int8 array of shape ``(n_heads, n_seq, head_dim)``.
        scales:
            float32 array of shape ``(n_heads,)``.

        Returns
        -------
        float32 array of shape ``(n_heads, n_seq, head_dim)``.
        """
        if _RUST_AVAILABLE:
            try:
                return _sq.dequantize_kv_heads_int8(kv_q, scales)
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_dequantize(kv_q, scales)

    def decode_step_update(
        self,
        kv_cache: np.ndarray,
        new_kv: np.ndarray,
        step: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Append a new KV slice at position *step* and re-quantize.

        Parameters
        ----------
        kv_cache:
            float32 KV cache of shape ``(n_heads, max_seq, head_dim)``.
        new_kv:
            float32 new KV slice of shape ``(n_heads, 1, head_dim)``.
        step:
            Current sequence position (0-indexed).

        Returns
        -------
        kv_q:
            Updated int8 cache of shape ``(n_heads, step + 1, head_dim)``.
        scales:
            float32 per-head scales of shape ``(n_heads,)``.
        """
        kv_cache[:, step : step + 1, :] = new_kv
        valid_cache = kv_cache[:, : step + 1, :]
        return self.quantize_kv(valid_cache)

    # ------------------------------------------------------------------ #
    #  NumPy fallback implementations                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _numpy_quantize(
        kv: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_heads, n_seq, head_dim = kv.shape
        scales = np.zeros(n_heads, dtype=np.float32)
        out = np.zeros((n_heads, n_seq, head_dim), dtype=np.int8)

        for h in range(n_heads):
            abs_max = np.abs(kv[h]).max()
            scale = 1.0 if abs_max == 0 else abs_max / 127.0
            scales[h] = scale
            out[h] = np.clip(np.round(kv[h] / scale), -127, 127).astype(np.int8)

        return out, scales

    @staticmethod
    def _numpy_dequantize(
        kv_q: np.ndarray, scales: np.ndarray
    ) -> np.ndarray:
        n_heads = kv_q.shape[0]
        out = kv_q.astype(np.float32)
        for h in range(n_heads):
            out[h] *= scales[h]
        return out
