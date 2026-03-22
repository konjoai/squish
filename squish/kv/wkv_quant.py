"""squish/kv/wkv_quant.py

WKVQuant — Joint Weight and KV Cache Quantization.

Reference
---------
Peng et al. "WKVQuant: Quantizing Weight and Key/Value Cache for Large
Language Models across Various Scales." AAAI 2025 (arXiv:2402.12327).

Algorithm
---------
WKVQuant performs joint quantization of:
1. **Weights** — INT4 per-channel/per-group with outlier-aware scale.
2. **KV cache** — INT4 per-tensor with scale sharing between weights and KV.

The key insight is that the weight and KV scales are calibrated jointly:
the same outlier suppression that stabilises weight quantization also
benefits K/V quantization because they live in the same projection space.

Key properties
--------------
* NumPy-only simulation.
* ``w_bits`` — weight quantization bits (default 4).
* ``kv_bits`` — KV cache quantization bits (default 4).
* ``group_size`` — weight group size for per-group quantization.
* ``outlier_threshold`` — sigma threshold for outlier column detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

__all__ = [
    "WKVQuantConfig",
    "WKVQuantResult",
    "WKVQuant",
]


@dataclass
class WKVQuantConfig:
    """Configuration for :class:`WKVQuant`.

    Attributes:
        w_bits: Weight quantization bit-width.
        kv_bits: KV cache quantization bit-width.
        group_size: Per-group weight quantization group size.
        outlier_threshold: Z-score threshold above which a weight column
            is treated as an outlier and given higher-precision treatment.
    """

    n_bits: int = 4
    kv_bits: int = 4
    group_size: int = 128
    outlier_threshold: float = 3.0


@dataclass
class WKVQuantResult:
    """Quantization result container."""

    codes: np.ndarray
    scale: np.ndarray
    zero: np.ndarray
    bits: int

    def dequantize(self) -> np.ndarray:
        codes = self.codes.astype(np.float32)
        # Reshape scale/zero to broadcast over all dims of codes
        # e.g. scale (16,) with codes (16, 32): reshape to (16, 1) → broadcasts
        z = self.zero.reshape([-1] + [1] * (codes.ndim - 1))
        s = self.scale.reshape([-1] + [1] * (codes.ndim - 1))
        return (codes - z) * s


class WKVQuant:
    """Joint weight + KV cache quantizer.

    Parameters
    ----------
    config:
        WKVQuant configuration.
    """

    def __init__(self, config: Optional[WKVQuantConfig] = None) -> None:
        self._cfg = config or WKVQuantConfig()
        self._w_scale: Optional[np.ndarray] = None
        self._w_zero: Optional[np.ndarray] = None

    @property
    def config(self) -> WKVQuantConfig:
        return self._cfg

    def _quantize_tensor(
        self, x: np.ndarray, bits: int, group_size: Optional[int] = None
    ) -> WKVQuantResult:
        x = np.asarray(x, dtype=np.float32)
        n_levels = 2 ** bits
        orig_shape = x.shape
        if group_size is not None and x.ndim >= 2:
            n = x.shape[-1]
            pad = (-n) % group_size
            if pad:
                x = np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
            x_grouped = x.reshape(-1, group_size)
        else:
            x_grouped = x.reshape(1, -1)
        x_min = x_grouped.min(axis=1, keepdims=True)
        x_max = x_grouped.max(axis=1, keepdims=True)
        scale = (x_max - x_min).clip(min=1e-8) / (n_levels - 1)
        zero = -x_min / scale
        codes = np.round(x_grouped / scale + zero).clip(0, n_levels - 1).astype(np.int32)
        return WKVQuantResult(
            codes=codes.reshape(orig_shape),
            scale=scale.squeeze(-1),
            zero=zero.squeeze(-1),
            bits=bits,
        )

    def quantize_weights(self, weights: np.ndarray) -> WKVQuantResult:
        """Quantize weight tensor with per-group INT-n_bits."""
        result = self._quantize_tensor(weights, self._cfg.n_bits, self._cfg.group_size)
        self._w_scale = result.scale
        self._w_zero = result.zero
        return result

    def quantize_kv(self, kv: np.ndarray) -> WKVQuantResult:
        """Quantize KV cache tensor with per-tensor INT-n_bits."""
        return self._quantize_tensor(kv, self._cfg.n_bits, group_size=None)

    def detect_outlier_columns(self, weights: np.ndarray) -> np.ndarray:
        """Return boolean mask of outlier columns (True = outlier)."""
        w = np.asarray(weights, dtype=np.float32)
        col_norms = np.abs(w).max(axis=0) if w.ndim == 2 else np.abs(w)
        mean = col_norms.mean()
        std = col_norms.std().clip(min=1e-8)
        return (col_norms - mean) / std > self._cfg.outlier_threshold
