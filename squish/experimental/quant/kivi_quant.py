"""
squish/quant/kivi_quant.py

KIVIQuantizer: Asymmetric INT2 KV Cache Quantization.

Reference
---------
Liu et al. "KIVI: A Tuning-Free Asymmetric 2-bit Quantization for KV Cache."
ICML 2024.

Algorithm
---------
KIVI quantises the KV cache tokens to low bit-widths (typically INT2) to
drastically reduce KV memory footprint.  The key insight is an *asymmetric
per-channel* quantisation strategy combined with keeping a small *residual
window* (the last ``residual_length`` positions) in full FP16 precision:

  1. For each channel (feature column), compute per-group min/max.
  2. Quantise using asymmetric linear mapping to ``[0, 2^bits - 1]``.
  3. The last ``residual_length`` KV positions are stored in FP32 to avoid
     compounding errors on the most recent, most-likely-to-be-attended tokens.
  4. At decode time, dequantise the compressed part and concatenate with the
     residual FP32 window.

Key properties
--------------
* ``bits`` — quantisation bits (default 2; also 4).
* ``group_size`` — tokens per scale group per channel (default 32).
* ``residual_length`` — number of recent tokens kept at full precision (default 32).
* ``per_channel`` — if True use per-channel groups; else per-tensor (default True).
* NumPy-only; no external dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class KIVIConfig:
    """Configuration for KIVIQuantizer."""

    bits: int = 2
    """Quantisation bits for the compressed KV region (2 or 4)."""

    group_size: int = 32
    """Number of tokens per quantisation group per channel."""

    residual_length: int = 32
    """Recent tokens kept in full FP32 precision (not compressed)."""

    per_channel: bool = True
    """If True, scale+zero are computed per channel; else per-tensor group."""

    def __post_init__(self) -> None:
        if self.bits not in (2, 4):
            raise ValueError(f"bits must be 2 or 4; got {self.bits}")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.residual_length < 0:
            raise ValueError("residual_length must be >= 0")

    @property
    def max_code(self) -> int:
        return 2 ** self.bits - 1


@dataclass
class KIVIStats:
    """Runtime counters for KIVIQuantizer."""

    compress_calls: int = 0
    decompress_calls: int = 0
    total_tokens_compressed: int = 0
    total_tokens_residual: int = 0

    @property
    def effective_bits(self) -> float:
        total = self.total_tokens_compressed + self.total_tokens_residual
        if total == 0:
            return 0.0
        return (
            self.total_tokens_compressed * self.config_bits +
            self.total_tokens_residual * 32
        ) / total

    def __init__(self, config_bits: int = 2) -> None:
        self.compress_calls = 0
        self.decompress_calls = 0
        self.total_tokens_compressed = 0
        self.total_tokens_residual = 0
        self.config_bits = config_bits


class KIVIQuantizer:
    """Asymmetric INT2/INT4 KV cache quantizer with FP32 residual window.

    Usage
    -----
    ::

        kivi = KIVIQuantizer()
        codes, scale, zero, residual = kivi.compress(kv_tensor)
        kv_approx = kivi.decompress(codes, scale, zero, residual)
    """

    def __init__(self, config: Optional[KIVIConfig] = None) -> None:
        self.config = config or KIVIConfig()
        self.stats = KIVIStats(config_bits=self.config.bits)

    def compress(
        self, kv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compress a KV tensor.

        Parameters
        ----------
        kv:
            Shape ``(seq_len, head_dim)`` float32 K or V tensor.

        Returns
        -------
        codes:
            Shape ``(compressed_len, head_dim)`` uint8.
        scale:
            Shape ``(n_groups, head_dim)`` float32.
        zero:
            Shape ``(n_groups, head_dim)`` float32.
        residual:
            Shape ``(residual_length, head_dim)`` float32 (last tokens).
        """
        self.stats.compress_calls += 1
        seq_len, hd = kv.shape
        cfg = self.config

        res_len = min(cfg.residual_length, seq_len)
        compressed_len = seq_len - res_len
        residual = kv[compressed_len:].astype(np.float32).copy()
        self.stats.total_tokens_residual += res_len

        if compressed_len == 0:
            empty_codes = np.empty((0, hd), dtype=np.uint8)
            empty_scale = np.empty((0, hd), dtype=np.float32)
            empty_zero = np.empty((0, hd), dtype=np.float32)
            return empty_codes, empty_scale, empty_zero, residual

        kv_c = kv[:compressed_len].astype(np.float32)
        self.stats.total_tokens_compressed += compressed_len

        gs = min(cfg.group_size, compressed_len)
        n_groups = math.ceil(compressed_len / gs)

        codes = np.empty((compressed_len, hd), dtype=np.uint8)
        scale = np.empty((n_groups, hd), dtype=np.float32)
        zero = np.empty((n_groups, hd), dtype=np.float32)
        max_code = cfg.max_code

        for g in range(n_groups):
            t0 = g * gs
            t1 = min(t0 + gs, compressed_len)
            group = kv_c[t0:t1]  # (gs_actual, hd)

            if cfg.per_channel:
                g_min = group.min(axis=0)          # (hd,)
                g_max = group.max(axis=0)
            else:
                g_min = group.min() * np.ones(hd)
                g_max = group.max() * np.ones(hd)

            s = (g_max - g_min) / max_code
            s = np.where(s < 1e-8, np.ones_like(s) * 1e-8, s)
            zp = -g_min / s

            scale[g] = s
            zero[g] = zp
            codes[t0:t1] = np.clip(
                np.round(group / s[None, :] + zp[None, :]), 0, max_code
            ).astype(np.uint8)

        return codes, scale, zero, residual

    def decompress(
        self,
        codes: np.ndarray,
        scale: np.ndarray,
        zero: np.ndarray,
        residual: np.ndarray,
    ) -> np.ndarray:
        """Decompress a KIVI-compressed KV tensor.

        Parameters
        ----------
        codes:
            Shape ``(compressed_len, head_dim)`` uint8.
        scale:
            Shape ``(n_groups, head_dim)`` float32.
        zero:
            Shape ``(n_groups, head_dim)`` float32.
        residual:
            Shape ``(residual_len, head_dim)`` float32.

        Returns
        -------
        kv_approx:
            Shape ``(compressed_len + residual_len, head_dim)`` float32.
        """
        self.stats.decompress_calls += 1
        if codes.shape[0] == 0:
            return residual.astype(np.float32)

        compressed_len, hd = codes.shape
        gs = math.ceil(compressed_len / scale.shape[0])
        n_groups = scale.shape[0]
        out = np.empty((compressed_len, hd), dtype=np.float32)

        for g in range(n_groups):
            t0 = g * gs
            t1 = min(t0 + gs, compressed_len)
            s = scale[g]
            zp = zero[g]
            out[t0:t1] = (codes[t0:t1].astype(np.float32) - zp[None, :]) * s[None, :]

        return np.concatenate([out, residual.astype(np.float32)], axis=0)
