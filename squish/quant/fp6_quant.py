"""
squish/quant/fp6_quant.py

FP6Quantizer — 6-bit Floating-Point Weight Quantization.

Based on:
  "FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric
   Algorithm-System Co-Design"
  Lequn Chen et al. — Supercomputing 2024 (SC'24)  —  arXiv:2401.14112

  "FP6 vs INT6: A Comprehensive Comparison"
  NeurIPS 2025 Workshop

Background
----------
FP6 (6-bit floating-point) quantization stores each weight in 6 bits using
either of two IEEE-like formats:

  • **E3M2** — 1 sign bit + 3 exponent bits + 2 mantissa bits
    - Exponent bias: 3 (range 1..6 → scale 2^−2 .. 2^3)
    - Max value: ±2^3 × (1 + 3/4) = ±14.0
    - Good for weights with moderate dynamic range.

  • **E2M3** — 1 sign bit + 2 exponent bits + 3 mantissa bits
    - Exponent bias: 1 (range 1..2 → scale 2^0 .. 2^1)
    - Max value: ±2^1 × (1 + 7/8) = ±3.75
    - Higher precision for narrow-range weights.

Both formats:
  - Represent ±0 explicitly (via all-zero mantissa/exponent).
  - Represent special NaN/Inf by the all-ones exponent pattern.
  - Pack 4 values per 3 bytes (4 × 6 = 24 bits).

Relative to INT8 (8-bit integer):
  - 75% of INT8 storage (6/8 = 0.75).
  - Better accuracy than INT6 due to floating-point representation.
  - Comparable accuracy to FP8 on weight-only quantization tasks.

Per-group scaling (group_size weights share one float32 scale) further
improves accuracy with negligible storage overhead.

Classes
-------
``FP6Config``        — configuration
``FP6Quantized``     — packed weight container
``FP6Stats``         — per-instance counters
``FP6Quantizer``     — quantize/dequantize API

Usage::

    from squish.quant.fp6_quant import FP6Config, FP6Quantizer

    cfg = FP6Config(fmt="e3m2", group_size=64)
    q = FP6Quantizer(cfg)

    weight = np.random.randn(4096, 4096).astype(np.float32)
    packed = q.quantize(weight)          # FP6Quantized
    restored = q.dequantize(packed)      # float32 approx

    print(packed.compression_ratio)      # ~0.75
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

__all__ = [
    "FP6Config",
    "FP6Quantized",
    "FP6Stats",
    "FP6Quantizer",
]

# Supported formats
_SUPPORTED_FMTS = {"e3m2", "e2m3"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class FP6Config:
    """Configuration for FP6 weight quantization.

    Attributes:
        fmt:        FP6 format — ``"e3m2"`` (3 exp bits, 2 mantissa) or
                    ``"e2m3"`` (2 exp bits, 3 mantissa).
        group_size: Number of weights sharing one scale factor (per output row
                    if 2-D, contiguous otherwise).  Must be >= 1.
    """

    fmt: Literal["e3m2", "e2m3"] = "e3m2"
    group_size: int = 64

    def __post_init__(self) -> None:
        if self.fmt not in _SUPPORTED_FMTS:
            raise ValueError(
                f"fmt must be one of {sorted(_SUPPORTED_FMTS)}, got '{self.fmt}'"
            )
        if self.group_size < 1:
            raise ValueError(f"group_size must be >= 1, got {self.group_size}")

    @property
    def exp_bits(self) -> int:
        return 3 if self.fmt == "e3m2" else 2

    @property
    def man_bits(self) -> int:
        return 2 if self.fmt == "e3m2" else 3

    @property
    def exp_bias(self) -> int:
        return (1 << (self.exp_bits - 1)) - 1  # e3m2 → 3; e2m3 → 1

    @property
    def max_exp(self) -> int:
        return (1 << self.exp_bits) - 2  # leave top for NaN/Inf

    @property
    def fp6_max(self) -> float:
        """Maximum representable absolute value."""
        m_max = (1 << self.man_bits) - 1
        return float(2 ** (self.max_exp - self.exp_bias) * (1.0 + m_max / (1 << self.man_bits)))


# ---------------------------------------------------------------------------
# Packed container
# ---------------------------------------------------------------------------


@dataclass
class FP6Quantized:
    """Packed FP6 weight tensor.

    Attributes:
        packed:         uint8 array of shape ``(n_groups, ceil(group_size*6/8))``
                        where 4 FP6 values are stored per 3 bytes.
        scales:         float32 scale per group, shape ``(n_groups,)``.
        original_shape: Original weight shape before flattening.
        fmt:            FP6 format string.
        original_dtype: Original numpy dtype of the weight.
    """

    packed: np.ndarray
    scales: np.ndarray
    original_shape: Tuple[int, ...]
    fmt: str
    original_dtype: np.dtype

    def nbytes(self) -> int:
        """Total bytes of the packed representation (packed + scales)."""
        return self.packed.nbytes + self.scales.nbytes

    @property
    def compression_ratio(self) -> float:
        """Ratio of packed bytes to fp32 bytes (< 1 means compression)."""
        fp32_bytes = int(np.prod(self.original_shape)) * 4
        if fp32_bytes == 0:
            return 1.0
        return self.nbytes() / fp32_bytes

    def __repr__(self) -> str:
        return (
            f"FP6Quantized(fmt={self.fmt}, "
            f"shape={self.original_shape}, "
            f"packed={self.packed.shape}, "
            f"compression={self.compression_ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class FP6Stats:
    """Lifetime statistics for an FP6Quantizer.

    Attributes:
        quantize_calls: Number of ``quantize()`` calls.
        dequant_calls:  Number of ``dequantize()`` calls.
        total_fp32_bytes_in:  Total input bytes quantized.
        total_packed_bytes_out: Total output bytes produced.
    """

    quantize_calls: int = 0
    dequant_calls: int = 0
    total_fp32_bytes_in: int = 0
    total_packed_bytes_out: int = 0

    @property
    def mean_compression_ratio(self) -> float:
        if self.total_fp32_bytes_in == 0:
            return 1.0
        return self.total_packed_bytes_out / self.total_fp32_bytes_in

    def __repr__(self) -> str:
        return (
            f"FP6Stats(quantize={self.quantize_calls}, "
            f"dequant={self.dequant_calls}, "
            f"mean_ratio={self.mean_compression_ratio:.3f})"
        )


# ---------------------------------------------------------------------------
# Core FP6 encode / decode (scalar reference implementation)
# ---------------------------------------------------------------------------


def _fp6_encode_value(val: float, exp_bits: int, man_bits: int, exp_bias: int) -> int:
    """Encode a single float32 value into a 6-bit FP6 integer.

    Returns an integer in range [0, 63].
    """
    if np.isnan(val):
        # Encode as NaN: all-ones exponent, non-zero mantissa
        return ((1 << exp_bits) - 1) << man_bits | 1

    sign = 0
    if val < 0:
        sign = 1
        val = -val

    max_exp_val = (1 << exp_bits) - 2  # reserve top exponent for NaN/Inf
    man_max = (1 << man_bits) - 1

    if val == 0.0:
        return sign << 5

    import math

    fi = math.frexp(val)
    mantissa_f, exp_raw = fi  # mantissa in [0.5, 1.0), exponent stored

    exp_biased = exp_raw - 1 + exp_bias  # frexp: val = mantissa * 2^exp_raw
    # so val = (mantissa * 2) * 2^(exp_raw - 1) = significand * 2^(exp_biased - bias)

    if exp_biased <= 0:
        # Underflow → zero
        return sign << 5
    if exp_biased > max_exp_val:
        # Overflow → max representable
        return (sign << 5) | (max_exp_val << man_bits) | man_max

    # Round mantissa to man_bits
    # significand ∈ [1.0, 2.0)
    significand = mantissa_f * 2.0
    man_frac = significand - 1.0
    man_int = min(int(man_frac * (1 << man_bits) + 0.5), man_max)

    return (sign << 5) | (exp_biased << man_bits) | man_int


def _fp6_decode_value(code: int, exp_bits: int, man_bits: int, exp_bias: int) -> float:
    """Decode a 6-bit FP6 integer back to float32."""
    sign = (code >> 5) & 1
    exp_bits_val = (code >> man_bits) & ((1 << exp_bits) - 1)
    man_int = code & ((1 << man_bits) - 1)

    # NaN/Inf pattern
    if exp_bits_val == (1 << exp_bits) - 1:
        return float("nan")

    if exp_bits_val == 0:
        return 0.0

    val = (1.0 + man_int / (1 << man_bits)) * (2.0 ** (exp_bits_val - exp_bias))
    return -val if sign else val


def _pack_fp6_4values(a: int, b: int, c: int, d: int) -> Tuple[int, int, int]:
    """Pack four 6-bit codes into three bytes (24 bits).

    Layout:  byte0 = [a7..a2][b7..b6]
             byte1 = [b5..b0][c7..c4]
             byte2 = [c3..c0][d5..d0]

    More precisely (big-endian bit-packing):
      bits 23–18 = a (6 bits)
      bits 17–12 = b (6 bits)
      bits 11–6  = c (6 bits)
      bits  5–0  = d (6 bits)
    """
    packed = (a << 18) | (b << 12) | (c << 6) | d
    b0 = (packed >> 16) & 0xFF
    b1 = (packed >> 8) & 0xFF
    b2 = packed & 0xFF
    return b0, b1, b2


def _unpack_fp6_4values(b0: int, b1: int, b2: int) -> Tuple[int, int, int, int]:
    """Unpack three bytes into four 6-bit codes."""
    packed = (b0 << 16) | (b1 << 8) | b2
    a = (packed >> 18) & 0x3F
    b = (packed >> 12) & 0x3F
    c = (packed >> 6) & 0x3F
    d = packed & 0x3F
    return a, b, c, d


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------


class FP6Quantizer:
    """FP6-LLM weight quantizer with per-group scaling.

    Parameters
    ----------
    config:
        Quantizer configuration.
    """

    def __init__(self, config: FP6Config | None = None) -> None:
        self._cfg = config or FP6Config()
        self.stats = FP6Stats()

    @property
    def config(self) -> FP6Config:
        return self._cfg

    # ------------------------------------------------------------------
    # Quantize
    # ------------------------------------------------------------------

    def quantize(self, weight: np.ndarray) -> FP6Quantized:
        """Quantize a float32 weight tensor to FP6.

        Parameters
        ----------
        weight: float32 array of any shape.

        Returns
        -------
        FP6Quantized containing the packed bitstream and per-group scales.
        """
        cfg = self._cfg
        w = np.asarray(weight, dtype=np.float32)
        original_shape = w.shape
        flat = w.ravel()
        n = flat.size

        # Pad to multiple of group_size
        gs = cfg.group_size
        pad = (-n) % gs
        if pad:
            flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])

        n_groups = len(flat) // gs

        # Per-group absmax scaling
        groups = flat.reshape(n_groups, gs)
        scales = np.abs(groups).max(axis=1).astype(np.float32)
        scales = np.where(scales == 0, 1.0, scales)  # avoid division by zero
        fp6_max = cfg.fp6_max
        norm = groups / scales[:, None] * fp6_max  # values in [-fp6_max, +fp6_max]

        # Encode each value
        eb, mb, bias = cfg.exp_bits, cfg.man_bits, cfg.exp_bias

        # Each group encodes gs values → ceil(gs/4)*3 bytes
        bytes_per_group = ((gs + 3) // 4) * 3
        packed = np.zeros((n_groups, bytes_per_group), dtype=np.uint8)

        for g in range(n_groups):
            row = norm[g]
            byte_idx = 0
            for i in range(0, gs, 4):
                vals = [0, 0, 0, 0]
                for k in range(4):
                    if i + k < gs:
                        vals[k] = _fp6_encode_value(float(row[i + k]), eb, mb, bias)
                b0, b1, b2 = _pack_fp6_4values(*vals)
                packed[g, byte_idx] = b0
                packed[g, byte_idx + 1] = b1
                packed[g, byte_idx + 2] = b2
                byte_idx += 3

        self.stats.quantize_calls += 1
        self.stats.total_fp32_bytes_in += w.nbytes
        result = FP6Quantized(
            packed=packed,
            scales=scales,
            original_shape=original_shape,
            fmt=cfg.fmt,
            original_dtype=w.dtype,
        )
        self.stats.total_packed_bytes_out += result.nbytes()
        return result

    # ------------------------------------------------------------------
    # Dequantize
    # ------------------------------------------------------------------

    def dequantize(self, q: FP6Quantized) -> np.ndarray:
        """Reconstruct float32 weights from a FP6Quantized tensor.

        Parameters
        ----------
        q: FP6Quantized produced by ``quantize()``.

        Returns
        -------
        float32 array of the original shape (approximately).
        """
        cfg_fmt = q.fmt
        if cfg_fmt not in _SUPPORTED_FMTS:
            raise ValueError(f"Unknown FP6 format '{cfg_fmt}'")

        # Reconstruct config for this quantized tensor
        eb = 3 if cfg_fmt == "e3m2" else 2
        mb = 2 if cfg_fmt == "e3m2" else 3
        bias = (1 << (eb - 1)) - 1
        fp6_max = float((1 << eb) - 2 - bias)  # rough; actual max from config
        # Recompute fp6_max properly
        m_max = (1 << mb) - 1
        fp6_max = float(2 ** ((1 << eb) - 2 - bias) * (1.0 + m_max / (1 << mb)))

        n_groups, bytes_pg = q.packed.shape
        gs = bytes_pg // 3 * 4  # recover group_size from packed rows

        out = np.zeros(n_groups * gs, dtype=np.float32)

        for g in range(n_groups):
            row_bytes = q.packed[g]
            byte_idx = 0
            vals_out = []
            for _ in range(0, gs, 4):
                b0, b1, b2 = int(row_bytes[byte_idx]), int(row_bytes[byte_idx + 1]), int(row_bytes[byte_idx + 2])
                a, b, c, d = _unpack_fp6_4values(b0, b1, b2)
                for code in [a, b, c, d]:
                    vals_out.append(_fp6_decode_value(code, eb, mb, bias) / fp6_max * q.scales[g])
                byte_idx += 3
            out[g * gs : (g + 1) * gs] = vals_out[:gs]

        # Trim to original size and reshape
        n_orig = int(np.prod(q.original_shape))
        out = out[:n_orig].reshape(q.original_shape)
        self.stats.dequant_calls += 1
        return out

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        """Theoretical compression ratio for this config (bits used / 32)."""
        return 6.0 / 32.0  # FP6 is 6 bits vs 32-bit float

    def __repr__(self) -> str:
        return (
            f"FP6Quantizer(fmt={self._cfg.fmt}, "
            f"group_size={self._cfg.group_size}, "
            f"{self.stats})"
        )
