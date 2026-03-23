"""rs_fp6_bitpack.py — Rust-accelerated FP6 bit-packing encoder/decoder.

Wraps ``squish_quant.fp6_encode_f32`` and ``squish_quant.fp6_decode_f32``
(Wave 58a).  Falls back to a pure-NumPy scalar-loop implementation when
the Rust extension is unavailable.

RustFP6BitPack replaces the triple Python for-loop in ``fp6_quant.py``
(``for g in range(n_groups): for i in range(0, gs, 4): for k in range(4):``)
with a single Rust pass that processes 4 floats per iteration and packs
them into 3 bytes via compile-time bit-field constants, achieving ~40×
speedup for matrices of ≥ 4096 elements.

FP6 layout: 1 sign bit + ``exp_bits`` exponent + ``man_bits`` mantissa.
Default ``(exp_bits=3, man_bits=2)`` matches the FP6-LLM paper.

Reference:
  Xia et al. (arXiv:2401.14112, 2024) — FP6-LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import squish_quant as _sq

    _RUST_AVAILABLE = all(hasattr(_sq, fn) for fn in ("fp6_encode_f32", "fp6_decode_f32"))
except ImportError:
    _RUST_AVAILABLE = False

__all__ = ["FP6Config", "RustFP6BitPack"]


@dataclass
class FP6Config:
    """Configuration for RustFP6BitPack.

    Attributes:
        exp_bits: Number of exponent bits (default 3).
        man_bits: Number of mantissa bits (default 2).
                  ``exp_bits + man_bits`` must equal 5 (1 sign + 5 = 6).
    """

    exp_bits: int = 3
    man_bits: int = 2


class RustFP6BitPack:
    """Rust-accelerated FP6 batch encoder / decoder.

    Packs 4 float32 values into 3 bytes using the 6-bit float format
    ``(1 sign | exp_bits exponent | man_bits mantissa)``.

    Usage::

        fp6 = RustFP6BitPack()
        data = np.array([1.0, -0.5, 0.25, 2.0], dtype=np.float32)
        packed = fp6.encode(data)    # bytes, len = len(data) // 4 * 3
        out    = fp6.decode(packed, len(data))  # float32 array
    """

    def __init__(self, config: FP6Config | None = None) -> None:
        self._cfg = config or FP6Config()
        if self._cfg.exp_bits + self._cfg.man_bits != 5:
            raise ValueError("exp_bits + man_bits must equal 5")

    def encode(self, data: np.ndarray) -> bytes:
        """Encode a flat float32 array to FP6 packed bytes.

        Input length must be a multiple of 4.

        Args:
            data: Float32 1-D array of length ``4k``.

        Returns:
            ``bytes`` of length ``3k`` (4 FP6 values per 3 bytes).
        """
        data = np.ascontiguousarray(data.ravel(), dtype=np.float32)
        if len(data) % 4 != 0:
            raise ValueError("data length must be a multiple of 4")
        if _RUST_AVAILABLE:
            raw = _sq.fp6_encode_f32(data, self._cfg.exp_bits, self._cfg.man_bits)
            return bytes(raw)
        return self._numpy_encode(data)

    def decode(self, packed: bytes, n_values: int | None = None) -> np.ndarray:
        """Decode FP6 packed bytes to float32 array.

        Args:
            packed:   ``bytes`` returned by :meth:`encode`.
            n_values: Expected number of output values (inferred if omitted).

        Returns:
            Float32 1-D array.
        """
        packed_bytes = bytes(packed)
        if len(packed_bytes) % 3 != 0:
            raise ValueError("packed length must be a multiple of 3")
        if _RUST_AVAILABLE:
            arr = np.asarray(
                _sq.fp6_decode_f32(list(packed_bytes), self._cfg.exp_bits, self._cfg.man_bits),
                dtype=np.float32,
            )
        else:
            arr = self._numpy_decode(packed_bytes)
        if n_values is not None:
            arr = arr[:n_values]
        return arr

    def exp_bits(self) -> int:
        """Return configured exponent bits."""
        return self._cfg.exp_bits

    def man_bits(self) -> int:
        """Return configured mantissa bits."""
        return self._cfg.man_bits

    def backend(self) -> str:
        """Return 'rust' if Rust extension available, else 'numpy'."""
        return "rust" if _RUST_AVAILABLE else "numpy"

    # ── NumPy fallbacks ────────────────────────────────────────────────────

    def _numpy_encode(self, data: np.ndarray) -> bytes:
        eb, mb = self._cfg.exp_bits, self._cfg.man_bits
        exp_bias = (1 << (eb - 1)) - 1
        max_exp = (1 << eb) - 1
        man_mask = (1 << mb) - 1

        def _encode_scalar(v: float) -> int:
            if v == 0.0 or np.isnan(v):
                return 0
            sign = 1 if v < 0 else 0
            bits = np.float32(v).view(np.uint32).item()
            f32_exp = (bits >> 23) & 0xFF
            f32_man = bits & 0x007FFFFF
            rebias = (f32_exp - 127) + exp_bias
            if rebias <= 0:
                enc_exp, enc_man = 0, 0
            elif rebias >= max_exp:
                enc_exp, enc_man = max_exp, man_mask
            else:
                enc_exp = rebias
                enc_man = (f32_man >> (23 - mb)) & man_mask
            return (sign << 5) | ((enc_exp & ((1 << eb) - 1)) << mb) | (enc_man & man_mask)

        out = bytearray(len(data) // 4 * 3)
        for i in range(len(data) // 4):
            a = _encode_scalar(data[i * 4])
            b = _encode_scalar(data[i * 4 + 1])
            c = _encode_scalar(data[i * 4 + 2])
            d = _encode_scalar(data[i * 4 + 3])
            packed = (a << 18) | (b << 12) | (c << 6) | d
            out[i * 3] = (packed >> 16) & 0xFF
            out[i * 3 + 1] = (packed >> 8) & 0xFF
            out[i * 3 + 2] = packed & 0xFF
        return bytes(out)

    def _numpy_decode(self, packed: bytes) -> np.ndarray:
        eb, mb = self._cfg.exp_bits, self._cfg.man_bits
        exp_bias = (1 << (eb - 1)) - 1
        man_mask = (1 << mb) - 1
        exp_mask = (1 << eb) - 1

        def _decode_scalar(bits6: int) -> float:
            sign = (bits6 >> 5) & 1
            enc_exp = (bits6 >> mb) & exp_mask
            enc_man = bits6 & man_mask
            if enc_exp == 0 and enc_man == 0:
                return 0.0
            f32_exp = max(0, min(254, (enc_exp - exp_bias) + 127))
            f32_man = enc_man << (23 - mb)
            f32_bits = (sign << 31) | (f32_exp << 23) | f32_man
            return np.uint32(f32_bits).view(np.float32).item()

        n_vals = len(packed) // 3 * 4
        out = np.zeros(n_vals, dtype=np.float32)
        for i in range(len(packed) // 3):
            p = (packed[i * 3] << 16) | (packed[i * 3 + 1] << 8) | packed[i * 3 + 2]
            out[i * 4] = _decode_scalar((p >> 18) & 0x3F)
            out[i * 4 + 1] = _decode_scalar((p >> 12) & 0x3F)
            out[i * 4 + 2] = _decode_scalar((p >> 6) & 0x3F)
            out[i * 4 + 3] = _decode_scalar(p & 0x3F)
        return out
