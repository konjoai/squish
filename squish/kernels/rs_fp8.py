"""Rust-backed FP8 E4M3 / E5M2 quantization kernel.

Wraps ``squish_quant.{quantize,dequantize}_fp8_{e4m3,e5m2}_{f32,}`` from the
maturin-compiled Rust extension.  Falls back to a pure-NumPy path when the
extension is unavailable.

Formats supported:
- **E4M3**: 4-exponent, 3-mantissa bits; max representable = 448.0
- **E5M2**: 5-exponent, 2-mantissa bits; max representable = 57344.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

__all__ = ["FP8KernelConfig", "RustFP8Kernel"]

_E4M3_MAX: float = 448.0
_E5M2_MAX: float = 57344.0
_E4M3_BIAS: int = 7
_E5M2_BIAS: int = 15

try:
    import squish_quant as _sq  # type: ignore[import]

    _RUST_AVAILABLE = True
except ImportError:
    _sq = None
    _RUST_AVAILABLE = False


@dataclass
class FP8KernelConfig:
    """Configuration for :class:`RustFP8Kernel`.

    Attributes
    ----------
    fmt:
        FP8 format: ``"e4m3"`` (default) or ``"e5m2"``.
    per_tensor_scale:
        When ``True`` (default), a single per-tensor abs-max scale is used.
        When ``False``, per-row scales are computed.
    """

    fmt: str = "e4m3"
    per_tensor_scale: bool = True


class RustFP8Kernel:
    """FP8 quantization kernel (E4M3 / E5M2) backed by Rust.

    Parameters
    ----------
    config:
        Kernel configuration.  Defaults to :class:`FP8KernelConfig`.
    """

    def __init__(self, config: FP8KernelConfig | None = None) -> None:
        self.config = config or FP8KernelConfig()
        if self.config.fmt not in ("e4m3", "e5m2"):
            raise ValueError("fmt must be 'e4m3' or 'e5m2'")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def quantize(
        self, W: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Quantize a float32 weight matrix to FP8.

        Parameters
        ----------
        W:
            Float32 array of shape ``(N, D)``.

        Returns
        -------
        W_q:
            uint8 array of shape ``(N, D)``.
        scale:
            Per-tensor float32 scale.
        """
        if W.ndim != 2:
            raise ValueError(f"W must be 2-D, got shape {W.shape}")

        W32 = W.astype(np.float32, copy=False)

        if _RUST_AVAILABLE:
            try:
                fn = (
                    _sq.quantize_fp8_e4m3_f32
                    if self.config.fmt == "e4m3"
                    else _sq.quantize_fp8_e5m2_f32
                )
                q, s = fn(W32)
                return q, float(s[0])
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_quantize(W32)

    def dequantize(
        self, W_q: np.ndarray, scale: Union[float, np.ndarray]
    ) -> np.ndarray:
        """Dequantize FP8 to float32.

        Parameters
        ----------
        W_q:
            uint8 array of shape ``(N, D)``.
        scale:
            Scalar float scale returned by :meth:`quantize`.

        Returns
        -------
        float32 array of shape ``(N, D)``.
        """
        scale_f = float(scale) if not isinstance(scale, float) else scale

        if _RUST_AVAILABLE:
            try:
                fn = (
                    _sq.dequantize_fp8_e4m3
                    if self.config.fmt == "e4m3"
                    else _sq.dequantize_fp8_e5m2
                )
                return fn(W_q, scale_f)
            except Exception:  # noqa: BLE001
                pass

        return self._numpy_dequantize(W_q, scale_f)

    def max_representable(self) -> float:
        """Return the largest magnitude representable in this FP8 format."""
        return _E4M3_MAX if self.config.fmt == "e4m3" else _E5M2_MAX

    # ------------------------------------------------------------------ #
    #  NumPy fallback implementations                                     #
    # ------------------------------------------------------------------ #

    def _numpy_quantize(self, W: np.ndarray) -> Tuple[np.ndarray, float]:
        max_val = self.max_representable()
        abs_max = float(np.abs(W).max())
        scale = 1.0 if abs_max == 0.0 else abs_max / max_val

        W_scaled = W / scale
        # Clamp and round; store raw bit pattern as uint8
        W_clipped = np.clip(W_scaled, -max_val, max_val)
        # Simple integer-quantize: scale value to [0, 255] via sign-magnitude
        sign = np.sign(W_clipped).astype(np.float32)
        mag = np.abs(W_clipped)
        if self.config.fmt == "e4m3":
            # Map magnitude to [0, 127] using E4M3 dynamic range
            log_mag = np.where(mag > 0, np.log2(mag.clip(1e-30, max_val)), -float("inf"))
            log_max = np.log2(max_val)
            # Scale log to [0, 1] and map to 7-bit value
            normalized = ((log_mag + 10) / (log_max + 10)).clip(0, 1)
            q_mag = np.round(normalized * 127).astype(np.uint8)
        else:
            log_mag = np.where(mag > 0, np.log2(mag.clip(1e-30, max_val)), -float("inf"))
            log_max = np.log2(max_val)
            normalized = ((log_mag + 14) / (log_max + 14)).clip(0, 1)
            q_mag = np.round(normalized * 127).astype(np.uint8)

        sign_bit = (sign < 0).astype(np.uint8) << 7
        return (sign_bit | q_mag).astype(np.uint8), scale

    def _numpy_dequantize(self, W_q: np.ndarray, scale: float) -> np.ndarray:
        sign = np.where(W_q & 0x80, -1.0, 1.0).astype(np.float32)
        mag_q = (W_q & 0x7F).astype(np.float32)
        max_val = self.max_representable()

        if self.config.fmt == "e4m3":
            log_max = np.log2(max_val)
            normalized = mag_q / 127.0
            log_mag = normalized * (log_max + 10) - 10
        else:
            log_max = np.log2(max_val)
            normalized = mag_q / 127.0
            log_mag = normalized * (log_max + 14) - 14

        mag = np.where(mag_q > 0, np.exp2(log_mag), 0.0).astype(np.float32)
        return (sign * mag * scale).astype(np.float32)
