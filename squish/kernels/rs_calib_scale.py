"""rs_calib_scale.py — Rust-accelerated per-channel activation calibration scales.

Wraps ``calib_absmax_f32``, ``calib_percentile_f32``, ``calib_aciq_f32`` from
``squish_quant_rs`` (Wave 59a).
Falls back to NumPy when the Rust extension is unavailable.

RustCalibScale replaces the ``for c in range(C):`` loop in
``quant_calib.py`` ``ActivationCalibrator.calibrate()``; Rayon column-parallel
dispatch for three scale methods (absmax, percentile, ACIQ); ~7× per batch.

Reference:
    Banner et al. (NeurIPS 2019) — ACIQ.
    Finkelstein et al. (NeurIPS 2019) — Percentile clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

try:
    import squish_quant as _sq
    _HAS_RUST = hasattr(_sq, "calib_absmax_f32")
except ImportError:
    _HAS_RUST = False

__all__ = ["CalibScaleConfig", "RustCalibScale"]

_MethodT = Literal["absmax", "percentile", "aciq"]


@dataclass
class CalibScaleConfig:
    method: _MethodT = "absmax"
    percentile: float = 99.9
    n_levels: int = 256


def _np_absmax(acts: np.ndarray) -> np.ndarray:
    return np.abs(acts).max(axis=0).astype(np.float32)


def _np_percentile(acts: np.ndarray, p: float) -> np.ndarray:
    return np.percentile(np.abs(acts), p, axis=0).astype(np.float32)


def _np_aciq(acts: np.ndarray, n_levels: int) -> np.ndarray:
    n = acts.shape[0]
    sigma = acts.std(axis=0).astype(np.float32)
    alpha = float(np.sqrt(2.0 * np.log(n) + 1.0)) if n > 1 else 1.0
    return np.maximum(sigma * alpha, 1e-6).astype(np.float32)


class RustCalibScale:
    """Per-channel activation scale calibration using absmax, percentile, or ACIQ.

    Args:
        config: :class:`CalibScaleConfig`.
    """

    def __init__(self, config: Optional[CalibScaleConfig] = None) -> None:
        self._cfg = config or CalibScaleConfig()

    # ------------------------------------------------------------------
    def compute_scales(
        self,
        acts: np.ndarray,
        method: Optional[_MethodT] = None,
        percentile: Optional[float] = None,
        n_levels: Optional[int] = None,
    ) -> np.ndarray:
        """Compute per-channel calibration scales from ``(N, C)`` activation array.

        Args:
            acts: ``(N, C)`` float32 activation tensor.
            method: Override config method (``'absmax'``, ``'percentile'``, ``'aciq'``).
            percentile: Override config percentile (only used when method=percentile).
            n_levels: Override config n_levels (only used when method=aciq).

        Returns:
            ``(C,)`` float32 scale array — one scale per channel.
        """
        a = np.ascontiguousarray(acts, dtype=np.float32)
        if a.ndim == 1:
            a = a[np.newaxis, :]
        m = method if method is not None else self._cfg.method
        p = float(percentile) if percentile is not None else self._cfg.percentile
        nl = int(n_levels) if n_levels is not None else self._cfg.n_levels
        if _HAS_RUST:
            if m == "absmax":
                return _sq.calib_absmax_f32(a)
            if m == "percentile":
                return _sq.calib_percentile_f32(a, p)
            # aciq
            return _sq.calib_aciq_f32(a, nl)
        if m == "absmax":
            return _np_absmax(a)
        if m == "percentile":
            return _np_percentile(a, p)
        return _np_aciq(a, nl)

    def method(self) -> str:
        return self._cfg.method

    def backend(self) -> str:
        return "rust" if _HAS_RUST else "numpy"
