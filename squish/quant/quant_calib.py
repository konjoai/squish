# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""QuantCalib — Unified activation-range calibration for post-training quantisation.

Before deploying a quantised model it is necessary to determine per-channel (or
global) scale factors that map floating-point activations to integer codes with
minimal error.  This module implements three calibration strategies over a
representative calibration dataset:

* **MinMax** — symmetric range ``max(|x|) × 2``; fast and robust for
  well-behaved activation distributions.
* **Percentile** — clips at the *p*-th percentile of absolute values, reducing
  the influence of outliers at the cost of clipping error.
* **MSE** — grid-searches over 100 candidate clipping radii in
  ``[0.1 × max, 1.0 × max]`` and selects the one that minimises the mean
  squared reconstruction error after quant-dequant.

The calibrated per-channel (or global) scales are returned as a
:class:`CalibResult` dataclass ready to feed into an INT4/INT8 quantiser.

Usage::

    import numpy as np
    from squish.quant.quant_calib import CalibConfig, QuantCalibrator

    cfg   = CalibConfig(method="mse", n_bits=8, per_channel=True)
    calib = QuantCalibrator(cfg)

    activations = np.random.randn(128, 64, 4096).astype(np.float32)
    result = calib.calibrate(activations)

    print(result.scales.shape)   # (4096,) — one scale per channel
    print(calib.stats.total_calibrations)
"""

from __future__ import annotations

__all__ = [
    "CalibConfig",
    "CalibResult",
    "QuantCalibrator",
    "CalibStats",
]

from dataclasses import dataclass

import numpy as np


# ── Constants ─────────────────────────────────────────────────────────────────

_VALID_METHODS   = frozenset({"minmax", "percentile", "mse"})
_VALID_N_BITS    = frozenset({4, 8})
_MSE_N_CANDIDATES = 100


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class CalibConfig:
    """Configuration for the activation-range calibration pipeline.

    Attributes:
        method: Scale-selection method — ``"minmax"``, ``"percentile"``,
            or ``"mse"``.
        percentile: Percentile value used when ``method="percentile"``.
            Must be in ``(50, 100)``.
        n_bits: Quantisation bit-width — either 4 or 8.
        per_channel: When ``True``, a separate scale is computed for each
            channel (last dimension).  When ``False``, a single global scale
            is returned equal to the mean of per-channel scales.
    """

    method: str = "mse"
    percentile: float = 99.9
    n_bits: int = 8
    per_channel: bool = True

    def __post_init__(self) -> None:
        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"method must be one of {sorted(_VALID_METHODS)}; "
                f"got {self.method!r}"
            )
        if self.n_bits not in _VALID_N_BITS:
            raise ValueError(
                f"n_bits must be one of {sorted(_VALID_N_BITS)}; "
                f"got {self.n_bits}"
            )
        if not (50.0 < self.percentile < 100.0):
            raise ValueError(
                f"percentile must be in (50, 100); got {self.percentile}"
            )


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class CalibResult:
    """Result of a calibration pass.

    Attributes:
        scales: Per-channel float32 scales of shape ``(C,)`` when
            ``per_channel=True``, or a scalar array of shape ``()`` when
            ``per_channel=False``.
        method: Calibration method used to produce *scales*.
        n_bits: Quantisation bit-width for which the scales were optimised.
    """

    scales: np.ndarray
    method: str
    n_bits: int


# ── Stats ─────────────────────────────────────────────────────────────────────

@dataclass
class CalibStats:
    """Cumulative statistics for a :class:`QuantCalibrator` session.

    Attributes:
        total_calibrations: Number of times :meth:`QuantCalibrator.calibrate`
            has been called.
        total_channels: Cumulative number of channels calibrated across all
            calls.
    """

    total_calibrations: int = 0
    total_channels: int = 0


# ── Calibrator ────────────────────────────────────────────────────────────────

class QuantCalibrator:
    """Activation-range calibration pipeline for post-training quantisation.

    Given a batch of calibration activations, computes the scale factor(s)
    that minimise quantisation error under the configured method.

    Args:
        config: :class:`CalibConfig` specifying the method, bit-width, and
            channel granularity.
    """

    def __init__(self, config: CalibConfig) -> None:
        self.config = config
        self._stats = CalibStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calibrate(self, activations: np.ndarray) -> CalibResult:
        """Compute quantisation scales from calibration activations.

        The last dimension of *activations* is treated as the channel
        dimension.  All other dimensions are collapsed into a single batch
        axis before computing per-channel statistics.

        Args:
            activations: Float32 array of shape ``(batch, channels)`` or
                ``(batch, seq, channels)`` — any number of leading dims is
                supported as long as the last dim represents channels.

        Returns:
            :class:`CalibResult` with ``scales`` of shape ``(C,)`` when
            ``per_channel=True`` or shape ``()`` when ``per_channel=False``.
        """
        cfg = self.config
        acts = np.asarray(activations, dtype=np.float32)

        # Reshape to 2D: (N, C) — last dim is channels.
        C = acts.shape[-1]
        acts_2d = acts.reshape(-1, C)   # (N, C)

        n_levels = (1 << cfg.n_bits) - 1  # e.g. 255 for 8-bit

        per_channel_scales = np.empty(C, dtype=np.float32)

        for c in range(C):
            col = acts_2d[:, c]
            per_channel_scales[c] = self._compute_scale(
                col, n_levels, cfg
            )

        if cfg.per_channel:
            scales = per_channel_scales
        else:
            # Single global scale = mean of per-channel scales.
            scales = np.array(
                float(np.mean(per_channel_scales)), dtype=np.float32
            )

        self._stats.total_calibrations += 1
        self._stats.total_channels     += C

        return CalibResult(scales=scales, method=cfg.method, n_bits=cfg.n_bits)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_scale(
        col: np.ndarray,
        n_levels: int,
        cfg: CalibConfig,
    ) -> float:
        """Compute the scale for a single channel column.

        Args:
            col: 1-D float32 array of calibration values for one channel.
            n_levels: ``2^n_bits - 1``
            cfg: Calibration configuration.

        Returns:
            Optimal scale for this channel as a Python float.
        """
        abs_col = np.abs(col)
        max_abs  = float(np.max(abs_col)) if abs_col.size > 0 else 0.0

        if max_abs == 0.0:
            return 1.0  # Degenerate channel; return unit scale.

        if cfg.method == "minmax":
            range_val = max_abs * 2.0
            return float(range_val / n_levels)

        if cfg.method == "percentile":
            p_val = float(np.percentile(abs_col, cfg.percentile))
            range_val = p_val * 2.0
            return float(range_val / n_levels)

        # cfg.method == "mse"
        return QuantCalibrator._mse_scale(col, max_abs, n_levels)

    @staticmethod
    def _mse_scale(
        col: np.ndarray,
        max_abs: float,
        n_levels: int,
    ) -> float:
        """Grid-search for the scale that minimises quant-dequant MSE.

        Evaluates ``_MSE_N_CANDIDATES`` evenly spaced candidate clipping
        radii in ``[max_abs × 0.1, max_abs × 1.0]``.  For each candidate
        the data is symmetrically clipped, quantised, dequantised, and the
        MSE vs the original is computed.  The candidate with the lowest
        MSE is selected.

        Args:
            col: 1-D float32 activation column.
            max_abs: Maximum absolute value of *col*.
            n_levels: ``2^n_bits - 1``

        Returns:
            Scale factor ``2 × best_half_range / n_levels``.
        """
        # Candidate half-ranges (the clip bound, i.e. max representable abs
        # value): linearly spaced from 10% to 100% of the observed max.
        half_ranges = np.linspace(max_abs * 0.1, max_abs, _MSE_N_CANDIDATES)

        best_mse   = float("inf")
        best_scale = float(max_abs * 2.0 / n_levels)

        for hr in half_ranges:
            step = (2.0 * hr) / n_levels
            if step == 0.0:
                continue
            clipped = np.clip(col, -hr, hr)
            # Quantise-dequantise (symmetric uniform)
            recon = np.round(clipped / step) * step
            mse   = float(np.mean((col - recon) ** 2))
            if mse < best_mse:
                best_mse   = mse
                best_scale = step

        return best_scale

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> CalibStats:
        """Cumulative calibration statistics for this instance."""
        return self._stats
