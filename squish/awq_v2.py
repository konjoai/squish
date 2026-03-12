"""AWQv2 — Activation-Aware Weight Quantization v2 with per-channel scale and shift.

AWQ v2 extends the original AWQ algorithm by additionally learning a per-channel
zero-point shift (``shifts``) alongside the per-channel scale (``scales``).
Both are determined from activation statistics, concentrating quantisation
precision on *salient* input channels — those with large activation magnitudes
— while allowing less-activated channels to absorb more quantisation error.

For each input channel ``c``:

1. Search ``n_search_steps`` candidate scales linearly spaced between
   ``act_scales[c] * 0.1`` and ``act_scales[c] * 2.0``.
2. For each candidate scale ``s``:
   a. Pre-scale: ``w_scaled = W[:, c] / s``.
   b. Quantise: ``q = round(w_scaled).clip(0, 2^n_bits - 1)``
      (asymmetric unsigned, incorporating the zero-point implicitly).
   c. Dequantise: ``w_recon = q * s``.
   d. Error: ``||W[:, c] - w_recon||^2``.
3. Select ``s_c`` minimising reconstruction error.
4. Compute ``shift_c = round(-min(W[:, c] / s_c))``, clipped to
   ``[0, 2^n_bits - 1]`` — the zero-point that centres the unsigned range.

Reference:
    Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
    Compression and Acceleration", MLSys 2024.
    https://arxiv.org/abs/2306.00978

Usage::

    import numpy as np
    from squish.awq_v2 import AWQv2Config, AWQv2Calibrator, AWQv2Stats

    rng         = np.random.default_rng(0)
    out_f, in_f = 256, 512
    weights     = rng.standard_normal((out_f, in_f)).astype(np.float32)
    act_scales  = np.abs(rng.standard_normal(in_f)).astype(np.float32) + 0.1

    cfg   = AWQv2Config(n_bits=4, group_size=128, n_search_steps=20)
    calib = AWQv2Calibrator(cfg)

    opt_scales, opt_shifts = calib.calibrate(weights, act_scales)
    q_uint8 = calib.quantize(weights, opt_scales, opt_shifts)
    print(q_uint8.shape, q_uint8.dtype)  # (256, 512) uint8
    print(calib.stats)
"""

from __future__ import annotations

__all__ = ["AWQv2Config", "AWQv2Calibrator", "AWQv2Stats"]

from dataclasses import dataclass

import numpy as np


@dataclass
class AWQv2Config:
    """Configuration for AWQ v2 calibration.

    Attributes:
        n_bits: Quantisation bit-width; 4 or 8.
        group_size: Number of columns per quantisation group.
        n_search_steps: Number of candidate scale values to evaluate per
            input channel during the scale search.
    """

    n_bits:          int = 4
    group_size:      int = 128
    n_search_steps:  int = 20

    def __post_init__(self) -> None:
        if self.n_bits not in (4, 8):
            raise ValueError(
                f"n_bits must be 4 or 8; got {self.n_bits}"
            )
        if self.group_size < 8:
            raise ValueError(
                f"group_size must be >= 8; got {self.group_size}"
            )
        if self.n_search_steps < 1:
            raise ValueError(
                f"n_search_steps must be >= 1; got {self.n_search_steps}"
            )

    @property
    def q_max(self) -> int:
        """Maximum unsigned integer code (``2^n_bits - 1``)."""
        return (1 << self.n_bits) - 1


@dataclass
class AWQv2Stats:
    """Running statistics for an :class:`AWQv2Calibrator` session.

    Attributes:
        total_calibrations: Number of :meth:`AWQv2Calibrator.calibrate` calls.
        total_channels: Cumulative number of input channels calibrated.
    """

    total_calibrations: int = 0
    total_channels:     int = 0


class AWQv2Calibrator:
    """AWQ v2 per-channel scale-and-shift calibrator.

    Args:
        config: :class:`AWQv2Config` controlling bit-width, group size, and
            search granularity.
    """

    def __init__(self, config: AWQv2Config) -> None:
        self._config = config
        self._stats  = AWQv2Stats()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def calibrate(
        self,
        weights:    np.ndarray,
        act_scales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find per-channel optimal scales and zero-point shifts.

        For each input channel ``c``, this method searches ``n_search_steps``
        candidate scales in ``[act_scales[c]*0.1, act_scales[c]*2.0]`` and
        selects the one that minimises squared reconstruction error on
        ``W[:, c]``.

        Args:
            weights:    Float32 weight matrix of shape
                ``(out_features, in_features)``.
            act_scales: Float32 per-channel activation magnitude statistics,
                shape ``(in_features,)``.

        Returns:
            Tuple ``(opt_scales, opt_shifts)`` both of shape
            ``(in_features,)`` and dtype float32.

        Raises:
            ValueError: If shapes are incompatible.
        """
        weights    = np.asarray(weights,    dtype=np.float32)
        act_scales = np.asarray(act_scales, dtype=np.float32)

        if weights.ndim != 2:
            raise ValueError(
                f"weights must be 2-D (out_features, in_features); "
                f"got shape {weights.shape}"
            )
        out_features, in_features = weights.shape
        if act_scales.shape != (in_features,):
            raise ValueError(
                f"act_scales must have shape (in_features={in_features},); "
                f"got {act_scales.shape}"
            )

        cfg   = self._config
        q_max = float(cfg.q_max)

        opt_scales = np.empty(in_features, dtype=np.float32)
        opt_shifts = np.empty(in_features, dtype=np.float32)

        for c in range(in_features):
            w_col      = weights[:, c]  # (out_features,)
            act_s      = float(act_scales[c])

            # Avoid degenerate search range when act_scale is near zero
            low   = max(act_s * 0.1, 1e-8)
            high  = max(act_s * 2.0, 1e-7)
            if low >= high:
                high = low * 10.0

            candidates = np.linspace(low, high, cfg.n_search_steps,
                                     dtype=np.float32)

            best_scale = candidates[0]
            best_error = np.inf

            for s in candidates:
                # Pre-scale channel
                w_scaled = w_col / s

                # Compute zero-point: shift so that minimum of w_scaled maps
                # to 0 in the unsigned integer range.
                zp = np.round(-np.min(w_scaled))
                zp = float(np.clip(zp, 0.0, q_max))

                # Quantise with zero-point
                q = np.clip(np.round(w_scaled + zp), 0.0, q_max)

                # Dequantise
                w_recon = (q - zp) * s

                # Reconstruction error
                error = float(np.sum((w_col - w_recon) ** 2))

                if error < best_error:
                    best_error = error
                    best_scale = float(s)

            # Compute the optimal zero-point for the chosen scale
            w_col_scaled = w_col / best_scale
            best_shift   = float(
                np.clip(np.round(-np.min(w_col_scaled)), 0.0, q_max)
            )

            opt_scales[c] = best_scale
            opt_shifts[c] = best_shift

        self._stats.total_calibrations += 1
        self._stats.total_channels     += in_features

        return opt_scales, opt_shifts

    def quantize(
        self,
        weights: np.ndarray,
        scales:  np.ndarray,
        shifts:  np.ndarray,
    ) -> np.ndarray:
        """Apply the calibrated scales and shifts to produce uint8 weight codes.

        Args:
            weights: Float32 weight matrix of shape
                ``(out_features, in_features)``.
            scales:  Per-channel scales, shape ``(in_features,)``, float32.
            shifts:  Per-channel zero-point shifts, shape ``(in_features,)``,
                float32.

        Returns:
            Uint8 quantised weight matrix of shape ``(out_features, in_features)``.
        """
        weights = np.asarray(weights, dtype=np.float32)
        scales  = np.asarray(scales,  dtype=np.float32)
        shifts  = np.asarray(shifts,  dtype=np.float32)

        q_max = float(self._config.q_max)

        # Broadcast (in_features,) over the out_features dimension
        w_scaled = weights / scales[np.newaxis, :]       # (out_features, in_features)
        q        = np.round(w_scaled + shifts[np.newaxis, :])
        q        = np.clip(q, 0.0, q_max).astype(np.uint8)

        return q

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> AWQv2Stats:
        """Running calibration statistics."""
        return self._stats
