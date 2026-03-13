# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/adaptive_quantize.py

AdaptiveQuantize — Runtime precision switching under memory pressure.

When device memory pressure rises above configurable thresholds, automatically
downgrade the numerical precision used for KV cache and weight storage from
FP16 to INT8, and then from INT8 to INT4.  As pressure drops (e.g. because
requests complete and their KV caches are freed), the precision is upgraded
back toward FP16.

The :class:`PressureMonitor` tracks the fraction of capacity currently in use
and exposes the appropriate :attr:`current_precision` string.  The
:class:`AdaptiveQuantizer` uses the monitor to decide how to quantise an
``np.ndarray``: it returns a quantised array and a floating-point scale factor
that can be stored alongside the quantised data and used by
:meth:`dequantize` to reconstruct an approximation of the original.

Quantisation schemes:

* ``"fp16"`` — cast to ``float16``; scale is 1.0.
* ``"int8"`` — symmetric per-tensor linear quantisation to ``int8`` range
  ``[−127, 127]``; scale = ``max(|x|) / 127``.
* ``"int4"`` — symmetric per-tensor linear quantisation to ``int4`` range
  ``[−7, 7]`` stored in ``int8``; scale = ``max(|x|) / 7``.

Example usage::

    import numpy as np
    from squish.quant.adaptive_quantize import (
        PressureThresholds, PressureMonitor, AdaptiveQuantizer,
    )

    thresholds = PressureThresholds(int8_threshold=0.75, int4_threshold=0.90)
    monitor = PressureMonitor(thresholds, capacity_bytes=4 * 1024 ** 3)  # 4 GiB
    quantizer = AdaptiveQuantizer(monitor)

    x = np.random.randn(128, 256).astype(np.float32)

    monitor.update(used_bytes=int(0.80 * 4 * 1024 ** 3))  # 80% full → INT8
    q, scale = quantizer.quantize(x)
    x_approx = quantizer.dequantize(q, scale, precision="int8")

    print(f"precision={monitor.current_precision}, scale={scale:.6f}")
    print(quantizer.stats)
"""

from __future__ import annotations

__all__ = [
    "PressureThresholds",
    "QuantPrecision",
    "PressureMonitor",
    "AdaptiveQuantizer",
    "AdaptiveQuantStats",
]

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Precision constants
# ---------------------------------------------------------------------------


class QuantPrecision:
    """Named constants for supported quantisation precisions.

    Attributes:
        FP16: 16-bit floating point (half precision).
        INT8: 8-bit symmetric linear quantisation (range −127 to 127).
        INT4: 4-bit symmetric linear quantisation stored in int8 (range −7 to 7).
    """

    FP16: str = "fp16"
    INT8: str = "int8"
    INT4: str = "int4"


# ---------------------------------------------------------------------------
# Pressure thresholds
# ---------------------------------------------------------------------------


@dataclass
class PressureThresholds:
    """Memory-pressure thresholds that govern precision switching.

    Attributes:
        int8_threshold: Fraction of capacity at which the precision switches
                        from FP16 to INT8.  Must be in (0, 1).
        int4_threshold: Fraction of capacity at which the precision switches
                        from INT8 to INT4.  Must be greater than
                        ``int8_threshold`` and in (0, 1].
    """

    int8_threshold: float = 0.75
    int4_threshold: float = 0.90

    def __post_init__(self) -> None:
        if not (0.0 < self.int8_threshold < 1.0):
            raise ValueError(
                f"int8_threshold must be in (0, 1), got {self.int8_threshold}"
            )
        if not (0.0 < self.int4_threshold <= 1.0):
            raise ValueError(
                f"int4_threshold must be in (0, 1], got {self.int4_threshold}"
            )
        if self.int4_threshold <= self.int8_threshold:
            raise ValueError(
                f"int4_threshold ({self.int4_threshold}) must be greater than "
                f"int8_threshold ({self.int8_threshold})"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class AdaptiveQuantStats:
    """Cumulative quantisation call statistics.

    Attributes:
        total_quantize_calls: Total calls to :meth:`AdaptiveQuantizer.quantize`.
        fp16_calls:           Calls performed at FP16 precision.
        int8_calls:           Calls performed at INT8 precision.
        int4_calls:           Calls performed at INT4 precision.
    """

    total_quantize_calls: int = 0
    fp16_calls: int = 0
    int8_calls: int = 0
    int4_calls: int = 0


# ---------------------------------------------------------------------------
# Pressure monitor
# ---------------------------------------------------------------------------


class PressureMonitor:
    """Tracks memory utilisation and determines the active quantisation precision.

    Args:
        thresholds:      A :class:`PressureThresholds` instance defining the
                         precision-switching boundaries.
        capacity_bytes:  Total device memory capacity in bytes.  Must be >= 1.
    """

    def __init__(self, thresholds: PressureThresholds, capacity_bytes: int) -> None:
        if capacity_bytes < 1:
            raise ValueError(f"capacity_bytes must be >= 1, got {capacity_bytes}")
        self._thresholds = thresholds
        self._capacity = capacity_bytes
        self._used: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, used_bytes: int) -> None:
        """Update the current memory utilisation.

        Args:
            used_bytes: Current number of bytes in use.  Must be >= 0 and
                        <= ``capacity_bytes``.

        Raises:
            ValueError: if *used_bytes* is negative or exceeds capacity.
        """
        if used_bytes < 0:
            raise ValueError(f"used_bytes must be >= 0, got {used_bytes}")
        if used_bytes > self._capacity:
            raise ValueError(
                f"used_bytes ({used_bytes}) exceeds capacity ({self._capacity})"
            )
        self._used = used_bytes

    @property
    def pressure(self) -> float:
        """Current memory pressure as a fraction of capacity (0.0–1.0)."""
        return self._used / self._capacity

    @property
    def current_precision(self) -> str:
        """The quantisation precision appropriate for the current pressure level.

        Returns:
            One of the :class:`QuantPrecision` constants:

            * ``"fp16"``  — pressure < ``int8_threshold``
            * ``"int8"``  — ``int8_threshold`` <= pressure < ``int4_threshold``
            * ``"int4"``  — pressure >= ``int4_threshold``
        """
        p = self.pressure
        if p >= self._thresholds.int4_threshold:
            return QuantPrecision.INT4
        if p >= self._thresholds.int8_threshold:
            return QuantPrecision.INT8
        return QuantPrecision.FP16

    @property
    def capacity_bytes(self) -> int:
        """Total device memory capacity in bytes."""
        return self._capacity

    @property
    def used_bytes(self) -> int:
        """Currently reported memory usage in bytes."""
        return self._used


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------


_INT8_MAX: int = 127
_INT4_MAX: int = 7


class AdaptiveQuantizer:
    """Quantises tensors to the precision demanded by a :class:`PressureMonitor`.

    Args:
        monitor: A :class:`PressureMonitor` that dictates the active precision.
    """

    def __init__(self, monitor: PressureMonitor) -> None:
        self._monitor = monitor
        self._stats = AdaptiveQuantStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Quantise *x* to the precision currently demanded by the monitor.

        Args:
            x: Input array of any floating-point dtype.  Must be non-empty.

        Returns:
            A tuple ``(quantised, scale)`` where:

            * ``quantised`` is the quantised ``np.ndarray``.
              - FP16: ``float16`` array, same shape as *x*.
              - INT8: ``int8`` array, same shape as *x*; values in [−127, 127].
              - INT4: ``int8`` array, same shape as *x*; values in [−7, 7].
            * ``scale`` is a positive float.
              - FP16: always 1.0.
              - INT8/INT4: the per-tensor symmetric quantisation scale
                (``max(|x|) / range_max``).  Use this value with
                :meth:`dequantize` to reconstruct an approximation of *x*.

        Raises:
            ValueError: if *x* is empty.
        """
        if x.size == 0:
            raise ValueError("Input array x must be non-empty")

        precision = self._monitor.current_precision
        self._stats.total_quantize_calls += 1

        if precision == QuantPrecision.FP16:
            self._stats.fp16_calls += 1
            return x.astype(np.float16), 1.0

        if precision == QuantPrecision.INT8:
            self._stats.int8_calls += 1
            return _quantize_symmetric(x, _INT8_MAX)

        # INT4
        self._stats.int4_calls += 1
        return _quantize_symmetric(x, _INT4_MAX)

    def dequantize(self, q: np.ndarray, scale: float, precision: str) -> np.ndarray:
        """Reconstruct a float32 approximation from a quantised array.

        Args:
            q:         Quantised array as returned by :meth:`quantize`.
            scale:     The scale factor returned alongside *q*.
            precision: The precision string used when *q* was produced.
                       Must be one of the :class:`QuantPrecision` constants.

        Returns:
            A ``float32`` array of the same shape as *q*.

        Raises:
            ValueError: if *precision* is not a recognised constant, or if
                        *scale* is non-positive.
        """
        _valid = {QuantPrecision.FP16, QuantPrecision.INT8, QuantPrecision.INT4}
        if precision not in _valid:
            raise ValueError(
                f"precision must be one of {sorted(_valid)}, got '{precision}'"
            )
        if scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {scale}")

        if precision == QuantPrecision.FP16:
            return q.astype(np.float32)

        # INT8 and INT4 share the same dequantisation formula.
        return q.astype(np.float32) * scale

    @property
    def stats(self) -> AdaptiveQuantStats:
        """Cumulative quantisation call statistics."""
        return self._stats

    @property
    def monitor(self) -> PressureMonitor:
        """The :class:`PressureMonitor` backing this quantizer."""
        return self._monitor


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _quantize_symmetric(x: np.ndarray, range_max: int) -> tuple[np.ndarray, float]:
    """Symmetric per-tensor linear quantisation.

    Computes ``scale = max(|x|) / range_max``, clips, rounds, and casts to
    ``int8``.  If the input is all zeros the scale is set to 1.0 to avoid
    division by zero.

    Args:
        x:          Input floating-point array.
        range_max:  Maximum absolute value of the quantised representation
                    (127 for INT8, 7 for INT4).

    Returns:
        ``(quantised_int8, scale_float)``
    """
    abs_max = float(np.max(np.abs(x)))
    scale = abs_max / range_max if abs_max > 0.0 else 1.0
    q = np.clip(np.round(x / scale), -range_max, range_max).astype(np.int8)
    return q, scale
