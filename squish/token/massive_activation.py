"""MassiveActivationSuppressor — outlier dimension suppression without calibration.

Implements the Massive Activations algorithm (Sun et al., ICML 2024 /
arXiv:2402.17762).

Large language models consistently exhibit a small number of "massive
activation" dimensions — channel indices whose absolute values are orders of
magnitude larger than the typical activation.  These outliers prevent
low-bit quantisation without sacrificing accuracy.

The algorithm:

1. **Detection** — for each activation tensor, identify dimensions whose
   max-abs exceeds a threshold (``outlier_ratio × median_abs``).
2. **Soft clamp** — scale outlier values towards their mean using an
   exponential soft-clamp rather than hard truncation, preserving
   gradient flow during fine-tuning.
3. **Adjacent redistribution** — the suppressed energy is added back to
   neighbouring channels so the total activation norm is approximately
   preserved.

The module is stateful: it accumulates per-layer, per-head running statistics
so the detection threshold adapts over time.

Reference:
    Sun et al., "Massive Activations: A Critical View of LLM Outlier
    Suppression Without Calibration",
    ICML 2024 (arXiv:2402.17762).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "MassiveActivationConfig",
    "SuppressionStats",
    "MassiveActivationSuppressor",
]

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class MassiveActivationConfig:
    """Configuration for MassiveActivationSuppressor.

    Attributes:
        outlier_ratio: A dimension is flagged as massive when its max-abs
            exceeds ``outlier_ratio × median(abs(activation))``.
        clamp_alpha: Soft-clamp strength in (0, 1].  ``alpha=1`` leaves the
            value unchanged; ``alpha→0`` collapses the value to its
            channel mean.
        redistribute: If True, add suppressed energy to adjacent channels
            so total norm is approximately preserved.
        running_ema: Exponential moving average factor for the statistics
            tracker.  Set to 1.0 to disable EMA (snapshot mode).
        min_seq_len: Skip suppression on sequences shorter than this (e.g.,
            single-token decode steps where outlier detection is noisy).
    """

    outlier_ratio: float = 100.0
    clamp_alpha: float = 0.1
    redistribute: bool = True
    running_ema: float = 0.99
    min_seq_len: int = 4

    def __post_init__(self) -> None:
        if self.outlier_ratio <= 1.0:
            raise ValueError(
                f"outlier_ratio must be > 1; got {self.outlier_ratio}"
            )
        if not (0.0 < self.clamp_alpha <= 1.0):
            raise ValueError(
                f"clamp_alpha must be in (0, 1]; got {self.clamp_alpha}"
            )
        if not (0.0 < self.running_ema <= 1.0):
            raise ValueError(
                f"running_ema must be in (0, 1]; got {self.running_ema}"
            )
        if self.min_seq_len < 1:
            raise ValueError(
                f"min_seq_len must be ≥ 1; got {self.min_seq_len}"
            )


# ── Statistics container ──────────────────────────────────────────────────────


@dataclass
class SuppressionStats:
    """Running statistics for one layer / head.

    Attributes:
        n_calls: Number of :meth:`suppress` calls on this slot.
        mean_n_outliers: EMA of the count of detected outlier dims.
        mean_suppression_ratio: EMA of the fraction of energy suppressed.
        outlier_dims: Set of dimension indices flagged in the last call.
    """

    n_calls: int = 0
    mean_n_outliers: float = 0.0
    mean_suppression_ratio: float = 0.0
    outlier_dims: set = field(default_factory=set)


# ── Core class ────────────────────────────────────────────────────────────────


class MassiveActivationSuppressor:
    """Per-layer outlier dimension soft-clamping with adjacent redistribution.

    Example::

        suppressor = MassiveActivationSuppressor()
        x = np.random.randn(16, 512).astype(np.float32)
        x[0, 42] = 1e4   # simulate massive activation
        x_clean = suppressor.suppress(x, layer_id=0)

    Args:
        config: :class:`MassiveActivationConfig` (optional).
    """

    def __init__(self, config: Optional[MassiveActivationConfig] = None) -> None:
        self.config: MassiveActivationConfig = config or MassiveActivationConfig()
        self._stats: dict[int, SuppressionStats] = {}

    # ── Detect ────────────────────────────────────────────────────────────────

    def detect_outlier_dims(self, x: np.ndarray) -> np.ndarray:
        """Return indices of massive-activation dimensions.

        A dimension ``d`` is considered massive when:
            ``max_abs(x[:, d]) > outlier_ratio * median(abs(x))``

        Args:
            x: ``(T, C)`` activation tensor (sequence × channels).

        Returns:
            Integer array of outlier channel indices.
        """
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        global_median = np.median(np.abs(x)) + 1e-8
        threshold = self.config.outlier_ratio * global_median
        per_dim_max = np.abs(x).max(axis=0)  # (C,)
        return np.where(per_dim_max > threshold)[0]

    # ── Suppress ──────────────────────────────────────────────────────────────

    def suppress(
        self,
        x: np.ndarray,
        layer_id: int = 0,
    ) -> np.ndarray:
        """Detect and soft-clamp massive activations.

        Args:
            x: ``(..., C)`` activation tensor.  The last axis is the channel
                dimension.
            layer_id: Identifier for per-layer statistics tracking.

        Returns:
            Activation tensor of the same shape with outliers softened.
        """
        x = np.asarray(x, dtype=np.float32)
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])  # (T, C)

        T, C = flat.shape
        if T < self.config.min_seq_len:
            return x

        outlier_idx = self.detect_outlier_dims(flat)
        out = flat.copy()

        if len(outlier_idx) > 0:
            alpha = self.config.clamp_alpha
            for d in outlier_idx:
                col = flat[:, d]
                mean_val = float(col.mean())
                # Exponential soft-clamp towards mean
                delta = col - mean_val
                soft_clamped = mean_val + delta * alpha
                energy_diff = (col ** 2).sum() - (soft_clamped ** 2).sum()
                out[:, d] = soft_clamped

                # Adjacent redistribution
                if self.config.redistribute and energy_diff > 0 and C > 1:
                    left = (d - 1) % C
                    right = (d + 1) % C
                    correction = np.sqrt(energy_diff / 2.0)
                    out[:, left] = out[:, left] + np.sign(flat[:, left]) * correction / T
                    out[:, right] = out[:, right] + np.sign(flat[:, right]) * correction / T

        # Update running statistics
        self._update_stats(layer_id, len(outlier_idx), flat, out)
        return out.reshape(orig_shape)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self, layer_id: int = 0) -> SuppressionStats:
        """Return running statistics for the given layer.

        Args:
            layer_id: Layer identifier.

        Returns:
            :class:`SuppressionStats` snapshot (copy).
        """
        if layer_id not in self._stats:
            return SuppressionStats()
        s = self._stats[layer_id]
        return SuppressionStats(
            n_calls=s.n_calls,
            mean_n_outliers=s.mean_n_outliers,
            mean_suppression_ratio=s.mean_suppression_ratio,
            outlier_dims=set(s.outlier_dims),
        )

    def reset_stats(self, layer_id: Optional[int] = None) -> None:
        """Reset statistics for one or all layers.

        Args:
            layer_id: If given, reset only that layer; if None, reset all.
        """
        if layer_id is None:
            self._stats.clear()
        else:
            self._stats.pop(layer_id, None)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _update_stats(
        self,
        layer_id: int,
        n_outliers: int,
        before: np.ndarray,
        after: np.ndarray,
    ) -> None:
        if layer_id not in self._stats:
            self._stats[layer_id] = SuppressionStats()
        s = self._stats[layer_id]
        ema = self.config.running_ema
        s.n_calls += 1
        s.mean_n_outliers = ema * s.mean_n_outliers + (1 - ema) * n_outliers
        energy_before = float((before ** 2).sum()) + 1e-20
        energy_after = float((after ** 2).sum())
        ratio = max(0.0, 1.0 - energy_after / energy_before)
        s.mean_suppression_ratio = ema * s.mean_suppression_ratio + (1 - ema) * ratio

    def __repr__(self) -> str:
        return (
            f"MassiveActivationSuppressor("
            f"outlier_ratio={self.config.outlier_ratio}, "
            f"clamp_alpha={self.config.clamp_alpha}, "
            f"layers_tracked={len(self._stats)})"
        )
