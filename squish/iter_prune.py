"""IterPrune — Iterative magnitude-based unstructured pruning with cubic sparsity ramp.

Incrementally increases weight sparsity over ``n_steps`` steps following a
cubic polynomial schedule.  At each step, all weights whose absolute value
falls below the ``sparsity``-quantile threshold are zeroed, implementing
unstructured magnitude pruning.

The cubic schedule from Zhu & Gupta (2018) produces slow initial sparsity
growth followed by rapid convergence toward the target:

    s(t) = s_t - (s_t - s_i) * (1 - (t - t_0) / (t_e - t_0))^3

where ``s_i`` is ``initial_sparsity``, ``s_t`` is ``target_sparsity``,
``t_0`` is ``start_step``, and ``t_e`` is ``end_step``.  This yields
a gradual ramp that avoids destabilising the model early in training.

Reference:
    Zhu & Gupta, "To Prune, or Not to Prune: Exploring the Efficacy of
    Pruning for Model Compression", ICLR 2018.
    https://arxiv.org/abs/1710.01878

Usage::

    import numpy as np
    from squish.iter_prune import PruneSchedule, IterativePruner, IterPruneStats

    schedule = PruneSchedule(
        initial_sparsity=0.0,
        target_sparsity=0.7,
        n_steps=10,
        start_step=0,
    )
    pruner = IterativePruner(schedule)

    rng     = np.random.default_rng(0)
    weights = rng.standard_normal((512, 512)).astype(np.float32)

    for step in range(10):
        weights, actual_sparsity = pruner.prune_step(weights, step)
        print(f"step={step} sparsity={actual_sparsity:.3f}")

    print(pruner.stats)
"""

from __future__ import annotations

__all__ = ["PruneSchedule", "IterativePruner", "IterPruneStats"]

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class PruneSchedule:
    """Cubic polynomial sparsity ramp schedule.

    Attributes:
        initial_sparsity: Sparsity fraction at ``start_step``.  Must be in
            ``[0, 1)``.
        target_sparsity: Final sparsity fraction.  Must be in
            ``(initial_sparsity, 1]``.
        n_steps: Total number of pruning steps planned.
        start_step: First step at which pruning begins.
        end_step: Last step at which sparsity is still increasing.  Defaults
            to ``n_steps - 1`` when ``None``.
    """

    initial_sparsity: float = 0.0
    target_sparsity:  float = 0.7
    n_steps:          int   = 10
    start_step:       int   = 0
    end_step:         Optional[int] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.initial_sparsity < 1.0):
            raise ValueError(
                f"initial_sparsity must be in [0, 1); "
                f"got {self.initial_sparsity}"
            )
        if not (0.0 < self.target_sparsity <= 1.0):
            raise ValueError(
                f"target_sparsity must be in (0, 1]; "
                f"got {self.target_sparsity}"
            )
        if self.target_sparsity < self.initial_sparsity:
            raise ValueError(
                f"target_sparsity ({self.target_sparsity}) must be >= "
                f"initial_sparsity ({self.initial_sparsity})"
            )
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1; got {self.n_steps}")
        if self.end_step is None:
            self.end_step = self.n_steps - 1
        if self.end_step < self.start_step:
            raise ValueError(
                f"end_step ({self.end_step}) must be >= "
                f"start_step ({self.start_step})"
            )

    def current_sparsity(self, step: int) -> float:
        """Compute the target sparsity at *step* using the cubic schedule.

        Args:
            step: Current pruning step (0-indexed).

        Returns:
            Sparsity fraction in ``[initial_sparsity, target_sparsity]``,
            clamped to ``[0.0, 1.0]``.
        """
        # end_step is guaranteed non-None after __post_init__
        end = self.end_step  # type: ignore[assignment]

        if step <= self.start_step:
            return float(self.initial_sparsity)
        if step >= end:
            return float(self.target_sparsity)

        s_i = self.initial_sparsity
        s_t = self.target_sparsity
        t_0 = self.start_step
        t_e = end

        # Cubic: s_t - (s_t - s_i) * (1 - (t - t_0)/(t_e - t_0))^3
        progress  = (step - t_0) / (t_e - t_0)
        sparsity  = s_t - (s_t - s_i) * ((1.0 - progress) ** 3)

        return float(np.clip(sparsity, 0.0, 1.0))


@dataclass
class IterPruneStats:
    """Running statistics for an :class:`IterativePruner` session.

    Attributes:
        total_steps: Number of :meth:`IterativePruner.prune_step` calls.
        total_weights_zeroed: Cumulative count of weight elements set to zero.
        total_weights_processed: Cumulative total count of weight elements seen.
    """

    total_steps:              int = 0
    total_weights_zeroed:     int = 0
    total_weights_processed:  int = 0

    @property
    def avg_sparsity(self) -> float:
        """Mean sparsity fraction across all processed weight elements."""
        return self.total_weights_zeroed / max(1, self.total_weights_processed)


class IterativePruner:
    """Applies an iterative magnitude-pruning step using a :class:`PruneSchedule`.

    Args:
        schedule: :class:`PruneSchedule` controlling the sparsity ramp.
    """

    def __init__(self, schedule: PruneSchedule) -> None:
        self._schedule = schedule
        self._stats    = IterPruneStats()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def prune_step(
        self,
        weights: np.ndarray,
        step:    int,
    ) -> tuple[np.ndarray, float]:
        """Zero the smallest-magnitude weights to reach the scheduled sparsity.

        Args:
            weights: Float32 numpy array of any shape.  The array is not
                modified in place — a new array is returned.
            step: Current training / pruning step (0-indexed).

        Returns:
            Tuple ``(pruned_weights, actual_sparsity)`` where:

            * ``pruned_weights`` — float32 array of the same shape as
              *weights* with the lowest-magnitude weights set to zero.
            * ``actual_sparsity`` — float in ``[0, 1]`` reflecting the
              fraction of zero elements in the returned array.
        """
        weights = np.asarray(weights, dtype=np.float32)
        n_total = weights.size

        target_sparsity = self._schedule.current_sparsity(step)

        # Number of weights to zero
        n_to_zero = int(np.round(target_sparsity * n_total))
        n_to_zero = int(np.clip(n_to_zero, 0, n_total))

        pruned = weights.copy()

        if n_to_zero > 0:
            flat      = np.abs(pruned.ravel())
            # Partition to find the threshold at position n_to_zero - 1
            threshold = float(np.partition(flat, n_to_zero - 1)[n_to_zero - 1])
            # Zero all weights with |w| <= threshold
            pruned[np.abs(pruned) <= threshold] = 0.0

        n_zero = int(np.sum(pruned == 0.0))
        actual_sparsity = n_zero / n_total

        self._stats.total_steps             += 1
        self._stats.total_weights_zeroed    += n_zero
        self._stats.total_weights_processed += n_total

        return pruned, actual_sparsity

    def current_sparsity(self, step: int) -> float:
        """Delegate to :meth:`PruneSchedule.current_sparsity`.

        Args:
            step: Current step index.

        Returns:
            Target sparsity fraction for this step.
        """
        return self._schedule.current_sparsity(step)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def stats(self) -> IterPruneStats:
        """Running pruning statistics."""
        return self._stats
