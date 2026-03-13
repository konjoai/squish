# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""PipelineBubble — Pipeline-parallel bubble elimination for LLM serving.

In tensor-parallel + pipeline-parallel serving, pipeline bubbles occur when
decoder stages are idle waiting for the next micro-batch.  This module models
a 1F1B (one-forward-one-backward) interleaved schedule across pipeline stages
and computes effective utilisation and bubble fraction.

The 1F1B schedule minimises pipeline bubbles compared to a naive fill-drain
schedule:

    - Naive schedule: bubble fraction = (n_stages - 1) / n_microbatches
    - 1F1B schedule:  bubble fraction = (n_stages - 1) /
                      (n_microbatches + n_stages - 1)

Reference:
    Narayanan et al., "Efficient Large-Scale Language Model Training on GPU
    Clusters Using Megatron-LM", SC 2021.
    https://arxiv.org/abs/2104.04473

Usage::

    from squish.hardware.pipeline_bubble import BubbleEliminator, StageConfig
    import numpy as np

    cfg   = StageConfig(n_stages=4, n_microbatches=8, stage_latency_ms=10.0)
    elim  = BubbleEliminator(cfg)
    sched = elim.build_schedule()
    sim   = elim.simulate(sched)
    print(f"bubble={sched.bubble_fraction:.2%}, "
          f"throughput={sim['throughput_mbatch_per_ms']:.3f} mb/ms")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

__all__ = [
    "StageConfig",
    "StageSchedule",
    "BubbleEliminator",
    "BubbleStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class StageConfig:
    """Configuration for a pipeline-parallel stage schedule.

    Attributes:
        n_stages:          Number of pipeline stages (workers in sequence).
        n_microbatches:    Number of micro-batches to schedule.  Must be
                           >= ``n_stages`` for a well-utilised pipeline.
        stage_latency_ms:  Per-stage forward processing time in milliseconds.
                           Assumed uniform across all stages and micro-batches.
    """

    n_stages: int = 4
    n_microbatches: int = 8
    stage_latency_ms: float = 10.0

    def __post_init__(self) -> None:
        if self.n_stages < 1:
            raise ValueError(
                f"n_stages must be >= 1; got {self.n_stages}"
            )
        if self.n_microbatches < 1:
            raise ValueError(
                f"n_microbatches must be >= 1; got {self.n_microbatches}"
            )
        if self.stage_latency_ms <= 0.0:
            raise ValueError(
                f"stage_latency_ms must be > 0; got {self.stage_latency_ms}"
            )

    @property
    def theoretical_bubble_fraction(self) -> float:
        """Theoretical 1F1B bubble fraction formula.

        Returns ``(n_stages - 1) / (n_microbatches + n_stages - 1)``.
        """
        return (self.n_stages - 1) / (
            self.n_microbatches + self.n_stages - 1
        )


# ---------------------------------------------------------------------------
# StageSchedule
# ---------------------------------------------------------------------------


@dataclass
class StageSchedule:
    """A concrete pipeline schedule with slot assignments.

    The schedule is represented as a 2-D list where
    ``schedule[stage][slot] = microbatch_id`` (0-indexed) or ``-1`` for a
    pipeline bubble (idle slot).

    Attributes:
        schedule: ``list[list[int]]`` of shape
                  ``(n_stages, n_slots)`` where ``n_slots`` is the total
                  number of time slots required to drain the pipeline.
    """

    schedule: List[List[int]]

    @property
    def bubble_fraction(self) -> float:
        """Fraction of schedule slots that are pipeline bubbles.

        Computed as ``n_bubble_slots / total_slots``.
        Returns 0.0 for an empty schedule.
        """
        if not self.schedule:
            return 0.0
        total = sum(len(row) for row in self.schedule)
        if total == 0:
            return 0.0
        bubbles = sum(
            sum(1 for s in row if s == -1) for row in self.schedule
        )
        return bubbles / total

    @property
    def n_stages(self) -> int:
        """Number of pipeline stages in this schedule."""
        return len(self.schedule)

    @property
    def n_slots(self) -> int:
        """Number of time slots (columns) in the schedule."""
        if not self.schedule:
            return 0
        return max(len(row) for row in self.schedule)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class BubbleStats:
    """Cumulative statistics for :class:`BubbleEliminator`.

    Attributes:
        n_schedules_built:    Number of :meth:`~BubbleEliminator.build_schedule`
                              calls made.
        best_bubble_fraction: Lowest bubble fraction seen across all built
                              schedules.  Initialised to 1.0.
    """

    n_schedules_built: int = 0
    best_bubble_fraction: float = 1.0


# ---------------------------------------------------------------------------
# BubbleEliminator
# ---------------------------------------------------------------------------


class BubbleEliminator:
    """1F1B pipeline schedule builder and simulator.

    Constructs an interleaved pipeline schedule that minimises bubbles via
    the standard 1F1B approach: the first ``n_stages - 1`` warmup slots fill
    the pipeline one stage at a time; the steady-state phase processes one
    micro-batch per stage per cycle; the cooldown drains the pipeline.

    The resulting schedule is suitable for pure decoder (forward-only)
    inference pipelines.

    Args:
        config: :class:`StageConfig` instance.
    """

    def __init__(self, config: StageConfig) -> None:
        self._config = config
        self._stats = BubbleStats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_schedule(self) -> StageSchedule:
        """Build a 1F1B interleaved pipeline schedule.

        The schedule is constructed by tracking, for each stage, when the
        next available time slot begins (based on the previous stage's
        completion and its own last-used slot).  Each micro-batch propagates
        through all stages in order.

        Returns:
            :class:`StageSchedule` with all micro-batch assignments and
            explicit ``-1`` bubble markers.
        """
        cfg = self._config
        n_stages = cfg.n_stages
        n_mb = cfg.n_microbatches

        # stage_finish[s] = earliest slot at which stage s can next accept work.
        stage_finish: list[int] = [0] * n_stages
        # mb_finish[mb][s] = slot index at which micro-batch mb finishes stage s.
        mb_finish: list[list[int]] = [[-1] * n_stages for _ in range(n_mb)]

        for mb in range(n_mb):
            for s in range(n_stages):
                # Stage s can start when:
                # (a) the previous stage has finished this micro-batch, AND
                # (b) this stage has finished its previous micro-batch.
                if s == 0:
                    earliest_from_prev = mb  # pipeline can feed new mb every slot
                else:
                    earliest_from_prev = mb_finish[mb][s - 1]

                start = max(stage_finish[s], earliest_from_prev)
                finish = start + 1  # each stage takes 1 slot (normalised)
                mb_finish[mb][s] = finish
                stage_finish[s] = finish

        # Total slots needed = max finish across all stages for all micro-batches.
        total_slots = max(mb_finish[mb][s] for mb in range(n_mb) for s in range(n_stages))

        # Build the schedule array: schedule[stage][slot] = mb_id or -1.
        schedule: list[list[int]] = [
            [-1] * total_slots for _ in range(n_stages)
        ]

        for mb in range(n_mb):
            for s in range(n_stages):
                finish = mb_finish[mb][s]
                start_slot = finish - 1  # 1-slot duration, so start = finish - 1
                schedule[s][start_slot] = mb

        result = StageSchedule(schedule=schedule)

        self._stats.n_schedules_built += 1
        if result.bubble_fraction < self._stats.best_bubble_fraction:
            self._stats.best_bubble_fraction = result.bubble_fraction

        return result

    def simulate(self, schedule: StageSchedule) -> Dict[str, float]:
        """Simulate the wall-clock performance of a pipeline schedule.

        Args:
            schedule: :class:`StageSchedule` produced by
                      :meth:`build_schedule`.

        Returns:
            Dictionary with the following keys:

            ``total_time_ms``
                Wall-clock time from first micro-batch entering stage 0 to
                last micro-batch leaving the final stage.
            ``bubble_fraction``
                Fraction of schedule slots that are pipeline bubbles.
            ``throughput_mbatch_per_ms``
                Effective micro-batch throughput in micro-batches per
                millisecond.

        Raises:
            ValueError: if ``schedule.schedule`` is empty.
        """
        if not schedule.schedule:
            raise ValueError("schedule must be non-empty")

        cfg = self._config
        n_slots = schedule.n_slots
        total_time_ms = float(n_slots) * cfg.stage_latency_ms
        bubble_frac = schedule.bubble_fraction

        n_mb = cfg.n_microbatches
        throughput = n_mb / total_time_ms if total_time_ms > 0.0 else 0.0

        return {
            "total_time_ms": total_time_ms,
            "bubble_fraction": bubble_frac,
            "throughput_mbatch_per_ms": throughput,
        }

    @property
    def stats(self) -> BubbleStats:
        """Cumulative schedule build statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset cumulative statistics to their initial values."""
        self._stats = BubbleStats()
