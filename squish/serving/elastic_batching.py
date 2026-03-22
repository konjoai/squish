"""squish/serving/elastic_batching.py

ElasticBatchController — Adaptive batch sizing based on KV cache headroom.

Each ``tick`` call receives the current KV cache headroom (fraction free) and
the number of requests waiting in the queue.  The controller adjusts the
recommended batch size:

* If headroom < ``low_watermark``:  shrink by ``shrink_step`` (memory pressure).
* If headroom >= ``high_watermark`` and queue_depth >= ``drain_target``:
  grow by ``grow_step`` (spare capacity and demand).
* Otherwise: hold current batch size.

The recommended batch size is clamped to ``[min_batch, max_batch]``.
"""

from __future__ import annotations

__all__ = ["ElasticBatchConfig", "ElasticBatchState", "ElasticBatchController"]

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ElasticBatchConfig:
    """Configuration for ElasticBatchController.

    Parameters
    ----------
    min_batch:
        Minimum batch size.
    max_batch:
        Maximum batch size.
    low_watermark:
        KV headroom fraction below which the batch is shrunk.
    high_watermark:
        KV headroom fraction above which growth is permitted.
    drain_target:
        Minimum queue depth required to trigger a batch-size increase.
    grow_step:
        Number of requests added per grow tick.
    shrink_step:
        Number of requests removed per shrink tick.
    seed:
        RNG seed (reserved for future stochastic extensions).
    """

    min_batch: int = 1
    max_batch: int = 64
    low_watermark: float = 0.1
    high_watermark: float = 0.8
    drain_target: int = 4
    grow_step: int = 4
    shrink_step: int = 2
    seed: int = 0

    def __post_init__(self) -> None:
        if self.min_batch < 1:
            raise ValueError("min_batch must be >= 1")
        if self.max_batch < self.min_batch:
            raise ValueError("max_batch must be >= min_batch")
        if not (0.0 < self.low_watermark < self.high_watermark < 1.0):
            raise ValueError(
                "Must satisfy 0 < low_watermark < high_watermark < 1"
            )
        if self.drain_target < 1:
            raise ValueError("drain_target must be >= 1")
        if self.grow_step < 1:
            raise ValueError("grow_step must be >= 1")
        if self.shrink_step < 1:
            raise ValueError("shrink_step must be >= 1")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class ElasticBatchState:
    """Mutable state for ElasticBatchController.

    Attributes
    ----------
    current_batch_size:
        Current recommended batch size.
    n_shrinks:
        Cumulative number of shrink ticks.
    n_grows:
        Cumulative number of grow ticks.
    n_ticks:
        Total number of ticks processed.
    """

    current_batch_size: int
    n_shrinks: int = 0
    n_grows: int = 0
    n_ticks: int = 0


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ElasticBatchController:
    """Adaptive batch-size controller for LLM serving.

    Parameters
    ----------
    config:
        ``ElasticBatchConfig`` instance.
    """

    def __init__(self, config: ElasticBatchConfig) -> None:
        self.config = config

    def new_state(self) -> ElasticBatchState:
        """Create a fresh state starting at the midpoint batch size."""
        initial = max(
            self.config.min_batch,
            min(self.config.max_batch, self.config.min_batch),
        )
        return ElasticBatchState(current_batch_size=initial)

    def tick(
        self,
        kv_headroom: float,
        queue_depth: int,
        state: ElasticBatchState,
    ) -> Tuple[int, ElasticBatchState]:
        """Update recommended batch size for the current tick.

        Parameters
        ----------
        kv_headroom:
            Fraction of KV cache that is free, in ``[0, 1]``.
        queue_depth:
            Number of requests waiting to be batched.
        state:
            Current ``ElasticBatchState``.

        Returns
        -------
        recommended_batch_size:
            Integer batch size recommendation.
        state:
            Updated state.
        """
        current = state.current_batch_size
        n_shrinks = state.n_shrinks
        n_grows = state.n_grows

        if kv_headroom < self.config.low_watermark:
            # Memory pressure — shrink
            new_size = max(self.config.min_batch, current - self.config.shrink_step)
            n_shrinks += 1
        elif (
            kv_headroom >= self.config.high_watermark
            and queue_depth >= self.config.drain_target
        ):
            # Spare capacity + demand — grow
            new_size = min(self.config.max_batch, current + self.config.grow_step)
            n_grows += 1
        else:
            new_size = current

        new_state = ElasticBatchState(
            current_batch_size=new_size,
            n_shrinks=n_shrinks,
            n_grows=n_grows,
            n_ticks=state.n_ticks + 1,
        )
        return new_size, new_state

    @staticmethod
    def recommended_batch_size(state: ElasticBatchState) -> int:
        """Return the current recommended batch size from state."""
        return state.current_batch_size

    @staticmethod
    def stats(state: ElasticBatchState) -> dict:
        """Return a summary statistics dict."""
        return {
            "current_batch_size": state.current_batch_size,
            "n_shrinks": state.n_shrinks,
            "n_grows": state.n_grows,
            "n_ticks": state.n_ticks,
        }
