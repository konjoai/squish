"""
squish/speculative/draft_multiplexer.py

DraftMultiplexer — Intelligent runtime draft-strategy selection.

Key insight
-----------
Five draft families exist in squish (EAGLE-3, N-gram, Medusa, ConfSpec,
LayerSkip).  Currently the user or startup flag selects a single strategy for
the entire server lifetime.

``DraftMultiplexer`` maintains a per-task-type exponential moving average
(EMA) of each strategy's *acceptance rate* and *effective cost* (tokens per
wall-clock second), then selects the strategy with the best expected
``acceptance_rate / cost`` ratio for each incoming request.

Task types are classified from prompt-surface features (no model call) into
four categories: ``conversation``, ``rag``, ``coding``, ``math``.

The EMA is updated asynchronously after each request completes, using a
background queue so the selection path has negligible latency.

Usage::

    from squish.speculative.draft_multiplexer import (
        DraftMultiplexerConfig,
        DraftMultiplexer,
        DraftStrategy,
    )

    cfg = DraftMultiplexerConfig(
        strategies=["eagle3", "ngram", "layer_skip"],
        ema_alpha=0.1,
        min_samples=5,
    )
    mux = DraftMultiplexer(cfg)

    # On each new request:
    strategy = mux.select(prompt_tokens)
    print(f"Selected: {strategy}")   # e.g. DraftStrategy.EAGLE3

    # After the request completes:
    mux.update(strategy, task_type="coding", acceptance_rate=0.72, tps=48.3)
"""

from __future__ import annotations

__all__ = [
    "DraftStrategy",
    "DraftTaskType",
    "DraftMultiplexerConfig",
    "DraftMultiplexer",
    "StrategyStats",
]

import math
import re
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DraftStrategy(Enum):
    """Known draft strategy identifiers."""
    EAGLE3     = "eagle3"
    NGRAM      = "ngram"
    MEDUSA     = "medusa"
    CONF_SPEC  = "conf_spec"
    LAYER_SKIP = "layer_skip"
    NONE       = "none"


class DraftTaskType(Enum):
    """Coarse task classification used for per-task EMA tracking."""
    CONVERSATION = "conversation"
    RAG          = "rag"
    CODING       = "coding"
    MATH         = "math"
    UNKNOWN      = "unknown"


# ---------------------------------------------------------------------------
# Per-strategy statistics
# ---------------------------------------------------------------------------

@dataclass
class StrategyStats:
    """EMA-tracked statistics for one (strategy, task_type) pair.

    Attributes
    ----------
    acceptance_rate : float
        EMA of the fraction of draft tokens accepted by the verifier.
    tps : float
        EMA of accepted tokens per second (wall-clock).
    n_samples : int
        Number of request completions recorded.
    """

    acceptance_rate: float = 0.0
    tps:             float = 0.0
    n_samples:       int   = 0

    @property
    def score(self) -> float:
        """Combined selection score: acceptance_rate * tps."""
        return self.acceptance_rate * max(self.tps, 1e-6)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DraftMultiplexerConfig:
    """Configuration for DraftMultiplexer.

    Parameters
    ----------
    strategies : list[str]
        Allowed strategy names from :class:`DraftStrategy`.
    ema_alpha : float
        Smoothing factor for exponential moving averages (0 < α ≤ 1).
        Larger → more responsive; smaller → more stable.
    min_samples : int
        Minimum number of completed requests before EMA-based selection
        replaces the round-robin initialisation phase.
    default_strategy : str
        Strategy to use when no EMA data is available yet.
    cost_weight : float
        Weight for throughput (tps) relative to acceptance rate.
        ``score = acceptance_rate + cost_weight * normalised_tps``
    """

    strategies:       List[str] = field(
        default_factory=lambda: ["eagle3", "ngram", "layer_skip"]
    )
    ema_alpha:        float     = 0.1
    min_samples:      int       = 5
    default_strategy: str       = "eagle3"
    cost_weight:      float     = 0.3

    def __post_init__(self) -> None:
        valid = {s.value for s in DraftStrategy}
        for s in self.strategies:
            if s not in valid:
                raise ValueError(
                    f"Unknown strategy {s!r}; valid: {sorted(valid)}"
                )
        if not (0.0 < self.ema_alpha <= 1.0):
            raise ValueError(
                f"ema_alpha must be in (0, 1]; got {self.ema_alpha}"
            )
        if self.min_samples < 1:
            raise ValueError(
                f"min_samples must be ≥ 1; got {self.min_samples}"
            )
        if self.default_strategy not in {s.value for s in DraftStrategy}:
            raise ValueError(
                f"default_strategy {self.default_strategy!r} is not a valid "
                "DraftStrategy value"
            )


# ---------------------------------------------------------------------------
# Task classifier
# ---------------------------------------------------------------------------

_CODE_PATTERN = re.compile(
    r"```|def |class |import |#include|function |var |let |const |=>|->|\{|\}",
    re.IGNORECASE,
)
_MATH_PATTERN = re.compile(
    r"solve|calculate|integral|derivative|\\frac|\\sum|equation|theorem|proof",
    re.IGNORECASE,
)
_RAG_PATTERN  = re.compile(
    r"according to|based on|the document|the passage|context:|source:",
    re.IGNORECASE,
)


def classify_task(prompt: str) -> DraftTaskType:
    """Classify a prompt into a :class:`DraftTaskType` using regex heuristics.

    Parameters
    ----------
    prompt : str
        Raw text prompt (first 1 KB is sufficient).
    """
    sample = prompt[:1024]
    if _CODE_PATTERN.search(sample):
        return DraftTaskType.CODING
    if _MATH_PATTERN.search(sample):
        return DraftTaskType.MATH
    if _RAG_PATTERN.search(sample):
        return DraftTaskType.RAG
    return DraftTaskType.CONVERSATION


def classify_task_from_ids(
    token_ids: List[int],
    id_to_piece: Optional[Dict[int, str]] = None,
) -> DraftTaskType:
    """Classify task type from token IDs using a vocabulary heuristic.

    When ``id_to_piece`` is not provided, falls back to UNKNOWN.
    """
    if id_to_piece is None:
        return DraftTaskType.UNKNOWN
    sample = "".join(id_to_piece.get(t, "") for t in token_ids[:256])
    return classify_task(sample)


# ---------------------------------------------------------------------------
# Multiplexer
# ---------------------------------------------------------------------------

class DraftMultiplexer:
    """Select a draft strategy dynamically based on EMA acceptance statistics.

    Thread-safe: ``update()`` acquires a lock; ``select()`` reads without
    locking (EMA floats are updated atomically enough for Python's GIL).

    Parameters
    ----------
    config : DraftMultiplexerConfig
    """

    def __init__(self, config: DraftMultiplexerConfig) -> None:
        self._cfg    = config
        self._lock   = threading.Lock()
        # stats[task_type][strategy] → StrategyStats
        self._stats: Dict[str, Dict[str, StrategyStats]] = {
            tt.value: {s: StrategyStats() for s in config.strategies}
            for tt in DraftTaskType
        }
        # Round-robin cursor for initialisation phase
        self._rr_cursor: Dict[str, int] = {tt.value: 0 for tt in DraftTaskType}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select(
        self,
        prompt: Optional[str] = None,
        token_ids: Optional[List[int]] = None,
        task_type: Optional[str] = None,
    ) -> DraftStrategy:
        """Select the best draft strategy for an incoming request.

        At least one of *prompt*, *token_ids*, or *task_type* should be
        provided.  Task type classification priority:
        ``task_type`` > ``prompt`` > ``token_ids`` > ``UNKNOWN``.

        Parameters
        ----------
        prompt : str | None
            Raw prompt text for task classification.
        token_ids : list[int] | None
            Token IDs (used when prompt text is not available).
        task_type : str | None
            Pre-classified task type; skips heuristic classification.

        Returns
        -------
        DraftStrategy
        """
        tt = self._resolve_task(prompt, token_ids, task_type)
        tt_key = tt.value
        stats_map = self._stats[tt_key]

        # Initialisation phase: round-robin until min_samples
        min_n = min(s.n_samples for s in stats_map.values())
        if min_n < self._cfg.min_samples:
            return self._round_robin(tt_key)

        # EMA phase: pick highest score
        best_name  = self._cfg.default_strategy
        best_score = -1.0
        # Normalise tps for fair comparison
        all_tps = [s.tps for s in stats_map.values() if s.tps > 0]
        max_tps = max(all_tps) if all_tps else 1.0

        for name, stats in stats_map.items():
            score = (
                stats.acceptance_rate
                + self._cfg.cost_weight * (stats.tps / max_tps)
            )
            if score > best_score:
                best_score = score
                best_name  = name

        try:
            return self._apply_eagle_fallback(DraftStrategy(best_name))
        except ValueError:
            return DraftStrategy(self._cfg.default_strategy)

    def update(
        self,
        strategy: DraftStrategy,
        task_type: str,
        acceptance_rate: float,
        tps: float,
    ) -> None:
        """Record the outcome of a completed request and update EMA.

        Parameters
        ----------
        strategy : DraftStrategy
            The strategy that was used.
        task_type : str
            Task type (``DraftTaskType.value`` string).
        acceptance_rate : float
            Fraction of draft tokens accepted (0–1).
        tps : float
            Accepted tokens per second.
        """
        α = self._cfg.ema_alpha
        name = strategy.value
        if task_type not in self._stats:
            task_type = DraftTaskType.UNKNOWN.value
        stats_map = self._stats[task_type]
        if name not in stats_map:
            return

        with self._lock:
            s = stats_map[name]
            if s.n_samples == 0:
                s.acceptance_rate = acceptance_rate
                s.tps             = tps
            else:
                s.acceptance_rate = (1 - α) * s.acceptance_rate + α * acceptance_rate
                s.tps             = (1 - α) * s.tps + α * tps
            s.n_samples += 1

    def strategy_stats(
        self, task_type: str = "conversation"
    ) -> Dict[str, StrategyStats]:
        """Return a snapshot of strategy statistics for *task_type*."""
        return {k: v for k, v in self._stats.get(task_type, {}).items()}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_task(
        self,
        prompt: Optional[str],
        token_ids: Optional[List[int]],
        task_type: Optional[str],
    ) -> DraftTaskType:
        if task_type is not None:
            try:
                return DraftTaskType(task_type)
            except ValueError:
                pass
        if prompt is not None:
            return classify_task(prompt)
        return DraftTaskType.UNKNOWN

    def _round_robin(self, tt_key: str) -> DraftStrategy:
        strategies = self._cfg.strategies
        idx = self._rr_cursor[tt_key] % len(strategies)
        self._rr_cursor[tt_key] = (idx + 1) % len(strategies)
        try:
            strategy = DraftStrategy(strategies[idx])
        except ValueError:
            strategy = DraftStrategy(self._cfg.default_strategy)
        return self._apply_eagle_fallback(strategy)

    # ------------------------------------------------------------------
    # EAGLE integration (Wave 68)
    # ------------------------------------------------------------------

    def register_eagle_runner(self, runner: Optional[object]) -> None:
        """Register an :class:`~squish.speculative.eagle_head.EAGLEHeadRunner`.

        When a runner is registered, :meth:`select` applies a rolling-window
        fallback: if the runner's ``should_fallback()`` returns ``True`` while
        EAGLE3 is the selected strategy, the multiplexer downgrades the
        selection to NGRAM automatically.

        Pass ``None`` to unregister any previously registered runner.

        Args:
            runner: An :class:`~squish.speculative.eagle_head.EAGLEHeadRunner`
                instance or any object exposing a ``should_fallback() -> bool``
                method.  ``None`` clears the registration.
        """
        self._eagle_runner: Optional[object] = runner

    def _apply_eagle_fallback(self, strategy: DraftStrategy) -> DraftStrategy:
        """If the registered EAGLE runner signals fallback, downgrade to NGRAM.

        Only activates when *strategy* is :attr:`~DraftStrategy.EAGLE3` and an
        EAGLE runner is registered.
        """
        if strategy is not DraftStrategy.EAGLE3:
            return strategy
        runner = getattr(self, "_eagle_runner", None)
        if runner is None:
            return strategy
        try:
            if runner.should_fallback():  # type: ignore[attr-defined]
                return DraftStrategy.NGRAM
        except Exception:
            pass
        return strategy
