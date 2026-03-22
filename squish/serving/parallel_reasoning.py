"""ParallelReasoningScheduler: dispatch multiple reasoning chains concurrently.

For hard problems it is effective to run several independent reasoning chains
in parallel and aggregate with self-consistency or Best-of-N.  This module
implements the scheduling layer: estimate difficulty from a first-pass sample
(or a caller-supplied difficulty score), decide on how many chains to run, and
perform synchronous aggregation.

References:
  Wang et al., "Self-Consistency Improves Chain of Thought Reasoning in
  Language Models", ICLR 2023.
  Snell et al., arXiv 2408.03314, 2024.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ParallelReasoningConfig",
    "ParallelReasoningRequest",
    "ParallelReasoningResult",
    "ParallelReasoningScheduler",
]

GenerateFn = Callable[[str, int], List[str]]   # (prompt, n_chains) → [completion]
AggMethod = str  # "self_consistency" | "best_of_n"


@dataclass
class ParallelReasoningConfig:
    """Configuration for :class:`ParallelReasoningScheduler`.

    Attributes:
        max_chains: Maximum parallel chains to run for the hardest problems.
        min_chains: Minimum chains to run even for the easiest problems.
        aggregation: Default aggregation method.
        easy_threshold: Difficulty score below which ``min_chains`` is used.
        hard_threshold: Difficulty score above which ``max_chains`` is used.
        chain_budget_tokens: Per-chain token budget.
        seed: RNG seed.
    """

    max_chains: int = 8
    min_chains: int = 1
    aggregation: str = "self_consistency"
    easy_threshold: float = 1.0
    hard_threshold: float = 3.0
    chain_budget_tokens: int = 512
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_chains < 1:
            raise ValueError(f"max_chains must be ≥ 1, got {self.max_chains}")
        if self.min_chains < 1:
            raise ValueError(f"min_chains must be ≥ 1, got {self.min_chains}")
        if self.min_chains > self.max_chains:
            raise ValueError(
                f"min_chains ({self.min_chains}) must be ≤ max_chains ({self.max_chains})"
            )
        if self.hard_threshold <= self.easy_threshold:
            raise ValueError(
                f"hard_threshold ({self.hard_threshold}) must be > "
                f"easy_threshold ({self.easy_threshold})"
            )
        valid = {"self_consistency", "best_of_n"}
        if self.aggregation not in valid:
            raise ValueError(
                f"aggregation must be one of {valid}, got {self.aggregation!r}"
            )


@dataclass
class ParallelReasoningRequest:
    """A single reasoning request.

    Attributes:
        prompt: Input problem to solve.
        n_chains: Override number of chains (None = auto-dispatch).
        request_id: Unique identifier (auto-generated if empty).
    """

    prompt: str
    n_chains: Optional[int] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ParallelReasoningResult:
    """Result of one scheduled reasoning request.

    Attributes:
        winner: Aggregated best answer.
        chains: All generated chain strings.
        vote_counts: Answer → count mapping (for self-consistency).
        request_id: ID from the originating request.
        wall_seconds: Elapsed wall time.
    """

    winner: str
    chains: List[str]
    vote_counts: Dict[str, int]
    request_id: str
    wall_seconds: float

    @property
    def n_chains(self) -> int:
        return len(self.chains)


class ParallelReasoningScheduler:
    """Dispatch and aggregate parallel reasoning chains.

    Usage::

        cfg = ParallelReasoningConfig(max_chains=8, aggregation="self_consistency")
        scheduler = ParallelReasoningScheduler(cfg)
        req = ParallelReasoningRequest(prompt="Solve: 2+2=?")
        result = scheduler.schedule(req, generate_fn=my_generate)
        print(result.winner)
    """

    def __init__(self, config: ParallelReasoningConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def dispatch(self, difficulty_score: float) -> int:
        """Compute how many chains to run given a *difficulty_score*.

        Linearly interpolates between ``min_chains`` and ``max_chains``.
        """
        lo = self.config.easy_threshold
        hi = self.config.hard_threshold
        frac = float(np.clip((difficulty_score - lo) / (hi - lo), 0.0, 1.0))
        n = int(round(self.config.min_chains + frac * (self.config.max_chains - self.config.min_chains)))
        return int(np.clip(n, self.config.min_chains, self.config.max_chains))

    def aggregate(
        self,
        chains: List[str],
        method: Optional[str] = None,
    ) -> Tuple[str, Dict[str, int]]:
        """Aggregate *chains* into a single winner.

        Parameters
        ----------
        chains:
            List of chain-of-thought completion strings.
        method:
            ``"self_consistency"`` or ``"best_of_n"``.  Defaults to
            ``config.aggregation``.

        Returns
        -------
        (winner, vote_counts):
            The winning answer string and per-answer vote map.
        """
        method = method or self.config.aggregation
        if not chains:
            raise ValueError("chains must be non-empty")

        # Extract last non-empty line as answer
        answers = [self._extract_answer(c) for c in chains]
        counts: Dict[str, int] = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1

        if method == "self_consistency":
            winner = max(counts.keys(), key=lambda a: (counts[a], a))
        else:
            # best_of_n: prefer most frequent; same as majority vote here
            winner = max(counts.keys(), key=lambda a: (counts[a], a))

        return winner, counts

    def schedule(
        self,
        request: ParallelReasoningRequest,
        generate_fn: GenerateFn,
        difficulty_score: float = 2.0,
    ) -> ParallelReasoningResult:
        """Generate and aggregate chains for *request*.

        Parameters
        ----------
        request:
            The reasoning request.
        generate_fn:
            Callable ``(prompt, n) -> List[completion_str]``.
        difficulty_score:
            Pre-computed difficulty; used only when ``request.n_chains`` is None.
        """
        start = time.monotonic()
        n = request.n_chains if request.n_chains is not None else self.dispatch(difficulty_score)
        chains = generate_fn(request.prompt, n)
        if not chains:
            chains = ["[no output]"]
        winner, vote_counts = self.aggregate(chains, method=self.config.aggregation)
        elapsed = time.monotonic() - start
        return ParallelReasoningResult(
            winner=winner,
            chains=chains,
            vote_counts=vote_counts,
            request_id=request.request_id,
            wall_seconds=elapsed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(chain: str) -> str:
        """Return the last non-empty line as the extracted answer."""
        lines = [ln.strip() for ln in chain.splitlines() if ln.strip()]
        return lines[-1].lower() if lines else chain.strip().lower()
