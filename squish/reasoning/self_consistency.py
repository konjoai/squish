"""SelfConsistencyVoter: majority-vote aggregation over chain-of-thought paths.

Wang et al. (ICLR 2023) showed that sampling diverse reasoning chains and
taking the most-consistent final answer reliably outperforms greedy decoding.
This module provides the aggregation layer: given a list of chain-of-thought
completion strings it (1) extracts the final answer from each chain using a
configurable heuristic, (2) counts votes, and (3) returns the majority winner.

Reference: Wang et al., "Self-Consistency Improves Chain of Thought Reasoning
in Language Models", ICLR 2023.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "SelfConsistencyConfig",
    "SelfConsistencyResult",
    "SelfConsistencyVoter",
]


@dataclass
class SelfConsistencyConfig:
    """Configuration for :class:`SelfConsistencyVoter`.

    Attributes:
        k: Expected number of chains to aggregate (informational only;
            ``vote()`` accepts any non-empty list).
        temperature: Sampling temperature (informational; generation not done here).
        answer_pattern: Optional regex to extract the final answer from a chain.
            Captured group 1 is used.  If None, the last non-empty line is used.
        normalise_answers: Normalise extracted answers (strip, lower-case, collapse
            whitespace) before voting.
        seed: RNG seed (unused currently; reserved for future tie-breaking).
    """

    k: int = 8
    temperature: float = 0.9
    answer_pattern: Optional[str] = None
    normalise_answers: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.k < 1:
            raise ValueError(f"k must be ≥ 1, got {self.k}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


@dataclass
class SelfConsistencyResult:
    """Output of one :meth:`SelfConsistencyVoter.vote` call.

    Attributes:
        winner: The majority-vote answer string.
        vote_counts: Mapping from answer string to vote count.
        chains: Original chain-of-thought strings.
        extracted_answers: Answers extracted from each chain.
    """

    winner: str
    vote_counts: Dict[str, int]
    chains: List[str]
    extracted_answers: List[str]

    @property
    def n_chains(self) -> int:
        return len(self.chains)

    @property
    def winner_vote_share(self) -> float:
        total = sum(self.vote_counts.values())
        return self.vote_counts.get(self.winner, 0) / max(total, 1)


class SelfConsistencyVoter:
    """Extract final answers from CoT chains and pick the majority winner.

    Usage::

        cfg = SelfConsistencyConfig(k=8)
        voter = SelfConsistencyVoter(cfg)
        result = voter.vote(chains)
        print(result.winner, result.vote_counts)
    """

    def __init__(self, config: SelfConsistencyConfig) -> None:
        self.config = config
        self._pattern: Optional[re.Pattern[str]] = (
            re.compile(config.answer_pattern, re.IGNORECASE)
            if config.answer_pattern
            else None
        )

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def vote(self, chains: List[str]) -> SelfConsistencyResult:
        """Aggregate *chains* into a majority-vote answer.

        Parameters
        ----------
        chains:
            List of chain-of-thought completion strings.
        """
        if not chains:
            raise ValueError("chains must be non-empty")

        extracted = [self.extract_answer(c) for c in chains]
        vote_counts = self._count_votes(extracted)
        winner = self.majority_vote(vote_counts)

        return SelfConsistencyResult(
            winner=winner,
            vote_counts=vote_counts,
            chains=chains,
            extracted_answers=extracted,
        )

    def extract_answer(self, chain: str) -> str:
        """Extract the final answer from a single chain.

        Uses ``config.answer_pattern`` if provided, otherwise the last
        non-empty line.
        """
        if self._pattern is not None:
            m = self._pattern.search(chain)
            if m:
                raw = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                return self._normalise(raw)

        # Fallback: last non-empty line
        lines = [ln.strip() for ln in chain.splitlines() if ln.strip()]
        raw = lines[-1] if lines else chain.strip()
        return self._normalise(raw)

    def majority_vote(self, vote_counts: Dict[str, int]) -> str:
        """Return the answer with the most votes (ties broken alphabetically)."""
        if not vote_counts:
            raise ValueError("vote_counts must be non-empty")
        return max(vote_counts.keys(), key=lambda a: (vote_counts[a], a))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _count_votes(self, answers: List[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for ans in answers:
            counts[ans] = counts.get(ans, 0) + 1
        return counts

    def _normalise(self, text: str) -> str:
        if not self.config.normalise_answers:
            return text
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text
