"""PRMBeamSearch: step-level beam search guided by a process reward model.

Wang et al. (arXiv 2312.08935, NeurIPS 2024).  At each reasoning step the
beam is expanded, every candidate is scored by the PRM, and only the top
*beam_width* survivors advance.  The combined score blends step-level PRM
reward with the generator log-probability so that beam pruning accounts for
both reasoning quality and fluency.

Reference: Wang et al., "Math-Shepherd: Verify and Reinforce LLMs Step-by-Step
without Human Annotations", arXiv 2312.08935, NeurIPS 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "PRMBeamConfig",
    "PRMBeamCandidate",
    "PRMBeamResult",
    "PRMBeamSearch",
]


@dataclass
class PRMBeamConfig:
    """Configuration for :class:`PRMBeamSearch`.

    Attributes:
        beam_width: Maximum number of active candidates per step.
        max_steps: Maximum reasoning steps before forced termination.
        step_boundary: Token sequence that marks a reasoning-step boundary.
        prm_weight: Weight applied to the mean PRM step score.
        token_prob_weight: Weight applied to the accumulated log-probability.
        seed: RNG seed for expansion simulation.
    """

    beam_width: int = 8
    max_steps: int = 32
    step_boundary: str = "\n\n"
    prm_weight: float = 0.7
    token_prob_weight: float = 0.3
    seed: int = 0

    def __post_init__(self) -> None:
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be ≥ 1, got {self.beam_width}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be ≥ 1, got {self.max_steps}")
        total = self.prm_weight + self.token_prob_weight
        if not (0.999 <= total <= 1.001):
            raise ValueError(
                f"prm_weight + token_prob_weight must sum to 1.0, got {total:.4f}"
            )


@dataclass
class PRMBeamCandidate:
    """One candidate solution path in the beam.

    Attributes:
        tokens: Token-id sequence produced so far.
        prm_scores: Per-step PRM reward scores.
        log_prob: Accumulated generator log-probability.
        depth: Current reasoning-step depth.
        answer: Extracted final answer (empty until terminal).
    """

    tokens: List[int] = field(default_factory=list)
    prm_scores: List[float] = field(default_factory=list)
    log_prob: float = 0.0
    depth: int = 0
    answer: str = ""

    @property
    def mean_prm_score(self) -> float:
        """Mean PRM score across all steps (0 if none yet)."""
        return float(np.mean(self.prm_scores)) if self.prm_scores else 0.0

    def combined_score(self, prm_weight: float, token_prob_weight: float) -> float:
        """Weighted combination of PRM quality and generator probability."""
        return (
            prm_weight * self.mean_prm_score
            + token_prob_weight * float(np.tanh(-self.log_prob))
        )


@dataclass
class PRMBeamResult:
    """Result of one :meth:`PRMBeamSearch.search` call.

    Attributes:
        best_candidate: Candidate with the highest combined score.
        all_candidates: All surviving candidates sorted best-first.
        n_steps_taken: Actual number of step-iterations performed.
    """

    best_candidate: PRMBeamCandidate
    all_candidates: List[PRMBeamCandidate]
    n_steps_taken: int

    @property
    def best_answer(self) -> str:
        return self.best_candidate.answer


# ---------------------------------------------------------------------------
# Callable type aliases
# ---------------------------------------------------------------------------
PRMScorer = Callable[[List[int], int], float]   # (tokens, step) → score
ExpandFn = Callable[[List[int]], List[Tuple[List[int], float]]]  # tokens → [(next_tokens, lp)]
ExtractAnswer = Callable[[List[int]], Optional[str]]  # tokens → answer|None


class PRMBeamSearch:
    """Beam search over multi-step reasoning guided by a process reward model.

    Parameters
    ----------
    config:
        Beam-search hyper-parameters.

    Usage::

        cfg = PRMBeamConfig(beam_width=4, max_steps=8)
        searcher = PRMBeamSearch(cfg)
        result = searcher.search(
            seed_tokens=[1, 2, 3],
            prm_scorer=my_prm,
            expand_fn=my_expand,
            extract_answer=my_extract,
        )
        print(result.best_answer)
    """

    def __init__(self, config: PRMBeamConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def search(
        self,
        seed_tokens: List[int],
        prm_scorer: PRMScorer,
        expand_fn: ExpandFn,
        extract_answer: ExtractAnswer,
    ) -> PRMBeamResult:
        """Run PRM-guided beam search from *seed_tokens*.

        Parameters
        ----------
        seed_tokens:
            Initial prompt token ids.
        prm_scorer:
            Callable ``(tokens, step) -> prm_score``.
        expand_fn:
            Callable ``(tokens) -> List[(next_tokens, log_prob)]``.
        extract_answer:
            Callable ``(tokens) -> Optional[str]``.  Returns a string when
            the candidate has reached a terminal state.
        """
        beams: List[PRMBeamCandidate] = [
            PRMBeamCandidate(tokens=list(seed_tokens), log_prob=0.0, depth=0)
        ]
        step = 0

        while step < self.config.max_steps:
            candidates: List[PRMBeamCandidate] = []
            for beam in beams:
                expansions = expand_fn(beam.tokens)
                for next_tokens, lp in expansions:
                    new_tokens = beam.tokens + next_tokens
                    prm_score = prm_scorer(new_tokens, beam.depth)
                    answer = extract_answer(new_tokens) or ""
                    candidates.append(
                        PRMBeamCandidate(
                            tokens=new_tokens,
                            prm_scores=beam.prm_scores + [prm_score],
                            log_prob=beam.log_prob + lp,
                            depth=beam.depth + 1,
                            answer=answer,
                        )
                    )
            step += 1
            if not candidates:
                break
            candidates = self._prune_to_beam(candidates)
            beams = candidates
            # Early exit if all beams have a terminal answer
            if all(b.answer for b in beams):
                break

        beams.sort(
            key=lambda b: b.combined_score(
                self.config.prm_weight, self.config.token_prob_weight
            ),
            reverse=True,
        )
        return PRMBeamResult(
            best_candidate=beams[0],
            all_candidates=beams,
            n_steps_taken=step,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_to_beam(self, candidates: List[PRMBeamCandidate]) -> List[PRMBeamCandidate]:
        """Keep the *beam_width* highest-scoring candidates."""
        candidates.sort(
            key=lambda b: b.combined_score(
                self.config.prm_weight, self.config.token_prob_weight
            ),
            reverse=True,
        )
        return candidates[: self.config.beam_width]

    def _score_candidates(
        self,
        candidates: List[PRMBeamCandidate],
    ) -> List[float]:
        """Return list of combined scores for *candidates*."""
        return [
            c.combined_score(self.config.prm_weight, self.config.token_prob_weight)
            for c in candidates
        ]
