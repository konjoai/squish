"""BestOfNSampler: drawing the highest-reward completion from N samples.

Best-of-N scaling (Cobbe et al. NeurIPS 2021; Lightman et al. 2023; Snell et al.
arXiv 2408.03314) is among the simplest and most effective test-time compute
strategies.  *N* independent completions are generated; a reward function scores
each; the top-scoring completion is returned, with optional "majority-vote"
aggregation mode for categorical answers.

Reference: Snell et al., "Scaling LLM Test-Time Compute Optimally is More
Effective than Scaling Model Parameters", arXiv 2408.03314, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "BestOfNConfig",
    "BestOfNResult",
    "BestOfNSampler",
]

RewardFn = Callable[[str], float]


@dataclass
class BestOfNConfig:
    """Configuration for :class:`BestOfNSampler`.

    Attributes:
        n: Number of candidates to sample.
        temperature: Sampling temperature (used in simulation stubs).
        reward_aggregation: How to aggregate per-completion rewards.
            ``"max"`` selects the completion with the highest reward.
            ``"mean"`` returns the completion whose answer appears most
            frequently (majority-vote style), breaking ties by mean reward.
        seed: RNG seed.
    """

    n: int = 8
    temperature: float = 0.8
    reward_aggregation: str = "max"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"n must be ≥ 1, got {self.n}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        valid = {"max", "mean"}
        if self.reward_aggregation not in valid:
            raise ValueError(
                f"reward_aggregation must be one of {valid}, got {self.reward_aggregation!r}"
            )


@dataclass
class BestOfNResult:
    """Output of one :meth:`BestOfNSampler.sample` call.

    Attributes:
        best_completion: The winning completion string.
        all_scores: Reward scores for each of the *n* completions.
        best_index: Index of the winner in the original list.
    """

    best_completion: str
    all_scores: List[float]
    best_index: int

    @property
    def best_score(self) -> float:
        return self.all_scores[self.best_index]

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.all_scores)) if self.all_scores else 0.0


class BestOfNSampler:
    """Score *n* completions and return the best one.

    Usage::

        cfg = BestOfNConfig(n=8, reward_aggregation="max")
        sampler = BestOfNSampler(cfg)
        result = sampler.sample(completions, reward_fn=my_reward)
        print(result.best_completion)
    """

    def __init__(self, config: BestOfNConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def sample(self, completions: List[str], reward_fn: RewardFn) -> BestOfNResult:
        """Score *completions* and return the best.

        Parameters
        ----------
        completions:
            Pre-generated completion strings (length must match config.n or be ≤ it).
        reward_fn:
            Callable ``(completion: str) -> float``.
        """
        if not completions:
            raise ValueError("completions must be non-empty")
        scores = [reward_fn(c) for c in completions]
        if self.config.reward_aggregation == "max":
            best_idx = int(np.argmax(scores))
        else:
            best_idx = self._majority_best(completions, scores)
        return BestOfNResult(
            best_completion=completions[best_idx],
            all_scores=scores,
            best_index=best_idx,
        )

    def simulate(
        self,
        n: Optional[int] = None,
        answer_distribution: Optional[Dict[str, float]] = None,
    ) -> BestOfNResult:
        """Simulate Best-of-N with synthetic completions (useful for tests).

        Parameters
        ----------
        n:
            Override ``config.n`` for this call.
        answer_distribution:
            Mapping ``{answer: probability}``.  Defaults to a 3-class
            synthetic distribution if not provided.
        """
        n = n if n is not None else self.config.n
        if answer_distribution is None:
            answer_distribution = {"A": 0.5, "B": 0.3, "C": 0.2}

        keys = list(answer_distribution.keys())
        probs = np.array([answer_distribution[k] for k in keys], dtype=np.float64)
        probs /= probs.sum()
        chosen = self._rng.choice(len(keys), size=n, p=probs)
        completions = [keys[i] for i in chosen]
        scores = [probs[i] + self._rng.uniform(-0.05, 0.05) for i in chosen]
        scores = [float(np.clip(s, 0.0, 1.0)) for s in scores]

        if self.config.reward_aggregation == "max":
            best_idx = int(np.argmax(scores))
        else:
            best_idx = self._majority_best(completions, scores)

        return BestOfNResult(
            best_completion=completions[best_idx],
            all_scores=scores,
            best_index=best_idx,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _majority_best(self, completions: List[str], scores: List[float]) -> int:
        """Return the index of the completion whose answer appears most often.

        Ties in vote count are broken by mean reward.
        """
        vote_counts: Dict[str, int] = {}
        vote_scores: Dict[str, List[float]] = {}
        for comp, sc in zip(completions, scores):
            vote_counts[comp] = vote_counts.get(comp, 0) + 1
            vote_scores.setdefault(comp, []).append(sc)

        best_comp = max(
            vote_counts.keys(),
            key=lambda c: (vote_counts[c], float(np.mean(vote_scores[c]))),
        )
        # Return index of first occurrence
        return completions.index(best_comp)
