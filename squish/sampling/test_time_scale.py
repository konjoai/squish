"""TestTimeComputeRouter: difficulty-aware dispatch across compute strategies.

Snell et al. (Berkeley, arXiv 2408.03314, 2024).  Routes each request to one of four
compute strategies — greedy, top-p, best-of-N, or PRM beam search — based on a
lightweight difficulty estimate (entropy of the first-token distribution).  Achieves
2–3× answer accuracy at fixed wall-clock compute budget vs always-greedy dispatch.

Reference: Snell et al., "Scaling LLM Test-Time Compute Optimally", arXiv 2408.03314, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "ComputeStrategy",
    "TestTimeScaleConfig",
    "TestTimeScaleResult",
    "TestTimeComputeRouter",
]


class ComputeStrategy(Enum):
    """Available inference compute strategies."""

    GREEDY = "greedy"
    TOP_P = "top_p"
    BEST_OF_N = "best_of_n"
    PRM_BEAM = "prm_beam"


@dataclass
class TestTimeScaleConfig:
    """Configuration for :class:`TestTimeComputeRouter`.

    Attributes:
        easy_threshold: First-token entropy below which GREEDY is used.
        hard_threshold: First-token entropy above which PRM_BEAM is used.
        best_of_n_n: Number of samples for BEST_OF_N strategy.
        prm_beam_width: Beam width for PRM_BEAM strategy.
        top_p: Nucleus probability for TOP_P strategy.
        seed: RNG seed.
    """

    easy_threshold: float = 1.0
    hard_threshold: float = 3.0
    best_of_n_n: int = 8
    prm_beam_width: int = 4
    top_p: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if self.easy_threshold < 0.0:
            raise ValueError(
                f"easy_threshold must be ≥ 0, got {self.easy_threshold}"
            )
        if self.hard_threshold <= self.easy_threshold:
            raise ValueError(
                f"hard_threshold ({self.hard_threshold}) must exceed "
                f"easy_threshold ({self.easy_threshold})"
            )
        if self.best_of_n_n < 1:
            raise ValueError(f"best_of_n_n must be ≥ 1, got {self.best_of_n_n}")
        if self.prm_beam_width < 1:
            raise ValueError(
                f"prm_beam_width must be ≥ 1, got {self.prm_beam_width}"
            )
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


@dataclass
class TestTimeScaleResult:
    """Routing decision for one request.

    Attributes:
        strategy: Selected compute strategy.
        entropy: Estimated first-token distribution entropy.
        config_params: Strategy-specific parameters passed to the sampler.
    """

    strategy: ComputeStrategy
    entropy: float
    config_params: Dict


class TestTimeComputeRouter:
    """Route requests to the optimal compute strategy given a difficulty budget.

    Difficulty is estimated from the entropy of the first-token logit
    distribution:
    - Low entropy  → easy question → GREEDY decode
    - Medium entropy → moderate question → TOP_P or BEST_OF_N
    - High entropy → hard question → PRM_BEAM search

    Usage::

        cfg = TestTimeScaleConfig(easy_threshold=1.0, hard_threshold=3.0)
        router = TestTimeComputeRouter(cfg)
        logits = model.first_token_logits(prompt)
        result = router.route(logits)
        # → result.strategy, result.config_params

    """

    def __init__(self, config: TestTimeScaleConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)
        self._n_routed: Dict[ComputeStrategy, int] = {
            s: 0 for s in ComputeStrategy
        }

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def route(self, first_token_logits: np.ndarray) -> TestTimeScaleResult:
        """Decide a compute strategy from *first_token_logits*.

        Parameters
        ----------
        first_token_logits:
            1-D float array of raw logits over the vocabulary.
        """
        logits = np.asarray(first_token_logits, dtype=np.float64)
        entropy = self._entropy(logits)
        strategy = self._pick_strategy(entropy)
        self._n_routed[strategy] += 1
        params = self._strategy_params(strategy)
        return TestTimeScaleResult(
            strategy=strategy,
            entropy=float(entropy),
            config_params=params,
        )

    def route_from_probs(self, probs: np.ndarray) -> TestTimeScaleResult:
        """Decide from a *probability* distribution (already softmax'd)."""
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, None)
        probs = probs / probs.sum()
        entropy = float(-np.sum(probs * np.log(probs)))
        strategy = self._pick_strategy(entropy)
        self._n_routed[strategy] += 1
        return TestTimeScaleResult(
            strategy=strategy,
            entropy=entropy,
            config_params=self._strategy_params(strategy),
        )

    def routing_stats(self) -> Dict[str, int]:
        """Return per-strategy routing counts."""
        return {s.value: n for s, n in self._n_routed.items()}

    def reset_stats(self) -> None:
        """Reset routing counters."""
        for s in ComputeStrategy:
            self._n_routed[s] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _entropy(self, logits: np.ndarray) -> float:
        """Shannon entropy of the softmax distribution."""
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        probs = np.clip(probs, 1e-12, None)
        return float(-np.sum(probs * np.log(probs)))

    def _pick_strategy(self, entropy: float) -> ComputeStrategy:
        cfg = self.config
        if entropy <= cfg.easy_threshold:
            return ComputeStrategy.GREEDY
        if entropy <= (cfg.easy_threshold + cfg.hard_threshold) / 2.0:
            return ComputeStrategy.TOP_P
        if entropy <= cfg.hard_threshold:
            return ComputeStrategy.BEST_OF_N
        return ComputeStrategy.PRM_BEAM

    def _strategy_params(self, strategy: ComputeStrategy) -> Dict:
        cfg = self.config
        if strategy == ComputeStrategy.GREEDY:
            return {"temperature": 0.0}
        if strategy == ComputeStrategy.TOP_P:
            return {"top_p": cfg.top_p, "temperature": 1.0}
        if strategy == ComputeStrategy.BEST_OF_N:
            return {"n": cfg.best_of_n_n, "temperature": 0.8}
        return {"beam_width": cfg.prm_beam_width, "temperature": 0.7}
