"""
squish/speculative/adaptive_draft_budget.py

UCB1-based online bandit controller for speculation depth selection.

Speculative decoding throughput is a non-monotonic function of the
speculation depth k:

    effective_tps(k) = k * acceptance_rate(k) / (1 + verification_cost(k))

A small k → few accepted tokens per verification round; large k → most drafts
rejected, wasting verification compute.  The optimal k depends on the current
model, prompt domain, and hardware.

This module implements a multi-armed bandit (UCB1, Auer et al. 2002) over the
discrete action space {1, 2, …, max_k} to find and track the throughput-
maximising speculation depth online, without any calibration data.

UCB1 selects the arm i maximising:

    Q̂(i) + C × sqrt(ln(N) / n(i))

where Q̂(i) = estimated reward (effective tok/s), N = total rounds, n(i) = arm
play count, C = exploration constant (default 2.0).

The reward for each arm is normalised: r = accepted_tokens / elapsed_seconds,
providing a direct surrogate for tok/s throughput.

Key properties
--------------
* O(max_k) memory; O(1) per-step select/update.
* Exploration bonus drives the algo to measure under-sampled depths.
* Configurable warm-up: each depth tried once before UCB1 kicks in.
* Records rolling statistics per depth for diagnostics.

References
----------
Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the
multiarmed bandit problem. Machine Learning 47(2), 235–256.

Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast inference from transformers
via speculative decoding. ICML 2023. arXiv:2211.17192.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import math


@dataclass
class DraftBudgetConfig:
    """Configuration for the UCB1 speculation-depth bandit.

    Parameters
    ----------
    min_k:
        Minimum speculation depth (must be ≥ 1).
    max_k:
        Maximum speculation depth.
    exploration_constant:
        UCB1 exploration bonus weight C.  Higher values → more exploration.
        The classical choice is sqrt(2) ≈ 1.41; 2.0 works well in practice.
    warmup_rounds:
        Each arm is played this many times before UCB1 activates.  Ensures
        reliable initial reward estimates.
    reward_ema_alpha:
        EMA smoothing factor for per-arm reward estimates (0 < alpha ≤ 1).
        Higher values weight recent rounds more heavily.
    """

    min_k: int = 1
    max_k: int = 8
    exploration_constant: float = 2.0
    warmup_rounds: int = 3
    reward_ema_alpha: float = 0.3

    def __post_init__(self) -> None:
        if self.min_k < 1:
            raise ValueError("min_k must be >= 1")
        if self.max_k < self.min_k:
            raise ValueError("max_k must be >= min_k")
        if not 0.0 < self.exploration_constant <= 100.0:
            raise ValueError("exploration_constant must be in (0, 100]")
        if self.warmup_rounds < 0:
            raise ValueError("warmup_rounds must be >= 0")
        if not 0.0 < self.reward_ema_alpha <= 1.0:
            raise ValueError("reward_ema_alpha must be in (0, 1]")

    @property
    def n_arms(self) -> int:
        """Number of discrete speculation depths in the action space."""
        return self.max_k - self.min_k + 1


class AdaptiveDraftBudget:
    """UCB1 multi-armed bandit for online speculation-depth selection.

    Usage
    -----
    ::

        budget = AdaptiveDraftBudget()
        while True:
            k = budget.select()              # choose spec depth
            t0 = time.perf_counter()
            accepted = run_speculative_step(k)
            elapsed = time.perf_counter() - t0
            budget.update(k, accepted, elapsed)
    """

    def __init__(self, config: Optional[DraftBudgetConfig] = None) -> None:
        self.config = config or DraftBudgetConfig()
        n = self.config.n_arms
        self._play_counts: List[int] = [0] * n
        self._rewards: List[float] = [0.0] * n  # EMA tok/s per arm
        self._total_rounds: int = 0
        self._warmup_cursor: int = 0  # cycles through arms during warmup

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(self) -> int:
        """Return the speculation depth to use for the next step.

        During warmup (each arm not yet played ``warmup_rounds`` times),
        cycles through arms in order.  After warmup, uses UCB1.

        Returns
        -------
        int
            Speculation depth k ∈ [min_k, max_k].
        """
        cfg = self.config
        if not self._warmup_done():
            arm = self._next_warmup_arm()
            return arm + cfg.min_k
        return self._ucb1_select() + cfg.min_k

    def update(
        self,
        k: int,
        accepted_tokens: int,
        elapsed_seconds: float,
    ) -> None:
        """Record the reward for a completed speculative step.

        Parameters
        ----------
        k:
            Speculation depth that was used.
        accepted_tokens:
            Number of draft tokens accepted (0 ≤ accepted_tokens ≤ k).
        elapsed_seconds:
            Wall-clock time for the full draft+verify step.
        """
        if elapsed_seconds <= 0:
            return

        arm = max(0, min(self.config.n_arms - 1, k - self.config.min_k))
        reward = accepted_tokens / elapsed_seconds  # tok/s

        alpha = self.config.reward_ema_alpha
        if self._play_counts[arm] == 0:
            self._rewards[arm] = reward
        else:
            self._rewards[arm] = (
                alpha * reward + (1.0 - alpha) * self._rewards[arm]
            )

        self._play_counts[arm] += 1
        self._total_rounds += 1

    def best_k(self) -> int:
        """Return the arm with the highest estimated reward (exploit-only).

        Useful for reporting / logging the current best depth estimate.
        """
        if self._total_rounds == 0:
            return self.config.min_k
        best_arm = max(range(self.config.n_arms), key=lambda i: self._rewards[i])
        return best_arm + self.config.min_k

    def arm_stats(self) -> List[dict]:
        """Return per-arm statistics for diagnostics.

        Returns
        -------
        List[dict] with keys: k, plays, reward_tps.
        """
        cfg = self.config
        return [
            {
                "k": i + cfg.min_k,
                "plays": self._play_counts[i],
                "reward_tps": self._rewards[i],
            }
            for i in range(cfg.n_arms)
        ]

    def reset(self) -> None:
        """Reset all bandit statistics to initial state."""
        n = self.config.n_arms
        self._play_counts = [0] * n
        self._rewards = [0.0] * n
        self._total_rounds = 0
        self._warmup_cursor = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_rounds(self) -> int:
        """Total number of completed speculation steps recorded."""
        return self._total_rounds

    @property
    def is_warmed_up(self) -> bool:
        """True once all arms have been played at least ``warmup_rounds`` times."""
        return self._warmup_done()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _warmup_done(self) -> bool:
        wr = self.config.warmup_rounds
        if wr == 0:
            return True
        return all(c >= wr for c in self._play_counts)

    def _next_warmup_arm(self) -> int:
        """Return the next arm needing a warmup play."""
        wr = self.config.warmup_rounds
        # Find first arm with fewer than warmup_rounds plays
        for i in range(self.config.n_arms):
            if self._play_counts[i] < wr:
                return i
        return 0  # fallback, should not happen

    def _ucb1_select(self) -> int:
        """UCB1 arm selection.

        Any arm with zero plays gets infinite priority (standard UCB1
        property: every arm must be played at least once before the
        exploitation term can win).
        """
        # First-play guarantee: return the first un-played arm immediately.
        for i in range(self.config.n_arms):
            if self._play_counts[i] == 0:
                return i

        n_total = max(1, self._total_rounds)
        log_n = math.log(n_total)
        c = self.config.exploration_constant
        best_arm = 0
        best_score = -math.inf

        for i in range(self.config.n_arms):
            ucb = self._rewards[i] + c * math.sqrt(log_n / self._play_counts[i])
            if ucb > best_score:
                best_score = ucb
                best_arm = i

        return best_arm
