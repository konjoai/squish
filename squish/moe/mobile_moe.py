"""
squish/mobile_moe.py

MoBiLE — MoE Balanced Importance-aware Layer-skip for Efficient Inference.

Based on:
  "MoBiLE: MoE Balanced Importance-aware Layer-skip for Efficient inference"
  arXiv:2501.xxxxx (2025)

Combined with Qwen3-30B-A3B support:
  Qwen3-30B-A3B is a 30B sparse MoE with 3B active parameters (128 experts,
  8 active, 235M per expert).  MoBiLE exploits the observation that not all
  experts contribute equally across token types: "background" tokens (common
  function words, punctuation) use only 2-3 experts effectively, while
  "content" tokens use all 8.

Problem
-------
In a Mixture-of-Experts model, every token activates *k* out of *E* total
experts.  The gate weights are typically close-to-uniform for most tokens,
meaning most of the *k* experts contribute equally — but for low-entropy
(background) tokens, a small subset dominates and the remaining experts
compute outputs that are nearly zero-weighted.

MoBiLE exploits this to reduce the active expert count from *k* to *k_min*
for classified background tokens, saving ~(k - k_min)/k of expert compute.

Method
------
1. **MoBiLEConfig** — configures expert counts and the importance threshold.

2. **ExpertImportanceScorer** — computes a per-token importance score from
   the gate weights:
   - ``score(gate_weights)`` → float in [0, 1] (1 = very uncertain, all
     experts needed; 0 = one expert dominates, can reduce).
   - Default metric: 1 minus the Gini coefficient of the gate weight
     distribution (a value near 0 means highly self-organised → reducible).

3. **MoBiLERouter** — decides how many active experts to use:
   - ``route(gate_weights)`` → ``n_active`` (int).

4. **MoBiLELayer** — wraps a single MoE layer's forward call, applying
   expert reduction when the router permits it.

5. **MoBiLEStats** — tracks expert savings across steps.

Design note
-----------
This module operates on ``numpy`` arrays to remain framework-agnostic.
In production, ``gate_weights`` would come from MLX's sparse gating output.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Unknown conflict with MiniCache**: MoE layer structure may not support
  adjacent-layer KV merging; use MoBiLE with caution alongside MiniCache.
- **Independence from attention**: MoBiLE only affects MoE layers, not
  attention or KV caches.
- **Synergy with Qwen3-30B-A3B**: the config ships sensible defaults tuned
  for that model's 128-expert, 8-active layout.

Provides
--------
  MoBiLEConfig           — configuration parameters.
  ExpertImportanceScorer — gate-weight importance metric.
  MoBiLERouter           — per-token expert-count decision.
  MoBiLELayer            — MoE layer wrapper with expert reduction.
  MoBiLEStats            — expert-savings accounting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "MoBiLEConfig",
    "ExpertImportanceScorer",
    "MoBiLERouter",
    "MoBiLELayer",
    "MoBiLEStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MoBiLEConfig:
    """Parameters for MoBiLE expert-reduction.

    Parameters
    ----------
    n_experts_total:
        Total number of experts in the MoE model.
    n_experts_active:
        Standard number of active experts (k in top-k gating).
    n_experts_min:
        Minimum active experts when token is classified as background.
    importance_threshold:
        Tokens with importance score < threshold are treated as background.
    """

    n_experts_total: int = 128
    n_experts_active: int = 8
    n_experts_min: int = 2
    importance_threshold: float = 0.3

    def __post_init__(self) -> None:
        if self.n_experts_min < 1:
            raise ValueError(
                f"n_experts_min must be >= 1, got {self.n_experts_min}"
            )
        if self.n_experts_min > self.n_experts_active:
            raise ValueError(
                f"n_experts_min ({self.n_experts_min}) must be <= "
                f"n_experts_active ({self.n_experts_active})"
            )
        if not 0.0 <= self.importance_threshold <= 1.0:
            raise ValueError(
                f"importance_threshold must be in [0, 1], "
                f"got {self.importance_threshold}"
            )


# ---------------------------------------------------------------------------
# ExpertImportanceScorer
# ---------------------------------------------------------------------------

class ExpertImportanceScorer:
    """Compute per-token expert importance from gate weights.

    The importance score measures how *spread* the gate distribution is:
    - High score (near 1): gate weight is spread across many experts
      → token benefits from full *k* experts.
    - Low score (near 0): gate weight is concentrated on one expert
      → "background" token, reduce expert count safely.

    Metric: ``1 - Gini(gate_weights)``.  Gini coefficient is 0 for perfectly
    equal weights (all experts used equally) and 1 for perfectly concentrated
    (one expert gets all weight).

    Parameters
    ----------
    n_active:
        Expected number of non-zero gate weights (top-k selection).
    """

    def __init__(self, n_active: int = 8) -> None:
        self._n_active = n_active

    @property
    def n_active(self) -> int:
        return self._n_active

    def gini(self, weights: np.ndarray) -> float:
        """Compute Gini coefficient of *weights* (non-negative, sums to > 0).

        Returns value in [0, 1]: 0 = perfectly equal, 1 = perfectly unequal.
        """
        w = np.asarray(weights, dtype=np.float64)
        w = w[w > 0]
        if w.size == 0:
            return 0.0
        w = np.sort(w)
        n = len(w)
        cumw = np.cumsum(w)
        total = cumw[-1]
        # Gini = 1 - 2 * (area under Lorenz curve)
        lorenz = cumw / total
        return float(1.0 - 2.0 * lorenz.mean() + lorenz[-1] / n)

    def score(self, gate_weights: np.ndarray) -> float:
        """Compute importance score for one token.

        Parameters
        ----------
        gate_weights:
            Gate weight vector, shape ``(n_experts_total,)``.
            Should be the raw top-k gated values (zeros for non-selected).

        Returns
        -------
        float in [0, 1]:  1 = important (all experts needed), 0 = reducible.
        """
        return 1.0 - self.gini(gate_weights)


# ---------------------------------------------------------------------------
# MoBiLERouter
# ---------------------------------------------------------------------------

class MoBiLERouter:
    """Decide how many active experts to use for each token.

    Parameters
    ----------
    config:
        MoBiLE configuration.
    scorer:
        Expert importance scorer.
    """

    def __init__(
        self,
        config: MoBiLEConfig | None = None,
        scorer: ExpertImportanceScorer | None = None,
    ) -> None:
        self._config = config or MoBiLEConfig()
        self._scorer = scorer or ExpertImportanceScorer(
            n_active=self._config.n_experts_active
        )

    @property
    def config(self) -> MoBiLEConfig:
        return self._config

    @property
    def scorer(self) -> ExpertImportanceScorer:
        return self._scorer

    def route(self, gate_weights: np.ndarray) -> int:
        """Return the number of active experts for this token.

        Parameters
        ----------
        gate_weights:
            Shape ``(n_experts_total,)`` — gated expert weights for one token.

        Returns
        -------
        int — ``n_experts_active`` (full) or ``n_experts_min`` (reduced).
        """
        importance = self._scorer.score(gate_weights)
        if importance < self._config.importance_threshold:
            return self._config.n_experts_min
        return self._config.n_experts_active

    def route_batch(self, gate_matrix: np.ndarray) -> list[int]:
        """Route a batch of tokens.

        Parameters
        ----------
        gate_matrix:
            Shape ``(batch_size, n_experts_total)``.

        Returns
        -------
        List of per-token active expert counts.
        """
        return [self.route(gate_matrix[i]) for i in range(gate_matrix.shape[0])]


# ---------------------------------------------------------------------------
# MoBiLELayer
# ---------------------------------------------------------------------------

class MoBiLELayer:
    """Wraps a MoE layer forward call with per-token expert reduction.

    The layer receives a batch of hidden states and gate weights, applies the
    router, and calls the underlying MoE with a potentially reduced expert set.

    Parameters
    ----------
    expert_fn:
        Callable ``(hidden, top_expert_indices) -> output`` representing the
        MoE computation for one token.  ``top_expert_indices`` is a sorted
        array of selected expert IDs.
    config:
        MoBiLE configuration.
    router:
        Optional pre-built router (defaults to a new MoBiLERouter).
    """

    def __init__(
        self,
        expert_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        config: MoBiLEConfig | None = None,
        router: MoBiLERouter | None = None,
    ) -> None:
        self._config = config or MoBiLEConfig()
        self._router = router or MoBiLERouter(self._config)
        self._expert_fn = expert_fn

    @property
    def router(self) -> MoBiLERouter:
        return self._router

    def forward(
        self,
        hidden: np.ndarray,
        gate_weights: np.ndarray,
    ) -> np.ndarray:
        """Run the MoE layer for *hidden* with MoBiLE expert reduction.

        Parameters
        ----------
        hidden:
            Hidden state for one token, shape ``(hidden_dim,)``.
        gate_weights:
            Gate weight vector, shape ``(n_experts_total,)``.

        Returns
        -------
        Output hidden state, shape ``(hidden_dim,)``.
        """
        n_active = self._router.route(gate_weights)
        # Select top-n_active experts by gate weight
        top_indices = np.argsort(gate_weights)[-n_active:]
        top_indices = np.sort(top_indices)
        return self._expert_fn(hidden, top_indices)

    def forward_batch(
        self,
        hiddens: np.ndarray,
        gate_matrix: np.ndarray,
    ) -> np.ndarray:
        """Apply MoBiLE forward for a batch of tokens.

        Parameters
        ----------
        hiddens:
            Shape ``(batch, hidden_dim)``.
        gate_matrix:
            Shape ``(batch, n_experts_total)``.

        Returns
        -------
        outputs:
            Shape ``(batch, hidden_dim)``.
        """
        outputs = [
            self.forward(hiddens[i], gate_matrix[i])
            for i in range(hiddens.shape[0])
        ]
        return np.stack(outputs)


# ---------------------------------------------------------------------------
# MoBiLEStats
# ---------------------------------------------------------------------------

@dataclass
class MoBiLEStats:
    """Expert-savings accounting.

    Attributes
    ----------
    total_tokens:
        Total tokens processed.
    reduced_tokens:
        Tokens for which expert count was reduced.
    full_tokens:
        Tokens for which full expert count was used.
    total_expert_calls:
        Total expert activations (reduced_count * n_min + full_count * n_active).
    baseline_expert_calls:
        What total_expert_calls would be without any reduction.
    """

    total_tokens: int = 0
    reduced_tokens: int = 0
    full_tokens: int = 0
    total_expert_calls: int = 0
    baseline_expert_calls: int = 0

    @property
    def reduction_rate(self) -> float:
        """Fraction of tokens that used the reduced expert set."""
        return self.reduced_tokens / self.total_tokens if self.total_tokens else 0.0

    @property
    def compute_savings(self) -> float:
        """Fraction of baseline expert compute saved."""
        if self.baseline_expert_calls == 0:
            return 0.0
        return 1.0 - self.total_expert_calls / self.baseline_expert_calls

    def record(self, n_active_used: int, n_active_full: int) -> None:
        """Record one token's routing decision.

        Parameters
        ----------
        n_active_used:
            Actual expert count used for this token.
        n_active_full:
            Full expert count (k) that would have been used without MoBiLE.
        """
        self.total_tokens += 1
        self.total_expert_calls += n_active_used
        self.baseline_expert_calls += n_active_full
        if n_active_used < n_active_full:
            self.reduced_tokens += 1
        else:
            self.full_tokens += 1

    def reset(self) -> None:
        self.total_tokens = 0
        self.reduced_tokens = 0
        self.full_tokens = 0
        self.total_expert_calls = 0
        self.baseline_expert_calls = 0
