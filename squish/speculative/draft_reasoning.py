"""DraftReasoningVerifier: speculative-decoding acceptance for reasoning chains.

Extends standard speculative decoding to reasoning/chain-of-thought generation.
Because reasoning tokens form longer causal sequences a simple token-probability
threshold is insufficient; the verifier also evaluates whether the draft hidden
state is geometrically consistent with the context hidden states via cosine
similarity.

References:
  Leviathan et al., "Fast Inference from Transformers via Speculative Decoding",
  ICML 2023.  Adapted for reasoning by combining token probability with
  rolling cosine similarity over the recent context window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "DraftReasoningConfig",
    "DraftReasoningState",
    "DraftReasoningVerifier",
]


@dataclass
class DraftReasoningConfig:
    """Configuration for :class:`DraftReasoningVerifier`.

    Attributes:
        token_prob_threshold: Minimum draft token probability for acceptance
            (must be in ``(0, 1]``).
        cosine_threshold: Minimum mean cosine similarity of the draft hidden
            to the context window hiddens.
        context_window: Number of recent context hidden states to compare
            against.
        seed: RNG seed (used for calibration sampling).
    """

    token_prob_threshold: float = 0.6
    cosine_threshold: float = 0.85
    context_window: int = 32
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0.0 < self.token_prob_threshold <= 1.0):
            raise ValueError(
                f"token_prob_threshold must be in (0, 1], got {self.token_prob_threshold}"
            )
        if not (0.0 < self.cosine_threshold <= 1.0):
            raise ValueError(
                f"cosine_threshold must be in (0, 1], got {self.cosine_threshold}"
            )
        if self.context_window < 1:
            raise ValueError(
                f"context_window must be ≥ 1, got {self.context_window}"
            )


@dataclass
class DraftReasoningState:
    """Mutable verification statistics.

    Attributes:
        n_accepted: Total accepted draft tokens.
        n_rejected: Total rejected draft tokens.
        acceptance_history: Boolean history of each decision.
    """

    n_accepted: int = 0
    n_rejected: int = 0
    acceptance_history: List[bool] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.n_accepted + self.n_rejected


class DraftReasoningVerifier:
    """Accept or reject draft tokens using probability + cosine criteria.

    A draft token is accepted when **both** of the following hold:

    1. ``draft_token_prob >= config.token_prob_threshold``
    2. ``mean_cosine(draft_hidden, context_hiddens[-K:]) >= config.cosine_threshold``

    Usage::

        cfg = DraftReasoningConfig(token_prob_threshold=0.6, cosine_threshold=0.85)
        verifier = DraftReasoningVerifier(cfg)
        state = verifier.new_state()
        accepted = verifier.verify(prob, draft_hidden, context_hiddens, state)
    """

    def __init__(self, config: DraftReasoningConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Primary API
    # ------------------------------------------------------------------

    def new_state(self) -> DraftReasoningState:
        """Return a fresh verification state."""
        return DraftReasoningState()

    def verify(
        self,
        draft_token_prob: float,
        draft_hidden: np.ndarray,
        context_hiddens: List[np.ndarray],
        state: DraftReasoningState,
    ) -> bool:
        """Decide whether to accept the draft token.

        Parameters
        ----------
        draft_token_prob:
            Probability (0–1) assigned to the draft token by the draft model.
        draft_hidden:
            Hidden-state vector of shape ``(hidden_dim,)`` for this draft token.
        context_hiddens:
            Recent context hidden states (at most ``config.context_window`` used).
        state:
            Mutable statistics (updated in place).
        """
        draft_hidden = np.asarray(draft_hidden, dtype=np.float32)
        context = context_hiddens[-self.config.context_window :]

        prob_ok = draft_token_prob >= self.config.token_prob_threshold

        if context:
            sims = [
                self._cosine_sim(draft_hidden, np.asarray(h, dtype=np.float32))
                for h in context
            ]
            cosine_ok = float(np.mean(sims)) >= self.config.cosine_threshold
        else:
            # No context → rely on probability alone
            cosine_ok = True

        accepted = prob_ok and cosine_ok
        if accepted:
            state.n_accepted += 1
        else:
            state.n_rejected += 1
        state.acceptance_history.append(accepted)
        return accepted

    def acceptance_rate(self, state: DraftReasoningState) -> float:
        """Return the running acceptance rate (0–1)."""
        if state.total == 0:
            return 0.0
        return state.n_accepted / state.total

    def calibrate_threshold(
        self,
        valid_samples: List[Tuple[float, np.ndarray, List[np.ndarray], bool]],
        target_rate: float = 0.8,
    ) -> float:
        """Find the lowest combined threshold that achieves *target_rate* on samples.

        Parameters
        ----------
        valid_samples:
            List of ``(prob, draft_hidden, context_hiddens, label)`` tuples.
        target_rate:
            Desired minimum acceptance rate.

        Returns
        -------
        float:
            Recommended ``token_prob_threshold`` that achieves the target rate.
        """
        for thresh in np.linspace(0.1, 0.95, 18):
            thresh = float(thresh)
            accepted = sum(
                1
                for prob, dh, ctx, _ in valid_samples
                if prob >= thresh
            )
            rate = accepted / max(len(valid_samples), 1)
            if rate >= target_rate:
                return thresh
        return float(self.config.token_prob_threshold)

    def reset(self, state: DraftReasoningState) -> None:
        """Reset *state* statistics in place."""
        state.n_accepted = 0
        state.n_rejected = 0
        state.acceptance_history.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Numerically stable cosine similarity between two 1-D vectors."""
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
