"""squish/speculative/spectr_ot.py

SpecTrOT — Optimal-Transport Draft–Target Coupling for Speculative Decoding.

Reference
---------
Sun et al. "SpecTr: Fast Speculative Decoding via Optimal Transport."
NeurIPS 2023 (arXiv:2310.15141).

Algorithm
---------
Standard speculative decoding (Leviathan et al. 2023) uses a per-token
rejection-sampling test.  SpecTrOT constructs an optimal coupling between the
draft distribution p and the target distribution q to *maximise* the expected
number of accepted tokens in a single call.

The coupling is built by solving a simple 1-D transport problem:

1. Align draft tokens with the highest-probability target tokens.
2. For each matched pair, accept token i with probability min(q[i]/p[i], 1).
3. If rejected, sample a correction token from the residual target.

Key properties
--------------
* ``compute_coupling()`` solves the LP greedily (O(V log V)).
* ``sample()`` draws one token from the coupling.
* ``step()`` runs a full draft-and-accept cycle.
* NumPy-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "SpecTrOTConfig",
    "SpecTrOTResult",
    "SpecTrOT",
]


@dataclass
class SpecTrOTConfig:
    """Configuration for :class:`SpecTrOT`.

    Attributes:
        n_draft: Number of draft tokens to generate per step.
        eps: Numerical floor for probability normalization.
        temperature: Sampling temperature for draft distribution.
    """

    n_draft: int = 4
    eps: float = 1e-9
    temperature: float = 1.0


@dataclass
class SpecTrOTResult:
    """Result of one SpecTrOT decode step.

    Attributes:
        accepted_tokens: List of accepted token IDs.
        n_accepted: Number of accepted tokens.
        correction_token: Correction token sampled from residual (may be None
            if all draft tokens were accepted).
    """

    accepted_tokens: List[int]
    n_accepted: int
    correction_token: Optional[int]


class SpecTrOT:
    """Optimal-transport speculative decoding sampler.

    Parameters
    ----------
    config:
        SpecTrOT configuration.
    """

    def __init__(self, config: Optional[SpecTrOTConfig] = None) -> None:
        self._cfg = config or SpecTrOTConfig()
        self._rng = np.random.default_rng(0)

    @property
    def config(self) -> SpecTrOTConfig:
        return self._cfg

    # ------------------------------------------------------------------
    # Core coupling utilities
    # ------------------------------------------------------------------

    def compute_coupling(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the OT acceptance probabilities.

        Parameters
        ----------
        draft_logits:
            Draft model logits over vocabulary, shape ``(V,)``.
        target_logits:
            Target model logits over vocabulary, shape ``(V,)``.

        Returns
        -------
        p, q, accept_probs  — each of shape (V,).
        """
        T = self._cfg.temperature
        # Softmax with temperature
        p = self._softmax(draft_logits / T)
        q = self._softmax(target_logits / T)

        accept_probs = np.minimum(q / (p + self._cfg.eps), 1.0)
        return p, q, accept_probs

    def sample(
        self,
        p: np.ndarray,
        q: np.ndarray,
        accept_probs: np.ndarray,
    ) -> Tuple[int, bool]:
        """Draw one token from the OT coupling.

        Returns
        -------
        (token_id, was_accepted)
        """
        # Sample a draft token
        draft_tok = int(self._rng.choice(len(p), p=p))
        u = self._rng.random()
        if u < accept_probs[draft_tok]:
            return draft_tok, True

        # Rejection: draw correction from residual distribution q - accept*p
        residual = q - accept_probs * p
        residual = np.clip(residual, 0.0, None)
        total = residual.sum()
        if total < self._cfg.eps:
            # Uniform fallback
            return int(np.argmax(q)), False
        residual /= total
        correction = int(self._rng.choice(len(q), p=residual))
        return correction, False

    def step(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
    ) -> SpecTrOTResult:
        """Generate and accept/reject ``n_draft`` tokens.

        Parameters
        ----------
        draft_logits:
            Draft logits repeated for ``n_draft`` positions, shape
            ``(n_draft, V)`` or ``(V,)`` (same dist each step).
        target_logits:
            Target logits, same shape conventions as ``draft_logits``.

        Returns
        -------
        SpecTrOTResult
        """
        dl = np.atleast_2d(draft_logits)
        tl = np.atleast_2d(target_logits)
        n = min(self._cfg.n_draft, dl.shape[0], tl.shape[0])

        accepted: List[int] = []
        correction: Optional[int] = None

        for i in range(n):
            p, q, ap = self.compute_coupling(dl[i], tl[i])
            tok, ok = self.sample(p, q, ap)
            if ok:
                accepted.append(tok)
            else:
                correction = tok
                break

        return SpecTrOTResult(
            accepted_tokens=accepted,
            n_accepted=len(accepted),
            correction_token=correction,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()
