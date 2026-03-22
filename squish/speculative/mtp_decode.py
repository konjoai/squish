"""squish/speculative/mtp_decode.py

MTPDecode — Multi-Token Prediction Heads (DeepSeek-V3 Style).

Reference
---------
"DeepSeek-V3 Technical Report." DeepSeek-AI. arXiv:2412.19437, 2024.

Algorithm
---------
Multi-Token Prediction (MTP) attaches N auxiliary prediction heads to a
transformer model at training time.  At inference, these heads allow the
model to propose N next tokens in a single forward pass:

1. The main head predicts token t+1 as usual.
2. Auxiliary head_1 predicts token t+2 given the main head's hidden state.
3. Auxiliary head_k predicts token t+k+1 given head_{k-1}'s hidden state.
4. Token proposals are accepted greedily (or with lightweight tree-verify).

This module simulates the MTP decode path using NumPy-based logit sampling,
making it a drop-in algorithm layer that sits on top of any model backend.

Key properties
--------------
* NumPy-only; no GPU dependency.
* ``n_heads`` — number of auxiliary prediction heads (default 4).
* ``temperature`` — sampling temperature (1.0 = unmodified logits).
* ``greedy`` — if True, argmax is used instead of sampling.
* No tree-verify overhead: tokens are proposed greedily and used directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "MTPConfig",
    "MTPDraftResult",
    "MTPDecode",
]


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class MTPConfig:
    """Configuration for :class:`MTPDecode`.

    Attributes:
        n_heads: Number of auxiliary MTP heads.
        vocab_size: Vocabulary size (needed to size logit buffers).
        hidden_dim: Hidden state dimension.
        temperature: Sampling temperature for token selection.
        greedy: If True, use argmax instead of sampling.
    """

    n_heads: int = 4
    vocab_size: int = 32000
    hidden_size: int = 4096
    temperature: float = 1.0
    greedy: bool = True

    def __post_init__(self) -> None:
        if self.n_heads < 1:
            raise ValueError("n_heads must be >= 1")


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class MTPDraftResult:
    """Output of a single MTPDecode step.

    Attributes:
        tokens: Draft token IDs, shape ``(n_heads,)``.
        logprobs: Log-probabilities of each predicted token.
        accepted: Bitmask of which tokens were accepted (all True by default
            for greedy MTP; verification can update this).
    """

    draft_tokens: np.ndarray
    logprobs: np.ndarray
    accepted: np.ndarray

    @property
    def log_probs(self) -> np.ndarray:
        """Alias for logprobs (per-head log-probabilities)."""
        return self.logprobs

    @property
    def n_accepted(self) -> int:
        return int(self.accepted.sum())


# ── Module ────────────────────────────────────────────────────────────────────


class MTPDecode:
    """Multi-Token Prediction decoder.

    This class simulates an MTP decode pass.  In a real deployment the
    auxiliary head projections would be implemented inside the model forward
    method; here they are approximated with randomly-initialised linear
    projections for unit-testing purposes.

    Parameters
    ----------
    config:
        MTP configuration.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, config: Optional[MTPConfig] = None, seed: int = 0) -> None:
        self._cfg = config or MTPConfig()
        self._rng = np.random.default_rng(seed)
        # Simulate auxiliary head weight matrices W_i: (vocab_size, hidden_size)
        scale = 1.0 / np.sqrt(self._cfg.hidden_size)
        self._head_weights: List[np.ndarray] = [
            self._rng.standard_normal((self._cfg.vocab_size, self._cfg.hidden_size)).astype(
                np.float32
            )
            * scale
            for _ in range(self._cfg.n_heads)
        ]
        self._step_count: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def config(self) -> MTPConfig:
        return self._cfg

    @property
    def n_heads(self) -> int:
        return self._cfg.n_heads

    @property
    def step_count(self) -> int:
        return self._step_count

    def step(self, hidden_state: np.ndarray) -> MTPDraftResult:
        """Run a single MTP decode step.

        Parameters
        ----------
        hidden_state:
            Final hidden state of the previous token, shape ``(hidden_dim,)``
            or ``(1, hidden_dim)``.

        Returns
        -------
        MTPDraftResult
            Draft tokens and their log-probabilities.
        """
        h = np.asarray(hidden_state, dtype=np.float32).reshape(-1)
        if h.shape[0] != self._cfg.hidden_size:
            raise ValueError(
                f"hidden_state dim {h.shape[0]} != config.hidden_size {self._cfg.hidden_size}"
            )

        tokens = np.empty(self._cfg.n_heads, dtype=np.int64)
        logprobs = np.empty(self._cfg.n_heads, dtype=np.float32)

        current_h = h
        for i, W in enumerate(self._head_weights):
            logits = (W @ current_h).astype(np.float64)
            if self._cfg.temperature != 1.0 and self._cfg.temperature > 0:
                logits /= self._cfg.temperature
            log_probs = logits - np.log(np.sum(np.exp(logits - logits.max()))) - logits.max()
            if self._cfg.greedy:
                tok = int(np.argmax(log_probs))
            else:
                probs = np.exp(log_probs)
                probs /= probs.sum()
                tok = int(self._rng.choice(self._cfg.vocab_size, p=probs))
            tokens[i] = tok
            logprobs[i] = float(log_probs[tok])
            # Next head receives current hidden + a "residual" from token embedding
            # (approximated here as a random perturbation for simulation purposes)
            current_h = current_h + self._rng.standard_normal(self._cfg.hidden_size).astype(
                np.float32
            ) * 0.01

        self._step_count += 1
        return MTPDraftResult(
            draft_tokens=tokens,
            logprobs=logprobs,
            accepted=np.ones(self._cfg.n_heads, dtype=bool),
        )

    def verify_and_accept(
        self,
        result: MTPDraftResult,
        target_logprobs: np.ndarray,
    ) -> MTPDraftResult:
        """Apply lightweight rejection sampling against target log-probs.

        Parameters
        ----------
        result:
            Draft result from :meth:`step`.
        target_logprobs:
            Log-probabilities from the target model for each draft token,
            shape ``(n_heads,)``.

        Returns
        -------
        Updated :class:`MTPDraftResult` with ``accepted`` mask set.
        """
        target_logprobs = np.asarray(target_logprobs, dtype=np.float32)
        # If target_logprobs is 2D (n_heads, vocab), extract log-prob of chosen token
        if target_logprobs.ndim == 2:
            target_logprobs = target_logprobs[np.arange(len(result.draft_tokens)), result.draft_tokens]
        log_ratio = target_logprobs - result.logprobs
        # Accept if target_prob >= draft_prob (log_ratio >= 0) or with prob ratio
        u = self._rng.random(self._cfg.n_heads).astype(np.float32)
        accepted = (log_ratio >= 0) | (np.log(u) < log_ratio)
        # Causal: once rejected, all subsequent are also rejected
        first_reject = int(np.argmin(accepted)) if not accepted.all() else self._cfg.n_heads
        accepted[first_reject:] = False
        result.accepted = accepted
        return result

    def reset(self) -> None:
        """Reset step counter (weights are preserved)."""
        self._step_count = 0
