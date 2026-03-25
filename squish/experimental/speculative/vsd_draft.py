"""
VSDDraftHead — Variational Speculative Decoding Training Objective.

Inspired by: "Variational Speculative Decoding" (Feb 2026).
Problem: Draft models are trained on greedy/teacher-forced paths, but at
inference time decoding is stochastic.  This training-decoding distributional
gap degrades acceptance length.

VSD fixes this by training the draft head to maximise *sequence-level
acceptance probability* rather than token-level cross-entropy.  The loss is:

    L = -E[acceptance_len] + β · KL(p_draft || p_target)

Where:
  • acceptance_len ∈ {0, ..., draft_len} is the speculative decoding result.
  • KL term keeps the draft close to the target for quality preservation.

Reference improvement: +9.6% acceptance length over EAGLE-3 on MT-Bench /
HumanEval / GSM8K.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class VSDConfig:
    """Configuration for Variational Speculative Decoding."""
    kl_weight: float = 0.1           # β — KL divergence weight in loss
    n_candidates: int = 5            # number of draft sequences to evaluate
    temperature: float = 1.0         # temperature for draft sampling
    eps: float = 1e-9                # numerical stability floor

    def __post_init__(self) -> None:
        if self.kl_weight < 0:
            raise ValueError(f"kl_weight must be >= 0, got {self.kl_weight}")
        if self.n_candidates < 1:
            raise ValueError(f"n_candidates must be >= 1, got {self.n_candidates}")
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")


# ---------------------------------------------------------------------------
# Training example
# ---------------------------------------------------------------------------

@dataclass
class VSDTrainingExample:
    """A single sequence-level training example for the VSD objective."""
    context_ids: List[int]            # token context
    draft_tokens: List[int]           # draft sequence attempted
    target_logits: np.ndarray         # (draft_len, vocab) — target model logits
    accepted_mask: List[bool]         # which draft tokens were accepted

    @property
    def accepted_len(self) -> int:
        # Accepted length = run of True values from the start
        n = 0
        for v in self.accepted_mask:
            if v:
                n += 1
            else:
                break
        return n


# ---------------------------------------------------------------------------
# Loss computation
# ---------------------------------------------------------------------------

class VSDLoss:
    """Compute the VSD training loss for a draft-model output.

    The loss is differentiable w.r.t. draft_logits and can be used to
    fine-tune any draft head.  This implementation returns scalar numpy
    values for use with a custom training loop.
    """

    def __init__(self, config: VSDConfig) -> None:
        self.config = config

    def compute(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
        accepted_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Compute VSD loss components.

        Args:
            draft_logits:  (draft_len, vocab) — draft model log-probabilities
                           (pre-softmax logits).
            target_logits: (draft_len, vocab) — target model logits.
            accepted_mask: (draft_len,) bool — 1 where draft token was accepted.

        Returns:
            Dict with keys: 'total', 'acceptance_loss', 'kl_loss'.
        """
        draft_len = draft_logits.shape[0]
        cfg = self.config

        p_draft = self._softmax(draft_logits)   # (draft_len, vocab)
        p_target = self._softmax(target_logits) # (draft_len, vocab)

        # Acceptance-probability term: E[-acceptance_len]
        # acceptance_len = sum of prefix accept flags
        # The differentiable surrogate: -sum_t [ p_accept(t) * cumulative_prior ]
        acceptance_expected = self.acceptance_probability(draft_logits, target_logits)
        acceptance_loss = -acceptance_expected

        # KL( p_draft || p_target ) at each position, averaged
        kl = self._kl_divergence(p_draft, p_target)  # (draft_len,)
        kl_loss = float(kl.mean())

        total_loss = acceptance_loss + cfg.kl_weight * kl_loss

        return {
            "total": total_loss,
            "acceptance_loss": acceptance_loss,
            "kl_loss": kl_loss,
        }

    def acceptance_probability(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
    ) -> float:
        """Estimate expected acceptance length E[accepted_len].

        Under the speculative decoding acceptance rule:
            p_accept(t) = min(1, p_target(t|context) / p_draft(t|context))

        For the greedy draft token at each position.

        Args:
            draft_logits:  (draft_len, vocab)
            target_logits: (draft_len, vocab)

        Returns:
            Scalar expected number of accepted tokens.
        """
        draft_len = draft_logits.shape[0]
        p_draft = self._softmax(draft_logits)
        p_target = self._softmax(target_logits)

        greedy = np.argmax(p_draft, axis=-1)   # (draft_len,)
        p_d = p_draft[np.arange(draft_len), greedy].clip(min=self.config.eps)
        p_t = p_target[np.arange(draft_len), greedy]
        per_tok_accept = np.minimum(1.0, p_t / p_d)  # (draft_len,)

        # E[accepted_len] = sum_{n=0}^{draft_len-1} P(first n tokens accepted)
        # = sum_n prod_{i=0}^{n-1} p_accept(i)
        cumulative = 1.0
        expected = 0.0
        for n in range(draft_len):
            expected += cumulative * per_tok_accept[n]
            cumulative *= per_tok_accept[n]
        return float(expected)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        x = logits - logits.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
        """KL(p || q) per row."""
        return (p * np.log((p + eps) / (q + eps))).sum(axis=-1)


# ---------------------------------------------------------------------------
# Draft trainer
# ---------------------------------------------------------------------------

@dataclass
class VSDTrainerStats:
    training_steps: int = 0
    total_acceptance_rate: float = 0.0
    _acceptance_samples: int = 0

    @property
    def mean_acceptance_rate(self) -> float:
        if self._acceptance_samples == 0:
            return 0.0
        return self.total_acceptance_rate / self._acceptance_samples

    def __repr__(self) -> str:
        return (
            f"VSDTrainerStats(steps={self.training_steps}, "
            f"mean_accept={self.mean_acceptance_rate:.3f})"
        )


class VSDDraftTrainer:
    """Evaluates VSD loss over batches of training examples.

    In a full training loop::
        trainer = VSDDraftTrainer(VSDConfig())
        loss_dict = trainer.compute_loss(example, draft_fn)
        # backprop through draft_fn parameters using loss_dict['total']
    """

    def __init__(self, config: VSDConfig) -> None:
        self.config = config
        self._loss_fn = VSDLoss(config)
        self.stats = VSDTrainerStats()

    def compute_loss(
        self,
        example: VSDTrainingExample,
        draft_fn: Callable[[List[int]], np.ndarray],
    ) -> Dict[str, float]:
        """Compute VSD loss for one example.

        Args:
            example:  VSDTrainingExample containing context and target logits.
            draft_fn: callable (context_ids: List[int]) -> (draft_len, vocab)
                      logits from the draft head.

        Returns:
            Dict with 'total', 'acceptance_loss', 'kl_loss'.
        """
        draft_logits = draft_fn(example.context_ids)
        losses = self._loss_fn.compute(
            draft_logits=draft_logits,
            target_logits=example.target_logits,
            accepted_mask=np.array(example.accepted_mask, dtype=bool),
        )
        self.stats.training_steps += 1
        return losses

    def acceptance_rate(
        self,
        examples: List[VSDTrainingExample],
        draft_fn: Callable[[List[int]], np.ndarray],
    ) -> float:
        """Estimate mean acceptance rate across a list of examples.

        Returns:
            Mean expected acceptance rate in [0, 1].
        """
        rates = []
        for ex in examples:
            draft_logits = draft_fn(ex.context_ids)
            rate = self._loss_fn.acceptance_probability(
                draft_logits=draft_logits,
                target_logits=ex.target_logits,
            )
            draft_len = len(ex.draft_tokens)
            rate_normalised = rate / max(draft_len, 1)
            rates.append(rate_normalised)

        mean_rate = float(np.mean(rates)) if rates else 0.0
        self.stats.total_acceptance_rate += mean_rate
        self.stats._acceptance_samples += 1
        return mean_rate

    def __repr__(self) -> str:
        return f"VSDDraftTrainer(kl_weight={self.config.kl_weight}, {self.stats})"
