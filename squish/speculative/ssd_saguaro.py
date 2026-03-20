"""
SSD / SAGUARO — Speculative² Decoding.

Inspired by: "SSD: Speculative Speculative Decoding" (ICLR 2026).
Algorithm (SAGUARO):
  1.  While the draft model generates tokens, simultaneously predict the k most
      likely *verification outcomes* (how many tokens the main model will accept).
  2.  For each predicted outcome o_i, pre-fetch the next draft sequence that would
      follow if o_i were the actual verification result.
  3.  At verification time: look up the actual outcome in the pre-fetched table.
      If it is there → return the prefetched tokens immediately (zero drafting latency).
      If not → fall back to standard drafting from the verified prefix.

Reference result: up to 5× over autoregressive, up to 2× over optimised spec decode.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SSDConfig:
    """Configuration for SAGUARO speculative² decoding."""
    k_outcomes: int = 4          # number of anticipated verification outcomes
    draft_len: int = 8           # tokens per draft sequence
    acceptance_threshold: float = 0.3  # min probability to consider an outcome
    temperature: float = 1.0     # sampling temperature for pre-fetches
    greedy_verify: bool = True   # verify greedily (argmax) vs probabilistically

    def __post_init__(self) -> None:
        if self.k_outcomes < 1:
            raise ValueError(f"k_outcomes must be >= 1, got {self.k_outcomes}")
        if self.draft_len < 1:
            raise ValueError(f"draft_len must be >= 1, got {self.draft_len}")


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SSDOutcome:
    """A single anticipated verification outcome."""
    accepted_len: int            # predicted number of tokens the verifier accepts
    probability: float           # model's confidence in this outcome


@dataclass
class SSDPrefetchEntry:
    """Pre-fetched draft sequence for one anticipated outcome."""
    outcome: SSDOutcome
    tokens: List[int]            # pre-fetched continuation tokens


@dataclass
class VerifyResult:
    """Result from a single SAGUARO verify-and-select step."""
    accepted_tokens: List[int]   # tokens committed this step
    correction_token: int        # one corrected token appended after accepts
    prefetch_hit: bool           # was the prefetch table hit?
    accepted_len: int


@dataclass
class SSDStats:
    """Runtime statistics for SAGUARO."""
    decode_steps: int = 0
    prefetch_hits: int = 0
    prefetch_misses: int = 0
    total_tokens_generated: int = 0
    total_draft_tokens: int = 0

    @property
    def prefetch_hit_rate(self) -> float:
        total = self.prefetch_hits + self.prefetch_misses
        return self.prefetch_hits / total if total > 0 else 0.0

    @property
    def mean_tokens_per_step(self) -> float:
        return (
            self.total_tokens_generated / self.decode_steps
            if self.decode_steps > 0
            else 0.0
        )

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_tokens_generated / self.total_draft_tokens

    def __repr__(self) -> str:
        return (
            f"SSDStats(steps={self.decode_steps}, "
            f"hit_rate={self.prefetch_hit_rate:.1%}, "
            f"mean_tok/step={self.mean_tokens_per_step:.2f})"
        )


# ---------------------------------------------------------------------------
# SAGUARO engine
# ---------------------------------------------------------------------------

class SSDSaguaro:
    """SAGUARO: pre-speculate verification outcomes to eliminate drafting overhead.

    Usage (one decode step)::

        prefetches = ssd.prefetch_outcomes(context_ids, draft_tokens, draft_fn)
        result     = ssd.verify_and_select(main_logits, draft_tokens, prefetches)
        context_ids.extend(result.accepted_tokens + [result.correction_token])
    """

    def __init__(self, config: SSDConfig) -> None:
        self.config = config
        self.stats = SSDStats()

    # ------------------------------------------------------------------
    # Outcome prediction
    # ------------------------------------------------------------------

    def predict_outcomes(
        self,
        draft_logits: np.ndarray,
        target_logits: np.ndarray,
    ) -> List[SSDOutcome]:
        """Predict the k most likely acceptance lengths for this draft.

        Uses the speculative decoding acceptance probability formula:
            p_accept(t) = min(1, p_target(t) / p_draft(t))

        We compute the probability that exactly `n` tokens are accepted
        by marginalising the token-level acceptance probabilities, then
        keep the top-k outcomes by probability mass.

        Args:
            draft_logits:  (draft_len, vocab_size) — draft model logits.
            target_logits: (draft_len, vocab_size) — target model logits for same tokens.

        Returns:
            List of SSDOutcome, sorted descending by probability.
        """
        draft_len = draft_logits.shape[0]
        cfg = self.config

        # softmax to probabilities
        def _softmax(x: np.ndarray) -> np.ndarray:
            x = x - x.max(axis=-1, keepdims=True)
            e = np.exp(x)
            return e / e.sum(axis=-1, keepdims=True)

        p_draft = _softmax(draft_logits)    # (draft_len, vocab)
        p_target = _softmax(target_logits)  # (draft_len, vocab)

        # argmax draft tokens
        draft_tokens = np.argmax(p_draft, axis=-1)  # (draft_len,)

        # token-level acceptance probability
        p_accept_tok = np.minimum(
            1.0,
            p_target[np.arange(draft_len), draft_tokens]
            / (p_draft[np.arange(draft_len), draft_tokens].clip(min=1e-9))
        )  # (draft_len,)

        # P(exactly n accepted) = prod(p_accept[:n]) * (1-p_accept[n]) for n<draft_len
        outcomes: List[SSDOutcome] = []
        prob_all_accept = 1.0
        for n in range(draft_len + 1):
            if n == draft_len:
                prob = prob_all_accept
            else:
                prob = prob_all_accept * (1.0 - p_accept_tok[n])
                prob_all_accept *= p_accept_tok[n]

            if prob >= cfg.acceptance_threshold / draft_len:
                outcomes.append(SSDOutcome(accepted_len=n, probability=float(prob)))

        outcomes.sort(key=lambda o: o.probability, reverse=True)
        return outcomes[: cfg.k_outcomes]

    # ------------------------------------------------------------------
    # Pre-fetching
    # ------------------------------------------------------------------

    def prefetch_outcomes(
        self,
        context_ids: List[int],
        draft_tokens: List[int],
        draft_fn: Callable[[List[int]], List[int]],
    ) -> Dict[int, SSDPrefetchEntry]:
        """For each anticipated outcome, pre-fetch the next draft sequence.

        Args:
            context_ids:  current context token ids.
            draft_tokens: the pending draft tokens being verified.
            draft_fn:     callable (context: List[int]) -> List[int] that
                          produces the next draft_len tokens.

        Returns:
            Dict mapping accepted_len → SSDPrefetchEntry.
        """
        cfg = self.config
        prefetches: Dict[int, SSDPrefetchEntry] = {}

        for n in range(min(cfg.k_outcomes, len(draft_tokens) + 1)):
            # Hypothetical context if exactly n tokens are accepted
            hypo_context = list(context_ids) + draft_tokens[:n]
            next_draft = draft_fn(hypo_context)
            prefetches[n] = SSDPrefetchEntry(
                outcome=SSDOutcome(accepted_len=n, probability=0.0),
                tokens=next_draft,
            )
            self.stats.total_draft_tokens += len(next_draft)

        return prefetches

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_and_select(
        self,
        main_logits: np.ndarray,
        draft_tokens: List[int],
        prefetches: Dict[int, "SSDPrefetchEntry"],
    ) -> VerifyResult:
        """Verify draft tokens against the main model and select accepted tokens.

        The main model's logits for each draft position determine acceptance.
        If the resulting accepted_len is in prefetches → prefetch hit.

        Args:
            main_logits: (draft_len, vocab_size) — main model logits for draft positions.
            draft_tokens: draft token ids (length == main_logits.shape[0]).
            prefetches: pre-fetched next-draft table from prefetch_outcomes().

        Returns:
            VerifyResult with accepted tokens and correction token.
        """
        draft_len = len(draft_tokens)
        accepted: List[int] = []
        correction_token: int = 0

        for i, tok in enumerate(draft_tokens):
            logits_i = main_logits[i]  # (vocab,)
            if self.config.greedy_verify:
                best = int(np.argmax(logits_i))
            else:
                probs = self._softmax(logits_i)
                best = int(np.random.choice(len(probs), p=probs))

            if best == tok:
                accepted.append(tok)
            else:
                correction_token = best
                break

        accepted_len = len(accepted)

        # get correction from final position if all accepted
        if accepted_len == draft_len:
            correction_token = int(np.argmax(main_logits[-1]))

        hit = accepted_len in prefetches
        if hit:
            self.stats.prefetch_hits += 1
        else:
            self.stats.prefetch_misses += 1

        self.stats.decode_steps += 1
        self.stats.total_tokens_generated += accepted_len + 1  # +1 for correction

        return VerifyResult(
            accepted_tokens=accepted,
            correction_token=correction_token,
            prefetch_hit=hit,
            accepted_len=accepted_len,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / e.sum()

    def __repr__(self) -> str:
        return (
            f"SSDSaguaro(k_outcomes={self.config.k_outcomes}, "
            f"draft_len={self.config.draft_len}, {self.stats})"
        )
