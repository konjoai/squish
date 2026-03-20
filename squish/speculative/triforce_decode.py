"""TriForceDecoder — hierarchical speculative decoding for long-context LLMs.

Implements the TriForce algorithm (Sun et al., ICLR 2025 / arXiv:2404.11912).

Standard speculative decoding uses a small *draft model* to propose tokens
that the target verifies.  For very long contexts this breaks because the
draft model's KV cache does not fit in fast memory.

TriForce replaces the full draft-model KV cache with a *page subset* of the
target model's own KV cache:

  1. Select the top-K most-attended KV pages from the full context (the
     "retrieval draft KV").
  2. Run lightweight draft steps using only this compressed KV.
  3. Verify with the target model using the full KV cache (standard
     rejection sampling).

Because the draft KV is always a true subset of the target KV, the acceptance
criterion is identical to vanilla speculative decoding and the output
distribution is exactly the target distribution.

This module provides the TriForce scheduling and accept/reject logic as a
pure-Python/NumPy component compatible with every Squish backend.

Reference:
    Sun et al., "TriForce: Lossless Acceleration of Long Sequence LLM
    Decoding with Hierarchical Speculative Decoding",
    ICLR 2025 (arXiv:2404.11912).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "TriForceConfig",
    "TriForceDraftResult",
    "TriForceDecoder",
]

# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class TriForceConfig:
    """Configuration for TriForceDecoder.

    Attributes:
        draft_length: Number of tokens to speculatively draft per step.
        top_k_pages: How many KV pages to retain for the retrieval draft KV.
        page_size: Number of tokens per KV page.
        temperature: Sampling temperature for draft and target logits.
        seed: Random seed for reproducibility.
    """

    draft_length: int = 5
    top_k_pages: int = 8
    page_size: int = 16
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.draft_length < 1:
            raise ValueError(
                f"draft_length must be ≥ 1; got {self.draft_length}"
            )
        if self.top_k_pages < 1:
            raise ValueError(
                f"top_k_pages must be ≥ 1; got {self.top_k_pages}"
            )
        if self.page_size < 1:
            raise ValueError(
                f"page_size must be ≥ 1; got {self.page_size}"
            )
        if self.temperature <= 0:
            raise ValueError(
                f"temperature must be > 0; got {self.temperature}"
            )


# ── Result container ──────────────────────────────────────────────────────────


@dataclass
class TriForceDraftResult:
    """Result of one speculative draft-and-verify step.

    Attributes:
        accepted_tokens: Token IDs that were accepted by the verifier.
        n_accepted: Number of accepted tokens (0 … draft_length + 1).
        n_drafted: Number of tokens drafted.
        acceptance_rate: ``n_accepted / n_drafted``.
    """

    accepted_tokens: list[int]
    n_accepted: int
    n_drafted: int
    acceptance_rate: float


# ── Core class ────────────────────────────────────────────────────────────────


class TriForceDecoder:
    """Hierarchical speculative decoder using retrieval-based draft KV.

    In real deployment this class orchestrates calls to:
    * A *draft forward function* that accepts a page-subset KV cache.
    * A *target forward function* that uses the full KV cache.

    For standalone testing (all unit-test scenarios), both of these are
    supplied by the caller as callables with the signature::

        logits = fn(token_ids: list[int], kv_pages: list[np.ndarray])
            -> np.ndarray   # shape (vocab_size,)

    Example::

        cfg = TriForceConfig(draft_length=4, top_k_pages=4)
        decoder = TriForceDecoder(cfg)
        result  = decoder.step(
            context_ids=[1, 2, 3, ...],
            kv_cache=full_kv,      # list of page arrays
            draft_fn=my_draft_fn,
            target_fn=my_target_fn,
        )
        print(result.accepted_tokens)

    Args:
        config: :class:`TriForceConfig` (optional).
    """

    def __init__(self, config: Optional[TriForceConfig] = None) -> None:
        self.config: TriForceConfig = config or TriForceConfig()
        self._rng = np.random.default_rng(seed=self.config.seed)

    # ── Page selection ────────────────────────────────────────────────────────

    def select_top_k_pages(
        self,
        attention_weights: np.ndarray,
        n_pages: int,
    ) -> np.ndarray:
        """Select the top-K most-attended KV pages.

        Args:
            attention_weights: ``(n_pages,)`` float array of aggregate
                attention mass per page (e.g., sum over recent queries).
            n_pages: Total number of pages (used for bounds check).

        Returns:
            Sorted integer array of selected page indices (ascending).

        Raises:
            ValueError: If ``attention_weights`` has wrong length.
        """
        if len(attention_weights) != n_pages:
            raise ValueError(
                f"attention_weights length {len(attention_weights)} ≠ n_pages {n_pages}"
            )
        k = min(self.config.top_k_pages, n_pages)
        top_k = np.argsort(attention_weights)[-k:]
        return np.sort(top_k).astype(np.int32)

    # ── Accept / reject ───────────────────────────────────────────────────────

    def accept_reject(
        self,
        draft_logits: Sequence[np.ndarray],
        target_logits: Sequence[np.ndarray],
        draft_tokens: Sequence[int],
    ) -> list[int]:
        """Standard speculative decoding accept/reject with bonus token.

        For each position i:
            - Accept token ``draft_tokens[i]`` with probability
              ``min(1, target_p[i] / draft_p[i])``.
            - On rejection, sample a correction token from the *residual*
              distribution and stop.
        If all draft tokens are accepted, sample one additional bonus token
        from the last target distribution.

        Args:
            draft_logits: List of ``(vocab_size,)`` float arrays, one per
                draft position.
            target_logits: List of ``(vocab_size,)`` float arrays, one per
                draft position.
            draft_tokens: List of int token IDs proposed by the draft.

        Returns:
            List of accepted (and possible bonus) token IDs.

        Raises:
            ValueError: If the three sequences differ in length.
        """
        n = len(draft_tokens)
        if len(draft_logits) != n or len(target_logits) != n:
            raise ValueError(
                "draft_logits, target_logits, and draft_tokens must have equal length"
            )

        T = self.config.temperature
        accepted: list[int] = []

        for i in range(n):
            p_draft = self._to_prob(np.asarray(draft_logits[i]), T)
            p_target = self._to_prob(np.asarray(target_logits[i]), T)
            t = draft_tokens[i]
            ratio = float(p_target[t]) / (float(p_draft[t]) + 1e-12)
            u = float(self._rng.random())
            if u < min(1.0, ratio):
                accepted.append(t)
            else:
                # Correction token from residual
                residual = np.maximum(p_target - p_draft, 0.0)
                s = residual.sum()
                if s > 1e-8:
                    residual /= s
                    accepted.append(int(self._rng.choice(len(residual), p=residual)))
                else:
                    accepted.append(int(np.argmax(p_target)))
                return accepted

        # Bonus token from the last target distribution
        p_last = self._to_prob(np.asarray(target_logits[-1]), T)
        accepted.append(int(self._rng.choice(len(p_last), p=p_last)))
        return accepted

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(
        self,
        context_ids: list[int],
        kv_page_weights: np.ndarray,
        draft_fn,
        target_fn,
    ) -> TriForceDraftResult:
        """Execute one TriForce draft-and-verify step.

        Args:
            context_ids: Full context token IDs up to the current position.
            kv_page_weights: ``(n_pages,)`` attention weights per KV page
                (used by :meth:`select_top_k_pages`).
            draft_fn: Callable ``(token: int, page_idx: np.ndarray) ->
                np.ndarray`` returning logits for the *next* token.
            target_fn: Callable ``(tokens: list[int]) -> list[np.ndarray]``
                returning one logits array per token position.

        Returns:
            :class:`TriForceDraftResult` with accepted tokens and statistics.
        """
        cfg = self.config
        n_pages = len(kv_page_weights)
        pages = self.select_top_k_pages(kv_page_weights, n_pages)

        # Draft phase
        draft_tokens: list[int] = []
        draft_logits: list[np.ndarray] = []
        cur_token = context_ids[-1] if context_ids else 0
        for _ in range(cfg.draft_length):
            logits = np.asarray(draft_fn(cur_token, pages))
            p = self._to_prob(logits, cfg.temperature)
            t = int(self._rng.choice(len(p), p=p))
            draft_tokens.append(t)
            draft_logits.append(logits)
            cur_token = t

        # Target verification (batch)
        target_logits_list = target_fn(draft_tokens)

        accepted = self.accept_reject(
            draft_logits, target_logits_list, draft_tokens
        )

        n_accepted = len(accepted)
        return TriForceDraftResult(
            accepted_tokens=accepted,
            n_accepted=n_accepted,
            n_drafted=cfg.draft_length,
            acceptance_rate=n_accepted / cfg.draft_length,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_prob(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Convert logits to a probability distribution."""
        logits = logits.astype(np.float64)
        logits = logits / temperature
        logits -= logits.max()
        exp = np.exp(logits)
        return exp / exp.sum()

    def reset_rng(self, seed: Optional[int] = None) -> None:
        """Reset the internal random number generator.

        Args:
            seed: New seed; if None, uses ``config.seed``.
        """
        self._rng = np.random.default_rng(
            seed if seed is not None else self.config.seed
        )

    def __repr__(self) -> str:
        return (
            f"TriForceDecoder(draft_length={self.config.draft_length}, "
            f"top_k_pages={self.config.top_k_pages}, "
            f"page_size={self.config.page_size})"
        )
