"""
squish/prompt_lookup.py

Prompt Lookup Decoding — n-gram-based speculative token prediction.

Based on:
  "Prompt Lookup Decoding" — Saxena 2023
  (https://github.com/apoorvumang/prompt-lookup-decoding)

Key insight
-----------
When the model is tasked with copying or paraphrasing text that already
appears in the prompt (common in summarisation, RAG, code editing …), the
continuation tokens can be *predicted* by finding n-gram matches inside
the existing context rather than running a draft model.

The decoder:
  1. Looks for the last ``ngram_min_len``…``ngram_max_len`` tokens of the
     current context as sub-sequences *within* the context (excluding the
     very end).
  2. Uses the continuation after each match as speculative draft tokens.
  3. Verifies the draft with the full model in a batched step.
  4. Accepts the longest verified prefix.

When no n-gram match is found, the decoder falls back to a single full-model
forward pass (standard greedy decoding).

This module provides:
  * ``PromptLookupConfig`` — n-gram range and speculation window.
  * ``NGramIndex`` — builds and queries an n-gram lookup table from token ids.
  * ``PromptLookupDecoder`` — drives the full speculative decode loop.

Usage::

    from squish.prompt_lookup import PromptLookupConfig, PromptLookupDecoder

    cfg = PromptLookupConfig(ngram_min=2, ngram_max=5, max_speculative=5)
    dec = PromptLookupDecoder(
        full_forward=lambda ids: model_forward(ids),   # np (vocab,)
        config=cfg,
    )
    output_ids = dec.generate(prompt_ids, max_new_tokens=200)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = [
    "PromptLookupConfig",
    "NGramIndex",
    "PromptLookupDecoder",
    "PromptLookupStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PromptLookupConfig:
    """Configuration for Prompt Lookup Decoding.

    Parameters
    ----------
    ngram_min : int
        Minimum n-gram length used for matching (≥ 2 recommended).
    ngram_max : int
        Maximum n-gram length used for matching (≥ ngram_min).
    max_speculative : int
        Maximum number of speculative tokens proposed from a single match.
    """

    ngram_min:       int = 2
    ngram_max:       int = 5
    max_speculative: int = 5

    def __post_init__(self) -> None:
        if self.ngram_min < 1:
            raise ValueError("ngram_min must be ≥ 1")
        if self.ngram_max < self.ngram_min:
            raise ValueError("ngram_max must be ≥ ngram_min")
        if self.max_speculative < 1:
            raise ValueError("max_speculative must be ≥ 1")


# ---------------------------------------------------------------------------
# N-gram index
# ---------------------------------------------------------------------------

class NGramIndex:
    """Map n-gram token tuples to lists of following token sequences.

    The index is built *incrementally*: each push adds one new position.

    Parameters
    ----------
    ngram_min : int — shortest n-gram to index (default 2).
    ngram_max : int — longest n-gram to index (default 5).
    max_continuations : int — max continuation length stored per match.
    """

    def __init__(
        self,
        ngram_min: int = 2,
        ngram_max: int = 5,
        max_continuations: int = 5,
    ) -> None:
        if ngram_min < 1:
            raise ValueError("ngram_min must be ≥ 1")
        if ngram_max < ngram_min:
            raise ValueError("ngram_max must be ≥ ngram_min")
        self._min = ngram_min
        self._max = ngram_max
        self._max_cont = max_continuations
        # n-gram → list of (start_pos_after_match, token_sequence)
        self._table: dict[tuple[int, ...], list[list[int]]] = {}
        self._tokens: list[int] = []

    def build(self, token_ids: list[int]) -> None:
        """(Re)build the full index from ``token_ids``."""
        self._table = {}
        self._tokens = list(token_ids)
        for i in range(len(token_ids)):
            self._index_position(i)

    def push(self, token_id: int) -> None:
        """Append one token and index new n-grams ending at this position."""
        self._tokens.append(token_id)
        pos = len(self._tokens) - 1
        self._index_position(pos)

    def _index_position(self, pos: int) -> None:
        """Index all n-grams ending exactly at ``pos`` (exclusive)."""
        tokens = self._tokens
        n_tok  = len(tokens)
        for n in range(self._min, self._max + 1):
            start = pos - n
            if start < 0:
                break
            ng    = tuple(tokens[start : pos])
            after_pos = pos
            if after_pos >= n_tok:  # pragma: no cover
                continue
            cont  = tokens[after_pos : after_pos + self._max_cont]
            if not cont:  # pragma: no cover
                continue
            if ng not in self._table:
                self._table[ng] = []
            self._table[ng].append(list(cont))

    def find(self, query: list[int]) -> list[list[int]]:
        """Return continuations matching any n-gram suffix of ``query``.

        Parameters
        ----------
        query : list[int]
            Tail of the current token sequence (last k tokens).

        Returns
        -------
        List of candidate continuation sequences, longest n-gram match first.
        Returns empty list when no match found.
        """
        results: list[list[int]] = []
        for n in range(min(self._max, len(query)), self._min - 1, -1):
            ng = tuple(query[-n:])
            if ng in self._table:
                results.extend(self._table[ng])
                break  # Use longest match only
        return results


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------

@dataclass
class PromptLookupStats:
    """Statistics from a :class:`PromptLookupDecoder` generation run.

    Attributes
    ----------
    speculative_steps : int — decoding steps that used n-gram speculation.
    fallback_steps    : int — steps that fell back to single full forward.
    total_draft_tokens: int — speculative tokens proposed.
    total_accepted    : int — accepted speculative tokens.
    """

    speculative_steps: int = 0
    fallback_steps:    int = 0
    total_draft_tokens: int = 0
    total_accepted:    int = 0

    @property
    def acceptance_rate(self) -> float:
        return (
            self.total_accepted / self.total_draft_tokens
            if self.total_draft_tokens > 0 else 0.0
        )


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class PromptLookupDecoder:
    """Speculative decode using n-gram matches from the existing context.

    Parameters
    ----------
    full_forward : callable
        ``full_forward(token_ids: List[int]) -> np.ndarray`` of shape
        ``(vocab_size,)`` — the token-prediction logits for the next position
        given the full sequence.
    config : PromptLookupConfig
    """

    def __init__(
        self,
        full_forward: Callable[[list[int]], np.ndarray],
        config: PromptLookupConfig,
    ) -> None:
        self._fwd  = full_forward
        self._cfg  = config
        self._idx  = NGramIndex(
            ngram_min=config.ngram_min,
            ngram_max=config.ngram_max,
            max_continuations=config.max_speculative,
        )

    @staticmethod
    def _top_token(logits: np.ndarray) -> int:
        return int(np.argmax(np.asarray(logits)))

    def generate(
        self,
        input_ids: list[int],
        max_new_tokens: int = 128,
    ) -> tuple[list[int], PromptLookupStats]:
        """Generate tokens appended to ``input_ids``.

        Parameters
        ----------
        input_ids : list[int] — prompt token ids.
        max_new_tokens : int — max additional tokens to produce.

        Returns
        -------
        ``(all_ids, stats)`` where ``all_ids`` = prompt + generated.
        """
        ids    = list(input_ids)
        stats  = PromptLookupStats()
        cfg    = self._cfg

        # Build initial index from prompt
        self._idx.build(ids)
        generated = 0

        while generated < max_new_tokens:
            # Find candidates using context suffix
            candidates = self._idx.find(ids)
            candidates = [c[:cfg.max_speculative] for c in candidates if c]
            # Deduplicate and pick best (longest first)
            seen: set = set()
            unique_cands: list[list[int]] = []
            for c in candidates:
                key = tuple(c)
                if key not in seen:
                    seen.add(key)
                    unique_cands.append(c)
            unique_cands.sort(key=len, reverse=True)

            if not unique_cands:
                # Fallback: single greedy step
                logits = self._fwd(ids)
                tok    = self._top_token(logits)
                ids.append(tok)
                self._idx.push(tok)
                generated += 1
                stats.fallback_steps += 1
                continue

            # Pick the first (longest) candidate as draft
            draft = unique_cands[0][:min(cfg.max_speculative, max_new_tokens - generated)]
            stats.speculative_steps  += 1
            stats.total_draft_tokens += len(draft)

            # Verify draft tokens greedily
            ctx       = list(ids)
            accepted_toks: list[int] = []
            for d_tok in draft:
                logits   = self._fwd(ctx)
                v_tok    = self._top_token(logits)
                if v_tok == d_tok:
                    accepted_toks.append(d_tok)
                    ctx.append(d_tok)
                    stats.total_accepted += 1
                else:
                    # Accept verifier's correction as well
                    accepted_toks.append(v_tok)
                    ctx.append(v_tok)
                    break

            for tok in accepted_toks:
                ids.append(tok)
                self._idx.push(tok)
            generated += len(accepted_toks)

        return ids, stats
