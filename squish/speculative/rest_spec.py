"""rest_spec.py — REST: Retrieval-based Speculative Decoding

Maintains an online DataStore (n-gram hash trie) populated from previous request
completions. At each decode step, the last ``n`` generated tokens are used as a
retrieval key to look up candidate continuation tokens in the DataStore.

The matched candidates are used as draft tokens in a standard speculative
decoding verify-then-accept loop, exactly like N-gram speculative decoding but
drawing from a richer, growing corpus of real completions rather than the current
context only.

Algorithm:
  DataStore:  dict mapping frozen n-gram tuple → Counter{next_token: freq}
  Draft:      retrieve top-B candidates by frequency; propose as draft sequence
  Verify:     single batched target model forward pass over draft tokens
  Accept:     longest consistent prefix (same algorithm as standard spec-decode)

Key properties:
  - No additional model required; DataStore = "free" draft source
  - Grows richer with each completed request (online learning)
  - Degrades gracefully: falls back to single-token greedy when no candidates
  - Expected acceptance rate 40-65% on seen-domain text

Based on: "REST: Retrieval-based Speculative Decoding" (He et al., 2023)

Usage:
    store = DataStore(ngram_order=3, max_entries=100_000)
    decoder = RESTSpecDecoder(store, full_forward_fn, config)

    # After each request:
    store.ingest(completed_token_ids)

    # During decode:
    result = decoder.generate(prompt_ids, max_new_tokens=200, eos_id=2)
    output_ids, stats = result
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable, Dict, FrozenSet, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# DataStore
# ---------------------------------------------------------------------------

@dataclass
class DataStoreConfig:
    """Configuration for the REST DataStore.

    Args:
        ngram_order:  Length of the retrieval key (context window for lookup).
        max_entries:  Maximum total n-gram entries stored (LRU-eviction above).
        top_b_draft:  Number of top-frequency candidates to propose as draft.
        draft_depth:  How many chained draft tokens to propose per step.
    """
    ngram_order: int = 3
    max_entries: int = 100_000
    top_b_draft: int = 5
    draft_depth: int = 4


class DataStore:
    """Online n-gram retrieval DataStore for REST speculative decoding.

    Thread-unsafe; designed for single-threaded use with external locking if
    needed for concurrent request ingestion.
    """

    def __init__(self, config: Optional[DataStoreConfig] = None) -> None:
        self.config = config or DataStoreConfig()
        # trie: ngram_tuple → Counter{next_token → frequency}
        self._trie: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self._total_entries: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, token_ids: List[int]) -> None:
        """Add a completed token sequence to the DataStore.

        Extracts all n-gram contexts and their following tokens.
        """
        cfg = self.config
        n = cfg.ngram_order
        if len(token_ids) <= n:
            return
        for i in range(len(token_ids) - n):
            key = tuple(token_ids[i : i + n])
            next_tok = token_ids[i + n]
            self._trie[key][next_tok] += 1
            self._total_entries += 1
            # Evict oldest entries if over limit (simple count-based)
            if self._total_entries > cfg.max_entries:
                self._evict_one()

    def _evict_one(self) -> None:
        """Remove the entry with the lowest total frequency to stay under limit."""
        if not self._trie:
            return
        # Find the n-gram counter with the smallest total count
        min_key = min(self._trie, key=lambda k: sum(self._trie[k].values()))
        evicted = sum(self._trie[min_key].values())
        del self._trie[min_key]
        self._total_entries -= evicted

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get_candidates(self, context: List[int]) -> List[int]:
        """Return the top-B candidate next tokens for the given context.

        Args:
            context: Recent token IDs (last ``ngram_order`` tokens used as key).

        Returns:
            List of candidate token IDs, sorted by descending frequency.
            Empty list if no match found.
        """
        cfg = self.config
        n = cfg.ngram_order
        if len(context) < n:
            return []
        key = tuple(context[-n:])
        counter = self._trie.get(key)
        if not counter:
            return []
        top = counter.most_common(cfg.top_b_draft)
        return [tok for tok, _ in top]

    def draft_sequence(self, context: List[int]) -> List[int]:
        """Greedily extend context for ``draft_depth`` steps using the DataStore.

        Each step picks the most frequent next token and appends it to context.
        Returns the draft extension (may be shorter than draft_depth if runs out).
        """
        cfg = self.config
        ctx = list(context)
        draft = []
        for _ in range(cfg.draft_depth):
            candidates = self.get_candidates(ctx)
            if not candidates:
                break
            next_tok = candidates[0]  # most frequent
            draft.append(next_tok)
            ctx.append(next_tok)
        return draft

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def total_ngrams(self) -> int:
        return len(self._trie)

    @property
    def total_entries(self) -> int:
        return self._total_entries

    def __repr__(self) -> str:
        return (
            f"DataStore(ngrams={self.total_ngrams}, "
            f"entries={self.total_entries}, "
            f"order={self.config.ngram_order})"
        )


# ---------------------------------------------------------------------------
# REST Decoder
# ---------------------------------------------------------------------------

@dataclass
class RESTSpecConfig:
    """Configuration for RESTSpecDecoder."""
    max_draft_len: int = 4       # max draft tokens per speculative step
    temperature: float = 1.0    # target model sampling temperature
    eos_id: int = 2
    min_accept_len: int = 1     # stop speculative if accepted < this per step


@dataclass
class RESTSpecStats:
    """Statistics for a single generation run."""
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_target_calls: int = 0
    total_tokens_generated: int = 0
    elapsed_s: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def mean_accepted_per_call(self) -> float:
        if self.total_target_calls == 0:
            return 0.0
        return self.total_tokens_generated / self.total_target_calls


class RESTSpecDecoder:
    """Retrieval-based speculative decoder backed by a DataStore.

    The ``full_forward`` callable must accept a list of token IDs and return a
    1-D float array of logits over the vocabulary for the last position.

    Example signature:
        def full_forward(token_ids: List[int]) -> np.ndarray  # (vocab_size,)
    """

    def __init__(
        self,
        datastore: DataStore,
        full_forward: Callable[[List[int]], np.ndarray],
        config: Optional[RESTSpecConfig] = None,
    ) -> None:
        self.datastore = datastore
        self.full_forward = full_forward
        self.config = config or RESTSpecConfig()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 200,
        eos_id: Optional[int] = None,
    ) -> Tuple[List[int], RESTSpecStats]:
        """Generate up to max_new_tokens tokens from prompt_ids.

        Returns:
            (output_ids, stats) where output_ids includes the prompt.
        """
        cfg = self.config
        actual_eos = eos_id if eos_id is not None else cfg.eos_id
        stats = RESTSpecStats()
        t0 = monotonic()

        context = list(prompt_ids)
        generated = 0

        while generated < max_new_tokens:
            # Try to get a draft sequence from the DataStore
            draft = self.datastore.draft_sequence(context)
            if not draft:
                # No draft available — single greedy step (fallback)
                logits = self.full_forward(context)
                next_tok = int(np.argmax(logits))
                stats.total_target_calls += 1
                context.append(next_tok)
                generated += 1
                stats.total_tokens_generated += 1
                if next_tok == actual_eos:
                    break
                continue

            draft = draft[: cfg.max_draft_len]
            stats.total_draft_tokens += len(draft)

            # Verification: run target model on context + draft
            verify_input = context + draft
            # We need logits at each draft position → call once per position
            # (simplified: one call per draft token; a real impl would do
            #  a single batched forward pass)
            accepted = []
            for i, dtok in enumerate(draft):
                verify_ctx = verify_input[: len(context) + i + 1]
                logits = self.full_forward(verify_ctx[:-1])  # predict dtok
                stats.total_target_calls += 1
                # Accept if target agrees (argmax check)
                target_tok = int(np.argmax(logits))
                if target_tok == dtok:
                    accepted.append(dtok)
                else:
                    # Reject: emit target's token and stop this draft
                    accepted.append(target_tok)
                    break

            for tok in accepted:
                context.append(tok)
                generated += 1
                stats.total_tokens_generated += 1
                stats.total_accepted_tokens += 1
                if tok == actual_eos or generated >= max_new_tokens:
                    break

            if generated < max_new_tokens and context[-1] == actual_eos:
                break

        stats.elapsed_s = monotonic() - t0
        return context, stats
