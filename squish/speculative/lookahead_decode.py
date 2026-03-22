"""squish/speculative/lookahead_decode.py

Lookahead Decoding — 2D Jacobi Window Speculative Decode.

Reference
---------
Fu et al. "Break the Sequential Dependency of LLM Inference Using Lookahead
Decoding." ICML 2024 (arXiv 2402.02057).

Algorithm
---------
Lookahead decoding maintains a 2D sliding window:

* **W (window size)**: number of simultaneous Jacobi iterations (future steps).
* **N (n-gram size)**: length of n-gram candidates in each window column.

At each decode step:
1. The W Jacobi branches are speculated on in a single batched forward pass.
2. For each branch position we also maintain a cache of N-grams observed so
   far (the *lookahead cache*).
3. Any N-gram candidate whose prefix matches the current context is verified
   speculatively; if it would have been accepted, those tokens are committed
   as a block.

This module provides a NumPy simulation of the 2D window logic, which makes
the algorithm easy to unit-test and benchmark without a live LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

__all__ = [
    "LookaheadConfig",
    "LookaheadResult",
    "LookaheadDecode",
]


@dataclass
class LookaheadConfig:
    """Configuration for :class:`LookaheadDecode`.

    Attributes:
        window_size: W — number of parallel Jacobi branches.
        ngram_size:  N — n-gram length used for verification.
        max_candidates: Maximum n-gram candidates stored per context.
        vocab_size: Vocabulary size (used for random-greedy fallback tests).
        seed: RNG seed.
    """

    window_size: int = 4
    ngram_size: int = 4
    max_candidates: int = 256
    vocab_size: int = 32000
    seed: int = 0

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError(f"window_size must be >= 1; got {self.window_size}")
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2; got {self.ngram_size}")
        if self.vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1; got {self.vocab_size}")


@dataclass
class LookaheadResult:
    """Result of one lookahead decode step.

    Attributes:
        accepted_tokens: List of token ids committed this step (>= 1).
        n_verified: Number of n-gram candidates verified.
        cache_hits: Number of n-gram cache hits.
    """

    accepted_tokens: List[int]
    n_verified: int
    cache_hits: int

    @property
    def speedup_estimate(self) -> float:
        """Tokens per model call this step."""
        return float(len(self.accepted_tokens))


class LookaheadDecode:
    """2D Jacobi window lookahead decoder.

    Requires a **score function** ``score_fn(token_ids) -> logits`` that
    returns ``(vocab_size,)`` logits for the next token after the given
    context. In practice this would be your model's forward pass.

    Example::

        def score_fn(ctx):
            return np.random.randn(32000).astype(np.float32)

        cfg     = LookaheadConfig(window_size=4, ngram_size=4)
        decoder = LookaheadDecode(cfg, score_fn)

        # Start from a context
        ctx     = [1, 2, 3]
        result  = decoder.step(ctx)
        ctx    += result.accepted_tokens
    """

    def __init__(
        self,
        config: Optional[LookaheadConfig] = None,
        score_fn: Optional[Callable[[List[int]], np.ndarray]] = None,
    ) -> None:
        self._cfg = config or LookaheadConfig()
        self._score_fn = score_fn
        self._rng = np.random.default_rng(self._cfg.seed)
        # N-gram cache: context_ngram_prefix → list of candidate continuations
        self._ngram_cache: dict[tuple, List[List[int]]] = {}

    @property
    def config(self) -> LookaheadConfig:
        return self._cfg

    @property
    def cache_size(self) -> int:
        """Total number of cached n-gram candidates."""
        return sum(len(v) for v in self._ngram_cache.values())

    def _greedy(self, logits: np.ndarray) -> int:
        return int(np.argmax(logits))

    def _sample_or_greedy(self, logits: np.ndarray) -> int:
        """Sample or greedy depending on whether score_fn provided."""
        if self._score_fn is None:
            return int(self._rng.integers(0, self._cfg.vocab_size))
        return self._greedy(logits)

    def _add_ngram(self, context: List[int], continuation: List[int]) -> None:
        """Store an observed n-gram in the lookahead cache."""
        cfg = self._cfg
        n = cfg.ngram_size
        if len(continuation) < 1:
            return
        # Key is last (n-1) tokens of context
        key = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)
        cands = self._ngram_cache.setdefault(key, [])
        cands.append(list(continuation[:n]))
        # Trim to max candidates
        if len(cands) > cfg.max_candidates:
            cands.pop(0)

    def _lookup_candidates(self, context: List[int]) -> List[List[int]]:
        """Return stored n-gram candidates matching context tail."""
        cfg = self._cfg
        n = cfg.ngram_size
        key = tuple(context[-(n - 1):]) if len(context) >= n - 1 else tuple(context)
        return list(self._ngram_cache.get(key, []))

    def step(self, context: List[int]) -> LookaheadResult:
        """Execute one decode step with the lookahead window.

        Args:
            context: Current token sequence (prompt + generated so far).

        Returns:
            :class:`LookaheadResult` with accepted tokens.
        """
        cfg = self._cfg
        W = cfg.window_size
        N = cfg.ngram_size

        # --- Lookahead branch: speculate W future positions ---
        jacobi_tokens: List[int] = []
        ctx = list(context)
        for _ in range(W):
            if self._score_fn is not None:
                logits = self._score_fn(ctx)
            else:
                logits = np.zeros(cfg.vocab_size, dtype=np.float32)
            tok = self._sample_or_greedy(logits)
            jacobi_tokens.append(tok)
            ctx.append(tok)

        # Record this n-gram into the cache
        self._add_ngram(context, jacobi_tokens)

        # --- Verification: try n-gram candidates from cache ---
        candidates = self._lookup_candidates(context)
        n_verified = len(candidates)
        cache_hits = 0
        best_accepted: List[int] = []

        for cand in candidates:
            # Verify greedily: check if first token matches model prediction
            if self._score_fn is not None:
                logits = self._score_fn(context)
                predicted = self._greedy(logits)
            else:
                predicted = cand[0] if cand else jacobi_tokens[0]

            if cand and cand[0] == predicted:
                cache_hits += 1
                # Accept the whole n-gram (or as many tokens as match)
                accepted = [predicted]
                ctx_v = list(context) + [predicted]
                for tok in cand[1:len(cand)]:
                    if self._score_fn is not None:
                        lx = self._score_fn(ctx_v)
                        next_tok = self._greedy(lx)
                    else:
                        next_tok = tok
                    if next_tok == tok:
                        accepted.append(tok)
                        ctx_v.append(tok)
                    else:
                        break
                if len(accepted) > len(best_accepted):
                    best_accepted = accepted

        if not best_accepted:
            # Fall back to first Jacobi token
            if self._score_fn is not None:
                logits = self._score_fn(context)
                best_accepted = [self._greedy(logits)]
            else:
                best_accepted = [jacobi_tokens[0]]

        return LookaheadResult(
            accepted_tokens=best_accepted,
            n_verified=n_verified,
            cache_hits=cache_hits,
        )

    def reset_cache(self) -> None:
        """Clear the n-gram lookahead cache."""
        self._ngram_cache.clear()
