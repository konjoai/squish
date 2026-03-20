"""squish/speculative/rest_decode.py

RESTDecode — Retrieval-Based Speculative Decoding via N-Gram Datastore
(He et al., NAACL 2024 / arXiv:2311.08252).

Reference
---------
"REST: Retrieval-Based Speculative Decoding." He et al., NAACL 2024
(arXiv:2311.08252).

Algorithm
---------
1. Maintain an n-gram datastore mapping ``(t_0,...,t_{n-2}) → List[t_{n-1}]``.
2. At each decode step, look up the last ``n_gram - 1`` context tokens.
3. Propose up to ``top_k_draft`` next-token candidates from the datastore.
4. Verify proposals with a cheap speculative accept-reject: compare target
   probabilities vs. uniform over proposed set.
5. Accept tokens greedily until the first rejection; sample target distribution
   at the rejection point.

Complexity
----------
* O(1) datastore lookup (Python dict).
* The datastore is populated lazily from any token stream passed to
  ``add_to_datastore()``.
* Datastore is capped at ``max_datastore`` entries (oldest removed on overflow).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "RESTConfig",
    "RESTDraftResult",
    "RESTDecode",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class RESTConfig:
    """Configuration for :class:`RESTDecode`.

    Attributes:
        n_gram: Context window size for n-gram lookup (must be ≥ 2).
        max_datastore: Maximum number of n-gram entries to retain.
        top_k_draft: Max draft candidates to propose per step.
        temperature: Sampling temperature for fallback.
        seed: RNG seed (0 = deterministic).
    """

    n_gram: int = 3
    max_datastore: int = 65_536
    top_k_draft: int = 5
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_gram < 2:
            raise ValueError(f"n_gram must be ≥ 2; got {self.n_gram}")
        if self.max_datastore < 1:
            raise ValueError(f"max_datastore must be ≥ 1; got {self.max_datastore}")
        if self.top_k_draft < 1:
            raise ValueError(f"top_k_draft must be ≥ 1; got {self.top_k_draft}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class RESTDraftResult:
    """Outcome of one :meth:`RESTDecode.step` call.

    Attributes:
        accepted_tokens: Tokens accepted (including fallback token at end).
        n_accepted: Count of accepted speculative tokens (excluding fallback).
        n_proposed: Number of tokens proposed by the datastore.
        acceptance_rate: n_accepted / n_proposed (or 0.0 if no proposals).
    """

    accepted_tokens: List[int]
    n_accepted: int
    n_proposed: int
    acceptance_rate: float


# ── RESTDecode ────────────────────────────────────────────────────────────────


class RESTDecode:
    """Retrieval-based speculative n-gram decoder.

    Example::

        cfg = RESTConfig(n_gram=3, top_k_draft=4, seed=1)
        dec = RESTDecode(cfg)
        dec.add_to_datastore([1, 2, 3, 4, 3, 5, 1, 2, 7])

        def target_fn(last_token, context):
            vocab = 20
            probs = np.ones(vocab) / vocab
            return probs

        result = dec.step([1, 2], target_fn)
    """

    def __init__(self, config: Optional[RESTConfig] = None) -> None:
        self.config = config or RESTConfig()
        self._rng = np.random.default_rng(self.config.seed)
        # n-gram datastore: tuple[int, ...] → ordered unique list of next tokens
        self._store: "OrderedDict[Tuple[int, ...], List[int]]" = OrderedDict()
        self._total_accepted = 0
        self._total_proposed = 0
        self._total_steps = 0

    # ── Datastore ─────────────────────────────────────────────────────────────

    def add_to_datastore(self, token_ids: Sequence[int]) -> None:
        """Populate the datastore from a token sequence.

        Each consecutive ``n_gram``-length window produces one entry.

        Args:
            token_ids: Flat list of integer token ids.
        """
        n = self.config.n_gram
        ids = list(token_ids)
        for i in range(len(ids) - n + 1):
            ctx = tuple(ids[i : i + n - 1])
            next_tok = ids[i + n - 1]
            if ctx not in self._store:
                if len(self._store) >= self.config.max_datastore:
                    self._store.popitem(last=False)  # LRU evict
                self._store[ctx] = []
            if next_tok not in self._store[ctx]:
                self._store[ctx].append(next_tok)
            self._store.move_to_end(ctx)

    def datastore_size(self) -> int:
        """Return the number of unique n-gram contexts stored."""
        return len(self._store)

    # ── Decoding Step ─────────────────────────────────────────────────────────

    def step(
        self,
        context_ids: Sequence[int],
        target_fn: Callable[[int, List[int]], np.ndarray],
    ) -> RESTDraftResult:
        """Run one speculative decode step.

        Args:
            context_ids: Current context token ids (must be ≥ ``n_gram - 1``).
            target_fn: ``(last_token, context) → probabilities (vocab_size,)``
                callable implementing the target model.

        Returns:
            :class:`RESTDraftResult`.
        """
        cfg = self.config
        ctx = list(context_ids)
        lookup_key = tuple(ctx[-(cfg.n_gram - 1) :])

        # Lookup datastore candidates.
        candidates: List[int] = []
        if lookup_key in self._store:
            candidates = self._store[lookup_key][: cfg.top_k_draft]

        n_proposed = len(candidates)
        accepted: List[int] = []
        current_ctx = list(ctx)

        # Speculative accept-reject.
        for cand_token in candidates:
            probs = _safe_softmax(target_fn(current_ctx[-1], current_ctx), cfg.temperature)
            if probs[cand_token] >= (1.0 / len(probs)):
                accepted.append(cand_token)
                current_ctx.append(cand_token)
            else:
                # Rejection: sample from target and stop.
                fallback = int(self._rng.choice(len(probs), p=probs))
                accepted.append(fallback)
                self._total_accepted += len(accepted) - 1
                self._total_proposed += n_proposed
                self._total_steps += 1
                rate = (len(accepted) - 1) / n_proposed if n_proposed > 0 else 0.0
                return RESTDraftResult(accepted, len(accepted) - 1, n_proposed, rate)

        # All proposals accepted: sample one more from target.
        probs = _safe_softmax(
            target_fn(current_ctx[-1], current_ctx), cfg.temperature
        )
        fallback = int(self._rng.choice(len(probs), p=probs))
        accepted.append(fallback)

        n_accepted = n_proposed
        self._total_accepted += n_accepted
        self._total_proposed += n_proposed
        self._total_steps += 1
        rate = n_accepted / n_proposed if n_proposed > 0 else 0.0
        return RESTDraftResult(accepted, n_accepted, n_proposed, rate)

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def mean_acceptance_rate(self) -> float:
        """Mean acceptance rate (n_accepted / n_proposed over all steps)."""
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed

    def reset_stats(self) -> None:
        """Reset running acceptance statistics."""
        self._total_accepted = 0
        self._total_proposed = 0
        self._total_steps = 0

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"RESTDecode(n_gram={cfg.n_gram}, top_k_draft={cfg.top_k_draft}, "
            f"max_datastore={cfg.max_datastore})"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / max(temperature, 1e-9)
    scaled -= scaled.max()
    exp = np.exp(scaled)
    total = exp.sum()
    if total == 0.0:
        return np.ones(len(logits)) / len(logits)
    return exp / total
