"""squish/speculative/lade_decode.py

LADEDecoder — N-gram Lookahead Decoding with Parallel Verification
for accelerated inference without a draft model
(Fu et al., ICML 2024 / arXiv:2401.15077).

Reference
---------
"LADE: Lookahead Decoding with Verification for Accelerating Inference."
Fu et al., ICML 2024 (arXiv:2401.15077).

Algorithm
---------
LADE avoids a separate draft model by exploiting the model's own
n-gram statistics:

1. **Collection phase**: Maintain a rolling n-gram table keyed by the
   last (n-1) tokens; each entry stores the set of observed successor
   tokens from training / recent context.

2. **Lookahead branch generation**: From the current continuation,
   propose ``n_lookahead`` candidate tokens sampled from the n-gram table
   (or uniformly if no history exists).

3. **Verification pass**: The target model evaluates all branches in one
   forward pass (simulated here as sequential calls).

4. **Accept-reject**: Accept each candidate token with probability
   min(1, p_target / p_draft); on rejection, sample a fallback from the
   residual distribution.

Key properties
--------------
* NumPy-only simulation; no GPU dependency.
* ``n_gram`` — n-gram order (window of (n-1) tokens used as key).
* ``n_lookahead`` — lookahead depth / candidate tokens per step.
* ``max_ngram_table`` — maximum n-gram table size (evict oldest on overflow).
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from typing import Callable, DefaultDict, Dict, List, Optional, Set, Tuple

import numpy as np

__all__ = [
    "LADEConfig",
    "LADEDraftResult",
    "LADEDecoder",
]

# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class LADEConfig:
    """Configuration for :class:`LADEDecoder`.

    Attributes:
        n_gram: N-gram order; uses (n_gram-1) context tokens as key.
        n_lookahead: Candidate tokens proposed per decoding step.
        max_ngram_table: Maximum number of unique n-gram keys stored.
        temperature: Sampling temperature for target verify distribution.
        seed: Random seed.
    """

    n_gram: int = 3
    n_lookahead: int = 5
    max_ngram_table: int = 4096
    temperature: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_gram < 2:
            raise ValueError(f"n_gram must be ≥ 2; got {self.n_gram}")
        if self.n_lookahead < 1:
            raise ValueError(f"n_lookahead must be ≥ 1; got {self.n_lookahead}")
        if self.max_ngram_table < 1:
            raise ValueError(f"max_ngram_table must be ≥ 1; got {self.max_ngram_table}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0; got {self.temperature}")

    @property
    def window_size(self) -> int:  # server.py compatibility alias
        return self.n_lookahead

    @property
    def ngram_size(self) -> int:  # server.py compatibility alias
        return self.n_gram


# ── Result ────────────────────────────────────────────────────────────────────


@dataclass
class LADEDraftResult:
    """Outcome of one :meth:`LADEDecoder.step` call.

    Attributes:
        accepted_tokens: Token IDs accepted this step.
        n_accepted: Number of accepted tokens.
        n_proposed: Number of lookahead candidates proposed.
        acceptance_rate: n_accepted / max(n_proposed, 1).
    """

    accepted_tokens: List[int]
    n_accepted: int
    n_proposed: int
    acceptance_rate: float


# ── Decoder ───────────────────────────────────────────────────────────────────


class LADEDecoder:
    """Lookahead decoding with n-gram lookahead and parallel verification.

    Example::

        cfg = LADEConfig(n_gram=3, n_lookahead=4)
        decoder = LADEDecoder(cfg)

        def target_fn(token, context):
            logits = model.forward(token, context)
            return softmax(logits)

        decoder.update_ngram_table([1, 2, 3, 4, 5])
        result = decoder.step([1, 2, 3, 4], target_fn)
    """

    def __init__(self, config: Optional[LADEConfig] = None) -> None:
        self.config = config or LADEConfig()
        self._rng = np.random.default_rng(self.config.seed)
        # n-gram table: Tuple[int, ...] -> ordered set of successor tokens
        self._ngram_table: OrderedDict = OrderedDict()
        self._total_accepted: float = 0.0
        self._total_proposed: int = 0
        self._n_steps: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update_ngram_table(self, token_ids: List[int]) -> None:
        """Populate the n-gram table from an observed token sequence.

        Args:
            token_ids: Observed token IDs (e.g., recent context or training data).
        """
        n = self.config.n_gram
        for i in range(len(token_ids) - n + 1):
            key = tuple(token_ids[i : i + n - 1])
            succ = token_ids[i + n - 1]
            if key not in self._ngram_table:
                if len(self._ngram_table) >= self.config.max_ngram_table:
                    self._ngram_table.popitem(last=False)
                self._ngram_table[key] = set()
            self._ngram_table[key].add(succ)

    def step(
        self,
        context_ids: List[int],
        target_fn: Callable[[int, List[int]], np.ndarray],
    ) -> LADEDraftResult:
        """Run one LADE decoding step.

        Args:
            context_ids: Current context token IDs.
            target_fn: ``(last_token, context) -> probs (vocab_size,)``

        Returns:
            :class:`LADEDraftResult` with accepted tokens and statistics.
        """
        ctx = list(context_ids)
        last = ctx[-1] if ctx else 0
        n = self.config.n_gram

        # ── Phase 1: Propose lookahead candidates ────────────────────────
        candidates: List[int] = []
        for _ in range(self.config.n_lookahead):
            key = tuple(ctx[-(n - 1) :]) if len(ctx) >= n - 1 else tuple(ctx)
            if key in self._ngram_table and self._ngram_table[key]:
                choices = list(self._ngram_table[key])
                cand = int(self._rng.choice(choices))
            else:
                # Fallback: sample uniformly from a small vocab
                target_probs = np.asarray(target_fn(last, ctx), dtype=np.float32)
                target_probs = self._softmax(target_probs / self.config.temperature)
                cand = int(self._rng.choice(len(target_probs), p=target_probs))
            candidates.append(cand)

        # ── Phase 2: Verification ────────────────────────────────────────
        accepted: List[int] = []
        cur_ctx = ctx
        cur_last = last

        for i, cand in enumerate(candidates):
            target_probs = np.asarray(target_fn(cur_last, cur_ctx), dtype=np.float32)
            target_probs = self._softmax(target_probs / self.config.temperature)
            p_target = float(target_probs[cand])

            # Draft probability — uniform over n-gram successors
            n_key = tuple(cur_ctx[-(n - 1) :]) if len(cur_ctx) >= n - 1 else tuple(cur_ctx)
            if n_key in self._ngram_table and self._ngram_table[n_key]:
                p_draft = 1.0 / len(self._ngram_table[n_key])
            else:
                p_draft = 1.0 / len(target_probs)

            accept_prob = min(1.0, p_target / (p_draft + 1e-9))
            if self._rng.random() < accept_prob:
                accepted.append(cand)
                cur_ctx = cur_ctx + [cand]
                cur_last = cand
                # Update n-gram table with accepted token
                self.update_ngram_table([cur_last, cand])
            else:
                # Residual sampling
                draft_probs = np.full(len(target_probs), p_draft, dtype=np.float32)
                draft_probs /= draft_probs.sum()
                residual = np.maximum(target_probs - draft_probs, 0.0)
                s = residual.sum()
                if s > 1e-9:
                    residual /= s
                    fallback = int(self._rng.choice(len(residual), p=residual))
                else:
                    fallback = int(self._rng.choice(len(target_probs), p=target_probs))
                accepted.append(fallback)
                break

        n_acc = len(accepted)
        n_prop = len(candidates)
        self._total_accepted += n_acc
        self._total_proposed += n_prop
        self._n_steps += 1

        return LADEDraftResult(
            accepted_tokens=accepted,
            n_accepted=n_acc,
            n_proposed=n_prop,
            acceptance_rate=n_acc / max(n_prop, 1),
        )

    @property
    def mean_acceptance_rate(self) -> float:
        """Mean acceptance rate over all steps."""
        return self._total_accepted / max(self._total_proposed, 1)

    def reset_stats(self) -> None:
        """Reset acceptance statistics (does NOT clear n-gram table)."""
        self._total_accepted = 0.0
        self._total_proposed = 0
        self._n_steps = 0

    def n_ngram_entries(self) -> int:
        """Number of unique n-gram keys in the table."""
        return len(self._ngram_table)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-9)

    def __repr__(self) -> str:
        cfg = self.config
        return (
            f"LADEDecoder(n_gram={cfg.n_gram}, n_lookahead={cfg.n_lookahead}, "
            f"n_ngram_entries={self.n_ngram_entries()}, "
            f"mean_acceptance_rate={self.mean_acceptance_rate:.3f})"
        )
