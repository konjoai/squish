"""
squish/fr_spec.py

FR-Spec — Frequency-Ranked Vocabulary Compression for Speculative Decoding.

Based on:
  "FR-Spec: Accelerating Large Vocabulary Language Models via
   Frequency-Ranked Speculative Sampling"
  ACL 2025 — arXiv:2502.14856; github.com/thunlp/FR-Spec

Problem
-------
Speculative decoding with large-vocabulary models (Qwen3 has 151,936 tokens)
incurs significant overhead in the draft model's LM-head computation:
  draft_lm_head_cost ∝ hidden_dim × vocab_size  (matrix multiply)

FR-Spec observes that across realistic task distributions, only ~20-25% of
the vocabulary is ever drafted.  The remaining 75-80% of tokens are never
produced by the draft model, yet the full matrix multiply is paid every step.

Solution
--------
1. **Calibrate** — run the draft model on a representative corpus and rank
   tokens by draft-output frequency.  The ``top_k_fraction`` most frequent
   tokens (default 25%) form the ``FreqTokenSubset``.

2. **Compress** — slice the draft LM-head weight matrix to keep only the rows
   (or columns, depending on transposition convention) corresponding to the
   frequent-token indices.  The resulting ``FRSpecHead`` is ~4× smaller.

3. **Expand** — after the compressed draft produces logits over the frequent
   subset, scatter them back to the full vocabulary for acceptance checking.
   Tokens outside the subset receive logit = -inf during drafting only; the
   *target* model always uses its full vocabulary.

Conflict-resolved notes (Master Conflict Report)
-------------------------------------------------
- **Synergy with EAGLE-3**: FR-Spec modifies the draft LM-head internally.
  Apply FR-Spec calibration directly to the EAGLE-3 draft head when available.
- **Synergy with XGrammar**: structured outputs constrain to a small token
  subset that typically overlaps heavily with FR-Spec's frequent subset.
- No conflict with CommVQ, SpinQuant, or any KV technique.

Provides
--------
  FRSpecConfig       — calibration and compression parameters.
  FreqTokenSubset    — the calibrated frequent token index set.
  FRSpecHead         — compressed LM-head wrapper with expand/compress.
  FRSpecCalibrator   — corpus-driven calibration helper.
  FRSpecStats        — counters for monitoring head-reduction benefit.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np

__all__ = [
    "FRSpecConfig",
    "FreqTokenSubset",
    "FRSpecHead",
    "FRSpecCalibrator",
    "FRSpecStats",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FRSpecConfig:
    """Parameters controlling FR-Spec vocabulary compression.

    Parameters
    ----------
    vocab_size:
        Full vocabulary size (e.g. 151_936 for Qwen3).
    top_k_fraction:
        Fraction of vocabulary to retain in the frequent subset.
        FR-Spec paper uses 0.20–0.25, achieving 75–80% LM-head reduction.
    min_frequent_tokens:
        Hard lower bound on the frequent subset size regardless of fraction.
    max_calibration_samples:
        Maximum number of token lists used during calibration.
    """

    vocab_size: int = 151_936
    top_k_fraction: float = 0.25
    min_frequent_tokens: int = 256
    max_calibration_samples: int = 10_000

    @property
    def k(self) -> int:
        """Computed frequent-subset size."""
        raw = math.ceil(self.vocab_size * self.top_k_fraction)
        return max(raw, self.min_frequent_tokens)


# ---------------------------------------------------------------------------
# FreqTokenSubset
# ---------------------------------------------------------------------------

class FreqTokenSubset:
    """Immutable set of frequent-token indices with fast membership test.

    Attributes
    ----------
    indices:
        Sorted array of token IDs in the frequent subset (shape ``(k,)``).
    """

    def __init__(self, indices: "np.ndarray | Sequence[int]") -> None:
        arr = np.asarray(indices, dtype=np.int64)
        if arr.ndim != 1:
            raise ValueError("indices must be a 1-D array")
        self._indices: np.ndarray = np.sort(arr)
        self._set: frozenset[int] = frozenset(self._indices.tolist())

    @property
    def indices(self) -> np.ndarray:
        return self._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __contains__(self, token_id: object) -> bool:
        return token_id in self._set

    def __iter__(self):
        return iter(self._indices)

    def coverage(self, token_sequence: Sequence[int]) -> float:
        """Fraction of *token_sequence* tokens covered by this subset."""
        if not token_sequence:
            return 1.0
        hits = sum(1 for t in token_sequence if t in self._set)
        return hits / len(token_sequence)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def to_list(self) -> List[int]:
        return self._indices.tolist()

    @classmethod
    def from_list(cls, data: List[int]) -> "FreqTokenSubset":
        return cls(np.array(data, dtype=np.int64))


# ---------------------------------------------------------------------------
# FRSpecHead
# ---------------------------------------------------------------------------

class FRSpecHead:
    """Compressed draft LM-head that projects to the frequent-token subset.

    Parameters
    ----------
    full_weight:
        Original LM-head weight matrix, shape ``(vocab_size, hidden_dim)``
        (standard PyTorch/MLX convention: rows = output logit indices,
        columns = hidden features).
    subset:
        The ``FreqTokenSubset`` identifying which rows to retain.
    """

    def __init__(
        self,
        full_weight: np.ndarray,
        subset: FreqTokenSubset,
    ) -> None:
        if full_weight.ndim != 2:
            raise ValueError(
                f"full_weight must be 2-D (vocab_size, hidden_dim); "
                f"got shape {full_weight.shape}"
            )
        vocab_size, hidden_dim = full_weight.shape
        if len(subset) > vocab_size:
            raise ValueError(
                f"subset size ({len(subset)}) exceeds vocab_size ({vocab_size})"
            )
        self._full_vocab_size: int = vocab_size
        self._hidden_dim: int = hidden_dim
        self._subset: FreqTokenSubset = subset
        self._compressed_weight: np.ndarray = full_weight[subset.indices]  # (k, hidden)

    @property
    def subset(self) -> FreqTokenSubset:
        return self._subset

    @property
    def full_vocab_size(self) -> int:
        return self._full_vocab_size

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def compressed_weight(self) -> np.ndarray:
        """Weight matrix restricted to the frequent-token rows, shape ``(k, hidden_dim)``."""
        return self._compressed_weight

    @property
    def compression_ratio(self) -> float:
        """Fraction of the original LM-head compute retained."""
        return len(self._subset) / self._full_vocab_size

    def forward(self, hidden: np.ndarray) -> np.ndarray:
        """Compute compressed draft logits.

        Parameters
        ----------
        hidden:
            Last-layer hidden state, shape ``(hidden_dim,)`` or
            ``(batch, hidden_dim)``.

        Returns
        -------
        logits:
            Logits over the frequent subset only, shape ``(k,)`` or
            ``(batch, k)``.
        """
        return hidden @ self._compressed_weight.T

    def expand_logits(self, compressed_logits: np.ndarray) -> np.ndarray:
        """Scatter compressed logits back to the full vocabulary.

        Tokens outside the frequent subset receive ``-inf`` (will never be
        sampled during drafting; the target model always uses full vocab).

        Parameters
        ----------
        compressed_logits:
            Shape ``(k,)`` or ``(batch, k)``.

        Returns
        -------
        full_logits:
            Shape ``(vocab_size,)`` or ``(batch, vocab_size)``.
        """
        batch_mode = compressed_logits.ndim == 2
        if batch_mode:
            batch, k = compressed_logits.shape
            full = np.full((batch, self._full_vocab_size), -np.inf, dtype=compressed_logits.dtype)
            full[:, self._subset.indices] = compressed_logits
        else:
            k = compressed_logits.shape[0]
            full = np.full(self._full_vocab_size, -np.inf, dtype=compressed_logits.dtype)
            full[self._subset.indices] = compressed_logits
        return full

    def compress_logits(self, full_logits: np.ndarray) -> np.ndarray:
        """Extract the frequent-subset slice from full-vocabulary logits.

        Used for offline calibration or hybrid draft strategies.

        Parameters
        ----------
        full_logits:
            Shape ``(vocab_size,)`` or ``(batch, vocab_size)``.

        Returns
        -------
        compressed_logits:
            Shape ``(k,)`` or ``(batch, k)``.
        """
        if full_logits.ndim == 2:
            return full_logits[:, self._subset.indices]
        return full_logits[self._subset.indices]


# ---------------------------------------------------------------------------
# FRSpecCalibrator
# ---------------------------------------------------------------------------

class FRSpecCalibrator:
    """Build a :class:`FreqTokenSubset` from draft-model output samples.

    Usage::

        cal = FRSpecCalibrator(config)
        for batch in corpus:
            # draft_outputs is a list/array of token IDs produced by the
            # draft model across one batch
            cal.record(draft_outputs)
        subset = cal.build_subset()

    Parameters
    ----------
    config:
        FR-Spec configuration.  Only ``vocab_size``, ``top_k_fraction``, and
        ``min_frequent_tokens`` are used here.
    """

    def __init__(self, config: Optional[FRSpecConfig] = None) -> None:
        self._config = config or FRSpecConfig()
        self._counter: Counter[int] = Counter()
        self._n_samples: int = 0

    @property
    def n_samples(self) -> int:
        """Number of token-batch recordings accumulated so far."""
        return self._n_samples

    def record(self, token_ids: Iterable[int]) -> None:
        """Record a batch of token IDs produced by the draft model.

        Parameters
        ----------
        token_ids:
            Iterable of integer token IDs from one draft step or one batch.
        """
        if self._n_samples >= self._config.max_calibration_samples:
            return
        tokens = list(token_ids)
        self._counter.update(tokens)
        self._n_samples += 1

    def most_common(self, n: Optional[int] = None) -> List[tuple[int, int]]:
        """Return the ``n`` most frequent (token_id, count) pairs."""
        return self._counter.most_common(n)

    def build_subset(self) -> FreqTokenSubset:
        """Build and return the :class:`FreqTokenSubset` from recorded token IDs.

        If fewer distinct tokens than ``config.k`` were observed, the
        remaining slots are filled with the lowest-index token IDs to
        guarantee the requested subset size.
        """
        k = self._config.k
        vocab_size = self._config.vocab_size

        # Top-k by frequency
        top_tokens = [tok for tok, _ in self._counter.most_common(k)]

        # Pad with low-index tokens if not enough observed
        if len(top_tokens) < k:
            seen = set(top_tokens)
            for tid in range(vocab_size):
                if tid not in seen:
                    top_tokens.append(tid)
                if len(top_tokens) >= k:
                    break

        indices = np.array(top_tokens[:k], dtype=np.int64)
        return FreqTokenSubset(indices)

    def reset(self) -> None:
        """Clear all accumulated counts."""
        self._counter.clear()
        self._n_samples = 0


# ---------------------------------------------------------------------------
# FRSpecStats
# ---------------------------------------------------------------------------

@dataclass
class FRSpecStats:
    """Runtime counters for monitoring FR-Spec head-reduction benefit.

    Attributes
    ----------
    compressed_forwards:
        Number of compressed draft-head forward passes executed.
    full_forwards:
        Number of full-vocabulary draft-head forward passes executed
        (fallback path when FR-Spec is disabled).
    tokens_drafted:
        Total draft tokens produced.
    tokens_outside_subset:
        Draft tokens that fell outside the frequent subset
        (indicates subset quality degradation — should be near zero).
    """

    compressed_forwards: int = 0
    full_forwards: int = 0
    tokens_drafted: int = 0
    tokens_outside_subset: int = 0

    @property
    def compression_utilization(self) -> float:
        """Fraction of forwards that used the compressed head."""
        total = self.compressed_forwards + self.full_forwards
        return self.compressed_forwards / total if total else 0.0

    @property
    def subset_coverage_rate(self) -> float:
        """Fraction of drafted tokens that were inside the frequent subset."""
        if self.tokens_drafted == 0:
            return 1.0
        in_subset = self.tokens_drafted - self.tokens_outside_subset
        return in_subset / self.tokens_drafted

    def record_compressed(self, n_tokens: int = 1) -> None:
        self.compressed_forwards += 1
        self.tokens_drafted += n_tokens

    def record_full(self, n_tokens: int = 1) -> None:
        self.full_forwards += 1
        self.tokens_drafted += n_tokens

    def record_outside_subset(self, count: int = 1) -> None:
        self.tokens_outside_subset += count

    def reset(self) -> None:
        self.compressed_forwards = 0
        self.full_forwards = 0
        self.tokens_drafted = 0
        self.tokens_outside_subset = 0
