# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
"""squish/rag_prefetch.py

RAGPrefetch — Predictive document KV prefetch for retrieval-augmented
generation inference.

In retrieval-augmented generation (RAG) pipelines the same small corpus of
documents is retrieved repeatedly across user turns.  Without pre-warming,
every new turn must recompute the full KV representation of each retrieved
document from scratch, adding tens to hundreds of milliseconds of first-token
latency for long documents.  RAGPrefetch mitigates this by tracking access
patterns and surfacing the most-likely-needed documents as warmup candidates
before they are actually required.

Each document is identified by a SHA-256 hash of its token ID sequence, which
provides a compact, collision-resistant key that is stable across sessions.
The prefetcher maintains a score table where each entry accumulates an
integer access count and a decaying recency score.  On every call to
:meth:`RAGPrefetcher.record_access` all existing scores are multiplied by
``recency_decay`` before the new access is credited, implementing an
exponential moving average that naturally down-weights stale accesses.  This
combined frequency-recency ranking is used by
:meth:`RAGPrefetcher.get_warmup_candidates` to return the top-k documents
most likely to appear in the next retrieval call.

When the number of tracked documents exceeds ``max_docs`` the document with
the lowest current score is evicted, bounding memory usage.  The
``min_accesses`` threshold prevents cold documents from polluting the warmup
list before there is enough evidence of their utility.

Example usage::

    import numpy as np
    from squish.rag_prefetch import RAGConfig, RAGPrefetcher

    cfg       = RAGConfig(max_docs=1024, top_k=16, recency_decay=0.95,
                          min_accesses=2)
    prefetcher = RAGPrefetcher(cfg)

    doc_tokens = np.array([101, 2023, 2003, 1037, 3231], dtype=np.int64)
    h = prefetcher.record_access(doc_tokens)
    prefetcher.record_access(doc_tokens)   # second access clears min_accesses
    candidates = prefetcher.get_warmup_candidates()
    print(f"hash={h[:8]}…, candidates={candidates}")
    print(prefetcher.stats)
"""

from __future__ import annotations

__all__ = ["RAGConfig", "RAGPrefetcher", "RAGStats"]

import hashlib
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RAGConfig:
    """Configuration for the RAG document prefetch predictor.

    Attributes:
        max_docs:       Maximum number of documents to track simultaneously.
                        When exceeded, the lowest-scoring document is evicted.
        top_k:          Number of warmup candidates returned by
                        :meth:`~RAGPrefetcher.get_warmup_candidates`.
        recency_decay:  Multiplicative decay applied to all document scores on
                        each new access.  Must be in ``(0, 1]``.
        min_accesses:   Minimum number of times a document must have been
                        accessed before it is eligible as a warmup candidate.
    """

    max_docs: int = 1024
    top_k: int = 16
    recency_decay: float = 0.95
    min_accesses: int = 2

    def __post_init__(self) -> None:
        if self.max_docs < 1:
            raise ValueError(f"max_docs must be >= 1, got {self.max_docs}")
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if not (0.0 < self.recency_decay <= 1.0):
            raise ValueError(
                f"recency_decay must be in (0, 1], got {self.recency_decay}"
            )
        if self.min_accesses < 1:
            raise ValueError(
                f"min_accesses must be >= 1, got {self.min_accesses}"
            )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class RAGStats:
    """Aggregate statistics for a :class:`RAGPrefetcher`.

    Attributes:
        total_accesses: Total number of :meth:`~RAGPrefetcher.record_access`
                        calls.
        total_evictions: Total number of document evictions due to capacity.
        cache_hits:      Number of accesses where the document was already
                         tracked (i.e., a repeat access).
    """

    total_accesses: int = 0
    total_evictions: int = 0
    cache_hits: int = 0


# ---------------------------------------------------------------------------
# Prefetcher
# ---------------------------------------------------------------------------


class RAGPrefetcher:
    """Tracks document access patterns and predicts warmup candidates.

    Internally maintains a score table mapping SHA-256 hex digests to a
    ``(score, count)`` pair.  Scores are decayed on every access and
    incremented by 1.0 for the newly accessed document, implementing a
    frequency-weighted recency ranking.

    Args:
        config: A :class:`RAGConfig` instance.
    """

    def __init__(self, config: RAGConfig) -> None:
        self._cfg = config
        # Mapping from hex digest → running score (float).
        self._scores: dict[str, float] = {}
        # Mapping from hex digest → total access count (int).
        self._counts: dict[str, int] = {}

        self._total_accesses:  int = 0
        self._total_evictions: int = 0
        self._cache_hits:      int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_access(self, doc_tokens: np.ndarray) -> str:
        """Record an access to the document represented by *doc_tokens*.

        Decays all existing scores by ``recency_decay``, then credits the
        accessed document.  Evicts the lowest-scoring entry if the table
        exceeds ``max_docs``.

        Args:
            doc_tokens: 1-D integer array of token IDs representing the
                        document content.  The array is cast to ``int64``
                        before hashing so the digest is dtype-independent.

        Returns:
            The SHA-256 hex digest of the document.
        """
        tokens = np.asarray(doc_tokens, dtype=np.int64)
        digest = hashlib.sha256(tokens.tobytes()).hexdigest()

        # Decay all scores before crediting the new access.
        for key in self._scores:
            self._scores[key] *= self._cfg.recency_decay

        # Track whether this is a repeat access.
        if digest in self._scores:
            self._cache_hits += 1

        # Credit the accessed document.
        self._scores[digest] = self._scores.get(digest, 0.0) + 1.0
        self._counts[digest] = self._counts.get(digest, 0) + 1
        self._total_accesses += 1

        # Evict lowest-score document if over capacity.
        if len(self._scores) > self._cfg.max_docs:
            evict_key = min(self._scores, key=lambda k: self._scores[k])
            del self._scores[evict_key]
            del self._counts[evict_key]
            self._total_evictions += 1

        return digest

    def get_warmup_candidates(self) -> list[str]:
        """Return up to ``top_k`` document hashes ranked by frequency-recency score.

        Only documents with at least ``min_accesses`` total accesses are
        eligible.  Fewer than ``top_k`` candidates may be returned if not
        enough documents meet the threshold.

        Returns:
            A list of SHA-256 hex digests sorted by descending score.
        """
        eligible = [
            k for k, count in self._counts.items()
            if count >= self._cfg.min_accesses
        ]
        eligible.sort(key=lambda k: self._scores[k], reverse=True)
        return eligible[: self._cfg.top_k]

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_tracked(self) -> int:
        """Current number of documents in the score table."""
        return len(self._scores)

    @property
    def stats(self) -> RAGStats:
        """Return a snapshot of cumulative access and eviction counters."""
        return RAGStats(
            total_accesses=self._total_accesses,
            total_evictions=self._total_evictions,
            cache_hits=self._cache_hits,
        )
