# [Experimental] This module is part of Squish v6+ (Waves 19–26).
# Proof-of-concept quality: API and behaviour may change without notice.
# For stable core functionality see Waves 1–12 in MODULES.md.
#!/usr/bin/env python3
"""
squish/cache_warmup.py

CacheWarmup — Predictive KV cache pre-warming from access patterns.

By tracking which prefixes are accessed most frequently, we can pre-warm the KV
cache for likely-to-arrive requests, reducing TTFT (Time To First Token) on hot
paths.  Each distinct prefix is identified by a rolling hash of its token IDs
(truncated to ``max_prefix_tokens`` tokens), and an :class:`AccessRecord` is
maintained that counts how many times that prefix has been seen and when it was
last accessed.

:meth:`CacheWarmupPredictor.get_warmup_candidates` returns the top-k prefix
hashes that satisfy the minimum access threshold, ranked by a combined score of
access frequency and recency.  The calling code is responsible for actually
warming the cache; this module only provides the prediction signal.

Example usage::

    import time
    from squish.kv.cache_warmup import WarmupConfig, CacheWarmupPredictor

    config = WarmupConfig(top_k=8, min_access_count=2, max_prefix_tokens=128)
    predictor = CacheWarmupPredictor(config)

    tokens = [1, 2, 3, 4, 5]
    predictor.record_access(tokens, timestamp=time.monotonic())
    predictor.record_access(tokens, timestamp=time.monotonic())

    candidates = predictor.get_warmup_candidates()
    print(f"warmup candidates: {candidates}")
    print(f"tracked prefixes: {predictor.n_tracked}")
    print(predictor.stats)
"""

from __future__ import annotations

__all__ = [
    "WarmupConfig",
    "AccessRecord",
    "CacheWarmupPredictor",
    "WarmupStats",
]

from dataclasses import dataclass
from typing import Optional

import numpy as np  # noqa: F401  — imported for dtype compatibility in future extensions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class WarmupConfig:
    """Configuration for the predictive cache warmup predictor.

    Attributes:
        top_k:             Maximum number of prefix hashes to return from
                           :meth:`CacheWarmupPredictor.get_warmup_candidates`.
                           Must be >= 1.
        min_access_count:  Minimum number of times a prefix must have been seen
                           before it is considered a warmup candidate.  Prevents
                           one-off requests from polluting the warmup list.
                           Must be >= 1.
        max_prefix_tokens: Maximum number of leading tokens used to compute the
                           prefix hash.  Longer prefixes are silently truncated.
                           Must be >= 1.
    """

    top_k: int = 32
    min_access_count: int = 3
    max_prefix_tokens: int = 256

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        if self.min_access_count < 1:
            raise ValueError(
                f"min_access_count must be >= 1, got {self.min_access_count}"
            )
        if self.max_prefix_tokens < 1:
            raise ValueError(
                f"max_prefix_tokens must be >= 1, got {self.max_prefix_tokens}"
            )


# ---------------------------------------------------------------------------
# Access record
# ---------------------------------------------------------------------------


@dataclass
class AccessRecord:
    """Tracks access history for a single prefix hash.

    Attributes:
        prefix_hash:   Integer hash of the truncated prefix token sequence.
        access_count:  Total number of times this prefix has been recorded.
        last_access:   Timestamp of the most recent access (monotonic seconds).
    """

    prefix_hash: int
    access_count: int = 0
    last_access: float = 0.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class WarmupStats:
    """Cumulative statistics for the :class:`CacheWarmupPredictor`.

    Attributes:
        total_accesses:         Total calls to :meth:`record_access`.
        cache_warmups_issued:   Cumulative number of candidate hashes returned
                                by :meth:`get_warmup_candidates`.
        predicted_hits:         Number of subsequent accesses whose prefix hash
                                matched an issued warmup candidate (requires
                                :meth:`record_predicted_hit` to be called).
    """

    total_accesses: int = 0
    cache_warmups_issued: int = 0
    predicted_hits: int = 0


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class CacheWarmupPredictor:
    """Predictive KV-cache warmup advisor based on prefix access history.

    Maintains an in-memory table of :class:`AccessRecord` instances keyed by
    prefix hash.  On each :meth:`record_access` call the appropriate record is
    created or updated.  :meth:`get_warmup_candidates` returns the top-k prefix
    hashes whose access count meets ``min_access_count``, ranked by the score::

        score = access_count * (1 + log(1 + time_since_last_access + 1e-9))^{-1}

    which favours both frequently *and* recently accessed prefixes.

    Args:
        config: A :class:`WarmupConfig` describing the predictor's behaviour.
    """

    def __init__(self, config: WarmupConfig) -> None:
        self._config = config
        self._records: dict[int, AccessRecord] = {}
        self._stats = WarmupStats()
        # Track the most recent timestamp seen to compute relative recency.
        self._latest_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_access(self, prefix_tokens: list[int], timestamp: float) -> None:
        """Record an access for *prefix_tokens* at *timestamp*.

        The prefix is truncated to ``config.max_prefix_tokens`` before hashing.
        If the resulting hash has not been seen before, a new
        :class:`AccessRecord` is created with ``access_count=1``.

        Args:
            prefix_tokens: Sequence of token IDs forming the request's prefix.
                           Must be non-empty.
            timestamp:     Monotonic timestamp of the access (e.g. from
                           ``time.monotonic()``).  Must be >= 0.

        Raises:
            ValueError: if *prefix_tokens* is empty or *timestamp* is negative.
        """
        if not prefix_tokens:
            raise ValueError("prefix_tokens must be a non-empty list")
        if timestamp < 0.0:
            raise ValueError(f"timestamp must be >= 0, got {timestamp}")

        truncated = prefix_tokens[: self._config.max_prefix_tokens]
        h = _hash_tokens(truncated)

        if h not in self._records:
            self._records[h] = AccessRecord(prefix_hash=h, access_count=0, last_access=0.0)

        record = self._records[h]
        record.access_count += 1
        record.last_access = timestamp
        self._latest_ts = max(self._latest_ts, timestamp)
        self._stats.total_accesses += 1

    def get_warmup_candidates(self) -> list[int]:
        """Return prefix hashes most likely to benefit from KV pre-warming.

        Only prefixes with ``access_count >= config.min_access_count`` are
        eligible.  Candidates are ranked by a recency-weighted frequency score
        and at most ``config.top_k`` hashes are returned.

        Returns:
            An ordered list of prefix hash integers (highest priority first).
            The list may be shorter than ``top_k`` if fewer eligible prefixes
            exist.
        """
        eligible = [
            r for r in self._records.values()
            if r.access_count >= self._config.min_access_count
        ]
        if not eligible:
            return []

        scored = [
            (self._score(r), r.prefix_hash) for r in eligible
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [ph for _, ph in scored[: self._config.top_k]]

        self._stats.cache_warmups_issued += len(top)
        return top

    def record_predicted_hit(self, prefix_hash: int) -> None:
        """Notify the predictor that a pre-warmed prefix was actually used.

        Increments ``stats.predicted_hits``.  Call this when the serving layer
        confirms that a cache hit occurred for a prefix that was previously
        returned as a warmup candidate.

        Args:
            prefix_hash: The hash value that was a warmup candidate and was hit.
        """
        self._stats.predicted_hits += 1

    @property
    def n_tracked(self) -> int:
        """Number of distinct prefix hashes currently tracked."""
        return len(self._records)

    @property
    def stats(self) -> WarmupStats:
        """Cumulative predictor statistics."""
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(self, record: AccessRecord) -> float:
        """Compute a recency-weighted frequency score for *record*.

        The score rewards prefixes that are both frequently *and* recently
        accessed::

            score = access_count / (1 + age_seconds)

        where ``age_seconds = latest_ts - last_access``.
        """
        import math
        age = max(0.0, self._latest_ts - record.last_access)
        return record.access_count / (1.0 + age)


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------


def _hash_tokens(tokens: list[int]) -> int:
    """Compute a stable integer hash for a token list.

    Uses Python's built-in ``hash`` on a frozen ``tuple`` of token IDs.
    The result fits in a 64-bit integer on CPython.

    Args:
        tokens: Non-empty list of integer token IDs.
    """
    return hash(tuple(tokens))
