"""squish/serving/token_decode_cache.py — memoized single-token detokenization.

During streaming generation the scheduler calls ``tokenizer.decode([id])`` once
per generated token. That decode is deterministic per token id, so the same ids
(common words, punctuation, whitespace) get re-decoded constantly across a
response and across concurrent requests.

This cache memoizes ``id -> text`` so the work happens once per distinct id. It
is strictly behaviour-preserving: each id maps to exactly the string an isolated
``tokenizer.decode([id])`` produces — the same call the hot path already makes,
just not repeated. The cache is bounded so a pathological stream of distinct ids
cannot grow it without limit.
"""

from __future__ import annotations

from typing import Any

# Bounded so the cache cannot exceed a typical vocabulary's worth of entries.
_DEFAULT_MAX_ENTRIES = 1 << 16  # 65_536


class TokenDecodeCache:
    """Memoizes single-token decodes for the generation hot path."""

    __slots__ = ("_tokenizer", "_max_entries", "_cache")

    def __init__(self, tokenizer: Any, max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._tokenizer = tokenizer
        self._max_entries = max_entries
        self._cache: dict[int, str] = {}

    def decode(self, token_id: int) -> str:
        """Return the detokenized text for *token_id*, decoding on first sight.

        Uses ``is not None`` rather than truthiness so an id that legitimately
        decodes to an empty string is still cached (and not re-decoded).
        """
        cached = self._cache.get(token_id)
        if cached is not None:
            return cached
        text = self._tokenizer.decode([token_id])
        if len(self._cache) < self._max_entries:
            self._cache[token_id] = text
        return text

    def clear(self) -> None:
        """Drop all cached entries (e.g. when the tokenizer changes)."""
        self._cache.clear()

    @property
    def size(self) -> int:
        """Number of distinct ids currently cached."""
        return len(self._cache)
