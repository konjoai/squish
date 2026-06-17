"""Tests for TokenDecodeCache — memoized single-token detokenization."""
from __future__ import annotations

import pytest

from squish.serving.token_decode_cache import TokenDecodeCache


class _FakeTokenizer:
    """Records every decode call so we can assert on memoization."""

    def __init__(self, table: dict[int, str] | None = None):
        self.table = table or {}
        self.calls: list[list[int]] = []

    def decode(self, ids: list[int]) -> str:
        self.calls.append(list(ids))
        return self.table.get(ids[0], f"<{ids[0]}>")


class TestTokenDecodeCache:
    def test_matches_direct_decode(self):
        tok = _FakeTokenizer({1: "hello", 2: " world"})
        cache = TokenDecodeCache(tok)
        assert cache.decode(1) == "hello"
        assert cache.decode(2) == " world"
        # Equivalence with the call the hot path would otherwise make.
        assert cache.decode(1) == tok.decode([1])

    def test_memoizes_repeat_ids(self):
        tok = _FakeTokenizer({7: "x"})
        cache = TokenDecodeCache(tok)
        for _ in range(5):
            assert cache.decode(7) == "x"
        # decode([7]) invoked once by the cache (plus any direct test calls would add).
        assert tok.calls.count([7]) == 1
        assert cache.size == 1

    def test_caches_empty_string(self):
        # An id that decodes to "" must still be cached (is-not-None, not truthiness).
        tok = _FakeTokenizer({0: ""})
        cache = TokenDecodeCache(tok)
        assert cache.decode(0) == ""
        assert cache.decode(0) == ""
        assert tok.calls.count([0]) == 1

    def test_bounded_size(self):
        tok = _FakeTokenizer()
        cache = TokenDecodeCache(tok, max_entries=3)
        for i in range(10):
            cache.decode(i)
        assert cache.size == 3
        # Beyond the cap, decode still works (just not stored).
        assert cache.decode(99) == "<99>"

    def test_clear(self):
        tok = _FakeTokenizer({1: "a"})
        cache = TokenDecodeCache(tok)
        cache.decode(1)
        assert cache.size == 1
        cache.clear()
        assert cache.size == 0
        cache.decode(1)
        assert tok.calls.count([1]) == 2  # re-decoded after clear

    def test_rejects_nonpositive_max(self):
        with pytest.raises(ValueError):
            TokenDecodeCache(_FakeTokenizer(), max_entries=0)
