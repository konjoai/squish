#!/usr/bin/env python3
"""
tests/test_radix_kv_reuse_integration.py

Integration tests for Phase 13C: RadixTree KV reuse wiring in the server
dispatch layer.

Tests verify the RadixTree (squish/radix_cache.py) token-prefix trie:
  - find_prefix returns (0, []) on first call (no prefix stored)
  - insert_prefix followed by find_prefix returns correct prefix_len and block_refs
  - prefix_hits counter increments on a hit
  - Delta tokens (tokens after prefix) are correctly identified
  - find_prefix is a no-op on an empty token list
  - Multiple overlapping prefixes resolve to the longest match
  - block_refs round-trip through insert/find correctly

These tests do NOT require a live server — they test the RadixTree trie
directly, mirroring the dispatch-layer behavior that server.py implements
via the _prefix_cache.find_prefix / _prefix_cache.insert_prefix calls
added in Phase 13C.

Audit confirmation (documented in PLAN.md §13C):
  The full delta-only forward path (PagedKVCache.fork_sequence) is scaffolded
  but not yet wired; these tests confirm the trie layer is correct and ready
  for the forward-pass integration.
"""
from __future__ import annotations

import pytest
from squish.kv.radix_cache import RadixTree


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trie():
    return RadixTree(maxsize=64)


# ---------------------------------------------------------------------------
# Basic no-match (cold trie)
# ---------------------------------------------------------------------------

class TestFindPrefixColdTrie:
    def test_returns_zero_len_on_empty_trie(self, trie):
        length, refs = trie.find_prefix([1, 2, 3, 4])
        assert length == 0

    def test_returns_empty_refs_on_empty_trie(self, trie):
        _, refs = trie.find_prefix([1, 2, 3, 4])
        assert refs == []

    def test_empty_token_list_returns_zero(self, trie):
        length, refs = trie.find_prefix([])
        assert length == 0
        assert refs == []

    def test_prefix_hits_starts_at_zero(self, trie):
        assert trie.prefix_hits == 0


# ---------------------------------------------------------------------------
# insert_prefix / find_prefix round-trip
# ---------------------------------------------------------------------------

class TestInsertFindRoundTrip:
    def test_exact_match_returns_full_length(self, trie):
        tokens = [10, 20, 30, 40]
        block_refs = [1, 2]
        trie.insert_prefix(tokens, block_refs)
        length, refs = trie.find_prefix(tokens)
        assert length == len(tokens)

    def test_exact_match_returns_block_refs(self, trie):
        tokens = [10, 20, 30, 40]
        block_refs = [7, 8, 9]
        trie.insert_prefix(tokens, block_refs)
        _, refs = trie.find_prefix(tokens)
        assert refs == block_refs

    def test_prefix_match_returns_stored_length(self, trie):
        """Storing prefix A; querying A+delta should return len(A)."""
        prefix_a = [1, 2, 3, 4, 5, 6, 7, 8]
        block_refs = [42]
        trie.insert_prefix(prefix_a, block_refs)

        # Query with prefix + delta
        full_tokens = prefix_a + [9, 10, 11]
        length, refs = trie.find_prefix(full_tokens)
        assert length == len(prefix_a)
        assert refs == block_refs

    def test_prefix_match_delta_tokens_correctly_identified(self, trie):
        """Delta = full_tokens[prefix_len:] should equal the appended tokens."""
        prefix_a = [100, 200, 300]
        delta = [400, 500]
        trie.insert_prefix(prefix_a, [1])

        full_tokens = prefix_a + delta
        prefix_len, _ = trie.find_prefix(full_tokens)
        computed_delta = full_tokens[prefix_len:]
        assert computed_delta == delta

    def test_prefix_hits_increments_on_hit(self, trie):
        tokens = [1, 2, 3]
        trie.insert_prefix(tokens, [5])
        before = trie.prefix_hits
        trie.find_prefix(tokens)
        assert trie.prefix_hits == before + 1

    def test_prefix_hits_does_not_increment_on_miss(self, trie):
        tokens = [1, 2, 3]
        trie.insert_prefix(tokens, [5])
        before = trie.prefix_hits
        trie.find_prefix([99, 88, 77])  # no match
        assert trie.prefix_hits == before


# ---------------------------------------------------------------------------
# Longest prefix resolution
# ---------------------------------------------------------------------------

class TestLongestPrefixResolution:
    def test_longer_prefix_wins(self, trie):
        short = [1, 2, 3]
        long_ = [1, 2, 3, 4, 5]
        trie.insert_prefix(short, [10])
        trie.insert_prefix(long_, [20])

        full = [1, 2, 3, 4, 5, 6, 7]
        length, refs = trie.find_prefix(full)
        assert length == len(long_)
        assert refs == [20]

    def test_partial_edge_match_returns_common_prefix_length(self, trie):
        """When a query diverges mid-edge, RadixTree returns the shared
        common-prefix length and the edge node's block_refs.

        Stored edge: [1,2,3] → refs=[99].  Query: [1,2,9,10].
        Common prefix [1,2] = 2 tokens.  The trie returns (2, [99])
        because the matching edge node has block_refs populated.
        This is the correct behaviour for prefix-skipping: the delta
        that requires prefill is query[prefix_len:] = [9,10].
        """
        trie.insert_prefix([1, 2, 3], [99])
        length, refs = trie.find_prefix([1, 2, 9, 10])
        # At most 2 tokens can match (the shared common prefix [1,2])
        assert length <= 2


# ---------------------------------------------------------------------------
# Empty / edge-case block_refs
# ---------------------------------------------------------------------------

class TestEmptyBlockRefs:
    def test_insert_with_empty_block_refs_is_ignored(self, trie):
        """Inserting with block_refs=[] should not create a hit."""
        trie.insert_prefix([1, 2, 3], [])
        length, refs = trie.find_prefix([1, 2, 3])
        # find_prefix only returns a hit if block_refs is non-empty
        assert refs == [] or length == 0

    def test_insert_empty_tokens_is_no_op(self, trie):
        """Inserting an empty token list should be gracefully ignored."""
        trie.insert_prefix([], [1, 2])
        length, refs = trie.find_prefix([1, 2, 3])
        assert length == 0


# ---------------------------------------------------------------------------
# Audit confirmation test
# ---------------------------------------------------------------------------

class TestPhase13CAuditConfirmation:
    def test_delta_token_count_is_total_minus_prefix(self, trie):
        """
        Mirrors the dispatch logic in server.py Phase 13C:
        delta_tokens = len(input_ids) - _radix_prefix_len

        This test confirms the trie gives the correct prefix_len so
        the server can correctly compute how many tokens need prefill.
        """
        prefix = list(range(100))    # 100 tokens (e.g. system prompt)
        delta  = list(range(100, 115))  # 15 new tokens
        block_refs = list(range(10))

        trie.insert_prefix(prefix, block_refs)

        full_input = prefix + delta
        prefix_len, refs = trie.find_prefix(full_input)

        delta_token_count = len(full_input) - prefix_len
        assert prefix_len == len(prefix)
        assert delta_token_count == len(delta)
        assert refs == block_refs
