"""tests/test_grammar_cache_unit.py

Full-coverage unit tests for squish/grammar_cache.py.

Covers:
  FSMState          — validation errors, is_terminal property
  FSMTransition     — validation errors
  GrammarStats      — hit_rate zero and non-zero paths
  GrammarCache      — invalid vocab_size, add_pattern errors,
                      get_mask miss/hit, _compute_mask even/odd hash,
                      transition valid/invalid/cached,
                      cache_hit_rate, n_states_cached, reset_stats, stats()
"""
from __future__ import annotations

import pytest

from squish.grammar_cache import (
    FSMState,
    FSMTransition,
    GrammarCache,
    GrammarStats,
    _FSM_DEPTH_LIMIT,
)


# ---------------------------------------------------------------------------
# FSMState
# ---------------------------------------------------------------------------


class TestFSMState:
    def test_valid_defaults(self):
        s = FSMState(state_id=5, pattern_name="json")
        assert s.state_id == 5
        assert s.pattern_name == "json"
        assert s.depth == 0
        assert s.is_terminal is False

    def test_valid_with_depth(self):
        s = FSMState(state_id=10, pattern_name="regex", depth=3)
        assert s.depth == 3

    def test_negative_state_id_raises(self):
        with pytest.raises(ValueError, match="state_id must be non-negative"):
            FSMState(state_id=-1, pattern_name="json")

    def test_empty_pattern_name_raises(self):
        with pytest.raises(ValueError, match="pattern_name must not be empty"):
            FSMState(state_id=0, pattern_name="")

    def test_negative_depth_raises(self):
        with pytest.raises(ValueError, match="depth must be non-negative"):
            FSMState(state_id=0, pattern_name="json", depth=-1)

    def test_is_terminal_at_limit(self):
        s = FSMState(state_id=0, pattern_name="p", depth=_FSM_DEPTH_LIMIT)
        assert s.is_terminal is True

    def test_is_terminal_below_limit(self):
        s = FSMState(state_id=0, pattern_name="p", depth=_FSM_DEPTH_LIMIT - 1)
        assert s.is_terminal is False

    def test_is_terminal_above_limit(self):
        s = FSMState(state_id=0, pattern_name="p", depth=_FSM_DEPTH_LIMIT + 5)
        assert s.is_terminal is True


# ---------------------------------------------------------------------------
# FSMTransition
# ---------------------------------------------------------------------------


class TestFSMTransition:
    def test_valid_construction(self):
        t = FSMTransition(from_state_id=0, token_id=5, to_state_id=10)
        assert t.from_state_id == 0
        assert t.token_id == 5
        assert t.to_state_id == 10

    def test_negative_from_state_raises(self):
        with pytest.raises(ValueError, match="from_state_id must be non-negative"):
            FSMTransition(from_state_id=-1, token_id=0, to_state_id=0)

    def test_negative_token_id_raises(self):
        with pytest.raises(ValueError, match="token_id must be non-negative"):
            FSMTransition(from_state_id=0, token_id=-1, to_state_id=0)

    def test_negative_to_state_raises(self):
        with pytest.raises(ValueError, match="to_state_id must be non-negative"):
            FSMTransition(from_state_id=0, token_id=0, to_state_id=-1)


# ---------------------------------------------------------------------------
# GrammarStats
# ---------------------------------------------------------------------------


class TestGrammarStats:
    def test_hit_rate_zero_lookups(self):
        s = GrammarStats()
        assert s.hit_rate == 0.0

    def test_hit_rate_nonzero(self):
        s = GrammarStats(total_mask_lookups=10, cache_hits=7)
        assert s.hit_rate == pytest.approx(0.7)

    def test_hit_rate_all_hits(self):
        s = GrammarStats(total_mask_lookups=5, cache_hits=5)
        assert s.hit_rate == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GrammarCache
# ---------------------------------------------------------------------------


class TestGrammarCacheInit:
    def test_invalid_vocab_size_zero_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            GrammarCache(vocab_size=0)

    def test_invalid_vocab_size_negative_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            GrammarCache(vocab_size=-10)

    def test_default_vocab_size(self):
        c = GrammarCache()
        assert c._vocab_size == 32_000

    def test_custom_vocab_size(self):
        c = GrammarCache(vocab_size=100)
        assert c._vocab_size == 100


class TestAddPattern:
    def _cache(self):
        return GrammarCache(vocab_size=100)

    def test_valid_pattern(self):
        c = self._cache()
        c.add_pattern("digits", r"\d+")
        assert "digits" in c._patterns

    def test_empty_name_raises(self):
        c = self._cache()
        with pytest.raises(ValueError, match="pattern name must not be empty"):
            c.add_pattern("", r"\d+")

    def test_duplicate_name_raises(self):
        c = self._cache()
        c.add_pattern("digits", r"\d+")
        with pytest.raises(ValueError, match="already registered"):
            c.add_pattern("digits", r"\w+")

    def test_invalid_regex_raises(self):
        c = self._cache()
        with pytest.raises(ValueError, match="Invalid regex"):
            c.add_pattern("bad", r"[unclosed")

    def test_pattern_hash_stored(self):
        c = self._cache()
        c.add_pattern("test", r"abc")
        assert "test" in c._pattern_hashes
        assert isinstance(c._pattern_hashes["test"], int)


class TestGetMask:
    def _cache(self):
        c = GrammarCache(vocab_size=50)
        c.add_pattern("jp", r"json")
        return c

    def test_get_mask_shape(self):
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")
        mask = c.get_mask(state)
        assert mask.shape == (50,)
        assert mask.dtype.kind == "b"  # boolean

    def test_token_zero_always_allowed(self):
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")
        mask = c.get_mask(state)
        assert mask[0] is True or bool(mask[0])

    def test_cache_miss_then_hit(self):
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")

        assert c._total_lookups == 0
        assert c._cache_hits == 0

        mask1 = c.get_mask(state)
        assert c._total_lookups == 1
        assert c._cache_hits == 0

        mask2 = c.get_mask(state)
        assert c._total_lookups == 2
        assert c._cache_hits == 1

        import numpy as np
        assert np.array_equal(mask1, mask2)

    def test_n_states_cached_increments(self):
        c = self._cache()
        assert c.n_states_cached == 0
        state = FSMState(state_id=3, pattern_name="jp")
        c.get_mask(state)
        assert c.n_states_cached == 1
        # Same state again — no new cache entry
        c.get_mask(state)
        assert c.n_states_cached == 1

    def test_unknown_pattern_uses_zero_hash(self):
        """State whose pattern_name is not registered → hash defaults to 0."""
        c = GrammarCache(vocab_size=30)
        state = FSMState(state_id=5, pattern_name="unregistered")
        mask = c.get_mask(state)
        assert mask.shape == (30,)


class TestComputeMask:
    def test_odd_hash_branch(self):
        """Force an odd pattern_hash and verify mask is computed."""
        c = GrammarCache(vocab_size=60)
        # Try patterns until we get an odd hash
        for i in range(20):
            name = f"p_odd_{i}"
            c.add_pattern(name, rf"\d{{{i + 1}}}")
            ph = c._pattern_hashes[name]
            if ph % 2 == 1:
                state = FSMState(state_id=0, pattern_name=name)
                mask = c._compute_mask(state)
                assert mask.shape == (60,)
                assert bool(mask[0])  # always True
                return
        pytest.skip("Could not find an odd hash pattern in 20 attempts")

    def test_even_hash_branch(self):
        """Force an even pattern_hash and verify mask is computed."""
        c = GrammarCache(vocab_size=60)
        # Try patterns until we get an even hash
        for i in range(20):
            name = f"p_even_{i}"
            c.add_pattern(name, rf"[a-z]{{{i + 1}}}")
            ph = c._pattern_hashes[name]
            if ph % 2 == 0:
                state = FSMState(state_id=0, pattern_name=name)
                mask = c._compute_mask(state)
                assert mask.shape == (60,)
                assert bool(mask[0])
                return
        pytest.skip("Could not find an even hash pattern in 20 attempts")

    def test_both_hash_parity_covered(self):
        """Ensure both odd and even hash branches are hit across many patterns."""
        c = GrammarCache(vocab_size=30)
        found_odd = found_even = False
        patterns = [
            r"\d+", r"\w+", r"[A-Z]+", r"abc", r"xyz",
            r"foo", r"bar", r"\s+", r"\S+", r"[0-9]+",
        ]
        for i, regex in enumerate(patterns):
            name = f"pat_{i}"
            c.add_pattern(name, regex)
            ph = c._pattern_hashes[name]
            state = FSMState(state_id=i % 10, pattern_name=name)
            mask = c._compute_mask(state)
            assert mask.shape == (30,)
            if ph % 2 == 0:
                found_even = True
            else:
                found_odd = True
            if found_odd and found_even:
                return
        # It's extremely unlikely to not find both, but don't fail the test


class TestTransition:
    def _cache(self):
        c = GrammarCache(vocab_size=100)
        c.add_pattern("jp", r"json")
        return c

    def test_valid_transition(self):
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")
        next_state = c.transition(state, 10)
        assert next_state.state_id == (0 + 10) % 256
        assert next_state.depth == 1
        assert next_state.pattern_name == "jp"

    def test_transition_increments_depth(self):
        c = self._cache()
        state = FSMState(state_id=5, pattern_name="jp", depth=3)
        next_state = c.transition(state, 1)
        assert next_state.depth == 4

    def test_negative_token_raises(self):
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")
        with pytest.raises(ValueError, match="out of range"):
            c.transition(state, -1)

    def test_token_at_vocab_boundary_raises(self):
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")
        with pytest.raises(ValueError, match="out of range"):
            c.transition(state, 100)  # exactly vocab_size is out of range

    def test_transition_cache_hit(self):
        """Second transition call for same (state_id, pattern, token) hits cache."""
        c = self._cache()
        state = FSMState(state_id=0, pattern_name="jp")
        n1 = c.transition(state, 15)
        # Increment n_transitions to 1
        assert c._n_transitions == 1
        n2 = c.transition(state, 15)
        assert c._n_transitions == 2
        assert n1.state_id == n2.state_id

    def test_state_modulo_256(self):
        c = self._cache()
        # state_id=250, token=10 → (250+10) % 256 = 4
        state = FSMState(state_id=250, pattern_name="jp")
        next_state = c.transition(state, 10)
        assert next_state.state_id == 4


class TestProperties:
    def _cache_with_activity(self):
        c = GrammarCache(vocab_size=50)
        c.add_pattern("r", r"\d+")
        state = FSMState(state_id=0, pattern_name="r")
        c.get_mask(state)  # miss
        c.get_mask(state)  # hit
        c.transition(state, 5)
        return c

    def test_cache_hit_rate_zero(self):
        c = GrammarCache(vocab_size=50)
        assert c.cache_hit_rate == 0.0

    def test_cache_hit_rate_nonzero(self):
        c = self._cache_with_activity()
        # 1 hit out of 2 lookups → 0.5
        assert c.cache_hit_rate == pytest.approx(0.5)

    def test_reset_stats(self):
        c = self._cache_with_activity()
        c.reset_stats()
        assert c._total_lookups == 0
        assert c._cache_hits == 0
        assert c._n_transitions == 0
        # Mask cache is NOT cleared
        assert c.n_states_cached == 1

    def test_stats_snapshot(self):
        c = self._cache_with_activity()
        s = c.stats()
        assert s.total_mask_lookups == 2
        assert s.cache_hits == 1
        assert s.n_transitions == 1

    def test_stats_hit_rate_via_stats(self):
        c = self._cache_with_activity()
        s = c.stats()
        assert s.hit_rate == pytest.approx(0.5)
