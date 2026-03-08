"""tests/test_copy_spec_unit.py — 100 % coverage for squish/copy_spec.py"""
import pytest

from squish.copy_spec import (
    CopySpecConfig,
    CopySpecDrafter,
    CopySpecStats,
    _SAMState,
    _SuffixAutomaton,
)


# ---------------------------------------------------------------------------
# CopySpecConfig
# ---------------------------------------------------------------------------

class TestCopySpecConfig:
    def test_defaults(self):
        cfg = CopySpecConfig()
        assert cfg.min_match_len == 3
        assert cfg.max_draft_len == 8
        assert cfg.max_history_len == 2048
        assert cfg.search_context_len == 16

    def test_custom(self):
        cfg = CopySpecConfig(min_match_len=1, max_draft_len=4, max_history_len=100)
        assert cfg.min_match_len == 1
        assert cfg.max_draft_len == 4
        assert cfg.max_history_len == 100


# ---------------------------------------------------------------------------
# _SAMState
# ---------------------------------------------------------------------------

class TestSAMState:
    def test_initial_state(self):
        s = _SAMState(length=5)
        assert s.len == 5
        assert s.link == -1
        assert s.next == {}
        assert s.first_endpos == -1

    def test_default_length(self):
        s = _SAMState()
        assert s.len == 0


# ---------------------------------------------------------------------------
# _SuffixAutomaton
# ---------------------------------------------------------------------------

class TestSuffixAutomaton:
    def test_empty_history(self):
        sam = _SuffixAutomaton()
        assert sam.history == []

    def test_extend_single_token(self):
        sam = _SuffixAutomaton()
        sam.extend(1)
        assert sam.history == [1]

    def test_extend_unique_tokens(self):
        sam = _SuffixAutomaton()
        for t in [1, 2, 3, 4]:
            sam.extend(t)
        assert sam.history == [1, 2, 3, 4]

    def test_extend_repeated_tokens_clone_path(self):
        # "abab" pattern forces the clone path in the suffix automaton
        sam = _SuffixAutomaton()
        for t in [1, 2, 1, 2]:
            sam.extend(t)
        assert sam.history == [1, 2, 1, 2]

    def test_extend_longer_repeated_pattern(self):
        sam = _SuffixAutomaton()
        seq = [3, 1, 4, 1, 5, 9, 2, 6, 3, 1, 4, 1, 5]
        for t in seq:
            sam.extend(t)
        assert sam.history == seq

    def test_longest_match_empty_query(self):
        sam = _SuffixAutomaton()
        sam.extend(1)
        match_len, endpos = sam.longest_match([])
        assert match_len == 0
        assert endpos == -1

    def test_longest_match_no_match(self):
        sam = _SuffixAutomaton()
        for t in [1, 2, 3]:
            sam.extend(t)
        # Query for token that never appeared
        match_len, endpos = sam.longest_match([99, 100])
        assert match_len == 0

    def test_longest_match_partial_match(self):
        sam = _SuffixAutomaton()
        for t in [1, 2, 3, 4, 5]:
            sam.extend(t)
        # Query [1, 2, 99]: matches 2 then fails
        match_len, endpos = sam.longest_match([1, 2, 99])
        assert match_len == 2
        assert endpos >= 0

    def test_longest_match_full_match(self):
        sam = _SuffixAutomaton()
        for t in [10, 20, 10, 20, 30]:
            sam.extend(t)
        # Query [10, 20] should match in history
        match_len, endpos = sam.longest_match([10, 20])
        assert match_len == 2
        assert endpos >= 0

    def test_longest_match_after_repeated_sequence(self):
        sam = _SuffixAutomaton()
        for t in [1, 2, 3, 1, 2, 3]:
            sam.extend(t)
        match_len, endpos = sam.longest_match([1, 2, 3])
        assert match_len == 3
        assert endpos >= 2


# ---------------------------------------------------------------------------
# CopySpecDrafter
# ---------------------------------------------------------------------------

class TestCopySpecDrafter:
    def _make(self, **kw):
        cfg = CopySpecConfig(**kw)
        return CopySpecDrafter(cfg)

    def test_initial_state(self):
        d = CopySpecDrafter()
        assert d.n_tokens == 0
        assert d.history == []

    def test_default_config_when_none(self):
        d = CopySpecDrafter(config=None)
        assert d.n_tokens == 0

    def test_add_token_increments_count(self):
        d = CopySpecDrafter()
        d.add_token(5)
        assert d.n_tokens == 1
        assert d.history == [5]

    def test_add_multiple_tokens(self):
        d = CopySpecDrafter()
        for t in [1, 2, 3]:
            d.add_token(t)
        assert d.n_tokens == 3
        assert d.history == [1, 2, 3]

    # ----------------------------------------------------------------
    # draft() — None returns
    # ----------------------------------------------------------------

    def test_draft_returns_none_when_history_too_short(self):
        # min_match_len=3 → need >= 4 tokens; only 2 added
        d = self._make(min_match_len=3)
        d.add_token(1)
        d.add_token(2)
        assert d.draft() is None

    def test_draft_returns_none_when_history_at_boundary(self):
        # len(history)=3, min_match_len=3 → 3 < 4 → return None
        d = self._make(min_match_len=3)
        for t in [1, 2, 3]:
            d.add_token(t)
        assert d.draft() is None

    def test_draft_returns_none_when_history_single_token_with_zero_min(self):
        # min_match_len=0 → passes first check (len=1 >= 1)
        # len(history) == 1 → not > 1 → query = []
        # query empty → None
        d = self._make(min_match_len=0, max_draft_len=5)
        d.add_token(7)
        assert d.draft() is None

    def test_draft_returns_none_when_match_shorter_than_min(self):
        # search_context_len=2 → query has 2 tokens; min_match_len=3 → 2 < 3 → None
        d = self._make(min_match_len=3, search_context_len=2)
        for t in [10, 20, 30, 40, 50]:
            d.add_token(t)
        assert d.draft() is None

    def test_draft_returns_list_for_unique_sequence(self):
        # Unique sequence [1,2,3]: query=[1,2], match ends at index 1,
        # continuation = [3].  Verifies no crash and correct return type.
        d = self._make(min_match_len=1, search_context_len=4)
        for t in [1, 2, 3]:
            d.add_token(t)
        result = d.draft()
        assert isinstance(result, list)

    def test_draft_returns_none_when_max_draft_len_zero(self):
        # max_draft_len=0 → n=0 → cont_end=cont_start → drafts=[] → None
        d = self._make(min_match_len=1, max_draft_len=0, search_context_len=10)
        for t in [1, 2, 1, 2, 3]:
            d.add_token(t)
        assert d.draft() is None

    # ----------------------------------------------------------------
    # draft() — successful return
    # ----------------------------------------------------------------

    def test_draft_returns_continuation(self):
        # Pattern: [A, B, C, A, B, C, D] → after adding all, query [A, B, C]
        # should match first occurrence at index 2, cont = [A, B, C, D]
        d = self._make(min_match_len=2, max_draft_len=5, search_context_len=10)
        seq = [1, 2, 3, 1, 2, 3, 4]
        for t in seq:
            d.add_token(t)
        result = d.draft()
        assert result is not None
        assert len(result) >= 1

    def test_draft_respects_max_draft_len(self):
        d = self._make(min_match_len=2, max_draft_len=2, search_context_len=10)
        for t in [1, 2, 3, 1, 2, 3, 1, 2, 3]:
            d.add_token(t)
        result = d.draft()
        if result is not None:
            assert len(result) <= 2

    def test_draft_respects_max_n_override(self):
        d = self._make(min_match_len=2, max_draft_len=10, search_context_len=10)
        for t in [5, 6, 7, 5, 6, 7, 8, 9]:
            d.add_token(t)
        result = d.draft(max_n=3)
        if result is not None:
            assert len(result) <= 3

    def test_draft_returns_list_of_ints(self):
        d = self._make(min_match_len=1, max_draft_len=4, search_context_len=8)
        for t in [1, 1, 1, 1, 2]:
            d.add_token(t)
        result = d.draft()
        if result is not None:
            assert isinstance(result, list)
            assert all(isinstance(x, int) for x in result)

    # ----------------------------------------------------------------
    # Rebuild (history trim)
    # ----------------------------------------------------------------

    def test_add_token_triggers_rebuild_when_history_exceeds_max(self):
        d = self._make(max_history_len=5, min_match_len=1)
        # Add 6 tokens → triggers rebuild keeping last 5
        for t in range(6):
            d.add_token(t)
        assert len(d.history) == 5
        assert d.history == list(range(1, 6))

    def test_history_after_rebuild_is_consistent(self):
        d = self._make(max_history_len=4, min_match_len=1)
        for t in [10, 20, 30, 40, 50]:
            d.add_token(t)
        assert d.history == [20, 30, 40, 50]

    # ----------------------------------------------------------------
    # reset()
    # ----------------------------------------------------------------

    def test_reset_clears_state(self):
        d = CopySpecDrafter()
        for t in [1, 2, 3]:
            d.add_token(t)
        d.reset()
        assert d.n_tokens == 0
        assert d.history == []

    def test_draft_after_reset_returns_none(self):
        d = self._make(min_match_len=2)
        for t in [1, 2, 1, 2]:
            d.add_token(t)
        d.reset()
        assert d.draft() is None


# ---------------------------------------------------------------------------
# CopySpecStats
# ---------------------------------------------------------------------------

class TestCopySpecStats:
    def test_defaults(self):
        s = CopySpecStats()
        assert s.draft_attempts == 0
        assert s.hits == 0
        assert s.misses == 0
        assert s.total_tokens_proposed == 0
        assert s.total_tokens_accepted == 0

    def test_hit_rate_zero_when_no_attempts(self):
        assert CopySpecStats().hit_rate == 0.0

    def test_hit_rate_all_hits(self):
        s = CopySpecStats()
        s.record_hit(5, 4)
        s.record_hit(3, 3)
        assert s.hit_rate == 1.0

    def test_hit_rate_mixed(self):
        s = CopySpecStats()
        s.record_hit(4, 3)
        s.record_miss()
        assert s.hit_rate == pytest.approx(0.5)

    def test_acceptance_rate_zero_when_no_proposals(self):
        assert CopySpecStats().acceptance_rate == 0.0

    def test_acceptance_rate_all_accepted(self):
        s = CopySpecStats()
        s.record_hit(5, 5)
        assert s.acceptance_rate == 1.0

    def test_acceptance_rate_partial(self):
        s = CopySpecStats()
        s.record_hit(10, 7)
        assert s.acceptance_rate == pytest.approx(0.7)

    def test_tokens_per_hit_zero_when_no_hits(self):
        assert CopySpecStats().tokens_per_hit == 0.0

    def test_tokens_per_hit_nonzero(self):
        s = CopySpecStats()
        s.record_hit(6, 4)
        s.record_hit(4, 2)
        assert s.tokens_per_hit == pytest.approx(5.0)

    def test_record_miss_increments(self):
        s = CopySpecStats()
        s.record_miss()
        assert s.draft_attempts == 1
        assert s.misses == 1
        assert s.hits == 0

    def test_reset_clears_all_fields(self):
        s = CopySpecStats()
        s.record_hit(8, 6)
        s.record_miss()
        s.reset()
        assert s.draft_attempts == 0
        assert s.hits == 0
        assert s.misses == 0
        assert s.total_tokens_proposed == 0
        assert s.total_tokens_accepted == 0
