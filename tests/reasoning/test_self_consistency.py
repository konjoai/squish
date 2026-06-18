"""Tests for SelfConsistencyVoter — majority-vote aggregation over CoT chains.

Centred on the tie-break contract (alphabetically *first*), plus the vote()
integration and answer-extraction paths this previously-untested module exposes.
"""
from __future__ import annotations

import pytest

from squish.reasoning.self_consistency import (
    SelfConsistencyConfig,
    SelfConsistencyVoter,
)


def _voter(**kw) -> SelfConsistencyVoter:
    return SelfConsistencyVoter(SelfConsistencyConfig(**kw))


class TestMajorityVote:
    def test_clear_winner(self):
        v = _voter()
        assert v.majority_vote({"a": 3, "b": 1}) == "a"

    def test_tie_broken_alphabetically_first(self):
        # Both have 1 vote → must return the alphabetically-first ("apple").
        v = _voter()
        assert v.majority_vote({"banana": 1, "apple": 1}) == "apple"

    def test_tie_among_top_only_ignores_lower(self):
        # 'zeta' and 'alpha' tie at the top (2); 'mid' has fewer. Winner = 'alpha'.
        v = _voter()
        assert v.majority_vote({"zeta": 2, "alpha": 2, "mid": 1}) == "alpha"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _voter().majority_vote({})


class TestVoteIntegration:
    def test_vote_picks_majority(self):
        # No answer_pattern → the last non-empty line of each chain is the answer.
        v = _voter(normalise_answers=True)
        chains = ["reasoning here\n42", "a different path\n42", "wrong turn\n7"]
        result = v.vote(chains)
        assert result.winner == "42"
        assert result.vote_counts["42"] == 2
        assert result.n_chains == 3
        assert result.winner_vote_share == pytest.approx(2 / 3)

    def test_vote_tie_is_alphabetically_first(self):
        v = _voter()
        result = v.vote(["banana", "apple"])
        assert result.winner == "apple"

    def test_vote_empty_raises(self):
        with pytest.raises(ValueError):
            _voter().vote([])


class TestExtractAnswer:
    def test_last_nonempty_line_fallback(self):
        v = _voter(normalise_answers=False)
        assert v.extract_answer("step one\nstep two\nfinal") == "final"

    def test_pattern_extraction(self):
        v = _voter(answer_pattern=r"answer is (\w+)", normalise_answers=True)
        assert v.extract_answer("...the answer is Cat") == "cat"

    def test_normalisation_collapses_whitespace(self):
        v = _voter(normalise_answers=True)
        assert v.extract_answer("  Hello   World  ") == "hello world"
