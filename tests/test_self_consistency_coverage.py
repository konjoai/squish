"""Behavioral coverage for the validation and extraction edge paths of
``squish.reasoning.self_consistency`` left untested by the baseline suite.
Pure-Python; no MLX.
"""
from __future__ import annotations

import pytest

from squish.reasoning.self_consistency import (
    SelfConsistencyConfig,
    SelfConsistencyVoter,
)


def test_config_rejects_small_k():
    with pytest.raises(ValueError, match="k must be"):
        SelfConsistencyConfig(k=0)


def test_config_rejects_nonpositive_temperature():
    with pytest.raises(ValueError, match="temperature must be"):
        SelfConsistencyConfig(temperature=0.0)


def test_extract_answer_pattern_match():
    voter = SelfConsistencyVoter(SelfConsistencyConfig(k=2, answer_pattern=r"answer:\s*(\w+)"))
    assert voter.extract_answer("reasoning...\nanswer: forty") == "forty"


def test_extract_answer_pattern_no_match_falls_back_to_last_line():
    # Pattern present but absent from the chain → fallback to last non-empty line
    # (the 133→138 branch).
    voter = SelfConsistencyVoter(SelfConsistencyConfig(k=2, answer_pattern=r"answer:\s*(\w+)"))
    assert voter.extract_answer("step one\nstep two\n42") == "42"


def test_extract_answer_no_pattern_uses_last_line():
    voter = SelfConsistencyVoter(SelfConsistencyConfig(k=2))
    assert voter.extract_answer("thinking\nFINAL") == "final"  # normalised lowercase


def test_vote_majority():
    voter = SelfConsistencyVoter(SelfConsistencyConfig(k=3))
    result = voter.vote(["the answer is 7", "the answer is 7", "the answer is 9"])
    assert result.winner == "the answer is 7"
    assert result.n_chains == 3
    assert result.winner_vote_share == pytest.approx(2 / 3)


def test_majority_vote_empty_raises():
    voter = SelfConsistencyVoter(SelfConsistencyConfig(k=2))
    with pytest.raises(ValueError, match="non-empty"):
        voter.majority_vote({})
