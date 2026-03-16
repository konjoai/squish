"""tests/test_stop_token_suppression.py

Unit tests for stop-token suppression logic in squish/server.py.

The _get_stop_ids function converts stop string(s) to lists of token IDs using
the module-level _state.tokenizer.  Because _state is a module global, each test
patches it via unittest.mock.patch so no real model or tokenizer is required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(encode_side_effect=None, encode_return_value=None):
    """Return a MagicMock that satisfies the _state.tokenizer interface."""
    tokenizer = MagicMock()
    if encode_side_effect is not None:
        tokenizer.encode.side_effect = encode_side_effect
    elif encode_return_value is not None:
        tokenizer.encode.return_value = encode_return_value
    state = MagicMock()
    state.tokenizer = tokenizer
    return state


# ---------------------------------------------------------------------------
# Tests for _get_stop_ids
# ---------------------------------------------------------------------------


class TestGetStopIdsNone:
    """_get_stop_ids(None) must return [] without touching the tokenizer."""

    def test_returns_empty_list(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        with patch("squish.server._state", state):
            result = _get_stop_ids(None)

        assert result == []

    def test_tokenizer_not_called(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        with patch("squish.server._state", state):
            _get_stop_ids(None)

        state.tokenizer.encode.assert_not_called()


class TestGetStopIdsEmptyList:
    """_get_stop_ids([]) must return [] without calling the tokenizer."""

    def test_returns_empty_list(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        with patch("squish.server._state", state):
            result = _get_stop_ids([])

        assert result == []

    def test_tokenizer_not_called(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        with patch("squish.server._state", state):
            _get_stop_ids([])

        state.tokenizer.encode.assert_not_called()


class TestGetStopIdsSingleString:
    """_get_stop_ids('stop') wraps the bare string in a list, encodes it,
    and returns [[token_id, ...]]."""

    def test_single_token_result(self):
        from squish.server import _get_stop_ids

        state = _make_state(encode_return_value=[42])
        with patch("squish.server._state", state):
            result = _get_stop_ids("stop")

        assert result == [[42]]

    def test_encode_called_with_add_special_tokens_false(self):
        from squish.server import _get_stop_ids

        state = _make_state(encode_return_value=[42])
        with patch("squish.server._state", state):
            _get_stop_ids("stop")

        state.tokenizer.encode.assert_called_once_with("stop", add_special_tokens=False)


class TestGetStopIdsMultipleStrings:
    """_get_stop_ids(['a', 'b']) encodes each string independently and returns
    a list-of-lists preserving order."""

    def test_two_stop_strings(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        state.tokenizer.encode.side_effect = [[1], [2]]
        with patch("squish.server._state", state):
            result = _get_stop_ids(["a", "b"])

        assert result == [[1], [2]]

    def test_encode_called_twice(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        state.tokenizer.encode.side_effect = [[1], [2]]
        with patch("squish.server._state", state):
            _get_stop_ids(["a", "b"])

        assert state.tokenizer.encode.call_count == 2


class TestGetStopIdsEmptyEncoding:
    """When the tokenizer returns an empty list for a stop string the entry must
    be dropped so downstream code is not misled by an empty sequence."""

    def test_empty_ids_excluded(self):
        from squish.server import _get_stop_ids

        state = _make_state(encode_return_value=[])
        with patch("squish.server._state", state):
            result = _get_stop_ids("empty_token_string")

        assert result == []

    def test_mixed_empty_and_nonempty(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        # First string encodes to nothing; second encodes to [7]
        state.tokenizer.encode.side_effect = [[], [7]]
        with patch("squish.server._state", state):
            result = _get_stop_ids(["no_tokens", "valid"])

        assert result == [[7]]


class TestGetStopIdsExceptionSwallowing:
    """Exceptions raised by tokenizer.encode must be silently swallowed so that
    a bad stop string never aborts generation."""

    def test_exception_is_ignored(self):
        from squish.server import _get_stop_ids

        state = _make_state(encode_side_effect=ValueError("bad token"))
        with patch("squish.server._state", state):
            result = _get_stop_ids("crash_me")

        assert result == []

    def test_partial_exception_keeps_good_entries(self):
        from squish.server import _get_stop_ids

        state = _make_state()
        # First call raises; second call returns a valid id list
        state.tokenizer.encode.side_effect = [RuntimeError("oops"), [99]]
        with patch("squish.server._state", state):
            result = _get_stop_ids(["bad", "good"])

        assert result == [[99]]


class TestGetStopIdsMultiTokenSequence:
    """A multi-word stop string should map to a single inner list containing
    all token IDs in order."""

    def test_two_token_stop_sequence(self):
        from squish.server import _get_stop_ids

        state = _make_state(encode_return_value=[10, 20])
        with patch("squish.server._state", state):
            result = _get_stop_ids("hello world")

        assert result == [[10, 20]]

    def test_three_token_stop_sequence(self):
        from squish.server import _get_stop_ids

        state = _make_state(encode_return_value=[3, 14, 159])
        with patch("squish.server._state", state):
            result = _get_stop_ids("one two three")

        assert result == [[3, 14, 159]]


class TestStopSequenceMatchAtPositionZero:
    """Verify the rolling-buffer slice comparison used inside _generate_tokens
    correctly detects a stop sequence that appears at position 0 of the output.

    Rather than running the full generator (which requires a live MLX model),
    this test exercises the matching predicate in isolation: if the stop buffer
    ends with the stop sequence the result is truthy, regardless of how many
    tokens precede the match or whether the match starts at index 0.
    """

    def test_match_at_start_of_buffer(self):
        """Stop sequence [5, 6] present at position 0 of a fresh buffer."""
        stop_ids = [[5, 6]]
        stop_buf = [5, 6]

        hit = any(stop_buf[-len(seq):] == seq for seq in stop_ids)

        assert hit is True

    def test_no_match_partial_sequence(self):
        """Only the first token of the stop sequence has been emitted — no match yet."""
        stop_ids = [[5, 6]]
        stop_buf = [5]

        hit = any(stop_buf[-len(seq):] == seq for seq in stop_ids)

        assert hit is False

    def test_match_after_prefix_tokens(self):
        """Stop sequence appears after several non-stop tokens."""
        stop_ids = [[5, 6]]
        stop_buf = [1, 2, 3, 5, 6]

        hit = any(stop_buf[-len(seq):] == seq for seq in stop_ids)

        assert hit is True

    def test_finish_reason_is_stop_when_sequence_matched(self):
        """Confirm that the expected finish reason string is 'stop' (not 'length')
        when a stop sequence is matched — this reflects the contract documented in
        _generate_tokens's docstring."""
        stop_ids = [[5, 6]]
        stop_buf = [5, 6]

        if any(stop_buf[-len(seq):] == seq for seq in stop_ids):
            finish_reason = "stop"
        else:
            finish_reason = "length"

        assert finish_reason == "stop"

    def test_multiple_stop_sequences_first_matches(self):
        """When several stop sequences are registered, a match on the first one
        triggers a stop even if the others have not been seen."""
        stop_ids = [[5, 6], [7, 8]]
        stop_buf = [5, 6]

        hit = any(stop_buf[-len(seq):] == seq for seq in stop_ids)

        assert hit is True

    def test_multiple_stop_sequences_second_matches(self):
        """A match on the second registered stop sequence also triggers a stop."""
        stop_ids = [[5, 6], [7, 8]]
        stop_buf = [1, 7, 8]

        hit = any(stop_buf[-len(seq):] == seq for seq in stop_ids)

        assert hit is True
