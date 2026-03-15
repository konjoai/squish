#!/usr/bin/env python3
"""
tests/serving/test_stop_token_suppression.py

Unit tests for the stop token / stop sequence suppression logic in
squish/server.py (Phase 15).

Two complementary areas are covered:

1. ``_get_stop_ids(stop)``
   The pure helper that converts caller-supplied stop strings into lists of
   token IDs.  Tested by injecting a mock tokenizer so no MLX model is
   required.

2. Stop-buffer matching (inline decode-loop pattern)
   The rolling-buffer check ``stop_buf[-len(seq):] == seq`` is not extracted
   into a named function in server.py; it lives inline in three decode paths.
   A local ``_match_stop_sequence`` mirror is defined here so every branch
   can be exercised as a pure function.

3. Rolling-window trimmer
   The guard ``if len(stop_buf) > 64: stop_buf = stop_buf[-64:]`` is mirrored
   in ``_trim_stop_buf`` and tested for boundary conditions.

Coverage targets
----------------
_get_stop_ids
  - None input  →  empty list returned
  - Single string input  →  normalised to list and encoded
  - List of strings  →  one token-ID sequence per entry
  - String that encodes to an empty token list  →  excluded from result
  - tokenizer.encode() raising an exception  →  silently swallowed
  - Mixed valid / empty encodings  →  only valid entries kept

_match_stop_sequence (inline rolling-buffer logic)
  - Empty stop_ids list  →  never matches
  - Single-token sequence: match when token is at the buffer tail
  - Single-token sequence: no match when token is absent
  - Multi-token sequence: match when tail equals the full sequence
  - Multi-token partial tail  →  no match
  - Multiple candidate sequences: first match triggers True
  - Buffer shorter than the sequence  →  no match, no IndexError
  - Empty buffer  →  no match
  - Sequence equals the full buffer  →  match
  - Matching sequence in the middle (not at the tail)  →  no match

_trim_stop_buf (rolling-window guard)
  - Buffer under 64 tokens  →  returned unchanged
  - Buffer exactly 64 tokens  →  returned unchanged
  - Buffer of 65 tokens  →  trimmed to the last 64
  - Buffer well over 64 tokens  →  trimmed to the last 64
  - Empty buffer  →  returned unchanged
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Optional import of _get_stop_ids from squish.server
# ---------------------------------------------------------------------------
try:
    import squish.server as _srv
    _HAS_SERVER = True
except ImportError:  # pragma: no cover
    _HAS_SERVER = False


# ---------------------------------------------------------------------------
# Local mirrors of the inline decode-loop stop logic
# ---------------------------------------------------------------------------

def _match_stop_sequence(
    stop_buf: list[int],
    stop_ids: list[list[int]],
) -> bool:
    """
    Return True when the tail of *stop_buf* matches any token-ID sequence in
    *stop_ids*.

    Mirrors the pattern used in all three decode paths in server.py:

        for seq in stop_ids:
            if stop_buf[-len(seq):] == seq:
                ...
    """
    for seq in stop_ids:
        if stop_buf[-len(seq):] == seq:
            return True
    return False


def _trim_stop_buf(stop_buf: list[int], max_len: int = 64) -> list[int]:
    """
    Trim *stop_buf* to the last *max_len* entries.

    Mirrors the rolling-window guard in server.py:

        if len(stop_buf) > 64:
            stop_buf = stop_buf[-64:]
    """
    if len(stop_buf) > max_len:
        return stop_buf[-max_len:]
    return stop_buf


# ---------------------------------------------------------------------------
# Tests: _get_stop_ids via mock tokenizer
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_SERVER, reason="squish.server not importable")
class TestGetStopIds:
    """Tests for _get_stop_ids; a mock tokenizer avoids needing a real model."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tokenizer(encoding_map: dict[str, list[int]]) -> MagicMock:
        tok = MagicMock()
        tok.encode.side_effect = lambda text, **_kw: encoding_map.get(text, [])
        return tok

    def _call(self, stop, tokenizer):
        """Call _get_stop_ids with the given mock tokenizer injected."""
        with patch.object(_srv._state, "tokenizer", tokenizer):
            return _srv._get_stop_ids(stop)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_none_returns_empty_list(self):
        tok = self._make_tokenizer({})
        assert self._call(None, tok) == []

    def test_single_string_encoded_to_id_list(self):
        tok = self._make_tokenizer({"<|end|>": [100]})
        result = self._call("<|end|>", tok)
        assert result == [[100]]

    def test_list_of_strings_encoded_separately(self):
        tok = self._make_tokenizer({"<|end|>": [100], "STOP": [200, 201]})
        result = self._call(["<|end|>", "STOP"], tok)
        assert [100] in result
        assert [200, 201] in result
        assert len(result) == 2

    def test_empty_encoding_excluded_from_result(self):
        """A stop string that tokenises to no IDs must be silently dropped."""
        tok = self._make_tokenizer({"<|end|>": []})
        assert self._call("<|end|>", tok) == []

    def test_exception_during_encode_silently_ignored(self):
        tok = MagicMock()
        tok.encode.side_effect = RuntimeError("tokeniser offline")
        assert self._call("<|end|>", tok) == []

    def test_mixed_valid_and_empty_encodings(self):
        """Only the string with a non-empty encoding should appear in result."""
        tok = self._make_tokenizer({"<|end|>": [999], "empty": []})
        result = self._call(["<|end|>", "empty"], tok)
        assert result == [[999]]

    def test_multi_token_stop_string_preserved_as_sequence(self):
        tok = self._make_tokenizer({"</s>": [1, 2, 3]})
        result = self._call("</s>", tok)
        assert result == [[1, 2, 3]]


# ---------------------------------------------------------------------------
# Tests: _match_stop_sequence (inline rolling-buffer logic mirror)
# ---------------------------------------------------------------------------

class TestMatchStopSequence:

    def test_empty_stop_ids_never_matches(self):
        assert _match_stop_sequence([1, 2, 3], []) is False

    def test_single_token_match_at_buffer_tail(self):
        assert _match_stop_sequence([10, 20, 100], [[100]]) is True

    def test_single_token_no_match_when_absent(self):
        assert _match_stop_sequence([10, 20, 30], [[100]]) is False

    def test_multi_token_sequence_matches_at_tail(self):
        assert _match_stop_sequence([1, 2, 3, 4, 5], [[3, 4, 5]]) is True

    def test_multi_token_partial_tail_does_not_match(self):
        """Buffer ends with [4, 5] but the full sequence is [3, 4, 5]."""
        assert _match_stop_sequence([1, 2, 4, 5], [[3, 4, 5]]) is False

    def test_multiple_sequences_first_matching_one_triggers_true(self):
        stop_ids = [[99, 0], [3, 4, 5]]
        assert _match_stop_sequence([1, 2, 3, 4, 5], stop_ids) is True

    def test_buffer_shorter_than_sequence_no_match_no_error(self):
        """Sequence longer than buffer: Python slice returns empty list → no match."""
        assert _match_stop_sequence([1], [[1, 2, 3]]) is False

    def test_empty_buffer_returns_false(self):
        assert _match_stop_sequence([], [[100]]) is False

    def test_sequence_equal_to_full_buffer_matches(self):
        assert _match_stop_sequence([7, 8, 9], [[7, 8, 9]]) is True

    def test_matching_sequence_in_middle_not_at_tail_no_match(self):
        """[3, 4, 5] appears at positions 0-2; the tail is [5, 10] — no match."""
        assert _match_stop_sequence([3, 4, 5, 10], [[3, 4, 5]]) is False

    def test_none_of_multiple_sequences_match(self):
        stop_ids = [[88, 99], [200, 201]]
        assert _match_stop_sequence([1, 2, 3, 4, 5], stop_ids) is False

    def test_whitespace_only_sequence_matches_encoded_whitespace(self):
        """Whitespace tokens are valid stop sequences."""
        assert _match_stop_sequence([10, 32], [[32]]) is True


# ---------------------------------------------------------------------------
# Tests: _trim_stop_buf (rolling-window guard mirror)
# ---------------------------------------------------------------------------

class TestTrimStopBuf:

    def test_buffer_under_limit_returned_unchanged(self):
        buf = list(range(32))
        result = _trim_stop_buf(buf)
        assert result == buf

    def test_buffer_exactly_at_limit_returned_unchanged(self):
        buf = list(range(64))
        result = _trim_stop_buf(buf)
        assert result == buf
        assert len(result) == 64

    def test_buffer_one_over_limit_trimmed_to_last_64(self):
        buf = list(range(65))
        result = _trim_stop_buf(buf)
        assert result == list(range(1, 65))
        assert len(result) == 64

    def test_buffer_well_over_limit_keeps_last_64_entries(self):
        buf = list(range(200))
        result = _trim_stop_buf(buf)
        assert result == list(range(136, 200))
        assert len(result) == 64

    def test_empty_buffer_returned_unchanged(self):
        assert _trim_stop_buf([]) == []

    def test_custom_max_len_respected(self):
        buf = list(range(10))
        result = _trim_stop_buf(buf, max_len=4)
        assert result == [6, 7, 8, 9]

    def test_trim_preserves_exact_token_order(self):
        buf = [100, 200, 300, 400, 500]
        result = _trim_stop_buf(buf, max_len=3)
        assert result == [300, 400, 500]
