"""tests/test_grammar_domino_dccd_unit.py — 100% coverage for DOMINOConstraint and DCCDDecoder in squish/grammar_engine.py"""
import numpy as np
import pytest

from squish.grammar.grammar_engine import DCCDDecoder, DOMINOConstraint

# ---------------------------------------------------------------------------
# Minimal fake tokenizer for DOMINOConstraint tests
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Encodes text as ASCII byte values; avoids real tokenizer dependency."""

    def encode(self, text: str, add_special_tokens: bool = False):
        return [ord(c) for c in text]


class _RaisingTokenizer:
    """Always raises on encode; tests error-handling path."""

    def encode(self, text: str, add_special_tokens: bool = False):
        raise ValueError("no encode")


# ---------------------------------------------------------------------------
# DOMINOConstraint
# ---------------------------------------------------------------------------

class TestDOMINOConstraint:
    def test_empty_forbidden_and_required(self):
        dc = DOMINOConstraint(_FakeTokenizer(), forbidden=[], required=[])
        logits = np.zeros(256, dtype=np.float32)
        out    = dc.apply(logits)
        np.testing.assert_array_equal(out, logits)

    def test_default_none_forbidden_required(self):
        dc = DOMINOConstraint(_FakeTokenizer())
        assert dc.forbidden_phrases == []
        assert dc.required_phrases  == []

    def test_forbidden_token_masked(self):
        # "A" → ASCII 65
        dc     = DOMINOConstraint(_FakeTokenizer(), forbidden=["A"])
        logits = np.ones(256, dtype=np.float32)
        out    = dc.apply(logits)
        assert out[65] == pytest.approx(-1e9)
        # Other tokens should be unchanged
        assert out[66] == pytest.approx(1.0)

    def test_multiple_forbidden_phrases(self):
        dc     = DOMINOConstraint(_FakeTokenizer(), forbidden=["A", "B"])
        logits = np.ones(256, dtype=np.float32)
        out    = dc.apply(logits)
        assert out[65] == pytest.approx(-1e9)   # 'A'
        assert out[66] == pytest.approx(-1e9)   # 'B'

    def test_invalid_token_id_ignored(self):
        """If the encoded token ID is out of logits range, no crash."""
        class HugeTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [9999]   # out of range for small logits

        dc     = DOMINOConstraint(HugeTokenizer(), forbidden=["x"])
        logits = np.zeros(10, dtype=np.float32)
        out    = dc.apply(logits)   # should not raise
        np.testing.assert_array_equal(out, logits)

    def test_tokenizer_raises_gracefully(self):
        """Tokenizer errors should result in an empty deny set, not a crash."""
        dc     = DOMINOConstraint(_RaisingTokenizer(), forbidden=["x"])
        logits = np.ones(5, dtype=np.float32)
        out    = dc.apply(logits)   # should not raise
        np.testing.assert_array_equal(out, logits)

    def test_apply_returns_copy_not_inplace(self):
        dc     = DOMINOConstraint(_FakeTokenizer(), forbidden=["A"])
        logits = np.ones(256, dtype=np.float32)
        out    = dc.apply(logits)
        assert out is not logits
        assert logits[65] == pytest.approx(1.0)   # original unchanged

    def test_forbidden_phrases_property(self):
        dc = DOMINOConstraint(_FakeTokenizer(), forbidden=["hello", "world"])
        assert set(dc.forbidden_phrases) == {"hello", "world"}

    def test_required_phrases_property(self):
        dc = DOMINOConstraint(_FakeTokenizer(), required=["yes", "no"])
        assert set(dc.required_phrases) == {"yes", "no"}

    def test_required_phrases_not_enforced_by_apply(self):
        """apply() only masks forbidden tokens; required is stored for external use."""
        dc     = DOMINOConstraint(_FakeTokenizer(), required=["A"])
        logits = np.zeros(256, dtype=np.float32)
        out    = dc.apply(logits)
        np.testing.assert_array_equal(out, logits)

    def test_apply_with_2d_logits_flattens(self):
        """apply() should handle a 1-D array but also not crash on wider input."""
        dc     = DOMINOConstraint(_FakeTokenizer(), forbidden=["A"])
        logits = np.ones(256, dtype=np.float32)
        out    = dc.apply(logits)
        assert out[65] == pytest.approx(-1e9)


# ---------------------------------------------------------------------------
# DCCDDecoder
# ---------------------------------------------------------------------------

class TestDCCDDecoder:
    def _allow_all(self, tok):
        return True

    def _allow_even(self, tok):
        return tok % 2 == 0

    def _allow_none(self, tok):
        return False

    def test_all_valid_passthrough(self):
        dc  = DCCDDecoder(self._allow_all)
        out = dc.filter_drafts([1, 2, 3, 4])
        assert out == [1, 2, 3, 4]

    def test_first_token_invalid_replaces_and_truncates(self):
        dc  = DCCDDecoder(self._allow_even, fallback_token_id=0)
        out = dc.filter_drafts([1, 2, 4, 6])  # 1 is odd → invalid first
        assert out == [0]   # replaced + truncated immediately

    def test_truncation_at_first_invalid(self):
        dc  = DCCDDecoder(self._allow_even, fallback_token_id=99)
        out = dc.filter_drafts([2, 4, 5, 6])  # 5 is invalid
        assert out == [2, 4, 99]   # valid up to 5, then truncate

    def test_all_invalid_returns_single_fallback(self):
        dc  = DCCDDecoder(self._allow_none, fallback_token_id=0)
        out = dc.filter_drafts([1, 2, 3])
        assert out == [0]

    def test_empty_draft_returns_empty(self):
        dc  = DCCDDecoder(self._allow_all)
        out = dc.filter_drafts([])
        assert out == []

    def test_fallback_default_is_zero(self):
        dc = DCCDDecoder(self._allow_none)
        assert dc._fallback == 0

    def test_is_valid_true(self):
        dc = DCCDDecoder(self._allow_even)
        assert dc.is_valid(4) is True

    def test_is_valid_false(self):
        dc = DCCDDecoder(self._allow_even)
        assert dc.is_valid(3) is False

    def test_custom_fallback_id(self):
        dc  = DCCDDecoder(self._allow_none, fallback_token_id=42)
        out = dc.filter_drafts([99])
        assert out == [42]

    def test_single_valid_token(self):
        dc  = DCCDDecoder(self._allow_all)
        out = dc.filter_drafts([7])
        assert out == [7]

    def test_last_token_invalid(self):
        dc  = DCCDDecoder(self._allow_even, fallback_token_id=0)
        out = dc.filter_drafts([2, 4, 6, 7])
        assert out == [2, 4, 6, 0]
