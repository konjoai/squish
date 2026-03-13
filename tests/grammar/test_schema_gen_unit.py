"""tests/test_schema_gen_unit.py

Full-coverage unit tests for squish/schema_gen.py.

Covers:
  SchemaState       — construction
  SchemaGenEngine   — __init__ (valid, vocab_size<1, custom tokens, dup-id
                      branch), reset, constrain (all paths including _ST_SS,
                      wrong logits shape, empty stack), advance (all FSM
                      state transitions + error paths), valid_next_chars,
                      _token_to_category, _close_container
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.grammar.schema_gen import SchemaGenEngine, SchemaState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _engine(vocab_size: int = 32, **kw) -> SchemaGenEngine:
    return SchemaGenEngine(vocab_size=vocab_size, **kw)


def _logits(engine: SchemaGenEngine) -> np.ndarray:
    return np.ones(engine._vocab_size, dtype=np.float32)


# Default token IDs (from _DEFAULT_SPECIAL):
# "{": 0, "}": 1, "[": 2, "]": 3, '"': 4, ",": 5, ":": 6
# "true": 7, "false": 8, "null": 9, "digit": 10
LBRACE = 0
RBRACE = 1
LBRACK = 2
RBRACK = 3
QUOTE  = 4
COMMA  = 5
COLON  = 6
TRUE   = 7
FALSE  = 8
NULL   = 9
DIGIT  = 10
STRING_CHAR = 20  # any ID not in the special map


# ---------------------------------------------------------------------------
# SchemaState
# ---------------------------------------------------------------------------


class TestSchemaState:
    def test_construction(self):
        s = SchemaState(
            stack=["S"],
            expected_tokens={"{"} ,
            is_complete=False,
        )
        assert s.stack == ["S"]
        assert s.is_complete is False


# ---------------------------------------------------------------------------
# SchemaGenEngine — __init__
# ---------------------------------------------------------------------------


class TestEngineInit:
    def test_valid_construction(self):
        eng = _engine()
        assert eng._vocab_size == 32

    def test_vocab_size_zero_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            _engine(vocab_size=0)

    def test_vocab_size_negative_raises(self):
        with pytest.raises(ValueError, match="vocab_size must be >= 1"):
            _engine(vocab_size=-5)

    def test_custom_special_tokens_override_defaults(self):
        eng = _engine(vocab_size=100, special_tokens={"{": 50})
        assert eng._tok["{"] == 50

    def test_out_of_range_special_tokens_excluded(self):
        """Token IDs >= vocab_size should be filtered from _tok."""
        eng = _engine(vocab_size=5, special_tokens={"{": 0, "}": 10})
        # "}": 10 is out of range for vocab_size=5
        assert "{" in eng._tok
        assert "}" not in eng._tok or eng._tok.get("}") != 10

    def test_duplicate_id_in_special_tokens_first_wins(self):
        """When two categories share the same token ID, first registered wins."""
        # Force both "{" and "}" to map to ID 0 by overriding "}" default
        eng = _engine(vocab_size=50, special_tokens={"}": 0})
        # Both "{": 0 and "}": 0 are in _tok (from merged dict)
        # _id_to_cat[0] should be "{" (first seen during iteration)
        # This exercises the `if tid not in self._id_to_cat:` False branch
        cat = eng._id_to_cat.get(0)
        assert cat in ("{", "}")


# ---------------------------------------------------------------------------
# SchemaGenEngine — reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_returns_start_state(self):
        eng = _engine()
        state = eng.reset()
        assert state.stack == ["S"]
        assert not state.is_complete

    def test_reset_expected_tokens_include_value_starters(self):
        eng = _engine()
        state = eng.reset()
        # In start state, any value-starting token is expected
        assert "{" in state.expected_tokens or "string_char" in state.expected_tokens


# ---------------------------------------------------------------------------
# SchemaGenEngine — constrain
# ---------------------------------------------------------------------------


class TestConstrain:
    def test_wrong_logits_shape_raises(self):
        eng = _engine(vocab_size=32)
        wrong = np.ones(10, dtype=np.float32)
        state = eng.reset()
        with pytest.raises(ValueError, match="logits must have shape"):
            eng.constrain(wrong, state)

    def test_empty_stack_raises(self):
        eng = _engine(vocab_size=32)
        empty_state = SchemaState(stack=[], expected_tokens=set(), is_complete=False)
        logits = _logits(eng)
        with pytest.raises(ValueError, match="empty stack"):
            eng.constrain(logits, empty_state)

    def test_constrain_start_state_masks_non_value_tokens(self):
        eng = _engine(vocab_size=32)
        state = eng.reset()
        logits = _logits(eng)
        out = eng.constrain(logits, state)
        # "{" (ID 0) should be valid → not -inf
        assert out[LBRACE] > -1e30
        # ":" (ID 6) is NOT a value starter → should be -inf
        assert out[COLON] == pytest.approx(-np.inf)

    def test_constrain_in_string_state(self):
        """In _ST_SS state, structural tokens except '"' should be masked."""
        eng = _engine(vocab_size=32)
        # Manually advance to string state: S → OK → SS
        state = eng.reset()
        state = eng.advance(LBRACE, state)   # S → OK
        state = eng.advance(QUOTE, state)    # OK → SS (starting key string)
        assert state.stack[-1] == "SS"
        logits = _logits(eng)
        out = eng.constrain(logits, state)
        # In SS: only QUOTE should be masking applied (structural tokens except QUOTE → -inf)
        # String characters (non-structural) remain valid (not in _tok)
        # Verify QUOTE is valid (its logit unchanged) and at least one structural token is -inf
        assert out[QUOTE] > -1e30  # QUOTE is valid in SS
        # COLON (structural, not quote) should be masked
        assert out[COLON] == pytest.approx(-np.inf)

    def test_constrain_outside_string_some_valid_tokens(self):
        """Outside a string: only valid structural tokens have non-inf logits."""
        eng = _engine(vocab_size=32)
        state = eng.reset()
        state = eng.advance(LBRACE, state)  # now in OK: expects '"' or '}'
        logits = _logits(eng)
        out = eng.constrain(logits, state)
        assert out[QUOTE] > -1e30   # '"' is valid in OK
        assert out[RBRACE] > -1e30  # '}' is valid in OK
        assert out[DIGIT] == pytest.approx(-np.inf)   # digit not valid in OK

    def test_constrain_restores_valid_token_logits(self):
        """Valid tokens' logit values should equal the input logits at those positions."""
        eng = _engine(vocab_size=32)
        state = eng.reset()
        state = eng.advance(LBRACE, state)  # in OK
        rng = np.random.default_rng(42)
        logits = rng.standard_normal(32).astype(np.float32)
        out = eng.constrain(logits, state)
        assert out[QUOTE] == pytest.approx(logits[QUOTE])
        assert out[RBRACE] == pytest.approx(logits[RBRACE])

    def test_constrain_done_state_all_invalid(self):
        """In done state, valid_cats is empty → all logits should be -inf."""
        eng = _engine(vocab_size=32)
        # Navigate to done: emit a scalar value
        state = eng.reset()
        state = eng.advance(DIGIT, state)   # S with digit → done
        assert state.is_complete
        assert state.stack == ["D"]
        logits = _logits(eng)
        out = eng.constrain(logits, state)
        # In _ST_D: valid_cats is empty → all -inf
        assert np.all(out == pytest.approx(-np.inf))


# ---------------------------------------------------------------------------
# SchemaGenEngine — advance: all FSM transitions
# ---------------------------------------------------------------------------


class TestAdvance:
    def test_empty_stack_raises(self):
        eng = _engine()
        empty = SchemaState(stack=[], expected_tokens=set(), is_complete=False)
        with pytest.raises(ValueError, match="empty stack"):
            eng.advance(LBRACE, empty)

    # ── Start state ──────────────────────────────────────────────────────────

    def test_start_with_lbrace_goes_to_ok(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(LBRACE, state)
        assert next_s.stack == ["OK"]

    def test_start_with_lbrack_goes_to_av(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(LBRACK, state)
        assert next_s.stack == ["AV"]

    def test_start_with_quote_goes_to_d_ss(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(QUOTE, state)
        assert "SS" in next_s.stack
        assert "D" in next_s.stack

    def test_start_with_digit_goes_to_done(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(DIGIT, state)
        assert next_s.stack == ["D"]
        assert next_s.is_complete

    def test_start_with_true_goes_to_done(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(TRUE, state)
        assert next_s.is_complete

    def test_start_with_false_goes_to_done(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(FALSE, state)
        assert next_s.is_complete

    def test_start_with_null_goes_to_done(self):
        eng = _engine()
        state = eng.reset()
        next_s = eng.advance(NULL, state)
        assert next_s.is_complete

    def test_start_with_invalid_token_raises(self):
        eng = _engine()
        state = eng.reset()
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(COLON, state)  # colon is not a value starter

    # ── OK state: object waiting for key ────────────────────────────────────

    def test_ok_with_quote_enters_key_string(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)
        assert state.stack == ["OK"]
        next_s = eng.advance(QUOTE, state)
        # Stack should have OC below and SS on top
        assert next_s.stack == ["OC", "SS"]

    def test_ok_with_rbrace_closes_empty_object(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)
        next_s = eng.advance(RBRACE, state)  # empty object
        assert next_s.is_complete

    def test_ok_with_invalid_raises(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(DIGIT, state)

    # ── OC state: object waiting for colon ──────────────────────────────────

    def test_oc_with_colon_goes_to_ov(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)   # OK
        state = eng.advance(QUOTE, state)    # OC, SS
        state = eng.advance(QUOTE, state)    # pop SS → OC
        assert state.stack == ["OC"]
        next_s = eng.advance(COLON, state)
        assert next_s.stack == ["OV"]

    def test_oc_with_invalid_raises(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)
        state = eng.advance(QUOTE, state)
        state = eng.advance(QUOTE, state)  # pop SS → OC
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(COMMA, state)

    # ── OV state: object waiting for value ──────────────────────────────────

    def test_ov_with_digit_value(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(DIGIT, state)
        assert "OS" in next_s.stack

    def test_ov_with_nested_object(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(LBRACE, state)
        assert "OS" in next_s.stack
        assert "OK" in next_s.stack

    def test_ov_with_nested_array(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(LBRACK, state)
        assert "OS" in next_s.stack
        assert "AV" in next_s.stack

    def test_ov_with_string_value(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(QUOTE, state)
        assert "OS" in next_s.stack
        assert "SS" in next_s.stack

    def test_ov_with_true_value(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(TRUE, state)
        assert "OS" in next_s.stack

    def test_ov_with_false_value(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(FALSE, state)
        assert "OS" in next_s.stack

    def test_ov_with_null_value(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        next_s = eng.advance(NULL, state)
        assert "OS" in next_s.stack

    def test_ov_with_invalid_raises(self):
        eng = _engine()
        state = _advance_to_ov(eng)
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(COLON, state)

    # ── OS state: after object value ────────────────────────────────────────

    def test_os_with_comma_goes_to_ok(self):
        eng = _engine()
        state = _advance_to_os(eng)
        next_s = eng.advance(COMMA, state)
        assert next_s.stack[-1] == "OK"

    def test_os_with_rbrace_closes_object(self):
        eng = _engine()
        state = _advance_to_os(eng)
        next_s = eng.advance(RBRACE, state)
        assert next_s.is_complete

    def test_os_with_invalid_raises(self):
        eng = _engine()
        state = _advance_to_os(eng)
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(LBRACE, state)

    # ── AV state: array waiting for value ───────────────────────────────────

    def test_av_with_rbrack_closes_empty_array(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(RBRACK, state)
        assert next_s.is_complete

    def test_av_with_digit_value(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(DIGIT, state)
        assert "AS" in next_s.stack

    def test_av_with_nested_object(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(LBRACE, state)
        assert "AS" in next_s.stack
        assert "OK" in next_s.stack

    def test_av_with_nested_array(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(LBRACK, state)
        assert "AS" in next_s.stack
        assert "AV" in next_s.stack

    def test_av_with_string_value(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(QUOTE, state)
        assert "AS" in next_s.stack
        assert "SS" in next_s.stack

    def test_av_with_true(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(TRUE, state)
        assert "AS" in next_s.stack

    def test_av_with_false(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(FALSE, state)
        assert "AS" in next_s.stack

    def test_av_with_null(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        next_s = eng.advance(NULL, state)
        assert "AS" in next_s.stack

    def test_av_with_invalid_raises(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(COLON, state)

    # ── AS state: after array value ─────────────────────────────────────────

    def test_as_with_comma_goes_to_av(self):
        eng = _engine()
        state = _advance_to_as(eng)
        next_s = eng.advance(COMMA, state)
        assert next_s.stack[-1] == "AV"

    def test_as_with_rbrack_closes_array(self):
        eng = _engine()
        state = _advance_to_as(eng)
        next_s = eng.advance(RBRACK, state)
        assert next_s.is_complete

    def test_as_with_invalid_raises(self):
        eng = _engine()
        state = _advance_to_as(eng)
        with pytest.raises(ValueError, match="Unexpected token"):
            eng.advance(RBRACE, state)

    # ── SS state: inside a string ────────────────────────────────────────────

    def test_ss_with_quote_closes_string(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)   # OK
        state = eng.advance(QUOTE, state)    # OC, SS
        next_s = eng.advance(QUOTE, state)   # pop SS → OC
        assert next_s.stack == ["OC"]

    def test_ss_with_string_char_stays_in_ss(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)
        state = eng.advance(QUOTE, state)  # → ["OC", "SS"]
        # Emit a string character (any unregistered ID)
        next_s = eng.advance(STRING_CHAR, state)
        assert "SS" in next_s.stack

    # ── D (done) state ────────────────────────────────────────────────────────

    def test_advance_from_done_raises(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(DIGIT, state)  # complete with scalar
        assert state.is_complete
        with pytest.raises(ValueError, match="DONE state"):
            eng.advance(DIGIT, state)

    # ── Unknown state ─────────────────────────────────────────────────────────

    def test_advance_from_unknown_state_raises(self):
        eng = _engine()
        bad_state = SchemaState(
            stack=["UNKNOWN_STATE"],
            expected_tokens=set(),
            is_complete=False,
        )
        with pytest.raises(ValueError, match="Unknown FSM state"):
            eng.advance(LBRACE, bad_state)


# ---------------------------------------------------------------------------
# SchemaGenEngine — valid_next_chars
# ---------------------------------------------------------------------------


class TestValidNextChars:
    def test_start_state_returns_value_cats(self):
        eng = _engine()
        state = eng.reset()
        cats = eng.valid_next_chars(state)
        assert isinstance(cats, list)
        assert "{" in cats

    def test_empty_stack_returns_empty(self):
        eng = _engine()
        empty = SchemaState(stack=[], expected_tokens=set(), is_complete=False)
        assert eng.valid_next_chars(empty) == []

    def test_done_state_returns_empty(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(DIGIT, state)
        cats = eng.valid_next_chars(state)
        assert cats == []


# ---------------------------------------------------------------------------
# SchemaGenEngine — _token_to_category
# ---------------------------------------------------------------------------


class TestTokenToCategory:
    def test_registered_token_returns_category(self):
        eng = _engine()
        cat = eng._token_to_category(LBRACE)
        assert cat == "{"

    def test_unregistered_token_returns_string_char(self):
        eng = _engine()
        cat = eng._token_to_category(STRING_CHAR)
        assert cat == "string_char"


# ---------------------------------------------------------------------------
# SchemaGenEngine — _close_container
# ---------------------------------------------------------------------------


class TestCloseContainer:
    def test_nested_container_returns_to_parent(self):
        """Closing a nested container pops to parent's after-value state."""
        eng = _engine()
        # {"key": []} produces: start → OK → OC/SS → OC → OV → AS (array opened)
        # Closing the array goes back to OS
        state = eng.reset()
        state = eng.advance(LBRACE, state)  # OK
        state = eng.advance(QUOTE, state)   # OC, SS
        state = eng.advance(QUOTE, state)   # OC
        state = eng.advance(COLON, state)   # OV
        state = eng.advance(LBRACK, state)  # OV top replaced by OS, AV pushed
        state = eng.advance(RBRACK, state)  # close array → back to OS
        assert state.stack[-1] == "OS"

    def test_top_level_container_goes_to_done(self):
        """Closing the top-level container pushes _ST_D."""
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACK, state)   # top-level array
        state = eng.advance(DIGIT, state)    # one element → AS
        state = eng.advance(RBRACK, state)   # close → DONE
        assert state.is_complete
        assert state.stack == ["D"]

    def test_empty_top_level_object_goes_to_done(self):
        eng = _engine()
        state = eng.reset()
        state = eng.advance(LBRACE, state)   # OK
        state = eng.advance(RBRACE, state)   # empty object → DONE
        assert state.is_complete


# ---------------------------------------------------------------------------
# Full JSON generation walk-through
# ---------------------------------------------------------------------------


class TestFullJsonGeneration:
    def test_simple_object(self):
        """Walk through: {"key": 42}"""
        eng = _engine(vocab_size=64)
        state = eng.reset()
        state = eng.advance(LBRACE, state)     # {
        state = eng.advance(QUOTE, state)      # "
        state = eng.advance(STRING_CHAR, state) # k
        state = eng.advance(STRING_CHAR, state) # e
        state = eng.advance(STRING_CHAR, state) # y
        state = eng.advance(QUOTE, state)      # " (close key)
        state = eng.advance(COLON, state)      # :
        state = eng.advance(DIGIT, state)      # 42
        state = eng.advance(RBRACE, state)     # }
        assert state.is_complete

    def test_array_of_numbers(self):
        """Walk through: [1, 2, 3]"""
        eng = _engine(vocab_size=64)
        state = eng.reset()
        state = eng.advance(LBRACK, state)     # [
        state = eng.advance(DIGIT, state)      # 1
        state = eng.advance(COMMA, state)      # ,
        state = eng.advance(DIGIT, state)      # 2
        state = eng.advance(COMMA, state)      # ,
        state = eng.advance(DIGIT, state)      # 3
        state = eng.advance(RBRACK, state)     # ]
        assert state.is_complete

    def test_nested_structure(self):
        """Walk through: {"a": {"b": true}}"""
        eng = _engine(vocab_size=64)
        state = eng.reset()
        state = eng.advance(LBRACE, state)   # {
        state = eng.advance(QUOTE, state)    # "
        state = eng.advance(QUOTE, state)    # "  (close key)
        state = eng.advance(COLON, state)    # :
        state = eng.advance(LBRACE, state)   # {  (nested)
        state = eng.advance(QUOTE, state)    # "
        state = eng.advance(QUOTE, state)    # "  (close key)
        state = eng.advance(COLON, state)    # :
        state = eng.advance(TRUE, state)     # true
        state = eng.advance(RBRACE, state)   # }  (close inner)
        state = eng.advance(RBRACE, state)   # }  (close outer)
        assert state.is_complete

    def test_object_multiple_keys(self):
        """Walk through: {"a": 1, "b": 2}"""
        eng = _engine(vocab_size=64)
        state = eng.reset()
        state = eng.advance(LBRACE, state)
        state = eng.advance(QUOTE, state)    # "
        state = eng.advance(QUOTE, state)    # close key
        state = eng.advance(COLON, state)
        state = eng.advance(DIGIT, state)    # 1
        state = eng.advance(COMMA, state)    # , — back to OK
        state = eng.advance(QUOTE, state)    # "
        state = eng.advance(QUOTE, state)    # close key
        state = eng.advance(COLON, state)
        state = eng.advance(DIGIT, state)    # 2
        state = eng.advance(RBRACE, state)   # }
        assert state.is_complete

    def test_constrain_tid_none_branch(self):
        """Valid cat not in _tok (small vocab) exercises the tid-is-None branch."""
        # vocab_size=3 means only tokens 0, 1, 2 are valid
        # Default: "{": 0, "}": 1, "[": 2. "]": 3 is out of range.
        eng = SchemaGenEngine(vocab_size=3)
        state = eng.reset()
        state = eng.advance(LBRACK, state)   # now in AV state
        logits = np.ones(3, dtype=np.float32)
        # In AV state, valid_cats includes "]" (RBRACK), but "]" id=3 >= vocab_size=3
        # So tid is None → exercises the `if tid is not None:` False branch
        out = eng.constrain(logits, state)
        # Should not raise; result is well-defined
        assert out.shape == (3,)


# ---------------------------------------------------------------------------
# Helper functions to navigate to specific FSM states
# ---------------------------------------------------------------------------


def _advance_to_ov(eng: SchemaGenEngine) -> SchemaState:
    """Advance FSM to OV state: inside {"key": ..."""
    state = eng.reset()
    state = eng.advance(LBRACE, state)   # OK
    state = eng.advance(QUOTE, state)    # OC, SS
    state = eng.advance(QUOTE, state)    # close key → OC
    state = eng.advance(COLON, state)    # OV
    return state


def _advance_to_os(eng: SchemaGenEngine) -> SchemaState:
    """Advance FSM to OS state: after first key-value pair in object."""
    state = _advance_to_ov(eng)
    state = eng.advance(DIGIT, state)    # OV with digit → OS
    return state


def _advance_to_as(eng: SchemaGenEngine) -> SchemaState:
    """Advance FSM to AS state: inside [1..."""
    state = eng.reset()
    state = eng.advance(LBRACK, state)   # AV
    state = eng.advance(DIGIT, state)    # AV with digit → AS
    return state
