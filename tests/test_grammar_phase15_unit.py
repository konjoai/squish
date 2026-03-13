#!/usr/bin/env python3
"""
tests/test_grammar_phase15_unit.py

Unit tests for Phase 15D (schema cache) and Phase 15E (TagDispatch)
additions to squish/grammar_engine.py.

Coverage targets
────────────────
GrammarEngine._schema_cache (Phase 15D)
  - json_schema_grammar fallback (unavailable) returns None
  - first call compiles and caches
  - second call with same schema hits cache (no recompile)
  - different schemas get different cache entries
  - LRU eviction: adding > _SCHEMA_CACHE_MAXSIZE schemas drops oldest
  - GrammarMatcher is fresh per call (independent cursors)

TagDispatch (Phase 15E)
  - empty trigger_ids → never activates
  - single-token trigger detected on first match
  - multi-token trigger: partial match then full match
  - observe() before activation: buffering only
  - observe() after activation: delegates to engine.advance()
  - constrain_logits() before activation: pass-through
  - constrain_logits() after activation: delegates to engine
  - activated property reflects state
  - is_terminated() False before activation
  - is_terminated() proxies to state.is_terminated() after activation
  - is_terminated() AttributeError on state → False
  - tag_dispatch_for_schema factory: trigger tokenised correctly
  - tag_dispatch_for_schema factory: tokenise exception → empty trigger
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from squish.grammar_engine import (
    GrammarEngine,
    TagDispatch,
    _SCHEMA_CACHE_MAXSIZE,
)


# ---------------------------------------------------------------------------
# Helpers: build a mock xgrammar + GrammarEngine in "available" mode
# ---------------------------------------------------------------------------

def _make_mock_xgr():
    """Return a minimal mock xgrammar module."""
    xgr = MagicMock()
    xgr.TokenizerInfo.from_huggingface.return_value = MagicMock()
    xgr.GrammarCompiler.return_value = MagicMock()
    xgr.GrammarMatcher.return_value = MagicMock()
    return xgr


def _available_engine(xgr=None, tokenizer=None):
    """Return a GrammarEngine with _available=True via mock xgrammar."""
    if xgr is None:
        xgr = _make_mock_xgr()
    if tokenizer is None:
        tokenizer = MagicMock()
        tokenizer.encode.return_value = [10, 20, 30]
    with patch.dict(sys.modules, {"xgrammar": xgr}):
        engine = GrammarEngine(tokenizer)
    assert engine._available
    return engine, xgr, tokenizer


def _fallback_engine():
    """Return a GrammarEngine with _available=False (no xgrammar)."""
    with patch.dict(sys.modules, {"xgrammar": None}):
        engine = GrammarEngine(MagicMock())
    assert not engine._available
    return engine


# ---------------------------------------------------------------------------
# Phase 15D — schema cache
# ---------------------------------------------------------------------------

class TestSchemaCache:
    def test_cache_attr_exists(self):
        engine = _fallback_engine()
        assert hasattr(engine, "_schema_cache")

    def test_fallback_returns_none(self):
        engine = _fallback_engine()
        result = engine.json_schema_grammar({"type": "object"})
        assert result is None

    def test_available_first_call_compiles(self):
        engine, xgr, _ = _available_engine()
        schema = {"type": "object"}
        engine.json_schema_grammar(schema)
        # compiler was called once
        engine._compiler.compile_json_schema.assert_called_once()

    def test_same_schema_second_call_hits_cache(self):
        engine, xgr, _ = _available_engine()
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
        engine.json_schema_grammar(schema)
        engine.json_schema_grammar(schema)
        # compile called only once despite two calls
        assert engine._compiler.compile_json_schema.call_count == 1

    def test_key_ordering_irrelevant_to_cache_hit(self):
        engine, xgr, _ = _available_engine()
        schema_a = {"b": 2, "a": 1}
        schema_b = {"a": 1, "b": 2}          # same content, different key order
        engine.json_schema_grammar(schema_a)
        engine.json_schema_grammar(schema_b)
        assert engine._compiler.compile_json_schema.call_count == 1

    def test_different_schemas_get_separate_entries(self):
        engine, xgr, _ = _available_engine()
        engine.json_schema_grammar({"type": "string"})
        engine.json_schema_grammar({"type": "integer"})
        assert engine._compiler.compile_json_schema.call_count == 2

    def test_lru_eviction_at_maxsize(self):
        engine, xgr, _ = _available_engine()
        schemas = [{"type": "string", "minLength": i} for i in range(_SCHEMA_CACHE_MAXSIZE + 2)]
        for s in schemas:
            engine.json_schema_grammar(s)
        # Cache should not exceed maxsize
        assert len(engine._schema_cache) <= _SCHEMA_CACHE_MAXSIZE

    def test_each_call_returns_fresh_matcher(self):
        """Even with cache hit, GrammarMatcher is instantiated fresh each time."""
        engine, xgr, _ = _available_engine()
        schema = {"type": "object"}
        m1 = engine.json_schema_grammar(schema)
        m2 = engine.json_schema_grammar(schema)
        # GrammarMatcher constructor called twice
        assert xgr.GrammarMatcher.call_count == 2


# ---------------------------------------------------------------------------
# Phase 15E — TagDispatch
# ---------------------------------------------------------------------------

class TestTagDispatchEmptyTrigger:
    def test_never_activates_with_empty_trigger(self):
        engine = _fallback_engine()
        grammar_fn = MagicMock(return_value=object())
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[])
        for tok in [1, 2, 3, 4, 5]:
            td.observe(tok)
        assert not td.activated
        grammar_fn.assert_not_called()

    def test_constrain_logits_passthrough_with_empty_trigger(self):
        engine = _fallback_engine()
        td = TagDispatch(engine=engine, grammar_fn=MagicMock(), trigger_ids=[])
        logits = np.ones(10, dtype=np.float32)
        out = td.constrain_logits(logits)
        assert out is logits  # exact same object

    def test_is_terminated_false_with_empty_trigger(self):
        engine = _fallback_engine()
        td = TagDispatch(engine=engine, grammar_fn=MagicMock(), trigger_ids=[])
        assert td.is_terminated() is False


class TestTagDispatchSingleToken:
    def test_single_token_trigger_activates(self):
        engine, xgr, tok = _available_engine()
        mock_state = MagicMock()
        grammar_fn = MagicMock(return_value=mock_state)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[42])
        td.observe(42)
        assert td.activated
        grammar_fn.assert_called_once()

    def test_non_matching_token_does_not_activate(self):
        engine = _fallback_engine()
        grammar_fn = MagicMock()
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[42])
        td.observe(99)
        assert not td.activated
        grammar_fn.assert_not_called()

    def test_activation_sets_state(self):
        engine, xgr, _ = _available_engine()
        mock_state = MagicMock()
        grammar_fn = MagicMock(return_value=mock_state)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[7])
        td.observe(7)
        assert td._state is mock_state


class TestTagDispatchMultiToken:
    def test_partial_sequence_does_not_activate(self):
        engine = _fallback_engine()
        grammar_fn = MagicMock()
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[1, 2, 3])
        td.observe(1)
        td.observe(2)
        assert not td.activated

    def test_full_sequence_activates(self):
        engine = _fallback_engine()
        grammar_fn = MagicMock(return_value=MagicMock())
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[1, 2, 3])
        td.observe(1)
        td.observe(2)
        td.observe(3)
        assert td.activated

    def test_rolling_window_allows_late_match(self):
        """Noise tokens before trigger should still allow activation."""
        engine = _fallback_engine()
        grammar_fn = MagicMock(return_value=MagicMock())
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[5, 6])
        td.observe(1)
        td.observe(2)
        td.observe(5)
        td.observe(6)
        assert td.activated

    def test_window_does_not_miss_trigger_after_noise(self):
        """Window correctly discards old tokens."""
        engine = _fallback_engine()
        grammar_fn = MagicMock(return_value=MagicMock())
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[9, 10])
        for i in range(5):
            td.observe(i)
        td.observe(9)
        assert not td.activated  # only first token of trigger seen
        td.observe(10)
        assert td.activated


class TestTagDispatchPostActivation:
    def test_observe_after_activation_calls_engine_advance(self):
        engine, xgr, _ = _available_engine()
        mock_state = MagicMock()
        new_state   = MagicMock()
        engine.advance = MagicMock(return_value=new_state)
        grammar_fn = MagicMock(return_value=mock_state)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[77])
        td.observe(77)     # trigger activation
        td.observe(88)     # post-activation token
        engine.advance.assert_called_once_with(mock_state, 88)
        assert td._state is new_state

    def test_constrain_logits_delegates_after_activation(self):
        engine, xgr, _ = _available_engine()
        mock_state      = MagicMock()
        constrained_l   = np.zeros(5, dtype=np.float32)
        engine.constrain_logits = MagicMock(return_value=constrained_l)
        grammar_fn = MagicMock(return_value=mock_state)

        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[3])
        td.observe(3)

        logits = np.ones(5, dtype=np.float32)
        out = td.constrain_logits(logits)
        engine.constrain_logits.assert_called_once_with(logits, mock_state)
        assert out is constrained_l

    def test_constrain_logits_passthrough_before_activation(self):
        engine, xgr, _ = _available_engine()
        engine.constrain_logits = MagicMock()
        td = TagDispatch(engine=engine, grammar_fn=MagicMock(), trigger_ids=[99])
        logits = np.ones(5, dtype=np.float32)
        out = td.constrain_logits(logits)
        engine.constrain_logits.assert_not_called()
        assert out is logits


class TestTagDispatchIsTerminated:
    def test_is_terminated_false_before_activation(self):
        engine = _fallback_engine()
        td = TagDispatch(engine=engine, grammar_fn=MagicMock(), trigger_ids=[1])
        assert td.is_terminated() is False

    def test_is_terminated_proxies_state(self):
        engine = _fallback_engine()
        mock_state = MagicMock()
        mock_state.is_terminated.return_value = True
        grammar_fn = MagicMock(return_value=mock_state)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[50])
        td.observe(50)
        assert td.is_terminated() is True

    def test_is_terminated_false_when_state_is_terminated_false(self):
        engine = _fallback_engine()
        mock_state = MagicMock()
        mock_state.is_terminated.return_value = False
        grammar_fn = MagicMock(return_value=mock_state)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[50])
        td.observe(50)
        assert td.is_terminated() is False

    def test_is_terminated_attribute_error_returns_false(self):
        """Older xgrammar without is_terminated() → returns False safely."""
        engine = _fallback_engine()
        mock_state = MagicMock(spec=[])        # no attributes
        grammar_fn = MagicMock(return_value=mock_state)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[50])
        td.observe(50)
        assert td.is_terminated() is False

    def test_is_terminated_none_state_returns_false(self):
        engine = _fallback_engine()
        grammar_fn = MagicMock(return_value=None)
        td = TagDispatch(engine=engine, grammar_fn=grammar_fn, trigger_ids=[50])
        td.observe(50)
        assert td.is_terminated() is False


class TestTagDispatchFactory:
    def test_factory_tokenises_trigger(self):
        engine, xgr, tokenizer = _available_engine()
        tokenizer.encode.return_value = [10, 20]
        schema = {"type": "object"}
        td = engine.tag_dispatch_for_schema("<tool_call>", schema)
        assert isinstance(td, TagDispatch)
        assert td._trigger == [10, 20]

    def test_factory_exception_gives_empty_trigger(self):
        engine, xgr, tokenizer = _available_engine()
        tokenizer.encode.side_effect = RuntimeError("oops")
        td = engine.tag_dispatch_for_schema("<tool_call>", {"type": "object"})
        assert td._trigger == []
        assert not td.activated

    def test_factory_activates_on_correct_trigger(self):
        engine, xgr, tokenizer = _available_engine()
        tokenizer.encode.return_value = [5]
        # Ensure grammar_fn returns a usable state
        compiled = MagicMock()
        engine._compiler.compile_json_schema.return_value = compiled
        xgr.GrammarMatcher.return_value = MagicMock()

        td = engine.tag_dispatch_for_schema("X", {"type": "object"})
        td.observe(5)
        assert td.activated
