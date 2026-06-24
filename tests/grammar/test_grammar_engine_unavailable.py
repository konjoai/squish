"""Graceful-degradation contract for GrammarEngine when xgrammar is absent.

These paths run in any environment without the optional ``xgrammar`` dependency
(Linux CI, the published HF Space). Every grammar method must no-op cleanly —
returning ``None`` / the input / an empty list — never raise — so structured
output silently falls back to unconstrained decoding.
"""

from squish.grammar.grammar_engine import GrammarEngine


def _unavailable_engine() -> GrammarEngine:
    eng = GrammarEngine(tokenizer=None)
    # xgrammar is not a core dependency; on a machine without it the engine is
    # constructed in no-op mode. Force it so the contract holds even if a future
    # CI image happens to install xgrammar.
    eng._available = False
    return eng


def test_is_available_reflects_xgrammar_import():
    # Returns a bool either way; on the (xgrammar-free) test image this is False.
    assert isinstance(GrammarEngine.is_available(), bool)


def test_grammar_constructors_return_none_when_unavailable():
    eng = _unavailable_engine()
    assert eng.json_object_grammar() is None
    assert eng.regex_grammar("[0-9]+") is None
    assert eng.json_schema_grammar({"type": "object"}) is None


def test_constrain_logits_passes_through_when_unavailable():
    eng = _unavailable_engine()
    sentinel = object()
    # Unavailable → returned unchanged regardless of state (None or not).
    assert eng.constrain_logits(sentinel, None) is sentinel
    assert eng.constrain_logits(sentinel, object()) is sentinel


def test_advance_returns_state_unchanged_when_unavailable():
    eng = _unavailable_engine()
    assert eng.advance(None, 7) is None
    state = object()
    assert eng.advance(state, 7) is state


def test_jump_forward_tokens_empty_when_unavailable():
    eng = _unavailable_engine()
    assert eng.jump_forward_tokens(None) == []
    assert eng.jump_forward_tokens(object()) == []
