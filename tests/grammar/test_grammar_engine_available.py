"""Available-mode coverage for GrammarEngine via a numpy-backed fake xgrammar.

Complements test_grammar_engine_mask_precompute.py (which covers the precompute
loop) and test_grammar_engine_unavailable.py (the no-op contract). Here the
engine is constructed in *available* mode with a fake ``xgrammar`` whose
compiler/matcher are exercised, so the grammar-construction, logit-constraining,
FSM-advance and jump-forward branches all run. Pure numpy — host-agnostic.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from squish.grammar.grammar_engine import GrammarEngine

_VOCAB = 40


class _Tokenizer:
    vocab_size = _VOCAB

    def decode(self, ids: list[int]) -> str:
        return "x"

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]


class _Matcher:
    def __init__(self, compiled: object, fwd: str = "ab", raise_on: str | None = None) -> None:
        self.compiled = compiled
        self.accepted: list[int] = []
        self._fwd = fwd
        self._raise_on = raise_on

    def fill_next_token_bitmask(self, bitmask: np.ndarray, index: int) -> None:
        bitmask[:] = np.uint32(0xFFFFFFFF)

    def accept_token(self, token_id: int) -> None:
        if self._raise_on == "accept":
            raise RuntimeError("synthetic accept failure")
        self.accepted.append(token_id)

    def find_jump_forward_string(self) -> str:
        if self._raise_on == "jump":
            raise RuntimeError("synthetic jump failure")
        return self._fwd


class _Compiler:
    def compile_json_schema(self, schema: str) -> object:
        return ("schema", schema)

    def compile_builtin_json_grammar(self) -> object:
        return ("json",)

    def compile_regex(self, pattern: str) -> object:
        return ("regex", pattern)


def _make_xgr(*, allocate_raises: bool = False, matcher_kwargs: dict | None = None):
    mk = matcher_kwargs or {}

    class _Xgr:
        class TokenizerInfo:
            @staticmethod
            def from_huggingface(tok: object) -> object:
                return types.SimpleNamespace(vocab_size=tok.vocab_size)

        @staticmethod
        def GrammarCompiler(tok_info: object) -> object:
            return _Compiler()

        @staticmethod
        def GrammarMatcher(compiled: object) -> object:
            return _Matcher(compiled, **mk)

        @staticmethod
        def allocate_token_bitmask(n: int, vocab_size: int) -> np.ndarray:
            if allocate_raises:
                raise ValueError("synthetic allocate failure")
            return np.zeros((n, (vocab_size + 31) // 32), dtype=np.uint32)

        @staticmethod
        def apply_token_bitmask_inplace(logits_np: np.ndarray, bitmask: np.ndarray) -> None:
            return None

    return _Xgr()


def _engine(xgr: object) -> GrammarEngine:
    with patch.dict(sys.modules, {"xgrammar": xgr}):
        eng = GrammarEngine(_Tokenizer())
    assert eng._available
    return eng


def test_is_available_true_when_xgrammar_importable():
    with patch.dict(sys.modules, {"xgrammar": _make_xgr()}):
        assert GrammarEngine.is_available() is True


def test_precompute_outer_except_disables_mask():
    # allocate_token_bitmask raising sends _precompute_independent_mask into its
    # outer except, leaving the mask disabled (None) but the engine available.
    eng = _engine(_make_xgr(allocate_raises=True))
    assert eng._independent_mask is None


def test_json_object_and_regex_grammars_return_matchers():
    eng = _engine(_make_xgr())
    obj = eng.json_object_grammar()
    rgx = eng.regex_grammar("[0-9]+")
    assert obj.compiled == ("json",)
    assert rgx.compiled == ("regex", "[0-9]+")


def test_constrain_logits_applies_mask_and_returns_mx_array():
    eng = _engine(_make_xgr())
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = "f32"
    fake_mx.array = lambda a: ("MX", a)
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = fake_mx
    logits_mx = types.SimpleNamespace(astype=lambda dt: np.ones(_VOCAB, dtype=np.float32))
    with patch.dict(sys.modules, {"mlx": mlx_pkg, "mlx.core": fake_mx}):
        out = eng.constrain_logits(logits_mx, _Matcher(("json",)))
    assert out[0] == "MX"


def test_constrain_logits_swallows_masking_error():
    eng = _engine(_make_xgr())
    fake_mx = types.ModuleType("mlx.core")
    fake_mx.float32 = "f32"
    fake_mx.array = lambda a: ("MX", a)
    mlx_pkg = types.ModuleType("mlx")
    mlx_pkg.core = fake_mx
    logits_mx = types.SimpleNamespace(astype=lambda dt: np.ones(_VOCAB, dtype=np.float32))
    bad_state = object()  # lacks fill_next_token_bitmask → AttributeError, swallowed
    with patch.dict(sys.modules, {"mlx": mlx_pkg, "mlx.core": fake_mx}):
        out = eng.constrain_logits(logits_mx, bad_state)
    assert out is logits_mx


def test_advance_accepts_token_and_swallows_errors():
    eng = _engine(_make_xgr())
    ok_state = _Matcher(("json",))
    assert eng.advance(ok_state, 5) is ok_state
    assert ok_state.accepted == [5]
    # accept_token raising is swallowed; state is still returned.
    bad_state = _Matcher(("json",), raise_on="accept")
    assert eng.advance(bad_state, 9) is bad_state


def test_jump_forward_tokens_paths():
    eng = _engine(_make_xgr())
    # non-empty jump string → encoded ids
    assert eng.jump_forward_tokens(_Matcher(("json",), fwd="ab")) == [ord("a"), ord("b")]  # 97, 98
    # empty jump string → []
    assert eng.jump_forward_tokens(_Matcher(("json",), fwd="")) == []
    # raising find_jump_forward_string → [] (swallowed)
    assert eng.jump_forward_tokens(_Matcher(("json",), raise_on="jump")) == []


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
