"""tests/test_grammar_engine_mask_precompute.py

Behavioral coverage for GrammarEngine's context-independent bitmask precompute
loop and the combined-mask AND, exercised against a *numpy-backed* fake
xgrammar so the real per-token loop body runs.

The MagicMock-based fakes used elsewhere give ``_tok_info.vocab_size`` a Mock,
so ``range(vocab_size)`` raises and ``_precompute_independent_mask`` falls
straight into its outer ``except`` (mask left ``None``). That never exercises
the inner loop. Here ``allocate_token_bitmask`` returns a genuine
``np.uint32`` bitmask and the tokenizer yields a forbidden token, a normal
token, and a decode-failing token — so every branch of the loop executes and is
asserted on the resulting mask bits.

No xgrammar or MLX is required; the fakes are pure numpy, so the test is
host-agnostic (identical on macOS and Linux, CI or sandbox).
"""
from __future__ import annotations

import sys
import types
from unittest.mock import patch

import numpy as np
import pytest

from squish.grammar.grammar_engine import GrammarEngine

# Vocabulary kept tiny but spanning two 32-bit bitmask words (ceil(40/32) == 2).
_VOCAB = 40
_FORBIDDEN_TOKEN = 5   # decodes to a control char → masked out
_DECODE_FAIL_TOKEN = 7  # decode() raises → stays valid (exercises the except)


class _MaskTokenizer:
    """Tokenizer whose decode() drives each branch of the precompute loop."""

    vocab_size = _VOCAB

    def decode(self, ids: list[int]) -> str:
        tid = ids[0]
        if tid == _FORBIDDEN_TOKEN:
            return "\x00"  # NUL — entirely control chars → forbidden
        if tid == _DECODE_FAIL_TOKEN:
            raise ValueError("synthetic decode failure")
        return "x"  # ordinary printable → not forbidden

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) % self.vocab_size for c in text]


class _NumpyXgr:
    """Minimal xgrammar stand-in with a real numpy uint32 bitmask."""

    def __init__(self) -> None:
        self.applied: list[np.ndarray] = []

    class TokenizerInfo:
        @staticmethod
        def from_huggingface(tok: object) -> object:
            return types.SimpleNamespace(vocab_size=tok.vocab_size)

    @staticmethod
    def GrammarCompiler(tok_info: object) -> object:
        return object()

    def allocate_token_bitmask(self, n: int, vocab_size: int) -> np.ndarray:
        words = (vocab_size + 31) // 32
        return np.zeros((n, words), dtype=np.uint32)

    def apply_token_bitmask_inplace(self, logits_np: np.ndarray, bitmask: np.ndarray) -> None:
        self.applied.append(bitmask.copy())


class _AllAllowedState:
    """GrammarMatcher stand-in that marks every token valid (all bits set)."""

    def fill_next_token_bitmask(self, bitmask: np.ndarray, index: int) -> None:
        bitmask[:] = np.uint32(0xFFFFFFFF)


def _bit(mask: np.ndarray, token_id: int) -> int:
    """Return the 0/1 validity bit for *token_id* in the 2-D uint32 bitmask."""
    return int((mask[0, token_id >> 5] >> np.uint32(token_id & 31)) & np.uint32(1))


def _available_engine() -> tuple[GrammarEngine, _NumpyXgr]:
    xgr = _NumpyXgr()
    with patch.dict(sys.modules, {"xgrammar": xgr}):
        engine = GrammarEngine(_MaskTokenizer())
    assert engine._available, "engine should be in available mode with the numpy fake"
    return engine, xgr


class TestIndependentMaskPrecompute:
    def test_precompute_loop_sets_mask_bits_per_branch(self) -> None:
        engine, _ = _available_engine()
        mask = engine._independent_mask
        assert mask is not None
        # Forbidden token (decodes to a control char) → its bit is cleared.
        assert _bit(mask, _FORBIDDEN_TOKEN) == 0
        # Ordinary printable token → left valid (the `if` condition is False).
        assert _bit(mask, 0) == 1
        # Decode raised → the except branch keeps the token valid (not cleared).
        assert _bit(mask, _DECODE_FAIL_TOKEN) == 1

    def test_only_the_forbidden_token_is_cleared(self) -> None:
        engine, _ = _available_engine()
        mask = engine._independent_mask
        cleared = [t for t in range(_VOCAB) if _bit(mask, t) == 0]
        assert cleared == [_FORBIDDEN_TOKEN]


class TestApplyCombinedMask:
    def test_combined_mask_ands_independent_mask(self) -> None:
        engine, xgr = _available_engine()
        logits = np.ones(_VOCAB, dtype=np.float32)
        engine._apply_combined_mask(logits, _AllAllowedState())
        # State allowed everything; ANDing with the independent mask must yield
        # exactly the independent mask (forbidden token still cleared).
        assert len(xgr.applied) == 1
        np.testing.assert_array_equal(xgr.applied[-1], engine._independent_mask)

    def test_combined_mask_skips_and_when_independent_mask_is_none(self) -> None:
        engine, xgr = _available_engine()
        engine._independent_mask = None  # force the "no independent mask" branch
        logits = np.ones(_VOCAB, dtype=np.float32)
        engine._apply_combined_mask(logits, _AllAllowedState())
        # No AND applied → the bitmask stays the state's all-allowed mask.
        assert len(xgr.applied) == 1
        assert int(xgr.applied[-1][0, 0]) == 0xFFFFFFFF


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
