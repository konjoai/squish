"""Tests for context.prompt_compressor — TF-IDF sentence compression + dispatch.

LLMLingua is an optional dependency; its success/failure branches are covered by
injecting a lightweight stub module (the integration path, not core inference).
"""
from __future__ import annotations

import sys
import types

import pytest

from squish.context import prompt_compressor as pc
from squish.context.prompt_compressor import _sentence_split, _tfidf_compress, compress


def _make_sentences(words_each=6, n=6):
    # Distinct vocabulary per sentence so TF-IDF scores differ and ranking bites.
    return " ".join(
        " ".join(f"w{i}t{j}" for j in range(words_each)) + "."
        for i in range(n)
    )


class TestSentenceSplit:
    def test_splits_on_terminal_punctuation(self):
        assert _sentence_split("One. Two! Three?") == ["One.", "Two!", "Three?"]

    def test_strips_empty_and_whitespace(self):
        assert _sentence_split("  A.   B.  ") == ["A.", "B."]

    def test_single_sentence(self):
        assert _sentence_split("no terminator here") == ["no terminator here"]


class TestTfidfCompress:
    def test_fewer_than_two_sentences_unchanged(self):
        t = "only one sentence here"
        assert _tfidf_compress(t, ratio=0.5) == t

    def test_high_ratio_keeps_all(self):
        t = "Alpha beta. Gamma delta. Epsilon zeta."
        assert _tfidf_compress(t, ratio=0.99) == t  # keep >= n → unchanged

    def test_compresses_and_preserves_order(self):
        t = _make_sentences(n=6)
        out = _tfidf_compress(t, ratio=0.5)
        kept = _sentence_split(out)
        assert len(kept) == 3                       # ceil(0.5*6)
        # order preserved: kept sentences appear in original order
        original = _sentence_split(t)
        assert kept == [s for s in original if s in kept]

    def test_no_alphanumeric_tokens_unchanged(self):
        t = "+++. ---. ***."   # 3 sentences, zero vocab → V==0 path
        assert _tfidf_compress(t, ratio=0.5) == t

    def test_preserve_prefix_protected(self):
        body = _make_sentences(n=6)
        prefix = "SYSTEM PROMPT HEADER"
        t = prefix + " " + body
        out = _tfidf_compress(t, ratio=0.4, preserve_tokens=3)
        assert out.startswith("SYSTEM PROMPT HEADER")
        assert len(out) < len(t)

    def test_preserve_tokens_covers_whole_text(self):
        t = "short text only"
        assert _tfidf_compress(t, ratio=0.5, preserve_tokens=10) == t

    def test_token_less_sentence_among_real_ones(self):
        # Middle sentence has no [a-z0-9] tokens → the empty-token `continue`.
        t = "alpha beta gamma. +++. delta epsilon zeta."
        out = _tfidf_compress(t, ratio=0.5)
        assert len(_sentence_split(out)) == 2

    def test_repeated_token_shared_across_sentences(self):
        # "shared" recurs → exercises the "already in vocab" skip branch.
        t = "shared alpha beta. shared gamma delta. shared epsilon zeta."
        out = _tfidf_compress(t, ratio=0.3)    # ceil(0.3*3)=1 → keep 1 of 3
        assert len(_sentence_split(out)) == 1


class TestCompressDispatch:
    @pytest.mark.parametrize("ratio", [0.0, 1.0, 1.5, -0.2])
    def test_noop_ratios(self, ratio):
        t = "Alpha beta. Gamma delta."
        assert compress(t, ratio=ratio) == t

    def test_empty_text(self):
        assert compress("", ratio=0.5) == ""

    def test_min_tokens_skips(self):
        t = "Alpha beta. Gamma delta."
        assert compress(t, ratio=0.5, min_tokens=100) == t

    def test_tfidf_fallback_when_llmlingua_absent(self, monkeypatch):
        # Ensure llmlingua import fails → TF-IDF path runs and actually compresses.
        monkeypatch.setitem(sys.modules, "llmlingua", None)  # forces ImportError
        t = _make_sentences(n=6)
        out = compress(t, ratio=0.5)
        assert len(_sentence_split(out)) == 3

    def test_tfidf_internal_error_returns_original(self, monkeypatch):
        # Defensive fallback: if _tfidf_compress raises, compress returns input.
        monkeypatch.setitem(sys.modules, "llmlingua", None)

        def _boom(*a, **k):
            raise ValueError("simulated")

        monkeypatch.setattr(pc, "_tfidf_compress", _boom)
        t = "Alpha beta. Gamma delta."
        assert compress(t, ratio=0.5) == t


def _install_fake_llmlingua(monkeypatch, *, result="COMPRESSED", raise_in_compress=False):
    mod = types.ModuleType("llmlingua")

    class PromptCompressor:
        def __init__(self, **kw):
            pass

        def compress_prompt(self, text, **kw):
            if raise_in_compress:
                raise ValueError("llmlingua boom")
            # Echo received kwargs so the test can assert on preserve handling.
            PromptCompressor.last_kwargs = kw
            return {"compressed_prompt": result}

    mod.PromptCompressor = PromptCompressor
    monkeypatch.setitem(sys.modules, "llmlingua", mod)
    return PromptCompressor


class TestLLMLinguaBranch:
    def test_llmlingua_success_path(self, monkeypatch):
        _install_fake_llmlingua(monkeypatch, result="LLM_OUT")
        assert compress("Alpha beta. Gamma delta.", ratio=0.5) == "LLM_OUT"

    def test_llmlingua_preserve_tokens_sets_context(self, monkeypatch):
        cls = _install_fake_llmlingua(monkeypatch, result="LLM_OUT")
        text = "a b c d e f g h"
        compress(text, ratio=0.5, preserve_tokens=3)
        assert cls.last_kwargs["context"] == "a b c"

    def test_llmlingua_failure_falls_back_to_tfidf(self, monkeypatch):
        _install_fake_llmlingua(monkeypatch, raise_in_compress=True)
        t = _make_sentences(n=6)
        out = compress(t, ratio=0.5)        # llmlingua raises → TF-IDF runs
        assert len(_sentence_split(out)) == 3

    def test_llmlingua_preserve_tokens_exceeds_word_count(self, monkeypatch):
        # preserve_tokens >= word count → the context-setting branch is skipped.
        cls = _install_fake_llmlingua(monkeypatch, result="LLM_OUT")
        cls.last_kwargs = {}
        assert compress("a b c", ratio=0.5, preserve_tokens=100) == "LLM_OUT"
        assert "context" not in cls.last_kwargs
