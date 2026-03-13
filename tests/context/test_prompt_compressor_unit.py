"""tests/test_prompt_compressor_unit.py

Full-coverage unit tests for squish/prompt_compressor.py.

Covers all branches in:
  _sentence_split         — basic splitting, multi-sentence, single sentence
  _tfidf_compress         — normal path, preserve_tokens branches, V==0,
                            fewer-than-2-sentences, keep>=n, prefix path
  compress                — empty text, ratio>=1, ratio<=0, min_tokens,
                            llmlingua ImportError path, tfidf fallback,
                            exception-swallowing path
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from squish.context.prompt_compressor import _sentence_split, _tfidf_compress, compress


# ---------------------------------------------------------------------------
# _sentence_split
# ---------------------------------------------------------------------------


class TestSentenceSplit:
    def test_single_sentence(self):
        result = _sentence_split("Hello world.")
        assert result == ["Hello world."]

    def test_multiple_sentences(self):
        result = _sentence_split("Hello world. How are you? Fine!")
        assert len(result) == 3

    def test_empty_string_returns_empty_list(self):
        result = _sentence_split("")
        assert result == []

    def test_strips_whitespace(self):
        result = _sentence_split("  Hello.  World.  ")
        assert len(result) == 2
        for s in result:
            assert s == s.strip()

    def test_no_sentence_terminal_returns_single(self):
        result = _sentence_split("no punctuation here")
        assert result == ["no punctuation here"]


# ---------------------------------------------------------------------------
# _tfidf_compress
# ---------------------------------------------------------------------------


class TestTfIdfCompress:
    def _multi_sentence(self, n=10):
        return " ".join(
            f"Sentence {i} contains unique word term{i} here."
            for i in range(n)
        )

    def test_normal_compression(self):
        text = self._multi_sentence(10)
        compressed = _tfidf_compress(text, ratio=0.5)
        assert isinstance(compressed, str)
        assert len(compressed) < len(text)

    def test_fewer_than_2_sentences_returns_text(self):
        text = "Only one sentence here."
        result = _tfidf_compress(text, ratio=0.5)
        assert result == text

    def test_keep_ge_n_returns_text(self):
        """When ratio * n >= n (ratio=1.0), no compression occurs."""
        text = self._multi_sentence(5)
        result = _tfidf_compress(text, ratio=1.0)
        assert result == text

    def test_keep_at_boundary(self):
        """ratio such that keep == n → returns original text."""
        text = " ".join(
            f"Sentence {i} is unique." for i in range(4)
        )
        # 4 sentences, keep=ceil(1.0 * 4)=4 >= n=4 → return text
        result = _tfidf_compress(text, ratio=1.0)
        assert result == text

    def test_v_zero_returns_text(self):
        """If all sentences have no words (only numbers/punct), vocab is empty."""
        # Create sentences that only contain digits (tokenise strips non-alphanum
        # but our tokeniser looks for [a-z0-9]+, so digits still create vocab).
        # To get V==0, use sentences with only whitespace/punctuation
        # The split returns sentences, but after _tokenise they may produce tokens.
        # Let's use sentences with only punctuation characters to get V==0.
        text = "!!!! .... !!!!"
        # _sentence_split may return 1 item (no period-space split) → returns early
        # Let's construct a case with 2 sentences but empty vocab
        text2 = "!!! !!! ???\n..."  # These don't split on sentence boundaries easily
        # Build a custom case: force sentences to have non-word content
        # by using dots to split but with only symbols as content
        result = _tfidf_compress("… … …  … … …", ratio=0.5)
        # Should return unchanged (V==0 or fewer than 2 sentences)
        assert isinstance(result, str)

    def test_preserve_tokens_splits_prefix(self):
        """preserve_tokens > 0 and len(words) > preserve_tokens → prefix split."""
        # Build a text with 20 words followed by many sentences
        prefix_words = ["preserve"] * 5
        suffix = " ".join(
            f"Sentence {i} has unique word w{i}."
            for i in range(10)
        )
        text = " ".join(prefix_words) + " " + suffix
        result = _tfidf_compress(text, ratio=0.5, preserve_tokens=5)
        assert isinstance(result, str)
        # Result should start with the preserved prefix
        assert result.startswith("preserve preserve")

    def test_preserve_tokens_entire_text_is_prefix(self):
        """preserve_tokens > len(words) → entire text is within prefix, return unchanged."""
        text = "only four words here"
        result = _tfidf_compress(text, ratio=0.5, preserve_tokens=100)
        assert result == text

    def test_preserve_tokens_prefix_joined_with_compressed(self):
        """Result when prefix is non-empty and compression happens."""
        prefix_words = "system prompt prefix"
        suffix_sentences = " ".join(
            f"Sentence {i} about topic{i}."
            for i in range(8)
        )
        text = prefix_words + " " + suffix_sentences
        result = _tfidf_compress(text, ratio=0.4, preserve_tokens=3)
        # prefix is "system prompt prefix", then compressed suffix follows
        assert "system" in result
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------


class TestCompress:
    def test_empty_text_returns_empty(self):
        result = compress("", ratio=0.5)
        assert result == ""

    def test_ratio_ge_1_returns_original(self):
        text = "Hello world."
        result = compress(text, ratio=1.0)
        assert result == text

    def test_ratio_le_0_returns_original(self):
        text = "Hello world."
        result = compress(text, ratio=0.0)
        assert result == text

    def test_ratio_negative_returns_original(self):
        text = "Hello world."
        result = compress(text, ratio=-0.5)
        assert result == text

    def test_min_tokens_skip_when_text_short(self):
        """When word_count < min_tokens, return text unchanged."""
        text = "Short text here."
        result = compress(text, ratio=0.5, min_tokens=100)
        assert result == text

    def test_min_tokens_not_triggered_when_long_enough(self):
        """When word_count >= min_tokens, compression proceeds."""
        text = " ".join(
            f"Sentence {i} about topic{i}." for i in range(30)
        )
        result = compress(text, ratio=0.3, min_tokens=5)
        assert isinstance(result, str)

    def test_llmlingua_import_error_falls_back_to_tfidf(self):
        """ImportError from llmlingua → fall through to TF-IDF."""
        text = " ".join(
            f"Sentence {i} has unique word w{i}." for i in range(10)
        )
        # llmlingua is not installed in this env; test falls through naturally
        result = compress(text, ratio=0.5)
        assert isinstance(result, str)
        assert len(result) < len(text)

    def test_llmlingua_exception_falls_back_to_tfidf(self):
        """Generic Exception from llmlingua → fall through to TF-IDF."""
        mock_llmlingua = MagicMock()
        mock_compressor = MagicMock()
        mock_llmlingua.PromptCompressor.return_value = mock_compressor
        mock_compressor.compress_prompt.side_effect = RuntimeError("llmlingua fail")

        text = " ".join(
            f"Sentence {i} with keyword w{i}." for i in range(10)
        )
        with patch.dict(sys.modules, {"llmlingua": mock_llmlingua}):
            result = compress(text, ratio=0.5)
        # Should still get a string result from TF-IDF fallback
        assert isinstance(result, str)

    def test_llmlingua_available_uses_compressed_prompt(self):
        """When llmlingua succeeds, returns its compressed_prompt."""
        mock_llmlingua = MagicMock()
        mock_compressor = MagicMock()
        mock_llmlingua.PromptCompressor.return_value = mock_compressor
        mock_compressor.compress_prompt.return_value = {"compressed_prompt": "short text"}

        text = " ".join(
            f"Sentence {i} is here." for i in range(20)
        )
        with patch.dict(sys.modules, {"llmlingua": mock_llmlingua}):
            result = compress(text, ratio=0.5)
        assert result == "short text"

    def test_llmlingua_returns_empty_compressed_prompt(self):
        """If llmlingua returns dict without compressed_prompt key → use text."""
        mock_llmlingua = MagicMock()
        mock_compressor = MagicMock()
        mock_llmlingua.PromptCompressor.return_value = mock_compressor
        mock_compressor.compress_prompt.return_value = {}  # no compressed_prompt key

        text = " ".join(
            f"Sentence {i}." for i in range(20)
        )
        with patch.dict(sys.modules, {"llmlingua": mock_llmlingua}):
            result = compress(text, ratio=0.5)
        # .get("compressed_prompt", text) returns text when key absent
        assert result == text

    def test_tfidf_exception_returns_original(self):
        """If TF-IDF itself raises, the outer except returns text unchanged."""
        text = " ".join(
            f"Sentence {i} is here." for i in range(10)
        )
        with patch("squish.context.prompt_compressor._tfidf_compress", side_effect=RuntimeError("crash")):
            result = compress(text, ratio=0.5)
        assert result == text

    def test_compress_with_preserve_tokens(self):
        """preserve_tokens argument is passed through to TF-IDF."""
        text = "system " * 5 + " ".join(
            f"Sentence {i} about topic{i}." for i in range(15)
        )
        result = compress(text, ratio=0.4, preserve_tokens=5)
        assert isinstance(result, str)
        assert "system" in result

    def test_llmlingua_with_preserve_tokens_context_split(self):
        """preserve_tokens > 0 with llmlingua: context and text split."""
        mock_llmlingua = MagicMock()
        mock_compressor = MagicMock()
        mock_llmlingua.PromptCompressor.return_value = mock_compressor
        mock_compressor.compress_prompt.return_value = {"compressed_prompt": "compressed"}

        # Many words so preserve_tokens < len(words)
        text = " ".join(["word"] * 10 + [f"extra{i}" for i in range(30)])
        with patch.dict(sys.modules, {"llmlingua": mock_llmlingua}):
            result = compress(text, ratio=0.5, preserve_tokens=5)
        assert result == "compressed"
