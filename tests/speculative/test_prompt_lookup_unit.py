"""
tests/test_prompt_lookup_unit.py

Unit tests for squish/prompt_lookup.py — 100% coverage.
"""

import numpy as np
import pytest

from squish.speculative.prompt_lookup import (
    NGramIndex,
    PromptLookupConfig,
    PromptLookupDecoder,
    PromptLookupStats,
)

# ---------------------------------------------------------------------------
# PromptLookupConfig
# ---------------------------------------------------------------------------

class TestPromptLookupConfig:
    def test_defaults(self):
        cfg = PromptLookupConfig()
        assert cfg.ngram_min >= 1
        assert cfg.ngram_max >= cfg.ngram_min
        assert cfg.max_speculative >= 1

    def test_custom(self):
        cfg = PromptLookupConfig(ngram_min=3, ngram_max=7, max_speculative=8)
        assert cfg.ngram_min == 3
        assert cfg.ngram_max == 7
        assert cfg.max_speculative == 8

    @pytest.mark.parametrize("kwargs, exc", [
        ({"ngram_min": 0},                            "ngram_min"),
        ({"ngram_min": 3, "ngram_max": 2},            "ngram_max"),
        ({"max_speculative": 0},                      "max_speculative"),
    ])
    def test_validation(self, kwargs, exc):
        with pytest.raises(ValueError, match=exc):
            PromptLookupConfig(**kwargs)


# ---------------------------------------------------------------------------
# NGramIndex
# ---------------------------------------------------------------------------

class TestNGramIndex:
    def test_build_and_find(self):
        idx = NGramIndex(ngram_min=2, ngram_max=3)
        tokens = [1, 2, 3, 4, 1, 2, 5, 6]
        idx.build(tokens)
        # query [1, 2] should find continuation starting at position 5 → [5, 6]
        results = idx.find([1, 2])
        assert any(r[:2] == [5, 6] for r in results)

    def test_push_incremental(self):
        idx = NGramIndex(ngram_min=2, ngram_max=2)
        for t in [10, 20, 30, 10, 20, 99]:
            idx.push(t)
        results = idx.find([10, 20])
        assert results  # should find continuation [99]

    def test_no_match(self):
        idx = NGramIndex(ngram_min=2, ngram_max=3)
        idx.build([1, 2, 3])
        results = idx.find([9, 9])
        assert results == []

    def test_empty_query_no_crash(self):
        idx = NGramIndex(ngram_min=2, ngram_max=3)
        idx.build([1, 2, 3])
        results = idx.find([])
        assert results == []

    def test_single_token_query_too_short(self):
        idx = NGramIndex(ngram_min=2, ngram_max=3)
        idx.build([1, 2, 3])
        results = idx.find([1])
        assert results == []

    def test_invalid_ngram_min(self):
        with pytest.raises(ValueError):
            NGramIndex(ngram_min=0)

    def test_invalid_ngram_max(self):
        with pytest.raises(ValueError):
            NGramIndex(ngram_min=3, ngram_max=2)

    def test_continuation_capped_by_max_cont(self):
        idx = NGramIndex(ngram_min=2, ngram_max=2, max_continuations=2)
        idx.build([10, 20, 1, 2, 3, 4, 5])
        results = idx.find([10, 20])
        for r in results:
            assert len(r) <= 2  # max_continuations cap

    def test_long_sequence_with_multiple_matches(self):
        # [A, B] appears multiple times with different continuations
        tokens = [1, 2, 10, 1, 2, 20, 1, 2, 30]
        idx = NGramIndex(ngram_min=2, ngram_max=2)
        idx.build(tokens)
        results = idx.find([1, 2])
        # Should have found multiple continuations
        assert len(results) >= 1

    def test_rebuild_clears_old_data(self):
        idx = NGramIndex(ngram_min=2, ngram_max=2)
        idx.build([1, 2, 99])
        idx.build([5, 6, 77])   # complete rebuild
        results = idx.find([1, 2])
        assert results == []  # old data gone
        results2 = idx.find([5, 6])
        assert results2  # new data present

    def test_push_at_end_of_sequence_no_continuation(self):
        idx = NGramIndex(ngram_min=2, ngram_max=2)
        idx.build([1, 2])
        # [1,2] n-gram: pos=0 to 1, after_pos=2 → beyond end
        results = idx.find([1, 2])
        assert results == []  # no continuation past end


# ---------------------------------------------------------------------------
# PromptLookupStats
# ---------------------------------------------------------------------------

class TestPromptLookupStats:
    def test_acceptance_rate_zero(self):
        s = PromptLookupStats()
        assert s.acceptance_rate == 0.0

    def test_acceptance_rate(self):
        s = PromptLookupStats(total_draft_tokens=10, total_accepted=7)
        assert s.acceptance_rate == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# PromptLookupDecoder
# ---------------------------------------------------------------------------

class TestPromptLookupDecoder:
    def _greedy_forward(self, next_toks: list[int]):
        """Return a forward fn that cycles through next_toks."""
        state = {"i": 0}

        def fwd(ids):
            tok = next_toks[state["i"] % len(next_toks)]
            state["i"] += 1
            logits = np.full(100, -10.0)
            logits[tok] = 10.0
            return logits

        return fwd

    def test_generate_basic_length(self):
        cfg = PromptLookupConfig(ngram_min=2, ngram_max=3, max_speculative=3)
        fwd = self._greedy_forward([5])
        dec = PromptLookupDecoder(fwd, cfg)
        ids, stats = dec.generate([1, 2, 3], max_new_tokens=5)
        assert len(ids) == 8   # 3 prompt + 5 new
        assert (stats.speculative_steps + stats.fallback_steps) >= 1  # sanity

    def test_ngram_speculative_path(self):
        """Prompt contains [1, 2, 10] then [1, 2] again → speculative match."""
        cfg = PromptLookupConfig(ngram_min=2, ngram_max=2, max_speculative=3)
        # verifier always picks token matching the draft (token 10)
        def fwd(ids):
            # Return logits weighted toward token 10
            logits = np.full(50, -10.0)
            logits[10] = 10.0
            return logits

        # Prompt ends in [1, 2], and earlier in the prompt we have [1,2,10]
        prompt = [1, 2, 10, 5, 6, 1, 2]
        dec = PromptLookupDecoder(fwd, cfg)
        ids, stats = dec.generate(prompt, max_new_tokens=3)
        # Should have used speculative path at least once
        assert stats.speculative_steps >= 1
        assert stats.total_accepted >= 1

    def test_fallback_when_no_match(self):
        cfg = PromptLookupConfig(ngram_min=3, ngram_max=5)
        fwd = self._greedy_forward([7])
        dec = PromptLookupDecoder(fwd, cfg)
        # Prompt too short for any 3-gram match
        ids, stats = dec.generate([1, 2], max_new_tokens=3)
        assert stats.fallback_steps >= 1
        assert all(t == 7 for t in ids[2:])

    def test_draft_mismatch_inserts_verifier_token(self):
        """Draft proposes token X from n-gram; verifier corrects to Y."""
        # Prompt: [1, 2, 99, 1, 2] → n-gram [1,2] has continuation [99]
        # Verifier always picks 42
        cfg = PromptLookupConfig(ngram_min=2, ngram_max=2, max_speculative=1)

        def fwd(ids):
            logits = np.full(200, -10.0)
            logits[42] = 10.0   # verifier always returns 42
            return logits

        prompt = [1, 2, 99, 1, 2]
        dec = PromptLookupDecoder(fwd, cfg)
        ids, stats = dec.generate(prompt, max_new_tokens=4)
        assert 42 in ids  # verifier's correction appears

    def test_empty_prompt(self):
        cfg = PromptLookupConfig()
        fwd = self._greedy_forward([0])
        dec = PromptLookupDecoder(fwd, cfg)
        ids, _ = dec.generate([], max_new_tokens=3)
        assert len(ids) == 3

    def test_max_speculative_cap(self):
        """max_speculative=1 means at most 1 draft token per speculative step."""
        cfg = PromptLookupConfig(ngram_min=2, ngram_max=2, max_speculative=1)

        def fwd(ids):
            logits = np.full(100, -10.0)
            logits[5] = 10.0
            return logits

        prompt = [1, 2, 5, 1, 2]
        dec = PromptLookupDecoder(fwd, cfg)
        ids, stats = dec.generate(prompt, max_new_tokens=6)
        # With max_speculative=1, each speculative step produces at most 1 token
        if stats.speculative_steps > 0:
            assert stats.total_draft_tokens // max(1, stats.speculative_steps) <= 1
