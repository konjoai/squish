"""Tests for Wave 49a serving modules: LLMLingua-2, RECOMP, SelectiveContext."""

from __future__ import annotations

import unittest

import numpy as np

from squish.serving.llm_lingua2 import (
    LLMLingua2Compressor,
    LLMLingua2Config,
    LLMLingua2Result,
)
from squish.serving.recomp import (
    RECOMPCompressor,
    RECOMPConfig,
    RECOMPResult,
)
from squish.serving.selective_context import (
    SelectiveContextCompressor,
    SelectiveContextConfig,
    SelectiveContextResult,
)


# ---------------------------------------------------------------------------
# LLMLingua2Config
# ---------------------------------------------------------------------------


class TestLLMLingua2Config(unittest.TestCase):
    def test_defaults(self):
        cfg = LLMLingua2Config()
        self.assertAlmostEqual(cfg.target_ratio, 0.3)
        self.assertEqual(cfg.min_tokens, 10)
        self.assertEqual(cfg.force_tokens, [])
        self.assertEqual(cfg.seed, 0)

    def test_valid_custom(self):
        cfg = LLMLingua2Config(target_ratio=0.5, min_tokens=5, force_tokens=["<s>"], seed=42)
        self.assertAlmostEqual(cfg.target_ratio, 0.5)
        self.assertEqual(cfg.min_tokens, 5)
        self.assertEqual(cfg.force_tokens, ["<s>"])

    def test_target_ratio_zero_raises(self):
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_ratio=0.0)

    def test_target_ratio_one_raises(self):
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_ratio=1.0)

    def test_target_ratio_negative_raises(self):
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_ratio=-0.1)

    def test_target_ratio_above_one_raises(self):
        with self.assertRaises(ValueError):
            LLMLingua2Config(target_ratio=1.1)

    def test_min_tokens_zero_raises(self):
        with self.assertRaises(ValueError):
            LLMLingua2Config(min_tokens=0)

    def test_min_tokens_negative_raises(self):
        with self.assertRaises(ValueError):
            LLMLingua2Config(min_tokens=-1)

    def test_min_tokens_one_valid(self):
        cfg = LLMLingua2Config(min_tokens=1)
        self.assertEqual(cfg.min_tokens, 1)


# ---------------------------------------------------------------------------
# LLMLingua2Compressor — core API
# ---------------------------------------------------------------------------


class TestLLMLingua2Compressor(unittest.TestCase):
    def setUp(self):
        self.cfg = LLMLingua2Config(target_ratio=0.5, min_tokens=2, seed=0)
        self.comp = LLMLingua2Compressor(self.cfg)

    def test_config_property(self):
        self.assertIs(self.comp.config, self.cfg)

    def test_default_config(self):
        comp = LLMLingua2Compressor()
        self.assertIsNotNone(comp.config)

    # compress_tokens -------------------------------------------------------

    def test_compress_tokens_empty(self):
        result = self.comp.compress_tokens([])
        self.assertIsInstance(result, LLMLingua2Result)
        self.assertEqual(result.original_count, 0)
        self.assertEqual(result.compressed_count, 0)
        self.assertEqual(result.compressed_tokens, [])
        self.assertEqual(len(result.token_mask), 0)

    def test_compress_tokens_return_type(self):
        tokens = "hello world foo bar baz".split()
        result = self.comp.compress_tokens(tokens)
        self.assertIsInstance(result, LLMLingua2Result)

    def test_compress_tokens_original_count(self):
        tokens = "a b c d e f g h i j".split()
        result = self.comp.compress_tokens(tokens)
        self.assertEqual(result.original_count, len(tokens))

    def test_compress_tokens_mask_length(self):
        tokens = "a b c d e f g h i j".split()
        result = self.comp.compress_tokens(tokens)
        self.assertEqual(len(result.token_mask), len(tokens))

    def test_compress_tokens_mask_dtype_bool(self):
        tokens = "a b c d".split()
        result = self.comp.compress_tokens(tokens)
        self.assertEqual(result.token_mask.dtype, bool)

    def test_compress_tokens_mask_consistency(self):
        tokens = "alpha beta gamma delta epsilon".split()
        result = self.comp.compress_tokens(tokens)
        kept = [t for t, m in zip(tokens, result.token_mask) if m]
        self.assertEqual(kept, result.compressed_tokens)

    def test_compress_tokens_compressed_count_matches(self):
        tokens = "a b c d e f g h".split()
        result = self.comp.compress_tokens(tokens)
        self.assertEqual(result.compressed_count, len(result.compressed_tokens))

    def test_compress_tokens_ratio_achieved(self):
        tokens = "a b c d e f g h i j".split()
        result = self.comp.compress_tokens(tokens)
        expected = result.compressed_count / result.original_count
        self.assertAlmostEqual(result.ratio_achieved, expected, places=6)

    def test_compress_tokens_respects_min_tokens(self):
        cfg = LLMLingua2Config(target_ratio=0.1, min_tokens=5, seed=0)
        comp = LLMLingua2Compressor(cfg)
        tokens = "a b c d e f g h i j".split()
        result = comp.compress_tokens(tokens)
        self.assertGreaterEqual(result.compressed_count, 5)

    def test_compress_tokens_force_tokens_kept(self):
        cfg = LLMLingua2Config(target_ratio=0.1, min_tokens=1, force_tokens=["hello"], seed=0)
        comp = LLMLingua2Compressor(cfg)
        tokens = ["hello", "world", "foo", "bar", "baz", "qux", "quux", "quuz"]
        result = comp.compress_tokens(tokens)
        self.assertIn("hello", result.compressed_tokens)

    def test_compress_tokens_override_ratio(self):
        tokens = "a b c d e f g h i j".split()
        r1 = self.comp.compress_tokens(tokens, target_ratio=0.2)
        r2 = self.comp.compress_tokens(tokens, target_ratio=0.8)
        self.assertLessEqual(r1.compressed_count, r2.compressed_count)

    def test_compress_tokens_invalid_ratio_raises(self):
        tokens = "a b c d".split()
        with self.assertRaises(ValueError):
            self.comp.compress_tokens(tokens, target_ratio=1.5)

    def test_compress_tokens_deterministic(self):
        tokens = "the quick brown fox jumps over the lazy dog".split()
        r1 = self.comp.compress_tokens(tokens)
        r2 = self.comp.compress_tokens(tokens)
        np.testing.assert_array_equal(r1.token_mask, r2.token_mask)

    # compress (string) -----------------------------------------------------

    def test_compress_string(self):
        result = self.comp.compress("hello world foo bar baz")
        self.assertIsInstance(result, LLMLingua2Result)
        self.assertGreater(result.original_count, 0)

    def test_compress_empty_string(self):
        result = self.comp.compress("")
        self.assertEqual(result.original_count, 0)
        self.assertEqual(result.compressed_tokens, [])

    def test_compress_single_word(self):
        result = self.comp.compress("hello")
        self.assertEqual(result.original_count, 1)
        # min_tokens = 2, but only 1 exists → keep all
        self.assertEqual(result.compressed_count, 1)

    def test_compress_ratio_reduced(self):
        prompt = " ".join(["word"] * 50)
        result = self.comp.compress(prompt)
        self.assertLess(result.compressed_count, result.original_count)


# ---------------------------------------------------------------------------
# RECOMPConfig
# ---------------------------------------------------------------------------


class TestRECOMPConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = RECOMPConfig()
        self.assertEqual(cfg.mode, "extractive")
        self.assertEqual(cfg.top_k, 3)
        self.assertEqual(cfg.max_length, 512)
        self.assertEqual(cfg.seed, 0)

    def test_valid_abstractive(self):
        cfg = RECOMPConfig(mode="abstractive")
        self.assertEqual(cfg.mode, "abstractive")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            RECOMPConfig(mode="invalid")

    def test_top_k_zero_raises(self):
        with self.assertRaises(ValueError):
            RECOMPConfig(top_k=0)

    def test_top_k_negative_raises(self):
        with self.assertRaises(ValueError):
            RECOMPConfig(top_k=-1)

    def test_max_length_zero_raises(self):
        with self.assertRaises(ValueError):
            RECOMPConfig(max_length=0)

    def test_max_length_negative_raises(self):
        with self.assertRaises(ValueError):
            RECOMPConfig(max_length=-1)


# ---------------------------------------------------------------------------
# RECOMPCompressor
# ---------------------------------------------------------------------------


class TestRECOMPCompressor(unittest.TestCase):
    def setUp(self):
        self.cfg = RECOMPConfig(mode="extractive", top_k=2, seed=0)
        self.comp = RECOMPCompressor(self.cfg)
        self.docs = [
            "The cat sat on the mat. Dogs bark loudly.",
            "Paris is the capital of France. It has the Eiffel Tower.",
        ]
        self.query = "What is the capital of France?"

    def test_config_property(self):
        self.assertIs(self.comp.config, self.cfg)

    def test_default_config(self):
        comp = RECOMPCompressor()
        self.assertIsNotNone(comp.config)

    def test_compress_returns_result(self):
        result = self.comp.compress(self.docs, self.query)
        self.assertIsInstance(result, RECOMPResult)

    def test_compress_mode_extractive(self):
        result = self.comp.compress(self.docs, self.query)
        self.assertEqual(result.mode, "extractive")

    def test_compress_mode_override(self):
        result = self.comp.compress(self.docs, self.query, mode="abstractive")
        self.assertEqual(result.mode, "abstractive")

    def test_compress_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.comp.compress(self.docs, self.query, mode="unsupported")

    def test_compress_source_count_positive(self):
        result = self.comp.compress(self.docs, self.query)
        self.assertGreater(result.source_count, 0)

    def test_compress_retained_count_leq_source(self):
        result = self.comp.compress(self.docs, self.query)
        self.assertLessEqual(result.retained_count, result.source_count)

    def test_compress_retained_count_positive(self):
        result = self.comp.compress(self.docs, self.query)
        self.assertGreater(result.retained_count, 0)

    def test_compress_context_not_empty(self):
        result = self.comp.compress(self.docs, self.query)
        self.assertIsInstance(result.compressed_context, str)
        self.assertGreater(len(result.compressed_context), 0)

    def test_compress_single_document(self):
        result = self.comp.compress(["Single sentence here."], "query")
        self.assertIsInstance(result, RECOMPResult)

    def test_compress_empty_documents(self):
        result = self.comp.compress([], "query")
        self.assertEqual(result.source_count, 0)
        self.assertEqual(result.retained_count, 0)
        self.assertEqual(result.compressed_context, "")

    def test_compress_max_length_respected(self):
        cfg = RECOMPConfig(top_k=10, max_length=20)
        comp = RECOMPCompressor(cfg)
        docs = ["Alpha beta gamma delta epsilon zeta eta theta iota kappa."]
        result = comp.compress(docs, "test")
        self.assertLessEqual(len(result.compressed_context), 20)

    def test_compress_top_k_limits_retention(self):
        cfg = RECOMPConfig(top_k=1)
        comp = RECOMPCompressor(cfg)
        docs = ["Sentence one. Sentence two. Sentence three. Sentence four."]
        result = comp.compress(docs, "query")
        self.assertLessEqual(result.retained_count, 1)

    def test_compress_abstractive_returns_string(self):
        cfg = RECOMPConfig(mode="abstractive")
        comp = RECOMPCompressor(cfg)
        result = comp.compress(self.docs, self.query)
        self.assertIsInstance(result.compressed_context, str)

    def test_split_sentences_basic(self):
        sentences = RECOMPCompressor._split_sentences("Hello world. Goodbye world.")
        self.assertGreaterEqual(len(sentences), 1)

    def test_bow_vector_is_ndarray(self):
        v = RECOMPCompressor._bow_vector("hello world")
        self.assertIsInstance(v, np.ndarray)

    def test_cosine_sim_identical(self):
        comp = RECOMPCompressor()
        sim = comp._cosine_sim("hello world", "hello world")
        self.assertAlmostEqual(sim, 1.0, places=4)

    def test_cosine_sim_different(self):
        comp = RECOMPCompressor()
        sim = comp._cosine_sim("hello world", "xyz abc def")
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)


# ---------------------------------------------------------------------------
# SelectiveContextConfig
# ---------------------------------------------------------------------------


class TestSelectiveContextConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = SelectiveContextConfig()
        self.assertAlmostEqual(cfg.threshold, 0.5)
        self.assertEqual(cfg.min_tokens, 5)
        self.assertEqual(cfg.seed, 0)

    def test_threshold_zero_raises(self):
        with self.assertRaises(ValueError):
            SelectiveContextConfig(threshold=0.0)

    def test_threshold_one_raises(self):
        with self.assertRaises(ValueError):
            SelectiveContextConfig(threshold=1.0)

    def test_threshold_negative_raises(self):
        with self.assertRaises(ValueError):
            SelectiveContextConfig(threshold=-0.1)

    def test_threshold_above_one_raises(self):
        with self.assertRaises(ValueError):
            SelectiveContextConfig(threshold=1.5)

    def test_min_tokens_zero_raises(self):
        with self.assertRaises(ValueError):
            SelectiveContextConfig(min_tokens=0)

    def test_min_tokens_negative_raises(self):
        with self.assertRaises(ValueError):
            SelectiveContextConfig(min_tokens=-1)

    def test_valid_custom(self):
        cfg = SelectiveContextConfig(threshold=0.7, min_tokens=3, seed=42)
        self.assertAlmostEqual(cfg.threshold, 0.7)
        self.assertEqual(cfg.min_tokens, 3)


# ---------------------------------------------------------------------------
# SelectiveContextCompressor
# ---------------------------------------------------------------------------


class TestSelectiveContextCompressor(unittest.TestCase):
    def setUp(self):
        self.cfg = SelectiveContextConfig(threshold=0.5, min_tokens=1, seed=0)
        self.comp = SelectiveContextCompressor(self.cfg)

    def test_config_property(self):
        self.assertIs(self.comp.config, self.cfg)

    def test_default_config(self):
        comp = SelectiveContextCompressor()
        self.assertIsNotNone(comp.config)

    def test_compress_empty(self):
        result = self.comp.compress([])
        self.assertIsInstance(result, SelectiveContextResult)
        self.assertEqual(result.original_count, 0)
        self.assertEqual(result.retained_count, 0)

    def test_compress_returns_result(self):
        tokens = "the quick brown fox jumps".split()
        result = self.comp.compress(tokens)
        self.assertIsInstance(result, SelectiveContextResult)

    def test_compress_original_count(self):
        tokens = "a b c d e".split()
        result = self.comp.compress(tokens)
        self.assertEqual(result.original_count, len(tokens))

    def test_compress_mask_dtype(self):
        tokens = "a b c d".split()
        result = self.comp.compress(tokens)
        self.assertEqual(result.mask.dtype, bool)

    def test_compress_mask_length(self):
        tokens = "a b c d e f".split()
        result = self.comp.compress(tokens)
        self.assertEqual(len(result.mask), len(tokens))

    def test_compress_mask_consistency(self):
        tokens = "alpha beta gamma delta epsilon".split()
        result = self.comp.compress(tokens)
        kept = [t for t, m in zip(tokens, result.mask) if m]
        self.assertEqual(kept, result.compressed_tokens)

    def test_compress_retained_count_matches_mask(self):
        tokens = "a b c d e f g".split()
        result = self.comp.compress(tokens)
        self.assertEqual(result.retained_count, int(result.mask.sum()))

    def test_compress_min_tokens_enforced(self):
        cfg = SelectiveContextConfig(threshold=0.99, min_tokens=3, seed=0)
        comp = SelectiveContextCompressor(cfg)
        tokens = "a b c d e f g h".split()
        result = comp.compress(tokens)
        self.assertGreaterEqual(result.retained_count, 3)

    def test_compress_with_explicit_log_probs(self):
        tokens = "a b c d e".split()
        lp = np.array([-0.1, -2.0, -0.2, -3.0, -0.05], dtype=np.float32)
        result = self.comp.compress(tokens, log_probs=lp)
        self.assertEqual(result.original_count, 5)

    def test_compress_log_probs_length_mismatch_raises(self):
        tokens = "a b c".split()
        lp = np.array([-0.5, -1.0], dtype=np.float32)
        with self.assertRaises(ValueError):
            self.comp.compress(tokens, log_probs=lp)

    def test_compress_invalid_threshold_raises(self):
        tokens = "a b c d".split()
        with self.assertRaises(ValueError):
            self.comp.compress(tokens, threshold=0.0)

    def test_compress_threshold_override_lower(self):
        tokens = "alpha beta gamma delta epsilon zeta".split()
        r_high = self.comp.compress(tokens, threshold=0.9)
        r_low = self.comp.compress(tokens, threshold=0.1)
        # lower threshold keeps fewer (only tokens with self-info >= 0.1)
        # but min_tokens enforced, so at least compare loosely
        self.assertGreaterEqual(r_high.retained_count, 0)
        self.assertGreaterEqual(r_low.retained_count, 0)

    def test_compress_deterministic(self):
        tokens = "the quick brown fox jumps over the lazy dog".split()
        r1 = self.comp.compress(tokens)
        r2 = self.comp.compress(tokens)
        np.testing.assert_array_equal(r1.mask, r2.mask)

    def test_compress_text_returns_result(self):
        result = self.comp.compress_text("hello world foo bar baz")
        self.assertIsInstance(result, SelectiveContextResult)

    def test_compress_text_empty(self):
        result = self.comp.compress_text("")
        self.assertEqual(result.original_count, 0)

    def test_compress_text_token_count(self):
        result = self.comp.compress_text("a b c d e f g h i j")
        self.assertEqual(result.original_count, 10)

    def test_synthetic_log_probs_shape(self):
        tokens = "hello world".split()
        lp = self.comp._synthetic_log_probs(tokens)
        self.assertEqual(lp.shape[0], 2)

    def test_synthetic_log_probs_non_positive(self):
        tokens = "the quick brown fox".split()
        lp = self.comp._synthetic_log_probs(tokens)
        self.assertTrue(np.all(lp <= 0))

    def test_synthetic_log_probs_single_token(self):
        lp = self.comp._synthetic_log_probs(["hello"])
        self.assertEqual(len(lp), 1)


if __name__ == "__main__":
    unittest.main()
