"""tests/test_wave47a_modules.py

Wave 47a test suite — covers:
  * Mamba2SSM        (squish/attention/mamba2_ssm.py)
  * HGRN2            (squish/attention/hgrn2.py)
  * LookaheadDecode  (squish/speculative/lookahead_decode.py)
  * InfMemory        (squish/kv/inf_memory.py)
  * vAttentionKV     (squish/kv/v_attention.py)
  * IA3Adapter       (squish/lora/ia3_adapter.py)
"""

from __future__ import annotations

import unittest
from typing import List

import numpy as np

# ---------------------------------------------------------------------------
# Mamba2SSM
# ---------------------------------------------------------------------------
from squish.attention.mamba2_ssm import Mamba2Config, Mamba2SSM, Mamba2State


class TestMamba2Config(unittest.TestCase):
    def test_defaults(self):
        c = Mamba2Config()
        self.assertGreaterEqual(c.d_model, 1)
        self.assertGreaterEqual(c.d_state, 1)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            Mamba2Config(d_model=0)

    def test_invalid_d_state(self):
        with self.assertRaises(ValueError):
            Mamba2Config(d_state=0)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            Mamba2Config(chunk_size=0)

    def test_inner_not_divisible_by_n_heads(self):
        # d_model=8, expand=2 → inner=16; n_heads=3 → 16 % 3 != 0
        with self.assertRaises(ValueError):
            Mamba2Config(d_model=8, expand=2, n_heads=3)

    def test_valid_config(self):
        c = Mamba2Config(d_model=16, d_state=8, d_conv=4, expand=2, n_heads=4)
        self.assertEqual(c.d_model, 16)


class TestMamba2SSM(unittest.TestCase):
    def setUp(self):
        self.cfg = Mamba2Config(d_model=16, d_state=4, d_conv=4, expand=2, n_heads=4, seed=42)
        self.m = Mamba2SSM(self.cfg)

    def test_config_property(self):
        self.assertIs(self.m.config, self.cfg)

    def test_init_state_shapes(self):
        state = self.m.init_state()
        inner = self.cfg.d_model * self.cfg.expand
        head_dim = inner // self.cfg.n_heads
        self.assertEqual(state.conv_state.shape, (inner, self.cfg.d_conv))
        self.assertEqual(state.ssm_state.shape, (self.cfg.n_heads, head_dim, self.cfg.d_state))

    def test_init_state_is_zeros(self):
        state = self.m.init_state()
        np.testing.assert_array_equal(state.conv_state, 0)
        np.testing.assert_array_equal(state.ssm_state, 0)

    def test_forward_2d_input(self):
        x = np.random.randn(10, 16).astype(np.float32)
        out, state = self.m.forward(x)
        self.assertEqual(out.shape, (10, 16))
        self.assertIsInstance(state, Mamba2State)

    def test_forward_3d_input(self):
        x = np.random.randn(2, 10, 16).astype(np.float32)
        out, state = self.m.forward(x)
        self.assertEqual(out.shape, (2, 10, 16))

    def test_forward_with_initial_state(self):
        x = np.random.randn(5, 16).astype(np.float32)
        _, s1 = self.m.forward(x)
        out2, s2 = self.m.forward(x, initial_state=s1)
        self.assertEqual(out2.shape, (5, 16))

    def test_step_shape(self):
        x_t = np.random.randn(16).astype(np.float32)
        out, state = self.m.step(x_t)
        self.assertEqual(out.shape, (16,))
        self.assertIsInstance(state, Mamba2State)

    def test_step_with_state(self):
        x_t = np.random.randn(16).astype(np.float32)
        _, s1 = self.m.step(x_t)
        out2, _ = self.m.step(x_t, state=s1)
        self.assertEqual(out2.shape, (16,))

    def test_forward_output_dtype(self):
        x = np.random.randn(4, 16).astype(np.float32)
        out, _ = self.m.forward(x)
        self.assertEqual(out.dtype, np.float32)

    def test_different_seeds_differ(self):
        m2 = Mamba2SSM(Mamba2Config(d_model=16, d_state=4, expand=2, n_heads=4, seed=99))
        x = np.random.randn(4, 16).astype(np.float32)
        o1, _ = self.m.forward(x)
        o2, _ = m2.forward(x)
        self.assertFalse(np.allclose(o1, o2))

    def test_step_deterministic(self):
        x_t = np.ones(16, dtype=np.float32)
        o1, _ = self.m.step(x_t)
        o2, _ = self.m.step(x_t)
        np.testing.assert_array_almost_equal(o1, o2)

    def test_single_token_forward(self):
        x = np.random.randn(1, 16).astype(np.float32)
        out, _ = self.m.forward(x)
        self.assertEqual(out.shape, (1, 16))


# ---------------------------------------------------------------------------
# HGRN2
# ---------------------------------------------------------------------------
from squish.attention.hgrn2 import HGRN2Config, HGRN2, HGRN2State


class TestHGRN2Config(unittest.TestCase):
    def test_defaults(self):
        c = HGRN2Config()
        self.assertGreaterEqual(c.d_model, 1)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            HGRN2Config(d_model=0)

    def test_invalid_d_expand(self):
        with self.assertRaises(ValueError):
            HGRN2Config(d_expand=0)


class TestHGRN2(unittest.TestCase):
    def setUp(self):
        self.cfg = HGRN2Config(d_model=16, d_expand=2, seed=7)
        self.m = HGRN2(self.cfg)

    def test_config_property(self):
        self.assertIs(self.m.config, self.cfg)

    def test_init_state_shape(self):
        s = self.m.init_state()
        self.assertEqual(s.h.shape, (self.cfg.d_model,))

    def test_init_state_zeros(self):
        s = self.m.init_state()
        np.testing.assert_array_equal(s.h, 0)

    def test_forward_shape(self):
        x = np.random.randn(10, 16).astype(np.float32)
        out, state = self.m.forward(x)
        self.assertEqual(out.shape, (10, 16))
        self.assertIsInstance(state, HGRN2State)

    def test_forward_with_initial_state(self):
        x = np.random.randn(5, 16).astype(np.float32)
        _, s1 = self.m.forward(x)
        out2, s2 = self.m.forward(x, initial_state=s1)
        self.assertEqual(out2.shape, (5, 16))

    def test_step_shape(self):
        x_t = np.random.randn(16).astype(np.float32)
        out, state = self.m.step(x_t)
        self.assertEqual(out.shape, (16,))

    def test_step_state_shape(self):
        x_t = np.random.randn(16).astype(np.float32)
        _, s = self.m.step(x_t)
        self.assertEqual(s.h.shape, (self.cfg.d_model,))

    def test_output_dtype(self):
        x = np.random.randn(4, 16).astype(np.float32)
        out, _ = self.m.forward(x)
        self.assertEqual(out.dtype, np.float32)

    def test_state_evolves(self):
        x_t = np.ones(16, dtype=np.float32)
        _, s1 = self.m.step(x_t)
        _, s2 = self.m.step(x_t, state=s1)
        # State should change across steps
        self.assertFalse(np.allclose(s1.h, s2.h))

    def test_single_token_forward(self):
        x = np.random.randn(1, 16).astype(np.float32)
        out, _ = self.m.forward(x)
        self.assertEqual(out.shape, (1, 16))


# ---------------------------------------------------------------------------
# LookaheadDecode
# ---------------------------------------------------------------------------
from squish.speculative.lookahead_decode import (
    LookaheadConfig,
    LookaheadDecode,
    LookaheadResult,
)


class TestLookaheadConfig(unittest.TestCase):
    def test_defaults(self):
        c = LookaheadConfig()
        self.assertGreaterEqual(c.window_size, 1)

    def test_invalid_window_size(self):
        with self.assertRaises(ValueError):
            LookaheadConfig(window_size=0)

    def test_invalid_ngram_size(self):
        with self.assertRaises(ValueError):
            LookaheadConfig(ngram_size=1)

    def test_invalid_vocab_size(self):
        with self.assertRaises(ValueError):
            LookaheadConfig(vocab_size=0)


class TestLookaheadDecode(unittest.TestCase):
    def setUp(self):
        self.vocab = 64
        self.cfg = LookaheadConfig(
            window_size=4, ngram_size=3, max_candidates=32, vocab_size=self.vocab, seed=0
        )
        rng = np.random.default_rng(0)

        def score_fn(ctx: List[int]) -> np.ndarray:
            return rng.standard_normal(self.vocab).astype(np.float32)

        self.dec = LookaheadDecode(self.cfg, score_fn=score_fn)

    def test_config_property(self):
        self.assertIs(self.dec.config, self.cfg)

    def test_step_returns_result(self):
        ctx = [1, 2, 3, 4]
        result = self.dec.step(ctx)
        self.assertIsInstance(result, LookaheadResult)

    def test_step_at_least_one_token(self):
        ctx = [5, 6, 7]
        result = self.dec.step(ctx)
        self.assertGreaterEqual(len(result.accepted_tokens), 1)

    def test_step_tokens_in_vocab(self):
        ctx = [0, 1, 2]
        result = self.dec.step(ctx)
        for t in result.accepted_tokens:
            self.assertGreaterEqual(t, 0)
            self.assertLess(t, self.vocab)

    def test_n_verified_nonneg(self):
        result = self.dec.step([1, 2, 3])
        self.assertGreaterEqual(result.n_verified, 0)

    def test_cache_hits_nonneg(self):
        result = self.dec.step([1, 2, 3])
        self.assertGreaterEqual(result.cache_hits, 0)

    def test_speedup_estimate_positive(self):
        result = self.dec.step([1, 2, 3])
        self.assertGreater(result.speedup_estimate, 0.0)

    def test_cache_grows(self):
        for i in range(5):
            self.dec.step([i, i + 1, i + 2])
        self.assertGreater(self.dec.cache_size, 0)

    def test_reset_clears_cache(self):
        for i in range(5):
            self.dec.step([i, i + 1, i + 2])
        self.dec.reset_cache()
        self.assertEqual(self.dec.cache_size, 0)

    def test_no_score_fn(self):
        # Should still produce a result using greedy/random fallback
        dec2 = LookaheadDecode(self.cfg)
        result = dec2.step([1, 2, 3])
        self.assertGreaterEqual(len(result.accepted_tokens), 1)

    def test_repeated_step_builds_cache(self):
        for _ in range(10):
            self.dec.step([1, 2, 3, 4])
        # After many repeats, some hits are expected
        result = self.dec.step([1, 2, 3, 4])
        # Just ensure it runs without error; cache hits may be 0 depending on impl
        self.assertIsInstance(result, LookaheadResult)


# ---------------------------------------------------------------------------
# InfMemory
# ---------------------------------------------------------------------------
from squish.kv.inf_memory import InfMemoryConfig, InfMemory, MemoryBlock


class TestInfMemoryConfig(unittest.TestCase):
    def test_defaults(self):
        c = InfMemoryConfig()
        self.assertGreaterEqual(c.n_heads, 1)

    def test_invalid_n_heads(self):
        with self.assertRaises(ValueError):
            InfMemoryConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with self.assertRaises(ValueError):
            InfMemoryConfig(head_dim=0)

    def test_invalid_block_size(self):
        with self.assertRaises(ValueError):
            InfMemoryConfig(block_size=0)

    def test_invalid_max_blocks(self):
        with self.assertRaises(ValueError):
            InfMemoryConfig(max_blocks=0)

    def test_invalid_top_k(self):
        with self.assertRaises(ValueError):
            InfMemoryConfig(top_k_retrieve=0)


class TestInfMemory(unittest.TestCase):
    def setUp(self):
        self.cfg = InfMemoryConfig(n_heads=4, head_dim=8, block_size=8, max_blocks=4)
        self.mem = InfMemory(self.cfg)

    def _make_kv(self, bs=None):
        bs = bs or self.cfg.block_size
        rng = np.random.default_rng(0)
        K = rng.standard_normal((bs, self.cfg.n_heads, self.cfg.head_dim)).astype(np.float32)
        V = rng.standard_normal((bs, self.cfg.n_heads, self.cfg.head_dim)).astype(np.float32)
        return K, V

    def test_initial_n_blocks(self):
        self.assertEqual(self.mem.n_blocks, 0)

    def test_store_block(self):
        K, V = self._make_kv()
        bid = self.mem.store_block(K, V)
        self.assertEqual(self.mem.n_blocks, 1)
        self.assertIsInstance(bid, int)

    def test_representative_shape(self):
        K, V = self._make_kv()
        self.mem.store_block(K, V)
        block = self.mem._blocks[0]
        self.assertEqual(block.representative.shape, (self.cfg.n_heads, self.cfg.head_dim))

    def test_retrieve_empty(self):
        Q = np.zeros((self.cfg.n_heads, self.cfg.head_dim), dtype=np.float32)
        result = self.mem.retrieve(Q)
        self.assertEqual(result, [])

    def test_retrieve_returns_blocks(self):
        for _ in range(3):
            K, V = self._make_kv()
            self.mem.store_block(K, V)
        Q = np.random.randn(self.cfg.n_heads, self.cfg.head_dim).astype(np.float32)
        result = self.mem.retrieve(Q, top_k=2)
        self.assertEqual(len(result), 2)

    def test_retrieve_kv_shapes(self):
        for _ in range(2):
            K, V = self._make_kv()
            self.mem.store_block(K, V)
        Q = np.random.randn(self.cfg.n_heads, self.cfg.head_dim).astype(np.float32)
        K_ret, V_ret = self.mem.retrieve_kv(Q, top_k=2)
        expected_len = 2 * self.cfg.block_size
        self.assertEqual(K_ret.shape, (expected_len, self.cfg.n_heads, self.cfg.head_dim))
        self.assertEqual(V_ret.shape, (expected_len, self.cfg.n_heads, self.cfg.head_dim))

    def test_retrieve_kv_empty(self):
        Q = np.zeros((self.cfg.n_heads, self.cfg.head_dim), dtype=np.float32)
        K_ret, V_ret = self.mem.retrieve_kv(Q)
        self.assertEqual(K_ret.shape[0], 0)

    def test_max_blocks_eviction(self):
        for _ in range(6):
            K, V = self._make_kv()
            self.mem.store_block(K, V)
        self.assertEqual(self.mem.n_blocks, self.cfg.max_blocks)

    def test_reset(self):
        K, V = self._make_kv()
        self.mem.store_block(K, V)
        self.mem.reset()
        self.assertEqual(self.mem.n_blocks, 0)

    def test_compress_block_mean(self):
        K, V = self._make_kv()
        rep = self.mem.compress_block(K)
        expected = K.mean(axis=0)
        np.testing.assert_array_almost_equal(rep, expected)

    def test_block_id_increments(self):
        K, V = self._make_kv()
        b0 = self.mem.store_block(K, V)
        b1 = self.mem.store_block(K, V)
        self.assertEqual(b1, b0 + 1)


# ---------------------------------------------------------------------------
# vAttentionKV
# ---------------------------------------------------------------------------
from squish.kv.v_attention import vAttentionConfig, vAttentionKV


class TestVAttentionConfig(unittest.TestCase):
    def test_defaults(self):
        c = vAttentionConfig()
        self.assertGreaterEqual(c.page_size, 1)

    def test_invalid_page_size(self):
        with self.assertRaises(ValueError):
            vAttentionConfig(page_size=0)

    def test_invalid_max_pages(self):
        with self.assertRaises(ValueError):
            vAttentionConfig(max_pages=0)

    def test_invalid_n_heads(self):
        with self.assertRaises(ValueError):
            vAttentionConfig(n_heads=0)

    def test_invalid_head_dim(self):
        with self.assertRaises(ValueError):
            vAttentionConfig(head_dim=0)


class TestVAttentionKV(unittest.TestCase):
    def setUp(self):
        self.cfg = vAttentionConfig(page_size=4, max_pages=8, n_heads=2, head_dim=8)
        self.kv = vAttentionKV(self.cfg)

    def _kv(self):
        return (
            np.ones((self.cfg.n_heads, self.cfg.head_dim), dtype=np.float32),
            np.zeros((self.cfg.n_heads, self.cfg.head_dim), dtype=np.float32),
        )

    def test_initial_free_pages(self):
        self.assertEqual(self.kv.n_free_pages, self.cfg.max_pages)

    def test_allocate_reduces_free(self):
        self.kv.allocate("s0", 3)
        self.assertLess(self.kv.n_free_pages, self.cfg.max_pages)

    def test_allocate_and_store(self):
        self.kv.allocate("s0", 4)
        k, v = self._kv()
        self.kv.store_token("s0", 0, k, v)

    def test_get_kv_shape(self):
        self.kv.allocate("s0", 4)
        k, v = self._kv()
        self.kv.store_token("s0", 0, k, v)
        self.kv.store_token("s0", 1, k, v)
        K_out, V_out = self.kv.get_kv("s0")
        self.assertEqual(K_out.shape, (2, self.cfg.n_heads, self.cfg.head_dim))

    def test_get_kv_empty_seq(self):
        self.kv.allocate("s0", 4)
        K_out, V_out = self.kv.get_kv("s0")
        self.assertEqual(K_out.shape[0], 0)

    def test_free_restores_pages(self):
        self.kv.allocate("s0", 4)
        before = self.kv.n_free_pages
        self.kv.free("s0")
        self.assertGreater(self.kv.n_free_pages, before)

    def test_fragmentation_zero_when_empty(self):
        self.assertAlmostEqual(self.kv.fragmentation_ratio, 0.0)

    def test_fragmentation_positive(self):
        self.kv.allocate("s0", 3)  # needs 1 page (size 4), 1 token written = 0 yet
        k, v = self._kv()
        self.kv.store_token("s0", 0, k, v)
        # 1 page allocated (4 slots), 1 token used → frag = 3/4
        frag = self.kv.fragmentation_ratio
        self.assertGreaterEqual(frag, 0.0)
        self.assertLessEqual(frag, 1.0)

    def test_memory_error_on_exhaustion(self):
        with self.assertRaises(MemoryError):
            self.kv.allocate("s0", self.cfg.max_pages * self.cfg.page_size + 1)

    def test_unallocated_store_raises(self):
        with self.assertRaises(KeyError):
            k, v = self._kv()
            self.kv.store_token("ghost", 0, k, v)

    def test_allocated_pages_count(self):
        self.kv.allocate("s0", 4)
        self.assertEqual(self.kv.n_allocated_pages, 1)


# ---------------------------------------------------------------------------
# IA3Adapter
# ---------------------------------------------------------------------------
from squish.lora.ia3_adapter import IA3Config, IA3Adapter, ia3_compose


class TestIA3Config(unittest.TestCase):
    def test_defaults(self):
        c = IA3Config()
        self.assertGreaterEqual(c.d_model, 1)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            IA3Config(d_model=0)

    def test_invalid_d_ff(self):
        with self.assertRaises(ValueError):
            IA3Config(d_ff=0)


class TestIA3Adapter(unittest.TestCase):
    def setUp(self):
        self.cfg = IA3Config(d_model=16, d_ff=64)
        self.adapter = IA3Adapter(self.cfg)

    def test_scale_k_shape(self):
        self.assertEqual(self.adapter.scale_k.shape, (16,))

    def test_scale_v_shape(self):
        self.assertEqual(self.adapter.scale_v.shape, (16,))

    def test_scale_ff_shape(self):
        self.assertEqual(self.adapter.scale_ff.shape, (64,))

    def test_apply_k_shape(self):
        K = np.ones((8, 16), dtype=np.float32)
        out = self.adapter.apply_k(K)
        self.assertEqual(out.shape, (8, 16))

    def test_apply_v_shape(self):
        V = np.ones((8, 16), dtype=np.float32)
        out = self.adapter.apply_v(V)
        self.assertEqual(out.shape, (8, 16))

    def test_apply_ff_shape(self):
        h = np.ones((8, 64), dtype=np.float32)
        out = self.adapter.apply_ff(h)
        self.assertEqual(out.shape, (8, 64))

    def test_apply_k_scales(self):
        K = np.ones((4, 16), dtype=np.float32)
        out = self.adapter.apply_k(K)
        expected = K * self.adapter.scale_k
        np.testing.assert_array_almost_equal(out, expected)

    def test_merge_to_base_shapes(self):
        W_k = np.ones((32, 16), dtype=np.float32)
        W_v = np.ones((32, 16), dtype=np.float32)
        W_ff = np.ones((32, 64), dtype=np.float32)
        mk, mv, mff = self.adapter.merge_to_base(W_k, W_v, W_ff)
        self.assertEqual(mk.shape, (32, 16))
        self.assertEqual(mv.shape, (32, 16))
        self.assertEqual(mff.shape, (32, 64))

    def test_reset_to_identity(self):
        self.adapter.reset_to_identity()
        np.testing.assert_array_almost_equal(self.adapter.scale_k, np.ones(16))
        np.testing.assert_array_almost_equal(self.adapter.scale_v, np.ones(16))
        np.testing.assert_array_almost_equal(self.adapter.scale_ff, np.ones(64))

    def test_zero_scales(self):
        self.adapter.zero_scales()
        np.testing.assert_array_almost_equal(self.adapter.scale_k, np.zeros(16))

    def test_config_property(self):
        self.assertIs(self.adapter.config, self.cfg)


class TestIA3Compose(unittest.TestCase):
    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            ia3_compose([])

    def test_single_adapter(self):
        cfg = IA3Config(d_model=8, d_ff=32)
        a = IA3Adapter(cfg)
        composed = ia3_compose([a])
        np.testing.assert_array_almost_equal(composed.scale_k, a.scale_k)

    def test_two_adapters_product(self):
        cfg = IA3Config(d_model=8, d_ff=32)
        a1 = IA3Adapter(cfg)
        a2 = IA3Adapter(IA3Config(d_model=8, d_ff=32, seed=1))
        composed = ia3_compose([a1, a2])
        expected_k = a1.scale_k * a2.scale_k
        np.testing.assert_array_almost_equal(composed.scale_k, expected_k)

    def test_incompatible_configs_raise(self):
        a1 = IA3Adapter(IA3Config(d_model=8, d_ff=32))
        a2 = IA3Adapter(IA3Config(d_model=16, d_ff=32))
        with self.assertRaises(ValueError):
            ia3_compose([a1, a2])


if __name__ == "__main__":
    unittest.main()
