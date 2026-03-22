"""Tests for Wave 52b VLM-efficiency modules.

Covers: VisualKVQuant, CrossModalRouter, VideoKVReuse, VLMSpecDecode,
VLMBatchScheduler, ImageEncoderCache.

Run:
    python -m pytest tests/test_wave52b_modules.py -v
"""

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# VisualKVQuant
# ---------------------------------------------------------------------------
from squish.vision.visual_kv_quant import (
    VisualKVQuantConfig,
    VisualKVQuantState,
    VisualKVQuant,
)


class TestVisualKVQuantConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VisualKVQuantConfig()
        self.assertEqual(cfg.k_bits, 4)
        self.assertEqual(cfg.v_bits, 6)
        self.assertEqual(cfg.group_size, 32)
        self.assertTrue(cfg.text_passthrough)

    def test_invalid_k_bits(self):
        with self.assertRaises(ValueError):
            VisualKVQuantConfig(k_bits=3)

    def test_invalid_v_bits(self):
        with self.assertRaises(ValueError):
            VisualKVQuantConfig(v_bits=5)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            VisualKVQuantConfig(group_size=0)

    def test_valid_bits_combinations(self):
        for k in (1, 2, 4, 8):
            for v in (1, 2, 4, 6, 8):
                cfg = VisualKVQuantConfig(k_bits=k, v_bits=v)
                self.assertEqual(cfg.k_bits, k)


class TestVisualKVQuant(unittest.TestCase):
    def setUp(self):
        self.cfg = VisualKVQuantConfig(k_bits=4, v_bits=4, group_size=8)
        self.quant = VisualKVQuant(self.cfg)

    def _make_kv(self, d=16):
        rng = np.random.default_rng(99)
        return rng.random(d).astype(np.float32), rng.random(d).astype(np.float32)

    def test_new_state_defaults(self):
        state = self.quant.new_state()
        self.assertEqual(state.n_visual_tokens, 0)
        self.assertEqual(state.n_text_tokens, 0)
        self.assertTrue(state.in_visual_segment)

    def test_update_visual_token(self):
        state = self.quant.new_state()
        k, v = self._make_kv()
        self.quant.update(k, v, "patch_0", state)
        self.assertEqual(state.n_visual_tokens, 1)
        self.assertEqual(state.n_text_tokens, 0)

    def test_update_switches_on_boundary(self):
        state = self.quant.new_state()
        k, v = self._make_kv()
        self.quant.update(k, v, self.cfg.boundary_token, state)
        self.assertFalse(state.in_visual_segment)
        k2, v2 = self._make_kv()
        self.quant.update(k2, v2, "hello", state)
        self.assertEqual(state.n_text_tokens, 1)

    def test_get_kv_shape(self):
        state = self.quant.new_state()
        for i in range(8):
            k, v = self._make_kv(d=16)
            self.quant.update(k, v, f"tok_{i}", state)
        k_out, v_out = self.quant.get_kv(state)
        self.assertEqual(k_out.shape[0], 8)

    def test_roundtrip_reconstruction(self):
        cfg = VisualKVQuantConfig(k_bits=8, v_bits=8, group_size=8)
        quant = VisualKVQuant(cfg)
        state = quant.new_state()
        rng = np.random.default_rng(7)
        original_k = rng.random(16).astype(np.float32)
        original_v = rng.random(16).astype(np.float32)
        quant.update(original_k, original_v, "tok", state)
        k_out, v_out = quant.get_kv(state)
        np.testing.assert_allclose(k_out[0], original_k, atol=0.5)

    def test_empty_get_kv(self):
        state = self.quant.new_state()
        k, v = self.quant.get_kv(state)
        self.assertEqual(k.shape[0], 0)

    def test_memory_summary_keys(self):
        state = self.quant.new_state()
        k, v = self._make_kv()
        self.quant.update(k, v, "tok", state)
        summary = self.quant.memory_summary(state)
        self.assertIn("n_visual_tokens", summary)
        self.assertIn("compression_ratio", summary)

    def test_compression_ratio_range(self):
        state = self.quant.new_state()
        k, v = self._make_kv()
        self.quant.update(k, v, "tok", state)
        cr = state.compression_ratio
        self.assertGreaterEqual(cr, 0.0)
        self.assertLessEqual(cr, 1.0)

    def test_total_tokens_property(self):
        state = self.quant.new_state()
        for i in range(3):
            k, v = self._make_kv()
            self.quant.update(k, v, f"t{i}", state)
        self.quant.update(*self._make_kv(), self.cfg.boundary_token, state)
        for i in range(2):
            k, v = self._make_kv()
            self.quant.update(k, v, f"t{i}", state)
        self.assertEqual(state.total_tokens, 6)


# ---------------------------------------------------------------------------
# CrossModalRouter
# ---------------------------------------------------------------------------
from squish.vision.cross_modal_attn import (
    CrossModalConfig,
    CrossModalResult,
    CrossModalRouter,
)


class TestCrossModalConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = CrossModalConfig()
        self.assertAlmostEqual(cfg.top_k_ratio, 0.3)
        self.assertEqual(cfg.n_heads, 4)

    def test_invalid_top_k_ratio_zero(self):
        with self.assertRaises(ValueError):
            CrossModalConfig(top_k_ratio=0.0)

    def test_invalid_top_k_ratio_gt_one(self):
        with self.assertRaises(ValueError):
            CrossModalConfig(top_k_ratio=1.1)

    def test_invalid_n_heads(self):
        with self.assertRaises(ValueError):
            CrossModalConfig(n_heads=0)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            CrossModalConfig(temperature=0.0)


class TestCrossModalRouter(unittest.TestCase):
    def setUp(self):
        # hidden_dim=32 divisible by n_heads=4
        self.cfg = CrossModalConfig(top_k_ratio=0.5, n_heads=4, linear_dim=32, seed=0)
        self.router = CrossModalRouter(self.cfg)

    def _make_inputs(self, n_q=16, n_k=32, d=32):
        rng = np.random.default_rng(42)
        q = rng.random((n_q, d)).astype(np.float32)
        k = rng.random((n_k, d)).astype(np.float32)
        v = rng.random((n_k, d)).astype(np.float32)
        gate = rng.random(n_q).astype(np.float32)
        return q, k, v, gate

    def test_output_shape(self):
        q, k, v, gate = self._make_inputs()
        result = self.router.route(q, k, v, gate)
        self.assertEqual(result.output.shape, (16, 32))

    def test_attn_weights_shape(self):
        q, k, v, gate = self._make_inputs()
        result = self.router.route(q, k, v, gate)
        self.assertEqual(result.attn_weights.shape, (4, 16, 32))

    def test_counts_sum_to_n_q(self):
        q, k, v, gate = self._make_inputs()
        result = self.router.route(q, k, v, gate)
        self.assertEqual(result.n_full_attn + result.n_linear_attn, 16)

    def test_speedup_ratio_range(self):
        q, k, v, gate = self._make_inputs()
        result = self.router.route(q, k, v, gate)
        self.assertGreaterEqual(result.speedup_ratio, 0.0)
        self.assertLessEqual(result.speedup_ratio, 1.0)

    def test_all_full_attn(self):
        cfg = CrossModalConfig(top_k_ratio=1.0, n_heads=4)
        router = CrossModalRouter(cfg)
        q, k, v, gate = self._make_inputs()
        result = router.route(q, k, v, gate)
        self.assertEqual(result.n_linear_attn, 0)

    def test_output_dtype_float32(self):
        q, k, v, gate = self._make_inputs()
        result = self.router.route(q, k, v, gate)
        self.assertEqual(result.output.dtype, np.float32)

    def test_single_query(self):
        rng = np.random.default_rng(0)
        q = rng.random((1, 32)).astype(np.float32)
        k = rng.random((8, 32)).astype(np.float32)
        v = rng.random((8, 32)).astype(np.float32)
        gate = np.array([0.9], dtype=np.float32)
        result = self.router.route(q, k, v, gate)
        self.assertEqual(result.output.shape, (1, 32))


# ---------------------------------------------------------------------------
# VideoKVReuse
# ---------------------------------------------------------------------------
from squish.vision.video_kv_reuse import (
    VideoKVReuseConfig,
    VideoKVReuseState,
    VideoKVReuse,
)


class TestVideoKVReuseConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VideoKVReuseConfig()
        self.assertAlmostEqual(cfg.change_threshold, 0.15)
        self.assertEqual(cfg.token_dim, 128)

    def test_invalid_threshold_zero(self):
        with self.assertRaises(ValueError):
            VideoKVReuseConfig(change_threshold=0.0)

    def test_invalid_threshold_one(self):
        with self.assertRaises(ValueError):
            VideoKVReuseConfig(change_threshold=1.0)

    def test_invalid_token_dim(self):
        with self.assertRaises(ValueError):
            VideoKVReuseConfig(token_dim=0)


class TestVideoKVReuse(unittest.TestCase):
    def setUp(self):
        self.cfg = VideoKVReuseConfig(change_threshold=0.2, token_dim=8)
        self.reuser = VideoKVReuse(self.cfg)

    def _kv_fn(self, patches):
        n = patches.shape[0]
        return (
            np.random.rand(n, 8).astype(np.float32),
            np.random.rand(n, 8).astype(np.float32),
        )

    def test_new_state(self):
        state = self.reuser.new_state()
        self.assertIsNone(state.prev_patches)
        self.assertEqual(state.n_reused, 0)

    def test_first_frame_all_recomputed(self):
        state = self.reuser.new_state()
        patches = np.random.rand(16, 8).astype(np.float32)
        k, v = self.reuser.process_frame(patches, self._kv_fn, state)
        self.assertEqual(state.n_recomputed, 16)
        self.assertEqual(state.n_reused, 0)

    def test_second_frame_identical_patches_all_reused(self):
        state = self.reuser.new_state()
        patches = np.random.rand(16, 8).astype(np.float32)

        def identity_kv(p):
            return (np.zeros((p.shape[0], 8), dtype=np.float32),
                    np.zeros((p.shape[0], 8), dtype=np.float32))

        self.reuser.process_frame(patches, identity_kv, state)
        state_before = state.n_reused
        self.reuser.process_frame(patches.copy(), identity_kv, state)
        self.assertGreater(state.n_reused, state_before)

    def test_output_shape(self):
        state = self.reuser.new_state()
        patches = np.random.rand(16, 8).astype(np.float32)
        k, v = self.reuser.process_frame(patches, self._kv_fn, state)
        self.assertEqual(k.shape, (16, 8))
        self.assertEqual(v.shape, (16, 8))

    def test_reuse_ratio_zero_on_first_frame(self):
        state = self.reuser.new_state()
        patches = np.random.rand(16, 8).astype(np.float32)
        self.reuser.process_frame(patches, self._kv_fn, state)
        self.assertAlmostEqual(self.reuser.reuse_ratio(state), 0.0)

    def test_cosine_sim_matrix_shape(self):
        a = np.random.rand(10, 8).astype(np.float32)
        b = np.random.rand(10, 8).astype(np.float32)
        sims = self.reuser._cosine_sim_matrix(a, b)
        self.assertEqual(sims.shape, (10,))

    def test_cosine_sim_identical_is_one(self):
        a = np.random.rand(5, 8).astype(np.float32)
        sims = self.reuser._cosine_sim_matrix(a, a)
        np.testing.assert_allclose(sims, 1.0, atol=1e-5)

    def test_n_frames_increments(self):
        state = self.reuser.new_state()
        patches = np.random.rand(4, 8).astype(np.float32)
        for _ in range(3):
            self.reuser.process_frame(patches, self._kv_fn, state)
        self.assertEqual(state.n_frames, 3)


# ---------------------------------------------------------------------------
# VLMSpecDecode
# ---------------------------------------------------------------------------
from squish.vision.vlm_spec_decode import (
    VLMSpecConfig,
    VLMSpecState,
    VLMSpecDecode,
)


class TestVLMSpecConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VLMSpecConfig()
        self.assertEqual(cfg.draft_width, 4)
        self.assertEqual(cfg.max_draft_tokens, 8)
        self.assertTrue(cfg.visual_shared)

    def test_invalid_draft_width(self):
        with self.assertRaises(ValueError):
            VLMSpecConfig(draft_width=0)

    def test_invalid_max_draft_tokens(self):
        with self.assertRaises(ValueError):
            VLMSpecConfig(max_draft_tokens=0)


class TestVLMSpecDecode(unittest.TestCase):
    def setUp(self):
        self.cfg = VLMSpecConfig(draft_width=3, max_draft_tokens=4)
        self.spec = VLMSpecDecode(self.cfg)

    def _draft_fn(self, prompt, width):
        return [[i + 100, i + 101] for i in range(width)]

    def _verify_fn(self, full_seq, visual_kv):
        return full_seq[-2:]

    def test_new_state(self):
        state = self.spec.new_state()
        self.assertIsNone(state.visual_kv)
        self.assertEqual(state.n_accepted, 0)

    def test_encode_visual_shape(self):
        tokens = np.random.rand(16, 32).astype(np.float32)
        kv = self.spec.encode_visual(tokens)
        self.assertEqual(kv.shape, (16, 32))

    def test_speculate_returns_list(self):
        state = self.spec.new_state()
        visual_kv = self.spec.encode_visual(np.zeros((4, 8), dtype=np.float32))
        accepted = self.spec.speculate([1, 2, 3], self._draft_fn, self._verify_fn, visual_kv, state)
        self.assertIsInstance(accepted, list)

    def test_acceptance_rate_range(self):
        state = self.spec.new_state()
        visual_kv = self.spec.encode_visual(np.zeros((4, 8), dtype=np.float32))
        self.spec.speculate([1, 2, 3], self._draft_fn, self._verify_fn, visual_kv, state)
        rate = self.spec.acceptance_rate(state)
        self.assertGreaterEqual(rate, 0.0)
        self.assertLessEqual(rate, 1.0)

    def test_reset_clears_counters(self):
        state = self.spec.new_state()
        visual_kv = self.spec.encode_visual(np.zeros((4, 8), dtype=np.float32))
        self.spec.speculate([1, 2, 3], self._draft_fn, self._verify_fn, visual_kv, state)
        self.spec.reset(state)
        self.assertEqual(state.n_accepted, 0)
        self.assertEqual(state.n_rejected, 0)
        self.assertEqual(len(state.acceptance_history), 0)

    def test_reset_retains_visual_kv(self):
        state = self.spec.new_state()
        visual_kv = self.spec.encode_visual(np.ones((4, 8), dtype=np.float32))
        self.spec.speculate([1, 2, 3], self._draft_fn, self._verify_fn, visual_kv, state)
        self.spec.reset(state)
        self.assertIsNotNone(state.visual_kv)

    def test_max_draft_tokens_truncated(self):
        cfg = VLMSpecConfig(draft_width=2, max_draft_tokens=2)
        spec = VLMSpecDecode(cfg)
        calls = []

        def draft(prompt, width):
            return [[99, 100, 101, 102]] * width

        def verify(full_seq, vkv):
            calls.append(full_seq)
            return full_seq[-1:]

        state = spec.new_state()
        vkv = spec.encode_visual(np.zeros((2, 4), dtype=np.float32))
        spec.speculate([1], draft, verify, vkv, state)
        for seq in calls:
            self.assertLessEqual(len(seq) - 1, cfg.max_draft_tokens)

    def test_total_decisions_property(self):
        state = self.spec.new_state()
        visual_kv = self.spec.encode_visual(np.zeros((4, 8), dtype=np.float32))
        self.spec.speculate([1, 2, 3], self._draft_fn, self._verify_fn, visual_kv, state)
        self.assertEqual(state.total_decisions, state.n_accepted + state.n_rejected)


# ---------------------------------------------------------------------------
# VLMBatchScheduler
# ---------------------------------------------------------------------------
from squish.serving.vlm_scheduler import (
    VLMSchedulerConfig,
    VLMRequest,
    VLMBatch,
    VLMBatchScheduler,
)


class TestVLMSchedulerConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VLMSchedulerConfig()
        self.assertEqual(cfg.max_batch_size, 8)

    def test_invalid_low_threshold(self):
        with self.assertRaises(ValueError):
            VLMSchedulerConfig(low_res_threshold=0)

    def test_invalid_high_lte_low(self):
        with self.assertRaises(ValueError):
            VLMSchedulerConfig(low_res_threshold=500, high_res_threshold=400)

    def test_invalid_max_batch_size(self):
        with self.assertRaises(ValueError):
            VLMSchedulerConfig(max_batch_size=0)


class TestVLMRequest(unittest.TestCase):
    def test_auto_request_id(self):
        req = VLMRequest(prompt="test", image_height=224, image_width=224)
        self.assertTrue(len(req.request_id) > 0)

    def test_max_dim(self):
        req = VLMRequest(prompt="test", image_height=200, image_width=400)
        self.assertEqual(req.max_dim, 400)

    def test_is_video_default_false(self):
        req = VLMRequest(prompt="test", image_height=224, image_width=224)
        self.assertFalse(req.is_video)


class TestVLMBatchScheduler(unittest.TestCase):
    def setUp(self):
        self.cfg = VLMSchedulerConfig(
            low_res_threshold=336,
            high_res_threshold=672,
            max_batch_size=4,
        )
        self.sched = VLMBatchScheduler(self.cfg)

    def _req(self, h, w, is_video=False, fps=0.0):
        return VLMRequest(prompt="test", image_height=h, image_width=w,
                          is_video=is_video, fps=fps)

    def test_classify_low(self):
        self.assertEqual(self.sched.classify(self._req(224, 224)), "low")

    def test_classify_mid(self):
        self.assertEqual(self.sched.classify(self._req(500, 400)), "mid")

    def test_classify_high(self):
        self.assertEqual(self.sched.classify(self._req(1024, 768)), "high")

    def test_classify_video_flag(self):
        self.assertEqual(self.sched.classify(self._req(224, 224, is_video=True)), "video")

    def test_classify_video_fps(self):
        self.assertEqual(self.sched.classify(self._req(224, 224, fps=5.0)), "video")

    def test_batch_max_size_respected(self):
        reqs = [self._req(1024, 768) for _ in range(10)]
        batches = self.sched.batch(reqs)
        for b in batches:
            self.assertLessEqual(b.n_requests, 4)

    def test_batch_bucket_homogeneous(self):
        low_reqs = [self._req(224, 224) for _ in range(3)]
        high_reqs = [self._req(1024, 768) for _ in range(3)]
        batches = self.sched.batch(low_reqs + high_reqs)
        for b in batches:
            for req in b.requests:
                self.assertEqual(self.sched.classify(req), b.bucket)

    def test_schedule_high_before_low(self):
        reqs = [self._req(224, 224), self._req(1024, 768)]
        ordered = self.sched.schedule(reqs)
        self.assertEqual(self.sched.classify(ordered[0]), "high")
        self.assertEqual(self.sched.classify(ordered[1]), "low")

    def test_estimated_visual_tokens_positive(self):
        est = self.sched.estimated_visual_tokens(672, 672)
        self.assertGreater(est, 0)

    def test_batch_empty_input(self):
        batches = self.sched.batch([])
        self.assertEqual(len(batches), 0)

    def test_schedule_empty_input(self):
        ordered = self.sched.schedule([])
        self.assertEqual(len(ordered), 0)

    def test_vlm_batch_properties(self):
        reqs = [self._req(1024, 768) for _ in range(2)]
        batches = self.sched.batch(reqs)
        b = batches[0]
        self.assertEqual(b.n_requests, len(b.requests))
        self.assertEqual(b.total_visual_tokens, b.estimated_visual_tokens)


# ---------------------------------------------------------------------------
# ImageEncoderCache
# ---------------------------------------------------------------------------
from squish.vision.img_encoder_cache import (
    ImageEncoderCacheConfig,
    CacheEntry,
    ImageEncoderCache,
)


class TestImageEncoderCacheConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ImageEncoderCacheConfig()
        self.assertEqual(cfg.max_entries, 1000)
        self.assertEqual(cfg.token_dim, 256)

    def test_invalid_max_entries(self):
        with self.assertRaises(ValueError):
            ImageEncoderCacheConfig(max_entries=0)

    def test_invalid_token_dim(self):
        with self.assertRaises(ValueError):
            ImageEncoderCacheConfig(token_dim=0)


class TestImageEncoderCache(unittest.TestCase):
    def setUp(self):
        self.cfg = ImageEncoderCacheConfig(max_entries=5, token_dim=8)
        self.cache = ImageEncoderCache(self.cfg)

    def _tokens(self, n=4):
        return np.random.rand(n, 8).astype(np.float32)

    def test_miss_returns_none(self):
        self.assertIsNone(self.cache.get("nonexistent"))

    def test_put_and_get(self):
        tokens = self._tokens()
        self.cache.put("hash1", tokens)
        result = self.cache.get("hash1")
        np.testing.assert_array_almost_equal(result, tokens)

    def test_hit_count_increments(self):
        tokens = self._tokens()
        self.cache.put("h1", tokens)
        self.cache.get("h1")
        self.cache.get("h1")
        self.assertEqual(self.cache._store["h1"].hit_count, 2)

    def test_stats_hit_rate(self):
        tokens = self._tokens()
        self.cache.put("h1", tokens)
        self.cache.get("h1")
        self.cache.get("nonexistent")
        stats = self.cache.stats()
        self.assertAlmostEqual(stats["hit_rate"], 0.5)

    def test_lru_eviction_on_overflow(self):
        for i in range(5):
            self.cache.put(f"h{i}", self._tokens())
        self.assertEqual(len(self.cache), 5)
        self.cache.put("h_new", self._tokens())
        self.assertEqual(len(self.cache), 5)

    def test_encode_or_cached_cold(self):
        def encoder(h):
            return self._tokens()
        tokens = self.cache.encode_or_cached("img1", encoder)
        self.assertEqual(tokens.shape, (4, 8))

    def test_encode_or_cached_warm(self):
        calls = []
        def encoder(h):
            calls.append(h)
            return self._tokens()
        self.cache.encode_or_cached("img2", encoder)
        self.cache.encode_or_cached("img2", encoder)
        self.assertEqual(len(calls), 1)

    def test_clear(self):
        self.cache.put("h1", self._tokens())
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.assertIsNone(self.cache.get("h1"))

    def test_stats_after_clear(self):
        self.cache.put("h1", self._tokens())
        self.cache.get("h1")
        self.cache.clear()
        stats = self.cache.stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

    def test_contains_dunder(self):
        self.cache.put("h1", self._tokens())
        self.assertIn("h1", self.cache)
        self.assertNotIn("h2", self.cache)

    def test_update_existing_entry(self):
        old = self._tokens()
        self.cache.put("h1", old)
        new_tokens = self._tokens()
        self.cache.put("h1", new_tokens)
        result = self.cache.get("h1")
        np.testing.assert_array_almost_equal(result, new_tokens)

    def test_stats_keys_present(self):
        stats = self.cache.stats()
        for key in ("hits", "misses", "entries", "hit_rate"):
            self.assertIn(key, stats)


if __name__ == "__main__":
    unittest.main()
