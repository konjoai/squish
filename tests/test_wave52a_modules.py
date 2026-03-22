"""Tests for Wave 52a VLM-efficiency modules.

Covers: FastVPruner, VisionZip, LLaVAPruMerge, TokenPacker, FlashVStream,
DynamicResEncoder.

Run:
    python -m pytest tests/test_wave52a_modules.py -v
"""

import unittest

import numpy as np


# ---------------------------------------------------------------------------
# FastVPruner
# ---------------------------------------------------------------------------
from squish.vision.fast_v import FastVConfig, FastVPruneResult, FastVPruner


class TestFastVConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = FastVConfig()
        self.assertAlmostEqual(cfg.keep_ratio, 0.5)
        self.assertEqual(cfg.prune_layer, 2)
        self.assertEqual(cfg.min_keep, 1)
        self.assertEqual(cfg.score_aggregation, "mean")

    def test_invalid_keep_ratio_zero(self):
        with self.assertRaises(ValueError):
            FastVConfig(keep_ratio=0.0)

    def test_invalid_keep_ratio_gt_one(self):
        with self.assertRaises(ValueError):
            FastVConfig(keep_ratio=1.1)

    def test_invalid_prune_layer_negative(self):
        with self.assertRaises(ValueError):
            FastVConfig(prune_layer=-1)

    def test_invalid_min_keep_zero(self):
        with self.assertRaises(ValueError):
            FastVConfig(min_keep=0)

    def test_valid_max_aggregation(self):
        cfg = FastVConfig(score_aggregation="max")
        self.assertEqual(cfg.score_aggregation, "max")


class TestFastVPruner2D(unittest.TestCase):
    def setUp(self):
        self.cfg = FastVConfig(keep_ratio=0.5, min_keep=1)
        self.pruner = FastVPruner(self.cfg)

    def test_prune_2d_shape(self):
        rng = np.random.default_rng(0)
        attn = rng.random((10, 20)).astype(np.float32)
        result = self.pruner.prune(attn, n_visual=20)
        self.assertEqual(result.kept_indices.shape[0] + result.pruned_indices.shape[0], 20)

    def test_prune_2d_keep_count(self):
        rng = np.random.default_rng(1)
        attn = rng.random((8, 16)).astype(np.float32)
        result = self.pruner.prune(attn, n_visual=16)
        self.assertEqual(result.keep_count, 8)     # ceil(16 * 0.5) = 8

    def test_prune_result_properties(self):
        rng = np.random.default_rng(2)
        attn = rng.random((5, 12)).astype(np.float32)
        result = self.pruner.prune(attn, n_visual=12)
        self.assertEqual(result.keep_count + result.prune_count, 12)

    def test_actual_keep_ratio(self):
        rng = np.random.default_rng(3)
        attn = rng.random((4, 10)).astype(np.float32)
        result = self.pruner.prune(attn, n_visual=10)
        self.assertGreater(result.actual_keep_ratio, 0.0)
        self.assertLessEqual(result.actual_keep_ratio, 1.0)

    def test_apply_output_shape(self):
        rng = np.random.default_rng(4)
        tokens = rng.random((20, 64)).astype(np.float32)
        attn = rng.random((8, 20)).astype(np.float32)
        out_tokens, result = self.pruner.apply(tokens, attn)
        self.assertEqual(out_tokens.shape[0], result.keep_count)
        self.assertEqual(out_tokens.shape[1], 64)

    def test_min_keep_enforced(self):
        pruner = FastVPruner(FastVConfig(keep_ratio=0.01, min_keep=5))
        attn = np.ones((2, 6), dtype=np.float32)
        result = pruner.prune(attn, n_visual=6)
        self.assertGreaterEqual(result.keep_count, 5)

    def test_compression_ratio(self):
        result = self.pruner.prune(np.ones((4, 10), dtype=np.float32), n_visual=10)
        cr = self.pruner.compression_ratio(15)
        self.assertGreater(cr, 0.0)


class TestFastVPruner3D(unittest.TestCase):
    def test_prune_3d_shape(self):
        cfg = FastVConfig(keep_ratio=0.6, score_aggregation="max")
        pruner = FastVPruner(cfg)
        rng = np.random.default_rng(5)
        attn = rng.random((8, 10, 20)).astype(np.float32)
        result = pruner.prune(attn, n_visual=20)
        self.assertEqual(result.kept_indices.shape[0] + result.pruned_indices.shape[0], 20)

    def test_mean_aggregation_3d(self):
        cfg = FastVConfig(keep_ratio=0.5, score_aggregation="mean")
        pruner = FastVPruner(cfg)
        attn = np.random.rand(4, 6, 12).astype(np.float32)
        result = pruner.prune(attn, n_visual=12)
        self.assertEqual(result.scores.shape[0], 12)


# ---------------------------------------------------------------------------
# VisionZip
# ---------------------------------------------------------------------------
from squish.vision.vision_zip import VisionZipConfig, VisionZipResult, VisionZip


class TestVisionZipConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = VisionZipConfig()
        self.assertAlmostEqual(cfg.dominant_ratio, 0.1)
        self.assertAlmostEqual(cfg.contextual_keep_ratio, 0.1)

    def test_invalid_dominant_ratio(self):
        with self.assertRaises(ValueError):
            VisionZipConfig(dominant_ratio=0.0)

    def test_invalid_contextual_ratio(self):
        with self.assertRaises(ValueError):
            VisionZipConfig(contextual_keep_ratio=1.5)

    def test_invalid_min_tokens(self):
        with self.assertRaises(ValueError):
            VisionZipConfig(min_tokens=0)


class TestVisionZip(unittest.TestCase):
    def setUp(self):
        self.cfg = VisionZipConfig(dominant_ratio=0.2, contextual_keep_ratio=0.3)
        self.vz = VisionZip(self.cfg)

    def test_compress_total_leq_input(self):
        rng = np.random.default_rng(10)
        attn = rng.random((50,)).astype(np.float32)
        result = self.vz.compress(attn)
        self.assertLessEqual(len(result.kept_indices), 50)

    def test_compress_dominant_nonempty(self):
        attn = np.random.rand(40).astype(np.float32)
        result = self.vz.compress(attn)
        self.assertGreater(len(result.dominant_indices), 0)

    def test_apply_shape(self):
        rng = np.random.default_rng(11)
        tokens = rng.random((40, 128)).astype(np.float32)
        attn = rng.random((40,)).astype(np.float32)
        out, result = self.vz.apply(tokens, attn)
        self.assertEqual(out.shape[0], len(result.kept_indices))
        self.assertEqual(out.shape[1], 128)

    def test_compression_ratio_leq_one(self):
        attn = np.random.rand(30).astype(np.float32)
        result = self.vz.compress(attn)
        self.assertLessEqual(result.compression_ratio, 1.0)

    def test_min_tokens_enforced(self):
        cfg = VisionZipConfig(dominant_ratio=0.01, contextual_keep_ratio=0.01, min_tokens=5)
        vz = VisionZip(cfg)
        attn = np.ones(20, dtype=np.float32)
        result = vz.compress(attn)
        self.assertGreaterEqual(len(result.kept_indices), 5)

    def test_no_duplicate_kept_indices(self):
        attn = np.random.rand(50).astype(np.float32)
        result = self.vz.compress(attn)
        self.assertEqual(len(result.kept_indices), len(set(result.kept_indices.tolist())))


# ---------------------------------------------------------------------------
# LLaVAPruMerge
# ---------------------------------------------------------------------------
from squish.vision.llava_prumerge import (
    LLaVAPruMergeConfig,
    LLaVAPruMergeResult,
    LLaVAPruMerge,
)


class TestLLaVAPruMergeConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = LLaVAPruMergeConfig()
        self.assertEqual(cfg.n_clusters, 64)
        self.assertTrue(cfg.adaptive)

    def test_invalid_n_clusters(self):
        with self.assertRaises(ValueError):
            LLaVAPruMergeConfig(n_clusters=0)

    def test_invalid_position_weight(self):
        with self.assertRaises(ValueError):
            LLaVAPruMergeConfig(position_weight=1.5)


class TestLLaVAPruMerge(unittest.TestCase):
    def setUp(self):
        self.cfg = LLaVAPruMergeConfig(n_clusters=8, km_iters=5, adaptive=False)
        self.merger = LLaVAPruMerge(self.cfg)

    def test_merge_output_shape(self):
        rng = np.random.default_rng(20)
        keys = rng.random((64, 32)).astype(np.float32)
        pos = rng.random((64, 2)).astype(np.float32)
        result = self.merger.merge(keys, pos)
        self.assertEqual(result.merged_tokens.shape[1], 32)
        self.assertEqual(result.n_clusters_used, len(np.unique(result.cluster_labels)))

    def test_cluster_labels_range(self):
        rng = np.random.default_rng(21)
        keys = rng.random((32, 16)).astype(np.float32)
        pos = rng.random((32, 2)).astype(np.float32)
        result = self.merger.merge(keys, pos)
        self.assertTrue(np.all(result.cluster_labels >= 0))
        self.assertTrue(np.all(result.cluster_labels < self.cfg.n_clusters))

    def test_compression_ratio(self):
        rng = np.random.default_rng(22)
        keys = rng.random((64, 16)).astype(np.float32)
        pos = rng.random((64, 2)).astype(np.float32)
        result = self.merger.merge(keys, pos)
        self.assertLessEqual(result.compression_ratio, 1.0)
        self.assertGreater(result.compression_ratio, 0.0)

    def test_position_weight_zero(self):
        cfg = LLaVAPruMergeConfig(n_clusters=4, position_weight=0.0, km_iters=3, adaptive=False)
        merger = LLaVAPruMerge(cfg)
        keys = np.random.rand(16, 8).astype(np.float32)
        pos = np.zeros((16, 2), dtype=np.float32)
        result = merger.merge(keys, pos)
        self.assertIsNotNone(result)

    def test_fewer_keys_than_clusters(self):
        cfg = LLaVAPruMergeConfig(n_clusters=32, km_iters=5, adaptive=False)
        merger = LLaVAPruMerge(cfg)
        keys = np.random.rand(8, 16).astype(np.float32)
        pos = np.random.rand(8, 2).astype(np.float32)
        result = merger.merge(keys, pos)
        self.assertLessEqual(result.n_clusters_used, 8)


# ---------------------------------------------------------------------------
# TokenPacker
# ---------------------------------------------------------------------------
from squish.vision.token_packer import TokenPackerConfig, TokenPackerResult, TokenPacker


class TestTokenPackerConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TokenPackerConfig()
        self.assertEqual(cfg.n_anchor, 64)

    def test_invalid_n_anchor(self):
        with self.assertRaises(ValueError):
            TokenPackerConfig(n_anchor=0)

    def test_invalid_hidden_not_divisible(self):
        with self.assertRaises(ValueError):
            TokenPackerConfig(hidden_dim=63, n_heads=4)


class TestTokenPacker(unittest.TestCase):
    def setUp(self):
        self.cfg = TokenPackerConfig(n_anchor=16, hidden_dim=32, n_heads=4, seed=42)
        self.packer = TokenPacker(self.cfg)

    def test_pack_output_shape(self):
        patches = np.random.rand(64, 32).astype(np.float32)
        result = self.packer.pack(patches)
        self.assertEqual(result.packed.shape, (16, 32))

    def test_attn_weights_shape(self):
        patches = np.random.rand(64, 32).astype(np.float32)
        result = self.packer.pack(patches)
        self.assertEqual(result.attn_weights.shape, (4, 16, 64))

    def test_result_properties(self):
        patches = np.random.rand(50, 32).astype(np.float32)
        result = self.packer.pack(patches)
        self.assertEqual(result.n_anchor, 16)
        self.assertEqual(result.n_patches_in, 50)

    def test_set_anchors(self):
        new_anchors = np.random.rand(16, 32).astype(np.float32)
        self.packer.set_anchors(new_anchors)
        patches = np.random.rand(32, 32).astype(np.float32)
        result = self.packer.pack(patches)
        self.assertEqual(result.packed.shape, (16, 32))

    def test_attn_weights_sum_to_one(self):
        patches = np.random.rand(32, 32).astype(np.float32)
        result = self.packer.pack(patches)
        row_sums = result.attn_weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)

    def test_single_patch(self):
        patches = np.random.rand(1, 32).astype(np.float32)
        result = self.packer.pack(patches)
        self.assertEqual(result.packed.shape, (16, 32))


# ---------------------------------------------------------------------------
# FlashVStream
# ---------------------------------------------------------------------------
from squish.vision.flash_vstream import (
    FlashVStreamConfig,
    FrameEntry,
    FlashVStreamState,
    FlashVStream,
)


class TestFlashVStreamConfig(unittest.TestCase):
    def test_config_defaults(self):
        cfg = FlashVStreamConfig()
        self.assertEqual(cfg.sensory_window, 8)
        self.assertEqual(cfg.temporal_capacity, 32)

    def test_invalid_sensory_window(self):
        with self.assertRaises(ValueError):
            FlashVStreamConfig(sensory_window=0)

    def test_invalid_temporal_capacity(self):
        with self.assertRaises(ValueError):
            FlashVStreamConfig(temporal_capacity=0)


class TestFlashVStream(unittest.TestCase):
    def setUp(self):
        self.cfg = FlashVStreamConfig(
            sensory_window=3,
            temporal_capacity=5,
            token_dim=16,
            saliency_low_threshold=0.3,
        )
        self.streamer = FlashVStream(self.cfg)

    def _make_frame(self, n=4, saliency=0.5):
        return np.random.rand(n, 16).astype(np.float32), saliency

    def test_new_state(self):
        state = self.streamer.new_state()
        self.assertIsNone(state.spatial)
        self.assertEqual(len(state.temporal), 0)
        self.assertEqual(len(state.sensory), 0)

    def test_ingest_first_frame(self):
        state = self.streamer.new_state()
        kv, sal = self._make_frame()
        self.streamer.ingest(kv, sal, state)
        self.assertIsNotNone(state.spatial)

    def test_sensory_window_capped(self):
        state = self.streamer.new_state()
        for i in range(10):
            kv, sal = self._make_frame(saliency=0.9)
            self.streamer.ingest(kv, sal, state)
        self.assertLessEqual(len(state.sensory), self.cfg.sensory_window)

    def test_temporal_capacity_capped(self):
        state = self.streamer.new_state()
        for i in range(20):
            kv, sal = self._make_frame(saliency=0.9)
            self.streamer.ingest(kv, sal, state)
        self.assertLessEqual(len(state.temporal), self.cfg.temporal_capacity)

    def test_get_kv_shape(self):
        state = self.streamer.new_state()
        for i in range(4):
            kv, sal = self._make_frame(n=4, saliency=0.8)
            self.streamer.ingest(kv, sal, state)
        k, v = self.streamer.get_kv(state)
        total = state.total_tokens
        self.assertEqual(k.shape[0], total)

    def test_frame_entry_properties(self):
        kv = np.zeros((5, 16), dtype=np.float32)
        entry = FrameEntry(frame_idx=0, kv=kv, saliency=0.5)
        self.assertEqual(entry.n_tokens, 5)

    def test_memory_stats_keys(self):
        state = self.streamer.new_state()
        stats = self.streamer.memory_stats(state)
        self.assertIn("n_frames_seen", stats)
        self.assertIn("total_tokens", stats)

    def test_low_saliency_eviction(self):
        state = self.streamer.new_state()
        for i in range(6):
            sal = 0.1 if i % 2 == 0 else 0.9
            kv, _ = self._make_frame(saliency=sal)
            self.streamer.ingest(kv, sal, state)
        self.assertLessEqual(len(state.temporal), self.cfg.temporal_capacity)


# ---------------------------------------------------------------------------
# DynamicResEncoder
# ---------------------------------------------------------------------------
from squish.vision.dynamic_resolution import (
    DynamicResConfig,
    TileLayout,
    DynamicResResult,
    DynamicResEncoder,
)


class TestDynamicResConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DynamicResConfig()
        self.assertEqual(cfg.tile_size, 336)
        self.assertEqual(cfg.max_tiles, 12)
        self.assertTrue(cfg.include_summary)

    def test_invalid_tile_size(self):
        with self.assertRaises(ValueError):
            DynamicResConfig(tile_size=0)

    def test_invalid_max_tiles(self):
        with self.assertRaises(ValueError):
            DynamicResConfig(max_tiles=0)

    def test_invalid_min_tiles(self):
        with self.assertRaises(ValueError):
            DynamicResConfig(min_tiles=0)

    def test_min_gt_max(self):
        with self.assertRaises(ValueError):
            DynamicResConfig(min_tiles=10, max_tiles=5)


class TestTileLayout(unittest.TestCase):
    def test_n_tiles(self):
        layout = TileLayout(n_rows=3, n_cols=4, image_height=1008, image_width=1344)
        self.assertEqual(layout.n_tiles, 12)

    def test_aspect_ratio(self):
        layout = TileLayout(n_rows=2, n_cols=2, image_height=672, image_width=672)
        self.assertAlmostEqual(layout.aspect_ratio, 1.0)


class TestDynamicResEncoder(unittest.TestCase):
    def setUp(self):
        self.cfg = DynamicResConfig(tile_size=336, max_tiles=4, token_dim=32, include_summary=True)
        self.enc = DynamicResEncoder(self.cfg)

    def test_plan_layout_square(self):
        layout = self.enc.plan_layout(672, 672)
        self.assertLessEqual(layout.n_tiles, 4)
        self.assertGreaterEqual(layout.n_tiles, 1)

    def test_plan_layout_portrait(self):
        layout = self.enc.plan_layout(1024, 512)
        self.assertGreaterEqual(layout.n_rows, 1)

    def test_plan_layout_landscape(self):
        layout = self.enc.plan_layout(512, 1024)
        self.assertGreaterEqual(layout.n_cols, 1)

    def test_encode_token_count(self):
        def stub_enc(n, d):
            return np.zeros((n, d), dtype=np.float32)
        result = self.enc.encode(672, 672, stub_enc)
        self.assertGreater(result.total_tokens, 0)

    def test_encode_no_summary(self):
        cfg = DynamicResConfig(tile_size=336, max_tiles=4, token_dim=32, include_summary=False)
        enc = DynamicResEncoder(cfg)
        def stub_enc(n, d):
            return np.zeros((n, d), dtype=np.float32)
        result = enc.encode(336, 336, stub_enc)
        self.assertEqual(result.n_summary_tokens, 0)

    def test_encode_result_properties(self):
        def stub_enc(n, d):
            return np.zeros((n, d), dtype=np.float32)
        result = self.enc.encode(672, 672, stub_enc)
        self.assertEqual(result.total_tokens, result.n_summary_tokens + result.n_tile_tokens)

    def test_min_tiles_enforced(self):
        layout = self.enc.plan_layout(10, 10)
        self.assertGreaterEqual(layout.n_tiles, self.cfg.min_tiles)

    def test_tokens_array_dim(self):
        def stub_enc(n, d):
            return np.zeros((n, d), dtype=np.float32)
        result = self.enc.encode(336, 336, stub_enc)
        self.assertEqual(result.tokens.shape[1], 32)


if __name__ == "__main__":
    unittest.main()
