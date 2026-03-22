"""tests/test_wave54b_modules.py

Unit tests for Wave 54b attention / serving modules:
  - FlashAttn3Kernel
  - DoubleSparsityAttn
  - LASPLinearAttn
  - NaCLCache
  - KVMigrationManager
  - ElasticBatchController
"""

from __future__ import annotations

import unittest

import numpy as np

from squish.kernels.flash_attn3 import FlashAttn3Config, FlashAttn3Kernel
from squish.attention.double_sparse import (
    DoubleSparseConfig,
    DoubleSparseState,
    DoubleSparsityAttn,
)
from squish.attention.lasp_parallel import LASPConfig, LASPRingState, LASPLinearAttn
from squish.kv.nacl_cache import NaCLConfig, NaCLState, NaCLCache
from squish.serving.kv_migration import (
    KVMigrationConfig,
    MigrationRecord,
    KVMigrationManager,
)
from squish.serving.elastic_batching import (
    ElasticBatchConfig,
    ElasticBatchState,
    ElasticBatchController,
)


# ===========================================================================
# FlashAttn3Kernel Tests
# ===========================================================================

class TestFlashAttn3Config(unittest.TestCase):
    def test_defaults(self):
        cfg = FlashAttn3Config()
        self.assertEqual(cfg.block_size, 64)
        self.assertEqual(cfg.pingpong_stages, 2)
        self.assertTrue(cfg.causal)

    def test_invalid_block_size(self):
        with self.assertRaises(ValueError):
            FlashAttn3Config(block_size=0)

    def test_invalid_stages(self):
        with self.assertRaises(ValueError):
            FlashAttn3Config(pingpong_stages=0)


class TestFlashAttn3Kernel(unittest.TestCase):
    def setUp(self):
        self.cfg = FlashAttn3Config(block_size=8, causal=True, seed=0)
        self.kernel = FlashAttn3Kernel(self.cfg)

    def test_forward_output_shape(self):
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((16, 32)).astype(np.float32)
        K = rng.standard_normal((16, 32)).astype(np.float32)
        V = rng.standard_normal((16, 32)).astype(np.float32)
        out, lse = self.kernel.forward(Q, K, V)
        self.assertEqual(out.shape, (16, 32))
        self.assertEqual(lse.shape, (16,))

    def test_forward_non_causal(self):
        cfg = FlashAttn3Config(block_size=8, causal=False)
        kernel = FlashAttn3Kernel(cfg)
        rng = np.random.default_rng(1)
        Q = rng.standard_normal((8, 16)).astype(np.float32)
        K = rng.standard_normal((8, 16)).astype(np.float32)
        V = rng.standard_normal((8, 16)).astype(np.float32)
        out, lse = kernel.forward(Q, K, V)
        self.assertEqual(out.shape, (8, 16))

    def test_forward_output_dtype(self):
        rng = np.random.default_rng(2)
        Q = rng.standard_normal((8, 16)).astype(np.float32)
        K = rng.standard_normal((8, 16)).astype(np.float32)
        V = rng.standard_normal((8, 16)).astype(np.float32)
        out, _ = self.kernel.forward(Q, K, V)
        self.assertEqual(out.dtype, np.float32)

    def test_forward_invalid_dim_mismatch(self):
        Q = np.ones((4, 8), dtype=np.float32)
        K = np.ones((4, 16), dtype=np.float32)  # head_dim mismatch
        V = np.ones((4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.kernel.forward(Q, K, V)

    def test_forward_not_2d_raises(self):
        Q = np.ones((4, 4, 8), dtype=np.float32)
        K = np.ones((4, 8), dtype=np.float32)
        V = np.ones((4, 8), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.kernel.forward(Q, K, V)

    def test_call_alias_works(self):
        rng = np.random.default_rng(3)
        Q = rng.standard_normal((4, 8)).astype(np.float32)
        K = rng.standard_normal((4, 8)).astype(np.float32)
        V = rng.standard_normal((4, 8)).astype(np.float32)
        out1, _ = self.kernel.forward(Q, K, V)
        out2, _ = self.kernel(Q, K, V)
        np.testing.assert_array_equal(out1, out2)

    def test_causal_output_differs_noncausal(self):
        rng = np.random.default_rng(4)
        Q = rng.standard_normal((8, 16)).astype(np.float32)
        K = rng.standard_normal((8, 16)).astype(np.float32)
        V = rng.standard_normal((8, 16)).astype(np.float32)
        k_causal = FlashAttn3Kernel(FlashAttn3Config(block_size=4, causal=True))
        k_full = FlashAttn3Kernel(FlashAttn3Config(block_size=4, causal=False))
        out_c, _ = k_causal.forward(Q, K, V)
        out_f, _ = k_full.forward(Q, K, V)
        # They should differ for non-trivial inputs
        self.assertFalse(np.allclose(out_c, out_f))

    def test_single_token(self):
        rng = np.random.default_rng(5)
        Q = rng.standard_normal((1, 8)).astype(np.float32)
        K = rng.standard_normal((1, 8)).astype(np.float32)
        V = rng.standard_normal((1, 8)).astype(np.float32)
        out, lse = self.kernel.forward(Q, K, V)
        self.assertEqual(out.shape, (1, 8))

    def test_custom_scale(self):
        cfg = FlashAttn3Config(block_size=4, scale=1.0, causal=False)
        k = FlashAttn3Kernel(cfg)
        rng = np.random.default_rng(6)
        Q = rng.standard_normal((4, 8)).astype(np.float32)
        K = rng.standard_normal((4, 8)).astype(np.float32)
        V = rng.standard_normal((4, 8)).astype(np.float32)
        out, _ = k.forward(Q, K, V)
        self.assertEqual(out.shape, (4, 8))


# ===========================================================================
# DoubleSparsityAttn Tests
# ===========================================================================

class TestDoubleSparseConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = DoubleSparseConfig()
        self.assertEqual(cfg.n_heads, 8)
        self.assertEqual(cfg.head_keep_ratio, 0.5)

    def test_invalid_head_keep_ratio_zero(self):
        with self.assertRaises(ValueError):
            DoubleSparseConfig(head_keep_ratio=0.0)

    def test_invalid_head_keep_ratio_exceeds_one(self):
        with self.assertRaises(ValueError):
            DoubleSparseConfig(head_keep_ratio=1.5)

    def test_invalid_token_top_k(self):
        with self.assertRaises(ValueError):
            DoubleSparseConfig(token_top_k=0)

    def test_invalid_calibration_steps(self):
        with self.assertRaises(ValueError):
            DoubleSparseConfig(calibration_steps=0)


class TestDoubleSparsityAttn(unittest.TestCase):
    def setUp(self):
        self.cfg = DoubleSparseConfig(n_heads=4, head_dim=16, token_top_k=4,
                                      head_keep_ratio=0.5, calibration_steps=2, seed=0)
        self.attn = DoubleSparsityAttn(self.cfg)

    def test_new_state_all_heads_active(self):
        s = self.attn.new_state()
        np.testing.assert_array_equal(s.head_mask, np.ones(4))
        self.assertFalse(s.is_calibrated)

    def test_calibrate_increments_step(self):
        s = self.attn.new_state()
        grads = np.ones((4, 8, 16), dtype=np.float32)
        s2 = self.attn.calibrate(grads, s)
        self.assertEqual(s2.n_calibration_steps, 1)

    def test_calibrate_accumulates_importance(self):
        s = self.attn.new_state()
        grads = np.ones((4, 8, 16), dtype=np.float32)
        s = self.attn.calibrate(grads, s)
        s = self.attn.calibrate(grads, s)
        self.assertTrue(np.all(s.head_importance > 0))

    def test_finalise_calibration_prunes_heads(self):
        s = self.attn.new_state()
        grads = np.zeros((4, 8, 16), dtype=np.float32)
        grads[0] = 10.0  # make head 0 most important
        grads[1] = 5.0
        s = self.attn.calibrate(grads, s)
        s = self.attn.finalise_calibration(s)
        n_active = int(s.head_mask.sum())
        self.assertEqual(n_active, 2)  # head_keep_ratio=0.5 of 4 heads
        self.assertTrue(s.is_calibrated)

    def test_forward_output_shape(self):
        s = self.attn.new_state()
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((4, 8, 16)).astype(np.float32)
        K = rng.standard_normal((4, 8, 16)).astype(np.float32)
        V = rng.standard_normal((4, 8, 16)).astype(np.float32)
        out, _ = self.attn.forward(Q, K, V, s)
        self.assertEqual(out.shape, (4, 8, 16))

    def test_forward_pruned_heads_are_zero(self):
        s = self.attn.new_state()
        # Zero importance for heads 2 and 3
        grads = np.zeros((4, 8, 16), dtype=np.float32)
        grads[0] = 1.0
        grads[1] = 1.0
        s = self.attn.calibrate(grads, s)
        s = self.attn.finalise_calibration(s)
        rng = np.random.default_rng(1)
        Q = rng.standard_normal((4, 8, 16)).astype(np.float32)
        K = rng.standard_normal((4, 8, 16)).astype(np.float32)
        V = rng.standard_normal((4, 8, 16)).astype(np.float32)
        out, _ = self.attn.forward(Q, K, V, s)
        # Pruned heads should be zero
        for h in range(4):
            if s.head_mask[h] == 0.0:
                np.testing.assert_array_equal(out[h], np.zeros((8, 16)))

    def test_forward_invalid_n_heads(self):
        s = self.attn.new_state()
        Q = np.ones((2, 8, 16), dtype=np.float32)  # wrong n_heads
        K = np.ones((4, 8, 16), dtype=np.float32)
        V = np.ones((4, 8, 16), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.attn.forward(Q, K, V, s)


# ===========================================================================
# LASPLinearAttn Tests
# ===========================================================================

class TestLASPConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = LASPConfig()
        self.assertEqual(cfg.d_model, 256)

    def test_invalid_n_workers(self):
        with self.assertRaises(ValueError):
            LASPConfig(n_workers=0)

    def test_invalid_n_heads(self):
        with self.assertRaises(ValueError):
            LASPConfig(n_heads=0)

    def test_invalid_d_model(self):
        with self.assertRaises(ValueError):
            LASPConfig(d_model=0)


class TestLASPLinearAttn(unittest.TestCase):
    def setUp(self):
        self.cfg = LASPConfig(d_model=16, n_heads=2, head_dim=8, n_workers=2, seed=0)
        self.model = LASPLinearAttn(self.cfg)

    def test_new_state_zero_recv(self):
        s = self.model.new_state(worker_id=0)
        np.testing.assert_array_equal(s.recv_state, np.zeros((2, 8, 8)))
        self.assertEqual(s.worker_id, 0)
        self.assertEqual(s.n_steps, 0)

    def test_forward_output_shape(self):
        s = self.model.new_state()
        x = np.random.default_rng(0).standard_normal((4, 16)).astype(np.float32)
        out, s2 = self.model.forward(x, s)
        self.assertEqual(out.shape, (4, 16))

    def test_forward_increments_steps(self):
        s = self.model.new_state()
        x = np.ones((4, 16), dtype=np.float32)
        _, s2 = self.model.forward(x, s)
        self.assertEqual(s2.n_steps, 1)

    def test_forward_send_state_shape(self):
        s = self.model.new_state()
        x = np.random.default_rng(1).standard_normal((4, 16)).astype(np.float32)
        _, s2 = self.model.forward(x, s)
        self.assertEqual(s2.recv_state.shape, (2, 8, 8))

    def test_forward_invalid_shape(self):
        s = self.model.new_state()
        x = np.ones((4, 8), dtype=np.float32)  # wrong d_model
        with self.assertRaises(ValueError):
            self.model.forward(x, s)

    def test_ring_step_shapes(self):
        s0 = self.model.new_state()
        x = np.random.default_rng(2).standard_normal((4, 16)).astype(np.float32)
        local_out, send_state = self.model.ring_step(x, s0.recv_state)
        self.assertEqual(local_out.shape, (4, 16))
        self.assertEqual(send_state.shape, (2, 8, 8))

    def test_ring_step_state_changes(self):
        s0 = self.model.new_state()
        x = np.random.default_rng(3).standard_normal((4, 16)).astype(np.float32)
        _, send = self.model.ring_step(x, s0.recv_state)
        self.assertFalse(np.all(send == 0))

    def test_output_dtype(self):
        s = self.model.new_state()
        x = np.ones((4, 16), dtype=np.float32)
        out, _ = self.model.forward(x, s)
        self.assertEqual(out.dtype, np.float32)


# ===========================================================================
# NaCLCache Tests
# ===========================================================================

class TestNaCLConfig(unittest.TestCase):
    def test_invalid_max_budget(self):
        with self.assertRaises(ValueError):
            NaCLConfig(max_budget=0)

    def test_invalid_anchor_recent_sum(self):
        with self.assertRaises(ValueError):
            NaCLConfig(max_budget=4, k_anchor=3, k_recent=3)

    def test_invalid_n_heads(self):
        with self.assertRaises(ValueError):
            NaCLConfig(n_heads=0)

    def test_negative_k_anchor(self):
        with self.assertRaises(ValueError):
            NaCLConfig(k_anchor=-1)

    def test_negative_k_recent(self):
        with self.assertRaises(ValueError):
            NaCLConfig(k_recent=-1)


class TestNaCLCache(unittest.TestCase):
    def setUp(self):
        self.cfg = NaCLConfig(max_budget=8, k_anchor=2, k_recent=2, n_heads=2, head_dim=4, seed=42)
        self.cache = NaCLCache(self.cfg)

    def test_new_state_empty(self):
        s = self.cache.new_state()
        self.assertEqual(s.n_tokens, 0)
        self.assertFalse(s.is_full)
        self.assertAlmostEqual(s.utilization, 0.0)

    def test_update_adds_token(self):
        s = self.cache.new_state()
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        s2 = self.cache.update(k, v, s)
        self.assertEqual(s2.n_tokens, 1)

    def test_update_fills_budget(self):
        s = self.cache.new_state()
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        for _ in range(8):
            s = self.cache.update(k, v, s)
        self.assertTrue(s.is_full)
        self.assertAlmostEqual(s.utilization, 1.0)

    def test_update_evicts_when_full(self):
        s = self.cache.new_state()
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        for _ in range(9):  # 1 over budget
            s = self.cache.update(k, v, s)
        self.assertLessEqual(s.n_tokens, 8)

    def test_get_kv_shapes(self):
        s = self.cache.new_state()
        k = np.ones((2, 4), dtype=np.float32)
        v = np.ones((2, 4), dtype=np.float32)
        s = self.cache.update(k, v, s)
        K_out, V_out = self.cache.get_kv(s)
        self.assertEqual(K_out.shape, (1, 2, 4))
        self.assertEqual(V_out.shape, (1, 2, 4))

    def test_update_invalid_shape(self):
        s = self.cache.new_state()
        k = np.ones((4, 4), dtype=np.float32)  # wrong n_heads
        v = np.ones((2, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            self.cache.update(k, v, s)

    def test_evict_if_needed_when_not_full(self):
        s = self.cache.new_state()
        s2 = self.cache.evict_if_needed(s)
        self.assertEqual(s2.n_tokens, 0)  # unchanged

    def test_get_kv_empty_cache(self):
        s = self.cache.new_state()
        K_out, V_out = self.cache.get_kv(s)
        self.assertEqual(K_out.shape, (0, 2, 4))

    def test_anchor_tokens_preserved(self):
        """First k_anchor tokens should survive eviction."""
        s = self.cache.new_state()
        rng = np.random.default_rng(1)
        sentinel_k = rng.standard_normal((2, 4)).astype(np.float32)
        sentinel_v = rng.standard_normal((2, 4)).astype(np.float32)
        s = self.cache.update(sentinel_k, sentinel_v, s)  # position 0 = anchor
        k = np.zeros((2, 4), dtype=np.float32)
        v = np.zeros((2, 4), dtype=np.float32)
        for _ in range(10):
            s = self.cache.update(k, v, s)
        K_out, _ = self.cache.get_kv(s)
        # First slot should still be sentinel_k
        np.testing.assert_array_almost_equal(K_out[0], sentinel_k)


# ===========================================================================
# KVMigrationManager Tests
# ===========================================================================

class TestKVMigrationConfig(unittest.TestCase):
    def test_invalid_page_size(self):
        with self.assertRaises(ValueError):
            KVMigrationConfig(page_size=0)

    def test_invalid_low_watermark(self):
        with self.assertRaises(ValueError):
            KVMigrationConfig(low_watermark=0.0)

    def test_invalid_low_watermark_too_high(self):
        with self.assertRaises(ValueError):
            KVMigrationConfig(low_watermark=1.0)


class TestMigrationRecord(unittest.TestCase):
    def test_bytes_transferred(self):
        r = MigrationRecord(src_worker="w0", dst_worker="w1",
                            session_id="s0", n_pages=10)
        self.assertEqual(r.bytes_transferred, 10 * 2048)


class TestKVMigrationManager(unittest.TestCase):
    def setUp(self):
        self.cfg = KVMigrationConfig(page_size=16, low_watermark=0.2, seed=0)
        self.mgr = KVMigrationManager(self.cfg)
        self.mgr.register_worker("w0", 100)
        self.mgr.register_worker("w1", 100)

    def test_register_worker(self):
        mgr = KVMigrationManager(self.cfg)
        mgr.register_worker("a", 50)
        self.assertAlmostEqual(mgr.worker_headroom("a"), 1.0)

    def test_register_worker_invalid_capacity(self):
        with self.assertRaises(ValueError):
            self.mgr.register_worker("bad", 0)

    def test_worker_headroom_full(self):
        self.assertAlmostEqual(self.mgr.worker_headroom("w0"), 1.0)

    def test_allocate_pages_success(self):
        ok = self.mgr.allocate_pages("w0", "s1", 10)
        self.assertTrue(ok)
        self.assertAlmostEqual(self.mgr.worker_headroom("w0"), 0.9)

    def test_allocate_pages_fail_overflow(self):
        ok = self.mgr.allocate_pages("w0", "s1", 110)
        self.assertFalse(ok)

    def test_free_pages(self):
        self.mgr.allocate_pages("w0", "s1", 10)
        self.mgr.free_pages("w0", "s1")
        self.assertAlmostEqual(self.mgr.worker_headroom("w0"), 1.0)

    def test_migrate_transfers_pages(self):
        self.mgr.allocate_pages("w0", "sA", 20)
        rec = self.mgr.migrate("w0", "w1", "sA")
        self.assertEqual(rec.n_pages, 20)
        self.assertAlmostEqual(self.mgr.worker_headroom("w0"), 1.0)
        self.assertAlmostEqual(self.mgr.worker_headroom("w1"), 0.8)

    def test_migrate_unknown_session_raises(self):
        with self.assertRaises(KeyError):
            self.mgr.migrate("w0", "w1", "nonexistent")

    def test_migrate_unknown_worker_raises(self):
        with self.assertRaises(KeyError):
            self.mgr.migrate("unknown", "w1", "s1")

    def test_rebalance_returns_records(self):
        self.mgr.allocate_pages("w0", "s1", 90)  # w0 very full
        headrooms = {
            "w0": self.mgr.worker_headroom("w0"),
            "w1": self.mgr.worker_headroom("w1"),
        }
        recs = self.mgr.rebalance(headrooms)
        self.assertIsInstance(recs, list)

    def test_stats_keys(self):
        stats = self.mgr.stats()
        self.assertIn("n_workers", stats)
        self.assertIn("n_migrations", stats)
        self.assertIn("total_pages_migrated", stats)

    def test_worker_headroom_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.mgr.worker_headroom("nope")


# ===========================================================================
# ElasticBatchController Tests
# ===========================================================================

class TestElasticBatchConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = ElasticBatchConfig()
        self.assertEqual(cfg.min_batch, 1)
        self.assertEqual(cfg.max_batch, 64)

    def test_invalid_min_batch(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(min_batch=0)

    def test_max_less_than_min(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(min_batch=10, max_batch=5)

    def test_invalid_watermark_order(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(low_watermark=0.8, high_watermark=0.5)

    def test_watermark_boundary_zero(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(low_watermark=0.0, high_watermark=0.5)

    def test_watermark_boundary_one(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(low_watermark=0.3, high_watermark=1.0)

    def test_invalid_drain_target(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(drain_target=0)

    def test_invalid_grow_step(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(grow_step=0)

    def test_invalid_shrink_step(self):
        with self.assertRaises(ValueError):
            ElasticBatchConfig(shrink_step=0)


class TestElasticBatchController(unittest.TestCase):
    def setUp(self):
        self.cfg = ElasticBatchConfig(
            min_batch=2, max_batch=16,
            low_watermark=0.1, high_watermark=0.8,
            drain_target=4, grow_step=2, shrink_step=2, seed=0,
        )
        self.ctrl = ElasticBatchController(self.cfg)

    def test_new_state(self):
        s = self.ctrl.new_state()
        self.assertGreaterEqual(s.current_batch_size, self.cfg.min_batch)
        self.assertLessEqual(s.current_batch_size, self.cfg.max_batch)

    def test_tick_shrink_on_low_headroom(self):
        s = self.ctrl.new_state()
        s = ElasticBatchState(current_batch_size=8, n_shrinks=0, n_grows=0, n_ticks=0)
        new_bs, s2 = self.ctrl.tick(kv_headroom=0.05, queue_depth=0, state=s)
        self.assertLess(new_bs, 8)
        self.assertEqual(s2.n_shrinks, 1)

    def test_tick_grow_on_high_headroom(self):
        s = ElasticBatchState(current_batch_size=4, n_shrinks=0, n_grows=0, n_ticks=0)
        new_bs, s2 = self.ctrl.tick(kv_headroom=0.9, queue_depth=10, state=s)
        self.assertGreater(new_bs, 4)
        self.assertEqual(s2.n_grows, 1)

    def test_tick_hold_on_middle_headroom(self):
        s = ElasticBatchState(current_batch_size=8, n_shrinks=0, n_grows=0, n_ticks=0)
        new_bs, s2 = self.ctrl.tick(kv_headroom=0.5, queue_depth=10, state=s)
        self.assertEqual(new_bs, 8)
        self.assertEqual(s2.n_shrinks, 0)
        self.assertEqual(s2.n_grows, 0)

    def test_tick_no_grow_when_queue_below_drain_target(self):
        s = ElasticBatchState(current_batch_size=4, n_shrinks=0, n_grows=0, n_ticks=0)
        new_bs, _ = self.ctrl.tick(kv_headroom=0.9, queue_depth=2, state=s)
        self.assertEqual(new_bs, 4)  # queue too small to trigger grow

    def test_tick_clamps_to_max(self):
        s = ElasticBatchState(current_batch_size=15, n_shrinks=0, n_grows=0, n_ticks=0)
        new_bs, _ = self.ctrl.tick(kv_headroom=0.9, queue_depth=100, state=s)
        self.assertLessEqual(new_bs, self.cfg.max_batch)

    def test_tick_clamps_to_min(self):
        s = ElasticBatchState(current_batch_size=3, n_shrinks=0, n_grows=0, n_ticks=0)
        new_bs, _ = self.ctrl.tick(kv_headroom=0.01, queue_depth=0, state=s)
        self.assertGreaterEqual(new_bs, self.cfg.min_batch)

    def test_tick_increments_n_ticks(self):
        s = self.ctrl.new_state()
        _, s2 = self.ctrl.tick(kv_headroom=0.5, queue_depth=0, state=s)
        self.assertEqual(s2.n_ticks, 1)

    def test_recommended_batch_size(self):
        s = ElasticBatchState(current_batch_size=7, n_shrinks=0, n_grows=0, n_ticks=0)
        self.assertEqual(ElasticBatchController.recommended_batch_size(s), 7)

    def test_stats_keys(self):
        s = self.ctrl.new_state()
        stats = ElasticBatchController.stats(s)
        self.assertIn("current_batch_size", stats)
        self.assertIn("n_shrinks", stats)
        self.assertIn("n_grows", stats)
        self.assertIn("n_ticks", stats)

    def test_multiple_shrink_ticks(self):
        s = ElasticBatchState(current_batch_size=10, n_shrinks=0, n_grows=0, n_ticks=0)
        for _ in range(4):
            _, s = self.ctrl.tick(kv_headroom=0.05, queue_depth=0, state=s)
        self.assertEqual(s.n_shrinks, 4)
        self.assertGreaterEqual(s.current_batch_size, self.cfg.min_batch)


if __name__ == "__main__":
    unittest.main()
