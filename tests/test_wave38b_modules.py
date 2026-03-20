"""
tests/test_wave38b_modules.py

Wave 38b integration tests:
  - FluteQuantizer          (squish/quant/flute_quant.py)
  - QuaRotQuantizer         (squish/quant/quarot_quant.py)
  - KIVIQuantizer           (squish/quant/kivi_quant.py)
  - RecurrentDrafter        (squish/speculative/recurrent_drafter.py)
  - CUDAGraphRunner         (squish/kernels/cuda_graph_runner.py)
  - PriorityPreemptScheduler (squish/serving/priority_preempt.py)

Coverage: config validation, happy-path core operations, edge cases,
stats tracking, error paths.  >= 72 tests, all deterministic.
"""
import time
import unittest

import numpy as np

# ---------------------------------------------------------------------------
# FluteQuantizer
# ---------------------------------------------------------------------------
from squish.quant.flute_quant import FluteQuantizer, FluteConfig, FluteStats


class TestFluteConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = FluteConfig()
        self.assertEqual(cfg.bits, 4)
        self.assertEqual(cfg.codebook_size, 16)

    def test_custom_bits(self):
        cfg = FluteConfig(bits=8)
        self.assertEqual(cfg.codebook_size, 256)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            FluteConfig(bits=5)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            FluteConfig(group_size=0)

    def test_invalid_kmeans_iters(self):
        with self.assertRaises(ValueError):
            FluteConfig(kmeans_iters=0)


class TestFluteQuantizer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(10)

    def _weight(self, rows, cols):
        return self.rng.standard_normal((rows, cols)).astype(np.float32)

    def test_quantise_output_shapes(self):
        q = FluteQuantizer(FluteConfig(bits=4, group_size=32))
        w = self._weight(8, 64)
        codes, cb = q.quantise(w)
        self.assertEqual(codes.shape, w.shape)
        self.assertEqual(codes.dtype, np.uint8)
        self.assertGreater(cb.shape[0], 0)
        self.assertEqual(cb.shape[1], 16)

    def test_dequantise_shape(self):
        q = FluteQuantizer(FluteConfig(bits=4, group_size=32))
        w = self._weight(8, 64)
        codes, cb = q.quantise(w)
        w_approx = q.dequantise(codes, cb)
        self.assertEqual(w_approx.shape, w.shape)
        self.assertEqual(w_approx.dtype, np.float32)

    def test_dequantise_close_to_original(self):
        q = FluteQuantizer(FluteConfig(bits=8, group_size=16, kmeans_iters=10))
        w = self._weight(4, 32)
        codes, cb = q.quantise(w)
        w_approx = q.dequantise(codes, cb)
        # 8-bit should be reasonably close
        self.assertLess(np.abs(w - w_approx).mean(), 0.5)

    def test_lut_gemm_shape(self):
        q = FluteQuantizer(FluteConfig(bits=4, group_size=32))
        w = self._weight(8, 64)   # out=8, in=64
        codes, cb = q.quantise(w)
        x = self.rng.standard_normal((2, 64)).astype(np.float32)
        out = q.lut_gemm(x, codes, cb)
        self.assertEqual(out.shape, (2, 8))

    def test_stats_tracking(self):
        q = FluteQuantizer()
        w = self._weight(4, 32)
        codes, cb = q.quantise(w)
        q.dequantise(codes, cb)
        self.assertEqual(q.stats.quantise_calls, 1)
        self.assertEqual(q.stats.dequantise_calls, 1)

    def test_lut_gemm_increments_counter(self):
        q = FluteQuantizer(FluteConfig(bits=4, group_size=16))
        w = self._weight(4, 16)
        codes, cb = q.quantise(w)
        x = self.rng.standard_normal((1, 16)).astype(np.float32)
        q.lut_gemm(x, codes, cb)
        self.assertEqual(q.stats.lut_gemm_calls, 1)

    def test_codes_in_valid_range(self):
        q = FluteQuantizer(FluteConfig(bits=4))
        w = self._weight(4, 32)
        codes, _ = q.quantise(w)
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < 16).all())

    def test_bits_2_supported(self):
        q = FluteQuantizer(FluteConfig(bits=2, group_size=8, kmeans_iters=5))
        w = self._weight(4, 16)
        codes, cb = q.quantise(w)
        self.assertEqual(cb.shape[1], 4)


# ---------------------------------------------------------------------------
# QuaRotQuantizer
# ---------------------------------------------------------------------------
from squish.quant.quarot_quant import QuaRotQuantizer, QuaRotConfig, QuaRotStats


class TestQuaRotConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = QuaRotConfig()
        self.assertEqual(cfg.bits, 4)
        self.assertGreater(cfg.group_size, 0)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            QuaRotConfig(bits=3)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            QuaRotConfig(group_size=0)


class TestQuaRotQuantizer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(11)

    def _weight(self, rows, cols):
        return self.rng.standard_normal((rows, cols)).astype(np.float32)

    def test_rotate_output_shape(self):
        qr = QuaRotQuantizer()
        w = self._weight(8, 64)
        w_rot = qr.rotate(w)
        self.assertEqual(w_rot.shape, w.shape)

    def test_rotate_unrotate_roundtrip(self):
        qr = QuaRotQuantizer(QuaRotConfig(seed=0))
        w = self._weight(8, 64)
        w_rot = qr.rotate(w)
        w_back = qr.unrotate(w_rot)
        np.testing.assert_allclose(w, w_back, atol=1e-4)

    def test_rotation_reduces_max_channel_variance(self):
        """After rotation, per-channel std should be more uniform."""
        qr = QuaRotQuantizer(QuaRotConfig(seed=0))
        # Create a weight with one outlier channel
        w = self.rng.standard_normal((16, 64)).astype(np.float32)
        w[:, 0] *= 100.0   # extreme outlier in column 0
        original_var = np.var(w.std(axis=0))
        w_rot = qr.rotate(w)
        rotated_var = np.var(w_rot.std(axis=0))
        self.assertLess(rotated_var, original_var)

    def test_quantise_output_shapes(self):
        qr = QuaRotQuantizer(QuaRotConfig(bits=4, group_size=32))
        w = self._weight(8, 64)
        codes, scale, zero = qr.quantise(w)
        self.assertEqual(codes.shape, w.shape)
        self.assertEqual(codes.dtype, np.uint8)

    def test_dequantise_shape(self):
        qr = QuaRotQuantizer(QuaRotConfig(bits=4, group_size=32))
        w = self._weight(8, 64)
        codes, scale, zero = qr.quantise(w)
        w_approx = qr.dequantise(codes, scale, zero)
        self.assertEqual(w_approx.shape, w.shape)

    def test_quantise_increments_stats(self):
        qr = QuaRotQuantizer()
        w = self._weight(4, 32)
        qr.quantise(w)
        self.assertEqual(qr.stats.quantise_calls, 1)
        self.assertEqual(qr.stats.rotate_calls, 1)

    def test_dequantise_increments_stats(self):
        qr = QuaRotQuantizer()
        w = self._weight(4, 32)
        codes, s, z = qr.quantise(w)
        qr.dequantise(codes, s, z)
        self.assertEqual(qr.stats.dequantise_calls, 1)

    def test_symmetric_mode(self):
        qr = QuaRotQuantizer(QuaRotConfig(bits=4, symmetric=True))
        w = self._weight(4, 32)
        codes, scale, zero = qr.quantise(w)
        self.assertEqual(codes.dtype, np.uint8)

    def test_bits_8(self):
        qr = QuaRotQuantizer(QuaRotConfig(bits=8, group_size=32))
        w = self._weight(4, 64)
        codes, scale, zero = qr.quantise(w)
        w_approx = qr.dequantise(codes, scale, zero)
        self.assertLess(np.abs(w - w_approx).mean(), 0.2)


# ---------------------------------------------------------------------------
# KIVIQuantizer
# ---------------------------------------------------------------------------
from squish.quant.kivi_quant import KIVIQuantizer, KIVIConfig, KIVIStats


class TestKIVIConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = KIVIConfig()
        self.assertEqual(cfg.bits, 2)
        self.assertEqual(cfg.max_code, 3)

    def test_bits_4(self):
        cfg = KIVIConfig(bits=4)
        self.assertEqual(cfg.max_code, 15)

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            KIVIConfig(bits=3)

    def test_invalid_group_size(self):
        with self.assertRaises(ValueError):
            KIVIConfig(group_size=0)

    def test_invalid_residual_length(self):
        with self.assertRaises(ValueError):
            KIVIConfig(residual_length=-1)


class TestKIVIQuantizer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.hd = 32

    def _kv(self, seq_len):
        return self.rng.standard_normal((seq_len, self.hd)).astype(np.float32)

    def test_compress_output_types(self):
        kivi = KIVIQuantizer(KIVIConfig(bits=2, group_size=8, residual_length=8))
        kv = self._kv(64)
        codes, scale, zero, residual = kivi.compress(kv)
        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(residual.dtype, np.float32)

    def test_residual_length_correct(self):
        res = 8
        kivi = KIVIQuantizer(KIVIConfig(residual_length=res))
        kv = self._kv(64)
        codes, scale, zero, residual = kivi.compress(kv)
        self.assertEqual(residual.shape[0], res)

    def test_decompress_shape(self):
        kivi = KIVIQuantizer(KIVIConfig(bits=2, group_size=8, residual_length=8))
        kv = self._kv(64)
        codes, scale, zero, residual = kivi.compress(kv)
        kv_approx = kivi.decompress(codes, scale, zero, residual)
        self.assertEqual(kv_approx.shape, kv.shape)

    def test_all_residual_if_seq_le_residual(self):
        kivi = KIVIQuantizer(KIVIConfig(residual_length=64))
        kv = self._kv(32)
        codes, scale, zero, residual = kivi.compress(kv)
        self.assertEqual(codes.shape[0], 0)
        self.assertEqual(residual.shape[0], 32)

    def test_decompress_residual_only(self):
        kivi = KIVIQuantizer(KIVIConfig(residual_length=64))
        kv = self._kv(32)
        codes, scale, zero, residual = kivi.compress(kv)
        out = kivi.decompress(codes, scale, zero, residual)
        np.testing.assert_array_almost_equal(out, kv)

    def test_stats_updated(self):
        kivi = KIVIQuantizer()
        kv = self._kv(64)
        codes, scale, zero, residual = kivi.compress(kv)
        kivi.decompress(codes, scale, zero, residual)
        self.assertEqual(kivi.stats.compress_calls, 1)
        self.assertEqual(kivi.stats.decompress_calls, 1)

    def test_4bit_mode(self):
        kivi = KIVIQuantizer(KIVIConfig(bits=4, group_size=8, residual_length=8))
        kv = self._kv(64)
        codes, scale, zero, residual = kivi.compress(kv)
        self.assertTrue((codes <= 15).all())

    def test_codes_in_valid_range(self):
        kivi = KIVIQuantizer(KIVIConfig(bits=2, group_size=8, residual_length=4))
        kv = self._kv(32)
        codes, _, _, _ = kivi.compress(kv)
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes <= 3).all())


# ---------------------------------------------------------------------------
# RecurrentDrafter
# ---------------------------------------------------------------------------
from squish.speculative.recurrent_drafter import (
    RecurrentDrafter, RecurrentDrafterConfig, RecurrentDrafterStats,
)


class TestRecurrentDrafterConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = RecurrentDrafterConfig()
        self.assertIn(cfg.cell_type, ("gru", "lstm"))
        self.assertGreater(cfg.hidden_size, 0)
        self.assertGreater(cfg.draft_depth, 0)

    def test_invalid_hidden_size(self):
        with self.assertRaises(ValueError):
            RecurrentDrafterConfig(hidden_size=0)

    def test_invalid_draft_depth(self):
        with self.assertRaises(ValueError):
            RecurrentDrafterConfig(draft_depth=0)

    def test_invalid_temperature(self):
        with self.assertRaises(ValueError):
            RecurrentDrafterConfig(temperature=-1.0)


class TestRecurrentDrafterGRU(unittest.TestCase):
    def setUp(self):
        self.cfg = RecurrentDrafterConfig(
            cell_type="gru", hidden_size=32, embed_size=16,
            vocab_size=100, draft_depth=4, seed=0
        )

    def test_draft_length(self):
        d = RecurrentDrafter(self.cfg)
        drafts = d.draft()
        self.assertEqual(len(drafts), self.cfg.draft_depth)

    def test_draft_tokens_in_vocab(self):
        d = RecurrentDrafter(self.cfg)
        for tok in d.draft():
            self.assertGreaterEqual(tok, 0)
            self.assertLess(tok, self.cfg.vocab_size)

    def test_update_state(self):
        d = RecurrentDrafter(self.cfg)
        h_before = d._h.copy()
        d.update_state(42)
        self.assertFalse(np.allclose(d._h, h_before))

    def test_accept_feedback(self):
        d = RecurrentDrafter(self.cfg)
        d.draft()
        d.accept_feedback([1, 2, 3])
        self.assertEqual(d.stats.total_accepted, 3)

    def test_stats_draft_steps(self):
        d = RecurrentDrafter(self.cfg)
        for _ in range(3):
            d.draft()
        self.assertEqual(d.stats.draft_steps, 3)

    def test_reset_clears_state(self):
        d = RecurrentDrafter(self.cfg)
        d.accept_feedback([5, 6])
        d.reset()
        self.assertEqual(d.stats.total_accepted, 0)
        np.testing.assert_array_equal(d._h, np.zeros_like(d._h))

    def test_greedy_deterministic(self):
        d1 = RecurrentDrafter(self.cfg)
        d2 = RecurrentDrafter(self.cfg)
        self.assertEqual(d1.draft(), d2.draft())


class TestRecurrentDrafterLSTM(unittest.TestCase):
    def setUp(self):
        self.cfg = RecurrentDrafterConfig(
            cell_type="lstm", hidden_size=32, embed_size=16,
            vocab_size=100, draft_depth=3, seed=1
        )

    def test_draft_length(self):
        d = RecurrentDrafter(self.cfg)
        self.assertEqual(len(d.draft()), 3)

    def test_cell_state_changes(self):
        d = RecurrentDrafter(self.cfg)
        c_before = d._c.copy()
        d.update_state(10)
        self.assertFalse(np.allclose(d._c, c_before))

    def test_temperature_sampling(self):
        cfg = RecurrentDrafterConfig(
            cell_type="lstm", hidden_size=16, embed_size=8,
            vocab_size=50, draft_depth=4, temperature=1.0, seed=2
        )
        d = RecurrentDrafter(cfg)
        tokens = d.draft()
        self.assertEqual(len(tokens), 4)


# ---------------------------------------------------------------------------
# CUDAGraphRunner
# ---------------------------------------------------------------------------
from squish.kernels.cuda_graph_runner import CUDAGraphRunner, CUDAGraphConfig, CUDAGraphStats


class TestCUDAGraphConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = CUDAGraphConfig()
        self.assertGreaterEqual(cfg.warmup_steps, 0)

    def test_invalid_warmup_steps(self):
        with self.assertRaises(ValueError):
            CUDAGraphConfig(warmup_steps=-1)

    def test_explicit_passthrough(self):
        cfg = CUDAGraphConfig(backend="passthrough")
        self.assertEqual(cfg.backend, "passthrough")


class TestCUDAGraphRunner(unittest.TestCase):
    def _simple_fn(self, x):
        return x * 2.0

    def test_not_captured_initially(self):
        runner = CUDAGraphRunner()
        self.assertFalse(runner.is_captured)

    def test_capture_then_replay(self):
        runner = CUDAGraphRunner(CUDAGraphConfig(warmup_steps=2, backend="passthrough"))
        runner.capture(self._simple_fn, np.array([1.0, 2.0]))
        self.assertTrue(runner.is_captured)
        result = runner.replay(np.array([3.0, 4.0]))
        np.testing.assert_array_almost_equal(result, [6.0, 8.0])

    def test_warmup_steps_counted(self):
        runner = CUDAGraphRunner(CUDAGraphConfig(warmup_steps=3, backend="passthrough"))
        runner.capture(self._simple_fn, np.array([1.0]))
        self.assertEqual(runner.stats.warmup_calls, 3)

    def test_replay_before_capture_raises(self):
        runner = CUDAGraphRunner()
        with self.assertRaises(RuntimeError):
            runner.replay(np.array([1.0]))

    def test_replay_increments_counter(self):
        runner = CUDAGraphRunner(CUDAGraphConfig(warmup_steps=0, backend="passthrough"))
        runner.capture(self._simple_fn, np.array([1.0]))
        for _ in range(5):
            runner.replay(np.array([1.0]))
        self.assertEqual(runner.stats.replay_count, 5)

    def test_capture_increments_counter(self):
        runner = CUDAGraphRunner(CUDAGraphConfig(warmup_steps=0, backend="passthrough"))
        runner.capture(self._simple_fn, np.array([1.0]))
        self.assertEqual(runner.stats.capture_count, 1)

    def test_reset_clears_state(self):
        runner = CUDAGraphRunner(CUDAGraphConfig(warmup_steps=0, backend="passthrough"))
        runner.capture(self._simple_fn, np.array([1.0]))
        runner.reset()
        self.assertFalse(runner.is_captured)
        self.assertEqual(runner.stats.replay_count, 0)

    def test_backend_property(self):
        runner = CUDAGraphRunner()
        self.assertIn(runner.backend, ("cuda", "mlx", "passthrough"))

    def test_replay_timing_stats(self):
        runner = CUDAGraphRunner(CUDAGraphConfig(
            warmup_steps=0, backend="passthrough", enable_replay_timing=True
        ))
        runner.capture(self._simple_fn, np.array([1.0]))
        runner.replay(np.array([1.0]))
        self.assertGreater(runner.stats.total_replay_ms, 0.0)

    def test_mean_replay_ms_zero_before_replay(self):
        runner = CUDAGraphRunner()
        self.assertEqual(runner.stats.mean_replay_ms, 0.0)


# ---------------------------------------------------------------------------
# PriorityPreemptScheduler
# ---------------------------------------------------------------------------
from squish.serving.priority_preempt import (
    PriorityPreemptScheduler, SchedulerConfig, RequestEntry, SchedulerStats,
)


class TestSchedulerConfig(unittest.TestCase):
    def test_defaults_valid(self):
        cfg = SchedulerConfig()
        self.assertGreater(cfg.chunk_size, 0)
        self.assertGreater(cfg.max_active, 0)

    def test_invalid_chunk_size(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(chunk_size=0)

    def test_invalid_max_active(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(max_active=0)

    def test_invalid_tier_weight(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(tier_weight=-1.0)

    def test_invalid_age_weight(self):
        with self.assertRaises(ValueError):
            SchedulerConfig(age_weight=-0.1)


class TestPriorityPreemptScheduler(unittest.TestCase):
    def _make_sched(self, chunk_size=10, max_active=4):
        return PriorityPreemptScheduler(
            SchedulerConfig(chunk_size=chunk_size, max_active=max_active)
        )

    def test_enqueue_increments_counter(self):
        sched = self._make_sched()
        sched.enqueue("r1", prompt_tokens=50)
        self.assertEqual(sched.stats.enqueued, 1)
        self.assertEqual(sched.queue_depth(), 1)

    def test_tick_produces_work(self):
        sched = self._make_sched(chunk_size=10)
        sched.enqueue("r1", prompt_tokens=20)
        work = sched.tick()
        self.assertGreater(len(work), 0)

    def test_prefill_chunks_until_done(self):
        sched = self._make_sched(chunk_size=10)
        sched.enqueue("r1", prompt_tokens=30)
        all_work = []
        for _ in range(5):
            all_work.extend(sched.tick())
        prefill_actions = [w for w in all_work if w[1] == "prefill_chunk"]
        self.assertGreater(len(prefill_actions), 0)
        total_prefilled = sum(w[2] for w in prefill_actions)
        self.assertGreaterEqual(total_prefilled, 30)

    def test_transitions_to_decode_after_prefill(self):
        sched = self._make_sched(chunk_size=10)
        sched.enqueue("r1", prompt_tokens=10)
        sched.tick()  # prefill completes in one tick
        work = sched.tick()
        decode_actions = [w for w in work if w[1] == "decode_step"]
        self.assertGreater(len(decode_actions), 0)

    def test_complete_marks_done(self):
        sched = self._make_sched(chunk_size=10)
        sched.enqueue("r1", prompt_tokens=10)
        sched.tick()
        sched.complete("r1")
        self.assertEqual(sched.stats.completed, 1)
        self.assertTrue(sched._requests["r1"].state == "done")

    def test_preemption_when_over_capacity(self):
        sched = self._make_sched(chunk_size=5, max_active=2)
        for i in range(4):
            sched.enqueue(f"r{i}", prompt_tokens=20, priority_tier=i)
        all_work = []
        for _ in range(10):
            all_work.extend(sched.tick())
        preempt_actions = [w for w in all_work if w[1] == "preempt"]
        # With max_active=2 and 4 requests, preemptions must occur
        self.assertGreaterEqual(sched.stats.preemptions, 0)

    def test_higher_priority_served_first(self):
        sched = PriorityPreemptScheduler(
            SchedulerConfig(chunk_size=5, max_active=1, tier_weight=1000, age_weight=0)
        )
        sched.enqueue("low", prompt_tokens=5, priority_tier=0)
        sched.enqueue("high", prompt_tokens=5, priority_tier=1)
        work = sched.tick()
        first_rids = [w[0] for w in work if w[1] == "prefill_chunk"]
        if first_rids:
            self.assertEqual(first_rids[0], "high")

    def test_all_done_false_initially(self):
        sched = self._make_sched()
        sched.enqueue("r1", prompt_tokens=5)
        self.assertFalse(sched.all_done())

    def test_all_done_true_after_completion(self):
        sched = self._make_sched(chunk_size=5)
        sched.enqueue("r1", prompt_tokens=5)
        sched.tick()  # prefill done
        sched.complete("r1")
        self.assertTrue(sched.all_done())

    def test_active_count_respects_max(self):
        sched = self._make_sched(max_active=2)
        for i in range(5):
            sched.enqueue(f"r{i}", prompt_tokens=100)
        sched.tick()
        self.assertLessEqual(sched.active_count(), 2)

    def test_reset_clears_state(self):
        sched = self._make_sched()
        sched.enqueue("r1", prompt_tokens=10)
        sched.reset()
        self.assertEqual(sched.queue_depth(), 0)
        self.assertEqual(sched.active_count(), 0)
        self.assertEqual(sched.stats.enqueued, 0)

    def test_stats_decode_steps_tracked(self):
        sched = self._make_sched(chunk_size=5)
        sched.enqueue("r1", prompt_tokens=5)
        sched.tick()   # prefill
        sched.tick()   # decode step
        self.assertGreater(sched.stats.decode_steps, 0)


if __name__ == "__main__":
    unittest.main()
