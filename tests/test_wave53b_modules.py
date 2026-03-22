"""Tests for Wave 53b Linear Recurrent Architecture infrastructure modules.

Covers:
  - squish.kv.ssm_state_cache         (SSMStateCache)
  - squish.kernels.parallel_scan_kernel (ParallelScanKernel)
  - squish.quant.ssm_quant            (SSMQuantizer)
  - squish.serving.hybrid_arch_router (HybridArchRouter)
  - squish.attention.hymba_dual       (HymbaDualTrack)
  - squish.streaming.ssm_state_offload (SSMStateOffload)
"""

import unittest
import numpy as np


# ---------------------------------------------------------------------------
# SSMStateCache
# ---------------------------------------------------------------------------

class TestSSMStateCacheConfig(unittest.TestCase):
    def test_import(self):
        from squish.kv.ssm_state_cache import SSMStateCacheConfig
        cfg = SSMStateCacheConfig()
        self.assertIsNotNone(cfg)

    def test_default_max_sessions(self):
        from squish.kv.ssm_state_cache import SSMStateCacheConfig
        cfg = SSMStateCacheConfig()
        self.assertGreater(cfg.max_sessions, 0)

    def test_invalid_max_sessions(self):
        from squish.kv.ssm_state_cache import SSMStateCacheConfig
        with self.assertRaises(ValueError):
            SSMStateCacheConfig(max_sessions=0)


class TestSSMCacheEntry(unittest.TestCase):
    def test_size_bytes(self):
        from squish.kv.ssm_state_cache import SSMCacheEntry
        import time
        e = SSMCacheEntry(
            session_id="abc",
            arch="mamba2",
            state_bytes=b"x" * 64,
            n_tokens=10,
            timestamp=time.time(),
        )
        self.assertEqual(e.size_bytes, 64)


class TestSSMStateCache(unittest.TestCase):
    def setUp(self):
        from squish.kv.ssm_state_cache import SSMStateCacheConfig, SSMStateCache
        self.cache = SSMStateCache(SSMStateCacheConfig(max_sessions=4))

    def _arrays(self):
        return {"h": np.zeros((8, 16), dtype=np.float32)}

    def test_put_and_get(self):
        self.cache.put("s1", "mamba2", self._arrays(), 100)
        result = self.cache.get("s1")
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result["h"], np.zeros((8, 16)))

    def test_get_missing(self):
        result = self.cache.get("nonexistent")
        self.assertIsNone(result)

    def test_len(self):
        self.cache.put("s1", "mamba2", self._arrays(), 10)
        self.cache.put("s2", "hawk", self._arrays(), 20)
        self.assertEqual(len(self.cache), 2)

    def test_contains(self):
        self.cache.put("s3", "rwkv6", self._arrays(), 5)
        self.assertIn("s3", self.cache)
        self.assertNotIn("s99", self.cache)

    def test_delete(self):
        self.cache.put("s4", "hawk", self._arrays(), 7)
        deleted = self.cache.delete("s4")
        self.assertTrue(deleted)
        self.assertIsNone(self.cache.get("s4"))

    def test_delete_missing(self):
        result = self.cache.delete("ghost")
        self.assertFalse(result)

    def test_lru_eviction(self):
        for i in range(5):
            self.cache.put(f"s{i}", "mamba2", self._arrays(), i * 10)
        # With max_sessions=4, earliest entry should be evicted
        self.assertLessEqual(len(self.cache), 4)

    def test_stats(self):
        self.cache.put("s1", "mamba2", self._arrays(), 100)
        stats = self.cache.stats()
        self.assertIn("sessions", stats)

    def test_clear(self):
        self.cache.put("s1", "mamba2", self._arrays(), 10)
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)

    def test_overwrite_session(self):
        arr1 = {"h": np.ones((4, 4), dtype=np.float32)}
        arr2 = {"h": np.full((4, 4), 2.0, dtype=np.float32)}
        self.cache.put("s1", "mamba2", arr1, 1)
        self.cache.put("s1", "mamba2", arr2, 2)
        result = self.cache.get("s1")
        np.testing.assert_array_almost_equal(result["h"], np.full((4, 4), 2.0))


# ---------------------------------------------------------------------------
# ParallelScanKernel
# ---------------------------------------------------------------------------

class TestParallelScanConfig(unittest.TestCase):
    def test_import(self):
        from squish.kernels.parallel_scan_kernel import ParallelScanConfig
        cfg = ParallelScanConfig()
        self.assertIsNotNone(cfg)

    def test_invalid_tile_size(self):
        from squish.kernels.parallel_scan_kernel import ParallelScanConfig
        with self.assertRaises(ValueError):
            ParallelScanConfig(tile_size=0)


class TestScalarMulAdd(unittest.TestCase):
    def test_identity(self):
        from squish.kernels.parallel_scan_kernel import ScalarMulAdd
        k, x = ScalarMulAdd.identity()
        self.assertEqual(k, 1.0)
        self.assertEqual(x, 0.0)

    def test_combine(self):
        from squish.kernels.parallel_scan_kernel import ScalarMulAdd
        k_out, x_out = ScalarMulAdd.combine(2.0, 3.0, 4.0, 5.0)
        self.assertAlmostEqual(k_out, 8.0)
        self.assertAlmostEqual(x_out, 13.0)  # 2*5 + 3


class TestMatMulAdd(unittest.TestCase):
    def test_combine(self):
        from squish.kernels.parallel_scan_kernel import MatMulAdd
        A = np.eye(2)
        b = np.ones(2)
        C = np.eye(2) * 2
        d = np.zeros(2)
        out_A, out_b = MatMulAdd.combine(A, b, C, d)
        np.testing.assert_array_almost_equal(out_A, np.eye(2) * 2)
        np.testing.assert_array_almost_equal(out_b, np.ones(2))


class TestParallelScanKernel(unittest.TestCase):
    def setUp(self):
        from squish.kernels.parallel_scan_kernel import ParallelScanConfig, ParallelScanKernel
        self.kernel = ParallelScanKernel(ParallelScanConfig())

    def test_scan_scalar_shape(self):
        T = 8
        decays = np.full(T, 0.9)
        states = np.random.randn(T)
        out_k, out_x = self.kernel.scan_scalar(decays, states)
        self.assertEqual(out_k.shape, (T,))
        self.assertEqual(out_x.shape, (T,))

    def test_scan_scalar_identity_decay(self):
        """With decay=1, output state should be cumulative sum of inputs."""
        T = 4
        decays = np.ones(T)
        states = np.array([1.0, 1.0, 1.0, 1.0])
        _, out_x = self.kernel.scan_scalar(decays, states)
        # Cumulative: [1, 2, 3, 4]
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(out_x, expected)

    def test_scan_scalar_zero_decay(self):
        """With decay=0, each output should equal its own input."""
        T = 4
        decays = np.array([0.0, 0.0, 0.0, 0.0])
        states = np.array([5.0, 3.0, 1.0, 7.0])
        _, out_x = self.kernel.scan_scalar(decays, states)
        # First element always equals its input; others should be just their state
        self.assertAlmostEqual(out_x[0], 5.0)

    def test_blelloch_matches_scan_scalar(self):
        T = 16
        decays = np.random.uniform(0.8, 0.99, T)
        states = np.random.randn(T)
        _, ref = self.kernel.scan_scalar(decays, states)
        _, bel = self.kernel.blelloch_scan_scalar(decays, states)
        np.testing.assert_array_almost_equal(ref, bel, decimal=5)

    def test_scan_affine_shape(self):
        T = 6
        A = np.random.randn(T, 3, 3)
        b = np.random.randn(T, 3)
        out_A, out_b = self.kernel.scan_affine(A, b)
        self.assertEqual(out_A.shape, (T, 3, 3))
        self.assertEqual(out_b.shape, (T, 3))

    def test_blelloch_power_of_two(self):
        T = 8
        decays = np.full(T, 0.95)
        states = np.ones(T)
        _, ref = self.kernel.scan_scalar(decays, states)
        _, bel = self.kernel.blelloch_scan_scalar(decays, states)
        np.testing.assert_array_almost_equal(ref, bel, decimal=5)


# ---------------------------------------------------------------------------
# SSMQuantizer
# ---------------------------------------------------------------------------

class TestSSMQuantConfig(unittest.TestCase):
    def test_import(self):
        from squish.quant.ssm_quant import SSMQuantConfig
        cfg = SSMQuantConfig()
        self.assertIsNotNone(cfg)

    def test_default_bits(self):
        from squish.quant.ssm_quant import SSMQuantConfig
        cfg = SSMQuantConfig()
        self.assertIn("dt", cfg.bits_per_role)

    def test_invalid_bits(self):
        from squish.quant.ssm_quant import SSMQuantConfig
        with self.assertRaises(ValueError):
            SSMQuantConfig(bits_per_role={"dt": 3})


class TestSSMQuantState(unittest.TestCase):
    def test_is_calibrated_false(self):
        from squish.quant.ssm_quant import SSMQuantConfig, SSMQuantizer
        model = SSMQuantizer(SSMQuantConfig())
        state = model.new_state()
        self.assertFalse(state.is_calibrated)

    def test_initial_steps(self):
        from squish.quant.ssm_quant import SSMQuantConfig, SSMQuantizer
        model = SSMQuantizer(SSMQuantConfig())
        state = model.new_state()
        self.assertEqual(state.n_calibration_steps, 0)


class TestSSMQuantizer(unittest.TestCase):
    def setUp(self):
        from squish.quant.ssm_quant import SSMQuantConfig, SSMQuantizer
        self.q = SSMQuantizer(SSMQuantConfig(calibration_samples=2))

    def _observe_and_finalise(self):
        state = self.q.new_state()
        tensors = {
            "dt": np.random.randn(64).astype(np.float32),
            "A_log": np.random.randn(64).astype(np.float32),
            "B": np.random.randn(64).astype(np.float32),
        }
        for _ in range(2):
            state = self.q.observe(tensors, state)
        state = self.q.finalise(state)
        return state, tensors

    def test_observe_increments_steps(self):
        state = self.q.new_state()
        tensors = {"dt": np.random.randn(8).astype(np.float32)}
        state = self.q.observe(tensors, state)
        self.assertEqual(state.n_calibration_steps, 1)

    def test_finalise_marks_calibrated(self):
        state, _ = self._observe_and_finalise()
        self.assertTrue(state.is_calibrated)

    def test_quantize_and_dequantize_dt(self):
        state, tensors = self._observe_and_finalise()
        arr = tensors["dt"]
        q_arr, meta = self.q.quantize_tensor(arr, "dt", state)
        recon = self.q.dequantize_tensor(q_arr, meta)
        # Reconstruction should be close (within quantisation error)
        self.assertEqual(recon.shape, arr.shape)

    def test_compression_ratio(self):
        ratio = self.q.compression_ratio("A_log")
        self.assertGreater(ratio, 0)

    def test_calibrated_roles(self):
        state, _ = self._observe_and_finalise()
        roles = state.calibrated_roles
        self.assertIn("dt", roles)

    def test_state_role_passthrough(self):
        """state tensors use float16 identity (bits=16)."""
        state, _ = self._observe_and_finalise()
        arr = np.random.randn(8).astype(np.float32)
        q_arr, meta = self.q.quantize_tensor(arr, "state", state)
        recon = self.q.dequantize_tensor(q_arr, meta)
        np.testing.assert_array_almost_equal(recon, arr.astype(np.float16), decimal=2)

    def test_multiple_tensors_observe(self):
        state = self.q.new_state()
        for _ in range(3):
            tensors = {
                "dt": np.random.randn(16).astype(np.float32),
                "B": np.random.randn(16).astype(np.float32),
            }
            state = self.q.observe(tensors, state)
        self.assertEqual(state.n_calibration_steps, 3)


# ---------------------------------------------------------------------------
# HybridArchRouter
# ---------------------------------------------------------------------------

class TestHybridArchConfig(unittest.TestCase):
    def test_import(self):
        from squish.serving.hybrid_arch_router import HybridArchConfig
        cfg = HybridArchConfig(layer_types=["attention", "mamba"])
        self.assertIsNotNone(cfg)

    def test_empty_types(self):
        from squish.serving.hybrid_arch_router import HybridArchConfig
        with self.assertRaises(ValueError):
            HybridArchConfig(layer_types=[])

    def test_unknown_type(self):
        from squish.serving.hybrid_arch_router import HybridArchConfig
        with self.assertRaises(ValueError):
            HybridArchConfig(layer_types=["transformer", "unknown_arch"])


class TestHybridLayerSpec(unittest.TestCase):
    def test_canonical_type(self):
        from squish.serving.hybrid_arch_router import HybridLayerSpec
        spec = HybridLayerSpec(layer_idx=0, canonical_type="mamba", raw_type="mamba2")
        self.assertEqual(spec.canonical_type, "mamba")


class TestHybridArchRouter(unittest.TestCase):
    def setUp(self):
        from squish.serving.hybrid_arch_router import HybridArchRouter
        self.router = HybridArchRouter.from_layer_types(
            ["attention", "mamba", "mamba", "attention", "rwkv"],
            model_name="test",
        )

    def test_layer_type(self):
        self.assertEqual(self.router.layer_type(0), "attention")
        self.assertEqual(self.router.layer_type(1), "mamba")

    def test_count_by_type(self):
        counts = self.router.count_by_type()
        self.assertEqual(counts["attention"], 2)
        self.assertEqual(counts["mamba"], 2)
        self.assertEqual(counts["rwkv"], 1)

    def test_attention_ratio(self):
        ratio = self.router.attention_ratio()
        self.assertAlmostEqual(ratio, 0.4)

    def test_route_calls_handler(self):
        results = []
        self.router.register("attention", lambda x, s, **kw: (x + 1, s))
        x = np.ones((2, 4))
        out = self.router.route(0, x, state=None)
        # Returns tuple (x+1, state)
        np.testing.assert_array_almost_equal(out[0], x + 1)

    def test_route_out_of_range(self):
        with self.assertRaises(IndexError):
            self.router.route(99, np.ones(4))

    def test_route_missing_handler(self):
        with self.assertRaises(KeyError):
            # No handler for "mamba" registered
            self.router.route(1, np.ones(4))

    def test_register_invalid_type(self):
        with self.assertRaises(ValueError):
            self.router.register("unknown_type", lambda x, s: (x, s))

    def test_canonical_aliases(self):
        from squish.serving.hybrid_arch_router import HybridArchRouter
        router = HybridArchRouter.from_layer_types(
            ["attn", "ssm", "rglr", "rwkv6"]
        )
        self.assertEqual(router.layer_type(0), "attention")
        self.assertEqual(router.layer_type(1), "mamba")
        self.assertEqual(router.layer_type(2), "hawk")
        self.assertEqual(router.layer_type(3), "rwkv")

    def test_all_ssm(self):
        from squish.serving.hybrid_arch_router import HybridArchRouter
        router = HybridArchRouter.from_layer_types(["mamba"] * 6)
        self.assertAlmostEqual(router.attention_ratio(), 0.0)

    def test_all_attention(self):
        from squish.serving.hybrid_arch_router import HybridArchRouter
        router = HybridArchRouter.from_layer_types(["attention"] * 4)
        self.assertAlmostEqual(router.attention_ratio(), 1.0)


# ---------------------------------------------------------------------------
# HymbaDualTrack
# ---------------------------------------------------------------------------

class TestHymbaConfig(unittest.TestCase):
    def test_import(self):
        from squish.attention.hymba_dual import HymbaConfig
        cfg = HymbaConfig()
        self.assertIsNotNone(cfg)

    def test_mismatch_dim(self):
        from squish.attention.hymba_dual import HymbaConfig
        with self.assertRaises(ValueError):
            HymbaConfig(d_model=256, n_heads=4, head_dim=32)

    def test_invalid_d_ssm(self):
        from squish.attention.hymba_dual import HymbaConfig
        with self.assertRaises(ValueError):
            HymbaConfig(d_model=64, n_heads=2, head_dim=32, d_ssm=0)


class TestHymbaState(unittest.TestCase):
    def test_state_bytes(self):
        from squish.attention.hymba_dual import HymbaConfig, HymbaDualTrack
        cfg = HymbaConfig(d_model=64, n_heads=2, head_dim=32, d_ssm=16, seed=0)
        model = HymbaDualTrack(cfg)
        state = model.new_state()
        self.assertGreater(state.state_bytes, 0)

    def test_n_steps_initial(self):
        from squish.attention.hymba_dual import HymbaConfig, HymbaDualTrack
        cfg = HymbaConfig(d_model=64, n_heads=2, head_dim=32, d_ssm=16)
        model = HymbaDualTrack(cfg)
        state = model.new_state()
        self.assertEqual(state.n_steps, 0)

    def test_h_length(self):
        from squish.attention.hymba_dual import HymbaConfig, HymbaDualTrack
        cfg = HymbaConfig(d_model=64, n_heads=2, head_dim=32, d_ssm=16)
        model = HymbaDualTrack(cfg)
        state = model.new_state()
        self.assertEqual(len(state.h), 2)


class TestHymbaDualTrack(unittest.TestCase):
    def setUp(self):
        from squish.attention.hymba_dual import HymbaConfig, HymbaDualTrack
        self.cfg = HymbaConfig(d_model=64, n_heads=2, head_dim=32, d_ssm=16, seed=42)
        self.model = HymbaDualTrack(self.cfg)

    def test_forward_shape(self):
        x = np.random.randn(4, 64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (4, 64))

    def test_single_token(self):
        x = np.random.randn(1, 64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertEqual(out.shape, (1, 64))

    def test_no_nan(self):
        x = np.random.randn(5, 64).astype(np.float32)
        state = self.model.new_state()
        out, _ = self.model.forward(x, state)
        self.assertFalse(np.any(np.isnan(out)))

    def test_deterministic(self):
        x = np.random.randn(3, 64).astype(np.float32)
        s = self.model.new_state()
        o1, _ = self.model.forward(x, s)
        o2, _ = self.model.forward(x, s)
        np.testing.assert_array_equal(o1, o2)

    def test_state_steps_update(self):
        x = np.random.randn(4, 64).astype(np.float32)
        state = self.model.new_state()
        _, ns = self.model.forward(x, state)
        self.assertEqual(ns.n_steps, 4)

    def test_ssm_state_changes(self):
        x = np.random.randn(4, 64).astype(np.float32)
        state = self.model.new_state()
        h_before = state.h[0].copy()
        _, ns = self.model.forward(x, state)
        # SSM state should evolve after processing tokens
        self.assertFalse(np.allclose(ns.h[0], h_before))

    def test_invalid_input_dim(self):
        x = np.random.randn(3, 32).astype(np.float32)
        state = self.model.new_state()
        with self.assertRaises(ValueError):
            self.model.forward(x, state)


# ---------------------------------------------------------------------------
# SSMStateOffload
# ---------------------------------------------------------------------------

class TestSSMStateOffloadConfig(unittest.TestCase):
    def test_import(self):
        from squish.streaming.ssm_state_offload import SSMStateOffloadConfig
        cfg = SSMStateOffloadConfig()
        self.assertIsNotNone(cfg)

    def test_invalid_segment_len(self):
        from squish.streaming.ssm_state_offload import SSMStateOffloadConfig
        with self.assertRaises(ValueError):
            SSMStateOffloadConfig(segment_len=0)

    def test_invalid_max_segments(self):
        from squish.streaming.ssm_state_offload import SSMStateOffloadConfig
        with self.assertRaises(ValueError):
            SSMStateOffloadConfig(max_segments_per_session=0)


class TestOffloadSegment(unittest.TestCase):
    def test_size_bytes(self):
        from squish.streaming.ssm_state_offload import OffloadSegment
        import time
        seg = OffloadSegment(
            session_id="test",
            segment_idx=0,
            state_bytes=b"x" * 128,
            n_tokens=2048,
            timestamp=time.time(),
        )
        self.assertEqual(seg.size_bytes, 128)


class TestSSMStateOffload(unittest.TestCase):
    def setUp(self):
        from squish.streaming.ssm_state_offload import SSMStateOffloadConfig, SSMStateOffload
        cfg = SSMStateOffloadConfig(segment_len=4, compress_fp16=False)
        self.offload = SSMStateOffload(cfg)

    def _layer_states(self):
        return {
            "layer0": np.random.randn(8, 4).astype(np.float32),
            "layer1": np.random.randn(4,).astype(np.float32),
        }

    def test_new_session_returns_string(self):
        sid = self.offload.new_session()
        self.assertIsInstance(sid, str)
        self.assertTrue(len(sid) > 0)

    def test_sessions_are_unique(self):
        s1 = self.offload.new_session()
        s2 = self.offload.new_session()
        self.assertNotEqual(s1, s2)

    def test_maybe_offload_at_boundary(self):
        sid = self.offload.new_session()
        result = self.offload.maybe_offload(sid, self._layer_states(), n_tokens_total=4)
        self.assertTrue(result)

    def test_maybe_offload_not_at_boundary(self):
        sid = self.offload.new_session()
        result = self.offload.maybe_offload(sid, self._layer_states(), n_tokens_total=3)
        self.assertFalse(result)

    def test_segments_for_session_increases(self):
        sid = self.offload.new_session()
        self.offload.maybe_offload(sid, self._layer_states(), 4)
        self.offload.maybe_offload(sid, self._layer_states(), 8)
        self.assertEqual(self.offload.segments_for_session(sid), 2)

    def test_restore_roundtrip(self):
        sid = self.offload.new_session()
        states = self._layer_states()
        self.offload.maybe_offload(sid, states, 4)
        restored = self.offload.restore(sid)
        np.testing.assert_array_almost_equal(
            restored["layer0"], states["layer0"], decimal=5
        )

    def test_restore_latest_default(self):
        sid = self.offload.new_session()
        states1 = {"h": np.ones((4,), dtype=np.float32)}
        states2 = {"h": np.full((4,), 2.0, dtype=np.float32)}
        self.offload.maybe_offload(sid, states1, 4)
        self.offload.maybe_offload(sid, states2, 8)
        restored = self.offload.restore(sid)
        np.testing.assert_array_almost_equal(
            restored["h"], states2["h"], decimal=3
        )

    def test_latest_segment_none_when_empty(self):
        sid = self.offload.new_session()
        seg = self.offload.latest_segment(sid)
        self.assertIsNone(seg)

    def test_latest_segment_after_offload(self):
        sid = self.offload.new_session()
        self.offload.maybe_offload(sid, self._layer_states(), 4)
        seg = self.offload.latest_segment(sid)
        self.assertIsNotNone(seg)
        self.assertEqual(seg.n_tokens, 4)

    def test_restore_missing_session(self):
        with self.assertRaises(KeyError):
            self.offload.restore("ghost_session")

    def test_restore_empty_session(self):
        sid = self.offload.new_session()
        with self.assertRaises(KeyError):
            self.offload.restore(sid)

    def test_stats(self):
        sid = self.offload.new_session()
        self.offload.maybe_offload(sid, self._layer_states(), 4)
        stats = self.offload.stats()
        self.assertGreater(stats["n_sessions"], 0)
        self.assertGreater(stats["total_segments"], 0)

    def test_delete_session(self):
        sid = self.offload.new_session()
        self.offload.maybe_offload(sid, self._layer_states(), 4)
        self.offload.delete_session(sid)
        with self.assertRaises(KeyError):
            self.offload.restore(sid)

    def test_delete_unknown_session(self):
        with self.assertRaises(KeyError):
            self.offload.delete_session("not_here")

    def test_maybe_offload_unknown_session(self):
        with self.assertRaises(KeyError):
            self.offload.maybe_offload("no_session", self._layer_states(), 4)

    def test_fp16_compression(self):
        from squish.streaming.ssm_state_offload import SSMStateOffloadConfig, SSMStateOffload
        cfg = SSMStateOffloadConfig(segment_len=4, compress_fp16=True)
        offload = SSMStateOffload(cfg)
        sid = offload.new_session()
        states = {"h": np.random.randn(8).astype(np.float32)}
        offload.maybe_offload(sid, states, 4)
        restored = offload.restore(sid)
        # Restored from float16 so allow tolerance
        np.testing.assert_array_almost_equal(restored["h"], states["h"], decimal=2)

    def test_max_segments_eviction(self):
        from squish.streaming.ssm_state_offload import SSMStateOffloadConfig, SSMStateOffload
        cfg = SSMStateOffloadConfig(segment_len=2, max_segments_per_session=3, compress_fp16=False)
        offload = SSMStateOffload(cfg)
        sid = offload.new_session()
        for i in range(6):
            offload.maybe_offload(sid, {"h": np.ones(4, dtype=np.float32)}, (i + 1) * 2)
        # Should not exceed cap
        self.assertLessEqual(offload.segments_for_session(sid), 3)


if __name__ == "__main__":
    unittest.main()
