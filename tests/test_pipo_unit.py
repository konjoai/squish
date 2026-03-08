"""tests/test_pipo_unit.py — 100% coverage for squish/pipo.py"""
import threading
import time

import numpy as np
import pytest

from squish.pipo import (
    INT4BypassKernel,
    LayerWeightBuffer,
    PIPOConfig,
    PIPOScheduler,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_weight(out=16, in_=8):
    """Return (packed_uint8, scale) — simple fake INT4 weights."""
    # pack: two nibbles per byte
    n_bytes = in_ // 2
    w_u8 = np.full((out, n_bytes), 0x77, dtype=np.uint8)  # both nibbles = 7
    scale = np.ones(in_ // 64 + 1, dtype=np.float32)
    return w_u8, scale


def _simple_loader(n_layers=4, out=16, in_=8):
    """Return a callable that provides INT4 weights per layer."""
    weights = [_make_weight(out, in_) for _ in range(n_layers)]
    def loader(layer_idx):
        return weights[layer_idx]
    return loader


# ---------------------------------------------------------------------------
# PIPOConfig
# ---------------------------------------------------------------------------

class TestPIPOConfig:
    def test_defaults(self):
        cfg = PIPOConfig()
        assert cfg.n_prefetch_layers      == 1
        assert cfg.bypass_batch_threshold == 16
        assert cfg.dequant_cache_size      == 4
        assert cfg.group_size              == 64

    def test_custom(self):
        cfg = PIPOConfig(n_prefetch_layers=2, bypass_batch_threshold=8)
        assert cfg.n_prefetch_layers == 2
        assert cfg.bypass_batch_threshold == 8

    def test_invalid_n_prefetch_layers(self):
        with pytest.raises(ValueError, match="n_prefetch_layers"):
            PIPOConfig(n_prefetch_layers=0)

    def test_invalid_bypass_batch_threshold(self):
        with pytest.raises(ValueError, match="bypass_batch_threshold"):
            PIPOConfig(bypass_batch_threshold=-1)

    def test_invalid_dequant_cache_size(self):
        with pytest.raises(ValueError, match="dequant_cache_size"):
            PIPOConfig(dequant_cache_size=0)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            PIPOConfig(group_size=0)


# ---------------------------------------------------------------------------
# LayerWeightBuffer
# ---------------------------------------------------------------------------

class TestLayerWeightBuffer:
    def test_initial_not_ready(self):
        buf = LayerWeightBuffer()
        assert not buf.is_ready(0)

    def test_invalid_n_slots(self):
        with pytest.raises(ValueError, match="n_slots"):
            LayerWeightBuffer(n_slots=0)

    def test_begin_load_and_is_ready(self):
        buf  = LayerWeightBuffer()
        w, s = _make_weight()
        buf.begin_load(0, w, s)
        assert buf.is_ready(0)

    def test_wait_ready_returns_weights(self):
        buf  = LayerWeightBuffer()
        w, s = _make_weight()
        buf.begin_load(3, w, s)
        got_w, got_s = buf.wait_ready(3)
        np.testing.assert_array_equal(got_w, w)
        np.testing.assert_array_equal(got_s, s)

    def test_release_clears_slot(self):
        buf  = LayerWeightBuffer()
        w, s = _make_weight()
        buf.begin_load(5, w, s)
        buf.release(5)
        assert not buf.is_ready(5)

    def test_release_nonexistent_no_error(self):
        buf = LayerWeightBuffer()
        buf.release(99)  # should not raise

    def test_timeout_raises(self):
        buf = LayerWeightBuffer()
        with pytest.raises(TimeoutError):
            buf.wait_ready(layer_idx=77, timeout=0.01)

    def test_thread_safe_load(self):
        buf = LayerWeightBuffer()
        w, s = _make_weight()

        def load_after_delay():
            time.sleep(0.02)
            buf.begin_load(0, w, s)

        t = threading.Thread(target=load_after_delay)
        t.start()
        got_w, got_s = buf.wait_ready(0, timeout=2.0)
        t.join()
        np.testing.assert_array_equal(got_w, w)


# ---------------------------------------------------------------------------
# INT4BypassKernel
# ---------------------------------------------------------------------------

class TestINT4BypassKernel:
    def _make_weights(self, out=8, in_=16):
        """INT4 packed weights with all-1 nibbles and unit scales."""
        n_bytes = in_ // 2
        w_u8 = np.zeros((out, n_bytes), dtype=np.uint8)
        # Value 0x11 → lo = 1, hi = 1 → both values = 1 (positive nibble)
        w_u8[:] = 0x11
        scale = np.ones(max(in_ // 64, 1), dtype=np.float32)
        return w_u8, scale

    def test_dequantize_shape(self):
        kernel = INT4BypassKernel(cache_size=4, group_size=8)
        w, s = self._make_weights(out=4, in_=8)
        x = np.ones((2, 8), dtype=np.float32)
        out = kernel.matmul(x, w, s, layer_key=0,
                            batch_size=2, bypass_threshold=100)
        assert out.shape == (2, 4)

    def test_matmul_no_cache_miss(self):
        kernel = INT4BypassKernel(cache_size=2, group_size=8)
        w, s = self._make_weights(out=4, in_=8)
        x = np.ones((1, 8), dtype=np.float32)
        out = kernel.matmul(x, w, s, layer_key=1,
                            batch_size=1, bypass_threshold=10)
        # All weights = 1, x = 1 → each output element = sum of 8 ones = 8.0
        assert out.shape == (1, 4)
        assert np.allclose(out, 8.0, atol=1e-3)

    def test_cache_hit_same_result(self):
        kernel = INT4BypassKernel(cache_size=4, group_size=8)
        w, s  = self._make_weights(out=4, in_=8)
        x     = RNG.standard_normal((3, 8)).astype(np.float32)

        out1 = kernel.matmul(x, w, s, layer_key=7,
                             batch_size=1, bypass_threshold=100)
        out2 = kernel.matmul(x, w, s, layer_key=7,
                             batch_size=1, bypass_threshold=100)
        np.testing.assert_allclose(out1, out2)

    def test_no_cache_when_batch_above_threshold(self):
        kernel = INT4BypassKernel(cache_size=4, group_size=8)
        w, s  = self._make_weights(out=4, in_=8)
        x     = np.ones((20, 8), dtype=np.float32)
        # batch_size=20, threshold=10 → no caching
        kernel.matmul(x, w, s, layer_key=0,
                      batch_size=20, bypass_threshold=10)
        assert 0 not in kernel._cache  # not stored

    def test_lru_eviction(self):
        kernel = INT4BypassKernel(cache_size=2, group_size=8)
        w, s  = self._make_weights(out=4, in_=8)
        x     = np.ones((1, 8), dtype=np.float32)
        for key in [0, 1, 2]:       # 3 layers → first evicted after 3rd
            kernel.matmul(x, w, s, layer_key=key,
                          batch_size=1, bypass_threshold=100)
        assert 0 not in kernel._cache   # evicted
        assert 1 in kernel._cache
        assert 2 in kernel._cache

    def test_clear_cache(self):
        kernel = INT4BypassKernel(cache_size=4, group_size=8)
        w, s  = self._make_weights(out=4, in_=8)
        x     = np.ones((1, 8), dtype=np.float32)
        kernel.matmul(x, w, s, layer_key=0, batch_size=1, bypass_threshold=100)
        assert len(kernel._cache) == 1
        kernel.clear_cache()
        assert len(kernel._cache) == 0

    def test_6bit_nibble_values(self):
        """Specific nibble test: 0x8F → lo=15→-1, hi=8→-8."""
        kernel = INT4BypassKernel(cache_size=2, group_size=2)
        # 1 output × 2 input, group_size=2, scale=1
        w = np.array([[0x8F]], dtype=np.uint8)  # shape (1, 1)
        s = np.ones(1, dtype=np.float32)
        x = np.ones((1, 2), dtype=np.float32)
        out = kernel.matmul(x, w, s, layer_key=5, batch_size=1, bypass_threshold=100)
        # lo nibble of 0x8F = 0xF = 15 → 15-16 = -1
        # hi nibble of 0x8F = 0x8 = 8  → 8-16  = -8
        # unpacked via interleave: idx 0 = lo = -1, idx 1 = hi = -8
        # x @ W^T: x=(1,1,-1,-8)? No: shape is (out=1, in=2), x is (1,2)
        # unpacked weights: [[lo0, hi0]] = [[-1, -8]] shape (1,2)
        # x @ W^T = [[1,1]] @ [[-1],[-8]] = -9
        assert out.shape == (1, 1)
        assert np.isclose(out[0, 0], -9.0, atol=1.0)


# ---------------------------------------------------------------------------
# PIPOScheduler
# ---------------------------------------------------------------------------

class TestPIPOScheduler:
    def _make_scheduler(self, n_layers=4, out=16, in_=8):
        cfg    = PIPOConfig(n_prefetch_layers=1, bypass_batch_threshold=4, group_size=8)
        loader = _simple_loader(n_layers=n_layers, out=out, in_=in_)
        return PIPOScheduler(cfg, loader, n_layers=n_layers)

    def test_run_layer_output_shape(self):
        sched = self._make_scheduler(n_layers=4, out=16, in_=8)
        x     = np.ones((1, 8), dtype=np.float32)
        out   = sched.run_layer(0, x)
        assert out.shape == (1, 16)

    def test_run_all_layers(self):
        sched = self._make_scheduler(n_layers=4, out=16, in_=8)
        x     = np.ones((1, 8), dtype=np.float32)
        for i in range(4):
            x = sched.run_layer(i, x[:, :8])
        assert x is not None

    def test_throughput_positive_after_runs(self):
        sched = self._make_scheduler()
        x     = np.ones((2, 8), dtype=np.float32)
        sched.run_layer(0, x)
        assert sched.throughput_tps > 0.0

    def test_throughput_zero_before_runs(self):
        cfg    = PIPOConfig()
        loader = _simple_loader(n_layers=2)
        sched  = PIPOScheduler(cfg, loader, n_layers=2)
        assert sched.throughput_tps == 0.0

    def test_prefetch_async_does_not_block(self):
        sched = self._make_scheduler()
        t0    = time.time()
        sched.prefetch_async(1)
        elapsed = time.time() - t0
        assert elapsed < 0.5  # should be nearly instant

    def test_prefetch_async_out_of_range_noop(self):
        sched = self._make_scheduler(n_layers=2)
        # layer 99 out of range — should not raise
        sched._prefetch_layer(99)

    def test_run_layer_after_prefetch(self):
        cfg    = PIPOConfig(n_prefetch_layers=1, bypass_batch_threshold=1, group_size=8)
        loader = _simple_loader(n_layers=4)
        sched  = PIPOScheduler(cfg, loader, n_layers=4)
        sched.prefetch_async(0)
        time.sleep(0.05)
        x = np.ones((1, 8), dtype=np.float32)
        out = sched.run_layer(0, x)
        assert out.shape[0] == 1

    def test_sync_fallback_when_buffer_not_ready(self):
        cfg    = PIPOConfig(n_prefetch_layers=1, bypass_batch_threshold=100, group_size=8)
        loader = _simple_loader(n_layers=4)
        sched  = PIPOScheduler(cfg, loader, n_layers=4)
        # Do not prefetch — should fall back to sync load
        x   = np.ones((1, 8), dtype=np.float32)
        out = sched.run_layer(2, x)
        assert out is not None

    def test_prefetch_exception_is_swallowed(self):
        """_prefetch_layer() except Exception: pass (line 381) when loader raises."""
        def bad_loader(idx):
            raise RuntimeError("simulated load failure")

        cfg   = PIPOConfig(n_prefetch_layers=1, group_size=8)
        sched = PIPOScheduler(cfg, bad_loader, n_layers=2)
        # _prefetch_layer() should not propagate exceptions
        sched._prefetch_layer(0)  # exception must be swallowed silently
        # buffer remains empty (begin_load was never called)
        assert not sched._buffer.is_ready(0)
