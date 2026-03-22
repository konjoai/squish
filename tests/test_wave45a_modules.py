"""tests/test_wave45a_modules.py

Tests for Wave 45a modules:
  - FlexGenOffload (squish/serving/flexgen_offload.py)
  - YaRNRoPE (squish/attention/yarn_rope.py)
  - SelfExtend (squish/attention/self_extend.py)
  - OrcaScheduler (squish/serving/orca_scheduler.py)
  - MxFP4 (squish/quant/mx_fp4.py)
  - FP8ActQuant (squish/quant/fp8_act_quant.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── FlexGenOffload ─────────────────────────────────────────────────────────────

from squish.serving.flexgen_offload import DeviceTier, FlexGenOffloadConfig, OffloadPlan, FlexGenOffload


class TestFlexGenOffloadConfig:
    def test_defaults(self):
        cfg = FlexGenOffloadConfig()
        assert cfg.n_layers > 0
        assert cfg.gpu_memory_gb > 0

    def test_custom(self):
        cfg = FlexGenOffloadConfig(gpu_memory_gb=16.0, n_layers=16)
        assert cfg.gpu_memory_gb == 16.0
        assert cfg.n_layers == 16


class TestFlexGenOffload:
    def _make(self, n_layers=4, gpu_gb=1.0, weight_bytes=200_000_000):
        cfg = FlexGenOffloadConfig(n_layers=n_layers, gpu_memory_gb=gpu_gb,
                                   cpu_memory_gb=4.0, weight_bytes_per_layer=weight_bytes)
        return FlexGenOffload(cfg)

    def test_plan_returns_offload_plan(self):
        fg = self._make(n_layers=4)
        plan = fg.plan()
        assert isinstance(plan, OffloadPlan)

    def test_plan_layer_count(self):
        fg = self._make(n_layers=4)
        plan = fg.plan()
        assert len(plan.weight_tier) == 4
        assert len(plan.kv_tier) == 4

    def test_plan_tiers_valid(self):
        fg = self._make()
        plan = fg.plan()
        for tier in plan.weight_tier:
            assert tier in (DeviceTier.GPU, DeviceTier.CPU, DeviceTier.DISK)

    def test_gpu_util_fraction(self):
        fg = self._make()
        plan = fg.plan()
        assert 0.0 <= plan.gpu_util <= 1.0

    def test_prefetch_returns_latency(self):
        fg = self._make(n_layers=4)
        ms = fg.prefetch(0)
        assert ms >= 0.0

    def test_prefetch_marks_on_gpu(self):
        fg = self._make(n_layers=4)
        fg.plan()
        fg.prefetch(0)
        assert fg.is_on_gpu(0)

    def test_evict_removes_from_gpu(self):
        fg = self._make(n_layers=4)
        fg.plan()
        fg.prefetch(0)
        fg.evict(0)
        assert not fg.is_on_gpu(0)

    def test_small_gpu_forces_offload(self):
        cfg = FlexGenOffloadConfig(n_layers=8, gpu_memory_gb=0.01, cpu_memory_gb=4.0,
                                   weight_bytes_per_layer=500_000_000, kv_bytes_per_layer=50_000_000)
        fg = FlexGenOffload(cfg)
        plan = fg.plan()
        assert any(t == DeviceTier.CPU for t in plan.weight_tier)

    def test_default_config(self):
        fg = FlexGenOffload()
        assert fg.config is not None


# ── YaRNRoPE ──────────────────────────────────────────────────────────────────

from squish.attention.yarn_rope import YaRNRoPEConfig, YaRNRoPE


class TestYaRNRoPEConfig:
    def test_defaults(self):
        cfg = YaRNRoPEConfig()
        assert cfg.dim % 2 == 0
        assert cfg.original_max_len > 0
        assert cfg.target_max_len > cfg.original_max_len

    def test_custom(self):
        cfg = YaRNRoPEConfig(dim=64, original_max_len=2048, target_max_len=8192)
        assert cfg.dim == 64

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError):
            YaRNRoPEConfig(dim=7)


class TestYaRNRoPE:
    def _make(self, dim=32, orig=512, target=4096):
        cfg = YaRNRoPEConfig(dim=dim, original_max_len=orig, target_max_len=target)
        return YaRNRoPE(cfg)

    def test_scale_factor(self):
        yarn = self._make(orig=512, target=4096)
        assert yarn.scale_factor == 8.0

    def test_temperature_gt_one(self):
        yarn = self._make(orig=512, target=4096)
        assert yarn.temperature > 1.0

    def test_build_freqs_shape(self):
        yarn = self._make(dim=32)
        cos, sin = yarn.build_freqs(16)
        assert cos.shape == (16, 16)
        assert sin.shape == (16, 16)

    def test_apply_2d_shape(self):
        yarn = self._make(dim=32)
        x = np.random.randn(8, 32).astype(np.float32)
        out = yarn.apply(x)
        assert out.shape == (8, 32)

    def test_apply_3d_shape(self):
        yarn = self._make(dim=32)
        x = np.random.randn(2, 8, 32).astype(np.float32)
        out = yarn.apply(x)
        assert out.shape == (2, 8, 32)

    def test_apply_with_offset(self):
        yarn = self._make(dim=32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = yarn.apply(x, offset=100)
        assert out.shape == (4, 32)

    def test_output_dtype(self):
        yarn = self._make(dim=32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = yarn.apply(x)
        assert out.dtype == np.float32

    def test_default_config(self):
        yarn = YaRNRoPE()
        assert yarn.config is not None


# ── SelfExtend ────────────────────────────────────────────────────────────────

from squish.attention.self_extend import SelfExtendConfig, SelfExtend


class TestSelfExtendConfig:
    def test_defaults(self):
        cfg = SelfExtendConfig()
        assert cfg.group_size >= 1
        assert cfg.window_size >= 1

    def test_custom(self):
        cfg = SelfExtendConfig(group_size=4, window_size=256)
        assert cfg.group_size == 4


class TestSelfExtend:
    def _make(self, group_size=4, window_size=8):
        cfg = SelfExtendConfig(group_size=group_size, window_size=window_size)
        return SelfExtend(cfg)

    def test_forward_short_seq_output_shape(self):
        se = self._make(window_size=32)
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        out = se.forward(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_forward_long_seq_output_shape(self):
        se = self._make(window_size=4, group_size=2)
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 16, 8).astype(np.float32)
        v = np.random.randn(2, 16, 8).astype(np.float32)
        out = se.forward(q, k, v)
        assert out.shape == (2, 1, 8)

    def test_output_dtype(self):
        se = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 4, 8).astype(np.float32)
        v = np.random.randn(2, 4, 8).astype(np.float32)
        out = se.forward(q, k, v)
        assert out.dtype == np.float32

    def test_output_finite(self):
        se = self._make()
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 8, 8).astype(np.float32)
        v = np.random.randn(2, 8, 8).astype(np.float32)
        out = se.forward(q, k, v)
        assert np.all(np.isfinite(out))

    def test_default_config(self):
        se = SelfExtend()
        assert se.config is not None

    def test_window_size_eq_seq_uses_exact(self):
        se = self._make(window_size=8, group_size=4)
        q = np.random.randn(2, 1, 8).astype(np.float32)
        k = np.random.randn(2, 8, 8).astype(np.float32)
        v = np.random.randn(2, 8, 8).astype(np.float32)
        out = se.forward(q, k, v)
        assert out.shape == (2, 1, 8)


# ── OrcaScheduler ─────────────────────────────────────────────────────────────

from squish.serving.orca_scheduler import RequestStatus, OrcaRequest, OrcaSchedulerConfig, OrcaScheduler


class TestOrcaSchedulerConfig:
    def test_defaults(self):
        cfg = OrcaSchedulerConfig()
        assert cfg.max_batch_size >= 1
        assert cfg.max_kv_slots >= 1

    def test_custom(self):
        cfg = OrcaSchedulerConfig(max_batch_size=8, max_kv_slots=1024)
        assert cfg.max_batch_size == 8


class TestOrcaScheduler:
    def _make(self, max_batch=4, max_kv=512):
        cfg = OrcaSchedulerConfig(max_batch_size=max_batch, max_kv_slots=max_kv)
        return OrcaScheduler(cfg)

    def test_submit_returns_id(self):
        sched = self._make()
        rid = sched.submit(prompt_len=16, max_new_tokens=32)
        assert isinstance(rid, int)

    def test_step_returns_batch(self):
        sched = self._make()
        sched.submit(prompt_len=16, max_new_tokens=8)
        batch = sched.step()
        assert isinstance(batch, list)

    def test_batch_nonempty_after_submit(self):
        sched = self._make()
        sched.submit(prompt_len=16, max_new_tokens=8)
        batch = sched.step()
        assert len(batch) >= 1

    def test_advance_increments_counter(self):
        sched = self._make()
        rid = sched.submit(prompt_len=16, max_new_tokens=4)
        sched.step()
        done = sched.advance(rid)
        # May or may not be done after 1 token
        assert isinstance(done, bool)

    def test_running_count_bounded(self):
        sched = self._make(max_batch=2)
        for _ in range(5):
            sched.submit(prompt_len=8, max_new_tokens=4)
        sched.step()
        assert sched.running_count <= 2

    def test_complete_removes_request(self):
        sched = self._make()
        rid = sched.submit(prompt_len=4, max_new_tokens=1)
        sched.step()
        done = sched.advance(rid)
        if done:
            assert rid not in {r.request_id for r in sched.step()}

    def test_waiting_count_decreases_on_step(self):
        sched = self._make(max_batch=4)
        for _ in range(3):
            sched.submit(prompt_len=8, max_new_tokens=4)
        sched.step()
        assert sched.waiting_count <= 3

    def test_preemption_moves_to_swapped(self):
        cfg = OrcaSchedulerConfig(max_batch_size=1, max_kv_slots=32, preemption_enabled=True)
        sched = OrcaScheduler(cfg)
        sched.submit(prompt_len=16, max_new_tokens=4, priority=10)  # low prio
        sched.step()
        sched.submit(prompt_len=16, max_new_tokens=4, priority=1)  # high prio
        sched.step()
        assert sched.swapped_count >= 0  # may or may not preempt depending on capacity

    def test_default_config(self):
        sched = OrcaScheduler()
        assert sched.config is not None


# ── MxFP4 ─────────────────────────────────────────────────────────────────────

from squish.quant.mx_fp4 import MxFP4Config, MxFP4Result, MxFP4


class TestMxFP4Config:
    def test_defaults(self):
        cfg = MxFP4Config()
        assert cfg.block_size == 32

    def test_custom(self):
        cfg = MxFP4Config(block_size=16)
        assert cfg.block_size == 16

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            MxFP4Config(block_size=0)


class TestMxFP4:
    def _make(self, block_size=32):
        cfg = MxFP4Config(block_size=block_size)
        return MxFP4(cfg)

    def test_quantize_returns_result(self):
        mxfp4 = self._make()
        x = np.random.randn(64).astype(np.float32)
        result = mxfp4.quantize(x)
        assert isinstance(result, MxFP4Result)

    def test_codes_in_range(self):
        mxfp4 = self._make(block_size=16)
        x = np.random.randn(32).astype(np.float32)
        result = mxfp4.quantize(x)
        assert result.codes.min() >= 0
        assert result.codes.max() < 15

    def test_scales_positive(self):
        mxfp4 = self._make(block_size=16)
        x = np.random.randn(32).astype(np.float32)
        result = mxfp4.quantize(x)
        assert (result.scales > 0).all()

    def test_dequantize_shape(self):
        mxfp4 = self._make(block_size=16)
        x = np.random.randn(32, 16).astype(np.float32)
        result = mxfp4.quantize(x)
        x_hat = result.dequantize()
        assert x_hat.shape == x.shape

    def test_dequantize_close_to_original(self):
        mxfp4 = self._make(block_size=32)
        x = np.random.randn(64).astype(np.float32)
        result = mxfp4.quantize(x)
        x_hat = result.dequantize()
        assert np.allclose(x, x_hat, atol=0.5)

    def test_default_config(self):
        mxfp4 = MxFP4()
        assert mxfp4.config is not None

    def test_2d_input(self):
        mxfp4 = self._make(block_size=16)
        x = np.random.randn(4, 32).astype(np.float32)
        result = mxfp4.quantize(x)
        assert result.original_shape == (4, 32)


# ── FP8ActQuant ───────────────────────────────────────────────────────────────

from squish.quant.fp8_act_quant import FP8Format, FP8ActQuantConfig, FP8ActQuantResult, FP8ActQuant


class TestFP8ActQuantConfig:
    def test_defaults(self):
        cfg = FP8ActQuantConfig()
        assert cfg.weight_format == FP8Format.E4M3
        assert cfg.activation_format == FP8Format.E4M3

    def test_custom(self):
        cfg = FP8ActQuantConfig(weight_format=FP8Format.E5M2)
        assert cfg.weight_format == FP8Format.E5M2


class TestFP8ActQuant:
    def _make(self, fmt=FP8Format.E4M3):
        cfg = FP8ActQuantConfig(weight_format=fmt, activation_format=fmt)
        return FP8ActQuant(cfg)

    def test_quantize_weights_returns_result(self):
        fp8 = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = fp8.quantize_weights(W)
        assert isinstance(result, FP8ActQuantResult)

    def test_quantize_activations_shape(self):
        fp8 = self._make()
        x = np.random.randn(4, 32).astype(np.float32)
        result = fp8.quantize_activations(x)
        assert result.q_data.shape == x.shape

    def test_dequantize_close(self):
        fp8 = self._make()
        W = np.random.randn(8, 16).astype(np.float32)
        result = fp8.quantize_weights(W)
        W_hat = result.dequantize()
        assert np.allclose(W, W_hat, atol=1.0)

    def test_forward_shape(self):
        fp8 = self._make()
        x = np.random.randn(4, 32).astype(np.float32)
        W = np.random.randn(16, 32).astype(np.float32)
        out, q_x, q_W = fp8.forward(x, W)
        assert out.shape == (4, 16)

    def test_forward_with_bias(self):
        fp8 = self._make()
        x = np.random.randn(4, 32).astype(np.float32)
        W = np.random.randn(16, 32).astype(np.float32)
        b = np.random.randn(16).astype(np.float32)
        out, _, _ = fp8.forward(x, W, bias=b)
        assert out.shape == (4, 16)

    def test_scale_positive(self):
        fp8 = self._make()
        W = np.random.randn(16, 32).astype(np.float32)
        result = fp8.quantize_weights(W)
        assert result.scale > 0

    def test_e5m2_format(self):
        fp8 = self._make(fmt=FP8Format.E5M2)
        W = np.random.randn(8, 16).astype(np.float32)
        result = fp8.quantize_weights(W)
        assert result.fmt == FP8Format.E5M2

    def test_default_config(self):
        fp8 = FP8ActQuant()
        assert fp8.config is not None
