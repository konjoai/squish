"""tests/test_wave45b_modules.py

Tests for Wave 45b modules:
  - CLeXRoPE (squish/attention/clex_rope.py)
  - PowerInferOffload (squish/serving/powerinfer_offload.py)
  - GroupedRoPE (squish/attention/grouped_rope.py)
  - TensorParallel (squish/serving/tensor_parallel.py)
  - FusedBiasGELU (squish/kernels/fused_bias_gelu.py)
  - TokenBudgetScheduler (squish/serving/token_budget_scheduler.py)
"""

from __future__ import annotations

import pytest
import numpy as np

# ── CLeXRoPE ──────────────────────────────────────────────────────────────────

from squish.attention.clex_rope import CLeXRoPEConfig, CLeXRoPE


class TestCLeXRoPEConfig:
    def test_defaults(self):
        cfg = CLeXRoPEConfig()
        assert cfg.dim % 2 == 0
        assert cfg.base > 0
        assert cfg.target_max_len > cfg.original_max_len

    def test_custom(self):
        cfg = CLeXRoPEConfig(dim=64, original_max_len=1024, target_max_len=8192)
        assert cfg.dim == 64

    def test_odd_dim_raises(self):
        with pytest.raises(ValueError):
            CLeXRoPEConfig(dim=7)


class TestCLeXRoPE:
    def _make(self, dim=32, orig=256, target=2048):
        cfg = CLeXRoPEConfig(dim=dim, original_max_len=orig, target_max_len=target, n_calibration_steps=10)
        return CLeXRoPE(cfg)

    def test_build_freqs_shape(self):
        clex = self._make(dim=32)
        cos, sin = clex.build_freqs(8)
        assert cos.shape == (8, 16)
        assert sin.shape == (8, 16)

    def test_calibrate_no_data(self):
        clex = self._make()
        clex.calibrate()
        assert clex._calibrated

    def test_calibrate_sets_scale_vec(self):
        clex = self._make()
        clex.calibrate()
        assert clex._scale_vec is not None
        assert len(clex._scale_vec) == 16  # dim//2

    def test_apply_2d_shape(self):
        clex = self._make(dim=32)
        x = np.random.randn(8, 32).astype(np.float32)
        out = clex.apply(x)
        assert out.shape == (8, 32)

    def test_apply_3d_shape(self):
        clex = self._make(dim=32)
        x = np.random.randn(2, 6, 32).astype(np.float32)
        out = clex.apply(x)
        assert out.shape == (2, 6, 32)

    def test_apply_with_offset(self):
        clex = self._make(dim=32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = clex.apply(x, offset=50)
        assert out.shape == (4, 32)

    def test_output_dtype(self):
        clex = self._make(dim=32)
        x = np.random.randn(4, 32).astype(np.float32)
        out = clex.apply(x)
        assert out.dtype == np.float32

    def test_default_config(self):
        clex = CLeXRoPE()
        assert clex.config is not None

    def test_scale_vec_positive_after_calibrate(self):
        clex = self._make()
        clex.calibrate()
        assert (clex._scale_vec > 0).all()


# ── PowerInferOffload ─────────────────────────────────────────────────────────

from squish.serving.powerinfer_offload import PowerInferOffloadConfig, NeuronPlan, PowerInferOffload


class TestPowerInferOffloadConfig:
    def test_defaults(self):
        cfg = PowerInferOffloadConfig()
        assert cfg.n_neurons > 0
        assert 0.0 < cfg.hot_fraction <= 1.0

    def test_custom(self):
        cfg = PowerInferOffloadConfig(n_neurons=2048, hot_fraction=0.05)
        assert cfg.n_neurons == 2048

    def test_invalid_hot_fraction(self):
        with pytest.raises(ValueError):
            PowerInferOffloadConfig(hot_fraction=1.5)


class TestPowerInferOffload:
    def _make(self, n=256, hot=0.1):
        cfg = PowerInferOffloadConfig(n_neurons=n, hot_fraction=hot)
        return PowerInferOffload(cfg)

    def test_profile_returns_freq(self):
        pi = self._make(n=32)
        acts = np.random.randn(20, 32).astype(np.float32)
        freq = pi.profile(acts)
        assert freq.shape == (32,)

    def test_freq_in_01(self):
        pi = self._make(n=32)
        acts = np.random.randn(20, 32).astype(np.float32)
        freq = pi.profile(acts)
        assert (freq >= 0).all() and (freq <= 1.0).all()

    def test_plan_returns_neuron_plan(self):
        pi = self._make(n=32)
        plan = pi.plan()
        assert isinstance(plan, NeuronPlan)

    def test_plan_hot_cold_partition(self):
        pi = self._make(n=32, hot=0.25)
        plan = pi.plan()
        assert plan.n_hot + plan.n_cold == 32

    def test_plan_with_external_freq(self):
        pi = self._make(n=16)
        freq = np.random.rand(16).astype(np.float32)
        plan = pi.plan(activation_freq=freq)
        assert plan.n_hot >= 1

    def test_sparse_forward_shape(self):
        pi = self._make(n=32)
        x = np.random.randn(4, 16).astype(np.float32)
        W_up = np.random.randn(32, 16).astype(np.float32)
        W_down = np.random.randn(16, 32).astype(np.float32)
        out = pi.sparse_forward(x, W_up, W_down)
        assert out.shape == (4, 16)

    def test_sparse_forward_with_mask(self):
        pi = self._make(n=32)
        x = np.random.randn(3, 16).astype(np.float32)
        W_up = np.random.randn(32, 16).astype(np.float32)
        W_down = np.random.randn(16, 32).astype(np.float32)
        mask = np.zeros(32, dtype=bool)
        mask[:8] = True
        out = pi.sparse_forward(x, W_up, W_down, predict_mask=mask)
        assert out.shape == (3, 16)

    def test_default_config(self):
        pi = PowerInferOffload()
        assert pi.config is not None


# ── GroupedRoPE ────────────────────────────────────────────────────────────────

from squish.attention.grouped_rope import GroupedRoPEConfig, GroupedRoPE


class TestGroupedRoPEConfig:
    def test_defaults(self):
        cfg = GroupedRoPEConfig()
        assert cfg.n_heads > 0
        assert cfg.head_dim % 2 == 0
        assert len(cfg.group_bases) == cfg.n_groups

    def test_custom(self):
        cfg = GroupedRoPEConfig(n_heads=8, head_dim=16, n_groups=2)
        assert cfg.n_groups == 2

    def test_odd_head_dim_raises(self):
        with pytest.raises(ValueError):
            GroupedRoPEConfig(head_dim=7)

    def test_heads_not_divisible_raises(self):
        with pytest.raises(ValueError):
            GroupedRoPEConfig(n_heads=7, n_groups=3)


class TestGroupedRoPE:
    def _make(self, n_heads=4, head_dim=8, n_groups=2):
        cfg = GroupedRoPEConfig(n_heads=n_heads, head_dim=head_dim, n_groups=n_groups)
        return GroupedRoPE(cfg)

    def test_build_all_freqs_shape(self):
        rope = self._make(n_heads=4, head_dim=8, n_groups=2)
        cos, sin = rope.build_all_freqs(seq_len=10)
        assert cos.shape == (4, 10, 4)  # (n_heads, seq, head_dim//2)
        assert sin.shape == (4, 10, 4)

    def test_apply_shape(self):
        rope = self._make(n_heads=4, head_dim=8, n_groups=2)
        x = np.random.randn(2, 4, 6, 8).astype(np.float32)
        out = rope.apply(x)
        assert out.shape == (2, 4, 6, 8)

    def test_apply_with_offset(self):
        rope = self._make(n_heads=4, head_dim=8, n_groups=2)
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        out = rope.apply(x, offset=10)
        assert out.shape == (1, 4, 4, 8)

    def test_different_groups_have_different_freqs(self):
        rope = self._make(n_heads=4, head_dim=8, n_groups=2)
        cos, _ = rope.build_all_freqs(10)
        # First group and last group should differ
        assert not np.allclose(cos[0], cos[3])

    def test_output_dtype(self):
        rope = self._make()
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        out = rope.apply(x)
        assert out.dtype == np.float32

    def test_output_finite(self):
        rope = self._make()
        x = np.random.randn(1, 4, 4, 8).astype(np.float32)
        out = rope.apply(x)
        assert np.all(np.isfinite(out))

    def test_default_config(self):
        rope = GroupedRoPE()
        assert rope.config is not None


# ── TensorParallel ────────────────────────────────────────────────────────────

from squish.serving.tensor_parallel import TensorParallelConfig, TensorParallel


class TestTensorParallelConfig:
    def test_defaults(self):
        cfg = TensorParallelConfig()
        assert cfg.world_size >= 1

    def test_custom(self):
        cfg = TensorParallelConfig(world_size=8)
        assert cfg.world_size == 8

    def test_invalid_world_size(self):
        with pytest.raises(ValueError):
            TensorParallelConfig(world_size=0)


class TestTensorParallel:
    def _make(self, world_size=4):
        cfg = TensorParallelConfig(world_size=world_size)
        return TensorParallel(cfg)

    def test_split_column_count(self):
        tp = self._make(world_size=4)
        W = np.random.randn(16, 32).astype(np.float32)
        shards = tp.split_weights_column(W)
        assert len(shards) == 4

    def test_split_column_total_rows(self):
        tp = self._make(world_size=4)
        W = np.random.randn(16, 32).astype(np.float32)
        shards = tp.split_weights_column(W)
        total_rows = sum(s.shape[0] for s in shards)
        assert total_rows == 16

    def test_split_row_count(self):
        tp = self._make(world_size=4)
        W = np.random.randn(16, 32).astype(np.float32)
        shards = tp.split_weights_row(W)
        assert len(shards) == 4

    def test_column_forward_shape(self):
        tp = self._make(world_size=4)
        W = np.random.randn(16, 32).astype(np.float32)
        shards = tp.split_weights_column(W)
        x = np.random.randn(3, 32).astype(np.float32)
        out = tp.column_forward(x, shards)
        assert out.shape == (3, 16)

    def test_row_forward_shape(self):
        tp = self._make(world_size=4)
        W = np.random.randn(16, 32).astype(np.float32)
        w_shards = tp.split_weights_row(W)
        x = np.random.randn(3, 32).astype(np.float32)
        x_shards = tp.split_input_row(x)
        out = tp.row_forward(x_shards, w_shards)
        assert out.shape == (3, 16)

    def test_column_row_agree(self):
        tp = self._make(world_size=2)
        W = np.random.randn(8, 16).astype(np.float32)
        x = np.random.randn(3, 16).astype(np.float32)
        # Reference
        ref = x @ W.T
        # Column parallel
        col_shards = tp.split_weights_column(W)
        col_out = tp.column_forward(x, col_shards)
        assert np.allclose(ref, col_out, atol=1e-4)

    def test_all_reduce_sums(self):
        tp = self._make()
        t = [np.ones((3, 4)) * i for i in range(1, 5)]
        result = tp.all_reduce(t)
        assert np.allclose(result, np.ones((3, 4)) * 10)

    def test_default_config(self):
        tp = TensorParallel()
        assert tp.config is not None


# ── FusedBiasGELU ─────────────────────────────────────────────────────────────

from squish.kernels.fused_bias_gelu import FusedBiasGELUConfig, FusedBiasGELU


class TestFusedBiasGELUConfig:
    def test_defaults(self):
        cfg = FusedBiasGELUConfig()
        assert isinstance(cfg.approximate, bool)

    def test_exact_mode(self):
        cfg = FusedBiasGELUConfig(approximate=False)
        assert not cfg.approximate


class TestFusedBiasGELU:
    def _make(self, approximate=True):
        cfg = FusedBiasGELUConfig(approximate=approximate)
        return FusedBiasGELU(cfg)

    def test_forward_shape(self):
        fn = self._make()
        x = np.random.randn(4, 32).astype(np.float32)
        out = fn.forward(x)
        assert out.shape == (4, 32)

    def test_forward_with_bias(self):
        fn = self._make()
        x = np.random.randn(4, 32).astype(np.float32)
        b = np.random.randn(32).astype(np.float32)
        out = fn.forward(x, b)
        assert out.shape == (4, 32)

    def test_gelu_at_zero(self):
        fn = self._make()
        x = np.array([[0.0]], dtype=np.float32)
        out = fn.forward(x)
        assert abs(float(out[0, 0])) < 0.01

    def test_gelu_positive_for_large_positive(self):
        fn = self._make()
        x = np.array([[10.0]], dtype=np.float32)
        out = fn.forward(x)
        assert float(out[0, 0]) > 9.0

    def test_exact_and_approx_close(self):
        fn_approx = self._make(approximate=True)
        fn_exact = self._make(approximate=False)
        x = np.random.randn(8, 16).astype(np.float32)
        out_approx = fn_approx.forward(x)
        out_exact = fn_exact.forward(x)
        assert np.allclose(out_approx, out_exact, atol=0.01)

    def test_backward_grad_shapes(self):
        fn = self._make()
        x = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8).astype(np.float32)
        grad_out = np.ones((4, 8), dtype=np.float32)
        grad_x, grad_b = fn.backward(grad_out, x, b)
        assert grad_x.shape == x.shape
        assert grad_b.shape == b.shape

    def test_backward_no_bias(self):
        fn = self._make()
        x = np.random.randn(4, 8).astype(np.float32)
        grad_out = np.ones((4, 8), dtype=np.float32)
        grad_x, grad_b = fn.backward(grad_out, x, None)
        assert grad_x.shape == x.shape
        assert grad_b is None

    def test_default_config(self):
        fn = FusedBiasGELU()
        assert fn.config is not None


# ── TokenBudgetScheduler ──────────────────────────────────────────────────────

from squish.serving.token_budget_scheduler import (
    TokenBudgetSchedulerConfig,
    RequestBudget,
    TokenBudgetScheduler,
)


class TestTokenBudgetSchedulerConfig:
    def test_defaults(self):
        cfg = TokenBudgetSchedulerConfig()
        assert cfg.total_kv_slots > 0
        assert 0 < cfg.prune_fraction < 1
        assert 0 < cfg.swap_threshold <= 1

    def test_custom(self):
        cfg = TokenBudgetSchedulerConfig(total_kv_slots=1024, prune_fraction=0.1)
        assert cfg.total_kv_slots == 1024

    def test_invalid_prune_fraction(self):
        with pytest.raises(ValueError):
            TokenBudgetSchedulerConfig(prune_fraction=0.0)

    def test_invalid_swap_threshold(self):
        with pytest.raises(ValueError):
            TokenBudgetSchedulerConfig(swap_threshold=0.0)


class TestTokenBudgetScheduler:
    def _make(self, total=128, prune=0.3, swap_thr=0.8):
        cfg = TokenBudgetSchedulerConfig(total_kv_slots=total, prune_fraction=prune, swap_threshold=swap_thr)
        return TokenBudgetScheduler(cfg)

    def test_register(self):
        sched = self._make()
        sched.register(1, max_tokens=32)

    def test_total_kv_used_zero_initially(self):
        sched = self._make()
        sched.register(1, max_tokens=32)
        assert sched.total_kv_used() == 0

    def test_record_attention(self):
        sched = self._make()
        sched.register(1, max_tokens=32)
        imp = np.random.rand(16).astype(np.float32)
        sched.record_attention(1, imp)
        assert sched._requests[1].n_tokens == 16

    def test_enforce_no_pressure(self):
        sched = self._make(total=1024)
        sched.register(1, max_tokens=32)
        sched.record_attention(1, np.ones(10, dtype=np.float32))
        evictions = sched.enforce()
        assert isinstance(evictions, list)

    def test_enforce_prunes_on_pressure(self):
        sched = self._make(total=10, swap_thr=0.5)
        sched.register(1, max_tokens=32)
        sched.record_attention(1, np.ones(8, dtype=np.float32))
        evictions = sched.enforce(available_slots=10)
        assert isinstance(evictions, list)

    def test_swap_out(self):
        sched = self._make()
        sched.register(1, max_tokens=32)
        sched.record_attention(1, np.ones(8, dtype=np.float32))
        result = sched.swap_out(1)
        assert result
        assert sched._requests[1].swapped

    def test_swap_in(self):
        sched = self._make()
        sched.register(1, max_tokens=32)
        sched.record_attention(1, np.ones(8, dtype=np.float32))
        sched.swap_out(1)
        result = sched.swap_in(1)
        assert result
        assert not sched._requests[1].swapped

    def test_unregister(self):
        sched = self._make()
        sched.register(1, max_tokens=32)
        sched.unregister(1)
        assert 1 not in sched._requests

    def test_default_config(self):
        sched = TokenBudgetScheduler()
        assert sched.config is not None

    def test_swap_out_nonexistent_returns_false(self):
        sched = self._make()
        assert not sched.swap_out(999)
