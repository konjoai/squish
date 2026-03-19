"""tests/test_wave30_modules.py — Tests for Wave 30 inference optimization modules

Covers:
  - thermal_scheduler:      ThermalState, ThermalConfig, ThermalScheduler, ThermalScheduleParams
  - batched_draft_verify:   VerifyRequest, VerifyResult, BatchedDraftVerifierConfig, BatchedDraftVerifier, BatchVerifyStats
  - adaptive_rope:          RoPEScaleMode, AdaptiveRoPEConfig, AdaptiveRoPE
  - activation_offload:     OffloadConfig, OffloadStats, ActivationBank, ActivationOffloader
  - gear_kv:                GEARConfig, GEARQuantizedKV, GEARLayer, GEARManager, GEARStats
  - quant_rotary:           QuantRotaryConfig, QuantRotary
"""

from __future__ import annotations

import numpy as np
import pytest

# ============================================================
# thermal_scheduler
# ============================================================
from squish.serving.thermal_scheduler import (
    ThermalConfig,
    ThermalScheduleParams,
    ThermalScheduler,
    ThermalState,
    _BATCH_REDUCTION,
)


class TestThermalState:
    def test_enum_values_ordered(self):
        assert ThermalState.NOMINAL < ThermalState.WARM
        assert ThermalState.WARM < ThermalState.HOT
        assert ThermalState.HOT < ThermalState.CRITICAL

    def test_int_conversion(self):
        assert int(ThermalState.NOMINAL) == 0
        assert int(ThermalState.CRITICAL) == 3


class TestThermalConfig:
    def test_defaults(self):
        cfg = ThermalConfig()
        assert cfg.base_batch_size >= 1
        assert cfg.latency_ema_alpha > 0

    def test_custom_config(self):
        cfg = ThermalConfig(base_batch_size=4, base_max_tokens=256)
        assert cfg.base_batch_size == 4
        assert cfg.base_max_tokens == 256


class TestThermalScheduler:
    def _make_scheduler(self, base_batch=8) -> ThermalScheduler:
        return ThermalScheduler(ThermalConfig(
            base_batch_size=base_batch,
            sysctl_poll_interval=0,   # disable sysctl in tests
        ))

    def test_initial_state_nominal(self):
        sched = self._make_scheduler()
        assert sched.current_state == ThermalState.NOMINAL

    def test_active_params_nominal_returns_full_batch(self):
        sched = self._make_scheduler(base_batch=8)
        params = sched.active_params()
        assert params.batch_size == 8
        assert params.state == ThermalState.NOMINAL
        assert params.speculation_enabled

    def test_record_latency_sets_ema(self):
        sched = self._make_scheduler()
        sched.record_step_latency(20.0)
        assert sched.ema_latency_ms is not None

    def test_high_latency_triggers_throttle(self):
        sched = self._make_scheduler()
        # Feed 20 latency samples, steadily increasing to simulate throttle
        for _ in range(10):
            sched.record_step_latency(10.0)   # establish baseline
        for _ in range(20):
            sched.record_step_latency(100.0)  # 10× slowdown → critical
        assert sched.current_state in (ThermalState.HOT, ThermalState.CRITICAL)

    def test_batch_size_reduces_under_pressure(self):
        sched = self._make_scheduler(base_batch=8)
        sched.force_state(ThermalState.HOT)
        params = sched.active_params()
        assert params.batch_size < 8

    def test_critical_state_batch_is_quarter(self):
        sched = self._make_scheduler(base_batch=8)
        sched.force_state(ThermalState.CRITICAL)
        params = sched.active_params()
        expected = max(1, int(8 * _BATCH_REDUCTION[ThermalState.CRITICAL]))
        assert params.batch_size == expected

    def test_speculation_disabled_at_hot(self):
        sched = self._make_scheduler()
        sched.force_state(ThermalState.HOT)
        params = sched.active_params()
        assert not params.speculation_enabled

    def test_speculation_enabled_at_warm(self):
        sched = self._make_scheduler()
        sched.force_state(ThermalState.WARM)
        params = sched.active_params()
        assert params.speculation_enabled

    def test_invalid_latency_raises(self):
        sched = self._make_scheduler()
        with pytest.raises(ValueError):
            sched.record_step_latency(0.0)
        with pytest.raises(ValueError):
            sched.record_step_latency(-5.0)

    def test_reset_clears_state(self):
        sched = self._make_scheduler()
        sched.force_state(ThermalState.CRITICAL)
        sched.reset()
        assert sched.current_state == ThermalState.NOMINAL
        assert sched.ema_latency_ms is None

    def test_repr_contains_state_name(self):
        sched = self._make_scheduler()
        assert "NOMINAL" in repr(sched)


# ============================================================
# batched_draft_verify
# ============================================================
from squish.speculative.batched_draft_verify import (
    BatchedDraftVerifier,
    BatchedDraftVerifierConfig,
    BatchVerifyStats,
    VerifyRequest,
    VerifyResult,
)


def _make_model_fn(vocab: int = 50):
    """Model that always predicts argmax to be draft_token[i] (100% accept)."""
    def model_fn(token_ids, kv_ids, mask):
        batch, seq = token_ids.shape
        logits = np.full((batch, seq, vocab), -1e4, dtype=np.float32)
        for b in range(batch):
            for s in range(seq):
                tok = int(token_ids[b, s])
                logits[b, s, tok % vocab] = 100.0
        return logits
    return model_fn


def _make_reject_model_fn(vocab: int = 50):
    """Model that always predicts token 0 (rejects all drafts except token 0)."""
    def model_fn(token_ids, kv_ids, mask):
        batch, seq = token_ids.shape
        logits = np.full((batch, seq, vocab), -1e4, dtype=np.float32)
        logits[:, :, 0] = 100.0
        return logits
    return model_fn


class TestVerifyRequest:
    def test_basic_creation(self):
        req = VerifyRequest(
            request_id="r1",
            draft_tokens=[1, 2, 3],
            context_ids=[0],
            kv_cache_id="kv-r1",
        )
        assert req.request_id == "r1"
        assert req.draft_tokens == [1, 2, 3]


class TestBatchedDraftVerifier:
    def _make_verifier(self, max_draft=5) -> BatchedDraftVerifier:
        cfg = BatchedDraftVerifierConfig(max_draft_len=max_draft)
        return BatchedDraftVerifier(cfg)

    def test_add_request_increments_pending(self):
        verifier = self._make_verifier()
        req = VerifyRequest("r1", [1, 2, 3], [], "kv-1")
        verifier.add_request(req)
        assert verifier.pending_count() == 1

    def test_verify_all_returns_results_for_all_requests(self):
        verifier = self._make_verifier()
        for i in range(3):
            req = VerifyRequest(f"r{i}", [i + 1, i + 2], [], f"kv-{i}")
            verifier.add_request(req)
        results = verifier.verify_all(_make_model_fn())
        assert len(results) == 3
        for i in range(3):
            assert f"r{i}" in results

    def test_all_draft_tokens_accepted_by_identity_model(self):
        verifier = self._make_verifier()
        draft = [10, 20, 30]
        req = VerifyRequest("r1", draft, [], "kv-1")
        verifier.add_request(req)
        results = verifier.verify_all(_make_model_fn(vocab=100))
        assert results["r1"].n_accepted == len(draft)

    def test_all_tokens_rejected_by_reject_model(self):
        verifier = self._make_verifier()
        # Draft with tokens that don't match model prediction (0)
        req = VerifyRequest("r1", [5, 6, 7], [], "kv-1")
        verifier.add_request(req)
        results = verifier.verify_all(_make_reject_model_fn(vocab=50))
        assert results["r1"].n_accepted == 0

    def test_bonus_token_present(self):
        verifier = self._make_verifier()
        req = VerifyRequest("r1", [1, 2], [], "kv-1")
        verifier.add_request(req)
        results = verifier.verify_all(_make_model_fn(vocab=50))
        result = results["r1"]
        assert isinstance(result.bonus_token, int)

    def test_pending_cleared_after_verify(self):
        verifier = self._make_verifier()
        verifier.add_request(VerifyRequest("r1", [1], [], "kv-1"))
        verifier.verify_all(_make_model_fn())
        assert verifier.pending_count() == 0

    def test_verify_empty_pending_raises(self):
        verifier = self._make_verifier()
        with pytest.raises(RuntimeError):
            verifier.verify_all(_make_model_fn())

    def test_draft_truncated_to_max_draft_len(self):
        verifier = self._make_verifier(max_draft=3)
        req = VerifyRequest("r1", [1, 2, 3, 4, 5, 6], [], "kv-1")
        verifier.add_request(req)
        assert len(verifier._pending[0].draft_tokens) == 3

    def test_stats_tracked(self):
        verifier = self._make_verifier()
        for i in range(4):
            verifier.add_request(VerifyRequest(f"r{i}", [i + 1], [], f"kv-{i}"))
        verifier.verify_all(_make_model_fn())
        assert verifier.stats.total_batches == 1
        assert verifier.stats.total_requests == 4

    def test_accept_rate_is_one_for_identity_model(self):
        verifier = self._make_verifier()
        for i in range(3):
            verifier.add_request(VerifyRequest(f"r{i}", [i + 1, i + 2], [], f"kv-{i}"))
        verifier.verify_all(_make_model_fn(vocab=100))
        assert verifier.stats.accept_rate == 1.0

    def test_empty_request_id_raises(self):
        verifier = self._make_verifier()
        with pytest.raises(ValueError):
            verifier.add_request(VerifyRequest("", [1, 2], [], "kv"))

    def test_empty_draft_raises(self):
        verifier = self._make_verifier()
        with pytest.raises(ValueError):
            verifier.add_request(VerifyRequest("r1", [], [], "kv"))


# ============================================================
# adaptive_rope
# ============================================================
from squish.attention.adaptive_rope import (
    AdaptiveRoPE,
    AdaptiveRoPEConfig,
    RoPEScaleMode,
)


class TestRoPEScaleMode:
    def test_all_modes_exist(self):
        for mode in ("standard", "dynamic", "yarn", "ntk"):
            assert RoPEScaleMode(mode)


class TestAdaptiveRoPE:
    def _make_rope(self, mode=RoPEScaleMode.DYNAMIC, dim=64) -> AdaptiveRoPE:
        cfg = AdaptiveRoPEConfig(mode=mode, dim=dim, max_trained_len=4096)
        return AdaptiveRoPE(cfg)

    def test_get_rope_scale_standard_constant(self):
        rope = self._make_rope(RoPEScaleMode.STANDARD)
        for seq in (32, 512, 8192):
            assert rope.get_rope_scale(seq) == 10000.0

    def test_get_rope_scale_dynamic_short(self):
        rope = self._make_rope(RoPEScaleMode.DYNAMIC)
        # Short sequences should use the short base
        scale = rope.get_rope_scale(256)
        assert scale < 10000.0  # short_base < standard_base

    def test_get_rope_scale_dynamic_normal(self):
        rope = self._make_rope(RoPEScaleMode.DYNAMIC)
        scale = rope.get_rope_scale(1024)
        assert scale == 10000.0

    def test_get_rope_scale_dynamic_long(self):
        rope = self._make_rope(RoPEScaleMode.DYNAMIC)
        scale = rope.get_rope_scale(8192)
        assert scale > 10000.0  # long context → larger base

    def test_get_cos_sin_shape(self):
        rope = self._make_rope(dim=64)
        cos, sin = rope.get_cos_sin(seq_len=128)
        assert cos.shape == (128, 32)  # (seq, dim//2)
        assert sin.shape == (128, 32)

    def test_get_cos_sin_dtype(self):
        rope = self._make_rope(dim=64)
        cos, sin = rope.get_cos_sin(seq_len=32, dtype=np.float32)
        assert cos.dtype == np.float32

    def test_cos_sin_cached_on_second_call(self):
        rope = self._make_rope(dim=64)
        cos1, _ = rope.get_cos_sin(seq_len=64)
        cos2, _ = rope.get_cos_sin(seq_len=64)
        assert cos1 is cos2  # same object → cached

    def test_apply_output_shape(self):
        rope = self._make_rope(dim=64)
        cos, sin = rope.get_cos_sin(seq_len=16)
        x = np.random.default_rng(0).standard_normal((2, 4, 16, 64)).astype(np.float32)
        out = rope.apply(x, cos, sin)
        assert out.shape == x.shape

    def test_apply_preserves_dtype(self):
        rope = self._make_rope(dim=64)
        cos, sin = rope.get_cos_sin(seq_len=16, dtype=np.float32)
        x = np.ones((1, 2, 16, 64), dtype=np.float32)
        out = rope.apply(x, cos, sin)
        assert out.dtype == np.float32

    def test_apply_wrong_dim_raises(self):
        rope = self._make_rope(dim=64)
        cos, sin = rope.get_cos_sin(seq_len=8)  # cos.shape[-1] = 32
        x = np.ones((1, 8, 128))  # d_k=128, cos only has 32 pairs
        with pytest.raises(ValueError):
            rope.apply(x, cos, sin)

    def test_yarn_mode_scales_base(self):
        rope = self._make_rope(RoPEScaleMode.YARN, dim=64)
        # Yarn should scale up for long sequences
        scale_long = rope.get_rope_scale(8192)
        scale_short = rope.get_rope_scale(4096)
        assert scale_long >= scale_short

    def test_ntk_mode_scales_base(self):
        rope = self._make_rope(RoPEScaleMode.NTK, dim=64)
        scale_long = rope.get_rope_scale(8192)
        assert scale_long > 10000.0

    def test_clear_cache(self):
        rope = self._make_rope(dim=64)
        rope.get_cos_sin(seq_len=32)
        assert len(rope._cache) > 0
        rope.clear_cache()
        assert len(rope._cache) == 0


# ============================================================
# activation_offload
# ============================================================
from squish.hardware.activation_offload import (
    ActivationBank,
    ActivationOffloader,
    OffloadConfig,
    OffloadStats,
)


class TestActivationBank:
    def test_put_returns_nbytes(self):
        bank = ActivationBank()
        arr = np.ones((4, 8), dtype=np.float32)
        nbytes = bank.put(0, arr)
        assert nbytes == arr.nbytes

    def test_get_returns_stored_array(self):
        bank = ActivationBank()
        arr = np.arange(12, dtype=np.float32).reshape(3, 4)
        bank.put(2, arr)
        out = bank.get(2)
        np.testing.assert_array_equal(out, arr)

    def test_get_consume_once(self):
        bank = ActivationBank()
        bank.put(0, np.ones(8))
        bank.get(0)
        with pytest.raises(KeyError):
            bank.get(0)

    def test_get_missing_raises_key_error(self):
        bank = ActivationBank()
        with pytest.raises(KeyError):
            bank.get(99)

    def test_contains(self):
        bank = ActivationBank()
        bank.put(1, np.zeros(4))
        assert bank.contains(1)
        assert not bank.contains(2)

    def test_clear_returns_freed_bytes(self):
        bank = ActivationBank()
        arr = np.ones((10, 10), dtype=np.float32)
        bank.put(0, arr)
        bank.put(1, arr)
        freed = bank.clear()
        assert freed >= 2 * arr.nbytes
        assert len(bank) == 0

    def test_put_deep_copies_array(self):
        bank = ActivationBank()
        arr = np.ones(5, dtype=np.float32)
        bank.put(0, arr)
        arr[0] = 99.0
        out = bank.get(0)
        assert out[0] == 1.0  # deep copy — not affected by mutation


class TestActivationOffloader:
    def test_begin_prefill_activates_on_long_seq(self):
        offloader = ActivationOffloader(OffloadConfig(threshold=512))
        active = offloader.begin_prefill(seq_len=1024)
        assert active is True
        assert offloader.is_active

    def test_begin_prefill_inactive_on_short_seq(self):
        offloader = ActivationOffloader(OffloadConfig(threshold=512))
        active = offloader.begin_prefill(seq_len=100)
        assert active is False
        assert not offloader.is_active

    def test_save_and_load_roundtrip(self):
        offloader = ActivationOffloader(OffloadConfig(threshold=0))
        offloader.begin_prefill(seq_len=100)
        arr = np.arange(24, dtype=np.float32).reshape(3, 8)
        offloader.save(0, arr)
        out = offloader.load(0)
        np.testing.assert_array_equal(out, arr)

    def test_has_activation_after_save(self):
        offloader = ActivationOffloader()
        offloader.begin_prefill(4096)
        offloader.save(5, np.ones(16))
        assert offloader.has_activation(5)

    def test_has_activation_false_before_save(self):
        offloader = ActivationOffloader()
        assert not offloader.has_activation(0)

    def test_end_prefill_clears_storage(self):
        offloader = ActivationOffloader(OffloadConfig(threshold=0))
        offloader.begin_prefill(100)
        offloader.save(0, np.ones(4))
        offloader.end_prefill()
        assert not offloader.has_activation(0)

    def test_stats_tracks_offloaded_bytes(self):
        offloader = ActivationOffloader(OffloadConfig(threshold=0))
        offloader.begin_prefill(seq_len=100)
        arr = np.ones((8, 8), dtype=np.float32)
        offloader.save(0, arr)
        assert offloader.stats.offloaded_bytes >= arr.nbytes

    def test_invalid_layer_idx_raises(self):
        offloader = ActivationOffloader()
        offloader.begin_prefill(100)
        with pytest.raises(ValueError):
            offloader.save(-1, np.ones(4))

    def test_non_ndarray_raises(self):
        offloader = ActivationOffloader()
        offloader.begin_prefill(100)
        with pytest.raises(TypeError):
            offloader.save(0, [1, 2, 3])  # list, not ndarray

    def test_load_missing_raises(self):
        offloader = ActivationOffloader()
        offloader.begin_prefill(100)
        with pytest.raises(KeyError):
            offloader.load(99)

    def test_stats_type(self):
        offloader = ActivationOffloader()
        assert isinstance(offloader.stats, OffloadStats)

    def test_reset_stats(self):
        offloader = ActivationOffloader(OffloadConfig(threshold=0))
        offloader.begin_prefill(10)
        offloader.save(0, np.ones(8))
        offloader.reset_stats()
        assert offloader.stats.total_saves == 0


# ============================================================
# gear_kv
# ============================================================
from squish.kv.gear_kv import (
    GEARConfig,
    GEARLayer,
    GEARManager,
    GEARQuantizedKV,
    GEARStats,
)


class TestGEARConfig:
    def test_defaults(self):
        cfg = GEARConfig()
        assert cfg.rank > 0
        assert cfg.kv_bits in (4, 8)

    def test_invalid_kv_bits(self):
        with pytest.raises(ValueError):
            GEARLayer(GEARConfig(kv_bits=3))

    def test_invalid_rank(self):
        with pytest.raises(ValueError):
            GEARLayer(GEARConfig(rank=0))


class TestGEARLayer:
    def _make_kv(self, seq=16, d_k=32, seed=0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((seq, d_k)).astype(np.float32)

    def test_quantize_returns_gear_quantized_kv(self):
        layer = GEARLayer(GEARConfig(rank=4, kv_bits=8))
        kv = self._make_kv()
        qkv = layer.quantize(kv)
        assert isinstance(qkv, GEARQuantizedKV)

    def test_quantize_shapes(self):
        seq, d_k = 16, 32
        layer = GEARLayer(GEARConfig(rank=4, kv_bits=8))
        kv = self._make_kv(seq, d_k)
        qkv = layer.quantize(kv)
        assert qkv.q_data.shape == (seq, d_k)
        assert qkv.U.shape[0] == d_k

    def test_reconstruct_shape_matches_original(self):
        seq, d_k = 16, 32
        layer = GEARLayer(GEARConfig(rank=4, kv_bits=8))
        kv = self._make_kv(seq, d_k)
        qkv = layer.quantize(kv)
        kv_approx = layer.reconstruct(qkv)
        assert kv_approx.shape == (seq, d_k)

    def test_reconstruct_better_than_raw_quant(self):
        """GEAR reconstruction should have lower MSE than raw INT8 quantization."""
        seq, d_k = 32, 64
        layer_gear = GEARLayer(GEARConfig(rank=8, kv_bits=8))
        layer_no_gear = GEARLayer(GEARConfig(rank=1, kv_bits=8))  # almost no correction
        kv = self._make_kv(seq, d_k)
        qkv_gear = layer_gear.quantize(kv)
        qkv_no = layer_no_gear.quantize(kv)
        approx_gear = layer_gear.reconstruct(qkv_gear)
        approx_no = layer_no_gear.reconstruct(qkv_no)
        mse_gear = float(np.mean((kv - approx_gear) ** 2))
        mse_no = float(np.mean((kv - approx_no) ** 2))
        assert mse_gear <= mse_no

    def test_quantize_1d_raises(self):
        layer = GEARLayer()
        with pytest.raises(ValueError):
            layer.quantize(np.ones(16))

    def test_nbytes_lower_than_original(self):
        """GEARQuantizedKV bytes should be ideally less than original FP32."""
        seq, d_k = 64, 64
        layer = GEARLayer(GEARConfig(rank=8, kv_bits=4))
        kv = self._make_kv(seq, d_k)
        qkv = layer.quantize(kv)
        # Original FP32 bytes
        orig_bytes = kv.nbytes
        gear_bytes = qkv.nbytes()
        # GEAR with INT4 + rank-8 correction should be < 2× original
        assert gear_bytes < 2 * orig_bytes


class TestGEARManager:
    def test_quantize_and_reconstruct_layer(self):
        mgr = GEARManager(GEARConfig(rank=4, kv_bits=8), n_layers=4)
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((16, 32)).astype(np.float32)
        vals = rng.standard_normal((16, 32)).astype(np.float32)
        q_k, q_v = mgr.quantize_layer(0, keys, vals)
        k_approx, v_approx = mgr.reconstruct_layer(0, q_k, q_v)
        assert k_approx.shape == keys.shape
        assert v_approx.shape == vals.shape

    def test_stats_type(self):
        mgr = GEARManager()
        assert isinstance(mgr.stats, GEARStats)

    def test_compression_ratio_populated(self):
        mgr = GEARManager(GEARConfig(rank=4, kv_bits=8), n_layers=2)
        rng = np.random.default_rng(0)
        keys = rng.standard_normal((16, 32)).astype(np.float32)
        vals = rng.standard_normal((16, 32)).astype(np.float32)
        mgr.quantize_layer(0, keys, vals)
        assert mgr.stats.compression_ratio > 0

    def test_reset_stats(self):
        mgr = GEARManager()
        rng = np.random.default_rng(0)
        k, v = rng.standard_normal((8, 16)).astype(np.float32), rng.standard_normal((8, 16)).astype(np.float32)
        mgr.quantize_layer(0, k, v)
        mgr.reset_stats()
        assert mgr.stats.total_kv_quantized == 0


# ============================================================
# quant_rotary
# ============================================================
from squish.quant.quant_rotary import QuantRotary, QuantRotaryConfig


class TestQuantRotaryConfig:
    def test_defaults(self):
        cfg = QuantRotaryConfig()
        assert cfg.in_bits == 8
        assert cfg.out_bits == 8
        assert cfg.symmetric

    def test_invalid_in_bits_raises(self):
        with pytest.raises(ValueError):
            QuantRotary(QuantRotaryConfig(in_bits=3))

    def test_invalid_out_bits_raises(self):
        with pytest.raises(ValueError):
            QuantRotary(QuantRotaryConfig(out_bits=2))

    def test_invalid_granularity_raises(self):
        with pytest.raises(ValueError):
            QuantRotary(QuantRotaryConfig(scale_granularity="per_token"))


class TestQuantRotary:
    def _make_cos_sin(self, seq=8, d_k=32):
        half = d_k // 2
        base = 10000.0
        inv_freq = 1.0 / (base ** (np.arange(0, d_k, 2) / d_k))
        pos = np.arange(seq)
        ang = np.outer(pos, inv_freq)
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def _make_int8_tensor(self, shape, seed=0) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        q_int = rng.integers(-127, 128, size=shape, dtype=np.int8)
        scale = np.ones((*shape[:-1], 1), dtype=np.float32) * 0.01
        return q_int, scale

    def test_rotate_and_quantize_output_shapes(self):
        qr = QuantRotary()
        s, d = 8, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s, d), seed=1)
        q_out, k_out, qs, ks = qr.rotate_and_quantize(
            q_int, k_int, cos, sin, q_scale, k_scale
        )
        assert q_out.shape == (s, d)
        assert k_out.shape == (s, d)

    def test_output_dtype_int8(self):
        qr = QuantRotary()
        s, d = 4, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s, d), seed=2)
        q_out, k_out, _, _ = qr.rotate_and_quantize(
            q_int, k_int, cos, sin, q_scale, k_scale
        )
        assert q_out.dtype == np.int8
        assert k_out.dtype == np.int8

    def test_output_in_valid_int8_range(self):
        qr = QuantRotary()
        s, d = 4, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s, d), seed=3)
        q_out, k_out, _, _ = qr.rotate_and_quantize(
            q_int, k_int, cos, sin, q_scale, k_scale
        )
        assert q_out.min() >= -128
        assert q_out.max() <= 127

    def test_scale_out_positive(self):
        qr = QuantRotary()
        s, d = 4, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s, d), seed=4)
        _, _, qs, ks = qr.rotate_and_quantize(
            q_int, k_int, cos, sin, q_scale, k_scale
        )
        assert (qs > 0).all()
        assert (ks > 0).all()

    def test_odd_d_k_raises(self):
        qr = QuantRotary()
        q_int = np.zeros((4, 33), dtype=np.int8)
        k_int = np.zeros((4, 33), dtype=np.int8)
        cos = np.ones((4, 16), dtype=np.float32)
        sin = np.ones((4, 16), dtype=np.float32)
        scale = np.ones((4, 1), dtype=np.float32)
        with pytest.raises(ValueError):
            qr.rotate_and_quantize(q_int, k_int, cos, sin, scale, scale)

    def test_mismatched_shapes_raises(self):
        qr = QuantRotary()
        s, d = 4, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s + 2, d), seed=5)
        with pytest.raises(ValueError):
            qr.rotate_and_quantize(q_int, k_int, cos, sin, q_scale, k_scale)

    def test_dequantize_helper(self):
        qr = QuantRotary()
        x = np.array([[127, -127, 0, 64]], dtype=np.int8)
        scale = np.array([[0.1]], dtype=np.float32)
        out = qr.dequantize(x, scale)
        np.testing.assert_allclose(out[0, 0], 12.7, atol=0.01)

    def test_granularity_row(self):
        qr = QuantRotary(QuantRotaryConfig(scale_granularity="row"))
        s, d = 4, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s, d), seed=6)
        q_out, k_out, qs, ks = qr.rotate_and_quantize(
            q_int, k_int, cos, sin, q_scale, k_scale
        )
        assert q_out.shape == (s, d)

    def test_4bit_io(self):
        qr = QuantRotary(QuantRotaryConfig(in_bits=4, out_bits=4))
        s, d = 4, 32
        cos, sin = self._make_cos_sin(s, d)
        q_int, q_scale = self._make_int8_tensor((s, d))
        k_int, k_scale = self._make_int8_tensor((s, d), seed=7)
        q_out, k_out, _, _ = qr.rotate_and_quantize(
            q_int, k_int, cos, sin, q_scale, k_scale
        )
        assert q_out.shape == (s, d)

    def test_repr_contains_bits(self):
        qr = QuantRotary(QuantRotaryConfig(in_bits=8, out_bits=8))
        assert "8" in repr(qr)
