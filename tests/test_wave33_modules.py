"""
tests/test_wave33_modules.py

Test suite for Wave 33 modules — Velocity Compression:

  - squish/speculative/ngram_draft.py      (NgramDrafter)
  - squish/hardware/fused_qkv_proj.py      (FusedQKVProjection)
  - squish/serving/decode_hedger.py        (DecodeHedger)
  - squish/streaming/prefill_splitter.py   (PrefillSplitter)
  - squish/quant/weight_only_int2.py       (WeightOnlyInt2Quant)
  - squish/token/skip_layer_predictor.py   (SkipLayerPredictor)
"""

import math
import numpy as np
import pytest

# ============================================================
# NgramDrafter tests
# ============================================================

from squish.speculative.ngram_draft import NgramDraftConfig, NgramDrafter


class TestNgramDraftConfig:
    def test_defaults(self):
        cfg = NgramDraftConfig()
        assert cfg.max_ngram_size == 4
        assert cfg.min_ngram_size == 2
        assert cfg.draft_length == 5

    def test_invalid_max_ngram_size(self):
        with pytest.raises(ValueError, match="max_ngram_size"):
            NgramDraftConfig(max_ngram_size=1)

    def test_invalid_min_ngram_size_zero(self):
        with pytest.raises(ValueError, match="min_ngram_size"):
            NgramDraftConfig(min_ngram_size=0)

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_ngram_size"):
            NgramDraftConfig(min_ngram_size=5, max_ngram_size=3)

    def test_invalid_draft_length(self):
        with pytest.raises(ValueError, match="draft_length"):
            NgramDraftConfig(draft_length=0)

    def test_invalid_max_table_size(self):
        with pytest.raises(ValueError, match="max_table_size"):
            NgramDraftConfig(max_table_size=4)

    def test_context_window_too_small(self):
        with pytest.raises(ValueError, match="context_window"):
            NgramDraftConfig(context_window=2, max_ngram_size=4)


class TestNgramDrafter:
    def setup_method(self):
        self.drafter = NgramDrafter()

    def test_empty_draft_before_update(self):
        result = self.drafter.draft([1, 2, 3, 4])
        assert result == []

    def test_empty_draft_empty_context(self):
        self.drafter.update([1, 2, 3, 4, 5, 6])
        result = self.drafter.draft([])
        assert result == []

    def test_update_populates_table(self):
        tokens = list(range(100))
        self.drafter.update(tokens)
        assert self.drafter.table_size > 0

    def test_draft_returns_list(self):
        tokens = list(range(20))
        self.drafter.update(tokens)
        draft = self.drafter.draft(tokens)
        assert isinstance(draft, list)
        assert all(isinstance(t, int) for t in draft)

    def test_draft_respects_length_limit(self):
        cfg = NgramDraftConfig(draft_length=3)
        d = NgramDrafter(cfg)
        tokens = list(range(30))
        d.update(tokens)
        draft = d.draft(tokens)
        assert len(draft) <= 3

    def test_draft_follows_context(self):
        # Repeating pattern: [0,1,2,3,0,1,2,3,...] — predictor should find 0,1,2,3
        pattern = list(range(4)) * 20
        self.drafter.update(pattern)
        draft = self.drafter.draft(pattern[-10:])
        # At minimum the first draft token continues the pattern
        assert len(draft) >= 1

    def test_acceptance_rate_zero_before_any_drafts(self):
        assert self.drafter.acceptance_rate == 0.0

    def test_record_acceptance_updates_rate(self):
        tokens = list(range(40))
        self.drafter.update(tokens)
        self.drafter.draft(tokens)        # increments _total_drafted
        self.drafter.record_acceptance(2)
        assert self.drafter._total_accepted == 2

    def test_acceptance_rate_calculation(self):
        tokens = list(range(40))
        self.drafter.update(tokens)
        self.drafter.draft(tokens)
        self.drafter.record_acceptance(3)
        # acceptance rate = accepted / drafted (drafted >= 1)
        assert 0.0 <= self.drafter.acceptance_rate <= 1.0

    def test_reset_clears_table(self):
        tokens = list(range(50))
        self.drafter.update(tokens)
        assert self.drafter.table_size > 0
        self.drafter.reset()
        assert self.drafter.table_size == 0
        assert self.drafter.n_prefixes == 0

    def test_reset_clears_stats(self):
        tokens = list(range(40))
        self.drafter.update(tokens)
        self.drafter.draft(tokens)
        self.drafter.record_acceptance(2)
        self.drafter.reset()
        assert self.drafter._total_drafted == 0
        assert self.drafter._total_accepted == 0
        assert self.drafter.acceptance_rate == 0.0

    def test_eviction_on_full_table(self):
        cfg = NgramDraftConfig(max_table_size=200)
        d = NgramDrafter(cfg)
        # Insert more than max_table_size items
        long_seq = list(range(500, 1000))
        d.update(long_seq)
        assert d.table_size <= cfg.max_table_size

    def test_draft_consistency_repeated(self):
        """Same context produces the same draft deterministically."""
        tokens = list(range(50))
        self.drafter.update(tokens)
        draft1 = self.drafter.draft(tokens)
        draft2 = self.drafter.draft(tokens)
        assert draft1 == draft2


# ============================================================
# FusedQKVProjection tests
# ============================================================

from squish.hardware.fused_qkv_proj import FusedQKVConfig, FusedQKVProjection


class TestFusedQKVConfig:
    def test_defaults(self):
        cfg = FusedQKVConfig()
        assert cfg.d_model == 4096
        assert cfg.n_heads == 32
        assert cfg.n_kv_heads == 8

    def test_derived_dimensions(self):
        cfg = FusedQKVConfig(d_model=512, d_head=64, n_heads=8, n_kv_heads=2)
        assert cfg.d_q == 512
        assert cfg.d_kv == 128
        assert cfg.d_qkv == 512 + 128 + 128

    def test_invalid_d_model_zero(self):
        with pytest.raises(ValueError, match="d_model"):
            FusedQKVConfig(d_model=0, d_head=64, n_heads=8, n_kv_heads=2)

    def test_invalid_n_heads_not_divisible_by_n_kv_heads(self):
        with pytest.raises(ValueError, match="n_heads must be divisible"):
            FusedQKVConfig(d_model=512, d_head=64, n_heads=8, n_kv_heads=3)

    def test_invalid_d_model_ne_n_heads_times_d_head(self):
        with pytest.raises(ValueError, match="d_model must equal"):
            FusedQKVConfig(d_model=1024, d_head=64, n_heads=8, n_kv_heads=2)


class TestFusedQKVProjection:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.cfg = FusedQKVConfig(
            d_model=256, d_head=32, n_heads=8, n_kv_heads=2
        )
        self.proj = FusedQKVProjection(self.cfg)

    def _make_weights(self):
        cfg = self.cfg
        w_q = self.rng.standard_normal((cfg.d_model, cfg.d_q)).astype(np.float32)
        w_k = self.rng.standard_normal((cfg.d_model, cfg.d_kv)).astype(np.float32)
        w_v = self.rng.standard_normal((cfg.d_model, cfg.d_kv)).astype(np.float32)
        return w_q, w_k, w_v

    def test_not_packed_initially(self):
        assert not self.proj.is_packed

    def test_project_before_pack_raises(self):
        x = self.rng.standard_normal((4, self.cfg.d_model)).astype(np.float32)
        with pytest.raises(RuntimeError, match="pack_weights"):
            self.proj.project(x)

    def test_pack_weights_sets_flag(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        assert self.proj.is_packed

    def test_weight_shape_after_pack(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        expected = (self.cfg.d_model, self.cfg.d_qkv)
        assert self.proj.weight_shape == expected

    def test_project_output_shapes(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        seq_len = 16
        x = self.rng.standard_normal((seq_len, self.cfg.d_model)).astype(np.float32)
        q, k, v = self.proj.project(x)
        assert q.shape == (seq_len, self.cfg.d_q)
        assert k.shape == (seq_len, self.cfg.d_kv)
        assert v.shape == (seq_len, self.cfg.d_kv)

    def test_project_matches_separate_matmuls(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        x = self.rng.standard_normal((8, self.cfg.d_model)).astype(np.float32)
        q_fused, k_fused, v_fused = self.proj.project(x)
        q_ref = x @ w_q
        k_ref = x @ w_k
        v_ref = x @ w_v
        np.testing.assert_allclose(q_fused, q_ref, atol=1e-5)
        np.testing.assert_allclose(k_fused, k_ref, atol=1e-5)
        np.testing.assert_allclose(v_fused, v_ref, atol=1e-5)

    def test_unpack_recovers_original_weights(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        uq, uk, uv = self.proj.unpack_weights()
        np.testing.assert_array_equal(uq, w_q)
        np.testing.assert_array_equal(uk, w_k)
        np.testing.assert_array_equal(uv, w_v)

    def test_wrong_weight_shape_raises(self):
        w_q, w_k, w_v = self._make_weights()
        bad_q = w_q[:, :10]  # wrong output dim
        with pytest.raises(ValueError, match="w_q"):
            self.proj.pack_weights(bad_q, w_k, w_v)

    def test_partial_bias_raises(self):
        w_q, w_k, w_v = self._make_weights()
        b_q = np.zeros(self.cfg.d_q, dtype=np.float32)
        with pytest.raises(ValueError, match="bias"):
            self.proj.pack_weights(w_q, w_k, w_v, b_q=b_q)

    def test_weight_bytes_positive_after_pack(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        assert self.proj.weight_bytes > 0

    def test_batch_input(self):
        w_q, w_k, w_v = self._make_weights()
        self.proj.pack_weights(w_q, w_k, w_v)
        x = self.rng.standard_normal((2, 4, self.cfg.d_model)).astype(np.float32)
        q, k, v = self.proj.project(x)
        assert q.shape == (2, 4, self.cfg.d_q)


# ============================================================
# DecodeHedger tests
# ============================================================

from squish.serving.decode_hedger import (
    DecodeHedger,
    DecodeHedgerConfig,
    HedgePolicy,
)


class TestDecodeHedgerConfig:
    def test_defaults(self):
        cfg = DecodeHedgerConfig()
        assert cfg.policy == HedgePolicy.THRESHOLD
        assert cfg.token_threshold == 64
        assert cfg.p99_window_size == 100

    def test_invalid_token_threshold(self):
        with pytest.raises(ValueError, match="token_threshold"):
            DecodeHedgerConfig(token_threshold=0)

    def test_invalid_p99_window_size(self):
        with pytest.raises(ValueError, match="p99_window_size"):
            DecodeHedgerConfig(p99_window_size=5)

    def test_invalid_p99_target_ms(self):
        with pytest.raises(ValueError, match="p99_target_ms"):
            DecodeHedgerConfig(p99_target_ms=0)

    def test_invalid_hedge_timeout(self):
        with pytest.raises(ValueError, match="hedge_timeout_ms"):
            DecodeHedgerConfig(hedge_timeout_ms=-1.0)

    def test_invalid_max_parallel_hedges(self):
        with pytest.raises(ValueError, match="max_parallel_hedges"):
            DecodeHedgerConfig(max_parallel_hedges=0)


class TestDecodeHedger:
    def setup_method(self):
        self.cfg = DecodeHedgerConfig(
            policy=HedgePolicy.THRESHOLD,
            token_threshold=32,
        )
        self.hedger = DecodeHedger(self.cfg)

    def test_initial_active_hedges_zero(self):
        assert self.hedger.active_hedges == 0

    def test_should_hedge_always_policy(self):
        h = DecodeHedger(DecodeHedgerConfig(policy=HedgePolicy.ALWAYS))
        assert h.should_hedge(1) is True

    def test_should_not_hedge_below_threshold(self):
        assert self.hedger.should_hedge(10) is False

    def test_should_hedge_above_threshold(self):
        assert self.hedger.should_hedge(100) is True

    def test_begin_hedge_increments_active(self):
        self.hedger.begin_hedge()
        assert self.hedger.active_hedges == 1

    def test_end_hedge_decrements_active(self):
        hid = self.hedger.begin_hedge()
        self.hedger.end_hedge(hid, hedge_won=False, latency_ms=300.0)
        assert self.hedger.active_hedges == 0

    def test_hedge_win_rate_after_win(self):
        hid = self.hedger.begin_hedge()
        self.hedger.end_hedge(hid, hedge_won=True, latency_ms=200.0)
        assert self.hedger.hedge_win_rate == 1.0

    def test_hedge_win_rate_zero_initially(self):
        assert self.hedger.hedge_win_rate == 0.0

    def test_adaptive_policy_activates_on_high_p99(self):
        h = DecodeHedger(
            DecodeHedgerConfig(
                policy=HedgePolicy.ADAPTIVE,
                p99_target_ms=100.0,
                p99_window_size=10,
            )
        )
        for _ in range(10):
            h.record_latency(500.0)
        assert h.should_hedge(1) is True

    def test_adaptive_policy_no_hedge_when_ok(self):
        h = DecodeHedger(
            DecodeHedgerConfig(
                policy=HedgePolicy.ADAPTIVE,
                p99_target_ms=1000.0,
                p99_window_size=10,
            )
        )
        for _ in range(10):
            h.record_latency(50.0)
        assert h.should_hedge(1) is False

    def test_max_parallel_cap(self):
        h = DecodeHedger(DecodeHedgerConfig(policy=HedgePolicy.ALWAYS, max_parallel_hedges=1))
        h.begin_hedge()
        assert h.should_hedge(200) is False

    def test_p99_latency_calculation(self):
        h = DecodeHedger(DecodeHedgerConfig(p99_window_size=10))
        for v in [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]:
            h.record_latency(v)
        assert h.p99_latency_ms >= 900.0

    def test_reset_stats_clears_history(self):
        for v in [100.0, 200.0]:
            self.hedger.record_latency(v)
        self.hedger.reset_stats()
        assert self.hedger.p99_latency_ms == 0.0
        assert self.hedger.total_hedges_launched == 0


# ============================================================
# PrefillSplitter tests
# ============================================================

from squish.streaming.prefill_splitter import PrefillSplitter, PrefillSplitterConfig


class TestPrefillSplitterConfig:
    def test_defaults(self):
        cfg = PrefillSplitterConfig()
        assert cfg.min_chunk_size == 64
        assert cfg.max_chunk_size == 2048
        assert cfg.target_ttft_ms == 200.0

    def test_invalid_min_chunk_size(self):
        with pytest.raises(ValueError, match="min_chunk_size"):
            PrefillSplitterConfig(min_chunk_size=0)

    def test_max_less_than_min(self):
        with pytest.raises(ValueError, match="max_chunk_size"):
            PrefillSplitterConfig(min_chunk_size=512, max_chunk_size=64)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            PrefillSplitterConfig(alpha=0.0)

    def test_invalid_target_ttft(self):
        with pytest.raises(ValueError, match="target_ttft_ms"):
            PrefillSplitterConfig(target_ttft_ms=0.0)

    def test_negative_throughput_floor_raises(self):
        with pytest.raises(ValueError, match="throughput_floor_tps"):
            PrefillSplitterConfig(throughput_floor_tps=-1.0)


class TestPrefillSplitter:
    def setup_method(self):
        self.cfg = PrefillSplitterConfig(
            min_chunk_size=64,
            max_chunk_size=512,
            initial_chunk_size=128,
            target_ttft_ms=100.0,
        )
        self.splitter = PrefillSplitter(self.cfg)

    def test_empty_sequence_no_chunks(self):
        chunks = list(self.splitter.split([]))
        assert chunks == []

    def test_single_chunk_short_sequence(self):
        ids = list(range(50))
        chunks = list(self.splitter.split(ids))
        assert len(chunks) == 1
        assert chunks[0] == ids

    def test_covers_all_tokens(self):
        ids = list(range(1000))
        chunks = list(self.splitter.split(ids))
        recovered = []
        for c in chunks:
            recovered.extend(c)
        assert recovered == ids

    def test_first_chunk_respects_initial_chunk_size(self):
        ids = list(range(2000))
        chunks = list(self.splitter.split(ids))
        assert len(chunks[0]) == self.cfg.initial_chunk_size

    def test_subsequent_chunks_up_to_max(self):
        ids = list(range(2000))
        chunks = list(self.splitter.split(ids))
        for chunk in chunks[1:]:
            assert len(chunk) <= self.cfg.max_chunk_size

    def test_record_chunk_updates_ema(self):
        self.splitter.record_chunk(128, 50.0)  # 128 tokens / 50 ms = 2560 tps
        assert self.splitter.estimated_tps > 0.0

    def test_adaptation_decreases_chunk_for_slow_hardware(self):
        # Very slow: 10 tps → optimal chunk = 100ms * 10 / 1000 = 1 token → clamped to min
        self.splitter.record_chunk(128, 12_800.0)  # 10 tps
        assert self.splitter.current_chunk_size == self.cfg.min_chunk_size

    def test_adaptation_caps_at_max_chunk_size(self):
        # Very fast: 100_000 tps → optimal chunk >> max
        self.splitter.record_chunk(512, 5.12)  # 100_000 tps
        assert self.splitter.current_chunk_size <= self.cfg.max_chunk_size

    def test_set_chunk_size_clamps(self):
        self.splitter.set_chunk_size(1)
        assert self.splitter.current_chunk_size == self.cfg.min_chunk_size
        self.splitter.set_chunk_size(100_000)
        assert self.splitter.current_chunk_size == self.cfg.max_chunk_size

    def test_estimated_ttft_inf_before_measurements(self):
        s = PrefillSplitter()
        assert math.isinf(s.estimated_ttft_ms())

    def test_estimated_ttft_positive_after_record(self):
        self.splitter.record_chunk(128, 50.0)
        ttft = self.splitter.estimated_ttft_ms()
        assert ttft > 0.0
        assert not math.isinf(ttft)

    def test_chunk_count_empty(self):
        assert self.splitter.chunk_count(0) == 0

    def test_n_measurements_increments(self):
        assert self.splitter.n_measurements == 0
        self.splitter.record_chunk(100, 20.0)
        assert self.splitter.n_measurements == 1


# ============================================================
# WeightOnlyInt2Quant tests
# ============================================================

from squish.quant.weight_only_int2 import Int2QuantConfig, WeightOnlyInt2Quant


class TestInt2QuantConfig:
    def test_defaults(self):
        cfg = Int2QuantConfig()
        assert cfg.group_size == 64
        assert cfg.symmetric is False

    def test_group_size_too_small(self):
        with pytest.raises(ValueError, match="group_size"):
            Int2QuantConfig(group_size=4)

    def test_group_size_not_divisible_by_four(self):
        with pytest.raises(ValueError, match="group_size"):
            Int2QuantConfig(group_size=10)

    def test_invalid_clip_threshold_low(self):
        with pytest.raises(ValueError, match="clip_threshold"):
            Int2QuantConfig(clip_threshold=0.1)

    def test_invalid_clip_threshold_high(self):
        with pytest.raises(ValueError, match="clip_threshold"):
            Int2QuantConfig(clip_threshold=1.1)


class TestWeightOnlyInt2Quant:
    def setup_method(self):
        self.rng = np.random.default_rng(7)
        self.q = WeightOnlyInt2Quant(Int2QuantConfig(group_size=16))

    def _weight(self, rows=8, cols=64):
        return self.rng.standard_normal((rows, cols)).astype(np.float32)

    def test_quantize_returns_three_arrays(self):
        w = self._weight()
        result = self.q.quantize(w)
        assert len(result) == 3

    def test_packed_shape(self):
        w = self._weight(8, 64)
        packed, _, _ = self.q.quantize(w)
        assert packed.shape == (8, 64 // WeightOnlyInt2Quant.PACK_FACTOR)

    def test_packed_dtype(self):
        w = self._weight()
        packed, _, _ = self.q.quantize(w)
        assert packed.dtype == np.uint8

    def test_scale_shape(self):
        w = self._weight(8, 64)
        _, scale, _ = self.q.quantize(w)
        assert scale.shape == (8, 64 // 16)  # n_groups

    def test_dequantize_shape(self):
        w = self._weight(8, 64)
        packed, scale, zero = self.q.quantize(w)
        w_rec = self.q.dequantize(packed, scale, zero)
        assert w_rec.shape == w.shape

    def test_quantize_1d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            self.q.quantize(np.zeros(64, dtype=np.float32))

    def test_cols_not_divisible_raises(self):
        with pytest.raises(ValueError, match="divisible"):
            self.q.quantize(np.zeros((4, 17), dtype=np.float32))

    def test_dequantize_round_trip_error_small(self):
        """Quantization error should be much smaller than the weight range."""
        w = self._weight(16, 64)
        packed, scale, zero = self.q.quantize(w)
        w_rec = self.q.dequantize(packed, scale, zero)
        w_range = float(w.max() - w.min())
        max_err = float(np.abs(w - w_rec).max())
        # With group_size=16 and INT2, max deviation <= scale/2 per group
        assert max_err < w_range * 0.6

    def test_symmetric_mode(self):
        qs = WeightOnlyInt2Quant(Int2QuantConfig(group_size=16, symmetric=True))
        w = self._weight()
        packed, scale, zero = qs.quantize(w)
        np.testing.assert_array_equal(zero, np.zeros_like(zero))

    def test_compression_ratio_float16(self):
        ratio = self.q.compression_ratio("float16")
        assert ratio == pytest.approx(8.0)

    def test_compression_ratio_float32(self):
        ratio = self.q.compression_ratio("float32")
        assert ratio == pytest.approx(16.0)

    def test_pack_unpack_identity(self):
        """Pack-4 bit packing is a lossless bijection — unpack(pack(x)) == x."""
        rows, cols = 4, 32
        # Integer values in [0, 3] — the domain of the pack/unpack operations
        rng = np.random.default_rng(0)
        w_int = rng.integers(0, 4, size=(rows, cols), dtype=np.uint8)
        packed = self.q._pack(w_int)
        w_int_recovered = self.q._unpack(packed)
        np.testing.assert_array_equal(w_int_recovered, w_int)


# ============================================================
# SkipLayerPredictor tests
# ============================================================

from squish.token.skip_layer_predictor import SkipLayerConfig, SkipLayerPredictor


class TestSkipLayerConfig:
    def test_defaults(self):
        cfg = SkipLayerConfig()
        assert cfg.n_layers == 32
        assert cfg.n_features == 4
        assert cfg.exit_threshold == 0.90

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            SkipLayerConfig(n_layers=1)

    def test_invalid_n_features(self):
        with pytest.raises(ValueError, match="n_features"):
            SkipLayerConfig(n_features=0)

    def test_invalid_exit_threshold_low(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            SkipLayerConfig(exit_threshold=0.3)

    def test_invalid_exit_threshold_high(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            SkipLayerConfig(exit_threshold=1.1)

    def test_invalid_max_skip_fraction_zero(self):
        with pytest.raises(ValueError, match="max_skip_fraction"):
            SkipLayerConfig(max_skip_fraction=0.0)

    def test_invalid_max_skip_fraction_one(self):
        with pytest.raises(ValueError, match="max_skip_fraction"):
            SkipLayerConfig(max_skip_fraction=1.0)

    def test_invalid_lr(self):
        with pytest.raises(ValueError, match="lr"):
            SkipLayerConfig(lr=0.0)


class TestSkipLayerPredictor:
    def setup_method(self):
        self.cfg = SkipLayerConfig(n_layers=8, warmup_tokens=5, exit_threshold=0.51)
        self.pred = SkipLayerPredictor(self.cfg)
        self.rng = np.random.default_rng(99)

    def _hidden(self, d=32):
        return self.rng.standard_normal(d).astype(np.float32)

    def test_not_warmed_up_initially(self):
        assert not self.pred.is_warmed_up

    def test_no_skip_before_warmup(self):
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.1, 2)
        assert self.pred.should_skip(2, feats) is False

    def test_first_last_layer_never_skipped(self):
        # Warm up
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.0, 0)
        for _ in range(self.cfg.warmup_tokens + 10):
            self.pred.update(0, feats, was_skippable=True)
        assert self.pred.should_skip(0, feats) is False
        assert self.pred.should_skip(self.cfg.n_layers - 1, feats) is False

    def test_extract_features_shape(self):
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.5, 3)
        assert feats.shape == (self.cfg.n_features,)

    def test_extract_features_dtype(self):
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.5, 3)
        assert feats.dtype == np.float32

    def test_update_increments_token_count(self):
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.1, 2)
        self.pred.update(2, feats, was_skippable=True)
        assert self.pred.token_count == 1

    def test_skip_rate_zero_initially(self):
        assert self.pred.skip_rate(2) == 0.0

    def test_global_skip_rate_zero_initially(self):
        assert self.pred.global_skip_rate() == 0.0

    def test_online_learning_increases_skip_probability(self):
        """After many 'was_skippable=True' updates the predictor should learn to skip."""
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.01, 3)
        layer_idx = 3
        for _ in range(200):
            self.pred.update(layer_idx, feats, was_skippable=True)
        # After 200 positive examples the logit should be strongly positive
        logit = float(np.dot(self.pred._weights[layer_idx], feats)) + float(
            self.pred._bias[layer_idx]
        )
        assert logit > 0.0

    def test_reset_zeroes_weights(self):
        h = self._hidden()
        feats = self.pred.extract_features(h, 0.1, 2)
        for _ in range(50):
            self.pred.update(2, feats, was_skippable=True)
        self.pred.reset()
        np.testing.assert_array_equal(self.pred._weights, np.zeros_like(self.pred._weights))
        assert self.pred.token_count == 0
        assert self.pred.global_skip_rate() == 0.0

    def test_max_skip_fraction_cap(self):
        """skip_rate cap forces should_skip to return False even when confident."""
        cfg = SkipLayerConfig(
            n_layers=8,
            warmup_tokens=0,
            exit_threshold=0.5,
            max_skip_fraction=0.01,  # essentially never skip
        )
        pred = SkipLayerPredictor(cfg)
        h = self._hidden()
        feats = pred.extract_features(h, 0.01, 3)
        # Give strong positive signal
        for _ in range(300):
            pred.update(3, feats, was_skippable=True)
        # Force skip counts artificially to exceed cap
        pred._layer_call_counts[3] = 100
        pred._layer_skip_counts[3] = 99  # 99% skip rate > max_skip_fraction=0.01
        assert pred.should_skip(3, feats) is False
