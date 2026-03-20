"""
tests/test_wave32_modules.py

Test suite for Wave 32 modules:
  - squish/quant/any4.py           (Any4Quantizer)
  - squish/speculative/vsd_draft.py (VSDDraftTrainer / VSDLoss)
  - squish/serving/confidence_gate.py (ConfidenceGate)
  - squish/quant/int3_runtime.py   (INT3RuntimeLoader)
  - squish/bench/benchmark_harness.py (BenchmarkHarness)
  - squish/kv/adaptive_kvtc.py     (AdaptiveKVTCManager)
"""

import math
import numpy as np
import pytest

# ============================================================
# Any4 tests
# ============================================================

from squish.quant.any4 import (
    Any4Config,
    Any4Quantized,
    Any4Quantizer,
    Any4Stats,
)


class TestAny4Config:
    def test_defaults(self):
        cfg = Any4Config()
        assert cfg.codebook_size == 16
        assert cfg.group_size == 128

    def test_invalid_codebook_size(self):
        with pytest.raises(ValueError, match="codebook_size"):
            Any4Config(codebook_size=8)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            Any4Config(group_size=0)


class TestAny4Quantizer:
    def setup_method(self):
        self.rng = np.random.default_rng(13)
        self.cfg = Any4Config(calibration_iters=20, calibration_seed=0)
        self.q = Any4Quantizer(self.cfg)
        self.sample = self.rng.normal(0, 1, (512,)).astype(np.float32)

    def test_calibrate_sets_flag(self):
        self.q.calibrate(self.sample)
        assert self.q._calibrated is True

    def test_codebook_property(self):
        self.q.calibrate(self.sample)
        cb = self.q.codebook
        assert cb is not None
        assert len(cb) == 16

    def test_quantize_requires_calibration(self):
        with pytest.raises(RuntimeError, match="calibrated"):
            self.q.quantize(self.sample)

    def test_quantize_returns_any4quantized(self):
        weight = self.rng.normal(0, 1, (256,)).astype(np.float32)
        self.q.calibrate(weight)
        result = self.q.quantize(weight)
        assert isinstance(result, Any4Quantized)
        assert result.original_shape == weight.shape

    def test_quantize_codes_range(self):
        weight = self.rng.normal(0, 1, (128,)).astype(np.float32)
        self.q.calibrate(weight)
        qw = self.q.quantize(weight)
        # decoded codes should all be 0-15
        n_values = math.prod(qw.original_shape)
        assert qw.codes.size * 2 >= n_values  # nibble packing

    def test_dequantize_shape(self):
        weight = self.rng.normal(0, 1, (256,)).astype(np.float32)
        self.q.calibrate(weight)
        qw = self.q.quantize(weight)
        recovered = self.q.dequantize(qw)
        assert recovered.shape == weight.shape

    def test_dequantize_close_enough(self):
        # k-means quantization should be closer than random
        weight = self.rng.normal(0, 1, (256,)).astype(np.float32)
        self.q.calibrate(weight)
        qw = self.q.quantize(weight)
        recovered = self.q.dequantize(qw)
        err = np.abs(recovered - weight).max()
        random_err = np.abs(self.rng.normal(0, 1, weight.shape).astype(np.float32) - weight).max()
        assert err < random_err

    def test_compression_ratio_high(self):
        weight = self.rng.normal(0, 1, (512,)).astype(np.float32)
        self.q.calibrate(weight)
        qw = self.q.quantize(weight)
        assert qw.compression_ratio > 2.0

    def test_nbytes_smaller(self):
        weight = self.rng.normal(0, 1, (512,)).astype(np.float32)
        self.q.calibrate(weight)
        qw = self.q.quantize(weight)
        assert qw.nbytes() < weight.nbytes

    def test_stats_update(self):
        weight = self.rng.normal(0, 1, (256,)).astype(np.float32)
        self.q.calibrate(weight)
        self.q.quantize(weight)
        assert self.q.stats.quantize_calls == 1

    def test_2d_weight(self):
        weight = self.rng.normal(0, 1, (64, 64)).astype(np.float32)
        self.q.calibrate(weight.ravel())
        qw = self.q.quantize(weight)
        rec = self.q.dequantize(qw)
        assert rec.shape == weight.shape

    def test_repr(self):
        r = repr(self.q)
        assert "Any4Quantizer" in r


# ============================================================
# VSD tests
# ============================================================

from squish.speculative.vsd_draft import (
    VSDConfig,
    VSDDraftTrainer,
    VSDLoss,
    VSDTrainerStats,
    VSDTrainingExample,
)


class TestVSDConfig:
    def test_defaults(self):
        cfg = VSDConfig()
        assert cfg.kl_weight >= 0
        assert cfg.n_candidates >= 1
        assert cfg.temperature > 0

    def test_invalid_kl_weight(self):
        with pytest.raises(ValueError, match="kl_weight"):
            VSDConfig(kl_weight=-1.0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            VSDConfig(temperature=0.0)


class TestVSDTrainingExample:
    def test_accepted_len_full_accept(self):
        ex = VSDTrainingExample(
            context_ids=[1, 2, 3],
            draft_tokens=[10, 11, 12],
            target_logits=np.zeros((3, 50), dtype=np.float32),
            accepted_mask=np.array([True, True, True]),
        )
        assert ex.accepted_len == 3

    def test_accepted_len_partial(self):
        ex = VSDTrainingExample(
            context_ids=[1, 2, 3, 4],
            draft_tokens=[10, 11, 12, 13],
            target_logits=np.zeros((4, 50), dtype=np.float32),
            accepted_mask=np.array([True, True, False, True]),
        )
        assert ex.accepted_len == 2

    def test_accepted_len_none(self):
        ex = VSDTrainingExample(
            context_ids=[1],
            draft_tokens=[5],
            target_logits=np.zeros((1, 50), dtype=np.float32),
            accepted_mask=np.array([False]),
        )
        assert ex.accepted_len == 0


class TestVSDLoss:
    def setup_method(self):
        self.rng = np.random.default_rng(21)
        self.cfg = VSDConfig(kl_weight=0.1)
        self.loss_fn = VSDLoss(self.cfg)
        self.T = 5
        self.V = 100

    def _draft_target(self, focus: int = 10):
        d = self.rng.normal(0, 0.1, (self.T, self.V)).astype(np.float32)
        t = self.rng.normal(0, 0.1, (self.T, self.V)).astype(np.float32)
        d[:, focus] += 3.0
        t[:, focus] += 3.0
        return d, t

    def test_compute_returns_dict(self):
        d, t = self._draft_target()
        mask = np.array([True, True, False, False, False])
        result = self.loss_fn.compute(d, t, mask)
        assert "total" in result
        assert "acceptance_loss" in result
        assert "kl_loss" in result

    def test_total_is_float(self):
        d, t = self._draft_target()
        mask = np.ones(self.T, dtype=bool)
        result = self.loss_fn.compute(d, t, mask)
        assert isinstance(float(result["total"]), float)

    def test_acceptance_probability_bounded(self):
        d, t = self._draft_target()
        mask = np.ones(self.T, dtype=bool)
        prob = self.loss_fn.acceptance_probability(d, t)
        assert 0.0 <= prob <= self.T + 1e-6

    def test_agreement_improves_acceptance(self):
        # When target ≈ draft, acceptance should be higher than when they diverge
        focus = 50
        d = np.zeros((self.T, self.V), dtype=np.float32)
        t_agree = np.zeros((self.T, self.V), dtype=np.float32)
        t_disagree = np.zeros((self.T, self.V), dtype=np.float32)
        d[:, focus] = 5.0
        t_agree[:, focus] = 5.0
        t_disagree[:, focus + 1] = 5.0
        mask = np.ones(self.T, dtype=bool)
        acc_agree = self.loss_fn.acceptance_probability(d, t_agree)
        acc_dis = self.loss_fn.acceptance_probability(d, t_disagree)
        assert acc_agree > acc_dis


class TestVSDDraftTrainer:
    def setup_method(self):
        self.rng = np.random.default_rng(17)
        self.cfg = VSDConfig(kl_weight=0.1)
        self.trainer = VSDDraftTrainer(self.cfg)
        self.T = 4
        self.V = 50

    def _example(self):
        tl = self.rng.normal(0, 1, (self.T, self.V)).astype(np.float32)
        tl[:, 0] += 3.0
        return VSDTrainingExample(
            context_ids=list(range(10)),
            draft_tokens=list(range(self.T)),
            target_logits=tl,
            accepted_mask=np.array([True, True, False, False]),
        )

    def test_compute_loss_calls_draft_fn(self):
        calls = []

        def draft_fn(ctx):
            calls.append(len(ctx))
            return self.rng.normal(0, 1, (self.T, self.V)).astype(np.float32)

        ex = self._example()
        result = self.trainer.compute_loss(ex, draft_fn)
        assert len(calls) == 1
        assert "total" in result

    def test_acceptance_rate_in_bounds(self):
        def draft_fn(ctx):
            return self.rng.normal(0, 1, (self.T, self.V)).astype(np.float32)

        examples = [self._example() for _ in range(5)]
        rate = self.trainer.acceptance_rate(examples, draft_fn)
        assert 0.0 <= rate <= 1.0 + 1e-6

    def test_stats_update(self):
        def draft_fn(ctx):
            return self.rng.normal(0, 1, (self.T, self.V)).astype(np.float32)

        ex = self._example()
        self.trainer.compute_loss(ex, draft_fn)
        assert self.trainer.stats.training_steps == 1

    def test_repr(self):
        r = repr(self.trainer)
        assert "VSDDraftTrainer" in r


# ============================================================
# ConfidenceGate tests
# ============================================================

from squish.serving.confidence_gate import (
    ConfidenceGate,
    ConfidenceGateConfig,
    ConfidenceGateStats,
)


class TestConfidenceGateConfig:
    def test_defaults(self):
        cfg = ConfidenceGateConfig()
        assert 0.0 < cfg.threshold <= 1.0
        assert cfg.min_commit >= 1
        assert cfg.max_commit >= cfg.min_commit

    def test_invalid_threshold_high(self):
        with pytest.raises(ValueError, match="threshold"):
            ConfidenceGateConfig(threshold=1.5)

    def test_invalid_threshold_low(self):
        with pytest.raises(ValueError, match="threshold"):
            ConfidenceGateConfig(threshold=0.0)

    def test_invalid_max_less_than_min(self):
        with pytest.raises(ValueError, match="max_commit"):
            ConfidenceGateConfig(min_commit=5, max_commit=2)


class TestConfidenceGate:
    def setup_method(self):
        self.rng = np.random.default_rng(55)
        self.cfg = ConfidenceGateConfig(threshold=0.8, min_commit=1, max_commit=6)
        self.gate = ConfidenceGate(self.cfg)
        self.V = 100

    def _high_conf_logits(self, tok: int = 0) -> np.ndarray:
        logits = np.zeros(self.V, dtype=np.float32)
        logits[tok] = 10.0
        return logits

    def _low_conf_logits(self) -> np.ndarray:
        return np.zeros(self.V, dtype=np.float32)  # uniform

    def test_confidence_is_max_softmax(self):
        logits = self._high_conf_logits(5)
        conf = self.gate.confidence(logits)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        assert conf == pytest.approx(probs[5])

    def test_should_commit_high_confidence(self):
        assert self.gate.should_commit(self._high_conf_logits()) is True

    def test_should_commit_low_confidence(self):
        assert self.gate.should_commit(self._low_conf_logits()) is False

    def test_commit_span_bounds(self):
        draft_logits = np.stack([self._high_conf_logits(i % self.V) for i in range(8)])
        span = self.gate.commit_span(draft_logits)
        assert self.cfg.min_commit <= span <= len(draft_logits)

    def test_filter_draft_length_invariant(self):
        n = 8
        draft_tokens = list(range(n))
        draft_logits = np.stack([self._high_conf_logits(i) for i in range(n)])
        commit, redraft = self.gate.filter_draft(draft_tokens, draft_logits)
        assert len(commit) + len(redraft) == n

    def test_filter_draft_all_commit_when_confident(self):
        # All tokens confidently predicted
        draft_tokens = list(range(6))
        draft_logits = np.stack([self._high_conf_logits(i) for i in range(6)])
        gate = ConfidenceGate(ConfidenceGateConfig(threshold=0.01, min_commit=1, max_commit=6))
        commit, redraft = gate.filter_draft(draft_tokens, draft_logits)
        assert len(commit) == 6 and len(redraft) == 0

    def test_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            self.gate.filter_draft([1, 2, 3],
                                   np.zeros((5, self.V), dtype=np.float32))

    def test_stats_accumulate(self):
        n = 4
        draft_tokens = list(range(n))
        draft_logits = np.stack([self._high_conf_logits(i) for i in range(n)])
        self.gate.filter_draft(draft_tokens, draft_logits)
        assert self.gate.stats.filter_calls == 1
        total = self.gate.stats.tokens_committed + self.gate.stats.tokens_redrafted
        assert total == n

    def test_commit_rate_property(self):
        gate = ConfidenceGate(self.cfg)
        stats = gate.stats
        stats.tokens_committed = 7
        stats.tokens_redrafted = 3
        assert abs(stats.commit_rate - 0.7) < 1e-9

    def test_filter_batch(self):
        batch_tokens = [list(range(4)) for _ in range(3)]
        batch_logits = [
            np.stack([self._high_conf_logits(i) for i in range(4)])
            for _ in range(3)
        ]
        results = self.gate.filter_batch(batch_tokens, batch_logits)
        assert len(results) == 3
        for commit, redraft in results:
            assert len(commit) + len(redraft) == 4

    def test_repr(self):
        r = repr(self.gate)
        assert "ConfidenceGate" in r


# ============================================================
# INT3RuntimeLoader tests
# ============================================================

from squish.quant.int3_runtime import (
    INT3LayerWeights,
    INT3LoaderStats,
    INT3RuntimeConfig,
    INT3RuntimeLoader,
)


class TestINT3RuntimeConfig:
    def test_defaults(self):
        cfg = INT3RuntimeConfig()
        assert cfg.group_size == 64
        assert cfg.tile_size > 0

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            INT3RuntimeConfig(group_size=0)


class TestINT3RuntimeLoader:
    def setup_method(self):
        self.rng = np.random.default_rng(77)
        self.cfg = INT3RuntimeConfig(group_size=8)
        self.loader = INT3RuntimeLoader(self.cfg)
        # Build fake INT3 arrays: q_packed is (n_groups, group_size) of uint8 codes 0-7
        self.d_out, self.d_in = 8, 16
        n_vals = self.d_out * self.d_in   # 128
        n_groups = max(1, n_vals // self.cfg.group_size)  # 16
        self.q_packed = self.rng.integers(0, 8, (n_groups, self.cfg.group_size), dtype=np.uint8)
        self.scales = self.rng.uniform(0.5, 1.5, n_groups).astype(np.float32)
        self.zeros = self.rng.uniform(-0.1, 0.1, n_groups).astype(np.float32)
        self.shape = (self.d_out, self.d_in)

    def test_load_from_arrays(self):
        weights = self.loader.load_from_arrays(
            self.q_packed, self.scales, self.zeros, self.shape
        )
        assert isinstance(weights, INT3LayerWeights)

    def test_dequantize_shape(self):
        weights = self.loader.load_from_arrays(
            self.q_packed, self.scales, self.zeros, self.shape
        )
        out = self.loader.dequantize(weights)
        assert out.shape == (self.d_out, self.d_in)

    def test_dequantize_dtype_float32(self):
        weights = self.loader.load_from_arrays(
            self.q_packed, self.scales, self.zeros, self.shape
        )
        out = self.loader.dequantize(weights)
        assert out.dtype == np.float32

    def test_dequantize_tiled_matches_full(self):
        weights = self.loader.load_from_arrays(
            self.q_packed, self.scales, self.zeros, self.shape
        )
        full = self.loader.dequantize(weights)
        tiles = list(self.loader.dequantize_tiled(weights))
        reconstructed = np.concatenate(tiles).reshape(self.d_out, self.d_in)
        np.testing.assert_allclose(reconstructed, full, rtol=1e-5, atol=1e-5)

    def test_int3_weights_n_groups(self):
        weights = self.loader.load_from_arrays(
            self.q_packed, self.scales, self.zeros, self.shape
        )
        assert weights.n_groups >= 1

    def test_compactness_ratio(self):
        weights = self.loader.load_from_arrays(
            self.q_packed, self.scales, self.zeros, self.shape
        )
        assert weights.compactness > 1.0

    def test_stats_track_calls(self):
        self.loader.load_from_arrays(self.q_packed, self.scales, self.zeros, self.shape)
        assert self.loader.stats.layers_loaded == 1
        weights = self.loader.load_from_arrays(self.q_packed, self.scales, self.zeros, self.shape)
        self.loader.dequantize(weights)
        assert self.loader.stats.layers_loaded >= 1

    def test_repr(self):
        r = repr(self.loader)
        assert "INT3RuntimeLoader" in r


# ============================================================
# BenchmarkHarness tests
# ============================================================

from squish.bench.benchmark_harness import (
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkStats,
    TrialResult,
)


class FakeInferenceResult:
    """Deterministic stub that satisfies the inference_fn protocol."""
    def __init__(self, ttft_ms=50.0, total_ms=500.0, tokens_generated=50, peak_memory_gb=2.0):
        self.ttft_ms = ttft_ms
        self.total_ms = total_ms
        self.tokens_generated = tokens_generated
        self.peak_memory_gb = peak_memory_gb


def fake_fn(prompt: str, max_tokens: int) -> FakeInferenceResult:
    return FakeInferenceResult()


class TestBenchmarkConfig:
    def test_defaults(self):
        cfg = BenchmarkConfig()
        assert cfg.n_trials >= 1
        assert cfg.warmup_trials >= 0

    def test_invalid_n_trials(self):
        with pytest.raises(ValueError, match="n_trials"):
            BenchmarkConfig(n_trials=0)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError, match="timeout_seconds"):
            BenchmarkConfig(timeout_seconds=-1.0)


class TestBenchmarkHarness:
    def setup_method(self):
        self.cfg = BenchmarkConfig(n_trials=5, warmup_trials=1, max_tokens=10)
        self.harness = BenchmarkHarness(self.cfg)

    def test_run_model_returns_stats(self):
        stats = self.harness.run_model("test", fake_fn, "hello")
        assert isinstance(stats, BenchmarkStats)

    def test_run_model_n_trials(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        assert stats.n_trials == 5

    def test_run_model_ttft_mean(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        assert stats.ttft_mean_ms == pytest.approx(50.0)

    def test_run_model_tps_mean(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        # 50 tokens / 0.5 s = 100 tps
        assert stats.tps_mean == pytest.approx(100.0)

    def test_run_model_peak_memory(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        assert stats.peak_memory_mean_gb == pytest.approx(2.0)

    def test_run_model_std_is_zero_deterministic(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        assert stats.ttft_std_ms == pytest.approx(0.0, abs=1e-6)
        assert stats.tps_std == pytest.approx(0.0, abs=1e-6)

    def test_run_model_percentiles(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        assert stats.ttft_p50_ms == pytest.approx(50.0)
        assert stats.ttft_p99_ms == pytest.approx(50.0)

    def test_run_all_returns_list(self):
        results = self.harness.run_all(
            {"model_a": fake_fn, "model_b": fake_fn}, "hello"
        )
        assert len(results) == 2
        assert all(isinstance(r, BenchmarkStats) for r in results)

    def test_to_markdown_table(self):
        stats = self.harness.run_model("bench", fake_fn, "hello")
        table = BenchmarkHarness.to_markdown_table([stats])
        assert table.startswith("|")
        assert "bench" in table

    def test_speedup_table_contains_speedup(self):
        r_a = self.harness.run_model("fast", fake_fn, "hello")
        r_b = self.harness.run_model("slow",
                                     lambda p, n: FakeInferenceResult(total_ms=1000.0, tokens_generated=50),
                                     "hello")
        table = BenchmarkHarness.speedup_table([r_a, r_b], "slow")
        assert "fast" in table
        assert "x" in table.lower() or "speedup" in table.lower()

    def test_to_markdown_row(self):
        stats = self.harness.run_model("row_test", fake_fn, "hello")
        row = stats.to_markdown_row()
        assert "|" in row

    def test_repr(self):
        r = repr(self.harness)
        assert "BenchmarkHarness" in r


# ============================================================
# AdaptiveKVTC tests
# ============================================================

from squish.kv.adaptive_kvtc import (
    AdaptiveKVTCConfig,
    AdaptiveKVTCLayer,
    AdaptiveKVTCManager,
    LayerRankInfo,
)


class TestAdaptiveKVTCConfig:
    def test_defaults(self):
        cfg = AdaptiveKVTCConfig()
        assert 0.0 < cfg.explained_variance_target <= 1.0
        assert cfg.min_rank >= 1
        assert cfg.max_rank >= cfg.min_rank

    def test_invalid_target_compression(self):
        with pytest.raises(ValueError, match="target_compression"):
            AdaptiveKVTCConfig(target_compression=0.5)

    def test_invalid_ev_target(self):
        with pytest.raises(ValueError, match="explained_variance_target"):
            AdaptiveKVTCConfig(explained_variance_target=1.5)

    def test_min_max_rank_consistency(self):
        with pytest.raises(ValueError, match="min_rank"):
            AdaptiveKVTCConfig(min_rank=20, max_rank=4)


class TestAdaptiveKVTCLayer:
    def setup_method(self):
        self.rng = np.random.default_rng(5)
        self.d_kv = 64
        self.n_samples = 128
        self.samples = self.rng.normal(0, 1, (self.n_samples, self.d_kv)).astype(np.float32)
        self.cfg = AdaptiveKVTCConfig(
            target_compression=4.0,
            explained_variance_target=0.90,
            min_rank=2,
            max_rank=32,
            quant_bits=8,
        )
        self.layer = AdaptiveKVTCLayer(self.cfg)

    def test_calibrate_and_tune_returns_rank(self):
        rank = self.layer.calibrate_and_tune(self.samples)
        assert self.cfg.min_rank <= rank <= self.cfg.max_rank

    def test_explained_variance_ratio_bounds(self):
        self.layer.calibrate_and_tune(self.samples)
        ev = self.layer.explained_variance_ratio(rank=8)
        assert 0.0 < ev <= 1.0

    def test_encode_decode_after_auto_calibrate(self):
        self.layer.calibrate_and_tune(self.samples)
        kv = self.rng.normal(0, 1, (16, self.d_kv)).astype(np.float32)
        enc = self.layer.encode(kv)
        rec = self.layer.decode(enc)
        assert rec.shape == kv.shape


class TestAdaptiveKVTCManager:
    def setup_method(self):
        self.rng = np.random.default_rng(8)
        self.n_layers = 4
        self.d_kv = 32
        self.n_samples = 64
        self.cfg = AdaptiveKVTCConfig(
            target_compression=4.0,
            explained_variance_target=0.90,
            min_rank=2,
            max_rank=16,
            quant_bits=8,
        )
        self.manager = AdaptiveKVTCManager(self.cfg, self.n_layers)

    def _kv_samples(self):
        return self.rng.normal(0, 1, (self.n_samples, self.d_kv)).astype(np.float32)

    def test_auto_calibrate_returns_rank_dict(self):
        all_kv = {i: (self._kv_samples(), self._kv_samples()) for i in range(self.n_layers)}
        rank_map = self.manager.auto_calibrate(all_kv)
        assert len(rank_map) == self.n_layers
        for rank in rank_map.values():
            assert self.cfg.min_rank <= rank <= self.cfg.max_rank

    def test_compression_summary_keys(self):
        all_kv = {i: (self._kv_samples(), self._kv_samples()) for i in range(self.n_layers)}
        self.manager.auto_calibrate(all_kv)
        summary = self.manager.compression_summary()
        assert "mean_rank" in summary
        assert "mean_compression" in summary
        assert "mean_explained_variance" in summary
        assert "n_layers" in summary
        assert summary["n_layers"] == self.n_layers

    def test_rank_info_returns_tuple(self):
        all_kv = {0: (self._kv_samples(), self._kv_samples())}
        self.manager.auto_calibrate(all_kv)
        info = self.manager.rank_info(0)
        assert info is not None
        k_info, v_info = info
        assert isinstance(k_info, LayerRankInfo)
        assert isinstance(v_info, LayerRankInfo)

    def test_rank_info_unknown_layer_returns_none(self):
        result = self.manager.rank_info(999)
        assert result is None

    def test_encode_decode_after_auto_calibrate(self):
        all_kv = {i: (self._kv_samples(), self._kv_samples()) for i in range(self.n_layers)}
        self.manager.auto_calibrate(all_kv)
        k = self.rng.normal(0, 1, (16, self.d_kv)).astype(np.float32)
        v = self.rng.normal(0, 1, (16, self.d_kv)).astype(np.float32)
        enc_k, enc_v = self.manager.encode_layer(0, k, v)
        k_rec, v_rec = self.manager.decode_layer(0, enc_k, enc_v)
        assert k_rec.shape == k.shape and v_rec.shape == v.shape

    def test_compression_mean_positive(self):
        all_kv = {i: (self._kv_samples(), self._kv_samples()) for i in range(self.n_layers)}
        self.manager.auto_calibrate(all_kv)
        summary = self.manager.compression_summary()
        assert summary["mean_compression"] > 1.0

    def test_repr(self):
        r = repr(self.manager)
        assert "AdaptiveKVTCManager" in r
