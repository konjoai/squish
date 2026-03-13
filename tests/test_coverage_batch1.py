"""Coverage tests for batch 1 modules: mix_kvq, quant_aware, quant_calib, fp8_quant,
long_context_chunk, adaptive_batcher, schema_validator, prefix_pool."""
import pytest
import numpy as np

from squish.kv.mix_kvq import (
    MixKVQConfig,
    ChannelScorer,
    MixKVQQuantizer,
    MixKVQStats,
)
from squish.quant.quant_aware import (
    QAConfig,
    QuantAwareCalibrator,
    QAStats,
)
from squish.quant.quant_calib import (
    CalibConfig,
    CalibResult,
    QuantCalibrator,
    CalibStats,
)
from squish.quant.fp8_quant import (
    FP8Config,
    FP8Tensor,
    FP8Quantizer,
    fp8_encode_e4m3,
    fp8_encode_e5m2,
    fp8_decode,
)
from squish.streaming.long_context_chunk import (
    ChunkConfig,
    LongContextChunker,
    ChunkStats,
)
from squish.serving.adaptive_batcher import (
    BatchObjective,
    BatchDecision,
    AdaptiveBatchController,
)
from squish.grammar.schema_validator import (
    ValidationResult,
    SchemaValidator,
)
from squish.kv.prefix_pool import (
    PrefixPoolConfig,
    PrefixEntry,
    PrefixPool,
    PrefixPoolStats,
)


# ===========================================================================
# mix_kvq
# ===========================================================================

class TestMixKVQConfig:
    def test_defaults(self):
        cfg = MixKVQConfig()
        assert cfg.fp16_fraction == 0.10
        assert cfg.int2_fraction == 0.50
        assert cfg.importance_weight == 0.50
        assert cfg.group_size == 64
        assert cfg.calibration_history == 128

    def test_valid_custom(self):
        cfg = MixKVQConfig(fp16_fraction=0.2, int2_fraction=0.3)
        assert cfg.fp16_fraction == 0.2
        assert cfg.int2_fraction == 0.3

    def test_fp16_fraction_below_zero(self):
        with pytest.raises(ValueError, match="fp16_fraction"):
            MixKVQConfig(fp16_fraction=-0.1)

    def test_fp16_fraction_above_one(self):
        with pytest.raises(ValueError, match="fp16_fraction"):
            MixKVQConfig(fp16_fraction=1.1)

    def test_int2_fraction_below_zero(self):
        with pytest.raises(ValueError, match="int2_fraction"):
            MixKVQConfig(int2_fraction=-0.1)

    def test_int2_fraction_above_one(self):
        with pytest.raises(ValueError, match="int2_fraction"):
            MixKVQConfig(int2_fraction=1.1)

    def test_fractions_sum_exceeds_one(self):
        with pytest.raises(ValueError, match=r"fp16_fraction \+ int2_fraction"):
            MixKVQConfig(fp16_fraction=0.6, int2_fraction=0.6)

    def test_importance_weight_below_zero(self):
        with pytest.raises(ValueError, match="importance_weight"):
            MixKVQConfig(importance_weight=-0.01)

    def test_importance_weight_above_one(self):
        with pytest.raises(ValueError, match="importance_weight"):
            MixKVQConfig(importance_weight=1.01)

    def test_fractions_sum_exactly_one(self):
        cfg = MixKVQConfig(fp16_fraction=0.5, int2_fraction=0.5)
        assert cfg.fp16_fraction + cfg.int2_fraction == 1.0


class TestChannelScorer:
    def setup_method(self):
        self.rng = np.random.default_rng(42)
        self.n = 16
        self.cfg = MixKVQConfig(fp16_fraction=0.1, int2_fraction=0.5, calibration_history=8)
        self.scorer = ChannelScorer(self.n, self.cfg)

    def test_n_channels_property(self):
        assert self.scorer.n_channels == self.n

    def test_difficulty_before_any_records(self):
        d = self.scorer.difficulty()
        assert d.shape == (self.n,)
        assert np.all(d == 0.0)

    def test_difficulty_with_partial_history(self):
        for _ in range(4):
            kv = self.rng.standard_normal(self.n).astype(np.float32)
            self.scorer.record(kv)
        d = self.scorer.difficulty()
        assert d.shape == (self.n,)
        assert np.all(d >= 0.0)

    def test_difficulty_with_full_history(self):
        for _ in range(10):
            kv = self.rng.standard_normal(self.n).astype(np.float32)
            self.scorer.record(kv)
        d = self.scorer.difficulty()
        assert d.shape == (self.n,)
        assert d.max() <= 1.0 + 1e-6

    def test_difficulty_normalised_to_one(self):
        for _ in range(8):
            kv = self.rng.standard_normal(self.n).astype(np.float32)
            self.scorer.record(kv)
        d = self.scorer.difficulty()
        assert abs(d.max() - 1.0) < 1e-5

    def test_difficulty_uniform_history(self):
        # All rows identical → outlier ratio = 1 everywhere → dmax > 0, result uniform
        for _ in range(8):
            kv = np.ones(self.n, dtype=np.float32)
            self.scorer.record(kv)
        d = self.scorer.difficulty()
        assert np.all(d >= 0.0)

    def test_importance_empty_key_matrix(self):
        q = self.rng.standard_normal(self.n).astype(np.float32)
        km = np.zeros((0, self.n), dtype=np.float32)
        imp = self.scorer.importance(q, km)
        assert imp.shape == (self.n,)
        assert np.all(imp == 0.0)

    def test_importance_non_empty(self):
        q = self.rng.standard_normal(self.n).astype(np.float32)
        km = self.rng.standard_normal((10, self.n)).astype(np.float32)
        imp = self.scorer.importance(q, km)
        assert imp.shape == (self.n,)
        assert imp.max() <= 1.0 + 1e-6

    def test_importance_all_zeros(self):
        q = np.zeros(self.n, dtype=np.float32)
        km = np.zeros((5, self.n), dtype=np.float32)
        imp = self.scorer.importance(q, km)
        assert np.all(imp == 0.0)

    def test_score_shape_and_range(self):
        q = self.rng.standard_normal(self.n).astype(np.float32)
        km = self.rng.standard_normal((10, self.n)).astype(np.float32)
        s = self.scorer.score(q, km)
        assert s.shape == (self.n,)
        assert np.all(s >= 0.0)

    def test_score_zero_combined(self):
        # No records + empty key matrix → all zeros → max == 0 → returns unchanged zeros
        scorer = ChannelScorer(4, MixKVQConfig())
        q = np.zeros(4, dtype=np.float32)
        km = np.zeros((0, 4), dtype=np.float32)
        s = scorer.score(q, km)
        assert np.all(s == 0.0)

    def test_assign_bits_values(self):
        q = self.rng.standard_normal(self.n).astype(np.float32)
        km = self.rng.standard_normal((10, self.n)).astype(np.float32)
        bits = self.scorer.assign_bits(q, km)
        assert bits.shape == (self.n,)
        assert set(np.unique(bits)).issubset({2, 4, 16})

    def test_assign_bits_fp16_at_top(self):
        q = self.rng.standard_normal(self.n).astype(np.float32)
        km = self.rng.standard_normal((10, self.n)).astype(np.float32)
        bits = self.scorer.assign_bits(q, km)
        assert np.any(bits == 16)

    def test_assign_bits_int2_at_bottom(self):
        q = self.rng.standard_normal(self.n).astype(np.float32)
        km = self.rng.standard_normal((10, self.n)).astype(np.float32)
        bits = self.scorer.assign_bits(q, km)
        assert np.any(bits == 2)

    def test_record_fills_history_wraparound(self):
        # Fill beyond history size to test wraparound
        for i in range(self.cfg.calibration_history + 3):
            kv = self.rng.standard_normal(self.n).astype(np.float32)
            self.scorer.record(kv)
        d = self.scorer.difficulty()
        assert d.shape == (self.n,)

    def test_default_config_when_none(self):
        scorer = ChannelScorer(8)
        assert scorer._cfg is not None
        assert isinstance(scorer._cfg, MixKVQConfig)


class TestMixKVQQuantizer:
    def setup_method(self):
        self.rng = np.random.default_rng(7)
        self.cfg = MixKVQConfig(fp16_fraction=0.25, int2_fraction=0.5, group_size=4)
        self.quant = MixKVQQuantizer(self.cfg)
        self.n = 8

    def _make_bit_map(self, pattern):
        return np.array(pattern, dtype=np.int32)

    def test_quantize_returns_three_items(self):
        kv = self.rng.standard_normal(self.n).astype(np.float32)
        bits = self._make_bit_map([16, 16, 4, 4, 4, 4, 2, 2])
        segs, scales, bm = self.quant.quantize(kv, bits)
        assert isinstance(segs, list)
        assert isinstance(scales, np.ndarray)
        assert isinstance(bm, np.ndarray)

    def test_quantize_dequantize_fp16_segment(self):
        kv = np.array([1.5, -2.5, 3.0, -1.0], dtype=np.float32)
        bits = self._make_bit_map([16, 16, 16, 16])
        segs, scales, bm = self.quant.quantize(kv, bits)
        out = self.quant.dequantize(segs, scales, bm)
        np.testing.assert_allclose(out, kv.astype(np.float16).astype(np.float32), atol=1e-3)

    def test_quantize_dequantize_int4_segment(self):
        kv = np.array([0.5, -0.5, 0.3, -0.3], dtype=np.float32)
        bits = self._make_bit_map([4, 4, 4, 4])
        segs, scales, bm = self.quant.quantize(kv, bits)
        out = self.quant.dequantize(segs, scales, bm)
        assert out.shape == (4,)
        assert out.dtype == np.float32

    def test_quantize_dequantize_int2_segment(self):
        kv = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        bits = self._make_bit_map([2, 2, 2, 2])
        segs, scales, bm = self.quant.quantize(kv, bits)
        out = self.quant.dequantize(segs, scales, bm)
        assert out.shape == (4,)

    def test_quantize_mixed_bits(self):
        kv = self.rng.standard_normal(self.n).astype(np.float32)
        bits = self._make_bit_map([16, 16, 4, 4, 2, 2, 4, 16])
        segs, scales, bm = self.quant.quantize(kv, bits)
        out = self.quant.dequantize(segs, scales, bm)
        assert out.shape == (self.n,)
        assert out.dtype == np.float32

    def test_default_config_when_none(self):
        q = MixKVQQuantizer()
        assert q._cfg is not None

    def test_quantize_bit_map_copied(self):
        kv = self.rng.standard_normal(self.n).astype(np.float32)
        bits = self._make_bit_map([4] * self.n)
        _, _, bm = self.quant.quantize(kv, bits)
        bits[0] = 999
        assert bm[0] == 4  # copy, not reference


class TestMixKVQStats:
    def test_initial_avg_bits_zero(self):
        s = MixKVQStats()
        assert s.avg_bits == 0.0

    def test_record_single_bit_map(self):
        s = MixKVQStats()
        bm = np.array([16, 4, 2, 4], dtype=np.int32)
        s.record(bm)
        assert s.total_quantized_channels == 4
        assert s.fp16_count == 1
        assert s.int4_count == 2
        assert s.int2_count == 1

    def test_avg_bits_calculation(self):
        s = MixKVQStats()
        bm = np.array([16, 4, 2], dtype=np.int32)
        s.record(bm)
        expected = (16 + 4 + 2) / 3
        assert abs(s.avg_bits - expected) < 1e-6

    def test_reset_clears_counts(self):
        s = MixKVQStats()
        bm = np.array([16, 4, 2], dtype=np.int32)
        s.record(bm)
        s.reset()
        assert s.total_quantized_channels == 0
        assert s.fp16_count == 0
        assert s.int4_count == 0
        assert s.int2_count == 0
        assert s.avg_bits == 0.0

    def test_multiple_records_accumulate(self):
        s = MixKVQStats()
        bm = np.array([16, 4], dtype=np.int32)
        s.record(bm)
        s.record(bm)
        assert s.total_quantized_channels == 4
        assert s.fp16_count == 2
        assert s.int4_count == 2


# ===========================================================================
# quant_aware
# ===========================================================================

class TestQAConfig:
    def test_defaults(self):
        cfg = QAConfig()
        assert cfg.method == "percentile"
        assert cfg.n_bits == 8
        assert cfg.per_channel is True

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            QAConfig(method="unknown")

    def test_percentile_zero(self):
        with pytest.raises(ValueError, match="percentile"):
            QAConfig(percentile=0.0)

    def test_percentile_above_100(self):
        with pytest.raises(ValueError, match="percentile"):
            QAConfig(percentile=100.1)

    def test_n_bits_too_low(self):
        with pytest.raises(ValueError, match="n_bits"):
            QAConfig(n_bits=1)

    def test_n_bits_too_high(self):
        with pytest.raises(ValueError, match="n_bits"):
            QAConfig(n_bits=17)

    def test_mse_grid_steps_zero(self):
        with pytest.raises(ValueError, match="mse_grid_steps"):
            QAConfig(mse_grid_steps=0)

    def test_valid_minmax(self):
        cfg = QAConfig(method="minmax")
        assert cfg.method == "minmax"

    def test_valid_mse(self):
        cfg = QAConfig(method="mse")
        assert cfg.method == "mse"


class TestQAStats:
    def test_dynamic_range_zero_when_min_is_zero(self):
        s = QAStats(n_batches=1, n_channels=4, method="minmax", max_scale=1.0, min_scale=0.0)
        assert s.dynamic_range_db == 0.0

    def test_dynamic_range_zero_when_max_is_zero(self):
        s = QAStats(n_batches=1, n_channels=4, method="minmax", max_scale=0.0, min_scale=0.0)
        assert s.dynamic_range_db == 0.0

    def test_dynamic_range_positive(self):
        s = QAStats(n_batches=1, n_channels=4, method="minmax", max_scale=100.0, min_scale=1.0)
        assert s.dynamic_range_db > 0.0

    def test_dynamic_range_formula(self):
        import math
        s = QAStats(n_batches=1, n_channels=4, method="minmax", max_scale=10.0, min_scale=1.0)
        expected = 20.0 * math.log10(10.0)
        assert abs(s.dynamic_range_db - expected) < 1e-6


class TestQuantAwareCalibrator:
    def setup_method(self):
        self.rng = np.random.default_rng(99)

    def _make_activations(self, batch=4, channels=8):
        return self.rng.standard_normal((batch, channels)).astype(np.float32)

    def test_record_and_compute_scales_minmax(self):
        cfg = QAConfig(method="minmax", n_bits=8)
        cal = QuantAwareCalibrator(cfg)
        cal.record(self._make_activations())
        scales = cal.compute_scales()
        assert scales.shape == (8,)
        assert np.all(scales > 0.0)

    def test_record_and_compute_scales_percentile(self):
        cfg = QAConfig(method="percentile", percentile=99.9, n_bits=8)
        cal = QuantAwareCalibrator(cfg)
        cal.record(self._make_activations())
        scales = cal.compute_scales()
        assert scales.shape == (8,)

    def test_record_and_compute_scales_mse(self):
        cfg = QAConfig(method="mse", n_bits=4, mse_grid_steps=8)
        cal = QuantAwareCalibrator(cfg)
        cal.record(self._make_activations())
        scales = cal.compute_scales()
        assert scales.shape == (8,)

    def test_compute_scales_raises_before_record(self):
        cal = QuantAwareCalibrator(QAConfig())
        with pytest.raises(RuntimeError, match="No activations recorded"):
            cal.compute_scales()

    def test_channel_mismatch_raises(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        cal.record(self._make_activations(4, 8))
        with pytest.raises(ValueError, match="Channel count mismatch"):
            cal.record(self._make_activations(4, 4))

    def test_3d_activations(self):
        cfg = QAConfig(method="minmax")
        cal = QuantAwareCalibrator(cfg)
        acts = self.rng.standard_normal((2, 5, 8)).astype(np.float32)
        cal.record(acts)
        scales = cal.compute_scales()
        assert scales.shape == (8,)

    def test_invalid_shape_raises(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        with pytest.raises(ValueError, match="must be 2-D"):
            cal.record(np.ones((2, 3, 4, 5), dtype=np.float32))

    def test_global_scale_per_channel_false(self):
        cfg = QAConfig(method="minmax", per_channel=False)
        cal = QuantAwareCalibrator(cfg)
        cal.record(self._make_activations())
        scales = cal.compute_scales()
        # All scales equal when per_channel=False
        assert np.all(scales == scales[0])

    def test_n_batches_property(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        assert cal.n_batches == 0
        cal.record(self._make_activations())
        assert cal.n_batches == 1

    def test_channels_property_none_before_record(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        assert cal.channels is None

    def test_channels_property_after_record(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        cal.record(self._make_activations(4, 8))
        assert cal.channels == 8

    def test_reset(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        cal.record(self._make_activations())
        cal.reset()
        assert cal.n_batches == 0
        assert cal.channels is None

    def test_stats_before_record(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        s = cal.stats()
        assert s.n_batches == 0
        assert s.n_channels == 0

    def test_stats_after_record(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        cal.record(self._make_activations(4, 8))
        s = cal.stats()
        assert s.n_batches == 1
        assert s.n_channels == 8
        assert s.max_scale > 0.0

    def test_scale_positive_where_nonzero_activations(self):
        cfg = QAConfig(method="minmax")
        cal = QuantAwareCalibrator(cfg)
        cal.record(self._make_activations())
        scales = cal.compute_scales()
        assert np.all(scales > 0.0)

    def test_multiple_records_accumulate(self):
        cal = QuantAwareCalibrator(QAConfig(method="minmax"))
        for _ in range(3):
            cal.record(self._make_activations())
        assert cal.n_batches == 3


# ===========================================================================
# quant_calib
# ===========================================================================

class TestCalibConfig:
    def test_defaults(self):
        cfg = CalibConfig()
        assert cfg.method == "mse"
        assert cfg.n_bits == 8

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method must be"):
            CalibConfig(method="invalid")

    def test_invalid_n_bits(self):
        with pytest.raises(ValueError, match="n_bits must be"):
            CalibConfig(n_bits=16)

    def test_invalid_n_bits_too_low(self):
        with pytest.raises(ValueError, match="n_bits must be"):
            CalibConfig(n_bits=2)

    def test_percentile_at_boundary_low(self):
        with pytest.raises(ValueError, match="percentile"):
            CalibConfig(percentile=50.0)

    def test_percentile_at_boundary_high(self):
        with pytest.raises(ValueError, match="percentile"):
            CalibConfig(percentile=100.0)

    def test_valid_minmax_4bit(self):
        cfg = CalibConfig(method="minmax", n_bits=4)
        assert cfg.n_bits == 4

    def test_valid_percentile(self):
        cfg = CalibConfig(method="percentile", percentile=99.9)
        assert cfg.method == "percentile"


class TestQuantCalibrator:
    def setup_method(self):
        self.rng = np.random.default_rng(55)

    def _acts(self, shape=(32, 16)):
        return self.rng.standard_normal(shape).astype(np.float32)

    def test_calibrate_minmax_2d(self):
        cal = QuantCalibrator(CalibConfig(method="minmax"))
        result = cal.calibrate(self._acts((32, 16)))
        assert isinstance(result, CalibResult)
        assert result.scales.shape == (16,)
        assert result.method == "minmax"
        assert result.n_bits == 8

    def test_calibrate_percentile_2d(self):
        cal = QuantCalibrator(CalibConfig(method="percentile"))
        result = cal.calibrate(self._acts((32, 16)))
        assert result.scales.shape == (16,)

    def test_calibrate_mse_2d(self):
        cal = QuantCalibrator(CalibConfig(method="mse"))
        result = cal.calibrate(self._acts((4, 4)))
        assert result.scales.shape == (4,)

    def test_calibrate_3d_input(self):
        acts = self.rng.standard_normal((2, 8, 16)).astype(np.float32)
        cal = QuantCalibrator(CalibConfig(method="minmax"))
        result = cal.calibrate(acts)
        assert result.scales.shape == (16,)

    def test_calibrate_per_channel_false(self):
        cal = QuantCalibrator(CalibConfig(method="minmax", per_channel=False))
        result = cal.calibrate(self._acts())
        # Global scale: scalar array
        assert result.scales.ndim == 0 or result.scales.shape == ()

    def test_calibrate_n_bits_4(self):
        cal = QuantCalibrator(CalibConfig(method="minmax", n_bits=4))
        result = cal.calibrate(self._acts())
        assert result.n_bits == 4

    def test_stats_increments(self):
        cal = QuantCalibrator(CalibConfig(method="minmax"))
        assert cal.stats.total_calibrations == 0
        cal.calibrate(self._acts((4, 8)))
        assert cal.stats.total_calibrations == 1
        assert cal.stats.total_channels == 8
        cal.calibrate(self._acts((4, 8)))
        assert cal.stats.total_calibrations == 2
        assert cal.stats.total_channels == 16

    def test_zero_channel_returns_unit_scale(self):
        acts = np.zeros((4, 4), dtype=np.float32)
        cal = QuantCalibrator(CalibConfig(method="minmax"))
        result = cal.calibrate(acts)
        assert np.all(result.scales == 1.0)

    def test_mse_scale_static(self):
        col = np.array([1.0, -1.0, 0.5, -0.5], dtype=np.float32)
        scale = QuantCalibrator._mse_scale(col, float(np.max(np.abs(col))), 255)
        assert scale > 0.0

    def test_compute_scale_minmax(self):
        col = np.array([2.0, -3.0, 1.0], dtype=np.float32)
        cfg = CalibConfig(method="minmax")
        s = QuantCalibrator._compute_scale(col, 255, cfg)
        expected = (3.0 * 2.0) / 255
        assert abs(s - expected) < 1e-5

    def test_compute_scale_percentile(self):
        rng = np.random.default_rng(1)
        col = rng.standard_normal(100).astype(np.float32)
        cfg = CalibConfig(method="percentile", percentile=99.9)
        s = QuantCalibrator._compute_scale(col, 255, cfg)
        assert s > 0.0

    def test_compute_scale_degenerate_zero(self):
        col = np.zeros(10, dtype=np.float32)
        cfg = CalibConfig(method="minmax")
        s = QuantCalibrator._compute_scale(col, 255, cfg)
        assert s == 1.0


# ===========================================================================
# fp8_quant
# ===========================================================================

class TestFP8Config:
    def test_defaults(self):
        cfg = FP8Config()
        assert cfg.fmt == "e4m3"
        assert cfg.block_size == 128

    def test_invalid_fmt(self):
        with pytest.raises(ValueError, match="fmt must be"):
            FP8Config(fmt="e6m1")

    def test_invalid_block_size_zero(self):
        with pytest.raises(ValueError, match="block_size must be a power of 2"):
            FP8Config(block_size=0)

    def test_invalid_block_size_non_power_of_two(self):
        with pytest.raises(ValueError, match="power of 2"):
            FP8Config(block_size=3)

    def test_max_val_e4m3(self):
        cfg = FP8Config(fmt="e4m3")
        assert cfg.max_val == 448.0

    def test_max_val_e5m2(self):
        cfg = FP8Config(fmt="e5m2")
        assert cfg.max_val == 57344.0

    def test_tiny_val_e4m3(self):
        cfg = FP8Config(fmt="e4m3")
        assert cfg.tiny_val == 2.0 ** -9

    def test_tiny_val_e5m2(self):
        cfg = FP8Config(fmt="e5m2")
        assert cfg.tiny_val == 2.0 ** -14

    def test_mantissa_bits_e4m3(self):
        assert FP8Config(fmt="e4m3").mantissa_bits == 3

    def test_mantissa_bits_e5m2(self):
        assert FP8Config(fmt="e5m2").mantissa_bits == 2

    def test_exponent_bits_e4m3(self):
        assert FP8Config(fmt="e4m3").exponent_bits == 4

    def test_exponent_bits_e5m2(self):
        assert FP8Config(fmt="e5m2").exponent_bits == 5


class TestFP8Tensor:
    def test_compression_ratio(self):
        data = np.zeros(128, dtype=np.uint8)
        scales = np.ones(1, dtype=np.float32)
        t = FP8Tensor(data=data, scales=scales, shape=(128,), fmt="e4m3")
        assert t.compression_ratio == 4.0

    def test_n_elements(self):
        data = np.zeros(12, dtype=np.uint8)
        scales = np.ones(1, dtype=np.float32)
        t = FP8Tensor(data=data, scales=scales, shape=(3, 4), fmt="e4m3")
        assert t.n_elements == 12

    def test_n_elements_1d(self):
        data = np.zeros(64, dtype=np.uint8)
        scales = np.ones(1, dtype=np.float32)
        t = FP8Tensor(data=data, scales=scales, shape=(64,), fmt="e4m3")
        assert t.n_elements == 64


class TestFP8Codec:
    def setup_method(self):
        self.rng = np.random.default_rng(21)

    def test_encode_e4m3_returns_uint8(self):
        x = self.rng.standard_normal(64).astype(np.float32)
        codes, scale = fp8_encode_e4m3(x)
        assert codes.dtype == np.uint8
        assert scale > 0.0

    def test_encode_e5m2_returns_uint8(self):
        x = self.rng.standard_normal(64).astype(np.float32)
        codes, scale = fp8_encode_e5m2(x)
        assert codes.dtype == np.uint8

    def test_encode_e4m3_with_provided_scale(self):
        x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        codes, scale = fp8_encode_e4m3(x, scale=1.0)
        assert scale == 1.0

    def test_encode_e5m2_with_provided_scale(self):
        x = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        codes, scale = fp8_encode_e5m2(x, scale=1.0)
        assert scale == 1.0

    def test_decode_e4m3_shape_preserved(self):
        x = self.rng.standard_normal(64).astype(np.float32)
        codes, scale = fp8_encode_e4m3(x)
        out = fp8_decode(codes, scale, fmt="e4m3")
        assert out.shape == (64,)
        assert out.dtype == np.float32

    def test_decode_e5m2(self):
        x = self.rng.standard_normal(64).astype(np.float32)
        codes, scale = fp8_encode_e5m2(x)
        out = fp8_decode(codes, scale, fmt="e5m2")
        assert out.shape == (64,)

    def test_encode_all_zeros(self):
        x = np.zeros(32, dtype=np.float32)
        codes, scale = fp8_encode_e4m3(x)
        assert scale == 1.0
        out = fp8_decode(codes, scale, fmt="e4m3")
        assert np.all(out == 0.0)

    def test_roundtrip_approximate(self):
        x = np.array([1.0, 2.0, -1.0, -2.0, 0.5], dtype=np.float32)
        codes, scale = fp8_encode_e4m3(x)
        out = fp8_decode(codes, scale, fmt="e4m3")
        # FP8 is lossy; just check signs are preserved
        assert np.all(np.sign(out[np.abs(x) > 0.01]) == np.sign(x[np.abs(x) > 0.01]))


class TestFP8Quantizer:
    def setup_method(self):
        self.rng = np.random.default_rng(33)

    def test_encode_decode_e4m3_per_channel(self):
        cfg = FP8Config(fmt="e4m3", per_channel=True)
        q = FP8Quantizer(cfg)
        x = self.rng.standard_normal((4, 32)).astype(np.float32)
        t = q.encode(x)
        out = q.decode(t)
        assert out.shape == x.shape

    def test_encode_decode_e5m2_per_channel(self):
        cfg = FP8Config(fmt="e5m2", per_channel=True)
        q = FP8Quantizer(cfg)
        x = self.rng.standard_normal((4, 32)).astype(np.float32)
        t = q.encode(x)
        out = q.decode(t)
        assert out.shape == x.shape

    def test_encode_decode_per_block(self):
        cfg = FP8Config(fmt="e4m3", block_size=16, per_channel=False)
        q = FP8Quantizer(cfg)
        x = self.rng.standard_normal(64).astype(np.float32)
        t = q.encode(x)
        out = q.decode(t)
        assert out.shape == (64,)

    def test_encode_saturate_nan(self):
        cfg = FP8Config(fmt="e4m3", saturate_nan=True)
        q = FP8Quantizer(cfg)
        x = np.array([np.nan, np.inf, -np.inf, 1.0], dtype=np.float32)
        t = q.encode(x)
        assert not np.any(np.isnan(q.decode(t)))

    def test_encode_1d_per_block(self):
        cfg = FP8Config(fmt="e4m3", per_channel=False, block_size=8)
        q = FP8Quantizer(cfg)
        x = self.rng.standard_normal(24).astype(np.float32)
        t = q.encode(x)
        assert t.data.dtype == np.uint8

    def test_relative_error_all_zeros(self):
        cfg = FP8Config(fmt="e4m3", per_channel=False, block_size=8)
        q = FP8Quantizer(cfg)
        orig = np.zeros(16, dtype=np.float32)
        decoded = np.zeros(16, dtype=np.float32)
        err = q.relative_error(orig, decoded)
        assert err == 0.0

    def test_relative_error_small_for_fp32(self):
        cfg = FP8Config(fmt="e4m3", per_channel=True)
        q = FP8Quantizer(cfg)
        x = self.rng.standard_normal((4, 32)).astype(np.float32)
        t = q.encode(x)
        out = q.decode(t)
        err = q.relative_error(x, out)
        assert 0.0 <= err < 1.0  # reasonable bound for FP8

    def test_fp8_tensor_stored_with_correct_fmt(self):
        cfg = FP8Config(fmt="e5m2", per_channel=False, block_size=16)
        q = FP8Quantizer(cfg)
        x = self.rng.standard_normal(32).astype(np.float32)
        t = q.encode(x)
        assert t.fmt == "e5m2"


# ===========================================================================
# long_context_chunk
# ===========================================================================

class TestChunkConfig:
    def test_defaults(self):
        cfg = ChunkConfig()
        assert cfg.max_chunk_size == 512
        assert cfg.min_chunk_size == 64

    def test_min_chunk_too_small(self):
        with pytest.raises(ValueError, match="min_chunk_size must be >= 1"):
            ChunkConfig(min_chunk_size=0)

    def test_max_chunk_too_small(self):
        with pytest.raises(ValueError, match="max_chunk_size must be >= 1"):
            ChunkConfig(max_chunk_size=0)

    def test_min_equals_max(self):
        with pytest.raises(ValueError, match="strictly less than max_chunk_size"):
            ChunkConfig(min_chunk_size=64, max_chunk_size=64)

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="strictly less than max_chunk_size"):
            ChunkConfig(min_chunk_size=100, max_chunk_size=50)

    def test_boundary_sensitivity_zero(self):
        with pytest.raises(ValueError, match="boundary_sensitivity must be > 0"):
            ChunkConfig(boundary_sensitivity=0.0)

    def test_embed_dim_zero(self):
        with pytest.raises(ValueError, match="embed_dim must be >= 1"):
            ChunkConfig(embed_dim=0)

    def test_valid_custom(self):
        cfg = ChunkConfig(max_chunk_size=256, min_chunk_size=32, embed_dim=64)
        assert cfg.max_chunk_size == 256


class TestChunkStats:
    def test_avg_chunk_size_zero_when_no_chunks(self):
        s = ChunkStats()
        assert s.avg_chunk_size == 0.0

    def test_avg_chunk_size_calculation(self):
        s = ChunkStats(total_chunk_calls=2, total_tokens_chunked=100, total_chunks_produced=4)
        assert s.avg_chunk_size == 25.0


class TestLongContextChunker:
    def setup_method(self):
        self.rng = np.random.default_rng(77)
        self.cfg = ChunkConfig(max_chunk_size=128, min_chunk_size=32, embed_dim=16)
        self.chunker = LongContextChunker(self.cfg)

    def _embeds(self, seq_len, embed_dim=16):
        return self.rng.standard_normal((seq_len, embed_dim)).astype(np.float32)

    def test_short_sequence_single_chunk(self):
        emb = self._embeds(64)
        chunks = self.chunker.chunk(emb)
        assert chunks == [(0, 64)]

    def test_chunks_cover_full_sequence(self):
        emb = self._embeds(512)
        chunks = self.chunker.chunk(emb)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 512
        # Contiguous
        for i in range(1, len(chunks)):
            assert chunks[i][0] == chunks[i-1][1]

    def test_no_chunk_exceeds_max(self):
        emb = self._embeds(512)
        chunks = self.chunker.chunk(emb)
        for s, e in chunks:
            assert e - s <= self.cfg.max_chunk_size

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            self.chunker.chunk(np.ones((4, 8, 16), dtype=np.float32))

    def test_wrong_embed_dim_raises(self):
        with pytest.raises(ValueError, match="embed_dim"):
            self.chunker.chunk(np.ones((128, 8), dtype=np.float32))

    def test_stats_updated_after_chunk(self):
        emb = self._embeds(256)
        _ = self.chunker.chunk(emb)
        assert self.chunker.stats.total_chunk_calls == 1
        assert self.chunker.stats.total_tokens_chunked == 256

    def test_multiple_calls_accumulate_stats(self):
        for _ in range(3):
            self.chunker.chunk(self._embeds(256))
        assert self.chunker.stats.total_chunk_calls == 3

    def test_repr_contains_config(self):
        r = repr(self.chunker)
        assert "LongContextChunker" in r
        assert "128" in r

    def test_exact_max_chunk_single_chunk(self):
        emb = self._embeds(self.cfg.max_chunk_size)
        chunks = self.chunker.chunk(emb)
        assert chunks == [(0, self.cfg.max_chunk_size)]

    def test_n_complete_windows_less_than_2_fixed_split(self):
        # min_chunk_size=32, max=128; seq_len=33 → 1 complete window (n_complete<2)
        emb = self._embeds(33)
        chunks = self.chunker.chunk(emb)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 33

    def test_uniform_embeddings_fixed_split(self):
        # Uniform embeddings → std=0 → fixed split
        emb = np.ones((256, 16), dtype=np.float32)
        chunks = self.chunker.chunk(emb)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 256
        for s, e in chunks:
            assert e - s <= self.cfg.max_chunk_size

    def test_high_sensitivity_fewer_chunks(self):
        cfg_hi = ChunkConfig(max_chunk_size=128, min_chunk_size=32,
                              boundary_sensitivity=100.0, embed_dim=16)
        chunker_hi = LongContextChunker(cfg_hi)
        emb = self.rng.standard_normal((512, 16)).astype(np.float32)
        chunks_hi = chunker_hi.chunk(emb)
        cfg_lo = ChunkConfig(max_chunk_size=128, min_chunk_size=32,
                              boundary_sensitivity=0.1, embed_dim=16)
        chunker_lo = LongContextChunker(cfg_lo)
        chunks_lo = chunker_lo.chunk(emb)
        # High sensitivity → more forced splits (or same, but at least high sens result valid)
        assert chunks_hi[-1][1] == 512

    def test_fixed_split_method(self):
        chunks = self.chunker._fixed_split(300)
        assert chunks[0][0] == 0
        assert chunks[-1][1] == 300
        for s, e in chunks:
            assert e - s <= 128


# ===========================================================================
# adaptive_batcher
# ===========================================================================

class TestBatchObjective:
    def test_defaults(self):
        obj = BatchObjective()
        assert obj.mode == "throughput"
        assert obj.max_batch_size == 32
        assert obj.min_batch_size == 1

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode must be"):
            BatchObjective(mode="invalid")

    def test_nonpositive_target_latency(self):
        with pytest.raises(ValueError, match="target_latency_ms"):
            BatchObjective(mode="latency", target_latency_ms=0.0)

    def test_max_batch_zero(self):
        with pytest.raises(ValueError, match="max_batch_size"):
            BatchObjective(max_batch_size=0)

    def test_min_batch_zero(self):
        with pytest.raises(ValueError, match="min_batch_size"):
            BatchObjective(min_batch_size=0)

    def test_min_greater_than_max(self):
        with pytest.raises(ValueError, match="min_batch_size"):
            BatchObjective(min_batch_size=10, max_batch_size=5)

    def test_valid_latency_mode(self):
        obj = BatchObjective(mode="latency", target_latency_ms=50.0)
        assert obj.mode == "latency"


class TestAdaptiveBatchController:
    def _make_ctrl(self, mode="throughput", target=100.0, max_bs=16, min_bs=1):
        obj = BatchObjective(mode=mode, target_latency_ms=target,
                              max_batch_size=max_bs, min_batch_size=min_bs)
        return AdaptiveBatchController(obj)

    def test_next_batch_negative_queue_raises(self):
        ctrl = self._make_ctrl()
        with pytest.raises(ValueError, match="queue_depth"):
            ctrl.next_batch(-1)

    def test_record_batch_size_zero_raises(self):
        ctrl = self._make_ctrl()
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            ctrl.record_observation(0, 10.0)

    def test_record_negative_latency_raises(self):
        ctrl = self._make_ctrl()
        with pytest.raises(ValueError, match="latency_ms must be >= 0"):
            ctrl.record_observation(4, -1.0)

    def test_throughput_mode_uses_queue_depth(self):
        ctrl = self._make_ctrl(mode="throughput", max_bs=8)
        d = ctrl.next_batch(queue_depth=5)
        assert d.batch_size == 5

    def test_throughput_mode_clamps_to_max(self):
        ctrl = self._make_ctrl(mode="throughput", max_bs=4)
        d = ctrl.next_batch(queue_depth=100)
        assert d.batch_size == 4

    def test_throughput_mode_zero_queue(self):
        ctrl = self._make_ctrl(mode="throughput", min_bs=1)
        d = ctrl.next_batch(queue_depth=0)
        assert d.batch_size == 1

    def test_throughput_estimated_latency_zero_no_model(self):
        ctrl = self._make_ctrl(mode="throughput")
        d = ctrl.next_batch(queue_depth=4)
        assert d.estimated_latency_ms == 0.0

    def test_latency_mode_no_model_returns_min(self):
        ctrl = self._make_ctrl(mode="latency", target=50.0, max_bs=8, min_bs=1)
        d = ctrl.next_batch(queue_depth=8)
        # No model → all estimates 0.0 ≤ target → picks cap (8)
        assert d.batch_size == 8

    def test_latency_mode_respects_target(self):
        ctrl = self._make_ctrl(mode="latency", target=60.0, max_bs=16)
        ctrl.record_observation(1, 20.0)
        ctrl.record_observation(8, 55.0)
        ctrl.record_observation(16, 120.0)
        d = ctrl.next_batch(queue_depth=16)
        assert d.estimated_latency_ms <= 60.0

    def test_latency_mode_falls_back_when_all_exceed_target(self):
        ctrl = self._make_ctrl(mode="latency", target=10.0, max_bs=4, min_bs=1)
        ctrl.record_observation(1, 50.0)
        ctrl.record_observation(2, 80.0)
        ctrl.record_observation(4, 200.0)
        d = ctrl.next_batch(queue_depth=4)
        assert "falling back" in d.reason

    def test_record_ema_update(self):
        ctrl = self._make_ctrl()
        ctrl.record_observation(4, 100.0)
        ctrl.record_observation(4, 200.0)
        model = ctrl.latency_model
        assert model[4] != 100.0  # EMA updated

    def test_latency_model_property_snapshot(self):
        ctrl = self._make_ctrl()
        ctrl.record_observation(2, 40.0)
        m = ctrl.latency_model
        m[2] = 9999.0
        assert ctrl.latency_model[2] == 40.0  # not mutated

    def test_estimate_latency_empty_model(self):
        ctrl = self._make_ctrl()
        assert ctrl._estimate_latency(4) == 0.0

    def test_estimate_latency_exact_hit(self):
        ctrl = self._make_ctrl()
        ctrl.record_observation(4, 50.0)
        assert ctrl._estimate_latency(4) == 50.0

    def test_estimate_latency_below_range_flat_extrapolation(self):
        ctrl = self._make_ctrl()
        ctrl.record_observation(8, 80.0)
        assert ctrl._estimate_latency(2) == 80.0

    def test_estimate_latency_above_range_flat_extrapolation(self):
        ctrl = self._make_ctrl()
        ctrl.record_observation(4, 50.0)
        assert ctrl._estimate_latency(100) == 50.0

    def test_estimate_latency_interpolation(self):
        ctrl = self._make_ctrl()
        ctrl.record_observation(2, 20.0)
        ctrl.record_observation(8, 80.0)
        est = ctrl._estimate_latency(5)
        assert 20.0 < est < 80.0

    def test_latency_mode_zero_queue_depth(self):
        ctrl = self._make_ctrl(mode="latency", target=100.0, max_bs=8, min_bs=1)
        d = ctrl.next_batch(queue_depth=0)
        assert isinstance(d.batch_size, int)
        assert d.batch_size >= 1


# ===========================================================================
# schema_validator
# ===========================================================================

class TestSchemaValidator:
    def setup_method(self):
        self.v = SchemaValidator()

    # -- JSON parse errors
    def test_invalid_json_returns_error(self):
        r = self.v.validate("{not valid json}", {})
        assert not r.valid
        assert "JSON parse error" in r.errors[0]

    def test_invalid_json_n_fields_zero(self):
        r = self.v.validate("[", {})
        assert r.n_fields_checked == 0

    # -- Type checks
    def test_string_valid(self):
        r = self.v.validate('"hello"', {"type": "string"})
        assert r.valid

    def test_string_invalid_type(self):
        r = self.v.validate("42", {"type": "string"})
        assert not r.valid
        assert "expected type 'string'" in r.errors[0]

    def test_number_valid_int(self):
        r = self.v.validate("42", {"type": "number"})
        assert r.valid

    def test_number_valid_float(self):
        r = self.v.validate("3.14", {"type": "number"})
        assert r.valid

    def test_integer_valid(self):
        r = self.v.validate("5", {"type": "integer"})
        assert r.valid

    def test_boolean_rejected_for_integer(self):
        r = self.v.validate("true", {"type": "integer"})
        assert not r.valid
        assert "boolean" in r.errors[0]

    def test_boolean_rejected_for_number(self):
        r = self.v.validate("true", {"type": "number"})
        assert not r.valid

    def test_boolean_valid(self):
        r = self.v.validate("true", {"type": "boolean"})
        assert r.valid

    def test_array_valid(self):
        r = self.v.validate("[1, 2, 3]", {"type": "array"})
        assert r.valid

    def test_object_valid(self):
        r = self.v.validate('{"a": 1}', {"type": "object"})
        assert r.valid

    def test_null_valid(self):
        r = self.v.validate("null", {"type": "null"})
        assert r.valid

    def test_unknown_type_error(self):
        r = self.v.validate('"hi"', {"type": "blob"})
        assert not r.valid
        assert "unknown schema type" in r.errors[0]

    # -- String constraints
    def test_min_length_pass(self):
        r = self.v.validate('"hello"', {"type": "string", "minLength": 3})
        assert r.valid

    def test_min_length_fail(self):
        r = self.v.validate('"hi"', {"type": "string", "minLength": 5})
        assert not r.valid
        assert "minLength" in r.errors[0]

    def test_max_length_pass(self):
        r = self.v.validate('"hi"', {"type": "string", "maxLength": 10})
        assert r.valid

    def test_max_length_fail(self):
        r = self.v.validate('"toolongstring"', {"type": "string", "maxLength": 5})
        assert not r.valid
        assert "maxLength" in r.errors[0]

    # -- Numeric constraints
    def test_minimum_pass(self):
        r = self.v.validate("5", {"type": "number", "minimum": 0})
        assert r.valid

    def test_minimum_fail(self):
        r = self.v.validate("-1", {"type": "number", "minimum": 0})
        assert not r.valid
        assert "minimum" in r.errors[0]

    def test_maximum_pass(self):
        r = self.v.validate("5", {"type": "number", "maximum": 10})
        assert r.valid

    def test_maximum_fail(self):
        r = self.v.validate("15", {"type": "number", "maximum": 10})
        assert not r.valid
        assert "maximum" in r.errors[0]

    # -- Object constraints
    def test_required_field_present(self):
        schema = {"type": "object", "required": ["name"]}
        r = self.v.validate('{"name": "Alice"}', schema)
        assert r.valid

    def test_required_field_missing(self):
        schema = {"type": "object", "required": ["name"]}
        r = self.v.validate('{"age": 30}', schema)
        assert not r.valid
        assert "missing required field" in r.errors[0]

    def test_properties_recursive(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
        r = self.v.validate('{"score": 0.5}', schema)
        assert r.valid

    def test_properties_recursive_fail(self):
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            }
        }
        r = self.v.validate('{"score": 1.5}', schema)
        assert not r.valid

    def test_properties_absent_key_not_checked(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        # "name" not present; no error expected
        r = self.v.validate('{"age": 30}', schema)
        assert r.valid

    # -- Array constraints
    def test_items_valid(self):
        schema = {"type": "array", "items": {"type": "number"}}
        r = self.v.validate("[1, 2, 3]", schema)
        assert r.valid

    def test_items_invalid(self):
        schema = {"type": "array", "items": {"type": "number"}}
        r = self.v.validate('[1, "two", 3]', schema)
        assert not r.valid

    def test_n_fields_checked_increments(self):
        schema = {
            "type": "object",
            "required": ["x", "y"],
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "string"}
            }
        }
        r = self.v.validate('{"x": 1, "y": "hello"}', schema)
        assert r.n_fields_checked >= 4

    # -- is_valid helper
    def test_is_valid_true(self):
        assert self.v.is_valid('"text"', {"type": "string"}) is True

    def test_is_valid_false(self):
        assert self.v.is_valid("42", {"type": "string"}) is False

    # -- validate_value helper
    def test_validate_value_returns_errors(self):
        errs = self.v.validate_value("short", {"type": "string", "minLength": 10})
        assert len(errs) == 1

    def test_validate_value_empty_on_success(self):
        errs = self.v.validate_value("hello world!", {"type": "string", "minLength": 1})
        assert errs == []

    # -- path propagation
    def test_path_included_in_error_message(self):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number", "minimum": 0.0}}
        }
        r = self.v.validate('{"score": -5}', schema)
        assert not r.valid
        assert "score" in r.errors[0]


# ===========================================================================
# prefix_pool
# ===========================================================================

class TestPrefixPoolConfig:
    def test_defaults(self):
        cfg = PrefixPoolConfig()
        assert cfg.max_entries == 256
        assert cfg.n_heads == 32
        assert cfg.head_dim == 128

    def test_max_entries_zero(self):
        with pytest.raises(ValueError, match="max_entries"):
            PrefixPoolConfig(max_entries=0)

    def test_n_heads_zero(self):
        with pytest.raises(ValueError, match="n_heads"):
            PrefixPoolConfig(n_heads=0)

    def test_head_dim_zero(self):
        with pytest.raises(ValueError, match="head_dim"):
            PrefixPoolConfig(head_dim=0)

    def test_invalid_eviction_policy(self):
        with pytest.raises(ValueError, match="eviction_policy"):
            PrefixPoolConfig(eviction_policy="random")

    def test_kv_n_heads_defaults_to_n_heads(self):
        cfg = PrefixPoolConfig(n_heads=8)
        assert cfg.kv_n_heads == 8

    def test_kv_n_heads_custom(self):
        cfg = PrefixPoolConfig(n_heads=8, kv_n_heads=4)
        assert cfg.kv_n_heads == 4

    def test_kv_n_heads_zero_raises(self):
        with pytest.raises(ValueError, match="kv_n_heads"):
            PrefixPoolConfig(n_heads=8, kv_n_heads=0)

    def test_lfu_policy_valid(self):
        cfg = PrefixPoolConfig(eviction_policy="lfu")
        assert cfg.eviction_policy == "lfu"


class TestPrefixPoolStats:
    def test_hit_rate_zero_when_empty(self):
        s = PrefixPoolStats()
        assert s.hit_rate == 0.0

    def test_hit_rate_calculation(self):
        s = PrefixPoolStats(n_hits=3, n_misses=1)
        assert s.hit_rate == 0.75

    def test_eviction_rate_zero_when_empty(self):
        s = PrefixPoolStats()
        assert s.eviction_rate == 0.0

    def test_eviction_rate_calculation(self):
        s = PrefixPoolStats(n_hits=4, n_misses=4, n_evictions=2)
        assert s.eviction_rate == 2 / 10


class TestPrefixPool:
    def setup_method(self):
        self.rng = np.random.default_rng(11)
        self.cfg = PrefixPoolConfig(max_entries=4, n_heads=2, head_dim=4)
        self.pool = PrefixPool(self.cfg)

    def _kv(self, seq_len=5):
        k = self.rng.standard_normal((2, seq_len, 4)).astype(np.float32)
        v = self.rng.standard_normal((2, seq_len, 4)).astype(np.float32)
        return k, v

    def test_put_and_get_hit(self):
        tokens = [1, 2, 3]
        k, v = self._kv()
        self.pool.put(tokens, k, v)
        result = self.pool.get(tokens)
        assert result is not None
        k2, v2 = result
        np.testing.assert_array_equal(k, k2)

    def test_get_miss_returns_none(self):
        result = self.pool.get([99, 100])
        assert result is None

    def test_contains_true(self):
        tokens = [10, 20]
        k, v = self._kv()
        self.pool.put(tokens, k, v)
        assert self.pool.contains(tokens) is True

    def test_contains_false(self):
        assert self.pool.contains([999]) is False

    def test_hit_increments_count(self):
        tokens = [1, 2]
        k, v = self._kv(3)
        self.pool.put(tokens, k, v)
        self.pool.get(tokens)
        self.pool.get(tokens)
        stats = self.pool.get_stats()
        assert stats.n_hits == 2

    def test_miss_increments_count(self):
        self.pool.get([42])
        self.pool.get([43])
        stats = self.pool.get_stats()
        assert stats.n_misses == 2

    def test_hit_rate_calculation(self):
        tokens = [1, 2, 3]
        k, v = self._kv()
        self.pool.put(tokens, k, v)
        self.pool.get(tokens)    # hit
        self.pool.get([99])      # miss
        assert abs(self.pool.hit_rate - 0.5) < 1e-9

    def test_hit_rate_zero_before_any_access(self):
        assert self.pool.hit_rate == 0.0

    def test_total_kv_saved(self):
        tokens = [1, 2, 3]
        k, v = self._kv(7)
        self.pool.put(tokens, k, v)
        self.pool.get(tokens)
        self.pool.get(tokens)
        assert self.pool.total_kv_saved == 14

    def test_size_after_puts(self):
        for i in range(3):
            k, v = self._kv()
            self.pool.put([i], k, v)
        assert self.pool.size == 3

    def test_lru_eviction_on_overflow(self):
        cfg = PrefixPoolConfig(max_entries=2, n_heads=2, head_dim=4, eviction_policy="lru")
        pool = PrefixPool(cfg)
        k, v = self._kv()
        pool.put([1], k, v)
        pool.put([2], k, v)
        pool.put([3], k, v)  # triggers eviction
        assert pool.size == 2
        stats = pool.get_stats()
        assert stats.n_evictions == 1

    def test_lfu_eviction_on_overflow(self):
        cfg = PrefixPoolConfig(max_entries=2, n_heads=2, head_dim=4, eviction_policy="lfu")
        pool = PrefixPool(cfg)
        k, v = self._kv()
        pool.put([1], k, v)
        pool.put([2], k, v)
        pool.get([1])  # increase hit_count for [1]
        pool.put([3], k, v)  # evicts [2] (hit_count=0)
        assert pool.size == 2

    def test_reinsertion_updates_without_eviction(self):
        k, v = self._kv()
        h1 = self.pool.put([1, 2], k, v)
        k2, v2 = self._kv()
        h2 = self.pool.put([1, 2], k2, v2)
        assert h1 == h2
        assert self.pool.size == 1

    def test_put_raises_on_wrong_ndim(self):
        tokens = [1]
        k = np.zeros((5, 4), dtype=np.float32)
        v = np.zeros((5, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="keys must be 3-D"):
            self.pool.put(tokens, k, v)

    def test_put_raises_on_shape_mismatch(self):
        tokens = [1]
        k = np.zeros((2, 5, 4), dtype=np.float32)
        v = np.zeros((2, 6, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="values shape"):
            self.pool.put(tokens, k, v)

    def test_evict_lru_noop_when_empty(self):
        self.pool.evict_lru()  # should not raise

    def test_evict_lfu_noop_when_empty(self):
        self.pool.evict_lfu()  # should not raise

    def test_evict_lru_removes_oldest(self):
        import time
        k, v = self._kv()
        self.pool.put([1], k, v)
        time.sleep(0.01)
        self.pool.put([2], k, v)
        self.pool.evict_lru()
        assert self.pool.size == 1
        # [1] was older, should be evicted
        assert not self.pool.contains([1])

    def test_evict_lfu_removes_least_used(self):
        k, v = self._kv()
        self.pool.put([1], k, v)
        self.pool.put([2], k, v)
        self.pool.get([2])  # [2] gets a hit; [1] stays at 0
        self.pool.evict_lfu()
        assert self.pool.size == 1
        assert not self.pool.contains([1])

    def test_get_stats_snapshot(self):
        k, v = self._kv()
        self.pool.put([1], k, v)
        self.pool.get([1])
        self.pool.get([99])
        s = self.pool.get_stats()
        assert s.n_hits == 1
        assert s.n_misses == 1
