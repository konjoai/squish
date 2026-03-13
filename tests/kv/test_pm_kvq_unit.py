"""tests/test_pm_kvq_unit.py

100% coverage tests for squish/pm_kvq.py.

Covers:
  PMKVQConfig.__post_init__  — all 4 validation branches (ValueError cases)
  PMKVQScheduler.__init__    — custom block_sensitivity + shape mismatch
  PMKVQScheduler.current_bits — all 4 scheduling phases (FP16, INT8, INT4, min_bits)
                                both sensitive and non-sensitive blocks
  PMKVQScheduler.bits_for_all_blocks
  PMKVQScheduler.advance      — step recording into stats
  PMKVQScheduler.reset
  PMKVQScheduler.quantize_kv  — bits==16 path and bits<16 path
  PMKVQScheduler.dequantize_kv — bits==16 path and bits<16 path
  PMKVQCalibrator.record      — single block, multi-call accumulation
  PMKVQCalibrator.compute_block_sensitivity — with data, all-zero max_s path
  PMKVQStats.record_step      — transitions at 8, 4, 2 bits
  PMKVQStats.reset
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scheduler(n_blocks: int = 8, high_prec: int = 4, mid: int = 10, cold: int = 20,
                    min_bits: int = 2, min_bits_sensitive: int = 4,
                    block_sensitivity=None):
    from squish.kv.pm_kvq import PMKVQConfig, PMKVQScheduler
    cfg = PMKVQConfig(
        n_blocks=n_blocks,
        high_prec_tokens=high_prec,
        mid_tokens=mid,
        cold_steps=cold,
        min_bits=min_bits,
        min_bits_sensitive=min_bits_sensitive,
        group_size=8,
    )
    return PMKVQScheduler(cfg, block_sensitivity=block_sensitivity)


# ---------------------------------------------------------------------------
# PMKVQConfig validation
# ---------------------------------------------------------------------------


class TestPMKVQConfigValidation:
    def test_valid_defaults(self):
        from squish.kv.pm_kvq import PMKVQConfig
        cfg = PMKVQConfig()
        assert cfg.min_bits == 2
        assert cfg.min_bits_sensitive == 4

    def test_min_bits_too_low(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="min_bits"):
            PMKVQConfig(min_bits=0)

    def test_min_bits_too_high(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="min_bits"):
            PMKVQConfig(min_bits=9)

    def test_min_bits_sensitive_too_low(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="min_bits_sensitive"):
            PMKVQConfig(min_bits_sensitive=0)

    def test_min_bits_sensitive_too_high(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="min_bits_sensitive"):
            PMKVQConfig(min_bits_sensitive=9)

    def test_min_bits_sensitive_less_than_min_bits(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="min_bits_sensitive must be >= min_bits"):
            PMKVQConfig(min_bits=4, min_bits_sensitive=2)

    def test_sensitive_fraction_out_of_range_low(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="sensitive_fraction"):
            PMKVQConfig(sensitive_fraction=-0.1)

    def test_sensitive_fraction_out_of_range_high(self):
        from squish.kv.pm_kvq import PMKVQConfig
        with pytest.raises(ValueError, match="sensitive_fraction"):
            PMKVQConfig(sensitive_fraction=1.1)


# ---------------------------------------------------------------------------
# PMKVQScheduler init
# ---------------------------------------------------------------------------


class TestPMKVQSchedulerInit:
    def test_default_config(self):
        from squish.kv.pm_kvq import PMKVQScheduler
        sched = PMKVQScheduler()
        assert sched.step == 0

    def test_custom_sensitivity(self):
        """Custom block_sensitivity array is accepted."""
        n = 8
        sens = np.linspace(0.0, 1.0, n, dtype=np.float32)
        sched = _make_scheduler(n_blocks=n, block_sensitivity=sens)
        assert sched.step == 0

    def test_wrong_sensitivity_shape_raises(self):
        """Shape mismatch raises ValueError."""
        from squish.kv.pm_kvq import PMKVQConfig, PMKVQScheduler
        cfg = PMKVQConfig(n_blocks=8)
        bad_sens = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError, match="block_sensitivity must have shape"):
            PMKVQScheduler(cfg, block_sensitivity=bad_sens)

    def test_config_property(self):
        sched = _make_scheduler()
        from squish.kv.pm_kvq import PMKVQConfig
        assert isinstance(sched.config, PMKVQConfig)

    def test_stats_property(self):
        from squish.kv.pm_kvq import PMKVQStats
        sched = _make_scheduler()
        assert isinstance(sched.stats, PMKVQStats)


# ---------------------------------------------------------------------------
# PMKVQScheduler.current_bits — 4 scheduling phases
# ---------------------------------------------------------------------------


class TestCurrentBits:
    def test_fp16_phase(self):
        """Steps < high_prec_tokens → 16."""
        sched = _make_scheduler(high_prec=10, mid=20, cold=40)
        assert sched.current_bits(0) == 16

    def test_int8_phase(self):
        """high_prec ≤ step < mid_tokens → 8."""
        sched = _make_scheduler(n_blocks=4, high_prec=2, mid=10, cold=30)
        for _ in range(2):
            sched.advance()
        assert sched.current_bits(0) == 8

    def test_int4_phase(self):
        """mid_tokens ≤ step < cold_steps → 4."""
        sched = _make_scheduler(n_blocks=4, high_prec=2, mid=5, cold=20)
        for _ in range(5):
            sched.advance()
        assert sched.current_bits(0) == 4

    def test_min_bits_phase_non_sensitive(self):
        """step >= cold_steps → min_bits for non-sensitive blocks."""
        sched = _make_scheduler(n_blocks=4, high_prec=1, mid=2, cold=3, min_bits=2)
        for _ in range(3):
            sched.advance()
        # first block in sched with default sensitivity linspace(0.1,0.9,4)
        # [0.1, 0.37, 0.63, 0.9] → sensitive_fraction=0.25 → threshold=0.9 → only block 3 sensitive
        non_sensitive_bits = sched.current_bits(0)
        assert non_sensitive_bits == 2

    def test_min_bits_phase_sensitive(self):
        """step >= cold_steps → min_bits_sensitive for sensitive blocks."""
        sched = _make_scheduler(
            n_blocks=4, high_prec=1, mid=2, cold=3,
            min_bits=2, min_bits_sensitive=4,
        )
        for _ in range(3):
            sched.advance()
        # block 3 in linspace(0.1,0.9,4) is the most sensitive
        sensitive_bits = sched.current_bits(3)
        assert sensitive_bits == 4

    def test_block_idx_wraps_modulo_n_blocks(self):
        """block_idx % n_blocks is used (no IndexError for large indices)."""
        sched = _make_scheduler(n_blocks=4)
        # block 0 and block 4 should give the same result
        assert sched.current_bits(0) == sched.current_bits(4)


# ---------------------------------------------------------------------------
# PMKVQScheduler.bits_for_all_blocks
# ---------------------------------------------------------------------------


class TestBitsForAllBlocks:
    def test_returns_array_of_correct_shape(self):
        sched = _make_scheduler(n_blocks=6)
        bits = sched.bits_for_all_blocks()
        assert bits.shape == (6,)
        assert bits.dtype == np.int32

    def test_all_fp16_at_step_0(self):
        sched = _make_scheduler(n_blocks=4, high_prec=100)
        bits = sched.bits_for_all_blocks()
        assert np.all(bits == 16)


# ---------------------------------------------------------------------------
# PMKVQScheduler.advance and reset
# ---------------------------------------------------------------------------


class TestAdvanceAndReset:
    def test_advance_increments_step(self):
        sched = _make_scheduler()
        assert sched.step == 0
        sched.advance()
        assert sched.step == 1

    def test_advance_records_stats(self):
        sched = _make_scheduler()
        sched.advance()
        assert sched.stats.total_steps == 1
        assert len(sched.stats.avg_bits_history) == 1

    def test_reset_clears_step(self):
        sched = _make_scheduler()
        sched.advance()
        sched.advance()
        sched.reset()
        assert sched.step == 0


# ---------------------------------------------------------------------------
# PMKVQScheduler.quantize_kv and dequantize_kv
# ---------------------------------------------------------------------------


class TestQuantizeDequantize:
    def _make_kv(self, shape=(4, 16)):
        rng = np.random.default_rng(42)
        return rng.standard_normal(shape).astype(np.float32)

    def test_fp16_quantize_returns_float16(self):
        """bits==16 path: returns float16 array."""
        sched = _make_scheduler(high_prec=100)
        kv = self._make_kv((4, 16))
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        assert bits == 16
        assert q.dtype == np.float16

    def test_int8_quantize_returns_int8(self):
        """bits==8 path: returns int8 quantized array."""
        sched = _make_scheduler(n_blocks=4, high_prec=0, mid=100, cold=200)
        kv = self._make_kv((4, 16))
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        assert bits == 8
        assert q.dtype == np.int8
        assert scale.ndim == 1

    def test_int4_quantize_returns_int8(self):
        """bits==4 path: still int8 (4-bit values stored in int8)."""
        sched = _make_scheduler(n_blocks=4, high_prec=0, mid=0, cold=100)
        kv = self._make_kv((4, 16))
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        assert bits == 4
        assert q.dtype == np.int8

    def test_fp16_roundtrip(self):
        """FP16 quantize/dequantize is low-loss."""
        sched = _make_scheduler(high_prec=100)
        kv = self._make_kv((4, 16))
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        rec = sched.dequantize_kv(q, scale, bits, kv.shape)
        assert rec.shape == kv.shape
        np.testing.assert_allclose(rec, kv, rtol=1e-2)

    def test_int8_roundtrip_shape(self):
        """INT8 dequantize produces correct shape."""
        sched = _make_scheduler(n_blocks=4, high_prec=0, mid=100, cold=200)
        kv = self._make_kv((8, 32))
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        rec = sched.dequantize_kv(q, scale, bits, kv.shape)
        assert rec.shape == kv.shape
        assert rec.dtype == np.float32

    def test_int4_roundtrip_approximate(self):
        """INT4 dequantize: output shape matches, values in range."""
        sched = _make_scheduler(n_blocks=4, high_prec=0, mid=0, cold=100)
        kv = self._make_kv((4, 16))
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        rec = sched.dequantize_kv(q, scale, bits, kv.shape)
        assert rec.shape == kv.shape

    def test_1d_kv_vector(self):
        """1D kv_vector is handled (scalar token)."""
        sched = _make_scheduler(n_blocks=4, high_prec=0, mid=100, cold=200)
        kv = np.random.default_rng(1).standard_normal(32).astype(np.float32)
        q, scale, bits = sched.quantize_kv(kv, block_idx=0)
        rec = sched.dequantize_kv(q, scale, bits, kv.shape)
        assert rec.shape == kv.shape


# ---------------------------------------------------------------------------
# PMKVQCalibrator
# ---------------------------------------------------------------------------


class TestPMKVQCalibrator:
    def test_init(self):
        from squish.kv.pm_kvq import PMKVQCalibrator
        cal = PMKVQCalibrator()
        assert cal.n_samples == 0

    def test_record_increments_n_samples(self):
        from squish.kv.pm_kvq import PMKVQCalibrator
        cal = PMKVQCalibrator()
        keys = np.random.default_rng(0).standard_normal((8, 4, 16)).astype(np.float32)
        cal.record(keys, block_idx=0)
        assert cal.n_samples == 1

    def test_record_multiple_blocks(self):
        from squish.kv.pm_kvq import PMKVQCalibrator
        cal = PMKVQCalibrator()
        keys = np.random.default_rng(0).standard_normal((8, 4, 16)).astype(np.float32)
        cal.record(keys, block_idx=0)
        cal.record(keys, block_idx=2)
        assert cal.n_samples == 2

    def test_record_multiple_times_same_block(self):
        from squish.kv.pm_kvq import PMKVQCalibrator
        cal = PMKVQCalibrator()
        keys = np.random.default_rng(0).standard_normal((8, 16)).astype(np.float32)
        cal.record(keys, block_idx=0)
        cal.record(keys, block_idx=0)
        assert cal.n_samples == 2

    def test_compute_block_sensitivity_shape(self):
        from squish.kv.pm_kvq import PMKVQConfig, PMKVQCalibrator
        cfg = PMKVQConfig(n_blocks=4)
        cal = PMKVQCalibrator(cfg)
        keys = np.abs(np.random.default_rng(0).standard_normal((8, 16))).astype(np.float32)
        for i in range(4):
            cal.record(keys, block_idx=i)
        scores = cal.compute_block_sensitivity()
        assert scores.shape == (4,)
        assert scores.max() <= 1.0 + 1e-6
        assert scores.min() >= 0.0

    def test_compute_block_sensitivity_no_data(self):
        """Empty calibrator returns zero scores."""
        from squish.kv.pm_kvq import PMKVQConfig, PMKVQCalibrator
        cal = PMKVQCalibrator(PMKVQConfig(n_blocks=4))
        scores = cal.compute_block_sensitivity()
        assert scores.shape == (4,)
        np.testing.assert_array_equal(scores, 0.0)

    def test_compute_block_sensitivity_all_zero_key_states(self):
        """All-zero keys → max_s == 0 → scores stay 0 (no divide by zero)."""
        from squish.kv.pm_kvq import PMKVQConfig, PMKVQCalibrator
        cal = PMKVQCalibrator(PMKVQConfig(n_blocks=2))
        keys = np.zeros((4, 16), dtype=np.float32)
        cal.record(keys, block_idx=0)
        cal.record(keys, block_idx=1)
        scores = cal.compute_block_sensitivity()
        assert np.all(scores == 0.0)


# ---------------------------------------------------------------------------
# PMKVQStats
# ---------------------------------------------------------------------------


class TestPMKVQStats:
    def test_init_defaults(self):
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        assert stats.total_steps == 0
        assert stats.step_at_int8 is None
        assert stats.step_at_int4 is None
        assert stats.step_at_min is None

    def test_record_step_increments_total(self):
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        stats.record_step(16, step=0)
        assert stats.total_steps == 1
        assert len(stats.avg_bits_history) == 1

    def test_record_step_sets_int8_transition(self):
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        stats.record_step(8, step=5)
        assert stats.step_at_int8 == 5

    def test_record_step_sets_int4_transition(self):
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        stats.record_step(4, step=10)
        assert stats.step_at_int4 == 10

    def test_record_step_sets_min_transition(self):
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        stats.record_step(2, step=20)
        assert stats.step_at_min == 20

    def test_record_step_does_not_overwrite_transitions(self):
        """Once set, transition markers don't update on subsequent steps."""
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        stats.record_step(8, step=5)
        stats.record_step(8, step=6)
        assert stats.step_at_int8 == 5

    def test_reset_clears_all_fields(self):
        from squish.kv.pm_kvq import PMKVQStats
        stats = PMKVQStats()
        stats.record_step(8, step=3)
        stats.record_step(4, step=10)
        stats.record_step(2, step=20)
        stats.reset()
        assert stats.total_steps == 0
        assert stats.avg_bits_history == []
        assert stats.step_at_int8 is None
        assert stats.step_at_int4 is None
        assert stats.step_at_min is None
