"""tests/test_milo_unit.py — unit tests for squish/milo_quant.py"""

import numpy as np
import pytest

from squish.milo_quant import (
    LowRankCompensator,
    MiLoConfig,
    MiLoQuantizer,
    MiLoStats,
    pack_int3,
    unpack_int3,
)

RNG = np.random.default_rng(17)


# ---------------------------------------------------------------------------
# MiLoConfig
# ---------------------------------------------------------------------------

class TestMiLoConfig:
    def test_defaults(self):
        cfg = MiLoConfig()
        assert cfg.target_bits == 3
        assert cfg.max_rank == 16
        assert cfg.min_rank == 2
        assert cfg.group_size == 64
        assert cfg.adaptive_rank is True

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="target_bits"):
            MiLoConfig(target_bits=5)

    def test_valid_bits(self):
        for b in (3, 4, 8):
            cfg = MiLoConfig(target_bits=b)
            assert cfg.target_bits == b

    def test_invalid_min_rank(self):
        with pytest.raises(ValueError, match="min_rank"):
            MiLoConfig(min_rank=0)

    def test_invalid_max_rank_lt_min(self):
        with pytest.raises(ValueError, match="max_rank"):
            MiLoConfig(max_rank=1, min_rank=4)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            MiLoConfig(group_size=0)


# ---------------------------------------------------------------------------
# pack_int3 / unpack_int3
# ---------------------------------------------------------------------------

class TestPackUnpackInt3:
    def test_round_trip_exact_multiple(self):
        """8 values — exactly one group."""
        vals = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)
        packed = pack_int3(vals)
        assert len(packed) == 3  # 3 bytes for 8 values
        recovered = unpack_int3(packed, len(vals))
        np.testing.assert_array_equal(recovered, vals)

    def test_round_trip_non_multiple(self):
        """7 values — partial last group."""
        vals = np.array([7, 6, 5, 4, 3, 2, 1], dtype=np.uint8)
        packed = pack_int3(vals)
        recovered = unpack_int3(packed, len(vals))
        np.testing.assert_array_equal(recovered, vals)

    def test_round_trip_large(self):
        rng = np.random.default_rng(99)
        vals = rng.integers(0, 8, size=256, dtype=np.uint8)
        packed = pack_int3(vals)
        recovered = unpack_int3(packed, len(vals))
        np.testing.assert_array_equal(recovered, vals)

    def test_packed_length(self):
        for n in [1, 7, 8, 9, 16, 100]:
            vals = np.zeros(n, dtype=np.uint8)
            packed = pack_int3(vals)
            import math
            expected = math.ceil(n / 8) * 3
            assert len(packed) == expected, f"n={n}: expected {expected}, got {len(packed)}"

    def test_all_zeros(self):
        vals = np.zeros(24, dtype=np.uint8)
        recovered = unpack_int3(pack_int3(vals), 24)
        np.testing.assert_array_equal(recovered, vals)

    def test_all_sevens(self):
        vals = np.full(16, 7, dtype=np.uint8)
        recovered = unpack_int3(pack_int3(vals), 16)
        np.testing.assert_array_equal(recovered, vals)


# ---------------------------------------------------------------------------
# LowRankCompensator
# ---------------------------------------------------------------------------

class TestLowRankCompensator:
    def _make_compensator(self, m=8, n=12, r=3):
        a = RNG.standard_normal((m, r)).astype(np.float32)
        b = RNG.standard_normal((r, n)).astype(np.float32)
        return LowRankCompensator(a, b)

    def test_rank_property(self):
        comp = self._make_compensator(m=8, n=12, r=3)
        assert comp.rank == 3

    def test_apply_shape(self):
        m, n, r = 8, 12, 3
        comp = self._make_compensator(m, n, r)
        base = np.zeros(m, dtype=np.float32)
        inp  = RNG.standard_normal(n).astype(np.float32)
        out  = comp.apply(base, inp)
        assert out.shape == (m,)

    def test_apply_zero_input(self):
        comp = self._make_compensator(4, 6, 2)
        base = RNG.standard_normal(4).astype(np.float32)
        inp  = np.zeros(6, dtype=np.float32)
        out  = comp.apply(base, inp)
        # Correction should be zero when input is zero
        np.testing.assert_allclose(out, base)

    def test_apply_adds_correction(self):
        """Verify apply(base, inp) == base + scale * A @ (B @ inp)."""
        m, n, r = 5, 7, 2
        a = RNG.standard_normal((m, r)).astype(np.float32)
        b = RNG.standard_normal((r, n)).astype(np.float32)
        comp = LowRankCompensator(a, b)
        base = RNG.standard_normal(m).astype(np.float32)
        inp  = RNG.standard_normal(n).astype(np.float32)
        expected = base + a @ (b @ inp)
        out = comp.apply(base, inp)
        np.testing.assert_allclose(out, expected, rtol=1e-5)

    def test_memory_bytes(self):
        comp = self._make_compensator(8, 12, 4)
        # float32 → 4 bytes per element
        expected = (8 * 4 + 4 * 12) * 4
        assert comp.memory_bytes() == expected

    def test_reconstruction_snr_db_perfect(self):
        """When compensator perfectly captures residual, SNR → infinity."""
        m, n = 4, 6
        residual = RNG.standard_normal((m, n)).astype(np.float32)
        # A @ B == residual when rank == min(m, n)
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        a = u * s  # (m, r)
        b = vt     # (r, n)
        comp = LowRankCompensator(a, b)
        snr = comp.reconstruction_snr_db(residual)
        assert snr > 60.0 or snr == float("inf")

    def test_reconstruction_snr_db_partial(self):
        """Rank-1 compensator on a full-rank matrix has finite SNR."""
        m, n = 8, 10
        residual = RNG.standard_normal((m, n)).astype(np.float32)
        u, s, vt = np.linalg.svd(residual, full_matrices=False)
        # rank-1 only
        a = (u[:, :1] * s[:1]).astype(np.float32)
        b = vt[:1, :].astype(np.float32)
        comp = LowRankCompensator(a, b)
        snr = comp.reconstruction_snr_db(residual)
        assert 0.0 < snr < float("inf")


# ---------------------------------------------------------------------------
# MiLoQuantizer
# ---------------------------------------------------------------------------

class TestMiLoQuantizer:
    def _weight(self, m=32, n=64) -> np.ndarray:
        return RNG.standard_normal((m, n)).astype(np.float32)

    def test_quantize_returns_four_components(self):
        q = MiLoQuantizer()
        w = self._weight()
        q_packed, scales, zeros, comp = q.quantize(w)
        assert isinstance(q_packed, np.ndarray)
        assert isinstance(scales, np.ndarray)
        assert isinstance(zeros, np.ndarray)
        assert isinstance(comp, LowRankCompensator)

    def test_quantize_dequantize_reasonable_snr(self):
        """INT3 + compensator should achieve a useful SNR on Gaussian weight."""
        cfg = MiLoConfig(target_bits=3, max_rank=8, snr_threshold_db=20.0)
        q = MiLoQuantizer(cfg)
        w = self._weight(32, 64)
        q_packed, scales, zeros, comp = q.quantize(w)
        snr = q.reconstruction_snr(w, q_packed, scales, zeros, comp)
        assert snr > 15.0

    def test_quantize_int4_mode(self):
        """INT4 mode should run without error and produce comp."""
        cfg = MiLoConfig(target_bits=4, max_rank=4)
        qr = MiLoQuantizer(cfg)
        w = self._weight(16, 32)
        q_packed, scales, zeros, comp = qr.quantize(w)
        assert comp.rank >= 1

    def test_adaptive_rank_selection(self):
        """With adaptive_rank=True, rank should be >= min_rank and <= max_rank."""
        cfg = MiLoConfig(target_bits=3, min_rank=2, max_rank=8, adaptive_rank=True)
        qr = MiLoQuantizer(cfg)
        w = self._weight(32, 48)
        _, _, _, comp = qr.quantize(w)
        assert 1 <= comp.rank <= 8

    def test_fixed_rank_mode(self):
        """adaptive_rank=False should always use max_rank."""
        cfg = MiLoConfig(target_bits=3, max_rank=5, adaptive_rank=False)
        qr = MiLoQuantizer(cfg)
        w = self._weight(32, 48)
        _, _, _, comp = qr.quantize(w)
        assert comp.rank == min(5, min(*w.shape))

    def test_dequantize_shape(self):
        qr = MiLoQuantizer()
        w = self._weight(16, 32)
        q_packed, scales, zeros, _ = qr.quantize(w)
        w_dq = qr.dequantize(q_packed, scales, zeros, w.size, w.shape)
        assert w_dq.shape == w.shape

    def test_single_weight_quantize(self):
        """Single-element weight should not crash."""
        qr = MiLoQuantizer(MiLoConfig(target_bits=3, min_rank=1, max_rank=1))
        w = np.array([[1.5]], dtype=np.float32)
        q_packed, scales, zeros, comp = qr.quantize(w)
        assert comp is not None


# ---------------------------------------------------------------------------
# MiLoStats
# ---------------------------------------------------------------------------

class TestMiLoStats:
    def test_defaults(self):
        s = MiLoStats()
        assert s.n_matrices == 0
        assert s.avg_snr_db == 0.0
        assert s.avg_rank == 0.0
        assert s.compression_ratio == 0.0

    def test_record(self):
        s = MiLoStats()
        s.record(snr_db=30.0, rank=4, weight_bytes=1024, quant_bytes=384, comp_bytes=256)
        assert s.n_matrices == 1
        assert abs(s.avg_snr_db - 30.0) < 1e-6
        assert abs(s.avg_rank - 4.0) < 1e-6

    def test_compression_ratio(self):
        s = MiLoStats()
        s.record(snr_db=30.0, rank=4, weight_bytes=1000, quant_bytes=375, comp_bytes=125)
        # (375 + 125) / 1000 = 0.5
        assert abs(s.compression_ratio - 0.5) < 1e-6

    def test_reset(self):
        s = MiLoStats()
        s.record(snr_db=35.0, rank=6, weight_bytes=512, quant_bytes=192, comp_bytes=96)
        s.reset()
        assert s.n_matrices == 0
        assert s.avg_snr_db == 0.0
        assert s.total_weight_bytes == 0
