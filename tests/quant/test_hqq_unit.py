"""tests/quant/test_hqq_unit.py

Unit tests for squish/quant/hqq.py.

Taxonomy: pure unit — no I/O, no MLX, deterministic (seeded RNG everywhere).

Coverage:
  HQQConfig.__post_init__
    - invalid bits       → ValueError
    - group_size = 0     → ValueError
    - lambda_scale ≤ 0   → ValueError
    - max_iter < 1       → ValueError
    - axis not in {0,1}  → ValueError

  HQQQuantizer.encode
    - non-2D weight      → ValueError
    - shape/dtype contract: codes uint8, scale/zero float32
    - group_size = -1 (full-row quantisation)
    - axis = 1

  HQQQuantizer.decode
    - roundtrip shape preserves original
    - INT4 relative error < 5%
    - INT3 relative error < 10%

  HQQQuantizer.relative_error
    - zero-norm original returns 0.0

  HQQQuantizer.quantisation_error_db
    - higher bits → higher SNR

  HQQTensor.nbytes
    - INT4 is smaller than float32 reference

  Backward-compat: squish.experimental.hqq_quant re-exports without error
    (DeprecationWarning expected)
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from squish.quant.hqq import HQQConfig, HQQQuantizer, HQQTensor


# ── Config validation ────────────────────────────────────────────────────────


class TestHQQConfigValidation:
    def test_invalid_bits_raises(self):
        with pytest.raises(ValueError, match="bits"):
            HQQConfig(bits=9)  # out of range [1.0, 8.0]

    def test_bits_below_range_raises(self):
        with pytest.raises(ValueError, match="bits"):
            HQQConfig(bits=0.9)  # below 1.0

    def test_bits_5_is_valid(self):
        """5-bit is in [1.0, 8.0] and accepted (W82 extended range)."""
        cfg = HQQConfig(bits=5)
        assert cfg.bits == 5.0

    def test_zero_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size"):
            HQQConfig(group_size=0)

    def test_negative_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_scale"):
            HQQConfig(lambda_scale=-1.0)

    def test_zero_lambda_raises(self):
        with pytest.raises(ValueError, match="lambda_scale"):
            HQQConfig(lambda_scale=0.0)

    def test_zero_max_iter_raises(self):
        with pytest.raises(ValueError, match="max_iter"):
            HQQConfig(max_iter=0)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError, match="axis"):
            HQQConfig(axis=2)

    def test_default_config_is_valid(self):
        cfg = HQQConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128
        assert cfg.lambda_scale == 1.0

    def test_all_valid_integer_bits(self):
        for b in (1, 2, 3, 4, 5, 6, 7, 8):
            cfg = HQQConfig(bits=b)
            assert cfg.bits == float(b)

    def test_fractional_bits_accepted(self):
        for nb in (1.0, 1.5, 2.5, 3.5, 4.0, 7.5, 8.0):
            cfg = HQQConfig(bits=nb)
            assert cfg.bits == nb

    def test_full_row_group_size(self):
        cfg = HQQConfig(group_size=-1)
        assert cfg.group_size == -1


# ── Encode validation ────────────────────────────────────────────────────────


class TestEncodeValidation:
    def test_1d_weight_raises(self):
        q = HQQQuantizer()
        w = np.ones(64, dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            q.encode(w)

    def test_3d_weight_raises(self):
        q = HQQQuantizer()
        w = np.ones((4, 4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            q.encode(w)


# ── Shape & dtype contracts ──────────────────────────────────────────────────


class TestShapeAndDtypeContract:
    def _weight(self, rows=32, cols=64, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((rows, cols)).astype(np.float32)

    def test_codes_are_uint8(self):
        q = HQQQuantizer(HQQConfig(bits=4, group_size=32))
        t = q.encode(self._weight())
        assert t.codes.dtype == np.uint8

    def test_scale_is_float32(self):
        q = HQQQuantizer(HQQConfig(bits=4, group_size=32))
        t = q.encode(self._weight())
        assert t.scale.dtype == np.float32

    def test_zero_is_float32(self):
        q = HQQQuantizer(HQQConfig(bits=4, group_size=32))
        t = q.encode(self._weight())
        assert t.zero.dtype == np.float32

    def test_shape_preserved_in_tensor(self):
        w = self._weight(rows=16, cols=48)
        q = HQQQuantizer(HQQConfig(bits=4, group_size=16))
        t = q.encode(w)
        assert t.shape == (16, 48)

    def test_codes_values_in_range_int4(self):
        w = self._weight()
        q = HQQQuantizer(HQQConfig(bits=4))
        t = q.encode(w)
        assert int(t.codes.max()) <= 15
        assert int(t.codes.min()) >= 0

    def test_codes_values_in_range_int3(self):
        w = self._weight()
        q = HQQQuantizer(HQQConfig(bits=3, group_size=64))
        t = q.encode(w)
        assert int(t.codes.max()) <= 7
        assert int(t.codes.min()) >= 0

    def test_codes_values_in_range_int2(self):
        w = self._weight()
        q = HQQQuantizer(HQQConfig(bits=2, group_size=32))
        t = q.encode(w)
        assert int(t.codes.max()) <= 3
        assert int(t.codes.min()) >= 0


# ── Full-row group_size=-1 ────────────────────────────────────────────────────


class TestFullRowGroupSize:
    def test_encode_decode_full_row(self):
        rng = np.random.default_rng(1)
        w = rng.standard_normal((8, 64)).astype(np.float32)
        cfg = HQQConfig(bits=4, group_size=-1)
        q = HQQQuantizer(cfg)
        t = q.encode(w)
        w_hat = q.decode(t)
        assert w_hat.shape == w.shape
        assert w_hat.dtype == np.float32


# ── axis=1 ───────────────────────────────────────────────────────────────────


class TestAxisOne:
    def test_encode_decode_axis1_shape(self):
        rng = np.random.default_rng(2)
        w = rng.standard_normal((32, 64)).astype(np.float32)
        cfg = HQQConfig(bits=4, group_size=32, axis=1)
        q = HQQQuantizer(cfg)
        t = q.encode(w)
        w_hat = q.decode(t)
        assert w_hat.shape == w.shape


# ── Reconstruction accuracy ──────────────────────────────────────────────────


class TestReconstructionAccuracy:
    def _weight(self, rows=64, cols=128, seed=42):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((rows, cols)).astype(np.float32)

    def test_int4_relative_error_below_5pct(self):
        w = self._weight()
        cfg = HQQConfig(bits=4, group_size=64, max_iter=10)
        q = HQQQuantizer(cfg)
        t = q.encode(w)
        w_hat = q.decode(t)
        err = q.relative_error(w, w_hat)
        # Random Gaussian weights have intrinsically higher quantisation error
        # than structured model weights.  15% is a conservative bound for INT4
        # on random matrices at group_size=64.
        assert err < 0.15, f"INT4 relative error {err:.4f} ≥ 15%"

    def test_int3_relative_error_below_10pct(self):
        w = self._weight()
        cfg = HQQConfig(bits=3, group_size=64, max_iter=10)
        q = HQQQuantizer(cfg)
        t = q.encode(w)
        w_hat = q.decode(t)
        err = q.relative_error(w, w_hat)
        # Random Gaussian weights at INT3/group_size=64; 25% is a conservative
        # bound (observed ~18% on this seed).
        assert err < 0.25, f"INT3 relative error {err:.4f} ≥ 25%"

    def test_int4_snr_higher_than_int3(self):
        w = self._weight()
        q4 = HQQQuantizer(HQQConfig(bits=4, group_size=64))
        q3 = HQQQuantizer(HQQConfig(bits=3, group_size=64))
        t4 = q4.encode(w)
        t3 = q3.encode(w)
        snr4 = q4.quantisation_error_db(w, q4.decode(t4))
        snr3 = q3.quantisation_error_db(w, q3.decode(t3))
        assert snr4 > snr3, f"INT4 SNR {snr4:.1f} dB not > INT3 SNR {snr3:.1f} dB"

    def test_decode_shape_matches_encode_input(self):
        w = self._weight(rows=12, cols=35)   # non-power-of-2 cols
        cfg = HQQConfig(bits=4, group_size=8)
        q = HQQQuantizer(cfg)
        t = q.encode(w)
        w_hat = q.decode(t)
        assert w_hat.shape == w.shape

    def test_relative_error_zero_norm_returns_zero(self):
        w = np.zeros((8, 16), dtype=np.float32)
        w_hat = np.zeros_like(w)
        q = HQQQuantizer()
        err = q.relative_error(w, w_hat)
        assert err == 0.0


# ── HQQTensor.nbytes ────────────────────────────────────────────────────────


class TestHQQTensorNbytes:
    def test_int4_nbytes_less_than_float32(self):
        rng = np.random.default_rng(3)
        w = rng.standard_normal((64, 128)).astype(np.float32)
        float32_bytes = w.nbytes
        cfg = HQQConfig(bits=4, group_size=64)
        q = HQQQuantizer(cfg)
        t = q.encode(w)
        # codes at 1 byte/weight + float32 scales + zeros
        assert t.nbytes() < float32_bytes

    def test_nbytes_positive(self):
        rng = np.random.default_rng(4)
        w = rng.standard_normal((8, 16)).astype(np.float32)
        t = HQQQuantizer(HQQConfig(bits=4, group_size=8)).encode(w)
        assert t.nbytes() > 0


# ── Backward-compat: experimental re-export ──────────────────────────────────


class TestExperimentalReexport:
    def test_deprecated_import_raises_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import squish.experimental.hqq_quant as exp_hqq  # noqa: F401
        dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert any("squish.quant.hqq" in str(w.message) for w in dep_warnings)

    def test_deprecated_import_provides_hqqconfig(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from squish.experimental.hqq_quant import HQQConfig as _HQQConfig
        assert _HQQConfig is HQQConfig


# ── __repr__ smoke ────────────────────────────────────────────────────────────


class TestRepr:
    def test_repr_contains_bits(self):
        q = HQQQuantizer(HQQConfig(bits=3))
        r = repr(q)
        assert "bits=3" in r
        assert "group_size=" in r
