"""Behavioral coverage for the edge/error paths of ``squish.quant.milo_quant``
left untested by the baseline suite. Pure-Python numpy; no MLX.
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quant import milo_quant as mq
from squish.quant.milo_quant import (
    LowRankCompensator,
    MiLoConfig,
    MiLoQuantizer,
    _dequantize_int_n,
    _quantize_int_n,
)


# ── _quantize_int_n / _dequantize_int_n ──────────────────────────────────────


def test_quantize_all_zero_group_uses_unit_scale():
    q, scales, zeros = _quantize_int_n(np.zeros(8, np.float32), bits=4, group_size=8)
    # vmax==0 → falls back to 1.0 (228), giving a positive scale = 1.0/(levels/2).
    assert scales[0] == pytest.approx(1.0 / 7.5)
    out = _dequantize_int_n(q, scales, zeros, n=8, bits=4, group_size=8, original_shape=(8,))
    assert np.all(out == 0.0)  # all-zero group reconstructs to zeros


def test_dequantize_without_original_shape_returns_flat():
    arr = np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32)
    q, scales, zeros = _quantize_int_n(arr, bits=4, group_size=4)
    out = _dequantize_int_n(q, scales, zeros, n=4, bits=4, group_size=4,
                            original_shape=None)  # 267→269 (no reshape)
    assert out.ndim == 1 and out.shape == (4,)


def test_quantize_dequantize_int3_roundtrip():
    arr = np.linspace(-1, 1, 64, dtype=np.float32)
    q, scales, zeros = _quantize_int_n(arr, bits=3, group_size=64)
    out = _dequantize_int_n(q, scales, zeros, n=64, bits=3, group_size=64,
                            original_shape=(64,))
    assert out.shape == (64,) and np.max(np.abs(out - arr)) < 0.5


# ── LowRankCompensator ───────────────────────────────────────────────────────


def test_compensator_properties_and_apply():
    a = np.ones((3, 2), np.float32)
    b = np.ones((2, 4), np.float32)
    comp = LowRankCompensator(a, b, scale=2.0)
    assert comp.rank == 2 and comp.scale == 2.0  # 322
    np.testing.assert_array_equal(comp.a, a)
    np.testing.assert_array_equal(comp.b, b)
    assert comp.memory_bytes() == a.nbytes + b.nbytes
    out = comp.apply(np.zeros(3, np.float32), np.ones(4, np.float32))
    # B@x = [4,4]; A@that = [8,8,8]; scale 2 → [16,16,16]
    np.testing.assert_array_equal(out, np.full(3, 16.0, np.float32))


def test_compensator_snr_perfect_is_inf():
    a = np.zeros((2, 1), np.float32)
    b = np.zeros((1, 3), np.float32)
    comp = LowRankCompensator(a, b)
    # residual == A@B (both zero) → err_pow 0 → inf (371)
    assert comp.reconstruction_snr_db(np.zeros((2, 3), np.float32)) == float("inf")


def test_compensator_snr_finite():
    a = np.ones((2, 1), np.float32)
    b = np.ones((1, 2), np.float32)
    comp = LowRankCompensator(a, b)
    snr = comp.reconstruction_snr_db(np.full((2, 2), 2.0, np.float32))  # residual != A@B
    assert np.isfinite(snr)


# ── MiLoQuantizer ────────────────────────────────────────────────────────────


def test_quantizer_config_property():
    cfg = MiLoConfig()
    assert MiLoQuantizer(cfg).config is cfg  # 393


def test_quantizer_quantize_dequantize_roundtrip():
    q = MiLoQuantizer(MiLoConfig())
    w = np.random.default_rng(0).standard_normal((16, 64)).astype(np.float32)
    packed, scales, zeros, comp = q.quantize(w, name="layer")
    w_dq = q.dequantize(packed, scales, zeros, n=w.size, original_shape=w.shape)
    assert w_dq.shape == (16, 64)
    snr = q.reconstruction_snr(w, packed, scales, zeros, comp)
    assert np.isfinite(snr)


def test_quantizer_reconstruction_snr_perfect_is_inf():
    q = MiLoQuantizer(MiLoConfig())
    w = np.zeros((8, 64), np.float32)  # zero weight → exact reconstruction → inf (502)
    packed, scales, zeros, comp = q.quantize(w)
    assert q.reconstruction_snr(w, packed, scales, zeros, comp) == float("inf")


def test_low_rank_decompose_1d_residual():
    q = MiLoQuantizer(MiLoConfig())
    comp = q._low_rank_decompose(np.array([0.1, 0.2, 0.3, 0.4], np.float32))  # 1-D → reshape (521)
    assert comp.a.shape[0] == 1 or comp.b.shape[1] == 4


def test_low_rank_decompose_svd_failure_returns_zero_compensator(monkeypatch):
    q = MiLoQuantizer(MiLoConfig())

    def _boom(*a, **k):
        raise np.linalg.LinAlgError("svd did not converge")

    monkeypatch.setattr(np.linalg, "svd", _boom)
    comp = q._low_rank_decompose(np.ones((4, 6), np.float32))  # 528-531
    assert comp.rank == 1 and np.all(comp.a == 0) and np.all(comp.b == 0)
