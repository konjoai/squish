"""Coverage for quantizer.py Rust-backed paths and pure-NumPy fallbacks.

The Rust extension (``squish_quant``) is not built on the Linux/CI runners, so
every ``_squish_quant``-backed branch is unreachable there. These tests inject a
fake ``_squish_quant`` (a passthrough recording stub) to exercise those branches,
and call the NumPy fallbacks directly (they run whenever the extension is absent).
Pure numpy + monkeypatch — host-agnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from squish.quant import quantizer as q


class _FakeSquishQuant:
    """Minimal stand-in for the squish_quant Rust extension. Each method returns
    a recognisable array so callers can assert the value is passed through."""

    # INT8
    def quantize_int8_f32(self, emb):
        return (np.ones_like(emb, dtype=np.int8), np.ones(emb.shape[0], np.float32))

    def quantize_int8_grouped(self, emb, gs):
        n, d = emb.shape
        return (np.ones((n, d), np.int8), np.ones((n, d // gs), np.float32))

    def dequantize_int8_f32(self, q_arr, s):
        return np.full((q_arr.shape[0], q_arr.shape[1]), 8.0, np.float32)

    def dequantize_int8_grouped(self, q_arr, s, gs):
        return np.full(q_arr.shape, 9.0, np.float32)

    # BF16-native
    def quantize_int4_asymmetric_bf16(self, arr, gs):
        n, d = arr.shape
        return (
            np.zeros((n, d // 2), np.uint8),
            np.ones((n, d // gs), np.float32),
            np.zeros((n, d // gs), np.float32),
        )

    def quantize_int8_bf16(self, arr):
        return (np.ones(arr.shape, np.int8), np.ones(arr.shape[0], np.float32))

    def quantize_int8_grouped_bf16(self, arr, gs):
        n, d = arr.shape
        return (np.ones((n, d), np.int8), np.ones((n, d // gs), np.float32))

    # INT4
    def quantize_int4_grouped(self, emb, gs):
        n, d = emb.shape
        return (np.zeros((n, d // 2), np.uint8), np.ones((n, d // gs), np.float32))

    def dequantize_int4_grouped(self, packed, scales, gs):
        return np.full((packed.shape[0], packed.shape[1] * 2), 4.0, np.float32)

    def quantize_int4_asymmetric_grouped(self, emb, gs):
        n, d = emb.shape
        return (
            np.zeros((n, d // 2), np.uint8),
            np.ones((n, d // gs), np.float32),
            np.zeros((n, d // gs), np.float32),
        )

    def dequantize_int4_asymmetric_grouped(self, packed, scales, offsets, gs):
        return np.full((packed.shape[0], packed.shape[1] * 2), 5.0, np.float32)

    def quantized_matmul_int4(self, w_codes, scales, offsets, x, gs):
        return np.full((x.shape[0], w_codes.shape[0]), 7.0, np.float32)

    def sqint2_residual_gemv(self, l_u16, r_u16, x):
        return np.full((x.shape[0], 1), 3.0, np.float32)


@pytest.fixture()
def fake_rust(monkeypatch):
    fake = _FakeSquishQuant()
    monkeypatch.setattr(q, "_squish_quant", fake)
    return fake


# ── pure-NumPy helpers (no extension needed) ───────────────────────────────────


def test_is_bf16_array():
    assert q._is_bf16_array(np.zeros(4, np.uint16)) is True
    assert q._is_bf16_array(np.zeros(4, np.float32)) is False


def test_quantized_matmul_int4_numpy_fallback(monkeypatch):
    monkeypatch.setattr(q, "_squish_quant", None)
    w_codes = np.zeros((4, 4), np.uint8)  # out_f=4, in_f=8
    scales = np.ones((4, 2), np.float32)
    offsets = np.zeros((4, 2), np.float32)
    x = np.ones((3, 8), np.float32)
    out = q.quantized_matmul_int4(w_codes, scales, offsets, x, group_size=4)
    assert out.shape == (3, 4) and out.dtype == np.float32


def test_sqint2_residual_gemv_numpy_fallback(monkeypatch):
    monkeypatch.setattr(q, "_squish_quant", None)
    L = np.ones((4, 2), np.float16)  # (out_f, rank)
    R = np.ones((2, 6), np.float16)  # (rank, in_f)
    x = np.ones((3, 6), np.float32)
    out = q.sqint2_residual_gemv(L, R, x)
    assert out.shape == (3, 4)


def test_mean_cosine_similarity_edges():
    a = np.array([[0.0, 0.0], [1.0, 0.0]], np.float32)
    b = np.array([[0.0, 0.0], [0.0, 1.0]], np.float32)
    # row 0: both zero → 1.0 ; row 1: orthogonal → 0.0 → mean 0.5
    assert q.mean_cosine_similarity(a, b) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="Shape mismatch"):
        q.mean_cosine_similarity(np.zeros((2, 2)), np.zeros((3, 2)))


# ── Rust-backed paths via the fake extension ───────────────────────────────────


def test_quantize_and_reconstruct_rust_per_row(fake_rust):
    emb = np.ones((2, 8), np.float32)
    res = q.quantize_embeddings(emb, backend="rust")
    recon = q.reconstruct_embeddings(res, backend="rust")
    assert recon.shape == (2, 8)


def test_quantize_and_reconstruct_rust_grouped(fake_rust):
    emb = np.ones((2, 8), np.float32)
    res = q._quantize_rust(emb, group_size=4)
    assert res.group_size == 4
    assert q._reconstruct_rust(res).shape == (2, 8)


def test_quantize_bf16_native_int4(fake_rust):
    arr = np.zeros((2, 8), np.uint16)
    out = q.quantize_bf16_native(arr, group_size=64, use_int4=True)
    assert set(out) == {"__q4a", "__s4a", "__z4a", "__shape"}


def test_quantize_bf16_native_int4_odd_dim_falls_back(fake_rust):
    arr = np.zeros((1, 7), np.uint16)  # odd d → packing impossible → None
    assert q.quantize_bf16_native(arr, use_int4=True) is None


def test_quantize_bf16_native_int8_grouped_and_plain(fake_rust):
    grouped = q.quantize_bf16_native(np.zeros((2, 8), np.uint16), use_int4=False)
    assert set(grouped) == {"__q", "__s", "__shape"}
    plain = q.quantize_bf16_native(np.zeros((1, 3), np.uint16), use_int4=False)
    assert set(plain) == {"__q", "__s", "__shape"}


def test_quantize_bf16_native_returns_none_without_extension(monkeypatch):
    monkeypatch.setattr(q, "_squish_quant", None)
    assert q.quantize_bf16_native(np.zeros((1, 8), np.uint16), use_int4=True) is None


def test_int4_rust_entrypoints(fake_rust):
    emb = np.ones((2, 8), np.float32)
    packed, scales = q.quantize_int4(emb, group_size=4)
    assert q.dequantize_int4(packed, scales, group_size=4).shape == (2, 8)
    p, s, o = q.quantize_int4_asymmetric(emb, group_size=4)
    assert q.dequantize_int4_asymmetric(p, s, o, group_size=4).shape == (2, 8)


def test_quantized_matmul_int4_rust_branch(fake_rust):
    out = q.quantized_matmul_int4(
        np.zeros((4, 4), np.uint8),
        np.ones((4, 2), np.float32),
        np.zeros((4, 2), np.float32),
        np.ones((3, 8), np.float32),
        group_size=4,
    )
    assert out[0, 0] == pytest.approx(7.0)  # came from the fake rust kernel


def test_sqint2_residual_gemv_rust_branch(fake_rust):
    out = q.sqint2_residual_gemv(
        np.ones((4, 2), np.float16), np.ones((2, 6), np.float16), np.ones((3, 6), np.float32)
    )
    assert out[0, 0] == pytest.approx(3.0)


def test_quantize_int4_asymmetric_mse_grid_search(fake_rust):
    # n>1 runs the pure-numpy MSE grid search before the final rust call.
    emb = np.array(
        [[0.0, 1.0, 2.0, 9.0, 0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        np.float32,
    )
    p, s, o = q.quantize_int4_asymmetric_mse(emb, group_size=4, n_clip_candidates=4)
    assert p.shape == (2, 4)


def test_quantize_int4_asymmetric_mse_1d_skips_search(fake_rust):
    # n==1 takes the early return (no grid search; 1-D scale invariance preserved).
    emb = np.arange(8, dtype=np.float32).reshape(1, 8)
    p, s, o = q.quantize_int4_asymmetric_mse(emb, group_size=4)
    assert p.shape == (1, 4)


def test_quantize_bf16_native_reshapes_1d_and_3d(fake_rust):
    out_1d = q.quantize_bf16_native(np.zeros(8, np.uint16), use_int4=True)
    assert out_1d["__shape"].tolist() == [1, 8]
    out_3d = q.quantize_bf16_native(np.zeros((2, 2, 8), np.uint16), use_int4=True)
    assert out_3d["__shape"].tolist() == [4, 8]


def test_quantize_bf16_native_mode_unavailable_returns_none(monkeypatch):
    class _NoBf16:
        pass  # neither quantize_int4_asymmetric_bf16 nor quantize_int8_bf16

    monkeypatch.setattr(q, "_squish_quant", _NoBf16())
    assert q.quantize_bf16_native(np.zeros((1, 8), np.uint16), use_int4=True) is None
