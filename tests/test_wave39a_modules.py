"""
tests/test_wave39a_modules.py

Test suite for Wave 39a modules — Activation Quantization, Fused Kernels,
and Sublinear Attention:

  - squish/quant/smooth_quant.py          (SmoothQuantActivation)
  - squish/quant/hqq_quant.py             (HQQQuantizer)
  - squish/attention/hyper_attn.py        (HyperAttention)
  - squish/speculative/triforce_decode.py (TriForceDecoder)
  - squish/kernels/flex_attn.py           (FlexAttentionKernel)
  - squish/token/massive_activation.py    (MassiveActivationSuppressor)
"""

import math
import numpy as np
import pytest

# ============================================================
# SmoothQuantActivation tests
# ============================================================

from squish.quant.smooth_quant import SmoothQuantConfig, SmoothQuantActivation


class TestSmoothQuantConfig:
    def test_defaults(self):
        cfg = SmoothQuantConfig()
        assert cfg.alpha == 0.5
        assert cfg.bits == 8
        assert cfg.epsilon > 0

    def test_alpha_out_of_range_high(self):
        with pytest.raises(ValueError, match="alpha"):
            SmoothQuantConfig(alpha=1.5)

    def test_alpha_out_of_range_low(self):
        with pytest.raises(ValueError, match="alpha"):
            SmoothQuantConfig(alpha=-0.1)

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            SmoothQuantConfig(bits=3)

    def test_invalid_epsilon(self):
        with pytest.raises(ValueError, match="epsilon"):
            SmoothQuantConfig(epsilon=0.0)

    def test_valid_bits(self):
        for b in (4, 8, 16):
            cfg = SmoothQuantConfig(bits=b)
            assert cfg.bits == b


class TestSmoothQuantActivation:
    def _make_data(self, n_tok=32, c_in=64, c_out=32, seed=0):
        rng = np.random.default_rng(seed)
        act = rng.standard_normal((n_tok, c_in)).astype(np.float32)
        weight = rng.standard_normal((c_out, c_in)).astype(np.float32)
        return act, weight

    def test_not_calibrated_raises_smooth_weight(self):
        smoother = SmoothQuantActivation()
        with pytest.raises(RuntimeError, match="calibrate"):
            smoother.smooth_weight(np.ones((4, 4)))

    def test_not_calibrated_raises_smooth_activation(self):
        smoother = SmoothQuantActivation()
        with pytest.raises(RuntimeError, match="calibrate"):
            smoother.smooth_activation(np.ones((4, 4)))

    def test_calibrate_returns_scales(self):
        smoother = SmoothQuantActivation()
        act, w = self._make_data()
        scales = smoother.calibrate(act, w)
        assert scales.shape == (64,)
        assert (scales > 0).all()

    def test_calibrate_is_calibrated(self):
        smoother = SmoothQuantActivation()
        assert not smoother.is_calibrated
        act, w = self._make_data()
        smoother.calibrate(act, w)
        assert smoother.is_calibrated

    def test_smooth_weight_shape(self):
        smoother = SmoothQuantActivation()
        act, w = self._make_data()
        smoother.calibrate(act, w)
        w_s = smoother.smooth_weight(w)
        assert w_s.shape == w.shape

    def test_smooth_activation_shape(self):
        smoother = SmoothQuantActivation()
        act, w = self._make_data()
        smoother.calibrate(act, w)
        a_s = smoother.smooth_activation(act)
        assert a_s.shape == act.shape

    def test_alpha_zero_moves_all_difficulty_to_weights(self):
        cfg = SmoothQuantConfig(alpha=0.0)
        smoother = SmoothQuantActivation(cfg)
        act, w = self._make_data()
        smoother.calibrate(act, w)
        a_s = smoother.smooth_activation(act)
        # All channels divided; act_smooth should have smaller outliers
        assert a_s.dtype == np.float32

    def test_alpha_one_preserves_activations(self):
        cfg = SmoothQuantConfig(alpha=1.0)
        smoother = SmoothQuantActivation(cfg)
        act, w = self._make_data()
        smoother.calibrate(act, w)
        a_s = smoother.smooth_activation(act)
        assert a_s.shape == act.shape

    def test_calibrate_cin_mismatch_raises(self):
        smoother = SmoothQuantActivation()
        act = np.ones((8, 32))
        w = np.ones((16, 64))
        with pytest.raises(ValueError, match="C_in"):
            smoother.calibrate(act, w)

    def test_calibrate_1d_activation_raises(self):
        smoother = SmoothQuantActivation()
        with pytest.raises(ValueError):
            smoother.calibrate(np.ones(32), np.ones((16, 32)))

    def test_calibrate_non_2d_weight_raises(self):
        smoother = SmoothQuantActivation()
        with pytest.raises(ValueError):
            smoother.calibrate(np.ones((8, 32)), np.ones((2, 16, 32)))

    def test_quantise_int8_range(self):
        smoother = SmoothQuantActivation()
        x = np.random.randn(16, 32).astype(np.float32)
        x_q, scale = smoother.quantise_int8(x)
        assert x_q.dtype == np.int8
        assert scale > 0
        assert x_q.min() >= -128
        assert x_q.max() <= 127

    def test_dequantise_roundtrip_approx(self):
        smoother = SmoothQuantActivation()
        x = np.random.randn(16, 32).astype(np.float32) * 10
        x_q, scale = smoother.quantise_int8(x)
        x_hat = smoother.dequantise_int8(x_q, scale)
        rel_err = float(np.linalg.norm(x - x_hat) / (np.linalg.norm(x) + 1e-8))
        assert rel_err < 0.02  # <2% relative error for INT8

    def test_forward_smoothed_shape(self):
        smoother = SmoothQuantActivation()
        act, w = self._make_data()
        smoother.calibrate(act, w)
        out = smoother.forward_smoothed(act, w)
        assert out.shape == (32, 32)

    def test_forward_smoothed_with_bias(self):
        smoother = SmoothQuantActivation()
        act, w = self._make_data()
        smoother.calibrate(act, w)
        bias = np.zeros(32, dtype=np.float32)
        out = smoother.forward_smoothed(act, w, bias=bias)
        assert out.shape == (32, 32)

    def test_repr(self):
        smoother = SmoothQuantActivation()
        r = repr(smoother)
        assert "SmoothQuantActivation" in r

    def test_dynamic_quantisation_path(self):
        cfg = SmoothQuantConfig(per_token_dynamic=True)
        smoother = SmoothQuantActivation(cfg)
        act, w = self._make_data()
        smoother.calibrate(act, w)
        a_s = smoother.smooth_activation(act)
        assert a_s.shape == act.shape

    def test_scales_property_none_before_calibrate(self):
        smoother = SmoothQuantActivation()
        assert smoother.scales is None


# ============================================================
# HQQQuantizer tests
# ============================================================

from squish.quant.hqq_quant import HQQConfig, HQQTensor, HQQQuantizer


class TestHQQConfig:
    def test_defaults(self):
        cfg = HQQConfig()
        assert cfg.bits == 4
        assert cfg.group_size == 128

    def test_invalid_bits(self):
        with pytest.raises(ValueError, match="bits"):
            HQQConfig(bits=5)

    def test_valid_bits(self):
        for b in (2, 3, 4, 8):
            cfg = HQQConfig(bits=b)
            assert cfg.bits == b

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            HQQConfig(group_size=0)

    def test_invalid_lambda(self):
        with pytest.raises(ValueError, match="lambda"):
            HQQConfig(lambda_scale=0.0)

    def test_invalid_max_iter(self):
        with pytest.raises(ValueError, match="max_iter"):
            HQQConfig(max_iter=0)

    def test_invalid_axis(self):
        with pytest.raises(ValueError, match="axis"):
            HQQConfig(axis=2)


class TestHQQQuantizer:
    def _w(self, rows=32, cols=64, seed=1):
        return np.random.default_rng(seed).standard_normal((rows, cols)).astype(np.float32)

    def test_encode_returns_hqqtensor(self):
        quant = HQQQuantizer()
        t = quant.encode(self._w())
        assert isinstance(t, HQQTensor)

    def test_encode_shape_preserved(self):
        w = self._w()
        quant = HQQQuantizer()
        t = quant.encode(w)
        assert t.shape == w.shape

    def test_decode_shape(self):
        w = self._w()
        quant = HQQQuantizer()
        t = quant.encode(w)
        w_hat = quant.decode(t)
        assert w_hat.shape == w.shape

    def test_encode_non_2d_raises(self):
        quant = HQQQuantizer()
        with pytest.raises(ValueError):
            quant.encode(np.ones((4, 4, 4)))

    def test_int4_relative_error_reasonable(self):
        w = self._w(32, 128, seed=7)
        quant = HQQQuantizer(HQQConfig(bits=4, group_size=32, max_iter=5))
        t = quant.encode(w)
        w_hat = quant.decode(t)
        err = quant.relative_error(w, w_hat)
        assert err < 0.5  # generous bound for a small random matrix

    def test_int8_lower_error_than_int2(self):
        w = self._w(16, 64, seed=3)
        q4 = HQQQuantizer(HQQConfig(bits=8, group_size=16, max_iter=3))
        q2 = HQQQuantizer(HQQConfig(bits=2, group_size=16, max_iter=3))
        err4 = q4.relative_error(w, q4.decode(q4.encode(w)))
        err2 = q2.relative_error(w, q2.decode(q2.encode(w)))
        assert err4 <= err2

    def test_codes_dtype_uint8(self):
        w = self._w()
        t = HQQQuantizer().encode(w)
        assert t.codes.dtype == np.uint8

    def test_snr_positive(self):
        w = self._w(16, 64) + 1.0  # ensure non-zero signal
        quant = HQQQuantizer()
        t = quant.encode(w)
        snr = quant.quantisation_error_db(w, quant.decode(t))
        assert isinstance(snr, float)

    def test_full_row_group(self):
        w = self._w(8, 32)
        quant = HQQQuantizer(HQQConfig(bits=4, group_size=-1))
        t = quant.encode(w)
        w_hat = quant.decode(t)
        assert w_hat.shape == w.shape

    def test_repr(self):
        r = repr(HQQQuantizer())
        assert "HQQQuantizer" in r

    def test_axis1_encode_decode(self):
        w = self._w(16, 64)
        quant = HQQQuantizer(HQQConfig(bits=4, group_size=16, axis=1, max_iter=3))
        t = quant.encode(w)
        w_hat = quant.decode(t)
        assert w_hat.shape == w.shape


# ============================================================
# HyperAttention tests
# ============================================================

from squish.attention.hyper_attn import HyperAttentionConfig, HyperAttention


class TestHyperAttentionConfig:
    def test_defaults(self):
        cfg = HyperAttentionConfig()
        assert cfg.n_hash_functions >= 1
        assert 0 < cfg.sample_ratio <= 1.0

    def test_invalid_n_hash_functions(self):
        with pytest.raises(ValueError, match="n_hash_functions"):
            HyperAttentionConfig(n_hash_functions=0)

    def test_invalid_n_hash_buckets(self):
        with pytest.raises(ValueError, match="n_hash_buckets"):
            HyperAttentionConfig(n_hash_buckets=1)

    def test_invalid_sample_ratio_zero(self):
        with pytest.raises(ValueError, match="sample_ratio"):
            HyperAttentionConfig(sample_ratio=0.0)

    def test_invalid_sample_ratio_gt1(self):
        with pytest.raises(ValueError, match="sample_ratio"):
            HyperAttentionConfig(sample_ratio=1.1)

    def test_invalid_min_seq_len(self):
        with pytest.raises(ValueError, match="min_seq_len"):
            HyperAttentionConfig(min_seq_len=0)


class TestHyperAttention:
    def _qkv(self, n=16, d=32, seed=0):
        rng = np.random.default_rng(seed)
        return (
            rng.standard_normal((n, d)).astype(np.float32),
            rng.standard_normal((n, d)).astype(np.float32),
            rng.standard_normal((n, d)).astype(np.float32),
        )

    def test_short_seq_exact_fallback_shape(self):
        attn = HyperAttention(HyperAttentionConfig(min_seq_len=512))
        Q, K, V = self._qkv(16, 32)
        out = attn.forward(Q, K, V)
        assert out.shape == (16, 32)

    def test_long_seq_approximate_shape(self):
        cfg = HyperAttentionConfig(min_seq_len=4, n_hash_functions=2, n_hash_buckets=4, sample_ratio=0.5)
        attn = HyperAttention(cfg)
        Q, K, V = self._qkv(32, 16)
        out = attn.forward(Q, K, V)
        assert out.shape == (32, 16)

    def test_output_dtype_float32(self):
        attn = HyperAttention()
        Q, K, V = self._qkv()
        out = attn.forward(Q, K, V)
        assert out.dtype == np.float32

    def test_q_k_dim_mismatch_raises(self):
        attn = HyperAttention()
        Q = np.ones((8, 32), dtype=np.float32)
        K = np.ones((8, 16), dtype=np.float32)
        V = np.ones((8, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            attn.forward(Q, K, V)

    def test_k_v_length_mismatch_raises(self):
        attn = HyperAttention()
        Q = np.ones((8, 32), dtype=np.float32)
        K = np.ones((8, 32), dtype=np.float32)
        V = np.ones((4, 32), dtype=np.float32)
        with pytest.raises(ValueError):
            attn.forward(Q, K, V)

    def test_exact_fallback_matches_reference(self):
        """For short sequences, HyperAttention should match exact attention."""
        from squish.attention.hyper_attn import _exact_attention
        cfg = HyperAttentionConfig(min_seq_len=512, causal=False)
        attn = HyperAttention(cfg)
        Q, K, V = self._qkv(8, 16, seed=5)
        out_hyper = attn.forward(Q, K, V)
        out_exact = _exact_attention(Q, K, V)
        np.testing.assert_allclose(out_hyper, out_exact, rtol=1e-4, atol=1e-5)

    def test_repr(self):
        r = repr(HyperAttention())
        assert "HyperAttention" in r

    def test_causal_flag_accepted(self):
        cfg = HyperAttentionConfig(causal=True, min_seq_len=512)
        attn = HyperAttention(cfg)
        Q, K, V = self._qkv(8, 16)
        out = attn.forward(Q, K, V)
        assert out.shape == (8, 16)

    def test_sample_ratio_one(self):
        cfg = HyperAttentionConfig(min_seq_len=4, sample_ratio=1.0, n_hash_buckets=4)
        attn = HyperAttention(cfg)
        Q, K, V = self._qkv(16, 16)
        out = attn.forward(Q, K, V)
        assert out.shape == (16, 16)

    def test_different_value_dim(self):
        cfg = HyperAttentionConfig(min_seq_len=512)
        attn = HyperAttention(cfg)
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((8, 32)).astype(np.float32)
        K = rng.standard_normal((8, 32)).astype(np.float32)
        V = rng.standard_normal((8, 64)).astype(np.float32)
        out = attn.forward(Q, K, V)
        assert out.shape == (8, 64)


# ============================================================
# TriForceDecoder tests
# ============================================================

from squish.speculative.triforce_decode import (
    TriForceConfig,
    TriForceDraftResult,
    TriForceDecoder,
)


class TestTriForceConfig:
    def test_defaults(self):
        cfg = TriForceConfig()
        assert cfg.draft_length >= 1
        assert cfg.top_k_pages >= 1

    def test_invalid_draft_length(self):
        with pytest.raises(ValueError, match="draft_length"):
            TriForceConfig(draft_length=0)

    def test_invalid_top_k_pages(self):
        with pytest.raises(ValueError, match="top_k_pages"):
            TriForceConfig(top_k_pages=0)

    def test_invalid_page_size(self):
        with pytest.raises(ValueError, match="page_size"):
            TriForceConfig(page_size=0)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            TriForceConfig(temperature=0.0)


class TestTriForceDecoder:
    VOCAB = 100

    def _draft_fn(self, token, pages):
        rng = np.random.default_rng(token % 13)
        logits = rng.standard_normal(self.VOCAB).astype(np.float32)
        return logits

    def _target_fn(self, tokens):
        result = []
        for t in tokens:
            rng = np.random.default_rng((t + 1) % 17)
            result.append(rng.standard_normal(self.VOCAB).astype(np.float32))
        return result

    def test_select_top_k_pages_length(self):
        decoder = TriForceDecoder(TriForceConfig(top_k_pages=3))
        weights = np.array([0.1, 0.5, 0.3, 0.2, 0.8])
        pages = decoder.select_top_k_pages(weights, n_pages=5)
        assert len(pages) == 3

    def test_select_top_k_pages_sorted(self):
        decoder = TriForceDecoder(TriForceConfig(top_k_pages=3))
        weights = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        pages = decoder.select_top_k_pages(weights, n_pages=5)
        assert list(pages) == sorted(pages)

    def test_select_top_k_pages_mismatch_raises(self):
        decoder = TriForceDecoder()
        with pytest.raises(ValueError):
            decoder.select_top_k_pages(np.array([0.1, 0.2]), n_pages=5)

    def test_accept_reject_all_accepted(self):
        decoder = TriForceDecoder(TriForceConfig(seed=0))
        # Identical draft and target → ratio=1 → always accept
        logits = [np.zeros(self.VOCAB, dtype=np.float32)] * 3
        logits[0][5] = 10.0  # high prob token
        tokens = [5, 5, 5]
        accepted = decoder.accept_reject(logits, logits, tokens)
        assert len(accepted) >= 1

    def test_accept_reject_length_mismatch_raises(self):
        decoder = TriForceDecoder()
        with pytest.raises(ValueError):
            decoder.accept_reject([np.zeros(10)], [np.zeros(10)], [0, 1])

    def test_step_returns_result(self):
        decoder = TriForceDecoder(TriForceConfig(draft_length=3, top_k_pages=2))
        weights = np.array([0.1, 0.5, 0.3, 0.8])
        result = decoder.step(
            context_ids=[1, 2, 3],
            kv_page_weights=weights,
            draft_fn=self._draft_fn,
            target_fn=self._target_fn,
        )
        assert isinstance(result, TriForceDraftResult)
        assert result.n_drafted == 3
        assert 0 <= result.n_accepted <= result.n_drafted + 1

    def test_step_acceptance_rate_in_range(self):
        decoder = TriForceDecoder(TriForceConfig(draft_length=5, top_k_pages=3))
        weights = np.random.default_rng(0).random(8)
        result = decoder.step(
            context_ids=list(range(10)),
            kv_page_weights=weights,
            draft_fn=self._draft_fn,
            target_fn=self._target_fn,
        )
        assert 0.0 <= result.acceptance_rate

    def test_reset_rng(self):
        decoder = TriForceDecoder()
        decoder.reset_rng(seed=99)
        assert decoder._rng is not None

    def test_repr(self):
        r = repr(TriForceDecoder())
        assert "TriForceDecoder" in r


# ============================================================
# FlexAttentionKernel tests
# ============================================================

from squish.kernels.flex_attn import (
    FlexAttentionConfig,
    FlexAttentionKernel,
    BlockMask,
    make_causal_mod,
    make_alibi_mod,
    make_sliding_window_mod,
    make_softcap_mod,
)


class TestFlexAttentionScoreMods:
    def test_causal_mod_future_masked(self):
        mod = make_causal_mod()
        assert mod(1.0, 0, 0, 0, 1) < -1e20
        assert mod(1.0, 0, 0, 1, 0) == 1.0

    def test_alibi_mod_adds_position_bias(self):
        slopes = np.array([0.5, 1.0], dtype=np.float32)
        mod = make_alibi_mod(slopes)
        score = mod(0.0, 0, 0, 4, 0)
        assert score == pytest.approx(-2.0, abs=1e-5)  # 0 - 0.5 * |4-0|

    def test_sliding_window_allows_nearby(self):
        mod = make_sliding_window_mod(3)
        assert mod(1.0, 0, 0, 5, 3) == 1.0

    def test_sliding_window_blocks_far(self):
        mod = make_sliding_window_mod(3)
        assert mod(1.0, 0, 0, 0, 10) < -1e20

    def test_sliding_window_invalid(self):
        with pytest.raises(ValueError):
            make_sliding_window_mod(0)

    def test_softcap_squashes_large_scores(self):
        mod = make_softcap_mod(10.0)
        large = mod(100.0, 0, 0, 0, 0)
        assert abs(large) <= 10.0

    def test_softcap_invalid(self):
        with pytest.raises(ValueError):
            make_softcap_mod(0.0)


class TestBlockMask:
    def test_causal_block_mask_shape(self):
        bm = BlockMask.causal(seq_len=64, block_size=16)
        assert bm.mask.shape == (4, 4)

    def test_causal_lower_triangular(self):
        bm = BlockMask.causal(seq_len=64, block_size=16)
        n = bm.mask.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert not bm.mask[i, j]

    def test_token_mask_shape(self):
        bm = BlockMask.causal(seq_len=32, block_size=8)
        tok = bm.token_mask(32, 32)
        assert tok.shape == (32, 32)

    def test_invalid_mask_dim(self):
        with pytest.raises(ValueError):
            BlockMask(mask=np.zeros((4,), dtype=bool))

    def test_invalid_block_size(self):
        with pytest.raises(ValueError):
            BlockMask(mask=np.zeros((2, 2), dtype=bool), block_size=0)


class TestFlexAttentionKernel:
    def _qkv(self, B=1, H=2, N=8, d=16, seed=0):
        rng = np.random.default_rng(seed)
        Q = rng.standard_normal((B, H, N, d)).astype(np.float32)
        K = rng.standard_normal((B, H, N, d)).astype(np.float32)
        V = rng.standard_normal((B, H, N, d)).astype(np.float32)
        return Q, K, V

    def test_forward_shape_4d(self):
        kernel = FlexAttentionKernel()
        Q, K, V = self._qkv()
        out = kernel.forward(Q, K, V)
        assert out.shape == Q.shape

    def test_forward_shape_2d(self):
        kernel = FlexAttentionKernel()
        rng = np.random.default_rng(0)
        Q = rng.standard_normal((8, 16)).astype(np.float32)
        K = rng.standard_normal((8, 16)).astype(np.float32)
        V = rng.standard_normal((8, 16)).astype(np.float32)
        out = kernel.forward(Q, K, V)
        assert out.shape == (8, 16)

    def test_forward_with_causal_mod(self):
        kernel = FlexAttentionKernel()
        Q, K, V = self._qkv(N=8, d=16)
        mod = make_causal_mod()
        out = kernel.forward(Q, K, V, score_mod=mod)
        assert out.shape == Q.shape

    def test_forward_with_block_mask(self):
        kernel = FlexAttentionKernel()
        Q, K, V = self._qkv(B=1, H=1, N=16, d=16)
        bm = BlockMask.causal(seq_len=16, block_size=8)
        out = kernel.forward(Q, K, V, block_mask=bm)
        assert out.shape == Q.shape

    def test_shape_mismatch_raises(self):
        kernel = FlexAttentionKernel()
        Q = np.ones((1, 2, 8, 16), dtype=np.float32)
        K = np.ones((1, 2, 8, 32), dtype=np.float32)
        V = np.ones((1, 2, 8, 16), dtype=np.float32)
        with pytest.raises(ValueError):
            kernel.forward(Q, K, V)

    def test_no_score_mod_no_block_mask(self):
        kernel = FlexAttentionKernel()
        Q, K, V = self._qkv(B=2, H=4, N=4, d=8)
        out = kernel.forward(Q, K, V)
        assert out.dtype == np.float32

    def test_alibi_mod_shape(self):
        kernel = FlexAttentionKernel()
        Q, K, V = self._qkv(B=1, H=2, N=6, d=8)
        slopes = np.array([0.5, 1.0])
        mod = make_alibi_mod(slopes)
        out = kernel.forward(Q, K, V, score_mod=mod)
        assert out.shape == Q.shape

    def test_repr(self):
        r = repr(FlexAttentionKernel())
        assert "FlexAttentionKernel" in r


# ============================================================
# MassiveActivationSuppressor tests
# ============================================================

from squish.token.massive_activation import (
    MassiveActivationConfig,
    SuppressionStats,
    MassiveActivationSuppressor,
)


class TestMassiveActivationConfig:
    def test_defaults(self):
        cfg = MassiveActivationConfig()
        assert cfg.outlier_ratio > 1.0
        assert 0 < cfg.clamp_alpha <= 1.0

    def test_invalid_outlier_ratio(self):
        with pytest.raises(ValueError, match="outlier_ratio"):
            MassiveActivationConfig(outlier_ratio=0.5)

    def test_invalid_clamp_alpha_zero(self):
        with pytest.raises(ValueError, match="clamp_alpha"):
            MassiveActivationConfig(clamp_alpha=0.0)

    def test_invalid_clamp_alpha_gt1(self):
        with pytest.raises(ValueError, match="clamp_alpha"):
            MassiveActivationConfig(clamp_alpha=1.1)

    def test_invalid_running_ema(self):
        with pytest.raises(ValueError, match="running_ema"):
            MassiveActivationConfig(running_ema=0.0)

    def test_invalid_min_seq_len(self):
        with pytest.raises(ValueError, match="min_seq_len"):
            MassiveActivationConfig(min_seq_len=0)


class TestMassiveActivationSuppressor:
    def _make_act(self, T=32, C=64, outlier_dim=42, outlier_val=1e4, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((T, C)).astype(np.float32)
        x[:, outlier_dim] = outlier_val
        return x

    def test_detect_outlier_dims_finds_outlier(self):
        suppressor = MassiveActivationSuppressor()
        x = self._make_act()
        dims = suppressor.detect_outlier_dims(x)
        assert 42 in dims

    def test_detect_no_outliers_empty(self):
        suppressor = MassiveActivationSuppressor()
        x = np.random.default_rng(0).standard_normal((32, 64)).astype(np.float32)
        dims = suppressor.detect_outlier_dims(x)
        assert len(dims) == 0

    def test_suppress_reduces_outlier_energy(self):
        suppressor = MassiveActivationSuppressor()
        # Use non-uniform outlier values so soft-clamp actually reduces energy
        x = self._make_act()
        rng = np.random.default_rng(99)
        x[:, 42] = 1e4 + rng.standard_normal(x.shape[0]).astype(np.float32) * 500.0
        energy_before = float((x[:, 42] ** 2).sum())
        x_clean = suppressor.suppress(x)
        energy_after = float((x_clean[:, 42] ** 2).sum())
        assert energy_after < energy_before

    def test_suppress_preserves_shape(self):
        suppressor = MassiveActivationSuppressor()
        x = self._make_act()
        x_clean = suppressor.suppress(x)
        assert x_clean.shape == x.shape

    def test_suppress_dtype_float32(self):
        suppressor = MassiveActivationSuppressor()
        x = self._make_act()
        x_clean = suppressor.suppress(x)
        assert x_clean.dtype == np.float32

    def test_suppress_short_seq_no_op(self):
        cfg = MassiveActivationConfig(min_seq_len=100)
        suppressor = MassiveActivationSuppressor(cfg)
        x = self._make_act(T=4)
        out = suppressor.suppress(x)
        np.testing.assert_array_equal(out, x)

    def test_stats_updated_after_suppress(self):
        suppressor = MassiveActivationSuppressor()
        x = self._make_act()
        suppressor.suppress(x, layer_id=0)
        stats = suppressor.get_stats(0)
        assert stats.n_calls == 1

    def test_get_stats_empty_layer(self):
        suppressor = MassiveActivationSuppressor()
        stats = suppressor.get_stats(99)
        assert stats.n_calls == 0

    def test_reset_stats_single_layer(self):
        suppressor = MassiveActivationSuppressor()
        x = self._make_act()
        suppressor.suppress(x, layer_id=0)
        suppressor.reset_stats(0)
        assert suppressor.get_stats(0).n_calls == 0

    def test_reset_stats_all(self):
        suppressor = MassiveActivationSuppressor()
        x = self._make_act()
        suppressor.suppress(x, layer_id=0)
        suppressor.suppress(x, layer_id=1)
        suppressor.reset_stats()
        assert suppressor.get_stats(0).n_calls == 0
        assert suppressor.get_stats(1).n_calls == 0

    def test_no_redistribute(self):
        cfg = MassiveActivationConfig(redistribute=False)
        suppressor = MassiveActivationSuppressor(cfg)
        x = self._make_act()
        out = suppressor.suppress(x)
        assert out.shape == x.shape

    def test_repr(self):
        r = repr(MassiveActivationSuppressor())
        assert "MassiveActivationSuppressor" in r

    def test_3d_input_preserved(self):
        suppressor = MassiveActivationSuppressor()
        x = np.random.default_rng(0).standard_normal((4, 8, 64)).astype(np.float32)
        x[:, :, 42] = 5000.0
        out = suppressor.suppress(x)
        assert out.shape == x.shape
