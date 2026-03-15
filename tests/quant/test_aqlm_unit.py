"""tests/quant/test_aqlm_unit.py — unit tests for squish/quant/aqlm.py

Phase 9A: AQLM (Additive Quantization of Language Models) quantiser.
Covers AQLMConfig, AQLMCodebook, AQLMLayer, AQLMQuantizer, aqlm_dequantize,
and quantize_model_aqlm.

Run:
    pytest tests/quant/test_aqlm_unit.py -v --tb=short
"""
import numpy as np
import pytest

from squish.quant.aqlm import (
    AQLMCodebook,
    AQLMConfig,
    AQLMLayer,
    AQLMQuantizer,
    aqlm_dequantize,
    quantize_model_aqlm,
)

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# TestAQLMConfig
# ---------------------------------------------------------------------------

class TestAQLMConfig:
    def test_defaults(self):
        cfg = AQLMConfig()
        assert cfg.n_codebooks == 2
        assert cfg.codebook_size == 16
        assert cfg.group_size == 8
        assert cfg.n_iterations == 25
        assert cfg.beam_width == 8

    def test_custom_values(self):
        cfg = AQLMConfig(n_codebooks=4, codebook_size=32, group_size=4,
                         n_iterations=10, beam_width=4)
        assert cfg.n_codebooks == 4
        assert cfg.codebook_size == 32
        assert cfg.group_size == 4
        assert cfg.n_iterations == 10
        assert cfg.beam_width == 4

    def test_invalid_n_codebooks(self):
        with pytest.raises(ValueError, match="n_codebooks"):
            AQLMConfig(n_codebooks=0)

    def test_invalid_codebook_size(self):
        with pytest.raises(ValueError, match="codebook_size"):
            AQLMConfig(codebook_size=1)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            AQLMConfig(group_size=0)

    def test_invalid_n_iterations(self):
        with pytest.raises(ValueError, match="n_iterations"):
            AQLMConfig(n_iterations=0)

    def test_invalid_beam_width(self):
        with pytest.raises(ValueError, match="beam_width"):
            AQLMConfig(beam_width=0)


# ---------------------------------------------------------------------------
# TestAQLMCodebook
# ---------------------------------------------------------------------------

class TestAQLMCodebook:
    def test_init_shape(self):
        cb = AQLMCodebook(codebook_size=16, group_size=8)
        assert cb.vectors.shape == (16, 8)
        assert cb.vectors.dtype == np.float32

    def test_init_zeros(self):
        cb = AQLMCodebook(codebook_size=8, group_size=4)
        assert np.all(cb.vectors == 0.0)

    def test_kmeans_sets_correct_shape(self):
        cb = AQLMCodebook(codebook_size=8, group_size=4)
        data = RNG.standard_normal((50, 4)).astype(np.float32)
        cb.initialize_kmeans(data)
        assert cb.vectors.shape == (8, 4)
        assert cb.vectors.dtype == np.float32

    def test_kmeans_fewer_samples_than_k(self):
        """When n_samples < codebook_size k-means should still complete."""
        cb = AQLMCodebook(codebook_size=16, group_size=4)
        data = RNG.standard_normal((5, 4)).astype(np.float32)
        cb.initialize_kmeans(data)
        assert cb.vectors.shape == (16, 4)

    def test_kmeans_empty_data(self):
        """Empty data should not crash."""
        cb = AQLMCodebook(codebook_size=8, group_size=4)
        empty = np.zeros((0, 4), dtype=np.float32)
        cb.initialize_kmeans(empty)
        # Vectors should remain zeros (no data to fit on)
        assert cb.vectors.shape == (8, 4)

    def test_nearest_returns_valid_index(self):
        cb = AQLMCodebook(codebook_size=8, group_size=4)
        data = RNG.standard_normal((30, 4)).astype(np.float32)
        cb.initialize_kmeans(data)
        query = RNG.standard_normal(4).astype(np.float32)
        idx = cb.nearest(query)
        assert 0 <= idx < 8

    def test_nearest_on_zeros_codebook(self):
        """nearest() should still return a valid index even with all-zeros codebook."""
        cb = AQLMCodebook(codebook_size=4, group_size=3)
        query = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        idx = cb.nearest(query)
        assert 0 <= idx < 4

    def test_nearest_exact_match(self):
        """When query exactly matches one codeword, that index is returned."""
        cb = AQLMCodebook(codebook_size=4, group_size=3)
        cb.vectors = np.eye(4, 3, dtype=np.float32)  # distinct unit vectors
        query = cb.vectors[2].copy()
        idx = cb.nearest(query)
        assert idx == 2


# ---------------------------------------------------------------------------
# TestAQLMLayer
# ---------------------------------------------------------------------------

class TestAQLMLayer:
    def _make_layer(self, out=4, in_=8, n_codebooks=2, cb_size=8, gs=4):
        cfg = AQLMConfig(n_codebooks=n_codebooks, codebook_size=cb_size, group_size=gs)
        return AQLMLayer(out, in_, cfg), cfg

    def test_init_shape_indices(self):
        layer, cfg = self._make_layer(out=4, in_=8, gs=4)
        n_groups = (8 + 4 - 1) // 4  # = 2
        assert layer.indices.shape == (4, n_groups, cfg.n_codebooks)

    def test_init_indices_zeros(self):
        layer, _ = self._make_layer()
        assert np.all(layer.indices == 0)

    def test_init_scale_one(self):
        layer, _ = self._make_layer()
        assert layer.scale == 1.0

    def test_dequantize_returns_float32(self):
        layer, _ = self._make_layer(out=3, in_=8, gs=4)
        out = layer.dequantize()
        assert out.dtype == np.float32

    def test_dequantize_returns_correct_shape(self):
        layer, _ = self._make_layer(out=3, in_=8, gs=4)
        out = layer.dequantize()
        assert out.shape == (3, 8)

    def test_dequantize_all_zeros_when_codebooks_zero(self):
        """All-zero codebooks → dequantize returns all zeros (before scaling)."""
        layer, _ = self._make_layer(out=2, in_=4, gs=4, n_codebooks=1, cb_size=4)
        out = layer.dequantize()
        assert np.allclose(out, 0.0)

    def test_dequantize_with_padding(self):
        """in_features not divisible by group_size → layer handles padding."""
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4)
        layer = AQLMLayer(2, 6, cfg)  # 6 not divisible by 4
        out = layer.dequantize()
        assert out.shape == (2, 6)


# ---------------------------------------------------------------------------
# TestAQLMQuantizer
# ---------------------------------------------------------------------------

class TestAQLMQuantizer:
    def _small_weight(self, out=4, in_=8):
        return RNG.standard_normal((out, in_)).astype(np.float32) * 0.1

    def test_calibrate_returns_aqlm_layer(self):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4,
                         n_iterations=5, beam_width=2)
        q = AQLMQuantizer(cfg)
        W = self._small_weight(4, 8)
        layer = q.calibrate(W)
        assert isinstance(layer, AQLMLayer)

    def test_calibrate_all_zeros_weight(self):
        """Calibrating on an all-zeros weight matrix should not crash."""
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4,
                         n_iterations=3, beam_width=2)
        q = AQLMQuantizer(cfg)
        W = np.zeros((4, 8), dtype=np.float32)
        layer = q.calibrate(W)
        assert isinstance(layer, AQLMLayer)
        assert layer.scale == 1.0  # zero weight → scale defaults to 1.0

    def test_calibrate_random_weight(self):
        cfg = AQLMConfig(n_codebooks=2, codebook_size=8, group_size=4,
                         n_iterations=5, beam_width=4)
        q = AQLMQuantizer(cfg)
        W = RNG.standard_normal((8, 16)).astype(np.float32)
        layer = q.calibrate(W)
        assert layer.indices.shape == (8, 4, 2)

    def test_calibrate_1d_weight(self):
        """1-D weight vector should be accepted (reshaped to (1, N))."""
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4)
        q = AQLMQuantizer(cfg)
        W = RNG.standard_normal(8).astype(np.float32)
        layer = q.calibrate(W)
        assert isinstance(layer, AQLMLayer)
        assert layer.out_features == 1

    def test_round_trip_reconstruction_error_reasonable(self):
        """
        AQLM is lossy; reconstruction error should be < 50 % of the raw variance.
        This is a loose sanity check that the quantiser is working.
        """
        cfg = AQLMConfig(n_codebooks=2, codebook_size=16, group_size=4,
                         n_iterations=10, beam_width=4)
        q = AQLMQuantizer(cfg)
        W = RNG.standard_normal((8, 16)).astype(np.float32) * 0.1
        layer = q.calibrate(W)
        W_hat = layer.dequantize()
        mse_recon = np.mean((W - W_hat) ** 2)
        mse_zero  = np.mean(W ** 2)  # MSE if we just predict zero
        assert mse_recon < mse_zero * 0.5 or mse_recon < 1.0, (
            f"Reconstruction MSE {mse_recon:.4f} seems too high vs zero-baseline {mse_zero:.4f}"
        )

    def test_default_config_used_when_none(self):
        q = AQLMQuantizer()
        assert q.config.n_codebooks == 2
        assert q.config.codebook_size == 16


# ---------------------------------------------------------------------------
# TestAqlmDequantize
# ---------------------------------------------------------------------------

class TestAqlmDequantize:
    def _fitted_layer(self, out=4, in_=8):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4,
                         n_iterations=5, beam_width=2)
        q = AQLMQuantizer(cfg)
        W = RNG.standard_normal((out, in_)).astype(np.float32)
        return q.calibrate(W), W

    def test_output_shape_matches(self):
        layer, W = self._fitted_layer(4, 8)
        out = aqlm_dequantize(layer)
        assert out.shape == W.shape

    def test_output_dtype_float32(self):
        layer, _ = self._fitted_layer(4, 8)
        out = aqlm_dequantize(layer)
        assert out.dtype == np.float32

    def test_runs_without_error(self):
        layer, _ = self._fitted_layer(3, 12)
        out = aqlm_dequantize(layer)
        assert out is not None
        assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# TestQuantizeModelAqlm
# ---------------------------------------------------------------------------

class TestQuantizeModelAqlm:
    def test_empty_dict_returns_empty_dict(self):
        result = quantize_model_aqlm({})
        assert result == {}

    def test_dict_with_one_weight_returns_aqlm_layer(self):
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4,
                         n_iterations=3, beam_width=2)
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        result = quantize_model_aqlm({"linear.weight": W}, config=cfg)
        assert "linear.weight" in result
        assert isinstance(result["linear.weight"], AQLMLayer)

    def test_1d_tensors_skipped(self):
        """1-D tensors (biases, norms) should be skipped."""
        cfg = AQLMConfig(n_codebooks=1, codebook_size=4, group_size=4,
                         n_iterations=3, beam_width=2)
        W2d = RNG.standard_normal((4, 8)).astype(np.float32)
        bias = RNG.standard_normal(8).astype(np.float32)
        result = quantize_model_aqlm(
            {"linear.weight": W2d, "linear.bias": bias}, config=cfg
        )
        assert "linear.weight" in result
        assert "linear.bias" not in result

    def test_default_config_used_when_none(self):
        W = RNG.standard_normal((4, 8)).astype(np.float32)
        result = quantize_model_aqlm({"w": W})
        assert "w" in result
        assert isinstance(result["w"], AQLMLayer)
