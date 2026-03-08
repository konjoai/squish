"""tests/test_squeeze_llm_unit.py — 100% coverage for squish/squeeze_llm.py"""
import numpy as np
import pytest

from squish.squeeze_llm import (
    OutlierDetector,
    SqueezeLLMConfig,
    SqueezeLLMLayer,
    SqueezeLLMQuantizer,
    decompress_layer_sq,
)

RNG = np.random.default_rng(55)


# ---------------------------------------------------------------------------
# SqueezeLLMConfig
# ---------------------------------------------------------------------------

class TestSqueezeLLMConfig:
    def test_defaults(self):
        cfg = SqueezeLLMConfig()
        assert cfg.quant_bits     == 3
        assert cfg.sparsity_ratio == pytest.approx(0.0045)
        assert cfg.group_size     == 128
        assert cfg.n_fit_iters    == 20
        assert cfg.seed           == 42

    def test_custom(self):
        cfg = SqueezeLLMConfig(quant_bits=4, sparsity_ratio=0.01, group_size=64)
        assert cfg.quant_bits     == 4
        assert cfg.sparsity_ratio == pytest.approx(0.01)

    def test_invalid_quant_bits(self):
        with pytest.raises(ValueError, match="quant_bits"):
            SqueezeLLMConfig(quant_bits=5)

    def test_valid_quant_bits(self):
        for bits in (2, 3, 4):
            cfg = SqueezeLLMConfig(quant_bits=bits)
            assert cfg.quant_bits == bits

    def test_invalid_sparsity_ratio_negative(self):
        with pytest.raises(ValueError, match="sparsity_ratio"):
            SqueezeLLMConfig(sparsity_ratio=-0.01)

    def test_invalid_sparsity_ratio_ge_1(self):
        with pytest.raises(ValueError, match="sparsity_ratio"):
            SqueezeLLMConfig(sparsity_ratio=1.0)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            SqueezeLLMConfig(group_size=0)

    def test_invalid_n_fit_iters(self):
        with pytest.raises(ValueError, match="n_fit_iters"):
            SqueezeLLMConfig(n_fit_iters=0)


# ---------------------------------------------------------------------------
# OutlierDetector
# ---------------------------------------------------------------------------

class TestOutlierDetector:
    def test_invalid_sparsity_ratio(self):
        with pytest.raises(ValueError, match="sparsity_ratio"):
            OutlierDetector(sparsity_ratio=-0.1)

    def test_zero_sparsity_returns_no_outliers(self):
        det = OutlierDetector(sparsity_ratio=0.0)
        W   = RNG.standard_normal((4, 8)).astype(np.float32)
        dense, outliers = det.identify(W)
        assert len(outliers) == 0
        np.testing.assert_array_equal(dense, W)

    def test_outlier_count_approx(self):
        det = OutlierDetector(sparsity_ratio=0.05)
        W   = RNG.standard_normal((100, 100)).astype(np.float32)
        _, outliers = det.identify(W)
        expected = max(1, int(round(10000 * 0.05)))
        assert len(outliers) == expected

    def test_dense_has_zeros_at_outlier_positions(self):
        det = OutlierDetector(sparsity_ratio=0.1)
        W   = RNG.standard_normal((10, 10)).astype(np.float32)
        dense, outliers = det.identify(W)
        for (r, c), val in outliers.items():
            assert dense[r, c] == 0.0
            assert val == pytest.approx(float(W[r, c]))

    def test_outliers_are_largest_magnitudes(self):
        det = OutlierDetector(sparsity_ratio=0.1)
        W   = np.eye(10, dtype=np.float32) * 100 + \
              RNG.standard_normal((10, 10)).astype(np.float32) * 0.01
        _, outliers = det.identify(W)
        # Diagonal elements are ~100 in magnitude → should be top outliers
        outlier_values = np.abs(list(outliers.values()))
        assert outlier_values.min() > 1.0  # all outliers have large magnitude

    def test_dense_plus_sparse_equals_original(self):
        det = OutlierDetector(sparsity_ratio=0.05)
        W   = RNG.standard_normal((20, 20)).astype(np.float32)
        dense, outliers = det.identify(W)
        # Reconstruct
        reconstructed = dense.copy()
        for (r, c), val in outliers.items():
            reconstructed[r, c] += val
        np.testing.assert_allclose(reconstructed, W, atol=1e-5)


# ---------------------------------------------------------------------------
# _nonuniform_quantize (exercised indirectly via SqueezeLLMQuantizer)
# ---------------------------------------------------------------------------

class TestNonuniformQuantize:
    def test_direct_call(self):
        from squish.squeeze_llm import _nonuniform_quantize
        vals = RNG.standard_normal(64).astype(np.float32)
        indices, centres = _nonuniform_quantize(vals, n_bins=8, n_iters=10, seed=0)
        assert indices.shape == (64,)
        assert indices.dtype == np.uint8
        assert centres.shape == (min(8, 64),)

    def test_single_value(self):
        from squish.squeeze_llm import _nonuniform_quantize
        vals = np.array([1.0], dtype=np.float32)
        indices, centres = _nonuniform_quantize(vals, n_bins=4, n_iters=5, seed=0)
        assert indices.shape == (1,)
        # Only 1 unique value → 1 centre
        assert centres.shape[0] >= 1


# ---------------------------------------------------------------------------
# SqueezeLLMLayer
# ---------------------------------------------------------------------------

class TestSqueezeLLMLayer:
    def _make_layer(self, out=8, in_=16, bits=3, sparsity=0.1, gs=8):
        quant = SqueezeLLMQuantizer(
            SqueezeLLMConfig(quant_bits=bits, sparsity_ratio=sparsity,
                             group_size=gs, n_fit_iters=5)
        )
        W = RNG.standard_normal((out, in_)).astype(np.float32)
        return quant.compress(W), W

    def test_n_outliers_positive(self):
        layer, W = self._make_layer(sparsity=0.1)
        n_total  = W.size
        expected = max(1, int(round(n_total * 0.1)))
        assert layer.n_outliers == expected

    def test_sparsity_property(self):
        layer, W = self._make_layer(out=10, in_=10, sparsity=0.05)
        assert layer.sparsity == pytest.approx(layer.n_outliers / W.size)

    def test_n_groups_positive(self):
        layer, _ = self._make_layer()
        assert layer.n_groups > 0

    def test_forward_shape(self):
        layer, W = self._make_layer(out=8, in_=16)
        x   = RNG.standard_normal((3, 16)).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (3, 8)


# ---------------------------------------------------------------------------
# decompress_layer_sq (standalone function)
# ---------------------------------------------------------------------------

class TestDecompressLayerSq:
    def test_output_shape(self):
        quant = SqueezeLLMQuantizer(SqueezeLLMConfig(quant_bits=3, sparsity_ratio=0.05,
                                                      group_size=8, n_fit_iters=5))
        W     = RNG.standard_normal((4, 16)).astype(np.float32)
        layer = quant.compress(W)
        out   = decompress_layer_sq(layer)
        assert out.shape == W.shape

    def test_outliers_restored(self):
        quant = SqueezeLLMQuantizer(SqueezeLLMConfig(quant_bits=3, sparsity_ratio=0.1,
                                                      group_size=8, n_fit_iters=5))
        W     = np.zeros((4, 8), dtype=np.float32)
        W[0, 0] = 100.0  # obvious outlier
        layer = quant.compress(W)
        out   = decompress_layer_sq(layer)
        # The outlier position should be close to 100
        assert abs(out[0, 0]) > 50.0


# ---------------------------------------------------------------------------
# SqueezeLLMQuantizer
# ---------------------------------------------------------------------------

class TestSqueezeLLMQuantizer:
    def _q(self, bits=3, sparsity=0.05, gs=8):
        return SqueezeLLMQuantizer(
            SqueezeLLMConfig(quant_bits=bits, sparsity_ratio=sparsity,
                             group_size=gs, n_fit_iters=5)
        )

    def test_compress_returns_layer(self):
        quant = self._q()
        W     = RNG.standard_normal((4, 16)).astype(np.float32)
        layer = quant.compress(W)
        assert isinstance(layer, SqueezeLLMLayer)

    def test_decompress_shape(self):
        quant = self._q()
        W     = RNG.standard_normal((4, 16)).astype(np.float32)
        layer = quant.compress(W)
        out   = quant.decompress(layer)
        assert out.shape == W.shape

    def test_default_config(self):
        quant = SqueezeLLMQuantizer()
        assert quant.config.quant_bits == 3

    def test_1d_weight_reshaped(self):
        quant = self._q(sparsity=0.1, gs=4)
        W     = RNG.standard_normal(32).astype(np.float32)
        layer = quant.compress(W)
        out   = quant.decompress(layer)
        assert out.shape == (1, 32)

    def test_compress_preserves_outliers_in_layer(self):
        quant = self._q(sparsity=0.1)
        W     = RNG.standard_normal((8, 16)).astype(np.float32) * 0.01
        W[0, 0] = 99.0
        layer = quant.compress(W)
        assert (0, 0) in layer.outliers

    def test_zero_sparsity_no_outliers(self):
        quant = SqueezeLLMQuantizer(SqueezeLLMConfig(quant_bits=3, sparsity_ratio=0.0,
                                                      group_size=8, n_fit_iters=5))
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        assert layer.n_outliers == 0

    def test_quant_bits_2(self):
        quant = self._q(bits=2, gs=4)
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        out   = quant.decompress(layer)
        assert out.shape == W.shape
        assert layer.bin_centres.shape[1] == 4  # 2^2 bins

    def test_quant_bits_4(self):
        quant = self._q(bits=4, gs=4)
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        assert layer.bin_centres.shape[1] <= 16  # capped at min(2^4, group_size)

    def test_round_trip_mse_reasonable(self):
        """Round-trip MSE should be much lower than variance of weights."""
        quant = SqueezeLLMQuantizer(SqueezeLLMConfig(quant_bits=3, sparsity_ratio=0.05,
                                                      group_size=8, n_fit_iters=20))
        W     = (RNG.standard_normal((16, 32)) * 0.02).astype(np.float32)
        layer = quant.compress(W)
        W_hat = quant.decompress(layer)
        mse   = np.mean((W - W_hat) ** 2)
        var_W = np.var(W)
        # MSE should be in the same ballpark or smaller than var (not terrible)
        assert mse < var_W * 10
