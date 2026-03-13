"""tests/test_vptq_unit.py — 100% coverage for squish/vptq.py"""
import numpy as np
import pytest

from squish.quant.vptq import (
    VPTQCodebook,
    VPTQConfig,
    VPTQLayer,
    VPTQQuantizer,
    decompress_layer,
)

RNG = np.random.default_rng(77)


# ---------------------------------------------------------------------------
# VPTQConfig
# ---------------------------------------------------------------------------

class TestVPTQConfig:
    def test_defaults(self):
        cfg = VPTQConfig()
        assert cfg.n_codebook_entries == 256
        assert cfg.group_size         == 8
        assert cfg.n_residual_entries == 16
        assert cfg.n_fit_iters        == 20
        assert cfg.seed               == 42

    def test_custom(self):
        cfg = VPTQConfig(n_codebook_entries=64, group_size=4, n_residual_entries=0)
        assert cfg.n_codebook_entries == 64
        assert cfg.n_residual_entries == 0

    def test_invalid_n_codebook_entries(self):
        with pytest.raises(ValueError, match="n_codebook_entries"):
            VPTQConfig(n_codebook_entries=1)

    def test_invalid_group_size(self):
        with pytest.raises(ValueError, match="group_size"):
            VPTQConfig(group_size=0)

    def test_invalid_n_residual_entries(self):
        with pytest.raises(ValueError, match="n_residual_entries"):
            VPTQConfig(n_residual_entries=-1)

    def test_invalid_n_fit_iters(self):
        with pytest.raises(ValueError, match="n_fit_iters"):
            VPTQConfig(n_fit_iters=0)


# ---------------------------------------------------------------------------
# VPTQCodebook
# ---------------------------------------------------------------------------

class TestVPTQCodebook:
    def _make(self, gs=4, n_entries=8, iters=5, seed=0):
        return VPTQCodebook(gs, n_entries, iters, seed)

    def test_initial_not_fitted(self):
        cb = self._make()
        assert not cb.is_fitted

    def test_encode_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError, match="fit"):
            cb.encode(np.zeros((3, 4), np.float32))

    def test_decode_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError, match="fit"):
            cb.decode(np.array([0], dtype=np.int64))

    def test_centroids_raises_before_fit(self):
        cb = self._make()
        with pytest.raises(RuntimeError):
            _ = cb.centroids

    def test_fit_sets_is_fitted(self):
        cb   = self._make(gs=4, n_entries=4)
        vecs = RNG.standard_normal((30, 4)).astype(np.float32)
        cb.fit(vecs)
        assert cb.is_fitted

    def test_centroids_shape_after_fit(self):
        cb   = self._make(gs=4, n_entries=8)
        vecs = RNG.standard_normal((40, 4)).astype(np.float32)
        cb.fit(vecs)
        # at most n_entries centroids (could be less if fewer data points)
        assert cb.centroids.shape[1] == 4

    def test_encode_returns_indices(self):
        cb   = self._make(gs=4, n_entries=8)
        vecs = RNG.standard_normal((20, 4)).astype(np.float32)
        cb.fit(vecs)
        idxs = cb.encode(vecs)
        assert idxs.dtype == np.int64
        assert idxs.shape == (20,)

    def test_decode_returns_vectors(self):
        cb   = self._make(gs=4, n_entries=8)
        vecs = RNG.standard_normal((20, 4)).astype(np.float32)
        cb.fit(vecs)
        idxs    = cb.encode(vecs)
        decoded = cb.decode(idxs)
        assert decoded.shape == (20, 4)
        assert decoded.dtype == np.float32

    def test_encode_decode_reduces_mse(self):
        cb   = self._make(gs=4, n_entries=32, iters=20)
        vecs = RNG.standard_normal((200, 4)).astype(np.float32)
        cb.fit(vecs)
        reconstructed = cb.decode(cb.encode(vecs))
        mse_vq   = np.mean((vecs - reconstructed) ** 2)
        mse_zero = np.mean(vecs ** 2)
        assert mse_vq <= mse_zero

    def test_single_vector(self):
        cb   = self._make(gs=4, n_entries=4)
        vecs = RNG.standard_normal((5, 4)).astype(np.float32)
        cb.fit(vecs)
        one_vec = vecs[:1]
        idx = cb.encode(one_vec)
        assert idx.shape == (1,)
        dec = cb.decode(idx)
        assert dec.shape == (1, 4)

    def test_fewer_data_than_k(self):
        """When n_samples < n_entries, k-means++ should still converge."""
        cb   = VPTQCodebook(group_size=4, n_codebook_entries=32, n_fit_iters=5)
        vecs = RNG.standard_normal((5, 4)).astype(np.float32)  # only 5 samples
        cb.fit(vecs)
        assert cb.is_fitted


# ---------------------------------------------------------------------------
# _kmeans (internal — exercised via VPTQCodebook)
# ---------------------------------------------------------------------------

class TestKmeans:
    def test_returns_k_or_fewer_centroids(self):
        from squish.quant.vptq import _kmeans
        X = RNG.standard_normal((10, 3)).astype(np.float32)
        centroids, labels = _kmeans(X, n_clusters=4, n_iters=5, seed=0)
        assert centroids.shape[0] <= 4
        assert labels.shape == (10,)


# ---------------------------------------------------------------------------
# VPTQLayer
# ---------------------------------------------------------------------------

class TestVPTQLayer:
    def _make_layer(self, shape=(8, 16)):
        cfg   = VPTQConfig(n_codebook_entries=16, group_size=4, n_residual_entries=4,
                           n_fit_iters=5, seed=1)
        quant = VPTQQuantizer(cfg)
        W     = RNG.standard_normal(shape).astype(np.float32) * 0.1
        return quant.compress(W), W

    def test_n_groups(self):
        layer, _ = self._make_layer((4, 8))
        assert layer.n_groups > 0

    def test_compressed_bits_positive(self):
        layer, _ = self._make_layer((4, 8))
        assert layer.compressed_bits > 0

    def test_forward_shape(self):
        layer, W = self._make_layer((8, 16))
        x   = RNG.standard_normal((3, 16)).astype(np.float32)
        out = layer.forward(x)
        assert out.shape == (3, 8)

    def test_layer_has_residual_when_configured(self):
        layer, _ = self._make_layer()
        assert layer.residual_cb is not None
        assert layer.residual_indices is not None

    def test_layer_no_residual_when_disabled(self):
        cfg   = VPTQConfig(n_codebook_entries=8, group_size=4, n_residual_entries=0,
                           n_fit_iters=5)
        quant = VPTQQuantizer(cfg)
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        assert layer.residual_cb is None
        assert layer.residual_indices is None


# ---------------------------------------------------------------------------
# decompress_layer (standalone function)
# ---------------------------------------------------------------------------

class TestDecompressLayer:
    def test_output_shape_matches_original(self):
        cfg   = VPTQConfig(n_codebook_entries=8, group_size=4, n_residual_entries=4,
                           n_fit_iters=5, seed=0)
        quant = VPTQQuantizer(cfg)
        W     = RNG.standard_normal((6, 12)).astype(np.float32)
        layer = quant.compress(W)
        W_hat = decompress_layer(layer)
        assert W_hat.shape == W.shape

    def test_residual_reduces_mse(self):
        # Compare with/without residual
        W     = RNG.standard_normal((8, 32)).astype(np.float32) * 0.02

        cfg_no_res = VPTQConfig(n_codebook_entries=32, group_size=4,
                                n_residual_entries=0, n_fit_iters=10, seed=0)
        cfg_res    = VPTQConfig(n_codebook_entries=32, group_size=4,
                                n_residual_entries=16, n_fit_iters=10, seed=0)

        layer_no  = VPTQQuantizer(cfg_no_res).compress(W)
        layer_res = VPTQQuantizer(cfg_res).compress(W)

        mse_no  = np.mean((W - decompress_layer(layer_no))  ** 2)
        mse_res = np.mean((W - decompress_layer(layer_res)) ** 2)
        assert mse_res <= mse_no + 1e-6  # residual should help at minimum equal


# ---------------------------------------------------------------------------
# VPTQQuantizer
# ---------------------------------------------------------------------------

class TestVPTQQuantizer:
    def test_compress_returns_layer(self):
        quant = VPTQQuantizer(VPTQConfig(n_codebook_entries=8, group_size=4,
                                         n_residual_entries=0, n_fit_iters=5))
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        assert isinstance(layer, VPTQLayer)

    def test_decompress_shape(self):
        quant = VPTQQuantizer(VPTQConfig(n_codebook_entries=8, group_size=4,
                                         n_residual_entries=0, n_fit_iters=5))
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        W_hat = quant.decompress(layer)
        assert W_hat.shape == W.shape

    def test_default_config(self):
        quant = VPTQQuantizer()
        assert quant.config.n_codebook_entries == 256

    def test_col_scales_applied(self):
        quant = VPTQQuantizer(VPTQConfig(n_codebook_entries=8, group_size=4,
                                         n_residual_entries=0, n_fit_iters=5))
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        assert layer.col_scales is not None
        assert layer.col_scales.shape == (W.shape[1],)

    def test_compress_large_weight(self):
        quant = VPTQQuantizer(VPTQConfig(n_codebook_entries=16, group_size=8,
                                         n_residual_entries=0, n_fit_iters=5))
        W     = RNG.standard_normal((32, 64)).astype(np.float32) * 0.05
        layer = quant.compress(W)
        W_hat = quant.decompress(layer)
        assert W_hat.shape == (32, 64)


# ---------------------------------------------------------------------------
# Coverage gap tests — exercise branches not reached by normal workflow
# ---------------------------------------------------------------------------

class TestCoverageGaps:
    def test_compressed_bits_no_residual(self):
        """False branch of `if self.residual_cb is not None:` in compressed_bits."""
        cfg   = VPTQConfig(n_codebook_entries=8, group_size=4,
                           n_residual_entries=0, n_fit_iters=5)
        quant = VPTQQuantizer(cfg)
        W     = RNG.standard_normal((4, 8)).astype(np.float32)
        layer = quant.compress(W)
        assert layer.residual_cb is None
        # compressed_bits with residual_cb=None → False branch → 306->308
        bits = layer.compressed_bits
        assert bits > 0

    def test_compress_nondivsible_weight_triggers_padding(self):
        """Weight size not divisible by group_size hits np.pad in _make_groups (line 399)."""
        cfg   = VPTQConfig(n_codebook_entries=8, group_size=4,
                           n_residual_entries=0, n_fit_iters=5)
        quant = VPTQQuantizer(cfg)
        # (3, 7) has size 21, pad = (4 - 21%4) % 4 = 3 → hits line 399
        W     = RNG.standard_normal((3, 7)).astype(np.float32)
        layer = quant.compress(W)
        W_hat = quant.decompress(layer)
        assert W_hat.shape == W.shape

    def test_decompress_pads_when_flat_too_short(self):
        """Manually constructed layer where decoded flat < n_orig hits else pad (line 351)."""
        # Build a tiny codebook with 2 groups of size 4 (decodes to 8 values)
        cb = VPTQCodebook(group_size=4, n_codebook_entries=4, n_fit_iters=5, seed=0)
        vecs = RNG.standard_normal((20, 4)).astype(np.float32)
        cb.fit(vecs)
        indices = np.zeros(2, dtype=np.int64)  # 2 groups → 8 decoded values

        # We set original_shape = (16,) so n_orig=16 > flat.size=8 → pad branch
        layer = VPTQLayer(
            primary_indices  = indices,
            residual_indices = None,
            primary_cb       = cb,
            residual_cb      = None,
            original_shape   = (16,),
            col_scales       = None,
        )
        out = decompress_layer(layer)
        assert out.shape == (16,)

    def test_decompress_no_col_scales(self):
        """col_scales=None hits False branch of `if layer.col_scales is not None:` (356->359)."""
        cb = VPTQCodebook(group_size=4, n_codebook_entries=4, n_fit_iters=5, seed=0)
        vecs = RNG.standard_normal((20, 4)).astype(np.float32)
        cb.fit(vecs)
        indices = np.array([0, 1, 0, 1], dtype=np.int64)  # 4 groups → 16 values

        layer = VPTQLayer(
            primary_indices  = indices,
            residual_indices = None,
            primary_cb       = cb,
            residual_cb      = None,
            original_shape   = (16,),
            col_scales       = None,  # ← False branch: no scaling applied
        )
        out = decompress_layer(layer)
        assert out.shape == (16,)
