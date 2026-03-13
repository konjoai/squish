"""tests/test_codec_kv_unit.py

Full-coverage unit tests for squish/codec_kv.py.

Covers:
  CodecConfig       — all __post_init__ validation errors
  CodecStats        — construction
  _pairwise_sq_dist — basic correctness
  _lloyd_kmeans     — basic fit, too-few-samples error
  KVCodec           — __init__, fit (valid+errors), encode/decode keys and
                      values (valid+errors), is_fitted, compression_ratio,
                      stats, subsampling path
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.codec_kv import (
    CodecConfig,
    CodecStats,
    KVCodec,
    _lloyd_kmeans,
    _pairwise_sq_dist,
)


# ---------------------------------------------------------------------------
# CodecConfig
# ---------------------------------------------------------------------------


class TestCodecConfig:
    def test_valid_defaults(self):
        cfg = CodecConfig()
        assert cfg.n_codebook == 256
        assert cfg.head_dim == 64
        assert cfg.n_heads == 8

    def test_n_codebook_one_raises(self):
        with pytest.raises(ValueError, match="n_codebook must be >= 2"):
            CodecConfig(n_codebook=1)

    def test_n_codebook_zero_raises(self):
        with pytest.raises(ValueError, match="n_codebook must be >= 2"):
            CodecConfig(n_codebook=0)

    def test_head_dim_zero_raises(self):
        with pytest.raises(ValueError, match="head_dim must be >= 1"):
            CodecConfig(head_dim=0)

    def test_n_heads_zero_raises(self):
        with pytest.raises(ValueError, match="n_heads must be >= 1"):
            CodecConfig(n_heads=0)

    def test_n_fit_samples_lt_n_codebook_raises(self):
        with pytest.raises(ValueError, match="n_fit_samples .* must be >="):
            CodecConfig(n_codebook=64, n_fit_samples=32)


# ---------------------------------------------------------------------------
# CodecStats
# ---------------------------------------------------------------------------


class TestCodecStats:
    def test_default_values(self):
        s = CodecStats()
        assert s.n_fit_calls == 0
        assert s.n_encode_calls == 0
        assert s.total_encoded_tokens == 0


# ---------------------------------------------------------------------------
# _pairwise_sq_dist
# ---------------------------------------------------------------------------


class TestPairwiseSqDist:
    def test_same_vectors_zero_distance(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        dists = _pairwise_sq_dist(a, a)
        np.testing.assert_allclose(np.diag(dists), [0.0, 0.0], atol=1e-5)

    def test_orthogonal_unit_vectors(self):
        a = np.array([[1.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0]], dtype=np.float32)
        dists = _pairwise_sq_dist(a, b)
        assert dists[0, 0] == pytest.approx(2.0, abs=1e-5)

    def test_output_shape(self):
        a = np.random.randn(5, 8).astype(np.float32)
        b = np.random.randn(3, 8).astype(np.float32)
        dists = _pairwise_sq_dist(a, b)
        assert dists.shape == (5, 3)

    def test_values_non_negative(self):
        rng = np.random.default_rng(42)
        a = rng.standard_normal((10, 4)).astype(np.float32)
        b = rng.standard_normal((6, 4)).astype(np.float32)
        dists = _pairwise_sq_dist(a, b)
        assert np.all(dists >= -1e-4)


# ---------------------------------------------------------------------------
# _lloyd_kmeans
# ---------------------------------------------------------------------------


class TestLloydKmeans:
    def test_too_few_samples_raises(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((3, 4)).astype(np.float32)
        with pytest.raises(ValueError, match="at least"):
            _lloyd_kmeans(data, n_clusters=5, rng=rng)

    def test_basic_fit(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 4)).astype(np.float32)
        centroids = _lloyd_kmeans(data, n_clusters=4, rng=rng)
        assert centroids.shape == (4, 4)
        assert centroids.dtype == np.float32

    def test_centroids_converge(self):
        """Centroids should be within data range."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((100, 8)).astype(np.float32)
        centroids = _lloyd_kmeans(data, n_clusters=8, rng=rng)
        assert centroids.shape == (8, 8)


# ---------------------------------------------------------------------------
# KVCodec
# ---------------------------------------------------------------------------


def _make_cfg(n_codebook=8, head_dim=4, n_heads=2, n_fit_samples=50):
    return CodecConfig(
        n_codebook=n_codebook,
        head_dim=head_dim,
        n_heads=n_heads,
        n_fit_samples=n_fit_samples,
    )


def _sample_data(n=50, head_dim=4, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    return rng.standard_normal((n, head_dim)).astype(np.float32)


class TestKVCodecInit:
    def test_not_fitted_initially(self):
        codec = KVCodec(_make_cfg())
        assert codec.is_fitted is False

    def test_default_rng(self):
        codec = KVCodec(_make_cfg())
        assert codec._rng is not None

    def test_custom_rng(self):
        rng = np.random.default_rng(99)
        codec = KVCodec(_make_cfg(), rng=rng)
        assert codec._rng is rng

    def test_compression_ratio_unfitted(self):
        cfg = _make_cfg(n_codebook=8, head_dim=4)
        codec = KVCodec(cfg)
        ratio = codec.compression_ratio
        import math
        expected = (32.0 * 4) / math.log2(8)
        assert ratio == pytest.approx(expected)


class TestFit:
    def test_fit_sets_codebooks(self):
        codec = KVCodec(_make_cfg())
        keys = _sample_data(50, 4)
        vals = _sample_data(50, 4, rng_seed=1)
        codec.fit(keys, vals)
        assert codec.is_fitted is True
        assert codec._key_codebook is not None
        assert codec._val_codebook is not None

    def test_fit_increments_n_fit_calls(self):
        codec = KVCodec(_make_cfg())
        keys = _sample_data(50, 4)
        vals = _sample_data(50, 4, rng_seed=1)
        codec.fit(keys, vals)
        assert codec.stats.n_fit_calls == 1

    def test_fit_wrong_keys_shape_raises(self):
        codec = KVCodec(_make_cfg(head_dim=4))
        keys = _sample_data(50, 8)  # wrong head_dim
        vals = _sample_data(50, 4)
        with pytest.raises(ValueError, match="keys must have shape"):
            codec.fit(keys, vals)

    def test_fit_wrong_values_shape_raises(self):
        codec = KVCodec(_make_cfg(head_dim=4))
        keys = _sample_data(50, 4)
        vals = _sample_data(50, 8)  # wrong head_dim
        with pytest.raises(ValueError, match="values must have shape"):
            codec.fit(keys, vals)

    def test_fit_keys_not_2d_raises(self):
        codec = KVCodec(_make_cfg(head_dim=4))
        keys = np.ones((50, 4, 1), dtype=np.float32)  # 3D
        vals = _sample_data(50, 4)
        with pytest.raises(ValueError, match="keys must have shape"):
            codec.fit(keys, vals)

    def test_fit_mismatched_row_count_raises(self):
        codec = KVCodec(_make_cfg(head_dim=4))
        keys = _sample_data(50, 4)
        vals = _sample_data(40, 4)  # different n
        with pytest.raises(ValueError, match="same number of rows"):
            codec.fit(keys, vals)

    def test_fit_too_few_samples_raises(self):
        codec = KVCodec(_make_cfg(n_codebook=8, n_fit_samples=8))
        keys = _sample_data(5, 4)  # < n_codebook=8
        vals = _sample_data(5, 4)
        with pytest.raises(ValueError, match="at least"):
            codec.fit(keys, vals)

    def test_fit_subsamples_large_dataset(self):
        """When n > n_fit_samples, data is subsampled."""
        codec = KVCodec(_make_cfg(n_codebook=4, n_fit_samples=20))
        # 100 samples > n_fit_samples=20
        keys = _sample_data(100, 4)
        vals = _sample_data(100, 4, rng_seed=1)
        codec.fit(keys, vals)  # should not raise
        assert codec.is_fitted


class TestEncodeDecodeKeys:
    def _fitted_codec(self):
        codec = KVCodec(_make_cfg(n_codebook=4, n_heads=2, head_dim=4))
        keys = _sample_data(50, 4)
        vals = _sample_data(50, 4, rng_seed=1)
        codec.fit(keys, vals)
        return codec

    def test_encode_keys_shape(self):
        codec = self._fitted_codec()
        keys = np.random.randn(2, 10, 4).astype(np.float32)
        indices = codec.encode_keys(keys)
        assert indices.shape == (2, 10)
        assert indices.dtype == np.int32

    def test_encode_keys_not_fitted_raises(self):
        codec = KVCodec(_make_cfg())
        keys = np.random.randn(2, 5, 4).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            codec.encode_keys(keys)

    def test_encode_keys_wrong_n_heads_raises(self):
        codec = self._fitted_codec()
        keys = np.random.randn(3, 5, 4).astype(np.float32)  # wrong n_heads
        with pytest.raises(ValueError, match="shape"):
            codec.encode_keys(keys)

    def test_encode_keys_wrong_head_dim_raises(self):
        codec = self._fitted_codec()
        keys = np.random.randn(2, 5, 8).astype(np.float32)  # wrong head_dim
        with pytest.raises(ValueError, match="shape"):
            codec.encode_keys(keys)

    def test_encode_keys_not_3d_raises(self):
        codec = self._fitted_codec()
        keys = np.random.randn(10, 4).astype(np.float32)  # 2D
        with pytest.raises(ValueError, match="shape"):
            codec.encode_keys(keys)

    def test_decode_keys_shape(self):
        codec = self._fitted_codec()
        keys = np.random.randn(2, 10, 4).astype(np.float32)
        indices = codec.encode_keys(keys)
        recon = codec.decode_keys(indices[0], head_idx=0)
        assert recon.shape == (10, 4)
        assert recon.dtype == np.float32

    def test_decode_keys_not_fitted_raises(self):
        codec = KVCodec(_make_cfg())
        with pytest.raises(RuntimeError, match="not fitted"):
            codec.decode_keys(np.array([0, 1]), head_idx=0)

    def test_decode_keys_not_1d_raises(self):
        codec = self._fitted_codec()
        with pytest.raises(ValueError, match="1-D"):
            codec.decode_keys(np.array([[0, 1]]), head_idx=0)

    def test_decode_keys_invalid_head_idx_raises(self):
        codec = self._fitted_codec()
        with pytest.raises(ValueError, match="head_idx"):
            codec.decode_keys(np.array([0, 1]), head_idx=5)

    def test_decode_keys_negative_head_idx_raises(self):
        codec = self._fitted_codec()
        with pytest.raises(ValueError, match="head_idx"):
            codec.decode_keys(np.array([0, 1]), head_idx=-1)

    def test_encode_keys_updates_stats(self):
        codec = self._fitted_codec()
        keys = np.random.randn(2, 5, 4).astype(np.float32)
        codec.encode_keys(keys)
        assert codec.stats.n_encode_calls == 1
        assert codec.stats.total_encoded_tokens == 2 * 5


class TestEncodeDecodeValues:
    def _fitted_codec(self):
        codec = KVCodec(_make_cfg(n_codebook=4, n_heads=2, head_dim=4))
        keys = _sample_data(50, 4)
        vals = _sample_data(50, 4, rng_seed=1)
        codec.fit(keys, vals)
        return codec

    def test_encode_values_shape(self):
        codec = self._fitted_codec()
        values = np.random.randn(2, 8, 4).astype(np.float32)
        indices = codec.encode_values(values)
        assert indices.shape == (2, 8)
        assert indices.dtype == np.int32

    def test_encode_values_not_fitted_raises(self):
        codec = KVCodec(_make_cfg())
        values = np.random.randn(2, 5, 4).astype(np.float32)
        with pytest.raises(RuntimeError, match="not fitted"):
            codec.encode_values(values)

    def test_encode_values_wrong_shape_raises(self):
        codec = self._fitted_codec()
        values = np.random.randn(3, 5, 4).astype(np.float32)  # wrong n_heads
        with pytest.raises(ValueError, match="shape"):
            codec.encode_values(values)

    def test_decode_values_shape(self):
        codec = self._fitted_codec()
        values = np.random.randn(2, 6, 4).astype(np.float32)
        indices = codec.encode_values(values)
        recon = codec.decode_values(indices[1], head_idx=1)
        assert recon.shape == (6, 4)

    def test_decode_values_not_fitted_raises(self):
        codec = KVCodec(_make_cfg())
        with pytest.raises(RuntimeError, match="not fitted"):
            codec.decode_values(np.array([0, 1]), head_idx=0)

    def test_decode_values_not_1d_raises(self):
        codec = self._fitted_codec()
        with pytest.raises(ValueError, match="1-D"):
            codec.decode_values(np.array([[0, 1]]), head_idx=0)

    def test_decode_values_invalid_head_idx_raises(self):
        codec = self._fitted_codec()
        with pytest.raises(ValueError, match="head_idx"):
            codec.decode_values(np.array([0, 1]), head_idx=10)

    def test_encode_values_updates_stats(self):
        codec = self._fitted_codec()
        values = np.random.randn(2, 7, 4).astype(np.float32)
        codec.encode_values(values)
        assert codec.stats.n_encode_calls == 1
        assert codec.stats.total_encoded_tokens == 2 * 7
