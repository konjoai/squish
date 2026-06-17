"""tests/test_kitty_channel_sensitivity.py

KITTY (arXiv 2511.18643) channel-sensitive INT2 KV quantization.

Five test classes:
  1. Pure utility functions (_channel_sensitivity_scores, _build_sensitive_mask)
  2. _quantize_int2_mixed / _dequantize_int2_mixed round-trip
  3. KVLayerCache integration (eviction + get_full_kv)
  4. HadamardKVCache.calibrate_channel_sensitivity method
  5. mlx-lm version guard (_check_mlx_lm_version)
"""
from __future__ import annotations

import sys
import types
import unittest.mock as mock

import numpy as np
import pytest

from squish.kv.kv_cache import (
    HadamardKVCache,
    KVLayerCache,
    QuantizedKVCache,
    _build_sensitive_mask,
    _channel_sensitivity_scores,
    _dequantize_int2_mixed,
    _dequantize_int2_per_channel,
    _quantize_int2_mixed,
    _quantize_int2_per_channel,
    _quantize_int4_per_channel,
)

RNG = np.random.default_rng(42)

HEAD_DIM = 64   # divisible by 4; standard test dimension


def _sample(n: int = 32, head_dim: int = HEAD_DIM) -> np.ndarray:
    """Random (n, head_dim) float16 activations."""
    return RNG.standard_normal((n, head_dim)).astype(np.float16)


def _outlier_sample(n: int = 32, head_dim: int = HEAD_DIM, hot_channels: int = 4) -> np.ndarray:
    """Activations where ``hot_channels`` dimensions have 10× higher variance."""
    arr = RNG.standard_normal((n, head_dim)).astype(np.float16)
    hot = RNG.integers(0, head_dim, size=hot_channels, endpoint=False)
    arr[:, hot] *= 10.0
    return arr.astype(np.float16), hot


# ---------------------------------------------------------------------------
# 1. Utility functions
# ---------------------------------------------------------------------------

class TestChannelSensitivityScores:
    def test_returns_per_channel_variance(self):
        arr = _sample(64, HEAD_DIM)
        scores = _channel_sensitivity_scores(arr)
        assert scores.shape == (HEAD_DIM,)
        assert scores.dtype == np.float32

    def test_high_variance_channels_rank_higher(self):
        arr, hot = _outlier_sample(64, HEAD_DIM, hot_channels=4)
        scores = _channel_sensitivity_scores(arr)
        ranked = np.argsort(-scores)
        # All 4 hot channels should appear in the top-8 highest-variance dims.
        top8 = set(ranked[:8])
        assert len(top8 & set(hot)) >= 3, "Most outlier channels should rank near the top"

    def test_constant_channel_gets_zero_variance(self):
        arr = _sample(32, HEAD_DIM)
        arr[:, 5] = 0.0
        scores = _channel_sensitivity_scores(arr)
        assert scores[5] == pytest.approx(0.0, abs=1e-6)

    def test_accepts_float32(self):
        arr = _sample().astype(np.float32)
        scores = _channel_sensitivity_scores(arr)
        assert scores.shape == (HEAD_DIM,)


class TestBuildSensitiveMask:
    def test_mask_size_rounded_to_multiple_of_4(self):
        scores = RNG.random(HEAD_DIM).astype(np.float32)
        for frac in (0.05, 0.1, 0.15, 0.25, 0.5):
            mask = _build_sensitive_mask(scores, HEAD_DIM, frac)
            n = int(mask.sum())
            assert n % 4 == 0, f"n_sensitive={n} not divisible by 4 at fraction={frac}"

    def test_at_least_4_sensitive_channels(self):
        scores = RNG.random(HEAD_DIM).astype(np.float32)
        mask = _build_sensitive_mask(scores, HEAD_DIM, fraction=0.01)
        assert mask.sum() >= 4

    def test_at_least_4_insensitive_channels(self):
        scores = RNG.random(HEAD_DIM).astype(np.float32)
        mask = _build_sensitive_mask(scores, HEAD_DIM, fraction=0.99)
        assert (~mask).sum() >= 4

    def test_top_channels_are_selected(self):
        scores = np.zeros(HEAD_DIM, dtype=np.float32)
        scores[:4] = 10.0   # first 4 are clearly most sensitive
        mask = _build_sensitive_mask(scores, HEAD_DIM, fraction=0.1)
        # Channels 0–3 must all be in the sensitive set.
        assert all(mask[:4])

    def test_invalid_fraction_raises(self):
        scores = RNG.random(HEAD_DIM).astype(np.float32)
        with pytest.raises(ValueError, match="fraction"):
            _build_sensitive_mask(scores, HEAD_DIM, fraction=0.0)
        with pytest.raises(ValueError, match="fraction"):
            _build_sensitive_mask(scores, HEAD_DIM, fraction=1.0)

    def test_small_head_dim_raises(self):
        scores = np.ones(4, dtype=np.float32)
        with pytest.raises(ValueError, match="head_dim"):
            _build_sensitive_mask(scores, 4, fraction=0.5)

    def test_output_is_boolean(self):
        scores = RNG.random(HEAD_DIM).astype(np.float32)
        mask = _build_sensitive_mask(scores, HEAD_DIM, fraction=0.1)
        assert mask.dtype == bool
        assert mask.shape == (HEAD_DIM,)


# ---------------------------------------------------------------------------
# 2. Mixed quantize / dequantize round-trip
# ---------------------------------------------------------------------------

class TestQuantizeInt2Mixed:
    def _mask(self, head_dim: int = HEAD_DIM, fraction: float = 0.125) -> np.ndarray:
        scores = RNG.random(head_dim).astype(np.float32)
        return _build_sensitive_mask(scores, head_dim, fraction)

    def test_output_shapes(self):
        n, d = 16, HEAD_DIM
        arr = _sample(n, d)
        mask = self._mask(d)
        p2, s2, p4, s4 = _quantize_int2_mixed(arr, mask)
        n_sens = int(mask.sum())
        n_ins = d - n_sens
        assert p2.shape == (n, n_ins // 4)
        assert s2.shape == (n,)
        assert p4.shape == (n, n_sens // 2)
        assert s4.shape == (n,)

    def test_dtypes(self):
        arr = _sample()
        mask = self._mask()
        p2, s2, p4, s4 = _quantize_int2_mixed(arr, mask)
        assert p2.dtype == np.uint8
        assert s2.dtype == np.float32
        assert p4.dtype == np.uint8
        assert s4.dtype == np.float32

    def test_roundtrip_shape(self):
        n, d = 32, HEAD_DIM
        arr = _sample(n, d)
        mask = self._mask(d)
        p2, s2, p4, s4 = _quantize_int2_mixed(arr, mask)
        rec = _dequantize_int2_mixed(p2, s2, p4, s4, mask, d)
        assert rec.shape == (n, d)
        assert rec.dtype == np.float16

    def test_sensitive_channels_have_lower_error_than_pure_int2(self):
        """INT4 storage for sensitive channels should give lower MSE than INT2."""
        n, d = 64, HEAD_DIM
        arr, hot = _outlier_sample(n, d, hot_channels=4)
        scores = _channel_sensitivity_scores(arr)
        mask = _build_sensitive_mask(scores, d, fraction=0.125)

        # Mixed reconstruction
        p2, s2, p4, s4 = _quantize_int2_mixed(arr, mask)
        rec_mixed = _dequantize_int2_mixed(p2, s2, p4, s4, mask, d)

        # Pure INT2 reconstruction
        p2_pure, s2_pure = _quantize_int2_per_channel(arr)
        rec_pure = _dequantize_int2_per_channel(p2_pure, s2_pure, d)

        # MSE on the hot (sensitive) channels should be lower for mixed
        mse_mixed = float(np.mean((arr[:, mask].astype(np.float32) -
                                   rec_mixed[:, mask].astype(np.float32)) ** 2))
        mse_pure  = float(np.mean((arr[:, mask].astype(np.float32) -
                                   rec_pure[:, mask].astype(np.float32)) ** 2))
        assert mse_mixed < mse_pure, (
            f"Mixed INT2+INT4 MSE ({mse_mixed:.4f}) should be < pure INT2 ({mse_pure:.4f})"
        )

    def test_insensitive_channels_unchanged_by_int2(self):
        """Non-sensitive channels follow the normal INT2 codec."""
        n, d = 16, HEAD_DIM
        arr = _sample(n, d)
        mask = self._mask(d)
        p2_m, s2_m, _, _ = _quantize_int2_mixed(arr, mask)
        p2_r, s2_r = _quantize_int2_per_channel(arr[:, ~mask])
        np.testing.assert_array_equal(p2_m, p2_r)
        np.testing.assert_array_almost_equal(s2_m, s2_r)

    def test_all_zero_input_reconstructs_near_zero(self):
        n, d = 8, HEAD_DIM
        arr = np.zeros((n, d), dtype=np.float16)
        mask = self._mask(d)
        p2, s2, p4, s4 = _quantize_int2_mixed(arr, mask)
        rec = _dequantize_int2_mixed(p2, s2, p4, s4, mask, d)
        assert np.allclose(rec, 0, atol=0.1)

    def test_snr_improvement_on_outlier_activations(self):
        """KITTY headline claim: mixed INT2+INT4 improves SNR over pure INT2."""
        n, d = 128, HEAD_DIM
        arr, _ = _outlier_sample(n, d, hot_channels=8)
        scores = _channel_sensitivity_scores(arr)
        mask = _build_sensitive_mask(scores, d, fraction=0.125)

        p2, s2, p4, s4 = _quantize_int2_mixed(arr, mask)
        rec_mixed = _dequantize_int2_mixed(p2, s2, p4, s4, mask, d)

        p2_pure, s2_pure = _quantize_int2_per_channel(arr)
        rec_pure = _dequantize_int2_per_channel(p2_pure, s2_pure, d)

        arr_f = arr.astype(np.float32)
        signal_power = float(np.mean(arr_f ** 2))
        noise_mixed = float(np.mean((arr_f - rec_mixed.astype(np.float32)) ** 2))
        noise_pure  = float(np.mean((arr_f - rec_pure.astype(np.float32)) ** 2))

        snr_mixed = 10 * np.log10(signal_power / (noise_mixed + 1e-12))
        snr_pure  = 10 * np.log10(signal_power / (noise_pure  + 1e-12))
        assert snr_mixed > snr_pure, (
            f"Mixed SNR ({snr_mixed:.1f} dB) should exceed pure INT2 ({snr_pure:.1f} dB)"
        )


# ---------------------------------------------------------------------------
# 3. KVLayerCache integration
# ---------------------------------------------------------------------------

class TestKVLayerCacheChannelSensitive:
    """Tests that the eviction path and get_full_kv use the mixed codec."""

    def _layer_with_mask(
        self, head_dim: int = HEAD_DIM, fraction: float = 0.125
    ) -> tuple:
        scores = RNG.random(head_dim).astype(np.float32)
        mask = _build_sensitive_mask(scores, head_dim, fraction)
        layer = KVLayerCache(window=4, kv_mode="int2")
        layer._channel_sensitive_mask = mask
        return layer, mask

    def _fill(self, layer: KVLayerCache, n: int = 8, n_heads: int = 2,
              head_dim: int = HEAD_DIM) -> None:
        for _ in range(n):
            k = RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            layer.append(k, v)

    def test_mixed_buffers_populated_after_eviction(self):
        layer, _ = self._layer_with_mask()
        self._fill(layer, n=8)
        assert layer._keys_old_q2 is not None
        assert layer._values_old_q2 is not None
        assert layer._keys_old_s2 is not None
        assert layer._values_old_s2 is not None

    def test_main_buffer_stores_insensitive_channels(self):
        head_dim = HEAD_DIM
        layer, mask = self._layer_with_mask(head_dim)
        self._fill(layer, n=8)
        n_ins = head_dim - int(mask.sum())
        # keys_old_q last dim should correspond to INT2-packed insensitive channels
        assert layer.keys_old_q.shape[-1] == n_ins // 4

    def test_sensitive_buffer_stores_int4_packed_sensitive_channels(self):
        head_dim = HEAD_DIM
        layer, mask = self._layer_with_mask(head_dim)
        self._fill(layer, n=8)
        n_sens = int(mask.sum())
        assert layer._keys_old_q2.shape[-1] == n_sens // 2

    def test_get_full_kv_returns_correct_head_dim(self):
        head_dim, n_heads = HEAD_DIM, 2
        layer, _ = self._layer_with_mask(head_dim)
        self._fill(layer, n=8, n_heads=n_heads, head_dim=head_dim)
        k_out, v_out = layer.get_full_kv()
        assert k_out.shape[-1] == head_dim
        assert v_out.shape[-1] == head_dim

    def test_get_full_kv_float16_output(self):
        layer, _ = self._layer_with_mask()
        self._fill(layer, n=8)
        k_out, v_out = layer.get_full_kv()
        assert k_out.dtype == np.float16
        assert v_out.dtype == np.float16

    def test_reconstruction_finite_values(self):
        layer, _ = self._layer_with_mask()
        self._fill(layer, n=8)
        k_out, v_out = layer.get_full_kv()
        assert np.all(np.isfinite(k_out))
        assert np.all(np.isfinite(v_out))

    def test_memory_bytes_includes_sensitive_buffer(self):
        layer_plain, _ = self._layer_with_mask()
        layer_mixed, _ = self._layer_with_mask()
        # Fill both the same way but only layer_mixed has the KITTY mask
        layer_plain._channel_sensitive_mask = None
        self._fill(layer_plain, n=8)
        self._fill(layer_mixed, n=8)
        # Mixed cache uses slightly more memory (INT4 for sensitive channels)
        assert layer_mixed.memory_bytes > layer_plain.memory_bytes

    def test_reset_clears_sensitive_buffers_keeps_mask(self):
        layer, mask = self._layer_with_mask()
        self._fill(layer, n=8)
        layer.reset()
        assert layer._keys_old_q2 is None
        assert layer._values_old_q2 is None
        # Mask is calibration data; it is preserved across resets.
        np.testing.assert_array_equal(layer._channel_sensitive_mask, mask)

    def test_pure_int2_mode_without_mask_unchanged(self):
        """Without a mask, INT2 eviction follows the existing path (no regression)."""
        head_dim, n_heads = HEAD_DIM, 2
        layer = KVLayerCache(window=4, kv_mode="int2")
        for _ in range(8):
            k = RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            layer.append(k, v)
        assert layer._keys_old_q2 is None
        assert layer.keys_old_q is not None
        k_out, v_out = layer.get_full_kv()
        assert k_out.shape[-1] == head_dim

    def test_n_compressed_incremented_on_mixed_eviction(self):
        layer, _ = self._layer_with_mask()
        self._fill(layer, n=8)  # window=4 → evicts 4 tokens
        assert layer._n_compressed == 4


# ---------------------------------------------------------------------------
# 4. HadamardKVCache.calibrate_channel_sensitivity
# ---------------------------------------------------------------------------

class TestCalibrateChannelSensitivity:
    def _sample_keys(
        self, n: int = 32, n_heads: int = 2, head_dim: int = HEAD_DIM
    ) -> list[np.ndarray]:
        return [
            RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            for _ in range(n)
        ]

    def test_returns_self(self):
        cache = HadamardKVCache(n_layers=2, window=4, mode="int2")
        keys = self._sample_keys()
        result = cache.calibrate_channel_sensitivity(keys, fraction=0.125)
        assert result is cache

    def test_mask_set_on_all_layers(self):
        n_layers = 4
        cache = HadamardKVCache(n_layers=n_layers, window=4, mode="int2")
        keys = self._sample_keys()
        cache.calibrate_channel_sensitivity(keys, fraction=0.125)
        for layer in cache._layers:
            assert layer._channel_sensitive_mask is not None
            assert layer._channel_sensitive_mask.dtype == bool

    def test_mask_consistent_across_layers(self):
        cache = HadamardKVCache(n_layers=3, window=4, mode="int2")
        keys = self._sample_keys()
        cache.calibrate_channel_sensitivity(keys, fraction=0.125)
        mask0 = cache._layers[0]._channel_sensitive_mask
        for layer in cache._layers[1:]:
            np.testing.assert_array_equal(layer._channel_sensitive_mask, mask0)

    def test_mask_shape_matches_head_dim(self):
        head_dim = HEAD_DIM
        cache = HadamardKVCache(n_layers=2, window=4, mode="int2")
        keys = self._sample_keys(head_dim=head_dim)
        cache.calibrate_channel_sensitivity(keys, fraction=0.125)
        assert cache._layers[0]._channel_sensitive_mask.shape == (head_dim,)

    def test_empty_sample_keys_raises(self):
        cache = HadamardKVCache(n_layers=2, window=4, mode="int2")
        with pytest.raises(ValueError, match="empty"):
            cache.calibrate_channel_sensitivity([], fraction=0.1)

    def test_invalid_fraction_raises(self):
        cache = HadamardKVCache(n_layers=2, window=4, mode="int2")
        keys = self._sample_keys()
        with pytest.raises(ValueError, match="fraction"):
            cache.calibrate_channel_sensitivity(keys, fraction=0.0)
        with pytest.raises(ValueError, match="fraction"):
            cache.calibrate_channel_sensitivity(keys, fraction=1.0)

    def test_mismatched_head_dim_raises(self):
        cache = HadamardKVCache(n_layers=2, window=4, mode="int2")
        keys = self._sample_keys(head_dim=HEAD_DIM)
        keys.append(RNG.standard_normal((2, HEAD_DIM + 4)).astype(np.float16))
        with pytest.raises(ValueError, match="head_dim"):
            cache.calibrate_channel_sensitivity(keys, fraction=0.125)

    def test_end_to_end_with_update(self):
        """After calibration, update() should use the mixed codec and reconstruct."""
        n_heads, head_dim = 2, HEAD_DIM
        cache = HadamardKVCache(n_layers=1, window=4, mode="int2", seed=7)
        sample_keys = self._sample_keys(n=16, n_heads=n_heads, head_dim=head_dim)
        cache.calibrate_channel_sensitivity(sample_keys, fraction=0.125)

        # Feed 8 tokens to layer 0 to trigger eviction (2D per-token shape).
        for _ in range(8):
            k = RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            v = RNG.standard_normal((n_heads, head_dim)).astype(np.float16)
            cache.update(0, k, v)

        layer = cache._layers[0]
        assert layer._keys_old_q2 is not None, "KITTY buffer should be populated"
        k_out, v_out = layer.get_full_kv()
        assert k_out.shape[-1] == head_dim
        assert np.all(np.isfinite(k_out))

    def test_calibrate_works_on_int8_mode_without_error(self):
        """For non-int2 mode, calibration stores the mask but eviction ignores it."""
        cache = HadamardKVCache(n_layers=2, window=4, mode="int8")
        keys = self._sample_keys()
        cache.calibrate_channel_sensitivity(keys, fraction=0.125)
        assert cache._layers[0]._channel_sensitive_mask is not None
        # No error — INT8 path just ignores the mask.


# ---------------------------------------------------------------------------
# 5. mlx-lm version guard
# ---------------------------------------------------------------------------

class TestCheckMlxLmVersion:
    def test_warns_on_bad_version(self, capsys):
        from squish.server import _check_mlx_lm_version, _MLX_LM_BAD_VERSION
        meta_mod = types.ModuleType("importlib.metadata")
        meta_mod.version = mock.MagicMock(return_value=_MLX_LM_BAD_VERSION)

        with mock.patch("sys.platform", "darwin"):
            with mock.patch("importlib.metadata.version", return_value=_MLX_LM_BAD_VERSION):
                _check_mlx_lm_version()

        out = capsys.readouterr().out
        assert "0.31.0" in out
        assert "UNSAFE" in out or "yanked" in out.lower() or "unsafe" in out.lower()

    def test_silent_on_safe_version(self, capsys):
        from squish.server import _check_mlx_lm_version
        with mock.patch("sys.platform", "darwin"):
            with mock.patch("importlib.metadata.version", return_value="0.31.1"):
                _check_mlx_lm_version()
        out = capsys.readouterr().out
        assert out == ""

    def test_silent_on_newer_version(self, capsys):
        from squish.server import _check_mlx_lm_version
        with mock.patch("sys.platform", "darwin"):
            with mock.patch("importlib.metadata.version", return_value="0.32.0"):
                _check_mlx_lm_version()
        assert capsys.readouterr().out == ""

    def test_silent_on_linux(self, capsys):
        from squish.server import _check_mlx_lm_version, _MLX_LM_BAD_VERSION
        with mock.patch("sys.platform", "linux"):
            with mock.patch("importlib.metadata.version", return_value=_MLX_LM_BAD_VERSION):
                _check_mlx_lm_version()
        assert capsys.readouterr().out == ""

    def test_silent_when_mlx_lm_not_installed(self, capsys):
        import importlib.metadata as _im

        from squish.server import _check_mlx_lm_version
        # The realistic "not installed" signal is PackageNotFoundError
        # (a subclass of ImportError), which the version probe swallows.
        with mock.patch("sys.platform", "darwin"):
            with mock.patch(
                "importlib.metadata.version",
                side_effect=_im.PackageNotFoundError("mlx-lm"),
            ):
                _check_mlx_lm_version()
        assert capsys.readouterr().out == ""

    def test_bad_version_constant_is_yanked_release(self):
        from squish.server import _MLX_LM_BAD_VERSION
        assert _MLX_LM_BAD_VERSION == "0.31.0"
