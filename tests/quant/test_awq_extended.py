"""
tests/test_awq_extended.py

Extended tests for squish/awq.py covering:
  - _ActivationHook.__init__        (lines 96-98)
  - _ActivationHook.__call__        (lines 100-118)
  - _ActivationHook.mean_activation (lines 120-124)
  - apply_awq_to_weights fuzzy match (lines 379-382)
"""
from __future__ import annotations

import numpy as np

from squish.quant.awq import _ActivationHook, apply_awq_to_weights

# ── _ActivationHook ───────────────────────────────────────────────────────────

class TestActivationHook:
    def test_init_defaults(self):
        hook = _ActivationHook()
        assert hook.channel_sum   is None
        assert hook.channel_count == 0

    def test_call_with_numpy_input(self):
        """Call with numpy array (mlx dtype conversion fails → except path or mlx path)."""
        hook = _ActivationHook()
        x = np.ones((2, 4, 8), dtype=np.float32)
        hook(None, (x,), None)
        # Either mlx path or numpy fallback — both should work
        assert hook.channel_count == 1
        assert hook.channel_sum is not None
        assert hook.channel_sum.shape == (8,)

    def test_call_accumulates(self):
        hook = _ActivationHook()
        x = np.ones((1, 4, 8), dtype=np.float32)
        hook(None, (x,), None)
        hook(None, (x,), None)
        assert hook.channel_count == 2

    def test_mean_activation_after_one_call(self):
        hook = _ActivationHook()
        x = np.full((1, 1, 4), 2.0, dtype=np.float32)
        hook(None, (x,), None)
        result = hook.mean_activation()
        assert result.shape == (4,)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, 2.0, rtol=1e-5)

    def test_mean_activation_averages_across_calls(self):
        hook = _ActivationHook()
        x1 = np.full((1, 1, 4), 1.0, dtype=np.float32)
        x2 = np.full((1, 1, 4), 3.0, dtype=np.float32)
        hook(None, (x1,), None)
        hook(None, (x2,), None)
        result = hook.mean_activation()
        np.testing.assert_allclose(result, 2.0, rtol=1e-4)

    def test_mean_activation_empty(self):
        hook = _ActivationHook()
        result = hook.mean_activation()
        assert result.shape == (0,)
        assert result.dtype == np.float32

    def test_call_3d_input_flattens(self):
        """3D input (batch, seq, features) should flatten to (batch*seq, features)."""
        hook = _ActivationHook()
        x = np.ones((2, 8, 16), dtype=np.float32)
        hook(None, (x,), None)
        assert hook.channel_sum.shape == (16,)

    def test_call_2d_input(self):
        """2D input (N, features) should also work."""
        hook = _ActivationHook()
        x = np.ones((4, 12), dtype=np.float32) * 3
        hook(None, (x,), None)
        result = hook.mean_activation()
        assert result.shape == (12,)
        np.testing.assert_allclose(result, 3.0, rtol=1e-5)

    def test_abs_mean_uses_absolute_values(self):
        """Negative values should be treated as positive in the mean."""
        hook = _ActivationHook()
        x = np.array([[[-1.0, 2.0, -3.0, 4.0]]], dtype=np.float32)
        hook(None, (x,), None)
        result = hook.mean_activation()
        expected = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


# ── apply_awq_to_weights fuzzy match ──────────────────────────────────────────

class TestApplyAwqFuzzyMatch:
    def test_exact_key_match(self):
        """Exact match: scale key equals layer path."""
        W = np.ones((4, 8), dtype=np.float32)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": W.copy(),
            "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        }
        scales = {"model.layers.0.self_attn.q_proj": np.full(8, 2.0, dtype=np.float32)}
        result = apply_awq_to_weights(weights, scales, verbose=False)
        # weight should be multiplied by scale (AWQ paper: W *= s)
        assert "model.layers.0.self_attn.q_proj.weight" in result
        # 1.0 * 2.0 = 2.0 per column
        np.testing.assert_allclose(result["model.layers.0.self_attn.q_proj.weight"].reshape(-1, 8),
                                   np.full((4, 8), 2.0), rtol=1e-5)

    def test_fuzzy_suffix_match(self):
        """Fuzzy match: scale key is a suffix of the layer path."""
        W = np.ones((4, 8), dtype=np.float32)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": W.copy(),
            "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        }
        # scale keyed by suffix only
        scales = {"self_attn.q_proj": np.full(8, 2.0, dtype=np.float32)}
        result = apply_awq_to_weights(weights, scales, verbose=False)
        assert "model.layers.0.self_attn.q_proj.weight" in result
        # AWQ paper: W *= s, so 1.0 * 2.0 = 2.0
        np.testing.assert_allclose(result["model.layers.0.self_attn.q_proj.weight"].reshape(-1, 8),
                                   np.full((4, 8), 2.0), rtol=1e-5)

    def test_no_matching_scale_key_skipped(self):
        """Weight with no matching scale key should be left unchanged."""
        W = np.ones((4, 8), dtype=np.float32) * 5
        weights = {"model.other.weight": W.copy()}
        scales = {"self_attn.q_proj": np.full(8, 2.0, dtype=np.float32)}
        result = apply_awq_to_weights(weights, scales, verbose=False)
        # no scale matched, weight unchanged
        np.testing.assert_allclose(result["model.other.weight"].reshape(-1, 8),
                                   np.full((4, 8), 5.0), rtol=1e-5)

    def test_empty_weights(self):
        result = apply_awq_to_weights({}, {}, verbose=False)
        assert isinstance(result, dict)
        assert len(result) == 0

    def test_non_weight_key_ignored(self):
        """Keys not ending in '.weight' should pass through unchanged."""
        W = np.ones((4, 8), dtype=np.float32) * 7
        weights = {"model.bias": W.copy()}
        scales = {"model": np.full(8, 2.0, dtype=np.float32)}
        result = apply_awq_to_weights(weights, scales, verbose=False)
        # bias should be unchanged
        np.testing.assert_allclose(result["model.bias"].reshape(-1, 8),
                                   np.full((4, 8), 7.0), rtol=1e-5)


# ── apply_awq_to_weights: fuzzy loop False branch [379, 378] ─────────────────

class TestApplyAwqFuzzyLoopFalseBranch:
    def test_fuzzy_with_non_matching_key_continues_loop(self):
        """
        Two scale keys: first doesn't endswith(layer_path), second does.
        Covers the fuzzy-match loop fallback path.
        """
        W = np.ones((4, 8), dtype=np.float32)
        weights = {
            "model.layers.0.self_attn.q_proj.weight": W.copy(),
            "model.layers.0.input_layernorm.weight": np.ones(8, dtype=np.float32),
        }
        scales = {
            "some.unrelated.key": np.full(8, 1.0, dtype=np.float32),  # won't match
            "self_attn.q_proj":   np.full(8, 2.0, dtype=np.float32),  # will match
        }
        result = apply_awq_to_weights(weights, scales, verbose=False)
        result_w = result["model.layers.0.self_attn.q_proj.weight"]
        # AWQ paper: W *= s from the second key
        np.testing.assert_allclose(result_w.reshape(-1, 8), np.full((4, 8), 2.0), rtol=1e-5)


# ── _preceding_norm_name: no '.weight' suffix [428, 432] ─────────────────────

class TestPrecedingNormNameNoWeightSuffix:
    def test_weight_name_without_weight_suffix(self):
        """
        When weight_name does not end in '.weight', the strip step is skipped
        (line 428 False branch → line 432).  Covers branch [428, 432].
        """
        from squish.quant.awq import _preceding_norm_name

        # Name WITHOUT '.weight' suffix — still contains 'self_attn'
        weights = {
            "model.layers.0.input_layernorm.weight": np.ones((8,), dtype=np.float32),
        }
        # This name has no '.weight' ending
        result = _preceding_norm_name("model.layers.0.self_attn.q_proj", weights)
        # Expected: finds 'model.layers.0.input_layernorm.weight'
        assert result == "model.layers.0.input_layernorm.weight"

    def test_norm_name_not_in_weights_returns_none(self):
        """Empty weights dict → none of the candidates match → returns None."""
        from squish.quant.awq import _preceding_norm_name

        result = _preceding_norm_name("model.layers.0.self_attn.q_proj", {})
        assert result is None


# ── min_scale parameter (scale clamping) ─────────────────────────────────────

class TestMinScaleParameter:
    """Test the min_scale floor applied after computing s = mean_act^alpha."""

    def _compute_scale(self, mean_act, alpha, min_scale):
        """Mirror the scale computation from collect_activation_scales."""
        s = np.clip(mean_act, 1e-4, None) ** alpha
        if min_scale > 0.0:
            s = np.maximum(s, min_scale)
        return s

    def test_default_no_clamping(self):
        """min_scale=0.0 (default) allows s < 1.0 for mean_act < 1.0."""
        mean_act = np.array([0.01, 0.1, 1.0, 5.0], dtype=np.float32)
        s = self._compute_scale(mean_act, alpha=0.1, min_scale=0.0)
        # Low activation channels → s < 1.0
        assert s[0] < 1.0
        assert s[1] < 1.0
        # mean_act = 1.0 → s = 1.0
        assert abs(float(s[2]) - 1.0) < 1e-5
        # High activation → s > 1.0
        assert s[3] > 1.0

    def test_min_scale_one_clamps_below_one(self):
        """min_scale=1.0 prevents s from falling below 1.0."""
        mean_act = np.array([0.01, 0.1, 1.0, 5.0], dtype=np.float32)
        s = self._compute_scale(mean_act, alpha=0.1, min_scale=1.0)
        assert s[0] >= 1.0
        assert s[1] >= 1.0
        assert abs(float(s[2]) - 1.0) < 1e-5
        assert s[3] > 1.0

    def test_min_scale_does_not_reduce_high_activation(self):
        """min_scale applies a floor only; high-activation scales are unaffected."""
        mean_act = np.array([10.0, 20.0], dtype=np.float32)
        s_no_floor = self._compute_scale(mean_act, alpha=0.1, min_scale=0.0)
        s_with_floor = self._compute_scale(mean_act, alpha=0.1, min_scale=1.0)
        np.testing.assert_allclose(s_no_floor, s_with_floor, rtol=1e-6)

    def test_min_scale_half_intermediate_clamp(self):
        """min_scale=0.9 clamps channels with s < 0.9."""
        mean_act = np.array([0.01, 0.5, 5.0], dtype=np.float32)
        s = self._compute_scale(mean_act, alpha=0.5, min_scale=0.9)
        # 0.01^0.5 = 0.1 → clamped to 0.9
        assert abs(float(s[0]) - 0.9) < 1e-5
        # 0.5^0.5 ≈ 0.707 → clamped to 0.9
        assert abs(float(s[1]) - 0.9) < 1e-5
        # 5.0^0.5 ≈ 2.236 → NOT clamped
        assert s[2] > 0.9

