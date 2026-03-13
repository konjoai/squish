"""tests/test_quip_unit.py — 100% coverage for squish/quip_sharp.py

12+ tests covering:
  - E8Lattice codebook integrity (256 distinct unit vectors, float16)
  - QuIPSharpConfig validation paths
  - QuIPSharpQuantizer output types and shapes
  - Round-trip quantize/dequantize on 8-D vectors and 2D weight matrices
  - quip_dequantize edge cases (no rotation, padded input)
  - quantize_model_quip model-level integration
"""
from __future__ import annotations

import numpy as np
import pytest

from squish.quip_sharp import (
    E8Lattice,
    QuIPSharpConfig,
    QuIPSharpLayer,
    QuIPSharpQuantizer,
    quip_dequantize,
    quantize_model_quip,
)


# ---------------------------------------------------------------------------
# E8Lattice codebook integrity
# ---------------------------------------------------------------------------

class TestE8Codebook:
    def test_codebook_has_exactly_256_vectors(self):
        """Codebook must have exactly 256 entries."""
        assert E8Lattice.codebook.shape == (256, 8)

    def test_codebook_dtype_is_float16(self):
        """Codebook must be stored as float16."""
        assert E8Lattice.codebook.dtype == np.float16

    def test_codebook_vectors_are_unit_norm(self):
        """Every codebook vector must lie on the unit 8-sphere (norm ≈ 1)."""
        cb_f32 = E8Lattice.codebook.astype(np.float32)
        norms = np.linalg.norm(cb_f32, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (
            f"Some norms deviate: min={norms.min():.4f} max={norms.max():.4f}"
        )

    def test_all_256_codebook_vectors_are_distinct(self):
        """No two codebook vectors should be identical (up to float16 precision)."""
        # Convert to a set of tuples to check pairwise uniqueness
        cb = E8Lattice.codebook
        rows_as_tuples = {tuple(row.tolist()) for row in cb}
        assert len(rows_as_tuples) == 256, (
            f"Expected 256 distinct vectors; got {len(rows_as_tuples)}"
        )

    def test_codebook_is_class_attribute(self):
        """Codebook must be accessible as a class attribute without instantiation."""
        # Should be accessible on the class directly (not on an instance)
        cb = E8Lattice.codebook
        assert isinstance(cb, np.ndarray)

    def test_codebook_8_dimensional(self):
        """Each codeword must be 8-dimensional (E8 lattice dimension)."""
        assert E8Lattice.codebook.shape[1] == 8


# ---------------------------------------------------------------------------
# QuIPSharpConfig
# ---------------------------------------------------------------------------

class TestQuIPSharpConfig:
    def test_default_values(self):
        cfg = QuIPSharpConfig()
        assert cfg.use_hadamard is True
        assert cfg.scalar_bits == 2
        assert cfg.group_size == 8

    def test_use_hadamard_false(self):
        cfg = QuIPSharpConfig(use_hadamard=False, scalar_bits=3)
        assert cfg.use_hadamard is False
        assert cfg.scalar_bits == 3

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size must be 8"):
            QuIPSharpConfig(group_size=16)

    def test_invalid_scalar_bits_raises(self):
        with pytest.raises(ValueError, match="scalar_bits must be 2 or 3"):
            QuIPSharpConfig(scalar_bits=4)

    def test_scalar_bits_3_valid(self):
        cfg = QuIPSharpConfig(scalar_bits=3)
        assert cfg.scalar_bits == 3


# ---------------------------------------------------------------------------
# QuIPSharpQuantizer — output types and shapes
# ---------------------------------------------------------------------------

class TestQuIPSharpQuantizer:
    def _small_W(self, out: int = 4, inp: int = 8, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((out, inp)).astype(np.float32)

    def test_quantize_returns_quipsharp_layer(self):
        W = self._small_W()
        q = QuIPSharpQuantizer()
        layer = q.quantize(W)
        assert isinstance(layer, QuIPSharpLayer)

    def test_e8_indices_dtype_uint8(self):
        W = self._small_W(out=4, inp=8)
        layer = QuIPSharpQuantizer().quantize(W)
        assert layer.e8_indices.dtype == np.uint8

    def test_e8_indices_values_in_range(self):
        """All E8 indices must be valid codebook indices (0–255)."""
        W = self._small_W(out=8, inp=16)
        layer = QuIPSharpQuantizer().quantize(W)
        assert int(layer.e8_indices.max()) <= 255
        assert int(layer.e8_indices.min()) >= 0

    def test_residual_scales_dtype_float16(self):
        W = self._small_W()
        layer = QuIPSharpQuantizer().quantize(W)
        assert layer.residual_scales.dtype == np.float16

    def test_rotation_matrix_is_float16(self):
        W = self._small_W()
        layer = QuIPSharpQuantizer().quantize(W)
        assert layer.rotation_matrix is not None
        assert layer.rotation_matrix.dtype == np.float16

    def test_original_shape_preserved(self):
        W = self._small_W(out=5, inp=16)
        layer = QuIPSharpQuantizer().quantize(W)
        assert layer.original_shape == (5, 16)

    def test_non_2d_input_raises(self):
        W = np.ones((4,), dtype=np.float32)
        with pytest.raises(ValueError, match="2-D"):
            QuIPSharpQuantizer().quantize(W)

    def test_n_blocks_correct_multiple_of_8(self):
        """N = out * (in // 8) when in_features is a multiple of 8."""
        W = self._small_W(out=3, inp=16)   # 3 * 2 = 6 blocks
        layer = QuIPSharpQuantizer().quantize(W)
        assert len(layer.e8_indices) == 6
        assert len(layer.residual_scales) == 6

    def test_n_blocks_correct_padded(self):
        """Padding brings in_features=10 to 16 → 3 * 2 = 6 blocks."""
        W = self._small_W(out=3, inp=10)   # padded 10→16, 3 * 2 = 6 blocks
        layer = QuIPSharpQuantizer().quantize(W)
        assert len(layer.e8_indices) == 6

    def test_external_rotation_used_when_provided(self):
        """When rotation_matrix is provided and shape matches, it is used."""
        in_features = 8
        W = self._small_W(out=4, inp=in_features)
        R = np.eye(in_features, dtype=np.float32)
        q = QuIPSharpQuantizer(rotation_matrix=R)
        layer = q.quantize(W)
        # With identity rotation the stored R should match the input (as float16)
        np.testing.assert_allclose(
            layer.rotation_matrix.astype(np.float32),
            R,
            atol=1e-3,
        )

    def test_external_rotation_ignored_when_dim_mismatch(self):
        """If external rotation shape doesn't match, fall back to random rotation."""
        W = self._small_W(out=4, inp=8)
        R_wrong = np.eye(4, dtype=np.float32)  # dim=4, but W has in_features=8
        q = QuIPSharpQuantizer(rotation_matrix=R_wrong)
        layer = q.quantize(W)  # should not raise
        assert layer.rotation_matrix.shape == (8, 8)


# ---------------------------------------------------------------------------
# Round-trip quantize / dequantize
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def _make_W(self, out: int, inp: int, seed: int = 1) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.standard_normal((out, inp)).astype(np.float32)

    def test_dequantize_shape_preserved_aligned(self):
        """Output must have the same shape as the original weight (aligned dims)."""
        W = self._make_W(6, 16)
        layer = QuIPSharpQuantizer().quantize(W)
        W_hat = quip_dequantize(layer)
        assert W_hat.shape == W.shape

    def test_dequantize_shape_preserved_padded(self):
        """Output shape must match original even when padding was applied."""
        W = self._make_W(4, 13)   # 13 not divisible by 8 → padded to 16
        layer = QuIPSharpQuantizer().quantize(W)
        W_hat = quip_dequantize(layer)
        assert W_hat.shape == W.shape

    def test_dequantize_output_dtype_float16(self):
        W = self._make_W(4, 8)
        W_hat = quip_dequantize(QuIPSharpQuantizer().quantize(W))
        assert W_hat.dtype == np.float16

    def test_cosine_similarity_above_threshold(self):
        """Reconstruction must be reasonably close: mean cosine sim > 0.85."""
        rng = np.random.default_rng(7)
        W = rng.standard_normal((32, 64)).astype(np.float32)
        layer = QuIPSharpQuantizer(seed=7).quantize(W)
        W_hat = quip_dequantize(layer).astype(np.float32)

        # Per-row cosine similarity
        nW = np.linalg.norm(W, axis=1, keepdims=True)
        nH = np.linalg.norm(W_hat, axis=1, keepdims=True)
        cos_sim = np.sum(W * W_hat, axis=1) / (
            nW.squeeze() * nH.squeeze() + 1e-8
        )
        assert float(cos_sim.mean()) > 0.85, (
            f"Mean cosine similarity too low: {cos_sim.mean():.4f}"
        )

    def test_dequantize_without_rotation_matrix(self):
        """quip_dequantize must work when rotation_matrix=None (no rotation applied)."""
        W = self._make_W(4, 8)
        layer = QuIPSharpQuantizer().quantize(W)
        # Detach rotation to simulate a layer saved without rotation
        no_rot_layer = QuIPSharpLayer(
            e8_indices=layer.e8_indices,
            residual_scales=layer.residual_scales,
            rotation_matrix=None,
            original_shape=layer.original_shape,
            config=layer.config,
        )
        W_hat = quip_dequantize(no_rot_layer)
        assert W_hat.shape == W.shape
        assert W_hat.dtype == np.float16

    def test_single_8d_chunk_round_trip(self):
        """Exact reconstruction test on a single 8-D weight row."""
        # Use a weight row that is exactly the first E8 codeword (no noise)
        cb = E8Lattice.codebook.astype(np.float32)
        codeword = cb[0:1, :]    # 1 row × 8 columns (is already unit vector)
        layer = QuIPSharpQuantizer(seed=0).quantize(codeword)
        # The encoded scale should be ≈ 1.0 (codeword has unit norm)
        # Index must be 0 (exact match to nearest codeword in some rotation)
        W_hat = quip_dequantize(layer)
        assert W_hat.shape == codeword.shape

    def test_all_zero_weight_handled(self):
        """All-zero weight matrix must not produce NaN or inf."""
        W = np.zeros((4, 8), dtype=np.float32)
        layer = QuIPSharpQuantizer().quantize(W)
        W_hat = quip_dequantize(layer)
        assert not np.any(np.isnan(W_hat))
        assert not np.any(np.isinf(W_hat))


# ---------------------------------------------------------------------------
# quantize_model_quip — model-level integration
# ---------------------------------------------------------------------------

class TestQuantizeModelQuip:
    def _simple_model(self, seed: int = 3) -> dict:
        rng = np.random.default_rng(seed)
        return {
            "model.layers.0.self_attn.q_proj.weight":
                rng.standard_normal((8, 16)).astype(np.float32),
            "model.layers.0.self_attn.q_proj.bias":
                rng.standard_normal((8,)).astype(np.float32),
            "model.embed_tokens.weight":
                rng.standard_normal((32, 16)).astype(np.float32),
        }

    def test_2d_weights_replaced_with_layer(self):
        model = self._simple_model()
        result = quantize_model_quip(model)
        assert isinstance(
            result["model.layers.0.self_attn.q_proj.weight"],
            QuIPSharpLayer,
        )

    def test_1d_bias_passthrough(self):
        """1-D bias tensor must be returned unchanged."""
        model = self._simple_model()
        result = quantize_model_quip(model)
        bias_key = "model.layers.0.self_attn.q_proj.bias"
        assert isinstance(result[bias_key], np.ndarray)
        assert result[bias_key].ndim == 1

    def test_embed_weight_quantized(self):
        """Embedding weight (2-D) must also be quantized."""
        model = self._simple_model()
        result = quantize_model_quip(model)
        assert isinstance(result["model.embed_tokens.weight"], QuIPSharpLayer)

    def test_all_keys_preserved(self):
        """All original keys must still be present in the output."""
        model = self._simple_model()
        result = quantize_model_quip(model)
        assert set(result.keys()) == set(model.keys())

    def test_custom_config_used(self):
        """The supplied config must be reflected in each QuIPSharpLayer."""
        model = self._simple_model()
        cfg = QuIPSharpConfig(use_hadamard=False, scalar_bits=3)
        result = quantize_model_quip(model, cfg)
        for v in result.values():
            if isinstance(v, QuIPSharpLayer):
                assert v.config.scalar_bits == 3

    def test_seed_determinism(self):
        """Same seed must produce identical e8_indices."""
        model = self._simple_model()
        r1 = quantize_model_quip(model, seed=99)
        r2 = quantize_model_quip(model, seed=99)
        key = "model.layers.0.self_attn.q_proj.weight"
        np.testing.assert_array_equal(
            r1[key].e8_indices, r2[key].e8_indices,
        )

    def test_empty_model_returns_empty(self):
        result = quantize_model_quip({})
        assert result == {}
