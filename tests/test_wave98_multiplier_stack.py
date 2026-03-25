"""tests/test_wave98_multiplier_stack.py — Wave 98: FFN sparsity wiring & EAGLE path fix.

Covers:
- MaskedFFN: wraps any callable mlp, applies binary mask to output
- MaskedFFN.__getattr__: transparently proxies inner module attributes
- MaskedFFN with all-ones mask: output == original
- MaskedFFN with all-zeros mask: output is zero
- patch_model_ffn_sparsity: replaces model.layers[i].mlp with MaskedFFN
- patch_model_ffn_sparsity: skips layers without a mask
- patch_model_ffn_sparsity: idempotent (double-patch is safe)
- patch_model_ffn_sparsity: model.model.layers layout
- unpatch_model_ffn_sparsity: restores original MLPs
- StructuredFfnSparsity integration: from_file → patch → apply
- auto_profile._eagle3_head_found: returns path when pattern present
- auto_profile._model_slug_score: correct token overlap
- auto_profile._detect_eagle3: finds head in ~/.squish/eagle-heads/
- auto_profile._detect_eagle3: adjacent dirs still take priority over slug dirs
- auto_profile._detect_eagle3: no false positive when slug score is 0
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ==============================================================================
# Helpers
# ==============================================================================

def _make_model(n_layers=4, with_mask_layers=(0, 2)):
    """Build a minimal fake MLX-style model for testing."""
    class _FakeMLP:
        def __init__(self, scale: float = 1.0):
            self.scale = scale
            self.weight = np.ones((8, 8), dtype=np.float32)

        def __call__(self, x):
            return x * self.scale

    class _FakeLayer:
        def __init__(self, idx: int):
            self.mlp = _FakeMLP(scale=float(idx + 1))

    class _FakeModel:
        def __init__(self):
            self.layers = [_FakeLayer(i) for i in range(n_layers)]

    return _FakeModel()


def _make_sparsity(n_layers=4, mask_layers=(0, 2), hidden_dim=8, sparsity=0.5):
    """Build a StructuredFfnSparsity with deterministic masks for testing."""
    from squish.runtime.structured_sparsity import StructuredFfnSparsity

    rng = np.random.default_rng(42)
    masks = {}
    for i in mask_layers:
        m = np.ones(hidden_dim, dtype=np.float32)
        n_zero = int(sparsity * hidden_dim)
        zero_idx = rng.choice(hidden_dim, n_zero, replace=False)
        m[zero_idx] = 0.0
        masks[i] = m
    return StructuredFfnSparsity(masks)


# ==============================================================================
# 1.  MaskedFFN
# ==============================================================================

class TestMaskedFFN(unittest.TestCase):

    def _cls(self):
        from squish.kernels.ffn_mask_patch import MaskedFFN
        return MaskedFFN

    def test_all_ones_mask_preserves_output(self):
        MaskedFFN = self._cls()
        inner = lambda x: x * 2  # noqa: E731
        mask = np.ones(8, dtype=np.float32)
        m = MaskedFFN(inner, mask)
        x = np.ones(8, dtype=np.float32)
        out = m(x)
        # mask is all-1 so output should equal inner output (x * 2)
        np.testing.assert_allclose(out, x * 2, rtol=1e-5)

    def test_all_zeros_mask_zeroes_output(self):
        MaskedFFN = self._cls()
        inner = lambda x: x * 99.0  # noqa: E731
        mask = np.zeros(8, dtype=np.float32)
        m = MaskedFFN(inner, mask)
        x = np.ones(8, dtype=np.float32)
        out = m(x)
        np.testing.assert_allclose(np.asarray(out), np.zeros(8), atol=1e-6)

    def test_partial_mask_zeros_correct_neurons(self):
        MaskedFFN = self._cls()
        inner = lambda x: x  # noqa: E731 identity
        mask = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        m = MaskedFFN(inner, mask)
        x = np.ones(8, dtype=np.float32)
        out = np.asarray(m(x))
        expected_zeros = [1, 3, 5, 7]
        for idx in expected_zeros:
            self.assertAlmostEqual(float(out[idx]), 0.0, places=5,
                                   msg=f"Neuron {idx} should be zeroed")
        expected_ones = [0, 2, 4, 6]
        for idx in expected_ones:
            self.assertAlmostEqual(float(out[idx]), 1.0, places=5,
                                   msg=f"Neuron {idx} should be kept")

    def test_repr_shows_layer_and_sparsity(self):
        MaskedFFN = self._cls()
        inner = lambda x: x  # noqa: E731
        mask = np.array([1, 0, 1, 0], dtype=np.float32)
        m = MaskedFFN(inner, mask, layer_idx=5)
        r = repr(m)
        self.assertIn("5", r)
        self.assertIn("50.0%", r)

    def test_getattr_proxies_to_inner(self):
        MaskedFFN = self._cls()

        class _Inner:
            weight = np.eye(4)
            bias = np.zeros(4)

        mask = np.ones(4, dtype=np.float32)
        m = MaskedFFN(_Inner(), mask)
        np.testing.assert_array_equal(m.weight, np.eye(4))
        np.testing.assert_array_equal(m.bias, np.zeros(4))

    def test_boolean_mask_treated_as_binary(self):
        MaskedFFN = self._cls()
        inner = lambda x: x  # noqa: E731
        mask = np.array([True, False, True, True], dtype=bool)
        m = MaskedFFN(inner, mask)
        x = np.ones(4, dtype=np.float32)
        out = np.asarray(m(x))
        self.assertAlmostEqual(float(out[1]), 0.0, places=5)
        self.assertAlmostEqual(float(out[0]), 1.0, places=5)


# ==============================================================================
# 2.  patch_model_ffn_sparsity
# ==============================================================================

class TestPatchModelFfnSparsity(unittest.TestCase):

    def _fn(self):
        from squish.kernels.ffn_mask_patch import patch_model_ffn_sparsity
        return patch_model_ffn_sparsity

    def test_patches_layers_with_masks(self):
        from squish.kernels.ffn_mask_patch import MaskedFFN
        fn = self._fn()
        model = _make_model(n_layers=4)
        sparsity = _make_sparsity(n_layers=4, mask_layers=(0, 2))
        n = fn(model, sparsity, verbose=False)
        self.assertEqual(n, 2)
        self.assertIsInstance(model.layers[0].mlp, MaskedFFN)
        self.assertIsInstance(model.layers[2].mlp, MaskedFFN)

    def test_skips_layers_without_mask(self):
        from squish.kernels.ffn_mask_patch import MaskedFFN
        fn = self._fn()
        model = _make_model(n_layers=4)
        sparsity = _make_sparsity(n_layers=4, mask_layers=(1,))
        fn(model, sparsity, verbose=False)
        # Layer 1 patched, others not
        self.assertIsInstance(model.layers[1].mlp, MaskedFFN)
        self.assertNotIsInstance(model.layers[0].mlp, MaskedFFN)
        self.assertNotIsInstance(model.layers[2].mlp, MaskedFFN)
        self.assertNotIsInstance(model.layers[3].mlp, MaskedFFN)

    def test_idempotent_double_patch(self):
        from squish.kernels.ffn_mask_patch import MaskedFFN
        fn = self._fn()
        model = _make_model(n_layers=2)
        sparsity = _make_sparsity(n_layers=2, mask_layers=(0,))
        fn(model, sparsity, verbose=False)
        original_masked = model.layers[0].mlp
        n2 = fn(model, sparsity, verbose=False)
        # Second call should patch 0 new layers (already patched)
        self.assertEqual(n2, 0)
        self.assertIs(model.layers[0].mlp, original_masked,
                      "Double-patch should not re-wrap the MaskedFFN")

    def test_model_model_layers_layout(self):
        """patch_model_ffn_sparsity should also work with model.model.layers."""
        from squish.kernels.ffn_mask_patch import MaskedFFN, patch_model_ffn_sparsity

        inner_model = _make_model(n_layers=3)
        outer = types.SimpleNamespace(model=inner_model)
        sparsity = _make_sparsity(n_layers=3, mask_layers=(0, 1))
        n = patch_model_ffn_sparsity(outer, sparsity, verbose=False)
        self.assertEqual(n, 2)
        self.assertIsInstance(inner_model.layers[0].mlp, MaskedFFN)

    def test_raises_if_no_layers(self):
        fn = self._fn()
        no_layers = types.SimpleNamespace()
        sparsity = _make_sparsity(n_layers=1, mask_layers=(0,))
        with self.assertRaises(AttributeError):
            fn(no_layers, sparsity, verbose=False)

    def test_returns_zero_when_no_matching_masks(self):
        fn = self._fn()
        model = _make_model(n_layers=2)
        # Masks for layers 10 and 11 — outside model range
        sparsity = _make_sparsity(n_layers=12, mask_layers=(10, 11))
        n = fn(model, sparsity, verbose=False)
        self.assertEqual(n, 0)

    def test_masked_ffn_output_is_correct(self):
        """End-to-end: patched model produces masked output."""
        from squish.kernels.ffn_mask_patch import patch_model_ffn_sparsity

        class _ScaledMLP:
            def __call__(self, x):
                return x * 3.0

        class _Layer:
            def __init__(self):
                self.mlp = _ScaledMLP()

        class _Model:
            def __init__(self):
                self.layers = [_Layer()]

        from squish.runtime.structured_sparsity import StructuredFfnSparsity
        mask = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        sp = StructuredFfnSparsity({0: mask})
        m = _Model()
        patch_model_ffn_sparsity(m, sp, verbose=False)
        x = np.ones(4, dtype=np.float32)
        out = np.asarray(m.layers[0].mlp(x))
        # Output should be (x * 3) * mask = [3., 0., 3., 0.]
        np.testing.assert_allclose(out, [3., 0., 3., 0.], rtol=1e-5)


# ==============================================================================
# 3.  unpatch_model_ffn_sparsity
# ==============================================================================

class TestUnpatchModelFfnSparsity(unittest.TestCase):

    def test_restores_original_mlp(self):
        from squish.kernels.ffn_mask_patch import (
            MaskedFFN, patch_model_ffn_sparsity, unpatch_model_ffn_sparsity,
        )
        model = _make_model(n_layers=3)
        originals = [layer.mlp for layer in model.layers]
        sparsity = _make_sparsity(n_layers=3, mask_layers=(0, 1, 2))
        patch_model_ffn_sparsity(model, sparsity, verbose=False)
        n = unpatch_model_ffn_sparsity(model)
        self.assertEqual(n, 3)
        for i, layer in enumerate(model.layers):
            self.assertIs(layer.mlp, originals[i],
                          f"Layer {i} mlp not restored after unpatch")

    def test_unpatch_no_patched_layers(self):
        from squish.kernels.ffn_mask_patch import unpatch_model_ffn_sparsity
        model = _make_model(n_layers=2)
        n = unpatch_model_ffn_sparsity(model)
        self.assertEqual(n, 0)

    def test_unpatch_model_with_no_layers(self):
        from squish.kernels.ffn_mask_patch import unpatch_model_ffn_sparsity
        m = types.SimpleNamespace()
        n = unpatch_model_ffn_sparsity(m)
        self.assertEqual(n, 0)


# ==============================================================================
# 4.  auto_profile EAGLE-3 head detection helpers
# ==============================================================================

class TestAutoProfileEagleHelpers(unittest.TestCase):

    def _cls(self):
        from squish.runtime.auto_profile import ModelCapabilityDetector
        return ModelCapabilityDetector

    def test_eagle3_head_found_returns_none_for_missing_dir(self):
        MCD = self._cls()
        patterns = ["eagle3_head.safetensors", "eagle3_head", "eagle_head.safetensors"]
        result = MCD._eagle3_head_found(Path("/tmp/__does_not_exist_squish_test__"), patterns)
        self.assertIsNone(result)

    def test_eagle3_head_found_returns_path_for_file(self):
        MCD = self._cls()
        patterns = ["eagle3_head.safetensors", "eagle3_head", "eagle_head.safetensors"]
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "eagle3_head.safetensors"
            f.touch()
            result = MCD._eagle3_head_found(Path(d), patterns)
            self.assertIsNotNone(result)
            # Should point to the directory
            self.assertEqual(Path(result), Path(d))

    def test_eagle3_head_found_returns_dir_path_for_subdirectory(self):
        MCD = self._cls()
        patterns = ["eagle3_head.safetensors", "eagle3_head", "eagle_head.safetensors"]
        with tempfile.TemporaryDirectory() as d:
            sub = Path(d) / "eagle3_head"
            sub.mkdir()
            result = MCD._eagle3_head_found(Path(d), patterns)
            self.assertIsNotNone(result)
            self.assertEqual(Path(result), sub)

    def test_model_slug_score_qwen3_8b(self):
        MCD = self._cls()
        model_path = Path("/Users/test/.squish/models/squish/qwen3/8b")
        score = MCD._model_slug_score(model_path, "eagle3-qwen3-instruct-8b")
        # "qwen3" and "8b" should both match
        self.assertGreaterEqual(score, 2)

    def test_model_slug_score_different_family_lower(self):
        MCD = self._cls()
        llama_path = Path("/Users/test/.squish/models/squish/llama3/8b")
        # qwen3 slug should score lower for llama model
        qwen_score = MCD._model_slug_score(llama_path, "eagle3-qwen3-instruct-8b")
        llama_score = MCD._model_slug_score(llama_path, "eagle3-llama31-instruct-8b")
        # Both contain "8b" but llama slug has more overlap with llama path
        self.assertGreaterEqual(llama_score, qwen_score,
                                "LLaMA slug should score >= Qwen slug for LLaMA model path")

    def test_detect_eagle3_finds_head_in_home_eagle_heads(self):
        """Detect a head saved to ~/.squish/eagle-heads/<slug>/ (pull-head default)."""
        MCD = self._cls()
        from squish.runtime.auto_profile import OptimizationProfile

        with tempfile.TemporaryDirectory() as tmp:
            # Simulate ~/.squish/eagle-heads/eagle3-qwen3-instruct-8b/
            heads_root = Path(tmp) / ".squish" / "eagle-heads"
            slug_dir = heads_root / "eagle3-qwen3-instruct-8b"
            slug_dir.mkdir(parents=True)
            (slug_dir / "eagle3_head.safetensors").touch()

            profile = OptimizationProfile()
            model_path = Path(tmp) / "models" / "qwen3" / "8b"
            model_path.mkdir(parents=True)
            comp_path = Path(tmp) / "models" / "qwen3-compressed"
            comp_path.mkdir(parents=True)

            with patch("squish.runtime.auto_profile.Path.home", return_value=Path(tmp)):
                MCD._detect_eagle3(profile, model_path, comp_path)

            self.assertTrue(profile.use_eagle3, "EAGLE head should be detected in eagle-heads dir")
            self.assertIsNotNone(profile.eagle3_head_dir)

    def test_detect_eagle3_adjacent_dir_takes_priority(self):
        """A head file adjacent to the model takes priority over ~/.squish/eagle-heads/."""
        MCD = self._cls()
        from squish.runtime.auto_profile import OptimizationProfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Adjacent head file (comp_path contains eagle head)
            comp_path = root / "model-compressed"
            comp_path.mkdir()
            (comp_path / "eagle3_head.safetensors").touch()

            # Also put one in heads dir
            heads_root = root / ".squish" / "eagle-heads" / "eagle3-qwen3-8b"
            heads_root.mkdir(parents=True)
            (heads_root / "eagle_head.safetensors").touch()

            model_path = root / "model-raw"
            model_path.mkdir()

            profile = OptimizationProfile()
            with patch("squish.runtime.auto_profile.Path.home", return_value=root):
                MCD._detect_eagle3(profile, model_path, comp_path)

            self.assertTrue(profile.use_eagle3)
            # Should point to comp_path (adjacent), not the heads dir
            self.assertEqual(
                Path(profile.eagle3_head_dir).resolve(),
                comp_path.resolve(),
                "Adjacent dir should take priority over ~/.squish/eagle-heads/",
            )

    def test_detect_eagle3_no_heads_dir_no_crash(self):
        """No ~/.squish/eagle-heads/ → no crash, profile unchanged."""
        MCD = self._cls()
        from squish.runtime.auto_profile import OptimizationProfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_path = root / "model"
            model_path.mkdir()
            comp_path = root / "model-c"
            comp_path.mkdir()

            profile = OptimizationProfile()
            # Home points to empty tmp dir (no .squish/eagle-heads/)
            with patch("squish.runtime.auto_profile.Path.home", return_value=root):
                MCD._detect_eagle3(profile, model_path, comp_path)

            self.assertFalse(profile.use_eagle3)
            self.assertFalse(profile.eagle3_head_dir)  # None or empty string = no head found


# ==============================================================================
# 5.  StructuredFfnSparsity round-trip with ffn_mask_patch
# ==============================================================================

class TestStructuredFfnSparsityIntegration(unittest.TestCase):

    def test_save_load_patch_roundtrip(self):
        """Write sparse_masks.npz, load via StructuredFfnSparsity, patch model."""
        from squish.kernels.ffn_mask_patch import MaskedFFN, patch_model_ffn_sparsity
        from squish.runtime.structured_sparsity import StructuredFfnSparsity

        with tempfile.TemporaryDirectory() as d:
            npz_path = Path(d) / "sparse_masks.npz"
            mask0 = np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype=np.float32)
            mask2 = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=np.float32)
            np.savez(str(npz_path), layer_0=mask0, layer_2=mask2)

            sp = StructuredFfnSparsity.from_file(str(npz_path))
            self.assertEqual(sp.n_layers, 2)
            self.assertTrue(sp.has_mask(0))
            self.assertTrue(sp.has_mask(2))
            self.assertFalse(sp.has_mask(1))

            model = _make_model(n_layers=4)
            n = patch_model_ffn_sparsity(model, sp, verbose=False)
            self.assertEqual(n, 2)
            self.assertIsInstance(model.layers[0].mlp, MaskedFFN)
            self.assertIsInstance(model.layers[2].mlp, MaskedFFN)

    def test_mask_values_match_after_load(self):
        """Masks survive save → load intact."""
        from squish.runtime.structured_sparsity import StructuredFfnSparsity

        with tempfile.TemporaryDirectory() as d:
            npz_path = Path(d) / "sparse_masks.npz"
            original = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
            np.savez(str(npz_path), layer_3=original)
            sp = StructuredFfnSparsity.from_file(str(npz_path))
            loaded = sp._masks[3]
            np.testing.assert_array_equal(loaded, original)


if __name__ == "__main__":
    unittest.main()
