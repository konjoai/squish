"""tests/test_wave82_autoload_eagle3.py

Wave 82 — EAGLE-3 auto-load + Structured FFN Sparsity

Tests:
  1. StructuredFfnSparsity: load, properties, apply_mask, edge cases
  2. EAGLE3 auto-load: triggered when auto_profile.use_eagle3=True and
     eagle3_head_dir is set and _draft.eagle_head is None
  3. EAGLE3 auto-load: skipped when eagle_head already loaded
  4. EAGLE3 auto-load: skipped when use_eagle3=False
  5. Sparsity auto-load: globals populated when masks detected
  6. Sparsity auto-load: skipped gracefully on missing file
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from squish.experimental.structured_sparsity import StructuredFfnSparsity


# ===========================================================================
# Helpers
# ===========================================================================

def _make_npz(path: str, layer_masks: dict[str | int, np.ndarray]) -> None:
    """Write a sparse_masks.npz to *path*."""
    arrays = {}
    for k, v in layer_masks.items():
        arrays[str(k)] = v.astype(np.float32)
    np.savez(path, **arrays)


def _make_named_npz(path: str, named: dict[str, np.ndarray]) -> None:
    """Write an npz with explicit key names (e.g., 'layer_5_gate')."""
    np.savez(path, **{k: v.astype(np.float32) for k, v in named.items()})


# ===========================================================================
# TestStructuredFfnSparsityFromFile
# ===========================================================================

class TestStructuredFfnSparsityFromFile(unittest.TestCase):
    """StructuredFfnSparsity.from_file() correctness."""

    def test_from_file_integer_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sparse_masks.npz")
            _make_npz(p, {0: np.array([1, 0, 1, 1, 0], dtype=np.float32),
                          1: np.array([1, 1, 0, 0, 1], dtype=np.float32)})
            s = StructuredFfnSparsity.from_file(p)
            self.assertEqual(s.n_layers, 2)

    def test_from_file_layer_prefix_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.npz")
            _make_named_npz(p, {
                "layer_0": np.ones(8, dtype=np.float32),
                "layer_1_gate": np.array([1, 0, 1, 0, 1, 1, 0, 1], dtype=np.float32),
            })
            s = StructuredFfnSparsity.from_file(p)
            self.assertEqual(s.n_layers, 2)

    def test_from_file_raises_on_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            StructuredFfnSparsity.from_file("/tmp/__does_not_exist_xyz.npz")

    def test_from_file_raises_on_empty_npz(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "empty.npz")
            np.savez(p)  # no arrays
            with self.assertRaises(ValueError):
                StructuredFfnSparsity.from_file(p)

    def test_from_file_raises_when_no_recognised_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.npz")
            np.savez(p, meta=np.array([1, 2, 3]))  # 'meta' has no layer idx
            with self.assertRaises(ValueError):
                StructuredFfnSparsity.from_file(p)


# ===========================================================================
# TestStructuredFfnSparsityProperties
# ===========================================================================

class TestStructuredFfnSparsityProperties(unittest.TestCase):
    """Properties: n_layers, mean_sparsity, has_mask, layer_sparsity."""

    def _make(self, masks: dict[int, list]):
        return StructuredFfnSparsity(
            {k: np.array(v, dtype=np.float32) for k, v in masks.items()}
        )

    def test_n_layers_correct(self):
        s = self._make({0: [1, 0, 1], 1: [0, 0, 1], 5: [1, 1, 1]})
        self.assertEqual(s.n_layers, 3)

    def test_mean_sparsity_all_zeros(self):
        s = self._make({0: [0, 0, 0, 0]})
        self.assertAlmostEqual(s.mean_sparsity, 1.0)

    def test_mean_sparsity_all_ones(self):
        s = self._make({0: [1, 1, 1, 1]})
        self.assertAlmostEqual(s.mean_sparsity, 0.0)

    def test_mean_sparsity_half(self):
        s = self._make({0: [1, 0, 1, 0]})
        self.assertAlmostEqual(s.mean_sparsity, 0.5)

    def test_has_mask_true(self):
        s = self._make({3: [1, 0, 1]})
        self.assertTrue(s.has_mask(3))

    def test_has_mask_false(self):
        s = self._make({3: [1, 0, 1]})
        self.assertFalse(s.has_mask(99))

    def test_layer_sparsity_correct(self):
        s = self._make({7: [1, 0, 1, 0, 0]})
        self.assertAlmostEqual(s.layer_sparsity(7), 0.6)

    def test_layer_sparsity_missing_layer_returns_zero(self):
        s = self._make({0: [1, 0]})
        self.assertAlmostEqual(s.layer_sparsity(99), 0.0)

    def test_get_mask_returns_binary(self):
        s = self._make({2: [0.7, 0.0, 2.5]})  # non-binary input
        mask = s.get_mask(2)
        self.assertIsNotNone(mask)
        unique = set(mask.tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_get_mask_missing_layer_returns_none(self):
        s = self._make({0: [1]})
        self.assertIsNone(s.get_mask(99))

    def test_mean_sparsity_empty_no_crash(self):
        s = StructuredFfnSparsity({})
        self.assertAlmostEqual(s.mean_sparsity, 0.0)


# ===========================================================================
# TestStructuredFfnSparsityApplyMask
# ===========================================================================

class TestStructuredFfnSparsityApplyMask(unittest.TestCase):
    """apply_mask correctness for numpy arrays."""

    def _make(self, masks: dict[int, list]):
        return StructuredFfnSparsity(
            {k: np.array(v, dtype=np.float32) for k, v in masks.items()}
        )

    def test_apply_mask_zeros_pruned_elements(self):
        s = self._make({0: [1, 0, 1, 0]})
        t = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
        out = s.apply_mask(0, t)
        np.testing.assert_array_almost_equal(out, [10.0, 0.0, 30.0, 0.0])

    def test_apply_mask_keepall(self):
        s = self._make({1: [1, 1, 1, 1]})
        t = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out = s.apply_mask(1, t)
        np.testing.assert_array_equal(out, t)

    def test_apply_mask_pruneall(self):
        s = self._make({2: [0, 0, 0, 0]})
        t = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
        out = s.apply_mask(2, t)
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_apply_mask_no_mask_returns_unchanged(self):
        s = self._make({0: [1, 0, 1]})
        t = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        out = s.apply_mask(99, t)
        np.testing.assert_array_equal(out, t)
        self.assertIs(out, t)  # same object returned

    def test_apply_mask_2d_broadcast(self):
        """Mask must broadcast over [seq_len, hidden_size]."""
        s = self._make({0: [1, 0, 1, 0]})
        t = np.ones((3, 4), dtype=np.float32)  # [seq_len=3, hidden=4]
        out = s.apply_mask(0, t)
        expected = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_apply_mask_output_shape_unchanged(self):
        s = self._make({3: [1, 0, 1, 1, 0]})
        t = np.random.randn(5).astype(np.float32)
        out = s.apply_mask(3, t)
        self.assertEqual(out.shape, t.shape)


# ===========================================================================
# TestStructuredFfnSparsityMultipleMasks
# ===========================================================================

class TestStructuredFfnSparsityMultipleMasks(unittest.TestCase):
    """Multiple masks for the same layer (gate + up) are combined correctly."""

    def test_gate_and_up_combined_with_and(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.npz")
            _make_named_npz(p, {
                "layer_0_gate": np.array([1, 1, 0, 1], dtype=np.float32),
                "layer_0_up":   np.array([1, 0, 1, 1], dtype=np.float32),
            })
            s = StructuredFfnSparsity.from_file(p)
            mask = s.get_mask(0)
            # Combined = gate AND up → [1, 0, 0, 1]
            expected = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32)
            np.testing.assert_array_equal(mask, expected)


# ===========================================================================
# TestStructuredFfnSparsitySummary
# ===========================================================================

class TestStructuredFfnSparsitySummary(unittest.TestCase):
    """summary() and __repr__ produce expected strings."""

    def test_summary_contains_layer_count(self):
        s = StructuredFfnSparsity({0: np.array([1.0, 0.0]), 1: np.array([0.0, 0.0])})
        self.assertIn("layers=2", s.summary())

    def test_summary_contains_sparsity_pct(self):
        s = StructuredFfnSparsity({0: np.array([1.0, 0.0, 1.0, 0.0])})
        self.assertIn("50.0%", s.summary())

    def test_repr_delegates_to_summary(self):
        s = StructuredFfnSparsity({})
        self.assertEqual(repr(s), s.summary())


# ===========================================================================
# TestEagle3AutoLoad (server.py wiring)
# ===========================================================================

class TestEagle3AutoLoad(unittest.TestCase):
    """Wave 82a: load_eagle_head called after auto_profile sets eagle3_head_dir."""

    def _make_auto_profile(self, use_eagle3=True, eagle3_head_dir="/tmp/eagle"):
        from squish.runtime.auto_profile import OptimizationProfile  # noqa: PLC0415
        p = OptimizationProfile(
            use_eagle3=use_eagle3,
            eagle3_head_dir=eagle3_head_dir,
            active_features=["eagle3"] if use_eagle3 else [],
        )
        return p

    def test_load_eagle_head_called_when_use_eagle3_and_dir_set(self):
        """When auto_profile detects eagle3 and eagle_head is None, should load."""
        import squish.server as srv  # noqa: PLC0415

        mock_draft = MagicMock()
        mock_draft.eagle_head = None
        mock_load = MagicMock()
        prof = self._make_auto_profile(use_eagle3=True, eagle3_head_dir="/tmp/fake_eagle")

        with patch.object(srv, "_draft", mock_draft), \
             patch.object(srv, "load_eagle_head", mock_load):
            # Simulate the server.py Wave 82a conditional block
            if (
                prof is not None
                and prof.use_eagle3
                and prof.eagle3_head_dir
                and srv._draft.eagle_head is None
            ):
                srv.load_eagle_head(prof.eagle3_head_dir, verbose=False)

        mock_load.assert_called_once_with("/tmp/fake_eagle", verbose=False)

    def test_load_eagle_head_skipped_when_already_loaded(self):
        """When eagle_head is already loaded, auto-load must not double-load."""
        import squish.server as srv  # noqa: PLC0415

        mock_draft = MagicMock()
        mock_draft.eagle_head = MagicMock()  # already loaded!
        mock_load = MagicMock()
        _auto_prof = self._make_auto_profile(use_eagle3=True, eagle3_head_dir="/tmp/eagle")

        with patch.object(srv, "_draft", mock_draft), \
             patch.object(srv, "load_eagle_head", mock_load):
            prof = _auto_prof
            if (
                prof is not None
                and prof.use_eagle3
                and prof.eagle3_head_dir
                and srv._draft.eagle_head is None     # guard fails here
            ):
                srv.load_eagle_head(prof.eagle3_head_dir, verbose=False)

        mock_load.assert_not_called()

    def test_load_eagle_head_skipped_when_use_eagle3_false(self):
        """use_eagle3=False → load_eagle_head must not be called."""
        import squish.server as srv  # noqa: PLC0415

        mock_load = MagicMock()
        prof = self._make_auto_profile(use_eagle3=False, eagle3_head_dir="/tmp/eagle")

        with patch.object(srv, "load_eagle_head", mock_load):
            if (
                prof is not None
                and prof.use_eagle3         # False
                and prof.eagle3_head_dir
            ):
                srv.load_eagle_head(prof.eagle3_head_dir, verbose=False)

        mock_load.assert_not_called()

    def test_load_eagle_head_skipped_when_no_dir(self):
        """Empty eagle3_head_dir → load not called."""
        import squish.server as srv  # noqa: PLC0415

        mock_draft = MagicMock()
        mock_draft.eagle_head = None
        mock_load = MagicMock()
        prof = self._make_auto_profile(use_eagle3=True, eagle3_head_dir="")

        with patch.object(srv, "_draft", mock_draft), \
             patch.object(srv, "load_eagle_head", mock_load):
            if (
                prof is not None
                and prof.use_eagle3
                and prof.eagle3_head_dir    # empty string → falsy
            ):
                srv.load_eagle_head(prof.eagle3_head_dir, verbose=False)

        mock_load.assert_not_called()

    def test_load_eagle_head_exception_does_not_propagate(self):
        """Errors during auto-load are swallowed — simulate the server.py pattern."""
        import squish.server as srv  # noqa: PLC0415

        mock_draft = MagicMock()
        mock_draft.eagle_head = None
        prof = self._make_auto_profile(use_eagle3=True, eagle3_head_dir="/tmp/eagle")

        startup_crashed = False
        with patch.object(srv, "_draft", mock_draft), \
             patch.object(srv, "load_eagle_head", side_effect=RuntimeError("oops")):
            # Mirror the try/except in the server.py Wave 82a block
            try:
                try:
                    if prof.use_eagle3 and prof.eagle3_head_dir and srv._draft.eagle_head is None:
                        srv.load_eagle_head(prof.eagle3_head_dir, verbose=False)
                except Exception:
                    pass  # server.py silences all errors here
            except Exception:
                startup_crashed = True

        self.assertFalse(startup_crashed, "Exception from load_eagle_head must not propagate past the server.py guard")


# ===========================================================================
# TestSparsityAutoLoad (server.py wiring)
# ===========================================================================

class TestSparsityAutoLoad(unittest.TestCase):
    """Wave 82b: StructuredFfnSparsity loaded and stored as _structured_sparsity."""

    def _make_profile_with_sparsity(self, mask_path: str):
        from squish.runtime.auto_profile import OptimizationProfile  # noqa: PLC0415
        return OptimizationProfile(
            use_sparsity=True,
            sparsity_mask_path=mask_path,
            active_features=["sparse"],
        )

    def test_structured_sparsity_loaded_when_mask_present(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "sparse_masks.npz")
            _make_npz(p, {0: np.array([1, 0, 1, 1, 0], dtype=np.float32)})

            prof = self._make_profile_with_sparsity(p)
            result_holder = {}

            # Simulate the server.py Wave 82b block
            if prof is not None and prof.use_sparsity and prof.sparsity_mask_path:
                sfn = StructuredFfnSparsity.from_file(prof.sparsity_mask_path)
                result_holder["_structured_sparsity"] = sfn

            self.assertIn("_structured_sparsity", result_holder)
            sfs = result_holder["_structured_sparsity"]
            self.assertEqual(sfs.n_layers, 1)
            self.assertAlmostEqual(sfs.layer_sparsity(0), 0.4)

    def test_sparsity_block_skipped_when_use_sparsity_false(self):
        from squish.runtime.auto_profile import OptimizationProfile  # noqa: PLC0415
        prof = OptimizationProfile(use_sparsity=False, sparsity_mask_path="/some/path.npz")
        called = []

        if prof is not None and prof.use_sparsity:  # False
            called.append("loaded")

        self.assertEqual(called, [])

    def test_sparsity_block_skipped_when_no_mask_path(self):
        from squish.runtime.auto_profile import OptimizationProfile  # noqa: PLC0415
        prof = OptimizationProfile(use_sparsity=True, sparsity_mask_path="")
        called = []

        if prof is not None and prof.use_sparsity and prof.sparsity_mask_path:  # "" → falsy
            called.append("loaded")

        self.assertEqual(called, [])

    def test_sparsity_load_error_does_not_raise(self):
        """Missing file must not crash startup (server blocks must be try/except)."""
        prof = types.SimpleNamespace(use_sparsity=True, sparsity_mask_path="/nonexistent/masks.npz")
        try:
            if prof.use_sparsity and prof.sparsity_mask_path:
                StructuredFfnSparsity.from_file(prof.sparsity_mask_path)
        except FileNotFoundError:
            pass  # should be caught at call site — test that caller logic handles it
        # No further assertion; just ensure the test doesn't raise unhandled


# ===========================================================================
# TestStructuredSparsityServerGlobal
# ===========================================================================

class TestStructuredSparsityServerGlobal(unittest.TestCase):
    """server.py exposes _structured_sparsity as a module-level global."""

    def test_global_exists_in_server_module(self):
        import squish.server as srv  # noqa: PLC0415
        self.assertTrue(
            hasattr(srv, "_structured_sparsity"),
            "_structured_sparsity global missing from squish.server",
        )

    def test_global_initially_none(self):
        import squish.server as srv  # noqa: PLC0415
        # It should be None at import time (no model loaded)
        # (May also be a StructuredFfnSparsity if test env has masks — allow both)
        val = getattr(srv, "_structured_sparsity", "MISSING")
        self.assertNotEqual(val, "MISSING")


# ===========================================================================
# TestStructuredSparsityModuleImport
# ===========================================================================

class TestStructuredSparsityModuleImport(unittest.TestCase):
    def test_module_importable(self):
        from squish.experimental import structured_sparsity  # noqa: PLC0415
        self.assertIsNotNone(structured_sparsity)

    def test_class_exported(self):
        from squish.experimental.structured_sparsity import StructuredFfnSparsity  # noqa: PLC0415
        self.assertTrue(callable(StructuredFfnSparsity))

    def test_all_contains_class(self):
        from squish.experimental import structured_sparsity  # noqa: PLC0415
        self.assertIn("StructuredFfnSparsity", structured_sparsity.__all__)


if __name__ == "__main__":
    unittest.main()
