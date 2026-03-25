"""tests/test_wave79_auto_profile.py

Wave 79 — Automatic optimization profiling

Tests for:
  - OptimizationProfile default values and status_line()
  - ModelCapabilityDetector.detect(): hardware detection pass
  - ModelCapabilityDetector.detect(): MoE detection from config.json
  - ModelCapabilityDetector.detect(): EAGLE-3 head detection from file presence
  - ModelCapabilityDetector.detect(): sparsity mask detection
  - OptimizationProfile.apply_defaults(): only overrides unset args fields
  - Compact single-line startup status (no verbose table when auto-profile active)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from squish.runtime.auto_profile import ModelCapabilityDetector, OptimizationProfile


# ============================================================================
# TestOptimizationProfileDefaults
# ============================================================================

class TestOptimizationProfileDefaults(unittest.TestCase):
    """OptimizationProfile must have safe defaults with no detection applied."""

    def test_default_kernel_path(self):
        p = OptimizationProfile()
        self.assertEqual(p.kernel_path, "fused_int4")

    def test_default_kv_mode(self):
        p = OptimizationProfile()
        self.assertEqual(p.kv_mode, "fp16")

    def test_default_use_eagle3_false(self):
        p = OptimizationProfile()
        self.assertFalse(p.use_eagle3)

    def test_default_use_sparsity_false(self):
        p = OptimizationProfile()
        self.assertFalse(p.use_sparsity)

    def test_default_use_moe_lazy_false(self):
        p = OptimizationProfile()
        self.assertFalse(p.use_moe_lazy)

    def test_default_chunk_prefill_size(self):
        p = OptimizationProfile()
        self.assertEqual(p.chunk_prefill_size, 512)

    def test_default_metal_cache_mb(self):
        p = OptimizationProfile()
        self.assertEqual(p.metal_cache_mb, 256)

    def test_active_features_initially_empty(self):
        p = OptimizationProfile()
        self.assertEqual(p.active_features, [])


# ============================================================================
# TestStatusLine
# ============================================================================

class TestStatusLine(unittest.TestCase):
    """status_line must produce a readable single-line summary."""

    def test_status_line_contains_model_name(self):
        p = OptimizationProfile()
        line = p.status_line("Qwen3-8B-int2", 2.5)
        self.assertIn("Qwen3-8B-int2", line)

    def test_status_line_contains_load_time(self):
        p = OptimizationProfile()
        line = p.status_line("model", 3.14)
        self.assertIn("3.1s", line)

    def test_status_line_contains_squish_prefix(self):
        p = OptimizationProfile()
        line = p.status_line("model", 1.0)
        self.assertTrue(line.startswith("squish"))

    def test_status_line_features_in_brackets(self):
        p = OptimizationProfile(active_features=["lut_int2", "eagle3"])
        line = p.status_line("model", 1.0)
        self.assertIn("[", line)
        self.assertIn("lut_int2", line)
        self.assertIn("eagle3", line)

    def test_status_line_no_brackets_when_no_features(self):
        p = OptimizationProfile(active_features=[])
        line = p.status_line("model", 1.0)
        self.assertNotIn("[", line)

    def test_status_line_is_single_line(self):
        p = OptimizationProfile(active_features=["lut_int2", "sparse", "moe-lazy"])
        line = p.status_line("MyModel", 4.0)
        self.assertNotIn("\n", line)


# ============================================================================
# TestApplyDefaults
# ============================================================================

class TestApplyDefaults(unittest.TestCase):
    """apply_defaults must only override fields at their argparse defaults."""

    def _make_args(self, **overrides):
        """Return a simple Namespace-like object with default field values."""
        args = types.SimpleNamespace(
            chunk_prefill_size=512,
            agent_kv=False,
            eagle_head_dir="",
            _blazing_metal_cache_mb=256,
        )
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    def test_chunk_prefill_size_updated_when_at_default(self):
        p = OptimizationProfile(chunk_prefill_size=128)
        args = self._make_args()
        p.apply_defaults(args)
        self.assertEqual(args.chunk_prefill_size, 128)

    def test_chunk_prefill_size_not_overridden_when_explicit(self):
        p = OptimizationProfile(chunk_prefill_size=128)
        args = self._make_args(chunk_prefill_size=256)   # explicit value
        p.apply_defaults(args)
        self.assertEqual(args.chunk_prefill_size, 256)   # unchanged

    def test_metal_cache_mb_updated_when_at_default(self):
        p = OptimizationProfile(metal_cache_mb=64)
        args = self._make_args()
        p.apply_defaults(args)
        self.assertEqual(args._blazing_metal_cache_mb, 64)

    def test_metal_cache_mb_not_overridden_when_explicit(self):
        p = OptimizationProfile(metal_cache_mb=64)
        args = self._make_args(_blazing_metal_cache_mb=128)
        p.apply_defaults(args)
        self.assertEqual(args._blazing_metal_cache_mb, 128)

    def test_eagle3_head_dir_set_when_use_eagle3(self):
        p = OptimizationProfile(use_eagle3=True, eagle3_head_dir="/models/eagle3")
        args = self._make_args()
        p.apply_defaults(args)
        self.assertEqual(args.eagle_head_dir, "/models/eagle3")

    def test_eagle3_head_dir_not_overridden_when_explicit(self):
        p = OptimizationProfile(use_eagle3=True, eagle3_head_dir="/models/eagle3")
        args = self._make_args(eagle_head_dir="/user/custom-eagle")
        p.apply_defaults(args)
        self.assertEqual(args.eagle_head_dir, "/user/custom-eagle")

    def test_agent_kv_enabled_on_int2_kv_mode(self):
        p = OptimizationProfile(kv_mode="int2")
        args = self._make_args()
        p.apply_defaults(args)
        self.assertTrue(args.agent_kv)

    def test_agent_kv_not_overridden_when_already_set(self):
        """Already-True agent_kv should not be touched (would be a no-op anyway)."""
        p = OptimizationProfile(kv_mode="fp16")
        args = self._make_args(agent_kv=True)
        p.apply_defaults(args)
        self.assertTrue(args.agent_kv)   # still True


# ============================================================================
# TestHardwareDetection
# ============================================================================

class TestHardwareDetection(unittest.TestCase):
    """ModelCapabilityDetector must produce correct settings from chip profiles."""

    def _make_chip_profile(self, generation_val=3, rec_kv_bits=4, rec_model_bits=2,
                           rec_chunk_ttft=128):
        """Build a minimal ChipProfile-like object."""
        try:
            from squish.hardware.chip_detector import (
                AppleChipGeneration,
                ChipProfile,
            )
            gen = AppleChipGeneration(generation_val)
            return ChipProfile(
                generation=gen,
                memory_bandwidth_gbps=100.0,
                neural_engine_tops=18.0,
                max_memory_gb=16,
                recommended_chunk_prefill=1024,
                recommended_kv_bits=rec_kv_bits,
                recommended_model_bits=rec_model_bits,
                recommended_chunk_prefill_ttft=rec_chunk_ttft,
            )
        except ImportError:
            self.skipTest("squish.hardware.chip_detector not available")

    def test_m3_16gb_selects_lut_int2_kernel(self):
        chip = self._make_chip_profile(generation_val=3, rec_model_bits=2)
        det = ModelCapabilityDetector()
        profile = det.detect(model_dir="", chip_profile=chip, ram_gb=16.0)
        self.assertEqual(profile.kernel_path, "lut_int2")

    def test_m3_24gb_selects_fused_int3_kernel(self):
        chip = self._make_chip_profile(
            generation_val=3, rec_model_bits=3, rec_kv_bits=4
        )
        det = ModelCapabilityDetector()
        profile = det.detect(model_dir="", chip_profile=chip, ram_gb=24.0)
        self.assertEqual(profile.kernel_path, "fused_int3")

    def test_m3_kv_mode_int4_on_tight_ram(self):
        chip = self._make_chip_profile(generation_val=3, rec_kv_bits=4)
        det = ModelCapabilityDetector()
        profile = det.detect(model_dir="", chip_profile=chip, ram_gb=16.0)
        self.assertEqual(profile.kv_mode, "int4")

    def test_m3_chunk_prefill_from_chip(self):
        chip = self._make_chip_profile(generation_val=3, rec_chunk_ttft=128)
        det = ModelCapabilityDetector()
        profile = det.detect(model_dir="", chip_profile=chip, ram_gb=16.0)
        self.assertEqual(profile.chunk_prefill_size, 128)

    def test_none_chip_profile_uses_safe_defaults(self):
        det = ModelCapabilityDetector()
        profile = det.detect(model_dir="", chip_profile=None, ram_gb=0.0)
        # Defaults should be safe (non-None / non-empty)
        self.assertIsInstance(profile.kernel_path, str)
        self.assertGreater(profile.chunk_prefill_size, 0)


# ============================================================================
# TestMoEDetection
# ============================================================================

class TestMoEDetection(unittest.TestCase):
    """ModelCapabilityDetector must detect MoE architectures from config.json."""

    def _write_config(self, tmp_dir: str, arch: str, extra: dict = None) -> None:
        config = {"architectures": [arch]}
        if extra:
            config.update(extra)
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config, f)

    def test_qwen3_moe_detected(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, "Qwen3MoeForCausalLM")
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertTrue(profile.use_moe_lazy)

    def test_mixtral_moe_detected(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, "MixtralForCausalLM")
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertTrue(profile.use_moe_lazy)

    def test_deepseek_v2_moe_detected(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, "DeepseekV2ForCausalLM")
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertTrue(profile.use_moe_lazy)

    def test_num_experts_field_triggers_moe(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, "SomeCustomArch", extra={"num_experts": 8})
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertTrue(profile.use_moe_lazy)

    def test_dense_qwen3_not_detected_as_moe(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, "Qwen3ForCausalLM")
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertFalse(profile.use_moe_lazy)

    def test_missing_config_json_no_crash(self):
        with tempfile.TemporaryDirectory() as d:
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            # Just check it returns a profile without raising
            self.assertIsInstance(profile, OptimizationProfile)

    def test_moe_feature_in_active_list(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_config(d, "Qwen3MoeForCausalLM")
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertIn("moe-lazy", profile.active_features)


# ============================================================================
# TestEagle3Detection
# ============================================================================

class TestEagle3Detection(unittest.TestCase):
    """ModelCapabilityDetector detects EAGLE-3 draft head files."""

    def test_eagle3_head_safetensors_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "eagle3_head.safetensors").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertTrue(profile.use_eagle3)

    def test_eagle3_head_dir_correct(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "eagle3_head.safetensors").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertEqual(profile.eagle3_head_dir, d)

    def test_eagle_head_safetensors_fallback(self):
        """eagle_head.safetensors (without '3') is also detected."""
        with tempfile.TemporaryDirectory() as d:
            Path(d, "eagle_head.safetensors").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertTrue(profile.use_eagle3)

    def test_eagle3_detection_in_compressed_dir(self):
        with tempfile.TemporaryDirectory() as d_model:
            comp = Path(d_model).parent / (Path(d_model).name + "-compressed")
            comp.mkdir(exist_ok=True)
            (comp / "eagle3_head.safetensors").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d_model, compressed_dir=str(comp))
            self.assertTrue(profile.use_eagle3)
            comp.joinpath("eagle3_head.safetensors").unlink()
            comp.rmdir()

    def test_no_eagle3_when_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertFalse(profile.use_eagle3)

    def test_eagle3_feature_in_active_list(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "eagle3_head.safetensors").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertIn("eagle3", profile.active_features)


# ============================================================================
# TestSparsityDetection
# ============================================================================

class TestSparsityDetection(unittest.TestCase):
    """ModelCapabilityDetector detects FFN sparsity mask files."""

    def test_sparse_masks_npz_detected(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "sparse_masks.npz").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d, compressed_dir=d)
            self.assertTrue(profile.use_sparsity)

    def test_sparsity_mask_path_set(self):
        with tempfile.TemporaryDirectory() as d:
            mask = Path(d, "sparse_masks.npz")
            mask.touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d, compressed_dir=d)
            self.assertEqual(profile.sparsity_mask_path, str(mask))

    def test_no_sparsity_when_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d, compressed_dir=d)
            self.assertFalse(profile.use_sparsity)

    def test_sparse_feature_in_active_list(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "sparse_masks.npz").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d, compressed_dir=d)
            self.assertIn("sparse", profile.active_features)


# ============================================================================
# TestBuildFeatureList
# ============================================================================

class TestBuildFeatureList(unittest.TestCase):
    """active_features list is built correctly from profile state."""

    def test_kernel_path_always_in_features(self):
        with tempfile.TemporaryDirectory() as d:
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d)
            self.assertIn(profile.kernel_path, profile.active_features)

    def test_no_duplicate_features(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "eagle3_head.safetensors").touch()
            Path(d, "sparse_masks.npz").touch()
            det = ModelCapabilityDetector()
            profile = det.detect(model_dir=d, compressed_dir=d)
            self.assertEqual(len(profile.active_features), len(set(profile.active_features)))

    def test_kv_int4_appears_in_features(self):
        try:
            from squish.hardware.chip_detector import AppleChipGeneration, ChipProfile
        except ImportError:
            self.skipTest("chip_detector not available")
        chip = ChipProfile(
            generation=AppleChipGeneration.M3,
            memory_bandwidth_gbps=100.0,
            neural_engine_tops=18.0,
            max_memory_gb=16,
            recommended_chunk_prefill=1024,
            recommended_kv_bits=4,
            recommended_model_bits=2,
            recommended_chunk_prefill_ttft=128,
        )
        det = ModelCapabilityDetector()
        with tempfile.TemporaryDirectory() as d:
            profile = det.detect(model_dir=d, chip_profile=chip, ram_gb=16.0)
        self.assertIn("kv-int4", profile.active_features)


# ============================================================================
# TestAutoProfileImport
# ============================================================================

class TestAutoProfileImport(unittest.TestCase):
    """Module must be importable and expose the expected public symbols."""

    def test_module_importable(self):
        from squish.runtime import auto_profile
        self.assertIsNotNone(auto_profile)

    def test_optimization_profile_exported(self):
        from squish.runtime.auto_profile import OptimizationProfile
        self.assertTrue(callable(OptimizationProfile))

    def test_model_capability_detector_exported(self):
        from squish.runtime.auto_profile import ModelCapabilityDetector
        self.assertTrue(callable(ModelCapabilityDetector))


if __name__ == "__main__":
    unittest.main()
