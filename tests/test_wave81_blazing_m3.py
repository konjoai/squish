"""tests/test_wave81_blazing_m3.py

Wave 81 — M3-16GB Sprint: Sub-3s TTFT for 7/8B models

Tests for:
  - ChipProfile new fields: recommended_model_bits, recommended_chunk_prefill_ttft
  - ChipDetector.detect_ram_gb() returns a float (platform-safe)
  - ChipDetector.get_recommended_model_bits() memory-tier logic
  - ChipDetector.get_ttft_chunk_size() halves for large models
  - server._has_quantized_layers() detects quantized vs plain models
  - server._blazing_preset_defaults() applies correct defaults to args
  - server._configure_blazing_mode() wires globals when --blazing set
  - cli._apply_blazing_m3_preset() applies correct quantisation defaults
"""
from __future__ import annotations

import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import squish.server as _srv
import squish.cli as _cli
from squish.hardware.chip_detector import (
    AppleChipGeneration,
    ChipDetector,
    ChipProfile,
    CHIP_PROFILES,
)

# Convenience aliases for the most-tested generation keys
_M3 = AppleChipGeneration.M3
_M5 = AppleChipGeneration.M5


# ============================================================================
# TestChipProfileBlazingFields — new dataclass fields
# ============================================================================

class TestChipProfileBlazingFields(unittest.TestCase):
    """ChipProfile must expose recommended_model_bits and recommended_chunk_prefill_ttft."""

    def test_recommended_model_bits_field_present(self):
        profile = CHIP_PROFILES[_M3]
        self.assertTrue(hasattr(profile, "recommended_model_bits"))

    def test_m3_profile_recommends_int2(self):
        profile = CHIP_PROFILES[_M3]
        self.assertEqual(profile.recommended_model_bits, 2)

    def test_m5_profile_recommends_int4(self):
        profile = CHIP_PROFILES[_M5]
        self.assertEqual(profile.recommended_model_bits, 4)

    def test_recommended_chunk_prefill_ttft_field_present(self):
        profile = CHIP_PROFILES[_M3]
        self.assertTrue(hasattr(profile, "recommended_chunk_prefill_ttft"))

    def test_m3_ttft_chunk_is_128(self):
        profile = CHIP_PROFILES[_M3]
        self.assertEqual(profile.recommended_chunk_prefill_ttft, 128)

    def test_m1_m2_ttft_chunk_is_64(self):
        for gen in (AppleChipGeneration.M1, AppleChipGeneration.M2):
            profile = CHIP_PROFILES.get(gen)
            if profile is not None:
                self.assertEqual(
                    profile.recommended_chunk_prefill_ttft, 64,
                    f"{gen} expected ttft_chunk=64, got {profile.recommended_chunk_prefill_ttft}",
                )


# ============================================================================
# TestChipDetectorDetectRamGb — detect_ram_gb()
# ============================================================================

class TestChipDetectorDetectRamGb(unittest.TestCase):
    """ChipDetector.detect_ram_gb() must return a float >=0."""

    def test_returns_float(self):
        result = ChipDetector.detect_ram_gb()
        self.assertIsInstance(result, float)

    def test_returns_nonnegative(self):
        result = ChipDetector.detect_ram_gb()
        self.assertGreaterEqual(result, 0.0)

    def test_returns_zero_on_non_darwin(self):
        """When platform is not Darwin, must return 0.0."""
        with patch("squish.hardware.chip_detector.platform.system", return_value="Linux"):
            result = ChipDetector.detect_ram_gb()
        self.assertEqual(result, 0.0)

    def test_returns_zero_on_sysctl_failure(self):
        import subprocess
        bad = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
        with patch("squish.hardware.chip_detector.platform.system", return_value="Darwin"):
            with patch("squish.hardware.chip_detector.subprocess.run", return_value=bad):
                result = ChipDetector.detect_ram_gb()
        self.assertEqual(result, 0.0)

    def test_parses_sysctl_physmem_correctly(self):
        """17179869184 bytes == 16 GB."""
        import subprocess
        good = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="17179869184\n", stderr=""
        )
        with patch("squish.hardware.chip_detector.platform.system", return_value="Darwin"):
            with patch("squish.hardware.chip_detector.subprocess.run", return_value=good):
                result = ChipDetector.detect_ram_gb()
        self.assertAlmostEqual(result, 16.0, places=2)


# ============================================================================
# TestGetRecommendedModelBits — memory-tier logic
# ============================================================================

class TestGetRecommendedModelBits(unittest.TestCase):
    """get_recommended_model_bits() must apply the correct memory-tier rules."""

    def _detector_with_m3(self) -> ChipDetector:
        det = ChipDetector.__new__(ChipDetector)
        det._profile = CHIP_PROFILES[_M3]
        return det

    def test_16gb_7b_model_returns_2(self):
        det = self._detector_with_m3()
        self.assertEqual(det.get_recommended_model_bits(ram_gb=16.0, model_params_b=7.0), 2)

    def test_18gb_7b_model_returns_3(self):
        """18 GB falls in the >16 ≤20 tier → INT3."""
        det = self._detector_with_m3()
        self.assertEqual(det.get_recommended_model_bits(ram_gb=18.0, model_params_b=7.0), 3)

    def test_24gb_7b_model_returns_4(self):
        """24 GB falls in the >20 ≤28 tier → INT4."""
        det = self._detector_with_m3()
        self.assertEqual(det.get_recommended_model_bits(ram_gb=24.0, model_params_b=7.0), 4)

    def test_small_model_always_int4(self):
        det = self._detector_with_m3()
        self.assertEqual(det.get_recommended_model_bits(ram_gb=8.0, model_params_b=1.5), 4)

    def test_unknown_ram_returns_chip_default(self):
        det = self._detector_with_m3()
        chip_default = CHIP_PROFILES[_M3].recommended_model_bits
        self.assertEqual(
            det.get_recommended_model_bits(ram_gb=0.0, model_params_b=7.0),
            chip_default,
        )


# ============================================================================
# TestGetTtftChunkSize — TTFT-optimised chunk size
# ============================================================================

class TestGetTtftChunkSize(unittest.TestCase):
    """get_ttft_chunk_size() must return chip base for small models, halved for large."""

    def _detector_with_m3(self) -> ChipDetector:
        det = ChipDetector.__new__(ChipDetector)
        det._profile = CHIP_PROFILES[_M3]
        return det

    def test_small_model_returns_ttft_base(self):
        det = self._detector_with_m3()
        result = det.get_ttft_chunk_size(model_size_gb=2.0)
        self.assertEqual(result, CHIP_PROFILES[_M3].recommended_chunk_prefill_ttft)

    def test_large_model_returns_halved_chunk(self):
        det = self._detector_with_m3()
        base = CHIP_PROFILES[_M3].recommended_chunk_prefill_ttft
        result = det.get_ttft_chunk_size(model_size_gb=10.0)
        self.assertLessEqual(result, base)
        self.assertGreaterEqual(result, 64)


# ============================================================================
# TestHasQuantizedLayers — server._has_quantized_layers()
# ============================================================================

class TestHasQuantizedLayers(unittest.TestCase):
    """_has_quantized_layers() must detect quantization via .bits attribute."""

    @staticmethod
    def _make_model(has_bits: bool, depth: int = 1) -> MagicMock:
        """Build a minimal mock model mimicking mlx_lm layout."""
        linear = MagicMock()
        if has_bits:
            linear.bits = 4         # INT4 quantised
        else:
            if hasattr(linear, "bits"):
                del linear.bits     # ensure attribute absent

        layer = MagicMock()
        # Use a plain dict so vars(layer) works
        layer.__dict__ = {"self_attn": linear}

        inner = MagicMock()
        inner.layers = [layer] * depth

        model = MagicMock()
        model.model = inner
        return model

    def test_quantized_model_returns_true(self):
        model = self._make_model(has_bits=True)
        self.assertTrue(_srv._has_quantized_layers(model))

    def test_unquantized_model_returns_false(self):
        model = self._make_model(has_bits=False)
        self.assertFalse(_srv._has_quantized_layers(model))

    def test_no_layers_returns_false(self):
        model = MagicMock()
        inner = MagicMock()
        inner.layers = []
        model.model = inner
        self.assertFalse(_srv._has_quantized_layers(model))

    def test_model_without_model_attribute_returns_false(self):
        model = MagicMock(spec=[])   # no attributes
        self.assertFalse(_srv._has_quantized_layers(model))

    def test_nested_sub_module_bits_detected(self):
        """bits on a nested sub-module of the layer should also be detected."""
        sub_linear = MagicMock()
        sub_linear.bits = 2

        proj = MagicMock()
        proj.__dict__ = {"q_proj": sub_linear}

        layer = MagicMock()
        layer.__dict__ = {"self_attn": proj}

        inner = MagicMock()
        inner.layers = [layer]

        model = MagicMock()
        model.model = inner

        self.assertTrue(_srv._has_quantized_layers(model))


# ============================================================================
# TestBlazingPresetDefaults — server._blazing_preset_defaults()
# ============================================================================

class TestBlazingPresetDefaults(unittest.TestCase):
    """_blazing_preset_defaults() must apply correctly-valued attributes."""

    def _fresh_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            agent_kv=False,
            chunk_prefill_size=512,
            no_chunk_prefill=True,
            fast_gelu=False,
            max_kv_size=None,
            _blazing_metal_cache_mb=256,
        )

    def test_agent_kv_set_true(self):
        args = self._fresh_args()
        _srv._blazing_preset_defaults(args)
        self.assertTrue(args.agent_kv)

    def test_chunk_prefill_size_set_128_default(self):
        args = self._fresh_args()
        _srv._blazing_preset_defaults(args)
        self.assertEqual(args.chunk_prefill_size, 128)

    def test_no_chunk_prefill_cleared(self):
        args = self._fresh_args()
        args.no_chunk_prefill = True
        _srv._blazing_preset_defaults(args)
        self.assertFalse(args.no_chunk_prefill)

    def test_fast_gelu_set_true(self):
        args = self._fresh_args()
        _srv._blazing_preset_defaults(args)
        self.assertTrue(args.fast_gelu)

    def test_max_kv_size_clamped_to_4096(self):
        args = self._fresh_args()
        args.max_kv_size = None
        _srv._blazing_preset_defaults(args)
        self.assertEqual(args.max_kv_size, 4096)

    def test_max_kv_size_not_increased(self):
        """If max_kv_size is already below 4096, must not be raised."""
        args = self._fresh_args()
        args.max_kv_size = 2048
        _srv._blazing_preset_defaults(args)
        self.assertEqual(args.max_kv_size, 2048)

    def test_metal_cache_mb_set_64(self):
        args = self._fresh_args()
        _srv._blazing_preset_defaults(args)
        self.assertEqual(args._blazing_metal_cache_mb, 64)

    def test_returns_same_args_object(self):
        args = self._fresh_args()
        result = _srv._blazing_preset_defaults(args)
        self.assertIs(result, args)

    def test_chip_profile_ttft_chunk_respected(self):
        """When chip_profile provides a recommended_chunk_prefill_ttft, it is used."""
        args = self._fresh_args()
        profile = MagicMock()
        profile.recommended_chunk_prefill_ttft = 64
        _srv._blazing_preset_defaults(args, chip_profile=profile)
        self.assertEqual(args.chunk_prefill_size, 64)


# ============================================================================
# TestConfigureBlazingMode — server._configure_blazing_mode()
# ============================================================================

class TestConfigureBlazingMode(unittest.TestCase):
    """_configure_blazing_mode() must set globals and args when --blazing active."""

    def test_noop_when_blazing_false(self):
        """Must not change globals when args.blazing is False."""
        original_mode = _srv._blazing_mode
        args = SimpleNamespace(blazing=False)
        _srv._configure_blazing_mode(args)
        self.assertEqual(_srv._blazing_mode, original_mode)

    def test_sets_blazing_mode_global(self):
        """_configure_blazing_mode must flip _blazing_mode global to True."""
        # Patch the ChipDetector inside the hardware module (local import path).
        with patch(
            "squish.hardware.chip_detector.ChipDetector.detect_ram_gb",
            return_value=16.0,
        ):
            args = SimpleNamespace(blazing=True)
            _srv._configure_blazing_mode(args)
        self.assertTrue(_srv._blazing_mode)
        # Reset globals so other tests are not affected.
        _srv._blazing_mode = False
        _srv._metal_cache_limit_mb = 256

    def test_sets_metal_cache_limit_global(self):
        """_configure_blazing_mode must set _metal_cache_limit_mb to 64."""
        with patch(
            "squish.hardware.chip_detector.ChipDetector.detect_ram_gb",
            return_value=16.0,
        ):
            args = SimpleNamespace(blazing=True)
            _srv._configure_blazing_mode(args)
        self.assertEqual(_srv._metal_cache_limit_mb, 64)
        # Reset globals so other tests are not affected.
        _srv._blazing_mode = False
        _srv._metal_cache_limit_mb = 256


# ============================================================================
# TestApplyBlazingM3Preset — cli._apply_blazing_m3_preset()
# ============================================================================

class TestApplyBlazingM3Preset(unittest.TestCase):
    """_apply_blazing_m3_preset() must apply M3-optimised quantisation defaults."""

    def _fresh_args(self) -> SimpleNamespace:
        return SimpleNamespace(
            blazing_m3=True,
            ffn_bits=4,
            attn_bits=None,
            embed_bits=6,
            group_size=64,
            hqq=False,
            _default_group_size=64,
        )

    def test_ffn_bits_set_to_2(self):
        args = self._fresh_args()
        _cli._apply_blazing_m3_preset(args)
        self.assertEqual(args.ffn_bits, 2)

    def test_attn_bits_set_to_4(self):
        args = self._fresh_args()
        _cli._apply_blazing_m3_preset(args)
        self.assertEqual(args.attn_bits, 4)

    def test_embed_bits_set_to_8(self):
        args = self._fresh_args()
        _cli._apply_blazing_m3_preset(args)
        self.assertEqual(args.embed_bits, 8)

    def test_group_size_set_to_32(self):
        args = self._fresh_args()
        _cli._apply_blazing_m3_preset(args)
        self.assertEqual(args.group_size, 32)

    def test_hqq_set_to_true(self):
        args = self._fresh_args()
        _cli._apply_blazing_m3_preset(args)
        self.assertTrue(args.hqq)

    def test_noop_when_blazing_m3_false(self):
        """Must not modify any field when blazing_m3=False."""
        args = self._fresh_args()
        args.blazing_m3 = False
        _cli._apply_blazing_m3_preset(args)
        self.assertEqual(args.ffn_bits, 4)   # unchanged
        self.assertIsNone(args.attn_bits)    # unchanged

    def test_user_override_ffn_bits_is_respected(self):
        """If user already set ffn_bits to non-default, preset must not override."""
        args = self._fresh_args()
        args.ffn_bits = 3        # user explicitly chose INT3
        _cli._apply_blazing_m3_preset(args)
        # Only the default (4) should be overridden; 3 stays
        self.assertEqual(args.ffn_bits, 3)

    def test_returns_same_args_object(self):
        args = self._fresh_args()
        result = _cli._apply_blazing_m3_preset(args)
        self.assertIs(result, args)


if __name__ == "__main__":
    unittest.main()
