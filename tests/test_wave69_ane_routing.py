"""tests/test_wave69_ane_routing.py

Unit tests for Wave 69: SQUIZD Apple Neural Engine Routing, CoreML
Conversion Pipeline, ANE Sub-8B Path.

Modules under test
──────────────────
* squish.platform.ane_router       — ANE detection + routing policy
* squish.convert_coreml            — CoreML export pipeline
* squish.loaders.coreml_loader     — CoreML appendix loader + runtime
* squish.serving.ane_server        — ANE serving path

All tests run without hardware or coremltools — NumPy fallback is used
throughout.  Hardware-specific behaviour is exercised via monkey-patching
and environment variable overrides.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers shared across test classes
# ---------------------------------------------------------------------------

class _FakeChipProfile:
    """Minimal stand-in for ChipProfile produced by ChipDetector.detect()."""
    def __init__(self, generation: int = 3):
        self.generation = generation
        self.neural_engine_tops = 18.0
        self.mlx_dispatch_unrestricted = True


class _FakeDetector:
    """Minimal stand-in for ChipDetector."""
    def __init__(self, generation: int = 3):
        self._profile = _FakeChipProfile(generation)

    def detect(self) -> _FakeChipProfile:
        return self._profile


def _make_squizd_with_appendix(tmp_dir: Path, payload: bytes = b"{}") -> Path:
    """Write a minimal fake .squizd file containing an ANE_COREML appendix."""
    from squish.convert_coreml import SQUIZD_APPENDIX_TAG
    path = tmp_dir / "model.squizd"
    header = SQUIZD_APPENDIX_TAG + struct.pack("<Q", len(payload))
    path.write_bytes(b"\x00" * 128 + header + payload)  # dummy 128-byte preamble
    return path


# =============================================================================
# 1. ANERouter — Initialisation
# =============================================================================

class TestANERouterInit(unittest.TestCase):
    def setUp(self):
        import squish.platform.ane_router as _mod
        _mod.reset_ane_router()

    def tearDown(self):
        import squish.platform.ane_router as _mod
        _mod.reset_ane_router()

    def test_imports_cleanly(self):
        from squish.platform.ane_router import ANERouter
        self.assertTrue(callable(ANERouter))

    def test_instantiation_no_args(self):
        from squish.platform.ane_router import ANERouter
        r = ANERouter(_detector_override=_FakeDetector(3))
        self.assertIsNotNone(r)

    def test_chip_generation_stored(self):
        from squish.platform.ane_router import ANERouter
        r = ANERouter(_detector_override=_FakeDetector(3))
        self.assertEqual(r._chip_generation, 3)

    def test_ane_budget_m1(self):
        from squish.platform.ane_router import ANERouter
        r = ANERouter(_detector_override=_FakeDetector(1))
        self.assertAlmostEqual(r._ane_budget_gb, 2.0)

    def test_ane_budget_m3(self):
        from squish.platform.ane_router import ANERouter
        r = ANERouter(_detector_override=_FakeDetector(3))
        self.assertAlmostEqual(r._ane_budget_gb, 4.0)

    def test_ane_budget_m5(self):
        from squish.platform.ane_router import ANERouter
        r = ANERouter(_detector_override=_FakeDetector(5))
        self.assertAlmostEqual(r._ane_budget_gb, 8.0)

    def test_singleton_get_ane_router(self):
        from squish.platform.ane_router import get_ane_router, reset_ane_router
        reset_ane_router()
        r1 = get_ane_router()
        r2 = get_ane_router()
        self.assertIs(r1, r2)

    def test_reset_clears_singleton(self):
        from squish.platform.ane_router import get_ane_router, reset_ane_router
        r1 = get_ane_router()
        reset_ane_router()
        r2 = get_ane_router()
        self.assertIsNot(r1, r2)


# =============================================================================
# 2. ANERouter — Routing decisions
# =============================================================================

class TestANERouterRouting(unittest.TestCase):

    def _router(self, generation: int = 3) -> Any:
        from squish.platform.ane_router import ANERouter
        return ANERouter(_detector_override=_FakeDetector(generation))

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_sub8b_routes_to_ane(self, _m):
        r = self._router(3)
        self.assertEqual(r.route(3_800_000_000), "ane")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_exactly_8b_routes_to_ane(self, _m):
        r = self._router(3)
        self.assertEqual(r.route(8_000_000_000), "ane")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_over_8b_routes_to_gpu(self, _m):
        r = self._router(3)
        self.assertEqual(r.route(8_000_000_001), "gpu")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_70b_routes_to_gpu(self, _m):
        r = self._router(4)
        self.assertEqual(r.route(70_000_000_000), "gpu")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_600m_routes_to_ane(self, _m):
        r = self._router(3)
        self.assertEqual(r.route(600_000_000), "ane")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_unknown_chip_routes_to_gpu(self, _m):
        r = self._router(0)
        self.assertEqual(r.route(1_000_000_000), "gpu")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=False)
    def test_linux_routes_to_gpu(self, _m):
        r = self._router(3)
        self.assertEqual(r.route(1_000_000_000), "gpu")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_get_policy_returns_policy_object(self, _m):
        from squish.platform.ane_router import ANERoutingPolicy
        r = self._router(3)
        p = r.get_policy(1_000_000_000)
        self.assertIsInstance(p, ANERoutingPolicy)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_policy_preferred_backend_matches_route(self, _m):
        r = self._router(3)
        p = r.get_policy(1_000_000_000)
        self.assertEqual(p.preferred_backend, r.route(1_000_000_000))

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_policy_has_reason_string(self, _m):
        r = self._router(3)
        p = r.get_policy(1_000_000_000)
        self.assertIsInstance(p.reason, str)
        self.assertTrue(len(p.reason) > 0)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_policy_chip_generation_correct(self, _m):
        r = self._router(4)
        p = r.get_policy(1_000_000_000)
        self.assertEqual(p.chip_generation, 4)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_policy_ane_budget_positive(self, _m):
        r = self._router(3)
        p = r.get_policy(1_000_000_000)
        self.assertGreater(p.ane_memory_budget_gb, 0)


# =============================================================================
# 3. ANERouter — Environment variable overrides
# =============================================================================

class TestANERouterEnvOverride(unittest.TestCase):

    def _router(self, generation: int = 3) -> Any:
        from squish.platform.ane_router import ANERouter
        return ANERouter(_detector_override=_FakeDetector(generation))

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_disabled_env_routes_gpu(self, _m):
        r = self._router(3)
        with patch.dict(os.environ, {"SQUISH_ANE_ENABLED": "0"}):
            self.assertEqual(r.route(600_000_000), "gpu")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_forced_env_routes_ane(self, _m):
        r = self._router(3)
        with patch.dict(os.environ, {"SQUISH_ANE_ENABLED": "1"}):
            self.assertEqual(r.route(600_000_000), "ane")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_forced_env_does_not_override_8b_limit(self, _m):
        r = self._router(3)
        with patch.dict(os.environ, {"SQUISH_ANE_ENABLED": "1"}):
            self.assertEqual(r.route(9_000_000_000), "gpu")

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_disabled_reason_mentions_env(self, _m):
        r = self._router(3)
        with patch.dict(os.environ, {"SQUISH_ANE_ENABLED": "0"}):
            p = r.get_policy(600_000_000)
            self.assertIn("SQUISH_ANE_ENABLED", p.reason)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_no_env_var_uses_chip_routing(self, _m):
        r = self._router(3)
        env_no_ane = {k: v for k, v in os.environ.items() if k != "SQUISH_ANE_ENABLED"}
        with patch.dict(os.environ, env_no_ane, clear=True):
            self.assertEqual(r.route(600_000_000), "ane")


# =============================================================================
# 4. ANERouter — Capability caching
# =============================================================================

class TestANERouterCaching(unittest.TestCase):

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_cache_caps_writes_file(self, _m):
        from squish.platform.ane_router import ANERouter
        with tempfile.TemporaryDirectory() as tmp:
            caps_path = Path(tmp) / "caps.json"
            r = ANERouter(_detector_override=_FakeDetector(3), _caps_path=caps_path)
            r.cache_caps()
            self.assertTrue(caps_path.exists())

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_cache_caps_content_keys(self, _m):
        from squish.platform.ane_router import ANERouter
        with tempfile.TemporaryDirectory() as tmp:
            caps_path = Path(tmp) / "caps.json"
            r = ANERouter(_detector_override=_FakeDetector(3), _caps_path=caps_path)
            r.cache_caps()
            data = json.loads(caps_path.read_text())
            self.assertIn("chip_generation", data)
            self.assertIn("ane_budget_gb", data)
            self.assertIn("ane_available", data)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_load_caps_returns_none_missing(self, _m):
        from squish.platform.ane_router import ANERouter
        r = ANERouter(_detector_override=_FakeDetector(3))
        result = r.load_caps(Path("/nonexistent/path/caps.json"))
        self.assertIsNone(result)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_load_caps_round_trips(self, _m):
        from squish.platform.ane_router import ANERouter
        with tempfile.TemporaryDirectory() as tmp:
            caps_path = Path(tmp) / "caps.json"
            r = ANERouter(_detector_override=_FakeDetector(4), _caps_path=caps_path)
            r.cache_caps()
            data = r.load_caps()
            self.assertIsNotNone(data)
            self.assertEqual(data["chip_generation"], 4)

    @patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True)
    def test_cache_caps_creates_parent_dirs(self, _m):
        from squish.platform.ane_router import ANERouter
        with tempfile.TemporaryDirectory() as tmp:
            caps_path = Path(tmp) / "deep" / "nested" / "caps.json"
            r = ANERouter(_detector_override=_FakeDetector(3), _caps_path=caps_path)
            r.cache_caps()
            self.assertTrue(caps_path.exists())


# =============================================================================
# 5. ANERouter — Platform guard
# =============================================================================

class TestANERouterPlatformGuard(unittest.TestCase):

    def test_is_ane_available_non_macos(self):
        from squish.platform.ane_router import ANERouter
        with patch("squish.platform.ane_router.ANERouter._is_macos", return_value=False):
            r = ANERouter(_detector_override=_FakeDetector(3))
            self.assertFalse(r.is_ane_available())

    def test_is_ane_available_unknown_chip(self):
        from squish.platform.ane_router import ANERouter
        with patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True):
            r = ANERouter(_detector_override=_FakeDetector(0))
            self.assertFalse(r.is_ane_available())

    def test_is_ane_available_valid_chip(self):
        from squish.platform.ane_router import ANERouter
        with patch("squish.platform.ane_router.ANERouter._is_macos", return_value=True):
            r = ANERouter(_detector_override=_FakeDetector(3))
            self.assertTrue(r.is_ane_available())

    def test_non_macos_policy_reason(self):
        from squish.platform.ane_router import ANERouter
        with patch("squish.platform.ane_router.ANERouter._is_macos", return_value=False):
            r = ANERouter(_detector_override=_FakeDetector(3))
            p = r.get_policy(1_000_000_000)
            self.assertIn("non-macOS", p.reason)
            self.assertFalse(p.enabled)


# =============================================================================
# 6. CoreMLConversionConfig
# =============================================================================

class TestCoreMLConversionConfig(unittest.TestCase):

    def test_default_quantization(self):
        from squish.convert_coreml import CoreMLConversionConfig
        cfg = CoreMLConversionConfig()
        self.assertEqual(cfg.quantization, "int4")

    def test_default_chunk_size(self):
        from squish.convert_coreml import CoreMLConversionConfig
        cfg = CoreMLConversionConfig()
        self.assertAlmostEqual(cfg.chunk_size_gb, 2.0)

    def test_custom_quantization(self):
        from squish.convert_coreml import CoreMLConversionConfig
        cfg = CoreMLConversionConfig(quantization="fp16")
        self.assertEqual(cfg.quantization, "fp16")

    def test_fuse_layernorm_default_true(self):
        from squish.convert_coreml import CoreMLConversionConfig
        cfg = CoreMLConversionConfig()
        self.assertTrue(cfg.fuse_layernorm)

    def test_merge_rope_default_true(self):
        from squish.convert_coreml import CoreMLConversionConfig
        cfg = CoreMLConversionConfig()
        self.assertTrue(cfg.merge_rope)

    def test_target_chip_default(self):
        from squish.convert_coreml import CoreMLConversionConfig
        cfg = CoreMLConversionConfig()
        self.assertEqual(cfg.target_chip, "ane")


# =============================================================================
# 7. CoreMLConverter — conversion logic
# =============================================================================

class TestCoreMLConverter(unittest.TestCase):

    def _weights(self, n: int = 4096) -> dict:
        rng = np.random.default_rng(seed=42)
        return {"weight": rng.standard_normal((n, n)).astype(np.float32)}

    def test_convert_returns_package(self):
        from squish.convert_coreml import CoreMLConverter, CoreMLPackage
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        self.assertIsInstance(pkg, CoreMLPackage)

    def test_package_has_at_least_one_chunk(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        self.assertGreater(pkg.chunk_count, 0)

    def test_total_param_count_positive(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        self.assertGreater(pkg.total_param_count, 0)

    def test_chunk_paths_exist(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        for chunk in pkg.chunks:
            self.assertTrue(chunk.mlpackage_path.exists())

    def test_large_model_produces_multiple_chunks(self):
        from squish.convert_coreml import CoreMLConverter, CoreMLConversionConfig
        rng = np.random.default_rng(0)
        # 32-layer model, 4096-dim: 32×4096×4096 * 4 bytes ≈ 2 GB  → > chunk_size
        weights = {f"layers.{i}.weight": rng.standard_normal((4096, 4096)).astype(np.float16)
                   for i in range(32)}
        cfg = CoreMLConversionConfig(chunk_size_gb=0.2, quantization="fp16")
        c = CoreMLConverter(config=cfg)
        pkg = c.convert(weights, layer_count=32)
        self.assertGreater(pkg.chunk_count, 1)

    def test_chunk_indices_sequential(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        for i, chunk in enumerate(pkg.chunks):
            self.assertEqual(chunk.index, i)

    def test_chunk_checksum_non_empty(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        for chunk in pkg.chunks:
            self.assertTrue(len(chunk.checksum) > 0)

    def test_manifest_has_required_keys(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        m = pkg.manifest()
        for key in ("header_bit", "chunk_count", "total_param_count", "chunks"):
            self.assertIn(key, m)

    def test_manifest_header_bit_is_6(self):
        from squish.convert_coreml import CoreMLConverter, SQUIZD_ANE_COREML_BIT
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        self.assertEqual(pkg.manifest()["header_bit"], SQUIZD_ANE_COREML_BIT)

    def test_coremltools_used_false_on_fallback(self):
        from squish.convert_coreml import CoreMLConverter
        c = CoreMLConverter()
        pkg = c.convert(self._weights())
        # In test env, coremltools is absent — simulation mode.
        self.assertFalse(pkg.coremltools_used)


# =============================================================================
# 8. CoreMLConverter — write_squizd_appendix
# =============================================================================

class TestCoreMLConverterAppendix(unittest.TestCase):

    def test_write_appendix_returns_byte_count(self):
        from squish.convert_coreml import CoreMLConverter
        with tempfile.TemporaryDirectory() as tmp:
            squizd = Path(tmp) / "model.squizd"
            squizd.write_bytes(b"\x00" * 64)
            c = CoreMLConverter()
            weights = {"weight": np.zeros((128, 128), dtype=np.float16)}
            pkg = c.convert(weights)
            n = c.write_squizd_appendix(pkg, squizd)
            self.assertIsInstance(n, int)
            self.assertGreater(n, 12)  # > ANML tag (4) + len (8)

    def test_write_appendix_tag_present(self):
        from squish.convert_coreml import CoreMLConverter, SQUIZD_APPENDIX_TAG
        with tempfile.TemporaryDirectory() as tmp:
            squizd = Path(tmp) / "model.squizd"
            squizd.write_bytes(b"\x00" * 64)
            c = CoreMLConverter()
            weights = {"weight": np.zeros((128, 128), dtype=np.float16)}
            pkg = c.convert(weights)
            c.write_squizd_appendix(pkg, squizd)
            data = squizd.read_bytes()
            self.assertIn(SQUIZD_APPENDIX_TAG, data)

    def test_write_appendix_payload_is_valid_json(self):
        from squish.convert_coreml import (
            CoreMLConverter, SQUIZD_APPENDIX_TAG, _APPENDIX_HEADER_SIZE
        )
        with tempfile.TemporaryDirectory() as tmp:
            squizd = Path(tmp) / "model.squizd"
            squizd.write_bytes(b"\x00" * 64)
            c = CoreMLConverter()
            weights = {"weight": np.zeros((128, 128), dtype=np.float16)}
            pkg = c.convert(weights)
            c.write_squizd_appendix(pkg, squizd)
            data = squizd.read_bytes()
            tag_idx = data.index(SQUIZD_APPENDIX_TAG)
            payload_len = struct.unpack("<Q", data[tag_idx + 4: tag_idx + 12])[0]
            payload = data[tag_idx + 12: tag_idx + 12 + payload_len]
            parsed = json.loads(payload)
            self.assertIsInstance(parsed, dict)


# =============================================================================
# 9. CoreMLLoaderConfig
# =============================================================================

class TestCoreMLLoaderConfig(unittest.TestCase):

    def test_default_fallback_true(self):
        from squish.loaders.coreml_loader import CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig()
        self.assertTrue(cfg.fallback_to_gpu)

    def test_default_extract_dir_empty(self):
        from squish.loaders.coreml_loader import CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig()
        self.assertEqual(cfg.extract_dir, "")

    def test_default_ane_router_none(self):
        from squish.loaders.coreml_loader import CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig()
        self.assertIsNone(cfg.ane_router)

    def test_custom_extract_dir(self):
        from squish.loaders.coreml_loader import CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig(extract_dir="/tmp/test")
        self.assertEqual(cfg.extract_dir, "/tmp/test")


# =============================================================================
# 10. CoreMLLoader — has_ane_appendix
# =============================================================================

class TestCoreMLLoaderHasAppendix(unittest.TestCase):

    def test_missing_file_returns_false(self):
        from squish.loaders.coreml_loader import CoreMLLoader
        loader = CoreMLLoader()
        self.assertFalse(loader.has_ane_appendix("/nonexistent/model.squizd"))

    def test_file_without_appendix_returns_false(self):
        from squish.loaders.coreml_loader import CoreMLLoader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 256)
            loader = CoreMLLoader()
            self.assertFalse(loader.has_ane_appendix(path))

    def test_file_with_appendix_returns_true(self):
        from squish.loaders.coreml_loader import CoreMLLoader
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.dumps({"chunks": [], "chunk_count": 0}).encode()
            path = _make_squizd_with_appendix(Path(tmp), manifest)
            loader = CoreMLLoader()
            self.assertTrue(loader.has_ane_appendix(path))

    def test_empty_file_returns_false(self):
        from squish.loaders.coreml_loader import CoreMLLoader
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"")
            loader = CoreMLLoader()
            self.assertFalse(loader.has_ane_appendix(path))


# =============================================================================
# 11. CoreMLLoader — fallback behaviour
# =============================================================================

class TestCoreMLLoaderFallback(unittest.TestCase):

    def test_missing_file_raises(self):
        from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig(fallback_to_gpu=True)
        loader = CoreMLLoader(cfg)
        with self.assertRaises(FileNotFoundError):
            loader.load("/nonexistent/model.squizd")

    def test_no_appendix_fallback_returns_runtime(self):
        from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig(fallback_to_gpu=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 256)
            loader = CoreMLLoader(cfg)
            runtime = loader.load(path)
            self.assertEqual(runtime.backend(), "gpu_fallback")

    def test_no_appendix_no_fallback_raises(self):
        from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig
        cfg = CoreMLLoaderConfig(fallback_to_gpu=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 256)
            loader = CoreMLLoader(cfg)
            with self.assertRaises(RuntimeError):
                loader.load(path)

    def test_fallback_available_respects_config(self):
        from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig
        cfg_true = CoreMLLoaderConfig(fallback_to_gpu=True)
        cfg_false = CoreMLLoaderConfig(fallback_to_gpu=False)
        self.assertTrue(CoreMLLoader(cfg_true).fallback_available())
        self.assertFalse(CoreMLLoader(cfg_false).fallback_available())

    def test_router_blocking_returns_fallback(self):
        from squish.loaders.coreml_loader import CoreMLLoader, CoreMLLoaderConfig
        mock_router = MagicMock()
        mock_router.route.return_value = "gpu"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 64)
            cfg = CoreMLLoaderConfig(fallback_to_gpu=True, ane_router=mock_router)
            loader = CoreMLLoader(cfg)
            rt = loader.load(path)
            self.assertEqual(rt.backend(), "gpu_fallback")


# =============================================================================
# 12. CoreMLRuntime
# =============================================================================

class TestCoreMLRuntime(unittest.TestCase):

    def _runtime(self, paths=None):
        from squish.loaders.coreml_loader import CoreMLRuntime
        return CoreMLRuntime(mlpackage_paths=paths or [], use_coreml=False)

    def test_is_loaded_true(self):
        rt = self._runtime()
        self.assertTrue(rt.is_loaded())

    def test_backend_gpu_fallback(self):
        rt = self._runtime()
        self.assertEqual(rt.backend(), "gpu_fallback")

    def test_predict_shape(self):
        rt = self._runtime()
        input_ids = np.array([[1, 2, 3]])
        out = rt.predict(input_ids)
        self.assertEqual(out.ndim, 2)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 32_000)

    def test_predict_batch_size_2(self):
        rt = self._runtime()
        input_ids = np.array([[1, 2], [3, 4]])
        out = rt.predict(input_ids)
        self.assertEqual(out.shape, (2, 32_000))

    def test_predict_dtype_float32(self):
        rt = self._runtime()
        out = rt.predict(np.array([[1, 2, 3]]))
        self.assertEqual(out.dtype, np.float32)

    def test_chunk_count_zero_on_empty(self):
        rt = self._runtime([])
        self.assertEqual(rt.chunk_count(), 0)

    def test_predict_deterministic_same_seed(self):
        from squish.loaders.coreml_loader import CoreMLRuntime
        rt = CoreMLRuntime([], use_coreml=False)
        ids = np.array([[5, 10, 15]])
        out1 = rt.predict(ids)
        out2 = rt.predict(ids)
        np.testing.assert_array_equal(out1, out2)


# =============================================================================
# 13. ANEServerConfig
# =============================================================================

class TestANEServerConfig(unittest.TestCase):

    def test_default_port(self):
        from squish.serving.ane_server import ANEServerConfig
        cfg = ANEServerConfig()
        self.assertEqual(cfg.port, 11436)

    def test_default_fallback_true(self):
        from squish.serving.ane_server import ANEServerConfig
        cfg = ANEServerConfig()
        self.assertTrue(cfg.fallback_to_gpu)

    def test_default_max_tokens(self):
        from squish.serving.ane_server import ANEServerConfig
        cfg = ANEServerConfig()
        self.assertEqual(cfg.max_tokens, 4_096)

    def test_default_temperature(self):
        from squish.serving.ane_server import ANEServerConfig
        cfg = ANEServerConfig()
        self.assertAlmostEqual(cfg.temperature, 1.0)

    def test_default_top_p(self):
        from squish.serving.ane_server import ANEServerConfig
        cfg = ANEServerConfig()
        self.assertAlmostEqual(cfg.top_p, 0.95)

    def test_custom_port(self):
        from squish.serving.ane_server import ANEServerConfig
        cfg = ANEServerConfig(port=8080)
        self.assertEqual(cfg.port, 8080)


# =============================================================================
# 14. ANEServingRuntime
# =============================================================================

class TestANEServingRuntime(unittest.TestCase):

    def _prepared_runtime(self, tmp_dir: Path):
        """Return an ANEServingRuntime prepared with a fallback-mode .squizd."""
        from squish.serving.ane_server import ANEServingRuntime, ANEServerConfig
        path = tmp_dir / "model.squizd"
        path.write_bytes(b"\x00" * 64)
        rt = ANEServingRuntime(config=ANEServerConfig(fallback_to_gpu=True))
        rt.prepare(path)
        return rt

    def test_not_ready_before_prepare(self):
        from squish.serving.ane_server import ANEServingRuntime
        rt = ANEServingRuntime()
        self.assertFalse(rt.is_ready())

    def test_ready_after_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            self.assertTrue(rt.is_ready())

    def test_backend_uninitialized(self):
        from squish.serving.ane_server import ANEServingRuntime
        rt = ANEServingRuntime()
        self.assertEqual(rt.backend(), "uninitialized")

    def test_backend_after_prepare(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            self.assertIn(rt.backend(), ("coreml_ane", "gpu_fallback"))

    def test_generate_stream_raises_if_not_ready(self):
        from squish.serving.ane_server import ANEServingRuntime
        rt = ANEServingRuntime()
        with self.assertRaises(RuntimeError):
            list(rt.generate_stream("hello"))

    def test_generate_stream_yields_tuples(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            tokens = list(rt.generate_stream("Hi", max_tokens=5, seed=0))
            self.assertTrue(len(tokens) > 0)
            for tok, fr in tokens:
                self.assertIsInstance(tok, str)
                self.assertTrue(fr is None or isinstance(fr, str))

    def test_generate_stream_last_token_has_finish_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            tokens = list(rt.generate_stream("Hi", max_tokens=5, seed=0))
            _, last_fr = tokens[-1]
            self.assertIn(last_fr, ("stop", "length"))

    def test_generate_stream_respects_max_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            tokens = list(rt.generate_stream("Hello world", max_tokens=3, seed=1))
            self.assertLessEqual(len(tokens), 3)

    def test_generate_non_streaming(self):
        from squish.serving.ane_server import GenerationResult
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            result = rt.generate("Hi", max_tokens=4, seed=0)
            self.assertIsInstance(result, GenerationResult)
            self.assertIsInstance(result.text, str)
            self.assertGreater(result.tokens_generated, 0)
            self.assertGreater(result.ttft_ms, 0)
            self.assertGreater(result.total_ms, 0)

    def test_generate_result_backend_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self._prepared_runtime(Path(tmp))
            result = rt.generate("Hi", max_tokens=2, seed=0)
            self.assertIn(result.backend, ("coreml_ane", "gpu_fallback"))


# =============================================================================
# 15. ANEServingRuntime — fallback mode
# =============================================================================

class TestANEServingRuntimeFallback(unittest.TestCase):

    def test_fallback_backend_label(self):
        from squish.serving.ane_server import ANEServingRuntime, ANEServerConfig
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 64)
            rt = ANEServingRuntime(ANEServerConfig(fallback_to_gpu=True))
            rt.prepare(path)
            # Without a real CoreML appendix, should use GPU fallback.
            self.assertEqual(rt.backend(), "gpu_fallback")

    def test_generate_deterministic_with_seed(self):
        from squish.serving.ane_server import ANEServingRuntime, ANEServerConfig
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 64)
            rt = ANEServingRuntime(ANEServerConfig(fallback_to_gpu=True))
            rt.prepare(path)
            r1 = rt.generate("Hello", max_tokens=5, seed=42)
            r2 = rt.generate("Hello", max_tokens=5, seed=42)
            self.assertEqual(r1.text, r2.text)

    def test_temperature_zero_deterministic(self):
        from squish.serving.ane_server import ANEServingRuntime, ANEServerConfig
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.squizd"
            path.write_bytes(b"\x00" * 64)
            rt = ANEServingRuntime(ANEServerConfig(fallback_to_gpu=True, temperature=0.0))
            rt.prepare(path)
            r1 = rt.generate("Hi", max_tokens=3, temperature=0.0)
            r2 = rt.generate("Hi", max_tokens=3, temperature=0.0)
            self.assertEqual(r1.text, r2.text)

    def test_generate_with_appendix_file(self):
        """Load from a file that has an ANE_COREML appendix (simulation only)."""
        from squish.serving.ane_server import ANEServingRuntime, ANEServerConfig
        with tempfile.TemporaryDirectory() as tmp:
            manifest = json.dumps({
                "header_bit": 6,
                "chunk_count": 1,
                "total_param_count": 1000000,
                "quantization": "int4",
                "fuse_layernorm": True,
                "merge_rope": True,
                "coremltools_used": False,
                "chunks": [
                    {
                        "index": 0,
                        "path": str(Path(tmp) / "chunk_000.mlpackage"),
                        "layer_start": 0,
                        "layer_end": 7,
                        "param_count": 1000000,
                        "size_bytes": 500000,
                        "checksum": "abc123",
                    }
                ],
            }).encode()
            path = _make_squizd_with_appendix(Path(tmp), manifest)
            rt = ANEServingRuntime(ANEServerConfig(fallback_to_gpu=True))
            rt.prepare(path)
            result = rt.generate("Hello ANE", max_tokens=4, seed=7)
            self.assertGreater(result.tokens_generated, 0)


# =============================================================================
# 16. Module constants and __all__ checks
# =============================================================================

class TestModuleExports(unittest.TestCase):

    def test_ane_router_all(self):
        from squish.platform import ane_router
        for sym in ane_router.__all__:
            self.assertTrue(hasattr(ane_router, sym), f"Missing: {sym}")

    def test_convert_coreml_all(self):
        import squish.convert_coreml as m
        for sym in m.__all__:
            self.assertTrue(hasattr(m, sym), f"Missing: {sym}")

    def test_coreml_loader_all(self):
        from squish.loaders import coreml_loader
        for sym in coreml_loader.__all__:
            self.assertTrue(hasattr(coreml_loader, sym), f"Missing: {sym}")

    def test_ane_server_all(self):
        from squish.serving import ane_server
        for sym in ane_server.__all__:
            self.assertTrue(hasattr(ane_server, sym), f"Missing: {sym}")

    def test_ane_param_limit_value(self):
        from squish.platform.ane_router import ANE_PARAM_LIMIT
        self.assertEqual(ANE_PARAM_LIMIT, 8_000_000_000)

    def test_squizd_header_bit_is_6(self):
        from squish.convert_coreml import SQUIZD_ANE_COREML_BIT
        self.assertEqual(SQUIZD_ANE_COREML_BIT, 6)

    def test_appendix_tag_length(self):
        from squish.convert_coreml import SQUIZD_APPENDIX_TAG
        self.assertEqual(len(SQUIZD_APPENDIX_TAG), 4)

    def test_appendix_tag_value(self):
        from squish.convert_coreml import SQUIZD_APPENDIX_TAG
        self.assertEqual(SQUIZD_APPENDIX_TAG, b"ANML")


if __name__ == "__main__":
    unittest.main()
