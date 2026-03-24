"""tests/test_wave70_benchmark_suite.py

Unit tests for Wave 70 benchmark infrastructure:

Modules under test
──────────────────
* squish.bench.squish_bench          — SquizdBenchmark, SquizdBenchConfig,
  SquizdModelResult, SquizdFormatVariant, FormatComparison, GGUFBaselineResult
* squish.hardware.capability_probe   — HardwareCapabilities, CapabilityProbe,
  get_capability_probe
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_inference_fn(
    ttft_ms: float = 30.0,
    n_tokens: int = 50,
    peak_gb: float = 2.0,
):
    """Return a zero-argument callable producing fixed measurements."""
    def fn() -> Tuple[float, int, float]:
        return ttft_ms, n_tokens, peak_gb
    return fn


def _make_noisy_inference_fn(
    base_ttft: float = 30.0,
    base_tps: float = 60.0,
    seed: int = 0,
):
    """Return an inference fn with reproducible noise for percentile tests."""
    rng = np.random.default_rng(seed)

    def fn() -> Tuple[float, int, float]:
        ttft = float(base_ttft + rng.normal(0, 2.0))
        tps = float(base_tps + rng.normal(0, 5.0))
        n_tok = max(1, int(100 * tps / base_tps))
        total_ms = (n_tok / max(tps, 0.01)) * 1000.0
        peak = float(rng.uniform(1.5, 3.0))
        return ttft, n_tok, peak

    return fn


# =============================================================================
# 1. SquizdFormatVariant
# =============================================================================

class TestSquizdFormatVariant(unittest.TestCase):
    def test_all_four_variants_exist(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        variants = list(SquizdFormatVariant)
        self.assertEqual(len(variants), 4)

    def test_astc_value(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        self.assertEqual(SquizdFormatVariant.ASTC.value, "squizd-astc")

    def test_int4_value(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        self.assertEqual(SquizdFormatVariant.INT4.value, "squizd-int4")

    def test_int4_sparse_value(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        self.assertEqual(SquizdFormatVariant.INT4_SPARSE.value, "squizd-int4-sparse")

    def test_full_value(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        self.assertEqual(SquizdFormatVariant.FULL.value, "squizd-full")

    def test_str_returns_value(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        self.assertEqual(str(SquizdFormatVariant.ASTC), "squizd-astc")


# =============================================================================
# 2. SquizdBenchConfig
# =============================================================================

class TestSquizdBenchConfig(unittest.TestCase):
    def test_defaults(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        cfg = SquizdBenchConfig()
        self.assertEqual(cfg.n_trials, 30)
        self.assertEqual(cfg.warmup_trials, 3)
        self.assertEqual(cfg.max_tokens, 100)

    def test_custom_trials(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        cfg = SquizdBenchConfig(n_trials=5, warmup_trials=1)
        self.assertEqual(cfg.n_trials, 5)
        self.assertEqual(cfg.warmup_trials, 1)

    def test_invalid_n_trials_raises(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        with self.assertRaises(ValueError):
            SquizdBenchConfig(n_trials=0)

    def test_invalid_warmup_raises(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        with self.assertRaises(ValueError):
            SquizdBenchConfig(warmup_trials=-1)

    def test_default_variants_all_four(self):
        from squish.bench.squish_bench import SquizdBenchConfig, SquizdFormatVariant
        cfg = SquizdBenchConfig()
        self.assertEqual(set(cfg.variants), set(SquizdFormatVariant))

    def test_default_models_nonempty(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        cfg = SquizdBenchConfig()
        self.assertGreater(len(cfg.models), 0)

    def test_default_models_contains_qwen(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        cfg = SquizdBenchConfig()
        self.assertTrue(any("Qwen" in m for m in cfg.models))

    def test_21_default_models(self):
        from squish.bench.squish_bench import SquizdBenchConfig
        cfg = SquizdBenchConfig()
        self.assertEqual(len(cfg.models), 21)


# =============================================================================
# 3. SquizdBenchmark.run_variant
# =============================================================================

class TestSquizdBenchmarkRunVariant(unittest.TestCase):
    def _bench(self, n_trials=5, warmup=1):
        from squish.bench.squish_bench import SquizdBenchConfig, SquizdBenchmark
        cfg = SquizdBenchConfig(n_trials=n_trials, warmup_trials=warmup)
        return SquizdBenchmark(cfg)

    def test_run_variant_returns_result(self):
        from squish.bench.squish_bench import SquizdFormatVariant, SquizdModelResult
        bench = self._bench()
        fn = _make_inference_fn()
        result = bench.run_variant("TestModel", SquizdFormatVariant.INT4, fn)
        self.assertIsInstance(result, SquizdModelResult)

    def test_result_model_name(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench()
        fn = _make_inference_fn()
        result = bench.run_variant("MyModel", SquizdFormatVariant.INT4, fn)
        self.assertEqual(result.model, "MyModel")

    def test_result_variant(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench()
        fn = _make_inference_fn()
        result = bench.run_variant("M", SquizdFormatVariant.ASTC, fn)
        self.assertEqual(result.variant, SquizdFormatVariant.ASTC)

    def test_ttft_p50_positive(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench()
        fn = _make_inference_fn(ttft_ms=50.0)
        result = bench.run_variant("M", SquizdFormatVariant.INT4, fn)
        self.assertGreater(result.ttft_p50_ms, 0)

    def test_n_trials_correct(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench(n_trials=7, warmup=2)
        fn = _make_inference_fn()
        result = bench.run_variant("M", SquizdFormatVariant.INT4, fn)
        self.assertEqual(result.n_trials, 7)

    def test_percentile_ordering_ttft(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench(n_trials=20, warmup=2)
        fn = _make_noisy_inference_fn()
        result = bench.run_variant("M", SquizdFormatVariant.FULL, fn)
        self.assertLessEqual(result.ttft_p50_ms, result.ttft_p95_ms)
        self.assertLessEqual(result.ttft_p95_ms, result.ttft_p99_ms)

    def test_percentile_ordering_tps(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench(n_trials=20, warmup=2)
        fn = _make_noisy_inference_fn(seed=99)
        result = bench.run_variant("M", SquizdFormatVariant.FULL, fn)
        self.assertLessEqual(result.tps_p50, result.tps_p95 + 1e-6)

    def test_disk_size_zero_when_no_path(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench()
        fn = _make_inference_fn()
        result = bench.run_variant("M", SquizdFormatVariant.INT4, fn)
        self.assertEqual(result.disk_size_gb, 0.0)

    def test_disk_size_from_real_file(self):
        from squish.bench.squish_bench import SquizdFormatVariant
        bench = self._bench()
        fn = _make_inference_fn()
        with tempfile.NamedTemporaryFile(suffix=".squizd", delete=False) as f:
            f.write(b"\x00" * 1024)
            p = Path(f.name)
        try:
            result = bench.run_variant("M", SquizdFormatVariant.INT4, fn, model_path=p)
            self.assertGreater(result.disk_size_gb, 0.0)
        finally:
            p.unlink(missing_ok=True)


# =============================================================================
# 4. SquizdBenchmark.to_markdown_table
# =============================================================================

class TestMarkdownTable(unittest.TestCase):
    def _make_result(self, model="M", variant=None):
        from squish.bench.squish_bench import SquizdFormatVariant, SquizdModelResult
        v = variant or SquizdFormatVariant.INT4
        return SquizdModelResult(
            model=model, variant=v,
            ttft_p50_ms=30.0, ttft_p95_ms=40.0, ttft_p99_ms=50.0,
            tps_p50=60.0, tps_p95=55.0, tps_p99=50.0,
            peak_memory_gb=2.0, disk_size_gb=4.0, ram_resident_gb=5.0,
            n_trials=30,
        )

    def test_table_contains_header_row(self):
        from squish.bench.squish_bench import SquizdBenchmark
        bench = SquizdBenchmark()
        table = bench.to_markdown_table([self._make_result()])
        self.assertIn("Model", table)
        self.assertIn("Variant", table)

    def test_table_contains_model_name(self):
        from squish.bench.squish_bench import SquizdBenchmark
        bench = SquizdBenchmark()
        table = bench.to_markdown_table([self._make_result("Qwen2.5-7B")])
        self.assertIn("Qwen2.5-7B", table)

    def test_table_rows_match_results(self):
        from squish.bench.squish_bench import SquizdBenchmark, SquizdFormatVariant
        bench = SquizdBenchmark()
        results = [self._make_result(f"M{i}") for i in range(3)]
        table = bench.to_markdown_table(results)
        for i in range(3):
            self.assertIn(f"M{i}", table)

    def test_table_empty_input(self):
        from squish.bench.squish_bench import SquizdBenchmark
        bench = SquizdBenchmark()
        table = bench.to_markdown_table([])
        self.assertIn("Model", table)


# =============================================================================
# 5. FormatComparison and compare_to_gguf
# =============================================================================

class TestFormatComparison(unittest.TestCase):
    def _squizd(self, model="M"):
        from squish.bench.squish_bench import SquizdFormatVariant, SquizdModelResult
        return SquizdModelResult(
            model=model, variant=SquizdFormatVariant.FULL,
            ttft_p50_ms=20.0, ttft_p95_ms=25.0, ttft_p99_ms=30.0,
            tps_p50=80.0, tps_p95=75.0, tps_p99=70.0,
            peak_memory_gb=2.0, disk_size_gb=3.5, ram_resident_gb=4.0,
            n_trials=30,
        )

    def _gguf(self, model="M"):
        from squish.bench.squish_bench import GGUFBaselineResult
        return GGUFBaselineResult(
            model=model, ttft_p50_ms=40.0, tps_p50=50.0,
            disk_size_gb=7.0, ram_resident_gb=8.0,
        )

    def test_compare_speedup_positive(self):
        from squish.bench.squish_bench import FormatComparison
        c = FormatComparison.compare(self._squizd(), self._gguf())
        self.assertGreater(c.ttft_speedup, 1.0)

    def test_compare_tps_gain_positive(self):
        from squish.bench.squish_bench import FormatComparison
        c = FormatComparison.compare(self._squizd(), self._gguf())
        self.assertGreater(c.tps_gain, 0.0)

    def test_compare_disk_ratio_less_than_one(self):
        from squish.bench.squish_bench import FormatComparison
        c = FormatComparison.compare(self._squizd(), self._gguf())
        self.assertLess(c.disk_ratio, 1.0)

    def test_compare_to_gguf_matches_models(self):
        from squish.bench.squish_bench import SquizdBenchmark
        bench = SquizdBenchmark()
        squizds = [self._squizd("Qwen2.5-7B"), self._squizd("Llama-3.1-8B")]
        ggufs = [self._gguf("Qwen2.5-7B"), self._gguf("Llama-3.1-8B")]
        comps = bench.compare_to_gguf(squizds, ggufs)
        self.assertEqual(len(comps), 2)

    def test_compare_to_gguf_skips_missing_baseline(self):
        from squish.bench.squish_bench import SquizdBenchmark
        bench = SquizdBenchmark()
        squizds = [self._squizd("ModelA"), self._squizd("ModelB")]
        ggufs = [self._gguf("ModelA")]
        comps = bench.compare_to_gguf(squizds, ggufs)
        self.assertEqual(len(comps), 1)

    def test_comparison_markdown_contains_header(self):
        from squish.bench.squish_bench import FormatComparison, SquizdBenchmark
        bench = SquizdBenchmark()
        comps = [FormatComparison.compare(self._squizd(), self._gguf())]
        md = bench.comparison_to_markdown(comps)
        self.assertIn("TTFT Speedup", md)

    def test_speedup_zero_ttft_safe(self):
        """TTFT speedup should not crash when squizd ttft_p50 is 0."""
        from squish.bench.squish_bench import FormatComparison, SquizdFormatVariant, SquizdModelResult
        zero_ttft = SquizdModelResult(
            model="M", variant=SquizdFormatVariant.INT4,
            ttft_p50_ms=0.0, ttft_p95_ms=0.0, ttft_p99_ms=0.0,
            tps_p50=50.0, tps_p95=45.0, tps_p99=40.0,
            peak_memory_gb=0.0, disk_size_gb=0.0, ram_resident_gb=0.0,
            n_trials=5,
        )
        c = FormatComparison.compare(zero_ttft, self._gguf())
        self.assertEqual(c.ttft_speedup, 0.0)


# =============================================================================
# 6. HardwareCapabilities
# =============================================================================

class TestHardwareCapabilities(unittest.TestCase):
    def test_m3_has_astc(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M3)
        self.assertTrue(caps.has_astc_texture_sampling)

    def test_m3_has_ane(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M3)
        self.assertTrue(caps.has_ane)

    def test_m3_has_metal3(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M3)
        self.assertTrue(caps.has_metal3)

    def test_m1_no_metal3(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M1)
        self.assertFalse(caps.has_metal3)

    def test_m1_no_mxfp4(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M1)
        self.assertFalse(caps.has_mxfp4)

    def test_m5_has_mxfp4(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M5)
        self.assertTrue(caps.has_mxfp4)

    def test_unknown_no_ane(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.UNKNOWN)
        self.assertFalse(caps.has_ane)

    def test_to_dict_roundtrip(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M3)
        d = caps.to_dict()
        restored = HardwareCapabilities.from_dict(d)
        self.assertEqual(caps, restored)

    def test_to_dict_json_serialisable(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M4)
        _ = json.dumps(caps.to_dict())  # must not raise

    def test_ane_budget_m5(self):
        from squish.hardware.capability_probe import HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        caps = HardwareCapabilities._from_generation(AppleChipGeneration.M5)
        self.assertEqual(caps.ane_memory_budget_gb, 24.0)


# =============================================================================
# 7. CapabilityProbe cache
# =============================================================================

class TestCapabilityProbeCache(unittest.TestCase):
    def test_load_cache_missing_returns_none(self):
        from squish.hardware.capability_probe import CapabilityProbe
        with tempfile.TemporaryDirectory() as td:
            probe = CapabilityProbe(Path(td) / "nonexistent.json")
            self.assertIsNone(probe.load_cache())

    def test_cache_round_trip(self):
        from squish.hardware.capability_probe import CapabilityProbe, HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "caps.json"
            probe = CapabilityProbe(p)
            caps = HardwareCapabilities._from_generation(AppleChipGeneration.M4)
            probe.cache(caps)
            loaded = probe.load_cache()
            self.assertEqual(loaded, caps)

    def test_cache_creates_directory(self):
        from squish.hardware.capability_probe import CapabilityProbe, HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        with tempfile.TemporaryDirectory() as td:
            cache_path = Path(td) / "subdir" / "caps.json"
            probe = CapabilityProbe(cache_path)
            caps = HardwareCapabilities._from_generation(AppleChipGeneration.M3)
            probe.cache(caps)
            self.assertTrue(cache_path.exists())

    def test_invalidate_removes_file(self):
        from squish.hardware.capability_probe import CapabilityProbe, HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "caps.json"
            probe = CapabilityProbe(p)
            caps = HardwareCapabilities._from_generation(AppleChipGeneration.M2)
            probe.cache(caps)
            probe.invalidate_cache()
            self.assertFalse(p.exists())

    def test_load_cache_corrupt_json_returns_none(self):
        from squish.hardware.capability_probe import CapabilityProbe
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "caps.json"
            p.write_text("{not valid json", encoding="utf-8")
            probe = CapabilityProbe(p)
            self.assertIsNone(probe.load_cache())

    def test_force_refresh_bypasses_cache(self):
        """probe(force_refresh=True) should not return the stale cached value."""
        from squish.hardware.capability_probe import CapabilityProbe, HardwareCapabilities
        from squish.hardware.chip_detector import AppleChipGeneration
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "caps.json"
            probe = CapabilityProbe(p)
            # Prime the cache with M1 data.
            m1 = HardwareCapabilities._from_generation(AppleChipGeneration.M1)
            probe.cache(m1)
            # force_refresh must re-probe (result may differ or not, but must not raise).
            fresh = probe.probe(force_refresh=True)
            self.assertIsInstance(fresh, HardwareCapabilities)


# =============================================================================
# 8. get_capability_probe convenience function
# =============================================================================

class TestGetCapabilityProbe(unittest.TestCase):
    def test_returns_capability_probe_instance(self):
        from squish.hardware.capability_probe import CapabilityProbe, get_capability_probe
        probe = get_capability_probe()
        self.assertIsInstance(probe, CapabilityProbe)

    def test_custom_cache_path(self):
        from squish.hardware.capability_probe import get_capability_probe
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "my_cache.json"
            probe = get_capability_probe(cache_path=p)
            self.assertEqual(probe._cache_path, p)


if __name__ == "__main__":
    unittest.main()
