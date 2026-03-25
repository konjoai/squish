"""tests/test_wave90_startup_lean.py

Wave 90 — Lean Startup Profiler + Core Cleanup

Tests for:
  - StartupTimer context manager: accumulates entries
  - StartupReport.total_ms accumulation
  - StartupReport.to_dict() structure and keys
  - StartupReport.slowest(n) sorted descending
  - measure_import_ms returns 0.0 for already-cached modules
  - measure_import_ms returns 0.0 for non-existent modules (no crash)
  - StartupReport disabled=True → no-op (empty entries)
  - FeatureState dataclass instantiates with sane defaults
  - FeatureState can be mutated
  - BlazingPreset dataclass fields
  - auto_blazing_eligible chip requirements
  - get_preset returns BlazingPreset
  - /v1/startup-profile endpoint shape (no server required: direct module test)
"""
from __future__ import annotations

import os
import sys
import unittest

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestStartupTimer — context manager and accumulation
# ============================================================================

class TestStartupTimer(unittest.TestCase):

    def _make_report(self, enabled=True):
        from squish.serving.startup_profiler import StartupReport
        return StartupReport(enabled=enabled)

    def test_timer_records_entry(self):
        from squish.serving.startup_profiler import StartupTimer, StartupPhase
        report = self._make_report()
        with StartupTimer(report, StartupPhase.MODEL_LOAD, "load model"):
            pass
        assert len(report._entries) == 1

    def test_timer_entry_has_positive_elapsed(self):
        from squish.serving.startup_profiler import StartupTimer, StartupPhase
        import time
        report = self._make_report()
        with StartupTimer(report, StartupPhase.MODEL_LOAD):
            time.sleep(0.001)  # 1 ms
        assert report._entries[0].elapsed_ms > 0

    def test_multiple_timers_accumulate(self):
        from squish.serving.startup_profiler import StartupTimer, StartupPhase
        report = self._make_report()
        for phase in (StartupPhase.IMPORTS, StartupPhase.CONFIG, StartupPhase.HW_DETECT):
            with StartupTimer(report, phase):
                pass
        assert len(report._entries) == 3

    def test_timer_no_op_when_disabled(self):
        from squish.serving.startup_profiler import StartupTimer, StartupPhase
        report = self._make_report(enabled=False)
        with StartupTimer(report, StartupPhase.MODEL_LOAD):
            pass
        assert len(report._entries) == 0

    def test_timer_does_not_suppress_exceptions(self):
        from squish.serving.startup_profiler import StartupTimer, StartupPhase
        report = self._make_report()
        with self.assertRaises(ValueError):
            with StartupTimer(report, StartupPhase.CONFIG, "raises"):
                raise ValueError("test error")


# ============================================================================
# TestStartupReport — total_ms, slowest, to_dict
# ============================================================================

class TestStartupReport(unittest.TestCase):

    def _filled_report(self):
        from squish.serving.startup_profiler import StartupReport, StartupTimer, StartupPhase
        import time
        report = StartupReport()
        for phase, label in [
            (StartupPhase.IMPORTS,   "imports-fast"),
            (StartupPhase.MODEL_LOAD, "model-load-slow"),
            (StartupPhase.HTTP_BIND,  "http-bind-medium"),
        ]:
            with StartupTimer(report, phase, label):
                time.sleep(0.002)  # small sleep to get measurable elapsed
        return report

    def test_total_ms_positive(self):
        report = self._filled_report()
        assert report.total_ms > 0

    def test_total_ms_is_sum(self):
        report = self._filled_report()
        expected = sum(e.elapsed_ms for e in report._entries)
        assert abs(report.total_ms - expected) < 1e-6

    def test_slowest_returns_n_entries(self):
        report = self._filled_report()
        result = report.slowest(2)
        assert len(result) == 2

    def test_slowest_sorted_descending(self):
        report = self._filled_report()
        result = report.slowest(3)
        times = [e.elapsed_ms for e in result]
        assert times == sorted(times, reverse=True)

    def test_to_dict_has_required_keys(self):
        report = self._filled_report()
        d = report.to_dict()
        for key in ("enabled", "total_ms", "phase_count", "entries", "slowest_5"):
            assert key in d, f"Missing key {key!r} in to_dict()"

    def test_to_dict_phase_count_correct(self):
        report = self._filled_report()
        d = report.to_dict()
        assert d["phase_count"] == 3

    def test_to_dict_entries_have_fields(self):
        report = self._filled_report()
        for entry in report.to_dict()["entries"]:
            for key in ("phase", "label", "elapsed_ms"):
                assert key in entry, f"Entry missing key {key!r}"

    def test_empty_report_total_ms_zero(self):
        from squish.serving.startup_profiler import StartupReport
        r = StartupReport()
        assert r.total_ms == 0.0


# ============================================================================
# TestMeasureImportMs
# ============================================================================

class TestMeasureImportMs(unittest.TestCase):

    def test_returns_zero_for_already_imported(self):
        from squish.serving.startup_profiler import measure_import_ms
        import sys
        import os  # noqa: F401 — already imported at top
        assert measure_import_ms("os") == 0.0
        assert measure_import_ms("sys") == 0.0

    def test_returns_zero_for_nonexistent_module(self):
        from squish.serving.startup_profiler import measure_import_ms
        # Non-existent module should return 0.0, not raise
        result = measure_import_ms("squish._totally_nonexistent_module_xyz")
        assert result == 0.0

    def test_returns_float(self):
        from squish.serving.startup_profiler import measure_import_ms
        result = measure_import_ms("os")
        assert isinstance(result, float)


# ============================================================================
# TestFeatureState — dataclass defaults and mutation
# ============================================================================

class TestFeatureState(unittest.TestCase):

    def test_instantiates_with_defaults(self):
        from squish.serving.feature_state import FeatureState
        fs = FeatureState()
        assert fs.model is None
        assert fs.tokenizer is None
        assert fs.model_loaded is False
        assert isinstance(fs.model_load_ms, float)

    def test_mutable_fields(self):
        from squish.serving.feature_state import FeatureState
        fs = FeatureState()
        fs.model_id = "qwen3:8b"
        fs.model_loaded = True
        fs.model_load_ms = 1234.5
        assert fs.model_id == "qwen3:8b"
        assert fs.model_loaded is True
        assert fs.model_load_ms == 1234.5

    def test_has_generation_fields(self):
        from squish.serving.feature_state import FeatureState
        fs = FeatureState()
        assert hasattr(fs, "max_tokens_default")
        assert hasattr(fs, "context_window")
        assert hasattr(fs, "temperature")

    def test_has_quantization_fields(self):
        from squish.serving.feature_state import FeatureState
        fs = FeatureState()
        assert hasattr(fs, "quant_bits")
        assert hasattr(fs, "is_blazing")

    def test_singleton_importable(self):
        from squish.serving.feature_state import _state, FeatureState
        assert isinstance(_state, FeatureState)


# ============================================================================
# TestBlazingHelpers — blazing.py
# ============================================================================

class TestBlazingHelpers(unittest.TestCase):

    def test_chip_families_blazing_contains_m3_m4_m5(self):
        from squish.serving.blazing import CHIP_FAMILIES_BLAZING
        assert "m3" in CHIP_FAMILIES_BLAZING
        assert "m4" in CHIP_FAMILIES_BLAZING
        assert "m5" in CHIP_FAMILIES_BLAZING

    def test_auto_blazing_m3_16gb_eligible(self):
        from squish.serving.blazing import auto_blazing_eligible
        assert auto_blazing_eligible("Apple M3 Pro", 16.0) is True

    def test_auto_blazing_m1_not_eligible(self):
        from squish.serving.blazing import auto_blazing_eligible
        assert auto_blazing_eligible("Apple M1", 16.0) is False

    def test_auto_blazing_m3_insufficient_ram(self):
        from squish.serving.blazing import auto_blazing_eligible
        assert auto_blazing_eligible("Apple M3", 8.0) is False

    def test_auto_blazing_m4_eligible(self):
        from squish.serving.blazing import auto_blazing_eligible
        assert auto_blazing_eligible("Apple M4 Max", 32.0) is True

    def test_get_preset_returns_blazing_preset(self):
        from squish.serving.blazing import get_preset, BlazingPreset
        preset = get_preset("Apple M3 Pro", 16.0)
        assert isinstance(preset, BlazingPreset)

    def test_get_preset_high_ram_uses_int4(self):
        from squish.serving.blazing import get_preset
        preset = get_preset("Apple M4 Max", 64.0)
        assert preset.quant_bits == 4

    def test_get_preset_default_fallback(self):
        from squish.serving.blazing import get_preset, BlazingPreset
        preset = get_preset("", 0.0)
        assert isinstance(preset, BlazingPreset)
        assert preset.quant_bits in (2, 4)

    def test_blazing_preset_has_all_fields(self):
        from squish.serving.blazing import BlazingPreset
        p = BlazingPreset()
        for field in ("quant_bits", "chunk_prefill_size", "max_kv_size",
                      "metal_cache_limit_mb", "fast_gelu", "note"):
            assert hasattr(p, field), f"BlazingPreset missing field {field!r}"


# ============================================================================
# TestStartupProfileEndpoint — /v1/startup-profile via module (no server)
# ============================================================================

class TestStartupProfileEndpoint(unittest.TestCase):
    """Test the startup_profiler module's to_dict() matches endpoint contract."""

    def test_global_report_to_dict_valid_when_disabled(self):
        from squish.serving.startup_profiler import _global_report
        # When SQUISH_TRACE_STARTUP is not set, report is disabled
        d = _global_report.to_dict()
        assert "enabled" in d
        assert isinstance(d["total_ms"], (int, float))
        assert isinstance(d["entries"], list)

    def test_startup_report_to_dict_full_structure(self):
        from squish.serving.startup_profiler import StartupReport, StartupTimer, StartupPhase
        report = StartupReport(enabled=True)
        with StartupTimer(report, StartupPhase.MODEL_LOAD, "test-model"):
            pass
        d = report.to_dict()
        assert d["enabled"] is True
        assert d["phase_count"] == 1
        assert len(d["entries"]) == 1
        assert len(d["slowest_5"]) == 1

    def test_startup_phase_enum_values(self):
        from squish.serving.startup_profiler import StartupPhase
        # All phases should be accessible
        for phase_name in ("IMPORTS", "CONFIG", "HW_DETECT", "MODEL_LOAD",
                           "KV_CACHE_INIT", "METAL_WARMUP", "HTTP_BIND", "OTHER"):
            assert hasattr(StartupPhase, phase_name), \
                f"StartupPhase missing {phase_name!r}"


if __name__ == "__main__":
    unittest.main()
