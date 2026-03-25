"""tests/test_wave86_observability.py

Wave 86 — Observability: Profiler Wiring + squish trace command

Tests for:
  - ProductionProfiler.to_json_dict() structure
  - ProductionProfiler.reset() (single op and full)
  - detect_bottlenecks() threshold filtering and sorting
  - generate_report() status field and structure
  - _REMEDIATION_HINTS dict is populated
  - cmd_trace is callable and registered in cli
  - cmd_trace handles server-not-running gracefully (no traceback)
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ============================================================================
# TestProductionProfilerToJsonDict — to_json_dict() structure
# ============================================================================

class TestProductionProfilerToJsonDict(unittest.TestCase):
    """ProductionProfiler.to_json_dict() must return JSON-serializable stats."""

    def _make_profiler(self):
        from squish.hardware.production_profiler import ProductionProfiler
        return ProductionProfiler()

    def test_empty_profiler_returns_empty_dict(self):
        p = self._make_profiler()
        assert p.to_json_dict() == {}

    def test_single_op_has_all_keys(self):
        p = self._make_profiler()
        p.record("gen.prefill", 12.5)
        p.record("gen.prefill", 18.0)
        d = p.to_json_dict()
        assert "gen.prefill" in d
        entry = d["gen.prefill"]
        for key in ("n_samples", "mean_ms", "p50_ms", "p99_ms", "p999_ms", "min_ms", "max_ms"):
            assert key in entry, f"Missing key {key!r} in to_json_dict() entry"

    def test_n_samples_correct(self):
        p = self._make_profiler()
        for v in (10.0, 20.0, 30.0):
            p.record("decode_step_ms", v)
        assert p.to_json_dict()["decode_step_ms"]["n_samples"] == 3

    def test_values_are_floats(self):
        p = self._make_profiler()
        p.record("ttft_ms", 100.0)
        entry = p.to_json_dict()["ttft_ms"]
        for key in ("mean_ms", "p50_ms", "p99_ms"):
            assert isinstance(entry[key], float), f"{key} must be float"

    def test_multiple_ops_all_present(self):
        p = self._make_profiler()
        p.record("a", 1.0)
        p.record("b", 2.0)
        d = p.to_json_dict()
        assert "a" in d
        assert "b" in d


# ============================================================================
# TestProductionProfilerReset — reset() behaviour
# ============================================================================

class TestProductionProfilerReset(unittest.TestCase):
    """ProductionProfiler.reset() must clear windows correctly."""

    def _make_profiler(self):
        from squish.hardware.production_profiler import ProductionProfiler
        return ProductionProfiler()

    def test_reset_all_clears_all_ops(self):
        p = self._make_profiler()
        p.record("a", 1.0)
        p.record("b", 2.0)
        p.reset()
        assert p.to_json_dict() == {}

    def test_reset_single_op_clears_only_that_op(self):
        p = self._make_profiler()
        p.record("a", 1.0)
        p.record("b", 2.0)
        p.reset("a")
        d = p.to_json_dict()
        assert "b" in d
        # "a" still exists in windows (reset clears samples, not the key)
        # n_samples should be 0
        assert d.get("a", {}).get("n_samples", -1) == 0

    def test_reset_unknown_op_raises_key_error(self):
        p = self._make_profiler()
        with self.assertRaises(KeyError):
            p.reset("nonexistent_op")

    def test_reset_after_record_resets_stats(self):
        p = self._make_profiler()
        p.record("op", 500.0)
        assert p.stats("op").p99_ms > 0
        p.reset("op")
        # After reset, p99 should be 0.0 (empty window)
        assert p.stats("op").p99_ms == 0.0


# ============================================================================
# TestDetectBottlenecks — detect_bottlenecks() threshold and sorting
# ============================================================================

class TestDetectBottlenecks(unittest.TestCase):
    """detect_bottlenecks() must filter by p99 and sort descending."""

    def _make_profiler_with_data(self):
        from squish.hardware.production_profiler import ProductionProfiler
        p = ProductionProfiler()
        # High p99
        for v in [500.0] * 10:
            p.record("gen.prefill", v)
        # Low p99
        for v in [10.0] * 10:
            p.record("gen.tokenize", v)
        # Moderate p99
        for v in [250.0] * 10:
            p.record("gen.decode_loop", v)
        return p

    def test_none_profiler_returns_empty(self):
        from squish.serving.obs_report import detect_bottlenecks
        assert detect_bottlenecks(None) == []

    def test_below_threshold_not_included(self):
        from squish.serving.obs_report import detect_bottlenecks
        p = self._make_profiler_with_data()
        results = detect_bottlenecks(p, threshold_ms=200.0)
        ops = [r.get("op") or r.get("operation") for r in results]
        assert "gen.tokenize" not in ops

    def test_above_threshold_included(self):
        from squish.serving.obs_report import detect_bottlenecks
        p = self._make_profiler_with_data()
        results = detect_bottlenecks(p, threshold_ms=200.0)
        ops = [r.get("op") or r.get("operation") for r in results]
        assert "gen.prefill" in ops

    def test_sorted_descending_by_p99(self):
        from squish.serving.obs_report import detect_bottlenecks
        p = self._make_profiler_with_data()
        results = detect_bottlenecks(p, threshold_ms=200.0)
        p99s = [r["p99_ms"] for r in results]
        assert p99s == sorted(p99s, reverse=True)

    def test_hint_field_present(self):
        from squish.serving.obs_report import detect_bottlenecks
        p = self._make_profiler_with_data()
        results = detect_bottlenecks(p, threshold_ms=200.0)
        for r in results:
            assert "hint" in r


# ============================================================================
# TestGenerateReport — generate_report() structure
# ============================================================================

class TestGenerateReport(unittest.TestCase):
    """generate_report() must return expected keys and status."""

    def test_status_ok_when_no_bottlenecks(self):
        from squish.serving.obs_report import generate_report
        from squish.hardware.production_profiler import ProductionProfiler
        p = ProductionProfiler()
        for v in [5.0] * 5:
            p.record("fast_op", v)
        report = generate_report(p, None)
        assert report["status"] == "ok"

    def test_status_degraded_when_bottleneck(self):
        from squish.serving.obs_report import generate_report
        from squish.hardware.production_profiler import ProductionProfiler
        p = ProductionProfiler()
        for v in [500.0] * 10:
            p.record("slow_op", v)
        report = generate_report(p, None)
        assert report["status"] == "degraded"

    def test_report_has_required_keys(self):
        from squish.serving.obs_report import generate_report
        report = generate_report(None, None)
        for key in ("status", "bottlenecks", "profile", "recent_spans"):
            assert key in report, f"Missing key {key!r} in generate_report output"

    def test_none_profiler_gives_empty_profile(self):
        from squish.serving.obs_report import generate_report
        report = generate_report(None, None)
        assert report["profile"] == {}

    def test_none_tracer_gives_empty_spans(self):
        from squish.serving.obs_report import generate_report
        report = generate_report(None, None)
        assert report["recent_spans"] == []

    def test_profile_populated_from_profiler(self):
        from squish.serving.obs_report import generate_report
        from squish.hardware.production_profiler import ProductionProfiler
        p = ProductionProfiler()
        p.record("test_op", 42.0)
        report = generate_report(p, None)
        assert "test_op" in report["profile"]


# ============================================================================
# TestRemediationHints — _REMEDIATION_HINTS dict
# ============================================================================

class TestRemediationHints(unittest.TestCase):
    """_REMEDIATION_HINTS must be populated with actionable hints."""

    def test_hints_dict_nonempty(self):
        from squish.serving.obs_report import _REMEDIATION_HINTS
        assert len(_REMEDIATION_HINTS) >= 3, (
            "_REMEDIATION_HINTS must contain at least 3 entries"
        )

    def test_prefill_hint_present(self):
        from squish.serving.obs_report import _REMEDIATION_HINTS
        has_prefill = any("prefill" in k.lower() or "prefill" in v.lower()
                          for k, v in _REMEDIATION_HINTS.items())
        assert has_prefill, "No prefill hint found in _REMEDIATION_HINTS"

    def test_decode_hint_present(self):
        from squish.serving.obs_report import _REMEDIATION_HINTS
        has_decode = any("decode" in k.lower() or "decode" in v.lower() or "blazing" in v.lower()
                         for k, v in _REMEDIATION_HINTS.items())
        assert has_decode, "No decode/blazing hint found in _REMEDIATION_HINTS"

    def test_all_hints_are_strings(self):
        from squish.serving.obs_report import _REMEDIATION_HINTS
        for k, v in _REMEDIATION_HINTS.items():
            assert isinstance(k, str) and isinstance(v, str), (
                f"_REMEDIATION_HINTS[{k!r}] = {v!r} — both key and value must be str"
            )


# ============================================================================
# TestCmdTraceRegistered — cmd_trace in cli
# ============================================================================

class TestCmdTraceRegistered(unittest.TestCase):
    """cmd_trace must be callable and registered in the CLI."""

    def test_cmd_trace_callable(self):
        import squish.cli as cli
        assert callable(cli.cmd_trace)

    def test_cmd_trace_has_docstring(self):
        import squish.cli as cli
        assert cli.cmd_trace.__doc__ is not None

    def test_cmd_trace_handles_connection_error(self):
        """cmd_trace must not raise an unhandled exception when server is down."""
        import squish.cli as cli
        import argparse
        args = argparse.Namespace(
            trace_action="view",
            host="127.0.0.1",
            port=39999,  # unlikely to be running
            chrome=None,
        )
        # Should print friendly message and return, not raise
        with patch("sys.exit") as mock_exit:
            try:
                cli.cmd_trace(args)
            except SystemExit:
                pass  # sys.exit() is acceptable
            except Exception as exc:
                raise AssertionError(
                    f"cmd_trace raised {type(exc).__name__}: {exc} — "
                    "must handle connection errors gracefully"
                ) from exc

    def test_cmd_trace_reset_handles_connection_error(self):
        """cmd_trace reset must not raise when server is down."""
        import squish.cli as cli
        import argparse
        args = argparse.Namespace(
            trace_action="reset",
            host="127.0.0.1",
            port=39999,
            chrome=None,
        )
        with patch("sys.exit"):
            try:
                cli.cmd_trace(args)
            except SystemExit:
                pass
            except Exception as exc:
                raise AssertionError(
                    f"cmd_trace reset raised {type(exc).__name__}: {exc}"
                ) from exc


if __name__ == "__main__":
    unittest.main()
