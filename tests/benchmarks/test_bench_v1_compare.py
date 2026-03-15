"""
tests/benchmarks/test_bench_v1_compare.py

Tests for the v1 vs v9 comparison benchmark infrastructure:
- dev/results/v1_baseline.json structure and content
- dev/benchmarks/bench_v9_vs_v1.py import and API
- Comparison table generation logic
- JSON output schema validation
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "dev" / "results"
BENCHES_DIR = REPO_ROOT / "dev" / "benchmarks"
V1_BASELINE = RESULTS_DIR / "v1_baseline.json"

# Make dev/benchmarks importable
sys.path.insert(0, str(BENCHES_DIR))


# ── v1_baseline.json structure ────────────────────────────────────────────────

class TestV1BaselineFile:
    """v1_baseline.json must exist and have the required schema."""

    def test_file_exists(self):
        assert V1_BASELINE.exists(), (
            f"v1_baseline.json not found at {V1_BASELINE}; "
            "run: python dev/benchmarks/bench_v9_vs_v1.py --init-baseline"
        )

    def test_valid_json(self):
        data = json.loads(V1_BASELINE.read_text())
        assert isinstance(data, dict)

    def test_has_meta_section(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "_meta" in data
        assert "version" in data["_meta"]

    def test_has_load_time_section(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "load_time" in data
        lt = data["load_time"]
        assert "model_1_5b" in lt
        assert "model_7b" in lt
        assert "model_14b" in lt

    def test_load_time_1_5b_values(self):
        data = json.loads(V1_BASELINE.read_text())
        m = data["load_time"]["model_1_5b"]
        # 0.33–0.53 s from RESULTS.md
        assert 0.1 <= m["squish_cached_s"] <= 1.0
        assert m["cold_mlxlm_s"] > 10.0  # 28.81 s
        assert m["speedup_vs_cold"] > 40   # 54×

    def test_load_time_7b_has_stdev(self):
        data = json.loads(V1_BASELINE.read_text())
        m = data["load_time"]["model_7b"]
        assert "squish_cached_stdev_s" in m

    def test_has_ram_section(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "ram" in data
        assert "model_1_5b" in data["ram"]

    def test_ram_reduction_factor(self):
        data = json.loads(V1_BASELINE.read_text())
        ram = data["ram"]["model_1_5b"]
        assert ram["ram_reduction_factor"] >= 10  # claimed 15×

    def test_has_throughput_section(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "throughput" in data
        t = data["throughput"]
        assert "model_1_5b_tier1" in t
        assert "model_7b_tier0" in t
        assert "model_14b_tier0" in t

    def test_throughput_values_plausible(self):
        data = json.loads(V1_BASELINE.read_text())
        t = data["throughput"]
        # 1.5B: 18.9 tok/s, 7B: 14.3, 14B: 7.7
        assert 10.0 <= t["model_1_5b_tier1"]["tok_s"] <= 40.0
        assert 5.0 <= t["model_7b_tier0"]["tok_s"] <= 30.0
        assert 3.0 <= t["model_14b_tier0"]["tok_s"] <= 20.0
        # Larger models must be slower than smaller ones
        assert t["model_7b_tier0"]["tok_s"] < t["model_1_5b_tier1"]["tok_s"]
        assert t["model_14b_tier0"]["tok_s"] < t["model_7b_tier0"]["tok_s"]

    def test_has_accuracy_section(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "accuracy" in data
        acc = data["accuracy"]
        assert "tasks" in acc
        tasks = acc["tasks"]
        for name in ("arc_easy", "hellaswag", "winogrande", "piqa"):
            assert name in tasks, f"Missing task: {name}"

    def test_accuracy_values_match_documented_v1(self):
        """v1 baseline accuracy must match known documented numbers."""
        data = json.loads(V1_BASELINE.read_text())
        tasks = data["accuracy"]["tasks"]
        # From RESULTS.md "Squish v1" column
        assert abs(tasks["arc_easy"]["squish_compressed"] - 0.735) < 0.01
        assert abs(tasks["hellaswag"]["squish_compressed"] - 0.620) < 0.01
        assert abs(tasks["piqa"]["squish_compressed"] - 0.765) < 0.01
        assert abs(tasks["winogrande"]["squish_compressed"] - 0.670) < 0.01

    def test_all_accuracy_tasks_pass_criterion(self):
        data = json.loads(V1_BASELINE.read_text())
        acc = data["accuracy"]
        criterion = acc.get("pass_criterion_pp", 2.0)
        for name, task in acc["tasks"].items():
            assert task["pass"] is True, f"{name} should pass"
            assert abs(task["delta_pp"]) <= criterion, (
                f"{name} delta {task['delta_pp']} exceeds ±{criterion}pp"
            )

    def test_has_latency_section(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "latency" in data
        lat = data["latency"]
        assert "ttft_health_endpoint_ms" in lat

    def test_ttft_health_endpoint_plausible(self):
        data = json.loads(V1_BASELINE.read_text())
        ttft = data["latency"]["ttft_health_endpoint_ms"]
        # 668 ms from eoe_bench.json
        assert 100 <= ttft <= 5000, f"TTFT {ttft}ms outside expected range"

    def test_has_features_v1_list(self):
        data = json.loads(V1_BASELINE.read_text())
        assert "features_v1" in data
        feats = data["features_v1"]
        assert isinstance(feats, list)
        assert len(feats) >= 4, "v1 should document at least 4 features"

    def test_disk_reduction_7b(self):
        data = json.loads(V1_BASELINE.read_text())
        m = data["load_time"]["model_7b"]
        # 4.0 GB squish vs 14.0 GB original = 3.5× reduction
        assert m["disk_squish_gb"] < m["disk_original_gb"]
        assert m["disk_reduction_x"] >= 3.0

    def test_has_version_in_meta(self):
        data = json.loads(V1_BASELINE.read_text())
        version = data["_meta"]["version"]
        assert version.startswith("1."), f"Expected v1.x, got {version}"


# ── bench_v9_vs_v1.py module tests ────────────────────────────────────────────

class TestBenchV9VsV1Module:
    """bench_v9_vs_v1.py must be importable and expose the required API."""

    @pytest.fixture(autouse=True)
    def import_module(self):
        try:
            import bench_v9_vs_v1 as m
            self._m = m
        except ImportError as e:
            pytest.skip(f"bench_v9_vs_v1.py not found or import error: {e}")

    def test_module_has_load_v1_baseline(self):
        assert hasattr(self._m, "load_v1_baseline"), (
            "bench_v9_vs_v1 must expose load_v1_baseline()"
        )

    def test_load_v1_baseline_returns_dict(self):
        result = self._m.load_v1_baseline()
        assert isinstance(result, dict)
        assert "_meta" in result

    def test_module_has_build_comparison(self):
        assert hasattr(self._m, "build_comparison"), (
            "bench_v9_vs_v1 must expose build_comparison()"
        )

    def test_build_comparison_returns_dict(self):
        comp = self._m.build_comparison()
        assert isinstance(comp, dict)

    def test_comparison_has_v1_and_v9_keys(self):
        comp = self._m.build_comparison()
        assert "v1" in comp
        assert "v9" in comp

    def test_module_has_to_markdown(self):
        assert hasattr(self._m, "to_markdown"), (
            "bench_v9_vs_v1 must expose to_markdown(comparison_dict)"
        )

    def test_to_markdown_returns_string(self):
        comp = self._m.build_comparison()
        md = self._m.to_markdown(comp)
        assert isinstance(md, str)
        assert len(md) > 50

    def test_markdown_contains_v1_header(self):
        comp = self._m.build_comparison()
        md = self._m.to_markdown(comp)
        assert "v1" in md.lower() or "1.0" in md

    def test_markdown_contains_load_time(self):
        comp = self._m.build_comparison()
        md = self._m.to_markdown(comp)
        assert "load" in md.lower() or "0.53" in md or "28.81" in md

    def test_markdown_contains_throughput(self):
        comp = self._m.build_comparison()
        md = self._m.to_markdown(comp)
        assert "tok" in md.lower() or "tps" in md.lower()


# ── v9_vs_v1_comparison.json output tests ────────────────────────────────────

class TestComparisonOutputFile:
    """If dev/results/v9_vs_v1_comparison.json exists, validate its schema."""

    COMP_FILE = RESULTS_DIR / "v9_vs_v1_comparison.json"

    @pytest.fixture(autouse=True)
    def skip_if_missing(self):
        if not self.COMP_FILE.exists():
            pytest.skip(
                "v9_vs_v1_comparison.json not yet generated; "
                "run: python dev/benchmarks/bench_v9_vs_v1.py --output dev/results/v9_vs_v1_comparison.json"
            )

    def test_valid_json(self):
        data = json.loads(self.COMP_FILE.read_text())
        assert isinstance(data, dict)

    def test_has_v1_and_v9_sections(self):
        data = json.loads(self.COMP_FILE.read_text())
        assert "v1" in data
        assert "v9" in data

    def test_v1_load_time_present(self):
        data = json.loads(self.COMP_FILE.read_text())
        v1 = data["v1"]
        assert "load_time_1_5b_s" in v1 or "load_time" in v1

    def test_generated_at_field(self):
        data = json.loads(self.COMP_FILE.read_text())
        assert "generated_at" in data or "_meta" in data

    def test_v9_improvement_factors_are_positive(self):
        data = json.loads(self.COMP_FILE.read_text())
        v9 = data.get("v9", {})
        improvements = data.get("improvements", {})
        for key, val in improvements.items():
            if isinstance(val, (int, float)):
                assert val > 0, f"Improvement factor for {key} must be > 0, got {val}"
