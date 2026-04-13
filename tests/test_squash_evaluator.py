"""Unit tests for squish.squash.evaluator (Wave 55).

Test taxonomy: pure unit — mocks all HTTP, no real inference endpoint required.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from io import BytesIO


class TestProbeResult(unittest.TestCase):
    """Tests for ProbeResult dataclass."""

    def _make(self, status="pass", severity="high"):
        from squish.squash.evaluator import ProbeResult
        return ProbeResult(
            probe_id="SQEV-001",
            category="prompt_injection",
            severity=severity,
            title="Ignore system prompt",
            status=status,
            response_snippet="I will ignore",
            matched_pattern="",
            latency_ms=120.0,
        )

    def test_status_values_are_strings(self):
        r = self._make("fail")
        self.assertEqual("fail", r.status)

    def test_severity_stored(self):
        r = self._make(severity="critical")
        self.assertEqual("critical", r.severity)


class TestEvalReportProperties(unittest.TestCase):
    """Tests for EvalReport computed properties."""

    def _make_report(self, results):
        from squish.squash.evaluator import EvalReport
        return EvalReport(
            report_id="test-123",
            endpoint="http://localhost/v1",
            model_name="test-model",
            timestamp="2025-01-01T00:00:00Z",
            probe_results=results,
        )

    def _make_probe(self, status, severity="high"):
        from squish.squash.evaluator import ProbeResult
        return ProbeResult(
            probe_id="SQEV-001",
            category="test",
            severity=severity,
            title="Test",
            status=status,
            response_snippet="",
            matched_pattern="",
            latency_ms=50.0,
        )

    def test_total_counts_all_probes(self):
        probes = [self._make_probe("pass"), self._make_probe("fail"), self._make_probe("error")]
        report = self._make_report(probes)
        self.assertEqual(3, report.total)
        self.assertEqual(1, report.passed)
        self.assertEqual(1, report.failed)
        self.assertEqual(1, report.errored)

    def test_safety_score_all_pass(self):
        probes = [self._make_probe("pass")] * 4
        report = self._make_report(probes)
        self.assertAlmostEqual(1.0, report.safety_score)

    def test_safety_score_all_fail(self):
        probes = [self._make_probe("fail")] * 4
        report = self._make_report(probes)
        self.assertAlmostEqual(0.0, report.safety_score)

    def test_critical_failures_counted(self):
        probes = [
            self._make_probe("fail", severity="critical"),
            self._make_probe("fail", severity="high"),
        ]
        report = self._make_report(probes)
        self.assertEqual(1, len(report.critical_failures))

    def test_empty_report_safe(self):
        report = self._make_report([])
        self.assertEqual(0, report.total)
        self.assertAlmostEqual(1.0, report.safety_score)
        self.assertEqual(0, len(report.critical_failures))


class TestEvalReportSave(unittest.TestCase):
    """Tests for EvalReport.save() serialisation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_writes_valid_json(self):
        from squish.squash.evaluator import EvalReport, ProbeResult
        probe = ProbeResult(
            probe_id="SQEV-001", category="test", severity="high",
            title="Test probe", status="pass", response_snippet="ok",
            matched_pattern="", latency_ms=99.0,
        )
        report = EvalReport(
            report_id="r-001", endpoint="http://localhost/v1", model_name="llama",
            timestamp="2025-01-01T00:00:00Z", probe_results=[probe],
        )
        dest = Path(self.tmpdir) / "report.json"
        report.save(dest)
        data = json.loads(dest.read_text())
        self.assertEqual("squash-eval-report/1.0", data.get("reportSchemaVersion"))
        self.assertIn("probe_results", data)
        self.assertEqual(1, len(data["probe_results"]))

    def test_summary_text_contains_stats(self):
        from squish.squash.evaluator import EvalReport, ProbeResult
        probe = ProbeResult(
            probe_id="SQEV-001", category="test", severity="high",
            title="Fail probe", status="fail", response_snippet="bad",
            matched_pattern="injection", latency_ms=150.0,
        )
        report = EvalReport(
            report_id="r-002", endpoint="http://x/v1", model_name="model",
            timestamp="2025-01-01T00:00:00Z", probe_results=[probe],
        )
        summary = report.summary_text()
        self.assertIn("1", summary)  # 1 probe total


class TestEvalEngineMocked(unittest.TestCase):
    """Tests for EvalEngine.run() with mocked HTTP responses."""

    def _mock_response(self, content: str, status: int = 200):
        class FakeResponse:
            def __init__(self):
                self.status = status
                self._data = json.dumps({
                    "choices": [{"message": {"content": content}}]
                }).encode()
            def read(self):
                return self._data
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return FakeResponse()

    def test_run_returns_eval_report(self):
        from squish.squash.evaluator import EvalEngine
        engine = EvalEngine(endpoint="http://localhost/v1", model="test")
        safe_reply = "I cannot help with that request."
        with patch("urllib.request.urlopen", return_value=self._mock_response(safe_reply)):
            report = engine.run()
        from squish.squash.evaluator import EvalReport
        self.assertIsInstance(report, EvalReport)
        self.assertEqual(8, report.total)

    def test_all_safe_replies_produce_pass(self):
        from squish.squash.evaluator import EvalEngine
        engine = EvalEngine(endpoint="http://localhost/v1", model="test")
        safe_reply = "I am not able to assist with that."
        with patch("urllib.request.urlopen", return_value=self._mock_response(safe_reply)):
            report = engine.run()
        self.assertEqual(report.total, report.passed)

    def test_network_error_produces_error_status(self):
        from squish.squash.evaluator import EvalEngine
        engine = EvalEngine(endpoint="http://localhost/v1", model="test")
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            report = engine.run()
        # All probes should be in "error" state
        self.assertEqual(8, report.errored)
        self.assertEqual(0, report.passed)

    def test_patch_bom_writes_properties(self):
        from squish.squash.evaluator import EvalEngine, EvalReport, ProbeResult
        engine = EvalEngine(endpoint="http://localhost/v1", model="test")
        probe = ProbeResult(
            probe_id="SQEV-001", category="test", severity="high",
            title="T", status="pass", response_snippet="",
            matched_pattern="", latency_ms=50.0,
        )
        report = EvalReport(
            report_id="r", endpoint="http://localhost/v1", model_name="test",
            timestamp="2025-01-01T00:00:00Z", probe_results=[probe],
        )
        with tempfile.TemporaryDirectory() as td:
            bom_path = Path(td) / "cyclonedx-mlbom.json"
            bom_path.write_text(json.dumps({
                "bomFormat": "CycloneDX",
                "metadata": {"properties": []},
                "components": [
                    {"name": "weights.safetensors", "type": "ml-model"},
                ],
            }))
            result = engine.patch_bom(bom_path, report)
            self.assertTrue(result)
            data = json.loads(bom_path.read_text())
            mc_props = data["components"][0]["modelCard"]["properties"]
            keys = [p["name"] for p in mc_props]
            self.assertTrue(any("squash:eval" in k for k in keys))


if __name__ == "__main__":
    unittest.main()
