"""Wave 15 — ComplianceReporter tests."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def _write_artifact(model_dir: Path, name: str, obj: dict) -> None:
    (model_dir / name).write_text(json.dumps(obj))


class TestComplianceReporterEmpty(unittest.TestCase):
    """HTML output with no artifacts present."""

    def test_generates_html_no_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            from squish.squash.report import ComplianceReporter
            html = ComplianceReporter.generate_html(model_dir)
            self.assertIsInstance(html, str)
            self.assertIn("<!DOCTYPE html>", html)
            self.assertIn("<html", html)

    def test_html_has_required_structure(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.report import ComplianceReporter
            html = ComplianceReporter.generate_html(Path(td))
            self.assertIn("<head>", html)
            self.assertIn("<body>", html)
            self.assertIn("</html>", html)

    def test_missing_dir_handled(self):
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "nonexistent"
            missing.mkdir()
            from squish.squash.report import ComplianceReporter
            # Should not raise — missing artifacts are silently skipped
            html = ComplianceReporter.generate_html(missing)
            self.assertIsInstance(html, str)


class TestComplianceReporterWithArtifacts(unittest.TestCase):
    """HTML output when artifacts are present."""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.model_dir = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_attest_section_present(self):
        _write_artifact(self.model_dir, "squash-attest.json", {
            "model_id": "test-model",
            "passed": True,
            "attested_at": "2025-01-01T00:00:00Z",
        })
        from squish.squash.report import ComplianceReporter
        html = ComplianceReporter.generate_html(self.model_dir)
        # Should include some content from attest record
        self.assertIn("test-model", html)

    def test_scan_section_present(self):
        _write_artifact(self.model_dir, "squash-scan.json", {
            "status": "clean",
            "critical": 0,
            "high": 0,
            "findings": [],
        })
        from squish.squash.report import ComplianceReporter
        html = ComplianceReporter.generate_html(self.model_dir)
        self.assertIn("Scan", html)

    def test_scan_with_findings(self):
        _write_artifact(self.model_dir, "squash-scan.json", {
            "status": "flagged",
            "critical": 1,
            "high": 2,
            "findings": [
                {"severity": "CRITICAL", "id": "CVE-2024-0001", "title": "Test CVE", "file": "model.bin"},
            ],
        })
        from squish.squash.report import ComplianceReporter
        html = ComplianceReporter.generate_html(self.model_dir)
        self.assertIn("CVE-2024-0001", html)

    def test_vex_section_present(self):
        _write_artifact(self.model_dir, "squash-vex-report.json", {
            "affected_count": 1,
            "not_affected_count": 5,
        })
        from squish.squash.report import ComplianceReporter
        html = ComplianceReporter.generate_html(self.model_dir)
        self.assertIn("VEX", html)

    def test_policy_section_present(self):
        _write_artifact(self.model_dir, "squash-policy-no-cvss.json", {
            "name": "no-cvss",
            "passed": True,
            "error_count": 0,
        })
        from squish.squash.report import ComplianceReporter
        html = ComplianceReporter.generate_html(self.model_dir)
        self.assertIn("Policy", html)

    def test_xss_escaping(self):
        """HTML-special characters in artifact data must be escaped."""
        _write_artifact(self.model_dir, "squash-attest.json", {
            "model_id": "<script>alert('xss')</script>",
            "passed": False,
        })
        from squish.squash.report import ComplianceReporter
        html = ComplianceReporter.generate_html(self.model_dir)
        self.assertNotIn("<script>alert", html)
        self.assertIn("&lt;script&gt;", html)


class TestComplianceReporterWrite(unittest.TestCase):
    """write() method creates file on disk."""

    def test_write_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            from squish.squash.report import ComplianceReporter
            out = ComplianceReporter.write(model_dir)
            self.assertIsInstance(out, Path)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 0)

    def test_write_default_filename(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            from squish.squash.report import ComplianceReporter
            out = ComplianceReporter.write(model_dir)
            self.assertEqual(out.name, "squash-report.html")

    def test_write_custom_output_path(self):
        with tempfile.TemporaryDirectory() as td:
            model_dir = Path(td)
            custom = model_dir / "custom-report.html"
            from squish.squash.report import ComplianceReporter
            out = ComplianceReporter.write(model_dir, output_path=custom)
            self.assertEqual(out, custom)
            self.assertTrue(out.exists())

    def test_write_content_is_html(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.report import ComplianceReporter
            out = ComplianceReporter.write(Path(td))
            content = out.read_text()
            self.assertIn("<!DOCTYPE html>", content)


class TestComplianceReporterDtypeContract(unittest.TestCase):
    """Shape / return type contracts."""

    def test_generate_html_returns_str(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.report import ComplianceReporter
            result = ComplianceReporter.generate_html(Path(td))
            self.assertIsInstance(result, str)

    def test_write_returns_path(self):
        with tempfile.TemporaryDirectory() as td:
            from squish.squash.report import ComplianceReporter
            result = ComplianceReporter.write(Path(td))
            self.assertIsInstance(result, Path)


if __name__ == "__main__":
    unittest.main()
