"""Tests for Wave 11 — SarifBuilder and /scan/{job_id}/sarif API endpoint."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from squish.squash.sarif import SarifBuilder
from squish.squash.scanner import ScanFinding, ScanResult


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_scan_result(findings: list[ScanFinding] | None = None) -> ScanResult:
    return ScanResult(
        scanned_path="/tmp/model",
        status="clean" if not findings else "unsafe",
        findings=findings or [],
        scanner_version="test-1.0",
    )


def _make_finding(severity: str = "high", cve: str = "CVE-2024-0001") -> ScanFinding:
    return ScanFinding(
        severity=severity,
        finding_id="TEST001",
        title=f"Vulnerable component ({cve})",
        detail="Package has known vulnerability.",
        file_path="/tmp/model/weights.pkl",
        cve=cve,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Unit tests — SarifBuilder
# ──────────────────────────────────────────────────────────────────────────────

class TestSarifBuilderStructure:
    def test_returns_valid_sarif_schema_keys(self):
        result = _make_scan_result()
        sarif = SarifBuilder.from_scan_result(result)
        assert sarif["version"] == "2.1.0"
        assert "$schema" in sarif
        assert "runs" in sarif
        assert len(sarif["runs"]) == 1

    def test_runs_has_tool_and_results(self):
        result = _make_scan_result([_make_finding()])
        sarif = SarifBuilder.from_scan_result(result)
        run = sarif["runs"][0]
        assert "tool" in run
        assert "results" in run
        assert run["tool"]["driver"]["name"] == "squash"

    def test_empty_findings_produces_valid_sarif(self):
        result = _make_scan_result([])
        sarif = SarifBuilder.from_scan_result(result)
        run = sarif["runs"][0]
        assert run["results"] == []

    def test_rules_populated_for_each_unique_rule_id(self):
        findings = [
            _make_finding("high", "CVE-2024-0001"),
            ScanFinding(
                severity="medium",
                finding_id="TEST002",
                title="Another finding",
                detail="detail",
                file_path="/tmp/model/config.json",
                cve="CVE-2024-0002",
            ),
        ]
        result = _make_scan_result(findings)
        sarif = SarifBuilder.from_scan_result(result)
        driver = sarif["runs"][0]["tool"]["driver"]
        rule_ids = {r["id"] for r in driver["rules"]}
        assert "TEST001" in rule_ids
        assert "TEST002" in rule_ids


class TestSarifSeverityMapping:
    @pytest.mark.parametrize("severity,expected", [
        ("critical", "error"),
        ("high",     "error"),
        ("medium",   "warning"),
        ("low",      "note"),
        ("info",     "note"),
    ])
    def test_severity_map(self, severity: str, expected: str):
        finding = ScanFinding(
            severity=severity,
            finding_id="MAP001",
            title="test",
            detail="test detail",
            file_path="/tmp/model",
            cve="CVE-2024-9999",
        )
        result = _make_scan_result([finding])
        sarif = SarifBuilder.from_scan_result(result)
        sarif_result = sarif["runs"][0]["results"][0]
        assert sarif_result["level"] == expected


class TestSarifWrite:
    def test_write_creates_file(self):
        result = _make_scan_result([_make_finding()])
        with tempfile.NamedTemporaryFile(suffix=".sarif", delete=False) as fh:
            path = Path(fh.name)
        try:
            SarifBuilder.write(result, path)
            assert path.exists()
            with open(path) as fh:
                data = json.load(fh)
            assert data["version"] == "2.1.0"
        finally:
            path.unlink(missing_ok=True)

    def test_write_creates_parent_dir(self):
        result = _make_scan_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "out.sarif"
            path.parent.mkdir(parents=True, exist_ok=True)
            SarifBuilder.write(result, path)
            assert path.exists()


class TestSarifFromPayload:
    def test_from_payload_returns_valid_sarif(self):
        payload = {
            "scanned_path": "/tmp/model",
            "scanner_version": "1.0",
            "findings": [
                {"id": "P001", "severity": "high", "title": "test",
                 "detail": "details", "file": "/tmp/model", "cve": "CVE-2024-0001"},
            ],
        }
        sarif = SarifBuilder.from_payload(payload)
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) == 1

    def test_from_payload_empty_findings(self):
        payload = {"scanned_path": "/tmp/m", "scanner_version": "0.1", "findings": []}
        sarif = SarifBuilder.from_payload(payload)
        assert sarif["runs"][0]["results"] == []


# ──────────────────────────────────────────────────────────────────────────────
# API endpoint — /scan/{job_id}/sarif
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def api_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    # Ensure no auth token bleeds from environment
    os.environ.pop("SQUASH_API_TOKEN", None)
    from squish.squash.api import app
    return TestClient(app)


class TestSarifApiEndpoint:
    def test_unknown_job_returns_404(self, api_client):
        resp = api_client.get("/scan/does-not-exist/sarif")
        assert resp.status_code == 404

    def test_pending_job_returns_202(self, api_client):
        from squish.squash.api import _scan_jobs
        _scan_jobs["pending-job-sarif"] = {"status": "pending", "result": None}
        resp = api_client.get("/scan/pending-job-sarif/sarif")
        assert resp.status_code == 202
        del _scan_jobs["pending-job-sarif"]

    def test_error_job_returns_400(self, api_client):
        from squish.squash.api import _scan_jobs
        _scan_jobs["err-job-sarif"] = {"status": "error", "result": {"error": "boom"}}
        resp = api_client.get("/scan/err-job-sarif/sarif")
        assert resp.status_code == 400
        del _scan_jobs["err-job-sarif"]

    def test_done_job_returns_sarif_200(self, api_client):
        from squish.squash.api import _scan_jobs
        _scan_jobs["done-job-sarif"] = {
            "status": "done",
            "result": {
                "scanned_path": "/tmp/m",
                "scanner_version": "1.0",
                "findings": [],
            },
        }
        resp = api_client.get("/scan/done-job-sarif/sarif")
        assert resp.status_code == 200
        body = resp.json()
        assert body["version"] == "2.1.0"
        assert "runs" in body
        del _scan_jobs["done-job-sarif"]


