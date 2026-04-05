"""Tests for Wave 12 — SbomDiff, PolicyHistory, squash diff CLI, POST /sbom/diff API."""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from squish.squash.sbom_builder import SbomDiff
from squish.squash.policy import PolicyHistory, PolicyResult, PolicyFinding


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_bom(
    component_hash: str = "abc123deadbeef",
    score: float | None = 0.85,
    policy_status: str = "passed",
    vuln_ids: list[str] | None = None,
) -> dict:
    vuln_ids = vuln_ids or []
    props = [{"name": "squash:policy_result", "value": policy_status}]
    comp: dict = {
        "hashes": [{"alg": "SHA-256", "content": component_hash}],
    }
    if score is not None:
        comp["modelCard"] = {
            "quantitativeAnalysis": {
                "performanceMetrics": [{"type": "arc_easy", "value": score}]
            }
        }
    return {
        "components": [comp],
        "metadata": {"properties": props},
        "vulnerabilities": [{"id": v} for v in vuln_ids],
    }


def _make_policy_result(passed: bool = True) -> PolicyResult:
    return PolicyResult(
        policy_name="nist-ai-rmf",
        passed=passed,
        findings=[
            PolicyFinding(
                rule_id="R01",
                severity="error",
                passed=passed,
                field="metadata.name",
                rationale="test",
                remediation="fix it",
            )
        ],
    )


# ──────────────────────────────────────────────────────────────────────────────
# SbomDiff unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSbomDiffCompare:
    def test_identical_boms_no_changes(self):
        bom = _make_bom()
        diff = SbomDiff.compare(bom, bom)
        assert not diff.hash_changed
        assert diff.score_delta is None or diff.score_delta == 0.0
        assert not diff.policy_status_changed
        assert diff.new_findings == []
        assert diff.resolved_findings == []
        assert not diff.has_regressions

    def test_detects_hash_change(self):
        bom_a = _make_bom(component_hash="abc")
        bom_b = _make_bom(component_hash="xyz")
        diff = SbomDiff.compare(bom_a, bom_b)
        assert diff.hash_changed

    def test_detects_new_findings(self):
        bom_a = _make_bom(vuln_ids=[])
        bom_b = _make_bom(vuln_ids=["CVE-2024-0001"])
        diff = SbomDiff.compare(bom_a, bom_b)
        assert "CVE-2024-0001" in diff.new_findings

    def test_detects_resolved_findings(self):
        bom_a = _make_bom(vuln_ids=["CVE-2024-0001"])
        bom_b = _make_bom(vuln_ids=[])
        diff = SbomDiff.compare(bom_a, bom_b)
        assert "CVE-2024-0001" in diff.resolved_findings

    def test_detects_policy_status_change(self):
        bom_a = _make_bom(policy_status="passed")
        bom_b = _make_bom(policy_status="failed")
        diff = SbomDiff.compare(bom_a, bom_b)
        assert diff.policy_status_changed

    def test_has_regressions_true_when_new_findings(self):
        bom_a = _make_bom(vuln_ids=[])
        bom_b = _make_bom(vuln_ids=["CVE-2024-9999"])
        diff = SbomDiff.compare(bom_a, bom_b)
        assert diff.has_regressions

    def test_has_regressions_false_when_only_resolved(self):
        bom_a = _make_bom(vuln_ids=["CVE-2024-9999"])
        bom_b = _make_bom(vuln_ids=[])
        diff = SbomDiff.compare(bom_a, bom_b)
        assert not diff.has_regressions

    def test_score_delta_computed(self):
        bom_a = _make_bom(score=0.80)
        bom_b = _make_bom(score=0.70)
        diff = SbomDiff.compare(bom_a, bom_b)
        assert diff.score_delta is not None
        assert abs(diff.score_delta - (-0.10)) < 1e-6

    def test_missing_metadata_does_not_raise(self):
        diff = SbomDiff.compare({}, {})
        assert not diff.hash_changed


# ──────────────────────────────────────────────────────────────────────────────
# PolicyHistory unit tests
# ──────────────────────────────────────────────────────────────────────────────

class TestPolicyHistory:
    def test_append_and_latest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "history.ndjson"
            ph = PolicyHistory(log_path)
            result = _make_policy_result(passed=True)
            ph.append(result, "/models/my_model")
            latest = ph.latest("/models/my_model")
            assert latest is not None
            assert latest["model"] == "/models/my_model"
            assert latest["passed"] is True

    def test_latest_returns_none_for_unknown_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ph = PolicyHistory(Path(tmpdir) / "h.ndjson")
            assert ph.latest("/nonexistent") is None

    def test_regressions_since_filters_by_time(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "h.ndjson"
            ph = PolicyHistory(log_path)
            # Append a failing result
            result = _make_policy_result(passed=False)
            ph.append(result, "/models/m1")
            since = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
            regressions = ph.regressions_since(since)
            assert any(r["model"] == "/models/m1" for r in regressions)

    def test_regressions_since_excludes_old_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "h.ndjson"
            ph = PolicyHistory(log_path)
            result = _make_policy_result(passed=False)
            ph.append(result, "/models/m1")
            # Use a future cutoff — nothing should match
            since = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=1)
            regressions = ph.regressions_since(since)
            assert regressions == []

    def test_multiple_appends_latest_returns_most_recent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "h.ndjson"
            ph = PolicyHistory(log_path)
            ph.append(_make_policy_result(passed=True), "/models/m1")
            ph.append(_make_policy_result(passed=False), "/models/m1")
            latest = ph.latest("/models/m1")
            assert latest["passed"] is False


# ──────────────────────────────────────────────────────────────────────────────
# squash diff CLI tests
# ──────────────────────────────────────────────────────────────────────────────

class TestSquashDiffCli:
    def _write_bom(self, path: Path, bom: dict) -> None:
        path.write_text(json.dumps(bom))

    def test_diff_identical_exits_0(self):
        from squish.squash.cli import main
        bom = _make_bom()
        with tempfile.TemporaryDirectory() as tmpdir:
            a = Path(tmpdir) / "a.json"
            b = Path(tmpdir) / "b.json"
            self._write_bom(a, bom)
            self._write_bom(b, bom)
            with patch("sys.argv", ["squash", "diff", str(a), str(b)]):
                try:
                    main()
                    exited = 0
                except SystemExit as e:
                    exited = e.code
        assert exited == 0

    def test_diff_with_regression_and_flag_exits_1(self):
        from squish.squash.cli import main
        bom_a = _make_bom(vuln_ids=[])
        bom_b = _make_bom(vuln_ids=["CVE-2024-REGRESSION"])
        with tempfile.TemporaryDirectory() as tmpdir:
            a = Path(tmpdir) / "a.json"
            b = Path(tmpdir) / "b.json"
            self._write_bom(a, bom_a)
            self._write_bom(b, bom_b)
            with patch("sys.argv", ["squash", "diff", str(a), str(b), "--exit-1-on-regression"]):
                try:
                    main()
                    exited = 0
                except SystemExit as e:
                    exited = e.code
        assert exited == 1

    def test_diff_no_regression_flag_exits_0_even_with_regression(self):
        from squish.squash.cli import main
        bom_a = _make_bom(vuln_ids=[])
        bom_b = _make_bom(vuln_ids=["CVE-2024-REGRESSION"])
        with tempfile.TemporaryDirectory() as tmpdir:
            a = Path(tmpdir) / "a.json"
            b = Path(tmpdir) / "b.json"
            self._write_bom(a, bom_a)
            self._write_bom(b, bom_b)
            with patch("sys.argv", ["squash", "diff", str(a), str(b)]):
                try:
                    main()
                    exited = 0
                except SystemExit as e:
                    exited = e.code
        assert exited == 0


# ──────────────────────────────────────────────────────────────────────────────
# POST /sbom/diff API endpoint
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def api_client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    os.environ.pop("SQUASH_API_TOKEN", None)
    from squish.squash.api import app
    return TestClient(app)


class TestSbomDiffApiEndpoint:
    def _write_bom(self, path: Path, bom: dict) -> None:
        path.write_text(json.dumps(bom))

    def test_identical_boms_returns_no_regressions(self, api_client, tmp_path):
        bom = _make_bom()
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        self._write_bom(a, bom)
        self._write_bom(b, bom)
        resp = api_client.post("/sbom/diff", json={"sbom_a_path": str(a), "sbom_b_path": str(b)})
        assert resp.status_code == 200
        body = resp.json()
        assert "has_regressions" in body
        assert not body["has_regressions"]

    def test_new_finding_detected_in_response(self, api_client, tmp_path):
        bom_a = _make_bom(vuln_ids=[])
        bom_b = _make_bom(vuln_ids=["CVE-2024-0001"])
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        self._write_bom(a, bom_a)
        self._write_bom(b, bom_b)
        resp = api_client.post("/sbom/diff", json={"sbom_a_path": str(a), "sbom_b_path": str(b)})
        assert resp.status_code == 200
        body = resp.json()
        assert "CVE-2024-0001" in body["new_findings"]
        assert body["has_regressions"] is True

    def test_missing_file_returns_404(self, api_client, tmp_path):
        bom = _make_bom()
        a = tmp_path / "exists.json"
        self._write_bom(a, bom)
        resp = api_client.post(
            "/sbom/diff",
            json={"sbom_a_path": str(a), "sbom_b_path": "/nonexistent/nope.json"},
        )
        assert resp.status_code == 404

    def test_response_schema_keys_present(self, api_client, tmp_path):
        bom = _make_bom()
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        self._write_bom(a, bom)
        self._write_bom(b, bom)
        resp = api_client.post("/sbom/diff", json={"sbom_a_path": str(a), "sbom_b_path": str(b)})
        body = resp.json()
        expected_keys = {
            "hash_changed", "score_delta", "policy_status_changed",
            "new_findings", "resolved_findings", "metadata_changes", "has_regressions",
        }
        assert expected_keys.issubset(body.keys())
