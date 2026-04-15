"""W71 — per-tenant EU AI Act conformance status.

Tests for:
- ``CloudDB.read_tenant_conformance()``
- ``GET /cloud/tenants/{tenant_id}/conformance``

Conformance gates:
- compliance_score >= 80.0
- attestation_pass_rate >= 0.8
- open_vex_alerts == 0
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from squish.squash.api import app
from squish.squash.cloud_db import CloudDB


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_db(tmp_path) -> CloudDB:
    return CloudDB(tmp_path / "w71.db")


def _add_vertex(db: CloudDB, tenant_id: str, *, passed: bool) -> None:
    db.append_vertex_result(
        tenant_id=tenant_id,
        model_resource_name="projects/p/locations/us-central1/models/m",
        passed=passed,
    )


def _add_vex(db: CloudDB, tenant_id: str) -> None:
    db.append_record(
        "vex_alerts", tenant_id,
        {"cve_id": "CVE-2025-0071", "severity": "high", "status": "open"},
    )


def _drive_compliance_below_80(db: CloudDB, tenant_id: str) -> None:
    """Record enough failures to push compliance score below 80.0."""
    for _ in range(5):
        db.inc_policy_stat(tenant_id, "data-minimisation", passed=False)


# ── CloudDB unit tests ────────────────────────────────────────────────────────

class TestCloudDBTenantConformance:
    def test_returns_dict(self, tmp_path):
        db = _make_db(tmp_path)
        result = db.read_tenant_conformance("t1")
        assert isinstance(result, dict)

    def test_unknown_tenant_non_conformant(self, tmp_path):
        """New tenant with no attestations: pass_rate=0.0 → non-conformant."""
        db = _make_db(tmp_path)
        result = db.read_tenant_conformance("ghost")
        assert result["conformant"] is False
        assert result["attestation_pass_rate"] == 0.0
        assert len(result["reasons"]) >= 1

    def test_all_passing_conformant(self, tmp_path):
        """Tenant with high compliance, all attestations passing, no VEX → conformant."""
        db = _make_db(tmp_path)
        tid = "good-tenant"
        db.upsert_tenant(tid, {"name": tid})
        _add_vertex(db, tid, passed=True)
        _add_vertex(db, tid, passed=True)
        result = db.read_tenant_conformance(tid)
        assert result["conformant"] is True
        assert result["reasons"] == []
        assert result["compliance_score"] >= 80.0
        assert result["attestation_pass_rate"] >= 0.8
        assert result["open_vex_alerts"] == 0

    def test_low_compliance_non_conformant(self, tmp_path):
        db = _make_db(tmp_path)
        tid = "low-comp"
        db.upsert_tenant(tid, {"name": tid})
        _drive_compliance_below_80(db, tid)
        _add_vertex(db, tid, passed=True)
        _add_vertex(db, tid, passed=True)
        result = db.read_tenant_conformance(tid)
        assert result["compliance_score"] < 80.0
        assert result["conformant"] is False
        assert any("compliance" in r.lower() for r in result["reasons"])

    def test_low_attestation_non_conformant(self, tmp_path):
        db = _make_db(tmp_path)
        tid = "low-att"
        db.upsert_tenant(tid, {"name": tid})
        # 1 pass + 9 fail → pass_rate = 0.10
        _add_vertex(db, tid, passed=True)
        for _ in range(9):
            _add_vertex(db, tid, passed=False)
        result = db.read_tenant_conformance(tid)
        assert result["conformant"] is False
        assert result["attestation_pass_rate"] < 0.8
        assert any("attestation" in r.lower() for r in result["reasons"])

    def test_open_vex_alerts_non_conformant(self, tmp_path):
        db = _make_db(tmp_path)
        tid = "vex-tenant"
        db.upsert_tenant(tid, {"name": tid})
        _add_vertex(db, tid, passed=True)
        _add_vertex(db, tid, passed=True)
        _add_vex(db, tid)
        result = db.read_tenant_conformance(tid)
        assert result["open_vex_alerts"] >= 1
        assert result["conformant"] is False
        assert any("vex" in r.lower() for r in result["reasons"])

    def test_reasons_list_populated_for_failures(self, tmp_path):
        """Multiple failing gates → multiple reasons."""
        db = _make_db(tmp_path)
        tid = "multi-fail"
        db.upsert_tenant(tid, {"name": tid})
        # Fail attestation (0.0) AND add VEX
        _add_vertex(db, tid, passed=False)
        _add_vex(db, tid)
        result = db.read_tenant_conformance(tid)
        assert result["conformant"] is False
        assert len(result["reasons"]) >= 2

    def test_conformant_tenant_has_empty_reasons(self, tmp_path):
        db = _make_db(tmp_path)
        tid = "clean"
        db.upsert_tenant(tid, {"name": tid})
        _add_vertex(db, tid, passed=True)
        _add_vertex(db, tid, passed=True)
        result = db.read_tenant_conformance(tid)
        assert result["conformant"] is True
        assert result["reasons"] == []


# ── API integration tests ─────────────────────────────────────────────────────

class TestCloudAPITenantConformance:
    @pytest.fixture(autouse=True)
    def _client(self):
        self.client = TestClient(app)

    def _vertex(self, tid: str, *, passed: bool) -> None:
        self.client.post(
            f"/cloud/tenants/{tid}/vertex-result",
            json={"model_resource_name": "projects/p/models/m",
                  "passed": passed, "labels": None},
        )

    def _vex(self, tid: str) -> None:
        self.client.post(
            "/cloud/vex/alert",
            json={"tenant_id": tid, "cve_id": "CVE-2025-0071",
                  "severity": "high"},
        )

    def test_get_returns_200(self):
        r = self.client.get("/cloud/tenants/any-tenant/conformance")
        assert r.status_code == 200

    def test_response_has_required_keys(self):
        r = self.client.get("/cloud/tenants/keys-tenant/conformance")
        body = r.json()
        for key in ("tenant_id", "conformant", "compliance_score",
                    "attestation_pass_rate", "open_vex_alerts", "reasons"):
            assert key in body, f"missing key: {key}"

    def test_tenant_id_echoed(self):
        r = self.client.get("/cloud/tenants/echo-me/conformance")
        assert r.json()["tenant_id"] == "echo-me"

    def test_unknown_tenant_non_conformant_no_attestations(self):
        r = self.client.get("/cloud/tenants/ghost-71/conformance")
        body = r.json()
        assert body["conformant"] is False
        assert body["attestation_pass_rate"] == 0.0

    def test_after_passing_attestations_pass_rate_improves(self):
        tid = "w71-pass"
        for _ in range(5):
            self._vertex(tid, passed=True)
        r = self.client.get(f"/cloud/tenants/{tid}/conformance")
        body = r.json()
        assert body["attestation_pass_rate"] == 1.0

    def test_after_failing_attestation_non_conformant(self):
        tid = "w71-fail"
        self._vertex(tid, passed=True)
        for _ in range(9):
            self._vertex(tid, passed=False)
        r = self.client.get(f"/cloud/tenants/{tid}/conformance")
        body = r.json()
        assert body["attestation_pass_rate"] < 0.8
        assert body["conformant"] is False

    def test_vex_alert_causes_non_conformance(self):
        tid = "w71-vex"
        for _ in range(3):
            self._vertex(tid, passed=True)
        self._vex(tid)
        r = self.client.get(f"/cloud/tenants/{tid}/conformance")
        body = r.json()
        assert body["open_vex_alerts"] >= 1
        assert body["conformant"] is False

    def test_conformant_field_is_bool(self):
        r = self.client.get("/cloud/tenants/bool-check/conformance")
        assert isinstance(r.json()["conformant"], bool)

    def test_reasons_is_list(self):
        r = self.client.get("/cloud/tenants/reasons-check/conformance")
        assert isinstance(r.json()["reasons"], list)

