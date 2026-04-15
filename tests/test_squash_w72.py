"""W72 — platform-wide EU AI Act conformance report.

Tests for:
- ``CloudDB.read_conformance_report()``
- ``GET /cloud/conformance-report``

Report fields:
- total_tenants, conformant_tenants, non_conformant_tenants
- non_conformant: [{tenant_id, compliance_score, attestation_pass_rate,
                    open_vex_alerts, reasons}]
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import squish.squash.api as _api_module
from squish.squash.api import app
from squish.squash.cloud_db import CloudDB


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_db(tmp_path) -> CloudDB:
    return CloudDB(tmp_path / "w72.db")


def _make_conformant(db: CloudDB, tenant_id: str) -> None:
    """Register tenant and give it passing attestations — zero VEX, 100% compliance."""
    db.upsert_tenant(tenant_id, {"name": tenant_id})
    db.append_vertex_result(
        tenant_id=tenant_id,
        model_resource_name="projects/p/models/m",
        passed=True,
    )
    db.append_vertex_result(
        tenant_id=tenant_id,
        model_resource_name="projects/p/models/m2",
        passed=True,
    )


def _make_non_conformant(db: CloudDB, tenant_id: str) -> None:
    """Register tenant and give it failing attestations (pass_rate=0)."""
    db.upsert_tenant(tenant_id, {"name": tenant_id})
    db.append_vertex_result(
        tenant_id=tenant_id,
        model_resource_name="projects/p/models/mbad",
        passed=False,
    )


# ── CloudDB unit tests ────────────────────────────────────────────────────────

class TestCloudDBConformanceReport:
    def test_returns_dict(self, tmp_path):
        db = _make_db(tmp_path)
        result = db.read_conformance_report()
        assert isinstance(result, dict)

    def test_empty_platform_all_zeros(self, tmp_path):
        db = _make_db(tmp_path)
        result = db.read_conformance_report()
        assert result["total_tenants"] == 0
        assert result["conformant_tenants"] == 0
        assert result["non_conformant_tenants"] == 0
        assert result["non_conformant"] == []

    def test_required_keys(self, tmp_path):
        db = _make_db(tmp_path)
        result = db.read_conformance_report()
        for key in ("total_tenants", "conformant_tenants",
                    "non_conformant_tenants", "non_conformant"):
            assert key in result, f"missing key: {key}"

    def test_single_conformant_tenant(self, tmp_path):
        db = _make_db(tmp_path)
        _make_conformant(db, "alpha")
        result = db.read_conformance_report()
        assert result["total_tenants"] == 1
        assert result["conformant_tenants"] == 1
        assert result["non_conformant_tenants"] == 0
        assert result["non_conformant"] == []

    def test_single_non_conformant_tenant(self, tmp_path):
        db = _make_db(tmp_path)
        _make_non_conformant(db, "beta")
        result = db.read_conformance_report()
        assert result["total_tenants"] == 1
        assert result["conformant_tenants"] == 0
        assert result["non_conformant_tenants"] == 1
        assert len(result["non_conformant"]) == 1
        nc = result["non_conformant"][0]
        assert nc["tenant_id"] == "beta"
        assert "attestation_pass_rate" in nc
        assert "reasons" in nc

    def test_mixed_tenants_counts(self, tmp_path):
        db = _make_db(tmp_path)
        _make_conformant(db, "good-1")
        _make_conformant(db, "good-2")
        _make_non_conformant(db, "bad-1")
        result = db.read_conformance_report()
        assert result["total_tenants"] == 3
        assert result["conformant_tenants"] == 2
        assert result["non_conformant_tenants"] == 1

    def test_counts_sum_to_total(self, tmp_path):
        db = _make_db(tmp_path)
        for i in range(3):
            _make_conformant(db, f"c-{i}")
        for i in range(2):
            _make_non_conformant(db, f"nc-{i}")
        result = db.read_conformance_report()
        assert (result["conformant_tenants"] + result["non_conformant_tenants"]
                == result["total_tenants"])

    def test_non_conformant_entry_has_required_keys(self, tmp_path):
        db = _make_db(tmp_path)
        _make_non_conformant(db, "nc-keys")
        result = db.read_conformance_report()
        entry = result["non_conformant"][0]
        for key in ("tenant_id", "compliance_score", "attestation_pass_rate",
                    "open_vex_alerts", "reasons"):
            assert key in entry, f"missing key in non_conformant entry: {key}"

    def test_vex_alert_tenant_in_non_conformant(self, tmp_path):
        db = _make_db(tmp_path)
        db.upsert_tenant("vex-one", {"name": "vex-one"})
        db.append_vertex_result("vex-one", "projects/p/models/m", passed=True)
        db.append_vertex_result("vex-one", "projects/p/models/m2", passed=True)
        db.append_record("vex_alerts", "vex-one",
                         {"cve_id": "CVE-2025-0072", "severity": "high"})
        result = db.read_conformance_report()
        nc_ids = [e["tenant_id"] for e in result["non_conformant"]]
        assert "vex-one" in nc_ids


# ── API integration tests ─────────────────────────────────────────────────────

class TestCloudAPIConformanceReport:
    @pytest.fixture(autouse=True)
    def _client(self):
        # Clear rate-window accumulation so this class doesn't exhaust the shared
        # per-IP budget and cause false 429s in unrelated later tests.
        _api_module._rate_window.pop("testclient", None)
        self.client = TestClient(app)
        yield
        _api_module._rate_window.pop("testclient", None)

    def _reg(self, tid: str) -> None:
        self.client.post("/cloud/tenant", json={"tenant_id": tid, "name": tid})

    def _vertex(self, tid: str, *, passed: bool) -> None:
        self.client.post(
            f"/cloud/tenants/{tid}/vertex-result",
            json={"model_resource_name": "projects/p/models/m",
                  "passed": passed, "labels": None},
        )

    def _vex(self, tid: str) -> None:
        self.client.post(
            "/cloud/vex/alert",
            json={"tenant_id": tid, "cve_id": "CVE-2025-0072",
                  "severity": "high"},
        )

    def test_get_returns_200(self):
        r = self.client.get("/cloud/conformance-report")
        assert r.status_code == 200

    def test_response_has_required_keys(self):
        r = self.client.get("/cloud/conformance-report")
        body = r.json()
        for key in ("total_tenants", "conformant_tenants",
                    "non_conformant_tenants", "non_conformant"):
            assert key in body, f"missing key: {key}"

    def test_non_conformant_is_list(self):
        r = self.client.get("/cloud/conformance-report")
        assert isinstance(r.json()["non_conformant"], list)

    def test_counts_are_ints(self):
        r = self.client.get("/cloud/conformance-report")
        body = r.json()
        assert isinstance(body["total_tenants"], int)
        assert isinstance(body["conformant_tenants"], int)
        assert isinstance(body["non_conformant_tenants"], int)

    def test_registered_conformant_tenant_counted(self):
        tid = "w72-conf"
        self._reg(tid)
        self._vertex(tid, passed=True)
        self._vertex(tid, passed=True)
        r = self.client.get("/cloud/conformance-report")
        body = r.json()
        assert body["total_tenants"] >= 1

    def test_failing_tenant_in_non_conformant_list(self):
        tid = "w72-fail"
        self._reg(tid)
        self._vertex(tid, passed=False)
        for _ in range(4):
            self._vertex(tid, passed=False)
        r = self.client.get("/cloud/conformance-report")
        body = r.json()
        nc_ids = [e["tenant_id"] for e in body["non_conformant"]]
        assert tid in nc_ids

    def test_non_conformant_entry_structure(self):
        tid = "w72-struct"
        self._reg(tid)
        self._vertex(tid, passed=False)
        r = self.client.get("/cloud/conformance-report")
        nc = [e for e in r.json()["non_conformant"] if e["tenant_id"] == tid]
        assert nc, f"tenant {tid!r} not found in non_conformant"
        entry = nc[0]
        for key in ("tenant_id", "compliance_score", "attestation_pass_rate",
                    "open_vex_alerts", "reasons"):
            assert key in entry, f"missing key in entry: {key}"

    def test_vex_alert_tenant_appears_as_non_conformant(self):
        tid = "w72-vex"
        self._reg(tid)
        self._vertex(tid, passed=True)
        self._vertex(tid, passed=True)
        self._vex(tid)
        r = self.client.get("/cloud/conformance-report")
        nc_ids = [e["tenant_id"] for e in r.json()["non_conformant"]]
        assert tid in nc_ids

    def test_counts_add_up(self):
        r = self.client.get("/cloud/conformance-report")
        body = r.json()
        assert (body["conformant_tenants"] + body["non_conformant_tenants"]
                == body["total_tenants"])
