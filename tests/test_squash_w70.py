"""tests/test_squash_w70.py — W70: platform attestation overview.

Tests for:
  - CloudDB.read_attestation_overview()  (8 unit tests)
  - GET /cloud/attestation-overview       (8 API integration tests)
"""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient

from squish.squash import api
from squish.squash.cloud_db import CloudDB


# ── CloudDB unit tests ────────────────────────────────────────────────────────

class TestCloudDBAttestationOverview:
    """Unit tests for CloudDB.read_attestation_overview()."""

    def setup_method(self) -> None:
        self.db = CloudDB(":memory:")

    def _reg(self, tid: str) -> None:
        self.db.upsert_tenant(tid, {})

    def _vertex(self, tid: str, passed: bool) -> None:
        self.db._conn.execute(
            "INSERT INTO vertex_results (tenant_id, model_resource_name, passed, labels, ts)"
            " VALUES (?, 'proj/model', ?, NULL, '2025-01-01T00:00:00Z')",
            (tid, int(passed)),
        )
        self.db._conn.commit()

    def _ado(self, tid: str, passed: bool) -> None:
        self.db._conn.execute(
            "INSERT INTO ado_results (tenant_id, pipeline_run_id, passed, variables, ts)"
            " VALUES (?, 'run-1', ?, NULL, '2025-01-02T00:00:00Z')",
            (tid, int(passed)),
        )
        self.db._conn.commit()

    # 1 -----------------------------------------------------------------------
    def test_returns_dict(self) -> None:
        result = self.db.read_attestation_overview()
        assert isinstance(result, dict)

    # 2 -----------------------------------------------------------------------
    def test_empty_platform_all_zeros(self) -> None:
        result = self.db.read_attestation_overview()
        assert result["total_attestations"] == 0
        assert result["tenants_covered"] == 0
        assert result["platform_pass_rate"] == 0.0
        assert result["tenants_with_failures"] == []

    # 3 -----------------------------------------------------------------------
    def test_total_attestations_across_tenants(self) -> None:
        self._reg("t1"); self._vertex("t1", True)
        self._reg("t2"); self._vertex("t2", False)
        result = self.db.read_attestation_overview()
        assert result["total_attestations"] == 2

    # 4 -----------------------------------------------------------------------
    def test_tenants_covered_count(self) -> None:
        for i in range(3):
            self._reg(f"t{i}")
        result = self.db.read_attestation_overview()
        assert result["tenants_covered"] == 3

    # 5 -----------------------------------------------------------------------
    def test_platform_pass_rate(self) -> None:
        self._reg("ta"); self._vertex("ta", True); self._vertex("ta", True)
        self._reg("tb"); self._vertex("tb", False)
        # 2 passed out of 3 total → ~0.6667
        result = self.db.read_attestation_overview()
        assert abs(result["platform_pass_rate"] - round(2 / 3, 4)) < 1e-6

    # 6 -----------------------------------------------------------------------
    def test_tenants_with_failures_populated(self) -> None:
        self._reg("ok"); self._vertex("ok", True)
        self._reg("bad"); self._vertex("bad", False)
        result = self.db.read_attestation_overview()
        tenant_ids = [e["tenant_id"] for e in result["tenants_with_failures"]]
        assert "bad" in tenant_ids
        assert "ok" not in tenant_ids

    # 7 -----------------------------------------------------------------------
    def test_all_passing_empty_failures_list(self) -> None:
        self._reg("a"); self._vertex("a", True)
        self._reg("b"); self._ado("b", True)
        result = self.db.read_attestation_overview()
        assert result["tenants_with_failures"] == []

    # 8 -----------------------------------------------------------------------
    def test_tenants_with_failures_has_required_keys(self) -> None:
        self._reg("x"); self._vertex("x", False)
        result = self.db.read_attestation_overview()
        assert len(result["tenants_with_failures"]) == 1
        entry = result["tenants_with_failures"][0]
        assert {"tenant_id", "failed", "pass_rate"} <= entry.keys()


# ── API integration tests ─────────────────────────────────────────────────────

class TestCloudAPIAttestationOverview:
    """Integration tests for GET /cloud/attestation-overview (W70)."""

    def setup_method(self) -> None:
        api._tenants.clear()
        api._inventory.clear()
        api._vex_alerts.clear()
        api._drift_events.clear()
        api._policy_stats.clear()
        api._vertex_results.clear()
        api._ado_results.clear()
        api._rate_window.clear()
        api._db = None
        self.client = TestClient(api.app, raise_server_exceptions=True)

    def _reg(self, tid: str) -> None:
        self.client.post("/cloud/tenant", json={"tenant_id": tid, "name": tid})

    def _vertex(self, tid: str, passed: bool) -> None:
        self.client.post(
            f"/cloud/tenants/{tid}/vertex-result",
            json={"model_resource_name": "proj/model", "passed": passed, "labels": None},
        )

    def _ado(self, tid: str, passed: bool) -> None:
        self.client.post(
            f"/cloud/tenants/{tid}/ado-result",
            json={"pipeline_run_id": "run-1", "passed": passed, "variables": None},
        )

    # 1 -----------------------------------------------------------------------
    def test_get_returns_200(self) -> None:
        r = self.client.get("/cloud/attestation-overview")
        assert r.status_code == 200

    # 2 -----------------------------------------------------------------------
    def test_response_has_required_keys(self) -> None:
        r = self.client.get("/cloud/attestation-overview")
        body = r.json()
        assert {"total_attestations", "tenants_covered", "platform_pass_rate", "tenants_with_failures"} <= body.keys()

    # 3 -----------------------------------------------------------------------
    def test_empty_platform_zeros(self) -> None:
        body = self.client.get("/cloud/attestation-overview").json()
        assert body["total_attestations"] == 0
        assert body["tenants_covered"] == 0
        assert body["platform_pass_rate"] == 0.0
        assert body["tenants_with_failures"] == []

    # 4 -----------------------------------------------------------------------
    def test_after_vertex_post_total_increments(self) -> None:
        self._reg("t1"); self._vertex("t1", True)
        body = self.client.get("/cloud/attestation-overview").json()
        assert body["total_attestations"] == 1
        assert body["tenants_covered"] == 1

    # 5 -----------------------------------------------------------------------
    def test_after_ado_post_total_increments(self) -> None:
        self._reg("t2"); self._ado("t2", True)
        body = self.client.get("/cloud/attestation-overview").json()
        assert body["total_attestations"] == 1
        assert body["tenants_covered"] == 1

    # 6 -----------------------------------------------------------------------
    def test_tenants_with_failures_appears(self) -> None:
        self._reg("good"); self._vertex("good", True)
        self._reg("bad"); self._vertex("bad", False)
        body = self.client.get("/cloud/attestation-overview").json()
        failure_ids = [e["tenant_id"] for e in body["tenants_with_failures"]]
        assert "bad" in failure_ids
        assert "good" not in failure_ids

    # 7 -----------------------------------------------------------------------
    def test_platform_pass_rate_is_float(self) -> None:
        self._reg("t1"); self._vertex("t1", True)
        body = self.client.get("/cloud/attestation-overview").json()
        assert isinstance(body["platform_pass_rate"], float)

    # 8 -----------------------------------------------------------------------
    def test_all_passing_no_failures_list(self) -> None:
        self._reg("a"); self._vertex("a", True)
        self._reg("b"); self._ado("b", True)
        body = self.client.get("/cloud/attestation-overview").json()
        assert body["tenants_with_failures"] == []
