"""W64 — Cross-tenant compliance overview tests.

Covers:
  - CloudDB.read_compliance_overview() (8 unit tests)
  - GET /cloud/compliance-overview (8 API integration tests)
"""
from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from squish.squash.cloud_db import CloudDB
from squish.squash.api import (
    app,
    _tenants,
    _policy_stats,
    _inventory,
    _vex_alerts,
    _drift_events,
    _rate_window,
)


# ── CloudDB unit tests ────────────────────────────────────────────────────────


class TestCloudDBComplianceOverview:
    def setup_method(self):
        self.db = CloudDB(path=":memory:")

    def _inject_failing_stats(self, tenant_id: str, passed: int = 1, failed: int = 9) -> None:
        """Insert policy_stats directly to produce a low compliance score."""
        self.db._conn.execute(
            "INSERT OR REPLACE INTO policy_stats "
            "(tenant_id, policy_name, passed, failed) VALUES (?, ?, ?, ?)",
            (tenant_id, "security_policy", passed, failed),
        )
        self.db._conn.commit()

    # 1. Result is a dict
    def test_returns_dict(self):
        result = self.db.read_compliance_overview()
        assert isinstance(result, dict)

    # 2. No tenants → all-zero counts, empty top_at_risk
    def test_empty_platform_all_zeros(self):
        result = self.db.read_compliance_overview()
        assert result["total_tenants"] == 0
        assert result["compliant_tenants"] == 0
        assert result["non_compliant_tenants"] == 0
        assert result["average_score"] == 0.0
        assert result["top_at_risk"] == []

    # 3. Single tenant with no policy failures → compliant_tenants=1
    def test_single_tenant_compliant(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        result = self.db.read_compliance_overview()
        assert result["compliant_tenants"] == 1
        assert result["non_compliant_tenants"] == 0

    # 4. Single tenant with injected failures → non_compliant_tenants=1
    def test_single_tenant_non_compliant(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self._inject_failing_stats("t1", passed=1, failed=9)  # score ≈ 10.0
        result = self.db.read_compliance_overview()
        assert result["non_compliant_tenants"] == 1
        assert result["compliant_tenants"] == 0

    # 5. Three tenants → total_tenants=3
    def test_total_count_correct(self):
        for tid in ("t1", "t2", "t3"):
            self.db.upsert_tenant(tid, {"name": tid})
        result = self.db.read_compliance_overview()
        assert result["total_tenants"] == 3

    # 6. average_score is a float
    def test_average_score_is_float(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        result = self.db.read_compliance_overview()
        assert isinstance(result["average_score"], float)

    # 7. top_at_risk sorted ascending by score (worst first)
    def test_top_at_risk_sorted_ascending(self):
        for tid in ("t1", "t2", "t3"):
            self.db.upsert_tenant(tid, {"name": tid})
        # t1: 10%, t2: 50%, t3: 100% (no failures)
        self._inject_failing_stats("t1", passed=1, failed=9)   # ~10.0
        self._inject_failing_stats("t2", passed=5, failed=5)   # ~50.0
        result = self.db.read_compliance_overview()
        at_risk = result["top_at_risk"]
        assert len(at_risk) >= 2
        assert at_risk[0]["score"] <= at_risk[-1]["score"]  # ascending

    # 8. Five non-compliant tenants → top_at_risk capped at 3
    def test_top_at_risk_capped_at_three(self):
        for i, tid in enumerate(("t1", "t2", "t3", "t4", "t5")):
            self.db.upsert_tenant(tid, {"name": tid})
            # All below 80.0 threshold (different fail counts to vary scores)
            self.db._conn.execute(
                "INSERT OR REPLACE INTO policy_stats "
                "(tenant_id, policy_name, passed, failed) VALUES (?, ?, ?, ?)",
                (tid, "sec", 1, 9 + i),
            )
        self.db._conn.commit()
        result = self.db.read_compliance_overview()
        assert result["total_tenants"] == 5
        assert len(result["top_at_risk"]) <= 3


# ── API integration tests ─────────────────────────────────────────────────────


class TestCloudAPIComplianceOverviewEndpoint:
    def setup_method(self):
        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        _rate_window.clear()  # prevent 429s in full-suite runs
        self.client = TestClient(app, raise_server_exceptions=True)

    # 1. GET returns 200
    def test_200_response(self):
        resp = self.client.get("/cloud/compliance-overview")
        assert resp.status_code == 200

    # 2. Response has all required keys
    def test_response_has_required_keys(self):
        resp = self.client.get("/cloud/compliance-overview")
        body = resp.json()
        for key in ("total_tenants", "compliant_tenants", "non_compliant_tenants",
                    "average_score", "top_at_risk"):
            assert key in body, f"missing key: {key}"

    # 3. Empty platform → zero counts, empty top_at_risk
    def test_empty_platform(self):
        resp = self.client.get("/cloud/compliance-overview")
        body = resp.json()
        assert body["total_tenants"] == 0
        assert body["compliant_tenants"] == 0
        assert body["non_compliant_tenants"] == 0
        assert body["average_score"] == 0.0
        assert body["top_at_risk"] == []

    # 4. Two tenants → total_tenants=2
    def test_total_tenants_count(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _tenants["globex"] = {"name": "Globex"}
        resp = self.client.get("/cloud/compliance-overview")
        assert resp.json()["total_tenants"] == 2

    # 5. Two tenants, no policy failures → compliant_tenants=2
    def test_compliant_count(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _tenants["globex"] = {"name": "Globex"}
        resp = self.client.get("/cloud/compliance-overview")
        body = resp.json()
        assert body["compliant_tenants"] == 2
        assert body["non_compliant_tenants"] == 0

    # 6. Two tenants (both score 100.0) → average_score > 0
    def test_average_score_nonzero_with_tenants(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _tenants["globex"] = {"name": "Globex"}
        resp = self.client.get("/cloud/compliance-overview")
        assert resp.json()["average_score"] > 0

    # 7. top_at_risk is a list
    def test_top_at_risk_is_list(self):
        resp = self.client.get("/cloud/compliance-overview")
        assert isinstance(resp.json()["top_at_risk"], list)

    # 8. Endpoint accessible without path parameter
    def test_no_path_parameter(self):
        # Confirm the route requires no {tenant_id} segment
        resp = self.client.get("/cloud/compliance-overview")
        assert resp.status_code == 200
