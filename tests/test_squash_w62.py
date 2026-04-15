"""W62 — Tenant compliance-score endpoint tests.

Covers:
  - CloudDB.read_tenant_compliance_score() (8 unit tests)
  - GET /cloud/tenants/{tenant_id}/compliance-score (8 API integration tests)
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


class TestCloudDBTenantComplianceScore:
    def setup_method(self):
        self.db = CloudDB(path=":memory:")

    # 1. Required keys present
    def test_returns_dict_with_required_keys(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        result = self.db.read_tenant_compliance_score("t1")
        assert set(result.keys()) == {"score", "grade", "policy_breakdown"}

    # 2. No policy checks → perfect score
    def test_no_policies_returns_perfect_score(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        result = self.db.read_tenant_compliance_score("t1")
        assert result["score"] == 100.0
        assert result["grade"] == "A"
        assert result["policy_breakdown"] == {}

    # 3. All passed → 100.0 / A
    def test_all_passed_returns_100(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        for _ in range(5):
            self.db.inc_policy_stat("t1", "SBOM_CHECK", passed=True)
        result = self.db.read_tenant_compliance_score("t1")
        assert result["score"] == 100.0
        assert result["grade"] == "A"

    # 4. All failed → 0.0 / F
    def test_all_failed_returns_0(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        for _ in range(5):
            self.db.inc_policy_stat("t1", "SBOM_CHECK", passed=False)
        result = self.db.read_tenant_compliance_score("t1")
        assert result["score"] == 0.0
        assert result["grade"] == "F"

    # 5. Mixed score computed correctly: 3 passed, 1 failed → 75.0
    def test_mixed_score_computed_correctly(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        for _ in range(3):
            self.db.inc_policy_stat("t1", "POLICY_A", passed=True)
        self.db.inc_policy_stat("t1", "POLICY_A", passed=False)
        result = self.db.read_tenant_compliance_score("t1")
        assert result["score"] == 75.0

    # 6. Grade thresholds (boundary values)
    @pytest.mark.parametrize("passed,total,expected_grade", [
        (90, 100, "A"),   # exactly 90 → A
        (75, 100, "B"),   # exactly 75 → B
        (60, 100, "C"),   # exactly 60 → C
        (45, 100, "D"),   # exactly 45 → D
        (44, 100, "F"),   # 44 → F (< 45)
    ])
    def test_grade_thresholds(self, passed, total, expected_grade):
        self.db.upsert_tenant("t1", {"name": "T1"})
        failed = total - passed
        for _ in range(passed):
            self.db.inc_policy_stat("t1", "P", passed=True)
        for _ in range(failed):
            self.db.inc_policy_stat("t1", "P", passed=False)
        result = self.db.read_tenant_compliance_score("t1")
        assert result["grade"] == expected_grade

    # 7. policy_breakdown contains per-policy rate field
    def test_policy_breakdown_contains_rate(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.inc_policy_stat("t1", "CHECK_A", passed=True)
        self.db.inc_policy_stat("t1", "CHECK_A", passed=False)
        result = self.db.read_tenant_compliance_score("t1")
        breakdown = result["policy_breakdown"]
        assert "CHECK_A" in breakdown
        entry = breakdown["CHECK_A"]
        assert "rate" in entry
        assert isinstance(entry["rate"], float)
        assert entry["rate"] == 50.0

    # 8. Score scoped to tenant — two tenants get independent results
    def test_score_scoped_to_tenant(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.upsert_tenant("t2", {"name": "T2"})
        # t1: all pass
        for _ in range(10):
            self.db.inc_policy_stat("t1", "POL", passed=True)
        # t2: all fail
        for _ in range(10):
            self.db.inc_policy_stat("t2", "POL", passed=False)
        r1 = self.db.read_tenant_compliance_score("t1")
        r2 = self.db.read_tenant_compliance_score("t2")
        assert r1["score"] == 100.0
        assert r2["score"] == 0.0
        assert r1["grade"] == "A"
        assert r2["grade"] == "F"


# ── API integration tests ─────────────────────────────────────────────────────


class TestCloudAPIComplianceScoreEndpoint:
    def setup_method(self):
        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        _rate_window.clear()  # prevent 429s in full-suite runs
        self.client = TestClient(app, raise_server_exceptions=True)

    # 1. 404 for unknown tenant
    def test_404_for_unknown_tenant(self):
        resp = self.client.get("/cloud/tenants/no-such-tenant/compliance-score")
        assert resp.status_code == 404

    # 2. 200 for known tenant
    def test_200_for_known_tenant(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        assert resp.status_code == 200

    # 3. Response has required keys
    def test_response_has_required_keys(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        body = resp.json()
        assert "score" in body
        assert "grade" in body
        assert "policy_breakdown" in body
        assert "tenant_id" in body

    # 4. New tenant (no policy checks) → score 100.0, grade 'A'
    def test_perfect_score_no_policies(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        body = resp.json()
        assert body["score"] == 100.0
        assert body["grade"] == "A"

    # 5. In-memory stats reflected in score
    def test_score_reflects_in_memory_stats(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        # Inject stats directly into in-memory store (mirrors what /drift-check writes)
        from collections import defaultdict
        _policy_stats["acme"] = {
            "SBOM_CHECK": {"passed": 1, "failed": 1},
        }
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        body = resp.json()
        assert body["score"] == 50.0

    # 6. Grade A on 100% pass rate
    def test_grade_a_on_high_pass_rate(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _policy_stats["acme"] = {
            "SBOM_CHECK": {"passed": 10, "failed": 0},
        }
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        assert resp.json()["grade"] == "A"

    # 7. Grade F on 0% pass rate
    def test_grade_f_on_low_pass_rate(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        _policy_stats["acme"] = {
            "SBOM_CHECK": {"passed": 0, "failed": 10},
        }
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        assert resp.json()["grade"] == "F"

    # 8. tenant_id echoed in response body
    def test_tenant_id_echoed_in_response(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/compliance-score")
        assert resp.json()["tenant_id"] == "acme"
