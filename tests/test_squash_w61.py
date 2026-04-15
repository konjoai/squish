"""W61 — Tenant summary endpoint tests.

16 tests:
  - TestCloudDBTenantSummary          (8 tests) — CloudDB.read_tenant_summary()
  - TestCloudAPITenantSummaryEndpoint (8 tests) — GET /cloud/tenants/{id}/summary
"""

from collections import defaultdict

import pytest
from fastapi.testclient import TestClient

from squish.squash.api import (
    app,
    _drift_events,
    _inventory,
    _policy_stats,
    _rate_window,
    _tenants,
    _vex_alerts,
)
from squish.squash.cloud_db import CloudDB


# ── CloudDB unit tests ────────────────────────────────────────────────────────


class TestCloudDBTenantSummary:
    def setup_method(self):
        self.db = CloudDB(path=":memory:")

    def test_returns_dict_with_required_keys(self):
        self.db.upsert_tenant("t1", {"name": "Acme"})
        result = self.db.read_tenant_summary("t1")
        assert set(result.keys()) == {
            "inventory_count",
            "vex_alert_count",
            "drift_event_count",
            "policy_stats",
        }

    def test_all_zero_for_empty_tenant(self):
        self.db.upsert_tenant("t_empty", {"name": "Empty"})
        result = self.db.read_tenant_summary("t_empty")
        assert result["inventory_count"] == 0
        assert result["vex_alert_count"] == 0
        assert result["drift_event_count"] == 0
        assert result["policy_stats"] == {}

    def test_zero_counts_for_unknown_tenant(self):
        # Unknown tenant — no raise, returns zeros
        result = self.db.read_tenant_summary("ghost")
        assert result["inventory_count"] == 0
        assert result["vex_alert_count"] == 0
        assert result["drift_event_count"] == 0
        assert result["policy_stats"] == {}

    def test_inventory_count_correct(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        for i in range(3):
            self.db.append_record("inventory", "t1", {"item": f"pkg-{i}"})
        result = self.db.read_tenant_summary("t1")
        assert result["inventory_count"] == 3

    def test_vex_alert_count_correct(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        for i in range(5):
            self.db.append_record("vex_alerts", "t1", {"cve": f"CVE-{i}", "severity": "HIGH"})
        result = self.db.read_tenant_summary("t1")
        assert result["vex_alert_count"] == 5

    def test_drift_event_count_correct(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        for i in range(4):
            self.db.append_record("drift_events", "t1", {"component": f"svc-{i}", "delta": 1})
        result = self.db.read_tenant_summary("t1")
        assert result["drift_event_count"] == 4

    def test_policy_stats_aggregated(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.inc_policy_stat("t1", "SBOM_CHECK", passed=True)
        self.db.inc_policy_stat("t1", "SBOM_CHECK", passed=False)
        self.db.inc_policy_stat("t1", "VEX_CHECK", passed=True)
        result = self.db.read_tenant_summary("t1")
        ps = result["policy_stats"]
        assert ps["SBOM_CHECK"] == {"passed": 1, "failed": 1}
        assert ps["VEX_CHECK"] == {"passed": 1, "failed": 0}

    def test_summary_scoped_to_tenant(self):
        self.db.upsert_tenant("t1", {"name": "T1"})
        self.db.upsert_tenant("t2", {"name": "T2"})
        for i in range(3):
            self.db.append_record("inventory", "t1", {"item": f"pkg-{i}"})
        self.db.append_record("inventory", "t2", {"item": "only-one"})
        r1 = self.db.read_tenant_summary("t1")
        r2 = self.db.read_tenant_summary("t2")
        assert r1["inventory_count"] == 3
        assert r2["inventory_count"] == 1


# ── API endpoint tests ────────────────────────────────────────────────────────


class TestCloudAPITenantSummaryEndpoint:
    def setup_method(self):
        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        _rate_window.clear()  # prevent 429 across suite runs
        self.client = TestClient(app, raise_server_exceptions=True)

    def test_404_for_unknown_tenant(self):
        resp = self.client.get("/cloud/tenants/ghost/summary")
        assert resp.status_code == 404
        assert "ghost" in resp.json()["detail"]

    def test_200_for_known_tenant(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        resp = self.client.get("/cloud/tenants/acme/summary")
        assert resp.status_code == 200

    def test_response_has_required_keys(self):
        _tenants["acme"] = {"name": "Acme Corp"}
        body = self.client.get("/cloud/tenants/acme/summary").json()
        for key in ("tenant_id", "tenant", "inventory_count", "vex_alert_count",
                    "drift_event_count", "policy_stats"):
            assert key in body, f"missing key: {key}"

    def test_tenant_id_echoed(self):
        _tenants["acme"] = {"name": "Acme"}
        body = self.client.get("/cloud/tenants/acme/summary").json()
        assert body["tenant_id"] == "acme"

    def test_tenant_object_echoed(self):
        _tenants["acme"] = {"name": "Acme"}
        body = self.client.get("/cloud/tenants/acme/summary").json()
        assert body["tenant"] == {"name": "Acme"}

    def test_zero_counts_for_empty_tenant(self):
        _tenants["empty"] = {"name": "Empty Corp"}
        body = self.client.get("/cloud/tenants/empty/summary").json()
        assert body["inventory_count"] == 0
        assert body["vex_alert_count"] == 0
        assert body["drift_event_count"] == 0
        assert body["policy_stats"] == {}

    def test_counts_reflect_in_memory_stores(self):
        _tenants["t1"] = {"name": "T1"}
        from collections import deque

        _inventory["t1"] = deque([{"item": "a"}, {"item": "b"}], maxlen=500)
        _vex_alerts["t1"] = deque([{"cve": "CVE-2025-1"}], maxlen=500)
        _drift_events["t1"] = deque([{"c": "svc"}, {"c": "svc2"}, {"c": "svc3"}], maxlen=500)
        body = self.client.get("/cloud/tenants/t1/summary").json()
        assert body["inventory_count"] == 2
        assert body["vex_alert_count"] == 1
        assert body["drift_event_count"] == 3

    def test_policy_stats_from_in_memory(self):
        _tenants["t1"] = {"name": "T1"}
        _policy_stats["t1"]["SBOM_CHECK"]["passed"] = 7
        _policy_stats["t1"]["SBOM_CHECK"]["failed"] = 2
        body = self.client.get("/cloud/tenants/t1/summary").json()
        ps = body["policy_stats"]
        assert ps["SBOM_CHECK"]["passed"] == 7
        assert ps["SBOM_CHECK"]["failed"] == 2
