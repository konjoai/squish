"""W60 — Tenant-scoped drift-events and policy-stats reads.

Tests:
  - CloudDB.read_drift_events(tenant_id)       (4)
  - CloudDB.read_tenant_policy_stats(tenant_id) (4)
  - GET /cloud/tenants/{id}/drift-events        (4)
  - GET /cloud/tenants/{id}/policy-stats        (4)

All API tests use TestClient with in-memory state; CloudDB tests use a
temporary SQLite path to exercise the real SQL path.
"""

from __future__ import annotations

import os
import tempfile

import pytest

pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient


# ── CloudDB.read_drift_events tests ──────────────────────────────────────────


class TestCloudDBDriftEvents:
    """Unit tests for CloudDB.read_drift_events(tenant_id) — SQL path."""

    def _db(self, path: str):
        from squish.squash.cloud_db import CloudDB
        return CloudDB(path)

    def test_read_drift_events_returns_empty_on_fresh_db(self) -> None:
        """A fresh tenant with no appended rows returns an empty list."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            db.upsert_tenant("de-t1", {"tenant_id": "de-t1", "name": "DE T1"})
            assert db.read_drift_events("de-t1") == []
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_drift_events_returns_data_after_write(self) -> None:
        """append_record → read_drift_events returns the written row."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            db.upsert_tenant("de-t2", {"tenant_id": "de-t2", "name": "DE T2"})
            db.append_record("drift_events", "de-t2", {"model_id": "m1", "severity": "warning"})
            result = db.read_drift_events("de-t2")
            assert len(result) == 1
            assert result[0]["model_id"] == "m1"
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_drift_events_unknown_tenant_returns_empty(self) -> None:
        """Querying a tenant that was never upserted returns []."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            assert db.read_drift_events("ghost-de-tenant") == []
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_drift_events_isolates_by_tenant(self) -> None:
        """Rows for tenant A are invisible from tenant B's read."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            db.upsert_tenant("de-ta", {"tenant_id": "de-ta", "name": "DE TA"})
            db.upsert_tenant("de-tb", {"tenant_id": "de-tb", "name": "DE TB"})
            db.append_record("drift_events", "de-ta", {"model_id": "ma", "severity": "info"})
            db.append_record("drift_events", "de-tb", {"model_id": "mb", "severity": "critical"})
            result_a = db.read_drift_events("de-ta")
            result_b = db.read_drift_events("de-tb")
            assert len(result_a) == 1 and result_a[0]["model_id"] == "ma"
            assert len(result_b) == 1 and result_b[0]["model_id"] == "mb"
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── CloudDB.read_tenant_policy_stats tests ───────────────────────────────────


class TestCloudDBTenantPolicyStats:
    """Unit tests for CloudDB.read_tenant_policy_stats(tenant_id) — SQL path."""

    def _db(self, path: str):
        from squish.squash.cloud_db import CloudDB
        return CloudDB(path)

    def test_read_tenant_policy_stats_returns_empty_on_fresh_db(self) -> None:
        """A fresh tenant with no evaluations returns {}."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            db.upsert_tenant("ps-t1", {"tenant_id": "ps-t1", "name": "PS T1"})
            assert db.read_tenant_policy_stats("ps-t1") == {}
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_tenant_policy_stats_returns_data_after_write(self) -> None:
        """inc_policy_stat → read_tenant_policy_stats returns accurate counts."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            db.upsert_tenant("ps-t2", {"tenant_id": "ps-t2", "name": "PS T2"})
            db.inc_policy_stat("ps-t2", "no_loose_deps", passed=True)
            db.inc_policy_stat("ps-t2", "no_loose_deps", passed=False)
            result = db.read_tenant_policy_stats("ps-t2")
            assert "no_loose_deps" in result
            assert result["no_loose_deps"]["passed"] == 1
            assert result["no_loose_deps"]["failed"] == 1
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_tenant_policy_stats_unknown_tenant_returns_empty(self) -> None:
        """Querying a tenant that was never upserted returns {}."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            assert db.read_tenant_policy_stats("ghost-ps-tenant") == {}
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_read_tenant_policy_stats_isolates_by_tenant(self) -> None:
        """Policies for tenant A are invisible from tenant B's read."""
        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = self._db(path)
            db.upsert_tenant("ps-ta", {"tenant_id": "ps-ta", "name": "PS TA"})
            db.upsert_tenant("ps-tb", {"tenant_id": "ps-tb", "name": "PS TB"})
            db.inc_policy_stat("ps-ta", "policy-a", passed=True)
            db.inc_policy_stat("ps-tb", "policy-b", passed=False)
            result_a = db.read_tenant_policy_stats("ps-ta")
            result_b = db.read_tenant_policy_stats("ps-tb")
            assert "policy-a" in result_a and "policy-b" not in result_a
            assert "policy-b" in result_b and "policy-a" not in result_b
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── GET /cloud/tenants/{id}/drift-events tests ───────────────────────────────


class TestCloudAPIDriftEventsEndpoint:
    """Tests for GET /cloud/tenants/{tenant_id}/drift-events (in-memory path)."""

    @pytest.fixture(autouse=True)
    def _client(self) -> None:
        from squish.squash.api import app, _tenants, _inventory, _vex_alerts, _drift_events, _policy_stats, _rate_window

        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        _rate_window.clear()  # reset per-IP sliding-window so full-suite runs never exhaust the bucket
        self.client = TestClient(app, raise_server_exceptions=True)

    def _create_tenant(self, tenant_id: str) -> None:
        resp = self.client.post("/cloud/tenant", json={"tenant_id": tenant_id, "name": f"Tenant {tenant_id}"})
        assert resp.status_code in (200, 201)

    def test_returns_empty_list_for_known_tenant(self) -> None:
        """Fresh tenant → endpoint returns count=0, events=[]."""
        self._create_tenant("de-api-empty")
        resp = self.client.get("/cloud/tenants/de-api-empty/drift-events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "de-api-empty"
        assert body["count"] == 0
        assert body["events"] == []

    def test_returns_data_after_drift_event_write(self) -> None:
        """POST /cloud/drift/event → GET returns the ingested event."""
        self._create_tenant("de-api-write")
        self.client.post("/cloud/drift/event", json={
            "tenant_id": "de-api-write",
            "model_id": "test-model",
            "bom_a": "/boms/v1.json",
            "bom_b": "/boms/v2.json",
            "severity": "warning",
        })
        resp = self.client.get("/cloud/tenants/de-api-write/drift-events")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 1
        assert body["events"][0]["model_id"] == "test-model"

    def test_unknown_tenant_returns_404(self) -> None:
        """Requesting drift events for an unregistered tenant_id returns 404."""
        resp = self.client.get("/cloud/tenants/ghost-de-tenant/drift-events")
        assert resp.status_code == 404

    def test_isolates_by_tenant(self) -> None:
        """Drift events written for tenant A do not appear under tenant B."""
        self._create_tenant("de-iso-a")
        self._create_tenant("de-iso-b")
        self.client.post("/cloud/drift/event", json={
            "tenant_id": "de-iso-a",
            "model_id": "model-a",
            "bom_a": "/a1.json",
            "bom_b": "/a2.json",
        })
        resp_a = self.client.get("/cloud/tenants/de-iso-a/drift-events")
        resp_b = self.client.get("/cloud/tenants/de-iso-b/drift-events")
        assert resp_a.json()["count"] == 1
        assert resp_b.json()["count"] == 0


# ── GET /cloud/tenants/{id}/policy-stats tests ───────────────────────────────


class TestCloudAPITenantPolicyStatsEndpoint:
    """Tests for GET /cloud/tenants/{tenant_id}/policy-stats (in-memory path)."""

    @pytest.fixture(autouse=True)
    def _client(self) -> None:
        from squish.squash.api import app, _tenants, _inventory, _vex_alerts, _drift_events, _policy_stats, _rate_window

        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        _rate_window.clear()  # reset per-IP sliding-window so full-suite runs never exhaust the bucket
        self.client = TestClient(app, raise_server_exceptions=True)

    def _create_tenant(self, tenant_id: str) -> None:
        resp = self.client.post("/cloud/tenant", json={"tenant_id": tenant_id, "name": f"Tenant {tenant_id}"})
        assert resp.status_code in (200, 201)

    def test_returns_empty_dict_for_known_tenant(self) -> None:
        """Fresh tenant → endpoint returns count=0, stats={}."""
        self._create_tenant("ps-api-empty")
        resp = self.client.get("/cloud/tenants/ps-api-empty/policy-stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["tenant_id"] == "ps-api-empty"
        assert body["count"] == 0
        assert body["stats"] == {}

    def test_returns_data_after_policy_eval(self) -> None:
        """POST /cloud/inventory/register → GET shows per-tenant policy stats."""
        self._create_tenant("ps-api-write")
        self.client.post("/cloud/inventory/register", json={
            "tenant_id": "ps-api-write",
            "model_id": "test-model",
            "model_path": "/models/test",
            "attestation_passed": True,
            "policy_results": {
                "enterprise-strict": {"passed": True, "error_count": 0, "warning_count": 0},
            },
        })
        resp = self.client.get("/cloud/tenants/ps-api-write/policy-stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] >= 1
        assert "enterprise-strict" in body["stats"]

    def test_unknown_tenant_returns_404(self) -> None:
        """Requesting policy stats for an unregistered tenant_id returns 404."""
        resp = self.client.get("/cloud/tenants/ghost-ps-tenant/policy-stats")
        assert resp.status_code == 404

    def test_isolates_by_tenant(self) -> None:
        """Policy stats written for tenant A do not appear under tenant B."""
        self._create_tenant("ps-iso-a")
        self._create_tenant("ps-iso-b")
        self.client.post("/cloud/inventory/register", json={
            "tenant_id": "ps-iso-a",
            "model_id": "model-a",
            "model_path": "/a",
            "attestation_passed": True,
            "policy_results": {
                "iso-policy": {"passed": True, "error_count": 0, "warning_count": 0},
            },
        })
        resp_a = self.client.get("/cloud/tenants/ps-iso-a/policy-stats")
        resp_b = self.client.get("/cloud/tenants/ps-iso-b/policy-stats")
        assert "iso-policy" in resp_a.json()["stats"]
        assert "iso-policy" not in resp_b.json()["stats"]
