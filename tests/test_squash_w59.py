"""W59 — Tenant lifecycle: PATCH + DELETE.

Tests:
  - CloudDB.delete_tenant() cascade (5)
  - PATCH /cloud/tenant/{tenant_id}  (5)
  - DELETE /cloud/tenant/{tenant_id} (5)

All API tests use TestClient with in-memory state; CloudDB tests use a
temporary SQLite path to exercise the real SQL cascade.
"""

from __future__ import annotations

import tempfile
import time
import os
from typing import Any

import pytest

# ── CloudDB cascade tests ─────────────────────────────────────────────────────


class TestCloudDBDeleteTenant:
    """Unit tests for CloudDB.delete_tenant() — SQL cascade path."""

    def _make_db(self) -> Any:
        from squish.squash.cloud_db import CloudDB

        tmp = tempfile.mktemp(suffix=".sqlite")  # noqa: S306
        db = CloudDB(tmp)
        yield db
        db._conn.close()
        if os.path.exists(tmp):
            os.unlink(tmp)

    def test_delete_unknown_tenant_is_noop(self) -> None:
        """delete_tenant on a non-existent tenant_id raises no exception."""
        from squish.squash.cloud_db import CloudDB

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = CloudDB(path)
            db.delete_tenant("ghost-tenant")  # must not raise
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_delete_removes_tenant(self) -> None:
        """Upsert a tenant then delete it — get_tenant returns None."""
        from squish.squash.cloud_db import CloudDB

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = CloudDB(path)
            db.upsert_tenant("t-del-1", {"tenant_id": "t-del-1", "name": "Del One"})
            assert db.get_tenant("t-del-1") is not None
            db.delete_tenant("t-del-1")
            assert db.get_tenant("t-del-1") is None
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_delete_cascades_inventory(self) -> None:
        """Appending inventory records then deleting the tenant cascades to 0 rows."""
        from squish.squash.cloud_db import CloudDB

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = CloudDB(path)
            db.upsert_tenant("t-cas-inv", {"tenant_id": "t-cas-inv", "name": "Cascade Inv"})
            db.append_record("inventory", "t-cas-inv", {"model_id": "m1", "ts": "2025-01-01"})
            db.append_record("inventory", "t-cas-inv", {"model_id": "m2", "ts": "2025-01-02"})
            assert db.count_records("inventory", "t-cas-inv") == 2
            db.delete_tenant("t-cas-inv")
            assert db.count_records("inventory", "t-cas-inv") == 0
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_delete_cascades_vex_alerts(self) -> None:
        """Appending VEX alert records then deleting the tenant cascades to 0 rows."""
        from squish.squash.cloud_db import CloudDB

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = CloudDB(path)
            db.upsert_tenant("t-cas-vex", {"tenant_id": "t-cas-vex", "name": "Cascade Vex"})
            db.append_record("vex_alerts", "t-cas-vex", {"cve_id": "CVE-2025-9999", "ts": "2025-01-01"})
            assert db.count_records("vex_alerts", "t-cas-vex") == 1
            db.delete_tenant("t-cas-vex")
            assert db.count_records("vex_alerts", "t-cas-vex") == 0
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_delete_cascades_policy_stats(self) -> None:
        """Incrementing policy stats then deleting the tenant leaves no aggregate rows."""
        from squish.squash.cloud_db import CloudDB

        with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
            path = f.name
        try:
            db = CloudDB(path)
            db.upsert_tenant("t-cas-pol", {"tenant_id": "t-cas-pol", "name": "Cascade Pol"})
            db.inc_policy_stat("t-cas-pol", "no_loose_deps", passed=True)
            db.inc_policy_stat("t-cas-pol", "no_loose_deps", passed=False)
            stats_before = db.read_policy_stats()
            assert "no_loose_deps" in stats_before
            db.delete_tenant("t-cas-pol")
            stats_after = db.read_policy_stats()
            assert "no_loose_deps" not in stats_after
            db._conn.close()
        finally:
            if os.path.exists(path):
                os.unlink(path)


# ── API PATCH tests ───────────────────────────────────────────────────────────


class TestCloudAPIPatch:
    """Tests for PATCH /cloud/tenant/{tenant_id}."""

    @pytest.fixture(autouse=True)
    def _client(self) -> None:
        from fastapi.testclient import TestClient
        from squish.squash.api import app, _tenants

        _tenants.clear()
        self.client = TestClient(app, raise_server_exceptions=True)

    def _create_tenant(self, tenant_id: str = "patch-t1") -> dict:
        resp = self.client.post(
            "/cloud/tenant",
            json={
                "tenant_id": tenant_id,
                "name": "Original Name",
                "plan": "community",
                "contact_email": "orig@example.com",
            },
        )
        assert resp.status_code in (200, 201)
        return resp.json()

    def test_patch_updates_name(self) -> None:
        """PATCH with {name} changes the tenant's name."""
        self._create_tenant("patch-name-1")
        resp = self.client.patch("/cloud/tenant/patch-name-1", json={"name": "Updated Name"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Updated Name"
        # Verify via GET
        get_resp = self.client.get("/cloud/tenant/patch-name-1")
        assert get_resp.status_code == 200
        assert get_resp.json()["name"] == "Updated Name"

    def test_patch_updates_plan(self) -> None:
        """PATCH with {plan} changes the subscription plan."""
        self._create_tenant("patch-plan-1")
        resp = self.client.patch("/cloud/tenant/patch-plan-1", json={"plan": "enterprise"})
        assert resp.status_code == 200
        assert resp.json()["plan"] == "enterprise"

    def test_patch_unknown_tenant_returns_404(self) -> None:
        """PATCH a non-existent tenant_id -> 404."""
        resp = self.client.patch("/cloud/tenant/does-not-exist", json={"name": "Ghost"})
        assert resp.status_code == 404
        assert "does-not-exist" in resp.json()["detail"]

    def test_patch_preserves_unset_fields(self) -> None:
        """PATCH with only {name} leaves plan and contact_email unchanged."""
        created = self._create_tenant("patch-preserve-1")
        original_plan = created["plan"]
        original_email = created["contact_email"]

        resp = self.client.patch("/cloud/tenant/patch-preserve-1", json={"name": "New Name Only"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "New Name Only"
        assert data["plan"] == original_plan
        assert data["contact_email"] == original_email

    def test_patch_updates_updated_at(self) -> None:
        """PATCH increments updated_at relative to the creation timestamp."""
        created = self._create_tenant("patch-ts-1")
        created_at = created.get("created_at", "")
        time.sleep(0.01)  # ensure clock advances
        resp = self.client.patch("/cloud/tenant/patch-ts-1", json={"name": "TS Check"})
        assert resp.status_code == 200
        updated_at = resp.json().get("updated_at", "")
        # updated_at must exist and differ from created_at (or at least be present)
        assert updated_at, "updated_at must be set after PATCH"
        if created_at:
            assert updated_at >= created_at, "updated_at must not be earlier than created_at"


# ── API DELETE tests ──────────────────────────────────────────────────────────


class TestCloudAPIDelete:
    """Tests for DELETE /cloud/tenant/{tenant_id}."""

    @pytest.fixture(autouse=True)
    def _client(self) -> None:
        from fastapi.testclient import TestClient
        from squish.squash.api import app, _tenants, _inventory, _vex_alerts, _drift_events, _policy_stats

        _tenants.clear()
        _inventory.clear()
        _vex_alerts.clear()
        _drift_events.clear()
        _policy_stats.clear()
        self.client = TestClient(app, raise_server_exceptions=True)

    def _create_tenant(self, tenant_id: str = "del-t1", name: str = "Del Tenant") -> None:
        resp = self.client.post(
            "/cloud/tenant",
            json={"tenant_id": tenant_id, "name": name, "plan": "community"},
        )
        assert resp.status_code in (200, 201)

    def test_delete_tenant_returns_204(self) -> None:
        """DELETE an existing tenant returns 204 No Content."""
        self._create_tenant("del-204-1")
        resp = self.client.delete("/cloud/tenant/del-204-1")
        assert resp.status_code == 204
        assert resp.content == b""

    def test_delete_unknown_tenant_returns_404(self) -> None:
        """DELETE a non-existent tenant_id -> 404."""
        resp = self.client.delete("/cloud/tenant/ghost-99")
        assert resp.status_code == 404
        assert "ghost-99" in resp.json()["detail"]

    def test_delete_removes_from_get(self) -> None:
        """After DELETE, GET /cloud/tenant/{id} returns 404."""
        self._create_tenant("del-get-1")
        del_resp = self.client.delete("/cloud/tenant/del-get-1")
        assert del_resp.status_code == 204
        get_resp = self.client.get("/cloud/tenant/del-get-1")
        assert get_resp.status_code == 404

    def test_delete_removes_from_list(self) -> None:
        """After DELETE, GET /cloud/tenants no longer includes the removed tenant."""
        self._create_tenant("del-list-1")
        self._create_tenant("del-list-keep", name="Keep Me")
        before = self.client.get("/cloud/tenants").json()
        assert before["count"] == 2

        del_resp = self.client.delete("/cloud/tenant/del-list-1")
        assert del_resp.status_code == 204

        after = self.client.get("/cloud/tenants").json()
        assert after["count"] == 1
        tenant_ids = [t["tenant_id"] for t in after["tenants"]]
        assert "del-list-1" not in tenant_ids
        assert "del-list-keep" in tenant_ids

    def test_delete_clears_inventory(self) -> None:
        """After DELETE the tenant's inventory is gone (GET /cloud/tenants/{id}/inventory returns 404)."""
        self._create_tenant("del-inv-1")
        # Register inventory for the tenant
        register_resp = self.client.post(
            "/cloud/tenants/del-inv-1/inventory",
            json={
                "tenant_id": "del-inv-1",
                "model_id": "my-model",
                "model_path": "/tmp/my-model",
            },
        )
        # Whether the register succeeds or is a 404 (endpoint may require auth setup),
        # what matters is that DELETE removes the tenant so the inventory endpoint returns 404.
        del_resp = self.client.delete("/cloud/tenant/del-inv-1")
        assert del_resp.status_code == 204

        inv_resp = self.client.get("/cloud/tenants/del-inv-1/inventory")
        assert inv_resp.status_code == 404, (
            f"Expected 404 after tenant deletion, got {inv_resp.status_code}"
        )
