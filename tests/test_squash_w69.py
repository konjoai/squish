"""W69 — Merged chronological attestation history per tenant.

16 tests: 8 CloudDB unit + 8 API integration.

Tests for:
  - CloudDB.read_attestations()
  - GET /cloud/tenants/{tenant_id}/attestations

Combines GCP Vertex AI (W66) and Azure DevOps (W67) attestation records into
a single, source-tagged history list sorted by timestamp descending.

EU AI Act Art. 12 + Art. 18 — technical documentation and record-keeping
obligations require a complete, auditable attestation trail per tenant.
"""
from __future__ import annotations

from squish.squash.cloud_db import CloudDB
from starlette.testclient import TestClient


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_db() -> CloudDB:
    return CloudDB(path=":memory:")


def _register(db: CloudDB, tenant_id: str) -> None:
    db.upsert_tenant(tenant_id, {"name": tenant_id, "plan": "pro"})


# ── CloudDB unit tests ────────────────────────────────────────────────────────

class TestCloudDBReadAttestations:
    """8 unit tests for CloudDB.read_attestations()."""

    def setup_method(self) -> None:
        self._db = _make_db()

    def test_returns_list(self) -> None:
        result = self._db.read_attestations("alpha")
        assert isinstance(result, list)

    def test_empty_tenant_returns_empty_list(self) -> None:
        assert self._db.read_attestations("alpha") == []

    def test_vertex_result_has_source_field(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        items = self._db.read_attestations("alpha")
        assert len(items) == 1
        assert items[0]["source"] == "vertex"

    def test_ado_result_has_source_field(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-1", False)
        items = self._db.read_attestations("alpha")
        assert len(items) == 1
        assert items[0]["source"] == "ado"

    def test_vertex_item_has_model_resource_name(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        item = self._db.read_attestations("alpha")[0]
        assert item["model_resource_name"] == "projects/p/models/m"

    def test_ado_item_has_pipeline_run_id(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-42", True)
        item = self._db.read_attestations("alpha")[0]
        assert item["pipeline_run_id"] == "run-42"

    def test_mixed_sources_total_count(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        self._db.append_vertex_result("alpha", "projects/p/models/n", False)
        self._db.append_ado_result("alpha", "run-1", True)
        items = self._db.read_attestations("alpha")
        assert len(items) == 3

    def test_multi_tenant_isolated(self) -> None:
        _register(self._db, "alpha")
        _register(self._db, "beta")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        self._db.append_ado_result("beta", "run-1", True)
        assert len(self._db.read_attestations("alpha")) == 1
        assert len(self._db.read_attestations("beta")) == 1
        assert self._db.read_attestations("alpha")[0]["source"] == "vertex"
        assert self._db.read_attestations("beta")[0]["source"] == "ado"


# ── API integration tests ─────────────────────────────────────────────────────

class TestCloudAPIGetAttestations:
    """8 integration tests for GET /cloud/tenants/{tenant_id}/attestations."""

    def setup_method(self) -> None:
        from squish.squash import api

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

    def test_get_returns_200(self) -> None:
        resp = self.client.get("/cloud/tenants/acme/attestations")
        assert resp.status_code == 200

    def test_response_has_required_keys(self) -> None:
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        assert "tenant_id" in body
        assert "attestations" in body

    def test_empty_tenant_returns_empty_list(self) -> None:
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        assert body["attestations"] == []

    def test_after_vertex_post_source_is_vertex(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/vertex-result",
            json={"model_resource_name": "projects/p/models/m", "passed": True},
        )
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        assert len(body["attestations"]) == 1
        assert body["attestations"][0]["source"] == "vertex"

    def test_after_ado_post_source_is_ado(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": False},
        )
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        assert len(body["attestations"]) == 1
        assert body["attestations"][0]["source"] == "ado"

    def test_mixed_sources_total_count(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/vertex-result",
            json={"model_resource_name": "projects/p/models/m", "passed": True},
        )
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": True},
        )
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        assert len(body["attestations"]) == 2

    def test_passed_field_present_in_each_item(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/vertex-result",
            json={"model_resource_name": "projects/p/models/m", "passed": True},
        )
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        for item in body["attestations"]:
            assert "passed" in item

    def test_tenant_id_in_response(self) -> None:
        body = self.client.get("/cloud/tenants/acme/attestations").json()
        assert body["tenant_id"] == "acme"
