"""W67 — Azure DevOps pipeline attestation result ingest.

Tests for:
  - CloudDB.append_ado_result() / read_ado_results() (8 unit tests)
  - POST /cloud/tenants/{tenant_id}/ado-result  (status 201)
  - GET  /cloud/tenants/{tenant_id}/ado-results (status 200)
  (8 API integration tests)

EU AI Act Art. 9 / ISO 42001 §8.4 — operators require a durable, per-tenant
record of Azure DevOps pipeline attestation outcomes for supply-chain traceability.
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

class TestCloudDBAdoResults:
    """8 unit tests for CloudDB ADO result persistence layer."""

    def setup_method(self) -> None:
        self._db = _make_db()

    def test_returns_list(self) -> None:
        result = self._db.read_ado_results("alpha")
        assert isinstance(result, list)

    def test_empty_returns_empty(self) -> None:
        assert self._db.read_ado_results("alpha") == []

    def test_append_stores_result(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-42", True)
        results = self._db.read_ado_results("alpha")
        assert len(results) == 1

    def test_passed_true_stored(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-1", True)
        assert self._db.read_ado_results("alpha")[0]["passed"] is True

    def test_passed_false_stored(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-2", False)
        assert self._db.read_ado_results("alpha")[0]["passed"] is False

    def test_variables_preserved(self) -> None:
        _register(self._db, "alpha")
        variables = {"BUILD_ID": "123", "ENV": "prod"}
        self._db.append_ado_result("alpha", "run-3", True, variables)
        stored = self._db.read_ado_results("alpha")[0]["variables"]
        assert stored == variables

    def test_variables_none_when_not_provided(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-4", True)
        assert self._db.read_ado_results("alpha")[0]["variables"] is None

    def test_multi_tenant_isolated(self) -> None:
        _register(self._db, "alpha")
        _register(self._db, "beta")
        self._db.append_ado_result("alpha", "run-a", True)
        self._db.append_ado_result("beta", "run-b", False)
        alpha = self._db.read_ado_results("alpha")
        beta = self._db.read_ado_results("beta")
        assert len(alpha) == 1
        assert alpha[0]["pipeline_run_id"] == "run-a"
        assert len(beta) == 1
        assert beta[0]["pipeline_run_id"] == "run-b"


# ── API integration tests ─────────────────────────────────────────────────────

class TestCloudAPIAdoResults:
    """8 integration tests for POST/GET /cloud/tenants/{tenant_id}/ado-result(s)."""

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

    def test_post_returns_201(self) -> None:
        resp = self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": True},
        )
        assert resp.status_code == 201

    def test_post_missing_run_id_returns_422(self) -> None:
        resp = self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"passed": True},
        )
        assert resp.status_code == 422

    def test_post_response_keys(self) -> None:
        resp = self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": True},
        )
        body = resp.json()
        assert "status" in body
        assert "tenant_id" in body
        assert "passed" in body

    def test_get_returns_200(self) -> None:
        resp = self.client.get("/cloud/tenants/acme/ado-results")
        assert resp.status_code == 200

    def test_get_empty_results(self) -> None:
        resp = self.client.get("/cloud/tenants/acme/ado-results")
        body = resp.json()
        assert body["results"] == []

    def test_get_after_post_has_result(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": True},
        )
        resp = self.client.get("/cloud/tenants/acme/ado-results")
        assert len(resp.json()["results"]) == 1

    def test_passed_false_stored(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-fail", "passed": False},
        )
        result = self.client.get("/cloud/tenants/acme/ado-results").json()["results"][0]
        assert result["passed"] is False

    def test_variables_in_get_response(self) -> None:
        variables = {"DEPLOY_ENV": "staging", "IMAGE_TAG": "v1.2.3"}
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={
                "pipeline_run_id": "run-vars",
                "passed": True,
                "variables": variables,
            },
        )
        result = self.client.get("/cloud/tenants/acme/ado-results").json()["results"][0]
        assert result["variables"] == variables
