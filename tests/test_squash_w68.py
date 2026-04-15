"""W68 — Per-tenant combined attestation score.

Tests for:
  - CloudDB.read_attestation_score()   (8 unit tests)
  - GET /cloud/tenants/{tenant_id}/attestation-score  (8 API integration tests)

Aggregates GCP Vertex AI (W66) and Azure DevOps (W67) attestation results into
a single ``{total, passed, failed, pass_rate}`` summary per tenant.

EU AI Act Art. 9 — operators require a continuous, cross-platform measure of
supply-chain attestation health to satisfy conformity-assessment obligations.
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

class TestCloudDBAttestationScore:
    """8 unit tests for CloudDB.read_attestation_score()."""

    def setup_method(self) -> None:
        self._db = _make_db()

    def test_returns_dict(self) -> None:
        result = self._db.read_attestation_score("alpha")
        assert isinstance(result, dict)

    def test_empty_tenant_all_zeros(self) -> None:
        score = self._db.read_attestation_score("alpha")
        assert score == {"total": 0, "passed": 0, "failed": 0, "pass_rate": 0.0}

    def test_vertex_only_counts(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        score = self._db.read_attestation_score("alpha")
        assert score["total"] == 1
        assert score["passed"] == 1
        assert score["failed"] == 0
        assert score["pass_rate"] == 1.0

    def test_ado_only_counts(self) -> None:
        _register(self._db, "alpha")
        self._db.append_ado_result("alpha", "run-1", False)
        score = self._db.read_attestation_score("alpha")
        assert score["total"] == 1
        assert score["passed"] == 0
        assert score["failed"] == 1
        assert score["pass_rate"] == 0.0

    def test_mixed_sources_total(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        self._db.append_ado_result("alpha", "run-1", True)
        score = self._db.read_attestation_score("alpha")
        assert score["total"] == 2

    def test_pass_rate_calculation(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m1", True)
        self._db.append_ado_result("alpha", "run-1", True)
        self._db.append_ado_result("alpha", "run-2", False)
        score = self._db.read_attestation_score("alpha")
        assert score["total"] == 3
        assert score["passed"] == 2
        assert score["failed"] == 1
        # 2/3 = 0.6667 rounded to 4 dp
        assert abs(score["pass_rate"] - round(2 / 3, 4)) < 1e-9

    def test_failed_count_correct(self) -> None:
        _register(self._db, "alpha")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        self._db.append_ado_result("alpha", "run-fail", False)
        score = self._db.read_attestation_score("alpha")
        assert score["failed"] == 1

    def test_multi_tenant_isolated(self) -> None:
        _register(self._db, "alpha")
        _register(self._db, "beta")
        self._db.append_vertex_result("alpha", "projects/p/models/m", True)
        self._db.append_ado_result("alpha", "run-a", True)
        # beta: no attestations
        score_a = self._db.read_attestation_score("alpha")
        score_b = self._db.read_attestation_score("beta")
        assert score_a["total"] == 2
        assert score_b["total"] == 0


# ── API integration tests ─────────────────────────────────────────────────────

class TestCloudAPIAttestationScore:
    """8 integration tests for GET /cloud/tenants/{tenant_id}/attestation-score."""

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
        resp = self.client.get("/cloud/tenants/acme/attestation-score")
        assert resp.status_code == 200

    def test_response_has_required_keys(self) -> None:
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        for key in ("tenant_id", "total", "passed", "failed", "pass_rate"):
            assert key in body

    def test_empty_tenant_score(self) -> None:
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        assert body["total"] == 0
        assert body["pass_rate"] == 0.0

    def test_after_vertex_post(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/vertex-result",
            json={"model_resource_name": "projects/p/models/m", "passed": True},
        )
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        assert body["total"] == 1
        assert body["passed"] == 1

    def test_after_ado_post(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": True},
        )
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        assert body["total"] == 1
        assert body["passed"] == 1

    def test_mixed_sources_counted(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/vertex-result",
            json={"model_resource_name": "projects/p/models/m", "passed": True},
        )
        self.client.post(
            "/cloud/tenants/acme/ado-result",
            json={"pipeline_run_id": "run-99", "passed": False},
        )
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        assert body["total"] == 2
        assert body["passed"] == 1
        assert body["failed"] == 1

    def test_pass_rate_is_float(self) -> None:
        self.client.post(
            "/cloud/tenants/acme/vertex-result",
            json={"model_resource_name": "projects/p/models/m", "passed": True},
        )
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        assert isinstance(body["pass_rate"], float)

    def test_tenant_id_in_response(self) -> None:
        body = self.client.get("/cloud/tenants/acme/attestation-score").json()
        assert body["tenant_id"] == "acme"
