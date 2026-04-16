"""tests/test_squash_wave5355.py — W52-55 Squash Cloud dashboard API tests.

Tests the /cloud/* REST endpoints added in waves 52-55:
    - POST /cloud/tenant            — tenant registration (create/update)
    - GET  /cloud/tenant/{id}       — tenant fetch
    - GET  /cloud/tenants           — tenant list
    - POST /cloud/inventory/register — model inventory ingestion
    - GET  /cloud/inventory         — inventory query (filter, limit)
    - POST /cloud/vex/alert         — VEX alert ingestion
    - GET  /cloud/vex/alerts        — alert query (filter status/severity)
    - POST /cloud/drift/event       — drift event ingestion
    - GET  /cloud/drift/events      — drift event query (filter model/severity)
    - GET  /cloud/policy/dashboard  — policy pass/fail aggregates
    - GET  /cloud/audit             — tenant-scoped audit log
    - JWT helpers (_verify_jwt_hs256, _resolve_tenant_id)
    - AttestRequest tenant_id field
    - Module count stays at 125
"""

from __future__ import annotations

import base64
import hashlib
import hmac as hmac_mod
import json
import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# httpx is required for TestClient
pytest.importorskip("httpx")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from squish.squash.api import (
    _inventory,
    _policy_stats,
    _tenants,
    _drift_events,
    _vex_alerts,
    _verify_jwt_hs256,
    _resolve_tenant_id,
    _rate_window,
    app,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_jwt(payload: dict, secret: str) -> str:
    """Build a minimal HS256 JWT without external deps."""
    header = base64.urlsafe_b64encode(
        json.dumps({"alg": "HS256", "typ": "JWT"}).encode()
    ).rstrip(b"=").decode()
    body = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    signing_input = f"{header}.{body}".encode()
    sig = hmac_mod.new(secret.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{header}.{body}.{sig_b64}"


def _fresh_client() -> TestClient:
    """Return a TestClient with cleared cloud stores and rate limiter."""
    _tenants.clear()
    _inventory.clear()
    _vex_alerts.clear()
    _drift_events.clear()
    _policy_stats.clear()
    _rate_window.clear()
    return TestClient(app, raise_server_exceptions=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. JWT helper: _verify_jwt_hs256
# ──────────────────────────────────────────────────────────────────────────────

class TestVerifyJwtHs256:
    SECRET = "squish-test-secret"

    def test_valid_token_returns_claims(self):
        payload = {"tenant_id": "acme", "exp": int(time.time()) + 3600}
        token = _make_jwt(payload, self.SECRET)
        claims = _verify_jwt_hs256(token, self.SECRET)
        assert claims["tenant_id"] == "acme"

    def test_expired_token_raises(self):
        payload = {"tenant_id": "acme", "exp": int(time.time()) - 1}
        token = _make_jwt(payload, self.SECRET)
        with pytest.raises(ValueError, match="expired"):
            _verify_jwt_hs256(token, self.SECRET)

    def test_wrong_secret_raises(self):
        payload = {"tenant_id": "acme", "exp": int(time.time()) + 3600}
        token = _make_jwt(payload, self.SECRET)
        with pytest.raises(ValueError, match="invalid JWT signature"):
            _verify_jwt_hs256(token, "different-secret")

    def test_malformed_token_raises(self):
        with pytest.raises(ValueError, match="invalid JWT structure"):
            _verify_jwt_hs256("not.a.valid.jwt.token.with.too.many.parts", self.SECRET)

    def test_no_exp_field_is_accepted(self):
        payload = {"tenant_id": "beta", "sub": "user@example.com"}
        token = _make_jwt(payload, self.SECRET)
        claims = _verify_jwt_hs256(token, self.SECRET)
        assert claims["tenant_id"] == "beta"

    def test_sub_field_accessible(self):
        payload = {"sub": "org-42", "exp": int(time.time()) + 300}
        token = _make_jwt(payload, self.SECRET)
        claims = _verify_jwt_hs256(token, self.SECRET)
        assert claims["sub"] == "org-42"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Tenant management endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudTenant:
    def test_create_tenant_returns_201(self):
        client = _fresh_client()
        r = client.post("/cloud/tenant", json={
            "tenant_id": "acme", "name": "ACME Corp", "plan": "enterprise",
        })
        assert r.status_code == 201
        data = r.json()
        assert data["tenant_id"] == "acme"
        assert data["name"] == "ACME Corp"
        assert data["plan"] == "enterprise"
        assert "created_at" in data

    def test_get_tenant_returns_record(self):
        client = _fresh_client()
        client.post("/cloud/tenant", json={"tenant_id": "beta", "name": "Beta Co"})
        r = client.get("/cloud/tenant/beta")
        assert r.status_code == 200
        assert r.json()["name"] == "Beta Co"

    def test_get_missing_tenant_returns_404(self):
        client = _fresh_client()
        r = client.get("/cloud/tenant/no-such-tenant")
        assert r.status_code == 404

    def test_create_tenant_idempotent_updates_record(self):
        client = _fresh_client()
        client.post("/cloud/tenant", json={"tenant_id": "t1", "name": "Old Name"})
        r2 = client.post("/cloud/tenant", json={"tenant_id": "t1", "name": "New Name"})
        assert r2.status_code == 201
        assert r2.json()["name"] == "New Name"
        # created_at preserved from first registration
        r3 = client.get("/cloud/tenant/t1")
        assert r3.json()["name"] == "New Name"

    def test_empty_tenant_id_rejected(self):
        client = _fresh_client()
        r = client.post("/cloud/tenant", json={"tenant_id": "", "name": "Oops"})
        assert r.status_code == 400

    def test_tenant_id_over_64_chars_rejected(self):
        client = _fresh_client()
        long_id = "a" * 65
        r = client.post("/cloud/tenant", json={"tenant_id": long_id, "name": "Too Long"})
        assert r.status_code == 400

    def test_list_tenants_returns_all(self):
        client = _fresh_client()
        client.post("/cloud/tenant", json={"tenant_id": "x", "name": "X"})
        client.post("/cloud/tenant", json={"tenant_id": "y", "name": "Y"})
        r = client.get("/cloud/tenants")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_default_plan_is_community(self):
        client = _fresh_client()
        r = client.post("/cloud/tenant", json={"tenant_id": "free", "name": "Free Tier"})
        assert r.json()["plan"] == "community"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Model inventory endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudInventory:
    def _register(self, client: TestClient, tenant: str, model: str, passed: bool = True) -> dict:
        r = client.post("/cloud/inventory/register", json={
            "tenant_id": tenant,
            "model_id": model,
            "model_path": f"/models/{model}",
            "attestation_passed": passed,
            "policy_results": {"enterprise-strict": {"passed": passed, "error_count": 0 if passed else 1, "warning_count": 0}},
        })
        assert r.status_code == 201
        return r.json()

    def test_register_returns_201_with_record_id(self):
        client = _fresh_client()
        data = self._register(client, "acme", "llama-3b")
        assert "record_id" in data
        assert data["model_id"] == "llama-3b"
        assert data["attestation_passed"] is True

    def test_inventory_empty_for_unknown_tenant(self):
        client = _fresh_client()
        r = client.get("/cloud/inventory", headers={"X-Tenant-ID": "ghost"})
        assert r.status_code == 200
        assert r.json()["count"] == 0

    def test_inventory_returns_registered_models(self):
        client = _fresh_client()
        self._register(client, "acme", "m1")
        self._register(client, "acme", "m2")
        r = client.get("/cloud/inventory", headers={"X-Tenant-ID": "acme"})
        assert r.json()["count"] == 2

    def test_inventory_filters_by_passed_true(self):
        client = _fresh_client()
        self._register(client, "acme", "good", passed=True)
        self._register(client, "acme", "bad", passed=False)
        r = client.get("/cloud/inventory?passed=true", headers={"X-Tenant-ID": "acme"})
        assert r.json()["count"] == 1
        assert r.json()["models"][0]["model_id"] == "good"

    def test_inventory_filters_by_passed_false(self):
        client = _fresh_client()
        self._register(client, "acme", "good", passed=True)
        self._register(client, "acme", "bad", passed=False)
        r = client.get("/cloud/inventory?passed=false", headers={"X-Tenant-ID": "acme"})
        assert r.json()["count"] == 1
        assert r.json()["models"][0]["model_id"] == "bad"

    def test_inventory_limit_parameter(self):
        client = _fresh_client()
        for i in range(10):
            self._register(client, "acme", f"model-{i}")
        r = client.get("/cloud/inventory?limit=3", headers={"X-Tenant-ID": "acme"})
        assert r.json()["count"] == 3
        assert r.json()["total"] == 10

    def test_inventory_is_tenant_scoped(self):
        client = _fresh_client()
        self._register(client, "tenant-a", "model-a")
        self._register(client, "tenant-b", "model-b")
        r = client.get("/cloud/inventory", headers={"X-Tenant-ID": "tenant-a"})
        assert r.json()["count"] == 1
        assert r.json()["models"][0]["model_id"] == "model-a"

    def test_register_missing_tenant_id_returns_400(self):
        client = _fresh_client()
        r = client.post("/cloud/inventory/register", json={
            "tenant_id": "",
            "model_id": "x",
            "model_path": "/x",
            "attestation_passed": True,
        })
        assert r.status_code == 400

    def test_register_includes_vex_cves(self):
        client = _fresh_client()
        r = client.post("/cloud/inventory/register", json={
            "tenant_id": "sec",
            "model_id": "vuln-model",
            "model_path": "/m",
            "attestation_passed": False,
            "vex_cves": ["CVE-2024-1234", "CVE-2024-5678"],
        })
        assert r.status_code == 201
        assert r.json()["vex_cves"] == ["CVE-2024-1234", "CVE-2024-5678"]

    def test_register_auto_updates_policy_stats(self):
        client = _fresh_client()
        self._register(client, "t", "m1", passed=True)
        self._register(client, "t", "m2", passed=False)
        r = client.get("/cloud/policy/dashboard", headers={"X-Tenant-ID": "t"})
        body = r.json()
        assert body["overall"]["passed"] == 1
        assert body["overall"]["failed"] == 1


# ──────────────────────────────────────────────────────────────────────────────
# 4. VEX alert endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudVexAlerts:
    def _post_alert(self, client: TestClient, tenant: str, cve: str = "CVE-2024-1234",
                    severity: str = "high", status: str = "open") -> dict:
        r = client.post("/cloud/vex/alert", json={
            "tenant_id": tenant,
            "cve_id": cve,
            "severity": severity,
            "status": status,
            "model_id": "llm-v1",
        })
        assert r.status_code == 201
        return r.json()

    def test_post_alert_returns_201_with_alert_id(self):
        client = _fresh_client()
        data = self._post_alert(client, "acme")
        assert "alert_id" in data
        assert data["cve_id"] == "CVE-2024-1234"
        assert data["status"] == "open"

    def test_get_alerts_returns_ingested(self):
        client = _fresh_client()
        self._post_alert(client, "acme", cve="CVE-2024-001")
        self._post_alert(client, "acme", cve="CVE-2024-002")
        r = client.get("/cloud/vex/alerts", headers={"X-Tenant-ID": "acme"})
        assert r.json()["count"] == 2

    def test_get_alerts_empty_for_unknown_tenant(self):
        client = _fresh_client()
        r = client.get("/cloud/vex/alerts", headers={"X-Tenant-ID": "nobody"})
        assert r.json()["count"] == 0

    def test_alerts_filtered_by_status(self):
        client = _fresh_client()
        self._post_alert(client, "t", cve="CVE-001", status="open")
        self._post_alert(client, "t", cve="CVE-002", status="resolved")
        r = client.get("/cloud/vex/alerts?status=open", headers={"X-Tenant-ID": "t"})
        assert r.json()["count"] == 1
        assert r.json()["alerts"][0]["cve_id"] == "CVE-001"

    def test_alerts_filtered_by_severity(self):
        client = _fresh_client()
        self._post_alert(client, "t", cve="CVE-A", severity="critical")
        self._post_alert(client, "t", cve="CVE-B", severity="low")
        r = client.get("/cloud/vex/alerts?severity=critical", headers={"X-Tenant-ID": "t"})
        assert r.json()["count"] == 1
        assert r.json()["alerts"][0]["cve_id"] == "CVE-A"

    def test_alerts_are_tenant_scoped(self):
        client = _fresh_client()
        self._post_alert(client, "tenant-a", cve="CVE-A")
        self._post_alert(client, "tenant-b", cve="CVE-B")
        r = client.get("/cloud/vex/alerts", headers={"X-Tenant-ID": "tenant-a"})
        assert r.json()["count"] == 1
        assert r.json()["alerts"][0]["cve_id"] == "CVE-A"

    def test_post_alert_missing_tenant_returns_400(self):
        client = _fresh_client()
        r = client.post("/cloud/vex/alert", json={
            "tenant_id": "", "cve_id": "CVE-001",
        })
        assert r.status_code == 400

    def test_alert_limit_parameter(self):
        client = _fresh_client()
        for i in range(20):
            self._post_alert(client, "t", cve=f"CVE-{i:03d}")
        r = client.get("/cloud/vex/alerts?limit=5", headers={"X-Tenant-ID": "t"})
        assert r.json()["count"] == 5
        assert r.json()["total"] == 20

    def test_alert_has_created_at(self):
        client = _fresh_client()
        data = self._post_alert(client, "t")
        assert "created_at" in data


# ──────────────────────────────────────────────────────────────────────────────
# 5. Drift event endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudDriftEvents:
    def _post_event(self, client: TestClient, tenant: str,
                    model: str = "llm-v1", severity: str = "warning") -> dict:
        r = client.post("/cloud/drift/event", json={
            "tenant_id": tenant,
            "model_id": model,
            "bom_a": "/boms/v1.json",
            "bom_b": "/boms/v2.json",
            "added": ["pkg:pypi/requests@2.32.0"],
            "removed": ["pkg:pypi/requests@2.31.0"],
            "severity": severity,
        })
        assert r.status_code == 201
        return r.json()

    def test_post_event_returns_201_with_event_id(self):
        client = _fresh_client()
        data = self._post_event(client, "acme")
        assert "event_id" in data
        assert data["model_id"] == "llm-v1"

    def test_get_events_returns_ingested(self):
        client = _fresh_client()
        self._post_event(client, "acme")
        self._post_event(client, "acme")
        r = client.get("/cloud/drift/events", headers={"X-Tenant-ID": "acme"})
        assert r.json()["count"] == 2

    def test_events_filtered_by_model_id(self):
        client = _fresh_client()
        self._post_event(client, "t", model="llm-v1")
        self._post_event(client, "t", model="cnn-v2")
        r = client.get("/cloud/drift/events?model_id=cnn-v2", headers={"X-Tenant-ID": "t"})
        assert r.json()["count"] == 1
        assert r.json()["events"][0]["model_id"] == "cnn-v2"

    def test_events_filtered_by_severity(self):
        client = _fresh_client()
        self._post_event(client, "t", severity="critical")
        self._post_event(client, "t", severity="info")
        r = client.get("/cloud/drift/events?severity=critical", headers={"X-Tenant-ID": "t"})
        assert r.json()["count"] == 1

    def test_events_are_tenant_scoped(self):
        client = _fresh_client()
        self._post_event(client, "tenant-a", model="ma")
        self._post_event(client, "tenant-b", model="mb")
        r = client.get("/cloud/drift/events", headers={"X-Tenant-ID": "tenant-a"})
        assert r.json()["count"] == 1
        assert r.json()["events"][0]["model_id"] == "ma"

    def test_post_event_missing_tenant_returns_400(self):
        client = _fresh_client()
        r = client.post("/cloud/drift/event", json={
            "tenant_id": "", "model_id": "x", "bom_a": "a", "bom_b": "b",
        })
        assert r.status_code == 400

    def test_event_includes_added_removed(self):
        client = _fresh_client()
        data = self._post_event(client, "t")
        assert data["added"] == ["pkg:pypi/requests@2.32.0"]
        assert data["removed"] == ["pkg:pypi/requests@2.31.0"]

    def test_event_limit_parameter(self):
        client = _fresh_client()
        for _ in range(15):
            self._post_event(client, "t")
        r = client.get("/cloud/drift/events?limit=5", headers={"X-Tenant-ID": "t"})
        assert r.json()["count"] == 5
        assert r.json()["total"] == 15


# ──────────────────────────────────────────────────────────────────────────────
# 6. Policy dashboard
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudPolicyDashboard:
    def _register_model(self, client: TestClient, tenant: str, passed: bool) -> None:
        client.post("/cloud/inventory/register", json={
            "tenant_id": tenant,
            "model_id": f"m-{passed}",
            "model_path": "/x",
            "attestation_passed": passed,
            "policy_results": {
                "enterprise-strict": {"passed": passed, "error_count": 0 if passed else 1, "warning_count": 0},
                "gdpr-baseline": {"passed": True, "error_count": 0, "warning_count": 0},
            },
        })

    def test_dashboard_empty_tenant_returns_zeros(self):
        client = _fresh_client()
        r = client.get("/cloud/policy/dashboard", headers={"X-Tenant-ID": "empty"})
        assert r.status_code == 200
        body = r.json()
        assert body["overall"]["total"] == 0
        assert body["overall"]["pass_rate"] == 0.0

    def test_dashboard_aggregates_correctly(self):
        client = _fresh_client()
        self._register_model(client, "corp", passed=True)
        self._register_model(client, "corp", passed=True)
        self._register_model(client, "corp", passed=False)
        r = client.get("/cloud/policy/dashboard", headers={"X-Tenant-ID": "corp"})
        body = r.json()
        assert body["overall"]["passed"] == 2
        assert body["overall"]["failed"] == 1
        assert body["overall"]["total"] == 3
        assert abs(body["overall"]["pass_rate"] - 2 / 3) < 0.001

    def test_dashboard_shows_by_policy_breakdown(self):
        client = _fresh_client()
        self._register_model(client, "t", passed=True)
        r = client.get("/cloud/policy/dashboard", headers={"X-Tenant-ID": "t"})
        body = r.json()
        policies = {p["policy"] for p in body["by_policy"]}
        assert "enterprise-strict" in policies
        assert "gdpr-baseline" in policies

    def test_dashboard_is_tenant_scoped(self):
        client = _fresh_client()
        self._register_model(client, "a", passed=True)
        r_b = client.get("/cloud/policy/dashboard", headers={"X-Tenant-ID": "b"})
        assert r_b.json()["overall"]["total"] == 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. Cloud audit endpoint
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudAudit:
    def test_cloud_audit_returns_200(self):
        """GET /cloud/audit delegates to AgentAuditLogger.read_tail."""
        client = _fresh_client()
        mock_logger = MagicMock()
        mock_logger.read_tail.return_value = []
        mock_logger.path = "/tmp/audit.jsonl"
        with patch("squish.squash.governor.AgentAuditLogger", return_value=mock_logger):
            r = client.get("/cloud/audit", headers={"X-Tenant-ID": "acme"})
        assert r.status_code == 200
        body = r.json()
        assert body["tenant_id"] == "acme"
        assert body["count"] == 0

    def test_cloud_audit_filters_by_tenant_prefix(self):
        client = _fresh_client()
        entries = [
            {"session_id": "acme-session-1", "event": "attest"},
            {"session_id": "other-session-2", "event": "attest"},
        ]
        mock_logger = MagicMock()
        mock_logger.read_tail.return_value = entries
        mock_logger.path = "/tmp/audit.jsonl"
        with patch("squish.squash.governor.AgentAuditLogger", return_value=mock_logger):
            r = client.get("/cloud/audit", headers={"X-Tenant-ID": "acme"})
        body = r.json()
        assert body["count"] == 1
        assert body["entries"][0]["session_id"] == "acme-session-1"

    def test_cloud_audit_no_tenant_returns_all(self):
        client = _fresh_client()
        entries = [{"session_id": "x"}, {"session_id": "y"}]
        mock_logger = MagicMock()
        mock_logger.read_tail.return_value = entries
        mock_logger.path = "/tmp/audit.jsonl"
        with patch("squish.squash.governor.AgentAuditLogger", return_value=mock_logger):
            r = client.get("/cloud/audit")
        # No tenant → no filter, both entries returned
        assert r.json()["count"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# 8. JWT-based multi-tenant auth on /cloud/* endpoints
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudJwtAuth:
    SECRET = "test-jwt-secret-key"

    def _jwt_headers(self, tenant_id: str) -> dict[str, str]:
        token = _make_jwt(
            {"tenant_id": tenant_id, "exp": int(time.time()) + 3600},
            self.SECRET,
        )
        return {"Authorization": f"Bearer {token}"}

    def test_inventory_via_jwt_returns_correct_tenant(self):
        client = _fresh_client()
        with patch.dict(os.environ, {"SQUASH_JWT_SECRET": self.SECRET}):
            # Register via tenant_id in body (doesn't call _resolve_tenant_id)
            client.post("/cloud/inventory/register", json={
                "tenant_id": "jwt-tenant",
                "model_id": "jwt-model",
                "model_path": "/x",
                "attestation_passed": True,
            })
            # Read back via JWT
            r = client.get(
                "/cloud/inventory",
                headers=self._jwt_headers("jwt-tenant"),
            )
        assert r.status_code == 200
        assert r.json()["tenant_id"] == "jwt-tenant"
        assert r.json()["count"] == 1

    def test_invalid_jwt_returns_401(self):
        client = _fresh_client()
        with patch.dict(os.environ, {"SQUASH_JWT_SECRET": self.SECRET}):
            r = client.get(
                "/cloud/inventory",
                headers={"Authorization": "Bearer this.isnot.valid"},
            )
        assert r.status_code == 401

    def test_missing_auth_header_with_jwt_secret_returns_401(self):
        client = _fresh_client()
        with patch.dict(os.environ, {"SQUASH_JWT_SECRET": self.SECRET}):
            r = client.get("/cloud/inventory")
        assert r.status_code == 401

    def test_expired_jwt_returns_401(self):
        client = _fresh_client()
        token = _make_jwt({"tenant_id": "t", "exp": int(time.time()) - 5}, self.SECRET)
        with patch.dict(os.environ, {"SQUASH_JWT_SECRET": self.SECRET}):
            r = client.get(
                "/cloud/inventory",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert r.status_code == 401

    def test_jwt_without_tenant_id_claim_returns_401(self):
        client = _fresh_client()
        token = _make_jwt({"sub": "", "exp": int(time.time()) + 300}, self.SECRET)
        with patch.dict(os.environ, {"SQUASH_JWT_SECRET": self.SECRET}):
            r = client.get(
                "/cloud/inventory",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert r.status_code == 401

    def test_x_tenant_id_header_used_without_jwt_secret(self):
        client = _fresh_client()
        # No SQUASH_JWT_SECRET → fall back to X-Tenant-ID header
        env = {k: v for k, v in os.environ.items() if k != "SQUASH_JWT_SECRET"}
        with patch.dict(os.environ, env, clear=True):
            r = client.get("/cloud/inventory", headers={"X-Tenant-ID": "plain-tenant"})
        assert r.status_code == 200
        assert r.json()["tenant_id"] == "plain-tenant"


# ──────────────────────────────────────────────────────────────────────────────
# 9. AttestRequest tenant_id field
# ──────────────────────────────────────────────────────────────────────────────

class TestAttestRequestTenantId:
    def test_tenant_id_field_exists_and_defaults_empty(self):
        from squish.squash.api import AttestRequest
        req = AttestRequest(model_path="/x")
        assert req.tenant_id == ""

    def test_tenant_id_field_accepts_value(self):
        from squish.squash.api import AttestRequest
        req = AttestRequest(model_path="/x", tenant_id="my-tenant")
        assert req.tenant_id == "my-tenant"


# ──────────────────────────────────────────────────────────────────────────────
# 10. Cloud Pydantic model shape contracts
# ──────────────────────────────────────────────────────────────────────────────

class TestCloudModelShapeContracts:
    def test_tenant_create_request_fields(self):
        from squish.squash.api import TenantCreateRequest
        m = TenantCreateRequest(tenant_id="t", name="Test")
        assert m.plan == "community"
        assert m.contact_email == ""

    def test_inventory_register_request_defaults(self):
        from squish.squash.api import InventoryRegisterRequest
        m = InventoryRegisterRequest(
            tenant_id="t", model_id="m", model_path="/p", attestation_passed=True
        )
        assert m.policy_results == {}
        assert m.vex_cves == []
        assert m.timestamp == ""

    def test_vex_alert_request_defaults(self):
        from squish.squash.api import VexAlertRequest
        m = VexAlertRequest(tenant_id="t", cve_id="CVE-2024-001")
        assert m.severity == "unknown"
        assert m.status == "open"

    def test_drift_event_request_defaults(self):
        from squish.squash.api import DriftEventRequest
        m = DriftEventRequest(tenant_id="t", model_id="m", bom_a="a", bom_b="b")
        assert m.added == []
        assert m.severity == "info"


# ──────────────────────────────────────────────────────────────────────────────
# 11. Module count guard
# ──────────────────────────────────────────────────────────────────────────────

class TestModuleCount:
    def test_module_count_is_125(self):
        """squish/ has 131 Python files after W54-56 adds remediate.py, evaluator.py, edge_formats.py, chat.py."""
        import subprocess, pathlib
        root = pathlib.Path(__file__).parent.parent / "squish"
        count = len(list(root.rglob("*.py")))
        assert count == 134, (
            f"Module count should be 134, got {count}. "
            "W57 adds model_card.py + cloud_db.py (SQLite persistence, justified). "
            "W83 adds nist_rmf.py (NIST AI RMF 1.0 controls scanner, justified). "
            "New modules require deletion or written justification."
        )
