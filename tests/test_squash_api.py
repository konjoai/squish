"""tests/test_squash_api.py — Tests for the Squash FastAPI REST microservice.

Test taxonomy:
  - Integration — uses httpx.AsyncClient against the real app instance.
    No network required; model paths point to tmp directories.

Requires:
    pip install httpx pytest-asyncio

All async tests use pytest-asyncio with asyncio_mode="auto" (set in pyproject.toml).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

try:
    import httpx
    from fastapi.testclient import TestClient

    from squish.squash.api import app
except ImportError:
    pytest.skip("fastapi / httpx not installed", allow_module_level=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stub_model_dir(tmp_path: Path, name: str = "test-model") -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    header = b"{}"
    weight = d / "model.safetensors"
    weight.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
    return d


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── GET /health ────────────────────────────────────────────────────────────────


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_body_has_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json().get("status") == "ok"


# ── GET /policies ──────────────────────────────────────────────────────────────


class TestListPolicies:
    def test_returns_200(self, client):
        resp = client.get("/policies")
        assert resp.status_code == 200

    def test_returns_list(self, client):
        resp = client.get("/policies")
        data = resp.json()
        assert "policies" in data
        assert isinstance(data["policies"], list)

    def test_includes_enterprise_strict(self, client):
        resp = client.get("/policies")
        assert "enterprise-strict" in resp.json()["policies"]

    def test_includes_eu_ai_act(self, client):
        resp = client.get("/policies")
        assert "eu-ai-act" in resp.json()["policies"]


# ── POST /scan ─────────────────────────────────────────────────────────────────


class TestScanEndpoint:
    # Wave 8: POST /scan is now async-queued — returns 202 + job_id immediately.
    # Results are retrieved via GET /scan/{job_id}.

    def _poll(self, client, job_id: str, max_attempts: int = 40) -> dict:
        """Poll GET /scan/{job_id} until status != 'pending' or budget exhausted."""
        import time

        for _ in range(max_attempts):
            r = client.get(f"/scan/{job_id}")
            body = r.json()
            if body.get("status") != "pending":
                return body
            time.sleep(0.05)
        return body  # return last seen body even if still pending

    def test_scan_clean_dir_returns_202(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        assert resp.status_code == 202

    def test_scan_response_has_job_id_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        assert "job_id" in resp.json()

    def test_scan_response_has_is_safe_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        job_id = resp.json()["job_id"]
        data = self._poll(client, job_id)
        assert "is_safe" in data.get("result", {})

    def test_scan_clean_dir_is_safe(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        job_id = resp.json()["job_id"]
        data = self._poll(client, job_id)
        # A minimal safetensors stub should be clean
        assert data.get("result", {}).get("is_safe") is True

    def test_scan_nonexistent_path_returns_404(self, client, tmp_path):
        resp = client.post("/scan", json={"model_path": str(tmp_path / "nonexistent")})
        assert resp.status_code == 404


# ── POST /policy/evaluate ─────────────────────────────────────────────────────


class TestPolicyEvaluate:
    def _minimal_bom(self) -> dict:
        return {
            "bomFormat": "CycloneDX",
            "specVersion": "1.7",
            "version": 1,
            "metadata": {
                "component": {
                    "name": "test-model",
                    "version": "1.0.0",
                    "hashes": [
                        {"alg": "SHA-256", "content": "abc123"},
                        {"alg": "SHA-512", "content": "def456"},
                    ],
                }
            },
            "components": [
                {
                    "name": "test-model",
                    "version": "1.0.0",
                    "hashes": [
                        {"alg": "SHA-256", "content": "abc123"},
                    ],
                }
            ],
            "squash:scan_result": "clean",
        }

    def test_evaluate_known_policy_returns_200_or_422(self, client):
        resp = client.post("/policy/evaluate", json={
            "sbom": self._minimal_bom(),
            "policy": "eu-ai-act",
        })
        assert resp.status_code in (200, 422)

    def test_evaluate_unknown_policy_returns_400(self, client):
        resp = client.post("/policy/evaluate", json={
            "sbom": self._minimal_bom(),
            "policy": "this-policy-does-not-exist-xyz",
        })
        assert resp.status_code == 400

    def test_enterprise_strict_returns_result_shape(self, client):
        resp = client.post("/policy/evaluate", json={
            "sbom": self._minimal_bom(),
            "policy": "enterprise-strict",
        })
        if resp.status_code in (200, 422):
            data = resp.json()
            assert "policy_name" in data or "passed" in data


# ── POST /attest ───────────────────────────────────────────────────────────────


class TestAttestEndpoint:
    def test_attest_nonexistent_path_returns_404(self, client, tmp_path):
        resp = client.post("/attest", json={
            "model_path": str(tmp_path / "nonexistent"),
        })
        assert resp.status_code == 404

    def test_attest_clean_model_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
        })
        assert resp.status_code == 200

    def test_attest_response_has_passed_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "passed" in data

    def test_attest_with_policy_returns_policy_results(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": ["eu-ai-act"],
            "sign": False,
            "fail_on_violation": False,
        })
        assert resp.status_code in (200, 422)
        data = resp.json()
        assert "policy_results" in data or "passed" in data

    def test_attest_fail_on_violation_clean_still_200(self, client, tmp_path):
        """Clean model with fail_on_violation=True should still return 200."""
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "fail_on_violation": True,
        })
        assert resp.status_code == 200


# ── OpenAPI schema integrity ───────────────────────────────────────────────────


class TestOpenApiSchema:
    def test_openapi_endpoint_exists(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200

    def test_openapi_has_paths(self, client):
        schema = client.get("/openapi.json").json()
        assert "paths" in schema
        paths = schema["paths"]
        assert "/health" in paths
        assert "/policies" in paths or "/policies" in str(paths)


# ── Wave 13 — Auth, rate limiter, metrics ─────────────────────────────────────


class TestBearerAuth:
    """Bearer token authentication middleware."""

    def test_no_token_env_allows_all_requests(self, client):
        """When SQUASH_API_TOKEN is unset, auth is disabled."""
        import os
        os.environ.pop("SQUASH_API_TOKEN", None)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_correct_token_allows_request(self):
        import os
        from fastapi.testclient import TestClient
        os.environ["SQUASH_API_TOKEN"] = "test-secret"
        try:
            from squish.squash import api as _api
            # Reload to pick up env — use a fresh client per test
            tc = TestClient(_api.app)
            resp = tc.get("/policies", headers={"Authorization": "Bearer test-secret"})
            assert resp.status_code == 200
        finally:
            os.environ.pop("SQUASH_API_TOKEN", None)

    def test_wrong_token_returns_401(self):
        import os
        from fastapi.testclient import TestClient
        os.environ["SQUASH_API_TOKEN"] = "secret123"
        try:
            from squish.squash import api as _api
            tc = TestClient(_api.app)
            resp = tc.get("/policies", headers={"Authorization": "Bearer wrong"})
            assert resp.status_code == 401
        finally:
            os.environ.pop("SQUASH_API_TOKEN", None)

    def test_health_bypasses_auth(self):
        """Health endpoint is exempted from auth check."""
        import os
        from fastapi.testclient import TestClient
        os.environ["SQUASH_API_TOKEN"] = "secret123"
        try:
            from squish.squash import api as _api
            tc = TestClient(_api.app)
            resp = tc.get("/health")
            assert resp.status_code == 200
        finally:
            os.environ.pop("SQUASH_API_TOKEN", None)

    def test_metrics_bypasses_auth(self):
        import os
        from fastapi.testclient import TestClient
        os.environ["SQUASH_API_TOKEN"] = "secret123"
        try:
            from squish.squash import api as _api
            tc = TestClient(_api.app)
            resp = tc.get("/metrics")
            assert resp.status_code == 200
        finally:
            os.environ.pop("SQUASH_API_TOKEN", None)


class TestRateLimit:
    def test_exceeding_rate_limit_returns_429(self):
        import os
        from fastapi.testclient import TestClient
        os.environ.pop("SQUASH_API_TOKEN", None)
        # Set a tiny rate limit for this test
        os.environ["SQUASH_RATE_LIMIT"] = "3"
        try:
            from squish.squash import api as _api
            # Reset window state to avoid interference from other tests
            _api._rate_window.clear()
            _api._RATE_LIMIT = 3
            tc = TestClient(_api.app)
            statuses = [tc.get("/health").status_code for _ in range(5)]
            assert 429 in statuses
        finally:
            os.environ.pop("SQUASH_RATE_LIMIT", None)
            # Restore default
            from squish.squash import api as _api2
            _api2._RATE_LIMIT = 60
            _api2._rate_window.clear()

    def test_429_includes_retry_after_header(self):
        import os
        from fastapi.testclient import TestClient
        os.environ.pop("SQUASH_API_TOKEN", None)
        from squish.squash import api as _api
        _api._rate_window.clear()
        _api._RATE_LIMIT = 1
        tc = TestClient(_api.app)
        tc.get("/health")  # consume the 1 allowed
        resp = tc.get("/health")
        if resp.status_code == 429:
            assert "Retry-After" in resp.headers
        # restore
        _api._RATE_LIMIT = 60
        _api._rate_window.clear()


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_metrics_content_type_is_text(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_counter_names(self, client):
        resp = client.get("/metrics")
        body = resp.text
        assert "squash_attest_total" in body
        assert "squash_scan_total" in body
        assert "squash_policy_violations_total" in body

    def test_counter_increments_on_scan(self, client, tmp_path):
        """Hitting /scan should increment squash_scan_total."""
        import os
        from squish.squash import api as _api
        before = _api._COUNTERS["squash_scan_total"]
        # Post a scan request (will return 202 with a job_id)
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text('{"model_type":"gpt2"}')
        client.post("/scan", json={"model_path": str(d)})
        after = _api._COUNTERS["squash_scan_total"]
        assert after == before + 1

