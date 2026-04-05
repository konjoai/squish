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
    def test_scan_clean_dir_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        assert resp.status_code == 200

    def test_scan_response_has_status_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        data = resp.json()
        assert "status" in data

    def test_scan_response_has_is_safe_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        data = resp.json()
        assert "is_safe" in data

    def test_scan_clean_dir_is_safe(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/scan", json={"model_path": str(d)})
        # A minimal safetensors stub should be clean
        assert resp.json()["is_safe"] is True

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
