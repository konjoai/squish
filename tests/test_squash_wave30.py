"""tests/test_squash_wave30.py — Wave 30 REST API endpoint tests.

Tests five new endpoints added to squish/squash/api.py:

    POST /vex/publish
    POST /attest/mlflow
    POST /attest/wandb
    POST /attest/huggingface
    POST /attest/langchain

Test taxonomy:
  - Integration — uses FastAPI TestClient against the real app, real pipeline.
    No handler mocking.  Model paths point to tmp directories created by
    _stub_model_dir().

All heavy work runs in-process via the thread-pool executor; no uvicorn needed.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient

    from squish.squash.api import app, _rate_window
except ImportError:
    pytest.skip("fastapi not installed", allow_module_level=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stub_model_dir(tmp_path: Path, name: str = "model") -> Path:
    """Create a minimal model directory that AttestPipeline can process."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    header = b"{}"
    weight = d / "model.safetensors"
    weight.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
    return d


@pytest.fixture()
def client():
    _rate_window.clear()  # reset per-IP sliding-window so full-suite runs never exhaust the bucket
    with TestClient(app) as c:
        yield c


# ── POST /vex/publish ─────────────────────────────────────────────────────────


class TestVexPublishEndpoint:
    def test_empty_entries_returns_200(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        assert resp.status_code == 200

    def test_empty_entries_returns_openvex_context(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        doc = resp.json()
        assert doc.get("@context") == "https://openvex.dev/ns/v0.2.0"

    def test_empty_entries_returns_openvex_type(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        doc = resp.json()
        assert doc.get("@type") == "OpenVEXDocument"

    def test_empty_entries_has_statements_list(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        doc = resp.json()
        assert isinstance(doc.get("statements"), list)

    def test_single_entry_appears_in_statements(self, client):
        entry = {
            "vulnerability": {"name": "CVE-2024-99999"},
            "products": [{"@id": "pkg:pypi/numpy@1.24.0"}],
            "status": "not_affected",
            "justification": "vulnerable_code_not_in_execute_path",
        }
        resp = client.post("/vex/publish", json={"entries": [entry]})
        assert resp.status_code == 200
        stmts = resp.json().get("statements", [])
        assert len(stmts) == 1
        assert stmts[0]["vulnerability"]["name"] == "CVE-2024-99999"

    def test_author_override_reflected_in_doc(self, client):
        resp = client.post(
            "/vex/publish",
            json={"entries": [], "author": "konjo-ai"},
        )
        assert resp.json().get("author") == "konjo-ai"

    def test_doc_id_override_reflected_in_doc(self, client):
        resp = client.post(
            "/vex/publish",
            json={"entries": [], "doc_id": "https://example.com/vex/123"},
        )
        assert resp.json().get("@id") == "https://example.com/vex/123"

    def test_default_author_is_squash(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        assert resp.json().get("author") == "squash"

    def test_timestamp_is_present(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        assert resp.json().get("timestamp")

    def test_spec_version_is_correct(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        assert resp.json().get("specVersion") == "0.2.0"

    def test_multiple_entries_all_appear_in_statements(self, client):
        entries = [
            {
                "vulnerability": {"name": f"CVE-2024-{10000 + i}"},
                "products": [],
                "status": "under_investigation",
            }
            for i in range(3)
        ]
        resp = client.post("/vex/publish", json={"entries": entries})
        assert resp.status_code == 200
        assert len(resp.json()["statements"]) == 3

    def test_missing_entries_field_uses_empty_default(self, client):
        resp = client.post("/vex/publish", json={})
        assert resp.status_code == 200
        assert resp.json().get("statements") == []


# ── POST /attest/mlflow ───────────────────────────────────────────────────────


class TestAttestMlflowEndpoint:
    def test_nonexistent_path_returns_404(self, client, tmp_path):
        resp = client.post(
            "/attest/mlflow",
            json={"model_path": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404

    def test_valid_model_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/mlflow", json={"model_path": str(d), "policies": []})
        assert resp.status_code == 200

    def test_response_has_passed_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/mlflow", json={"model_path": str(d), "policies": []})
        assert "passed" in resp.json()

    def test_response_has_model_id_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/mlflow", json={"model_path": str(d), "policies": []})
        assert "model_id" in resp.json()

    def test_response_has_artifacts_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/mlflow", json={"model_path": str(d), "policies": []})
        assert "artifacts" in resp.json()

    def test_custom_policy_accepted(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/mlflow",
            json={"model_path": str(d), "policies": ["eu-ai-act"]},
        )
        assert resp.status_code in (200, 400)  # 400 = ran but policy failed
        assert "passed" in resp.json()

    def test_response_has_policy_results_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/mlflow", json={"model_path": str(d), "policies": []})
        assert "policy_results" in resp.json()


# ── POST /attest/wandb ────────────────────────────────────────────────────────


class TestAttestWandbEndpoint:
    def test_nonexistent_path_returns_404(self, client, tmp_path):
        resp = client.post(
            "/attest/wandb",
            json={"model_path": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404

    def test_valid_model_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/wandb", json={"model_path": str(d), "policies": []})
        assert resp.status_code == 200

    def test_response_has_passed_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/wandb", json={"model_path": str(d), "policies": []})
        assert "passed" in resp.json()

    def test_response_has_model_id_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/wandb", json={"model_path": str(d), "policies": []})
        assert "model_id" in resp.json()

    def test_sign_flag_accepted_without_error(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/wandb",
            json={"model_path": str(d), "sign": True, "policies": []},
        )
        assert resp.status_code in (200, 400)

    def test_fail_on_violation_accepted(self, client, tmp_path):
        """Endpoint accepts fail_on_violation and returns a result dict (not 500)."""
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/wandb",
            json={"model_path": str(d), "fail_on_violation": True, "policies": []},
        )
        # With policies=[] the model passes → 200; field accepted without raising
        assert resp.status_code in (200, 400, 422)
        assert "passed" in resp.json()


# ── POST /attest/huggingface ──────────────────────────────────────────────────


class TestAttestHuggingFaceEndpoint:
    def test_nonexistent_path_returns_404(self, client, tmp_path):
        resp = client.post(
            "/attest/huggingface",
            json={"model_path": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404

    def test_offline_mode_returns_200(self, client, tmp_path):
        """Without repo_id the endpoint falls back to offline AttestPipeline."""
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/huggingface", json={"model_path": str(d), "policies": []})
        assert resp.status_code == 200

    def test_offline_mode_has_passed_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/huggingface", json={"model_path": str(d), "policies": []})
        assert "passed" in resp.json()

    def test_offline_mode_has_artifacts(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/huggingface", json={"model_path": str(d), "policies": []})
        assert "artifacts" in resp.json()

    def test_hf_token_field_accepted_in_schema(self, client, tmp_path):
        """hf_token should be accepted as a field even in offline mode."""
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/huggingface",
            json={"model_path": str(d), "hf_token": "hf_test", "policies": []},
        )
        # Should not fail schema validation
        assert resp.status_code in (200, 400)

    def test_repo_id_field_accepted_in_schema(self, client, tmp_path):
        """repo_id is a valid schema field; push failure returns 502 not 422."""
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/huggingface",
            json={"model_path": str(d), "repo_id": "test-org/test-model"},
        )
        # Push fails without real credentials → 502; field is valid → never 422
        assert resp.status_code != 422

    def test_custom_policies_pass_through(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/huggingface",
            json={"model_path": str(d), "policies": ["eu-ai-act"]},
        )
        assert "passed" in resp.json()


# ── POST /attest/langchain ────────────────────────────────────────────────────


class TestAttestLangchainEndpoint:
    def test_nonexistent_path_returns_404(self, client, tmp_path):
        resp = client.post(
            "/attest/langchain",
            json={"model_path": str(tmp_path / "nonexistent")},
        )
        assert resp.status_code == 404

    def test_valid_model_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/langchain", json={"model_path": str(d), "policies": []})
        assert resp.status_code == 200

    def test_response_has_passed_field(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/langchain", json={"model_path": str(d), "policies": []})
        assert "passed" in resp.json()

    def test_response_has_model_id(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/langchain", json={"model_path": str(d), "policies": []})
        assert "model_id" in resp.json()

    def test_custom_policy_accepted(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/langchain",
            json={"model_path": str(d), "policies": ["eu-ai-act"]},
        )
        assert resp.status_code in (200, 400)
        assert "passed" in resp.json()

    def test_sign_flag_accepted(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post(
            "/attest/langchain",
            json={"model_path": str(d), "sign": True},
        )
        assert resp.status_code in (200, 400)

    def test_response_has_policy_results(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest/langchain", json={"model_path": str(d)})
        assert "policy_results" in resp.json()


# ── Cross-endpoint structural contracts ───────────────────────────────────────


class TestEndpointStructuralContracts:
    """Verify the new endpoints appear in the OpenAPI schema."""

    def test_vex_publish_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/vex/publish" in schema["paths"]

    def test_attest_mlflow_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/attest/mlflow" in schema["paths"]

    def test_attest_wandb_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/attest/wandb" in schema["paths"]

    def test_attest_huggingface_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/attest/huggingface" in schema["paths"]

    def test_attest_langchain_in_openapi(self, client):
        schema = client.get("/openapi.json").json()
        assert "/attest/langchain" in schema["paths"]

    def test_all_new_endpoints_use_post_method(self, client):
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        for path in (
            "/vex/publish",
            "/attest/mlflow",
            "/attest/wandb",
            "/attest/huggingface",
            "/attest/langchain",
        ):
            assert "post" in paths[path], f"{path} missing POST method"

    def test_vex_publish_missing_entries_uses_default(self, client):
        """Pydantic default_factory — body can omit entries field."""
        resp = client.post("/vex/publish", json={"author": "test"})
        assert resp.status_code == 200
        assert resp.json()["statements"] == []


# ── Response schema contracts (VEX document fields) ───────────────────────────


class TestVexPublishResponseSchema:
    def test_at_id_is_string(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        assert isinstance(resp.json().get("@id"), str)

    def test_at_id_starts_with_https(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        assert resp.json()["@id"].startswith("https://")

    def test_timestamp_has_z_suffix(self, client):
        resp = client.post("/vex/publish", json={"entries": []})
        ts = resp.json().get("timestamp", "")
        assert ts.endswith("Z")

    def test_statements_preserves_justification(self, client):
        entry = {
            "vulnerability": {"name": "CVE-2024-55555"},
            "products": [],
            "status": "not_affected",
            "justification": "inline_mitigations_already_exist",
        }
        resp = client.post("/vex/publish", json={"entries": [entry]})
        stmt = resp.json()["statements"][0]
        assert stmt.get("justification") == "inline_mitigations_already_exist"

    def test_statements_preserves_impact_statement(self, client):
        entry = {
            "vulnerability": {"name": "CVE-2024-66666"},
            "products": [],
            "status": "not_affected",
            "impact_statement": "The vulnerable component is not reachable.",
        }
        resp = client.post("/vex/publish", json={"entries": [entry]})
        stmt = resp.json()["statements"][0]
        assert "The vulnerable component" in stmt.get("impact_statement", "")


# ── Pydantic model validation ─────────────────────────────────────────────────


class TestRequestModelValidation:
    def test_attest_integration_rejects_missing_model_path(self, client):
        resp = client.post("/attest/mlflow", json={})
        assert resp.status_code == 422

    def test_attest_hf_rejects_missing_model_path(self, client):
        resp = client.post("/attest/huggingface", json={})
        assert resp.status_code == 422

    def test_attest_wandb_rejects_missing_model_path(self, client):
        resp = client.post("/attest/wandb", json={})
        assert resp.status_code == 422

    def test_attest_langchain_rejects_missing_model_path(self, client):
        resp = client.post("/attest/langchain", json={})
        assert resp.status_code == 422

    def test_vex_publish_entries_must_be_list(self, client):
        # entries field must be a list; passing string should fail with 422
        resp = client.post("/vex/publish", json={"entries": "not-a-list"})
        assert resp.status_code == 422
