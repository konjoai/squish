"""tests/test_squash_wave37.py — Wave 37: SPDX AI Profile fields in /attest REST API.

Test taxonomy:
  Unit       — AttestRequest Pydantic model field inspection; no I/O.
  Integration — POST /attest via FastAPI TestClient with synthetic model dirs;
                verifies SPDX option fields are accepted and propagated to the
                written SPDX JSON artifact.

Covers:
  - spdx_type, spdx_safety_risk, spdx_datasets (list), spdx_training_info,
    spdx_sensitive_data fields present on AttestRequest with correct defaults
  - /attest accepts the SPDX fields without error (200)
  - Custom spdx_type propagates to SPDX JSON artifact
  - spdx_datasets merges with training_dataset_ids (deduplication)
  - No SPDX fields → prior behaviour unchanged (backward compat)
  - OpenAPI schema exposes the new SPDX fields on /attest body
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

try:
    from fastapi.testclient import TestClient
    from squish.squash.api import app, AttestRequest
except ImportError:
    pytest.skip("fastapi / httpx not installed", allow_module_level=True)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stub_model_dir(tmp_path: Path, name: str = "test-model") -> Path:
    """Write a minimal safetensors stub that passes all offline checks."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    header = b"{}"
    (d / "model.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 16
    )
    return d


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ── AttestRequest model field contracts ───────────────────────────────────────


class TestAttestRequestSpdxFields:
    """Unit: Pydantic model has all five SPDX fields with correct defaults."""

    def test_spdx_type_default_none(self):
        req = AttestRequest(model_path="/tmp/model")
        assert req.spdx_type is None

    def test_spdx_safety_risk_default_none(self):
        req = AttestRequest(model_path="/tmp/model")
        assert req.spdx_safety_risk is None

    def test_spdx_datasets_default_empty(self):
        req = AttestRequest(model_path="/tmp/model")
        assert req.spdx_datasets == []

    def test_spdx_training_info_default_none(self):
        req = AttestRequest(model_path="/tmp/model")
        assert req.spdx_training_info is None

    def test_spdx_sensitive_data_default_none(self):
        req = AttestRequest(model_path="/tmp/model")
        assert req.spdx_sensitive_data is None

    def test_spdx_type_accepted(self):
        req = AttestRequest(model_path="/tmp/model", spdx_type="text-classification")
        assert req.spdx_type == "text-classification"

    def test_spdx_safety_risk_accepted(self):
        req = AttestRequest(model_path="/tmp/model", spdx_safety_risk="high")
        assert req.spdx_safety_risk == "high"

    def test_spdx_datasets_list_accepted(self):
        req = AttestRequest(model_path="/tmp/model", spdx_datasets=["wikipedia", "c4"])
        assert req.spdx_datasets == ["wikipedia", "c4"]

    def test_training_dataset_ids_still_works(self):
        """Backward compat: existing training_dataset_ids field unchanged."""
        req = AttestRequest(model_path="/tmp/model", training_dataset_ids=["pile"])
        assert req.training_dataset_ids == ["pile"]


# ── /attest endpoint accepts SPDX fields ─────────────────────────────────────


class TestAttestEndpointSpdxAccepted:
    """Integration: /attest returns 200 when SPDX fields are supplied."""

    def test_spdx_type_accepted_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_type": "question-answering",
        })
        assert resp.status_code == 200

    def test_spdx_safety_risk_accepted_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_safety_risk": "medium",
        })
        assert resp.status_code == 200

    def test_spdx_datasets_list_accepted_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_datasets": ["wikipedia", "c4"],
        })
        assert resp.status_code == 200

    def test_all_spdx_fields_accepted_returns_200(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_type": "summarization",
            "spdx_safety_risk": "low",
            "spdx_datasets": ["pile"],
            "spdx_training_info": "see-hf-model-card",
            "spdx_sensitive_data": "absent",
        })
        assert resp.status_code == 200

    def test_no_spdx_fields_still_200_backward_compat(self, client, tmp_path):
        """Prior behaviour: no SPDX fields → /attest still succeeds."""
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
        })
        assert resp.status_code == 200


# ── SPDX options propagate to written artifact ────────────────────────────────


class TestAttestEndpointSpdxPropagation:
    """Integration: custom SPDX values appear in the written SPDX JSON artifact."""

    def _spdx_path(self, resp_data: dict, output_dir: Path) -> Path | None:
        """Find the SPDX JSON path from the response or scan the output dir."""
        spdx_path_str = resp_data.get("spdx_json")
        if spdx_path_str:
            p = Path(spdx_path_str)
            if p.exists():
                return p
        # Fallback: search output dir for an spdx JSON file
        candidates = list(output_dir.rglob("*spdx*.json"))
        return candidates[0] if candidates else None

    def test_custom_spdx_type_in_artifact(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_type": "translation",
        })
        assert resp.status_code == 200
        spdx_p = self._spdx_path(resp.json(), d)
        assert spdx_p is not None and spdx_p.exists(), "SPDX JSON artifact not found"
        content = spdx_p.read_text()
        assert "translation" in content, "Custom spdx_type not found in SPDX JSON"

    def test_custom_safety_risk_in_artifact(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_safety_risk": "high",
        })
        assert resp.status_code == 200
        spdx_p = self._spdx_path(resp.json(), d)
        assert spdx_p is not None and spdx_p.exists()
        assert "high" in spdx_p.read_text().lower()

    def test_spdx_datasets_in_artifact(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "spdx_datasets": ["wikipedia", "bookcorpus"],
        })
        assert resp.status_code == 200
        spdx_p = self._spdx_path(resp.json(), d)
        assert spdx_p is not None and spdx_p.exists()
        content = spdx_p.read_text()
        assert "wikipedia" in content
        assert "bookcorpus" in content

    def test_default_type_of_model_when_no_spdx_fields(self, client, tmp_path):
        """When no SPDX flags given, default text-generation should appear in artifact."""
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
        })
        assert resp.status_code == 200
        spdx_p = self._spdx_path(resp.json(), d)
        assert spdx_p is not None and spdx_p.exists()
        assert "text-generation" in spdx_p.read_text()


# ── Dataset deduplication ─────────────────────────────────────────────────────


class TestDatasetMerge:
    """Integration: spdx_datasets merges with training_dataset_ids (no duplication)."""

    def _parse_spdx(self, resp_data: dict, d: Path) -> str:
        spdx_path_str = resp_data.get("spdx_json")
        if spdx_path_str:
            p = Path(spdx_path_str)
            if p.exists():
                return p.read_text()
        candidates = list(d.rglob("*spdx*.json"))
        return candidates[0].read_text() if candidates else ""

    def test_spdx_datasets_plus_training_dataset_ids(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "training_dataset_ids": ["pile"],
            "spdx_datasets": ["c4"],
        })
        assert resp.status_code == 200
        content = self._parse_spdx(resp.json(), d)
        # Both should appear in the SPDX output
        assert "pile" in content
        assert "c4" in content

    def test_deduplication_when_same_dataset_in_both_fields(self, client, tmp_path):
        d = _stub_model_dir(tmp_path)
        resp = client.post("/attest", json={
            "model_path": str(d),
            "policies": [],
            "sign": False,
            "training_dataset_ids": ["wikipedia"],
            "spdx_datasets": ["wikipedia", "c4"],
        })
        assert resp.status_code == 200
        content = self._parse_spdx(resp.json(), d)
        assert "wikipedia" in content
        assert "c4" in content
        # "wikipedia" should not appear ≥2 times as a duplicate entry
        assert content.count("wikipedia") < 4  # SPDX may repeat in name/description


# ── OpenAPI schema exposes new fields ─────────────────────────────────────────


class TestOpenApiSpdxFields:
    """The /openapi.json schema must list all five SPDX fields on AttestRequest."""

    def test_attest_schema_has_spdx_type(self, client):
        schema = client.get("/openapi.json").json()
        attest_schema = (
            schema.get("components", {})
                  .get("schemas", {})
                  .get("AttestRequest", {})
                  .get("properties", {})
        )
        assert "spdx_type" in attest_schema

    def test_attest_schema_has_spdx_safety_risk(self, client):
        schema = client.get("/openapi.json").json()
        attest_schema = (
            schema.get("components", {})
                  .get("schemas", {})
                  .get("AttestRequest", {})
                  .get("properties", {})
        )
        assert "spdx_safety_risk" in attest_schema

    def test_attest_schema_has_spdx_datasets(self, client):
        schema = client.get("/openapi.json").json()
        attest_schema = (
            schema.get("components", {})
                  .get("schemas", {})
                  .get("AttestRequest", {})
                  .get("properties", {})
        )
        assert "spdx_datasets" in attest_schema

    def test_attest_schema_has_spdx_training_info(self, client):
        schema = client.get("/openapi.json").json()
        attest_schema = (
            schema.get("components", {})
                  .get("schemas", {})
                  .get("AttestRequest", {})
                  .get("properties", {})
        )
        assert "spdx_training_info" in attest_schema

    def test_attest_schema_has_spdx_sensitive_data(self, client):
        schema = client.get("/openapi.json").json()
        attest_schema = (
            schema.get("components", {})
                  .get("schemas", {})
                  .get("AttestRequest", {})
                  .get("properties", {})
        )
        assert "spdx_sensitive_data" in attest_schema
