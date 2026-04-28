"""tests/test_governor_middleware.py — Tests for SquashGovernor middleware.

Uses Starlette TestClient + a minimal FastAPI stub app — does NOT import the
real squish server.  Each test builds its own app + governor instance to keep
state isolation clean.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

squash_pkg = pytest.importorskip("squash", reason="squash-ai not installed")
import squash.governor as _gov_module
from squash.governor import SquashGovernor


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_app(strict: bool = False, min_accuracy_ratio: float = 0.92) -> FastAPI:
    """Build a minimal FastAPI app with SquashGovernor and a /ping route."""
    app = FastAPI()
    app.add_middleware(SquashGovernor, strict=strict, min_accuracy_ratio=min_accuracy_ratio)

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.get("/v1/sbom")
    async def sbom():
        model_dir = _gov_module._get_model_dir()
        if model_dir is None:
            return JSONResponse({"error": "no sidecar available"}, status_code=404)
        sidecar = model_dir / "cyclonedx-mlbom.json"
        if not sidecar.exists():
            return JSONResponse({"error": "no sidecar available"}, status_code=404)
        return JSONResponse(json.loads(sidecar.read_text()))

    @app.get("/v1/health/model")
    async def health_model():
        model_dir = _gov_module._get_model_dir()
        gov = _gov_module._INSTANCE
        gov_state = gov.boot_state if gov is not None else {
            "integrity_ok": None,
            "accuracy_ok": None,
            "strict_compliance": False,
        }
        return {
            "model": getattr(_gov_module._state, "model_name", None) if _gov_module._state else None,
            "model_dir": str(model_dir) if model_dir else None,
            "sbom_present": bool(model_dir and (model_dir / "cyclonedx-mlbom.json").exists()),
            "integrity_ok": gov_state.get("integrity_ok"),
            "accuracy_ok": gov_state.get("accuracy_ok"),
            "strict_compliance": gov_state.get("strict_compliance", False),
        }

    return app


def _stub_state(model_dir: str) -> MagicMock:
    state = MagicMock()
    state.model_dir  = model_dir
    state.model_name = "test-model"
    state.model      = object()   # truthy = loaded
    return state


def _write_valid_sidecar(model_dir: Path, arc_delta: str | None = "-3.4") -> Path:
    """Write a minimal valid CycloneDX sidecar with a known composite hash."""
    # Create a dummy weight file so the hash is non-empty.
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"fake-weights")

    import hashlib
    h = hashlib.sha256(b"fake-weights").hexdigest()
    composite = hashlib.sha256(h.encode()).hexdigest()

    metrics = []
    if arc_delta is not None:
        metrics.append({
            "type": "accuracy",
            "value": "70.6",
            "slice": "arc_easy",
            "deltaFromBaseline": arc_delta,
        })

    sidecar = model_dir / "cyclonedx-mlbom.json"
    sidecar.write_text(json.dumps({
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": "urn:uuid:test",
        "components": [{
            "type": "machine-learning-model",
            "name": "test-model",
            "hashes": [{"alg": "SHA-256", "content": composite}],
            "modelCard": {
                "modelParameters": {"task": "text-generation"},
                "quantitativeAnalysis": {"performanceMetrics": metrics},
            },
            "properties": [
                {"name": "squish:weight_hash:model.safetensors", "value": h}
            ],
        }],
    }))
    return sidecar


def _write_corrupted_sidecar(model_dir: Path) -> Path:
    """Write a sidecar where the composite hash does NOT match the weight file."""
    weight = model_dir / "model.safetensors"
    weight.write_bytes(b"fake-weights")

    sidecar = model_dir / "cyclonedx-mlbom.json"
    sidecar.write_text(json.dumps({
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": "urn:uuid:test",
        "components": [{
            "type": "machine-learning-model",
            "name": "test-model",
            "hashes": [{"alg": "SHA-256", "content": "deadbeef" * 8}],  # wrong hash
            "modelCard": {
                "modelParameters": {"task": "text-generation"},
                "quantitativeAnalysis": {"performanceMetrics": []},
            },
            "properties": [],
        }],
    }))
    return sidecar


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestMissingSidecar:
    def test_missing_sidecar_is_nonfatal_200(self):
        """No sidecar → pass-through; never 503 regardless of strict mode."""
        with tempfile.TemporaryDirectory() as td:
            stub = _stub_state(td)
            app  = _make_app(strict=True)
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/ping")
                assert r.status_code == 200


class TestIntegrityNonStrict:
    def test_corrupted_hash_non_strict_still_200(self):
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_corrupted_sidecar(tmp)
            stub = _stub_state(td)
            app  = _make_app(strict=False)
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/ping")
                assert r.status_code == 200


class TestIntegrityStrictFail:
    def test_corrupted_hash_strict_returns_503(self):
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_corrupted_sidecar(tmp)
            stub = _stub_state(td)
            app  = _make_app(strict=True)
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/ping")
                assert r.status_code == 503
                body = r.json()
                assert body["error"] == "model compliance check failed"


class TestAccuracyFail:
    def test_large_negative_delta_strict_503(self):
        """arc_easy delta -15.0 is below min_accuracy_ratio 0.92 → 503."""
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            # Write valid hash so integrity passes, bad accuracy so accuracy fails.
            _write_valid_sidecar(tmp, arc_delta="-15.0")
            stub = _stub_state(td)
            app  = _make_app(strict=True)
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/ping")
                assert r.status_code == 503
                assert r.json()["error"] == "model compliance check failed"


class TestAccuracyOk:
    def test_small_negative_delta_strict_200(self):
        """-3.4pp is within 0.92 ratio → 200 even in strict mode."""
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_valid_sidecar(tmp, arc_delta="-3.4")
            stub = _stub_state(td)
            app  = _make_app(strict=True)
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/ping")
                assert r.status_code == 200


class TestMemoise:
    def test_boot_checks_run_exactly_once(self):
        """Hash function should be called once regardless of request count."""
        call_count = {"n": 0}
        original_hash = SquashGovernor._check_integrity

        def counting_check(self_inner, model_dir, component):
            call_count["n"] += 1
            return True

        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_valid_sidecar(tmp, arc_delta="-3.4")
            stub = _stub_state(td)
            app  = _make_app(strict=False)
            with patch.object(_gov_module, "_state", stub):
                with patch.object(SquashGovernor, "_check_integrity", counting_check):
                    client = TestClient(app, raise_server_exceptions=False)
                    for _ in range(3):
                        client.get("/ping")
            assert call_count["n"] == 1, f"Expected 1 hash call, got {call_count['n']}"


class TestSbomRoute:
    def test_sbom_present_returns_200_with_bom_format(self):
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_valid_sidecar(tmp, arc_delta=None)
            stub = _stub_state(td)
            app  = _make_app()
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/v1/sbom")
                assert r.status_code == 200
                assert r.json().get("bomFormat") == "CycloneDX"

    def test_sbom_absent_returns_404(self):
        with tempfile.TemporaryDirectory() as td:
            stub = _stub_state(td)  # empty dir — no sidecar
            app  = _make_app()
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/v1/sbom")
                assert r.status_code == 404

    def test_no_model_loaded_returns_404(self):
        with patch.object(_gov_module, "_state", None):
            app  = _make_app()
            client = TestClient(app, raise_server_exceptions=False)
            r = client.get("/v1/sbom")
            assert r.status_code == 404


class TestHealthModelRoute:
    def test_health_model_returns_expected_keys(self):
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_valid_sidecar(tmp, arc_delta="-3.4")
            stub = _stub_state(td)
            app  = _make_app()
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/v1/health/model")
                assert r.status_code == 200
                body = r.json()
                for key in ("model_dir", "sbom_present", "integrity_ok",
                            "accuracy_ok", "strict_compliance"):
                    assert key in body, f"Missing key: {key}"

    def test_sbom_present_flag_true_when_sidecar_exists(self):
        with tempfile.TemporaryDirectory() as td:
            tmp  = Path(td)
            _write_valid_sidecar(tmp, arc_delta=None)
            stub = _stub_state(td)
            app  = _make_app()
            with patch.object(_gov_module, "_state", stub):
                client = TestClient(app, raise_server_exceptions=False)
                r = client.get("/v1/health/model")
                assert r.json()["sbom_present"] is True
