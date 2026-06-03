"""Wave 85 tests — real MLflow bridge + CLI/API integration."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from squish.squash.api import app
from squish.squash.cli import _cmd_mlflow_register
from squish.squash.mlflow_bridge import MlflowBridge, MlflowBridgeError, MlflowRegistration


@dataclass
class _FakePolicyResult:
    passed: bool = True
    error_count: int = 0
    warning_count: int = 0
    pass_count: int = 1


@dataclass
class _FakeScanResult:
    status: str = "clean"


class _FakeAttestResult:
    def __init__(self, output_dir: Path, *, passed: bool = True):
        self.model_id = "test-model"
        self.passed = passed
        self.output_dir = output_dir
        self.scan_result = _FakeScanResult(status="clean" if passed else "unsafe")
        self.policy_results = {"enterprise-strict": _FakePolicyResult(passed=passed)}
        self.cyclonedx_path = output_dir / "cyclonedx.json"
        self.spdx_json_path = output_dir / "sbom.spdx.json"
        self.spdx_tv_path = output_dir / "sbom.spdx"
        self.signature_path = output_dir / "sbom.sig"
        self.vex_report_path = output_dir / "vex.json"
        self.master_record_path = output_dir / "master.json"
        self.error = "" if passed else "policy violation"

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "passed": self.passed,
            "output_dir": str(self.output_dir),
        }


class _FakeRun:
    def __init__(self, run_id: str):
        self.info = SimpleNamespace(run_id=run_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeMlflowClient:
    def __init__(self):
        self.version_tags: list[tuple[str, str, str, str]] = []
        self.transitions: list[tuple[str, str, str]] = []
        self.aliases: list[tuple[str, str, str]] = []

    def create_model_version(self, *, name: str, source: str, run_id: str):
        self.created = (name, source, run_id)
        return SimpleNamespace(version="7")

    def set_model_version_tag(self, *, name: str, version: str, key: str, value: str):
        self.version_tags.append((name, version, key, value))

    def transition_model_version_stage(
        self, *, name: str, version: str, stage: str, archive_existing_versions: bool
    ):
        self.transitions.append((name, version, stage))

    def set_registered_model_alias(self, *, name: str, alias: str, version: str):
        self.aliases.append((name, alias, version))


class _FakeMlflow:
    def __init__(self):
        self.client = _FakeMlflowClient()
        self.tracking = SimpleNamespace(MlflowClient=lambda: self.client)
        self.params: list[tuple[str, str]] = []
        self.metrics: list[tuple[str, float]] = []
        self.tags: dict[str, str] = {}
        self.artifacts: list[tuple[str, str]] = []
        self.tracking_uri = ""
        self.run_name = ""

    def set_tracking_uri(self, uri: str):
        self.tracking_uri = uri

    def start_run(self, run_name=None):
        self.run_name = run_name or ""
        return _FakeRun("run-123")

    def log_param(self, key: str, value: str):
        self.params.append((key, value))

    def log_metric(self, key: str, value: float):
        self.metrics.append((key, value))

    def set_tags(self, tags: dict[str, str]):
        self.tags.update(tags)

    def log_artifacts(self, path: str, artifact_path: str = ""):
        self.artifacts.append((path, artifact_path))


def test_log_compress_run_logs_payload(tmp_path):
    mlflow = _FakeMlflow()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "a.txt").write_text("x", encoding="utf-8")

    with patch.object(MlflowBridge, "_import_mlflow", return_value=mlflow):
        run_id = MlflowBridge.log_compress_run(
            {
                "run_name": "wave85",
                "mlflow_tracking_uri": "http://mlflow.local",
                "params": {"model": "qwen", "bits": 4},
                "metrics": {"passed": 1.0, "bad": "not-a-number"},
                "tags": {"squash.passed": "true"},
                "artifacts_dir": artifacts,
            }
        )

    assert run_id == "run-123"
    assert mlflow.tracking_uri == "http://mlflow.local"
    assert ("model", "qwen") in mlflow.params
    assert ("bits", "4") in mlflow.params
    assert ("passed", 1.0) in mlflow.metrics
    assert mlflow.tags["squash.passed"] == "true"
    assert mlflow.artifacts and mlflow.artifacts[0][1] == "squash"


def test_register_model_returns_version():
    fake_client = _FakeMlflowClient()
    with patch.object(MlflowBridge, "_client", return_value=fake_client):
        version = MlflowBridge.register_model("run-123", "org/model")
    assert version == "7"
    assert fake_client.created[0] == "org/model"


def test_attest_and_register_raises_when_fail_on_violation(tmp_path):
    failed_result = _FakeAttestResult(tmp_path / "out", passed=False)
    with patch("squish.squash.mlflow_bridge.AttestPipeline.run", return_value=failed_result):
        with patch.object(MlflowBridge, "log_compress_run", return_value="run-123"):
            with pytest.raises(MlflowBridgeError):
                MlflowBridge.attest_and_register(
                    model_path=tmp_path,
                    model_name="org/model",
                    fail_on_violation=True,
                )


def test_mlflow_register_cli_success_outputs_json(tmp_path, capsys):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_result = _FakeAttestResult(tmp_path / "out", passed=True)
    fake_registration = MlflowRegistration(
        run_id="run-123",
        model_name="org/model",
        version="7",
        stage="Production",
        alias="champion",
        registered=True,
    )

    args = argparse.Namespace(
        model_path=str(model_dir),
        model_name="org/model",
        mlflow_tracking_uri=None,
        run_name=None,
        policies=None,
        sign=False,
        fail_on_violation=False,
        stage="Production",
        alias="champion",
    )

    with patch(
        "squish.squash.mlflow_bridge.MlflowBridge.attest_and_register",
        return_value=(fake_result, fake_registration),
    ):
        rc = _cmd_mlflow_register(args, quiet=True)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["attestation"]["passed"] is True
    assert payload["mlflow"]["registered"] is True
    assert payload["mlflow"]["version"] == "7"


def test_api_mlflow_register_returns_201_on_success(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    fake_result = _FakeAttestResult(tmp_path / "out", passed=True)
    fake_registration = MlflowRegistration(
        run_id="run-abc",
        model_name="org/model",
        version="9",
        stage="Production",
        alias="champion",
        registered=True,
    )

    client = TestClient(app)
    with patch(
        "squish.squash.mlflow_bridge.MlflowBridge.attest_and_register",
        return_value=(fake_result, fake_registration),
    ):
        resp = client.post(
            "/attest/mlflow/register",
            json={
                "model_path": str(model_dir),
                "model_name": "org/model",
                "policies": ["enterprise-strict"],
                "sign": False,
                "fail_on_violation": False,
                "stage": "Production",
                "alias": "champion",
            },
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["attestation"]["passed"] is True
    assert body["mlflow"]["registered"] is True
    assert body["mlflow"]["version"] == "9"
