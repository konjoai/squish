"""squish.squash.mlflow_bridge — real MLflow SDK registration bridge.

This module keeps MLflow integration optional (no hard dependency).
It provides a single high-level API:

    MlflowBridge.attest_and_register(...)

which performs:
1) Squash attestation run
2) MLflow run logging (params/metrics/artifacts/tags)
3) MLflow model registration
4) Optional stage transition + alias + compliance tags
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult


class MlflowBridgeError(RuntimeError):
    """Raised when MLflow registration cannot continue safely."""


@dataclass
class MlflowRegistration:
    """Result metadata for a model registration attempt."""

    run_id: str
    model_name: str
    version: str = ""
    stage: str = ""
    alias: str = ""
    tracking_uri: str = ""
    registered: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "version": self.version,
            "stage": self.stage,
            "alias": self.alias,
            "tracking_uri": self.tracking_uri,
            "registered": self.registered,
        }


class MlflowBridge:
    """Optional MLflow SDK integration for Squash attestation flows."""

    @staticmethod
    def _import_mlflow():
        try:
            import mlflow  # type: ignore[import-not-found]
        except ImportError as e:  # pragma: no cover - exercised in tests via patching
            raise ImportError(
                "mlflow is required for MLflow registration. Install with: pip install mlflow"
            ) from e
        return mlflow

    @classmethod
    def _client(cls):
        mlflow = cls._import_mlflow()
        return mlflow.tracking.MlflowClient()

    @classmethod
    def log_compress_run(cls, meta: dict[str, Any]) -> str:
        """Log a Squash run in MLflow and return the run_id.

        Expected keys in ``meta``:
        - run_name (optional)
        - mlflow_tracking_uri (optional)
        - params: dict
        - metrics: dict[str, float]
        - tags: dict
        - artifacts_dir (optional path-like)
        """
        mlflow = cls._import_mlflow()

        tracking_uri = str(meta.get("mlflow_tracking_uri") or "").strip()
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        run_name = meta.get("run_name")
        params = dict(meta.get("params") or {})
        metrics = dict(meta.get("metrics") or {})
        tags = {str(k): str(v) for k, v in dict(meta.get("tags") or {}).items()}
        artifacts_dir = meta.get("artifacts_dir")

        with mlflow.start_run(run_name=run_name) as run:
            for key, value in params.items():
                mlflow.log_param(str(key), str(value))

            for key, value in metrics.items():
                try:
                    mlflow.log_metric(str(key), float(value))
                except (TypeError, ValueError):
                    # Keep MLflow logging resilient to non-numeric accidental values.
                    continue

            if artifacts_dir:
                adir = Path(str(artifacts_dir))
                if adir.exists():
                    mlflow.log_artifacts(str(adir), artifact_path="squash")

            if tags:
                mlflow.set_tags(tags)

            return str(run.info.run_id)

    @classmethod
    def register_model(cls, run_id: str, model_name: str) -> str:
        """Register a model version from an MLflow run and return version."""
        if not run_id.strip():
            raise ValueError("run_id is required for MLflow model registration")
        if not model_name.strip():
            raise ValueError("model_name is required for MLflow model registration")

        client = cls._client()
        source = f"runs:/{run_id}/squash"
        model_version = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id,
        )
        version = str(getattr(model_version, "version", ""))
        if not version:
            raise MlflowBridgeError("MLflow did not return a model version")
        return version

    @classmethod
    def transition_to_production(cls, model_name: str, version: str) -> None:
        """Transition model version to Production with archival behavior."""
        client = cls._client()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True,
        )

    @classmethod
    def _transition_to_stage(cls, model_name: str, version: str, stage: str) -> None:
        client = cls._client()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=True,
        )

    @classmethod
    def _set_alias(cls, model_name: str, version: str, alias: str) -> None:
        client = cls._client()
        if hasattr(client, "set_registered_model_alias"):
            client.set_registered_model_alias(name=model_name, alias=alias, version=version)
        else:  # pragma: no cover - compatibility fallback for older MLflow
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key=f"alias.{alias}",
                value="true",
            )

    @classmethod
    def set_compliance_tags(
        cls,
        model_name: str,
        version: str,
        tags_dict: dict[str, Any],
    ) -> None:
        """Attach Squash compliance tags to an MLflow model version."""
        client = cls._client()
        for key, value in tags_dict.items():
            client.set_model_version_tag(
                name=model_name,
                version=version,
                key=str(key),
                value=str(value),
            )

    @staticmethod
    def _compliance_tags(result: AttestResult) -> dict[str, Any]:
        tags: dict[str, Any] = {
            "squash.passed": str(result.passed).lower(),
            "squash.scan_status": result.scan_result.status if result.scan_result else "skipped",
            "squash.model_id": result.model_id,
        }
        for pname, pres in result.policy_results.items():
            tags[f"squash.policy.{pname}.passed"] = str(pres.passed).lower()
            tags[f"squash.policy.{pname}.errors"] = pres.error_count
            tags[f"squash.policy.{pname}.warnings"] = pres.warning_count
        return tags

    @classmethod
    def attest_and_register(
        cls,
        *,
        model_path: Path,
        model_name: str,
        policies: list[str] | None = None,
        sign: bool = False,
        fail_on_violation: bool = False,
        stage: str | None = "Production",
        alias: str | None = "champion",
        mlflow_tracking_uri: str | None = None,
        run_name: str | None = None,
    ) -> tuple[AttestResult, MlflowRegistration]:
        """Run attestation, then log/register in MLflow when compliant."""
        model_path = Path(model_path)
        out_dir = model_path.parent / "squash"

        config = AttestConfig(
            model_path=model_path,
            output_dir=out_dir,
            policies=policies or ["enterprise-strict"],
            sign=sign,
            fail_on_violation=False,
        )
        result = AttestPipeline.run(config)

        compliance_tags = cls._compliance_tags(result)
        run_meta = {
            "run_name": run_name,
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "params": {
                "model_path": str(model_path),
                "model_name": model_name,
                "policies": ",".join(config.policies),
                "sign": sign,
            },
            "metrics": {
                "squash_passed": 1.0 if result.passed else 0.0,
                "policy_count": float(len(result.policy_results)),
            },
            "tags": compliance_tags,
            "artifacts_dir": str(result.output_dir),
        }
        run_id = cls.log_compress_run(run_meta)

        registration = MlflowRegistration(
            run_id=run_id,
            model_name=model_name,
            tracking_uri=mlflow_tracking_uri or "",
        )

        if not result.passed:
            if fail_on_violation:
                raise MlflowBridgeError("Attestation failed; refusing MLflow registration")
            return result, registration

        version = cls.register_model(run_id, model_name)
        registration.version = version
        registration.registered = True

        cls.set_compliance_tags(model_name, version, compliance_tags)

        stage_value = (stage or "").strip()
        if stage_value:
            if stage_value.lower() == "production":
                cls.transition_to_production(model_name, version)
                registration.stage = "Production"
            else:
                cls._transition_to_stage(model_name, version, stage_value)
                registration.stage = stage_value

        alias_value = (alias or "").strip()
        if alias_value:
            cls._set_alias(model_name, version, alias_value)
            registration.alias = alias_value

        return result, registration
