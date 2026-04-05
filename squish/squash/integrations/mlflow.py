"""squish.squash.integrations.mlflow — MLflow adapter for Squash attestation.

Attests a local model directory immediately after logging a run, then:
1. Uploads every Squash artifact (BOM, SPDX, policy reports, …) as MLflow
   run artifacts.
2. Sets MLflow tags that downstream dashboards and quality gates can query.

Usage::

    import mlflow
    from squish.squash.integrations.mlflow import MLflowSquash

    with mlflow.start_run() as run:
        # … your training / fine-tuning …
        mlflow.pytorch.log_model(model, "model")

        result = MLflowSquash.attest_run(
            run=run,
            model_path=Path("./output/llama-3.1-8b"),
            policies=["eu-ai-act", "nist-ai-rmf"],
        )
        # Tags squash.passed=true/false, squash.scan_status=clean, …
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlflow

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult

log = logging.getLogger(__name__)

# MLflow tag namespace
_TAG_PREFIX = "squash."


class MLflowSquash:
    """Attach Squash attestation artifacts and tags to an MLflow run."""

    @staticmethod
    def attest_run(
        run: "mlflow.ActiveRun | mlflow.entities.Run",
        model_path: Path,
        *,
        output_dir: Path | None = None,
        policies: list[str] | None = None,
        sign: bool = False,
        fail_on_violation: bool = False,
        artifact_prefix: str = "squash",
        **attest_kwargs,
    ) -> AttestResult:
        """Attest *model_path* and upload artifacts to the active MLflow run.

        Parameters
        ----------
        run:
            Active MLflow run context or a resolved
            :class:`mlflow.entities.Run` object.
        model_path:
            Local path to the model directory or file being attested.
        output_dir:
            Where Squash writes its artifacts; defaults to a ``squash/``
            subdirectory under *model_path*'s parent so they can be uploaded
            with a tidy prefix.
        policies:
            Policy templates to evaluate; defaults to ``["enterprise-strict"]``.
        sign:
            Sign the CycloneDX BOM with Sigstore.
        fail_on_violation:
            Raise on policy/scan failure.
        artifact_prefix:
            MLflow artifact path prefix, defaults to ``"squash"``.

        Returns
        -------
        AttestResult
        """
        try:
            import mlflow as _mlflow
        except ImportError as e:
            raise ImportError(
                "mlflow is required for MLflowSquash. "
                "Install with: pip install mlflow"
            ) from e

        out = output_dir or (model_path.parent / "squash")

        config = AttestConfig(
            model_path=model_path,
            output_dir=out,
            policies=policies if policies is not None else ["enterprise-strict"],
            sign=sign,
            fail_on_violation=fail_on_violation,
            **attest_kwargs,
        )
        result = AttestPipeline.run(config)

        # Upload every artifact in output_dir
        _mlflow.log_artifacts(str(out), artifact_path=artifact_prefix)

        # Set structured tags for dashboards and downstream quality gates
        tags = {
            f"{_TAG_PREFIX}passed": str(result.passed).lower(),
            f"{_TAG_PREFIX}scan_status": (
                result.scan_result.status if result.scan_result else "skipped"
            ),
        }
        for policy_name, pr in result.policy_results.items():
            tags[f"{_TAG_PREFIX}policy.{policy_name}.passed"] = str(pr.passed).lower()
            tags[f"{_TAG_PREFIX}policy.{policy_name}.errors"] = str(pr.error_count)
        _mlflow.set_tags(tags)

        if result.cyclonedx_path:
            log.info(
                "MLflow: tagged run %s with squash.passed=%s, artifacts at '%s/'",
                _mlflow.active_run().info.run_id if _mlflow.active_run() else "?",
                result.passed,
                artifact_prefix,
            )
        return result
