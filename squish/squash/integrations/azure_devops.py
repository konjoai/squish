"""squish.squash.integrations.azure_devops — Azure DevOps adapter for Squash attestation.

Provides a Python adapter that runs the Squash attestation pipeline and emits
Azure DevOps Pipeline logging commands so results appear inline in the
pipeline run UI and are surfaced as pipeline output variables for downstream
tasks.

Usage (Python Script task in ``azure-pipelines.yml``)::

    from pathlib import Path
    from squish.squash.integrations.azure_devops import AzureDevOpsSquash

    result = AzureDevOpsSquash.attest(
        model_path=Path("$(Build.ArtifactStagingDirectory)/model"),
        policies=["eu-ai-act", "nist-ai-rmf"],
        sign=False,
        fail_on_violation=True,
    )

ADO logging commands emitted to stdout:

- ``##vso[task.setvariable variable=SQUASH_PASSED;isOutput=true]true/false``
- ``##vso[task.setvariable variable=SQUASH_SCAN_STATUS;isOutput=true]clean/…``
- ``##vso[task.setvariable variable=SQUASH_CYCLONEDX_PATH;isOutput=true]…``
- ``##vso[task.setvariable variable=SQUASH_SPDX_JSON_PATH;isOutput=true]…``
- ``##vso[task.setvariable variable=SQUASH_MASTER_RECORD_PATH;isOutput=true]…``
- ``##vso[task.logissue type=error]…`` on policy violations when *fail_on_violation=True*
- ``##vso[task.complete result=Succeeded/Failed;]…``

No azure-pipelines-task-lib or azure-specific SDK required.  All result
communication happens via the ``##vso[...]`` stdout-logging protocol understood
by every ADO agent 2.x+.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # No ADO SDK imported at module level — all communication is stdout-based.

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult

log = logging.getLogger(__name__)

# Environment variable set by ADO agent on every pipeline run.
_ADO_AGENT_ENV_VAR = "TF_BUILD"

# Default artifact namespace used in ##vso[artifact.upload] commands.
_DEFAULT_ARTIFACT_NAME = "squash-attestation"


# ---------------------------------------------------------------------------
# Context detection
# ---------------------------------------------------------------------------


def is_ado_context() -> bool:
    """Return *True* if the current process is running inside an ADO pipeline.

    The ``TF_BUILD`` environment variable is set to ``"True"`` by the ADO
    agent on every pipeline job — it is the standard way to detect an
    Azure Pipelines execution context.
    """
    import os

    return os.environ.get(_ADO_AGENT_ENV_VAR) == "True"


# ---------------------------------------------------------------------------
# Low-level ADO logging command helpers
# ---------------------------------------------------------------------------


def _emit_vso(command: str, message: str = "", **props: str) -> None:
    """Write a single ``##vso[command prop=val;…]message`` line to stdout.

    Parameters
    ----------
    command:
        ADO logging command name, e.g. ``"task.setvariable"`` or
        ``"task.logissue"``.
    message:
        Optional message body appended after the closing ``]``.
    **props:
        Key-value properties embedded inside ``[command …]``.
    """
    if props:
        prop_str = ";".join(f"{k}={v}" for k, v in props.items()) + ";"
        bracket = f"{command} {prop_str}"
    else:
        bracket = command
    print(f"##vso[{bracket}]{message}", flush=True)


def set_variable(
    name: str,
    value: str,
    *,
    is_output: bool = True,
    is_secret: bool = False,
) -> None:
    """Emit ``##vso[task.setvariable…]value`` to the ADO agent.

    Parameters
    ----------
    name:
        Variable name used by downstream tasks via
        ``$(SquashAttest.SQUASH_PASSED)`` syntax.
    value:
        Variable value as a string.
    is_output:
        If *True* (default), the variable is surfaced as an output variable
        reachable by references like ``$(StepName.SQUASH_PASSED)`` in later
        pipeline steps.
    is_secret:
        If *True*, the agent masks the value in all log output.
    """
    _emit_vso(
        "task.setvariable",
        value,
        variable=name,
        isOutput=str(is_output).lower(),
        isSecret=str(is_secret).lower(),
    )


def log_issue(level: str, message: str) -> None:
    """Emit ``##vso[task.logissue type=error|warning]message``.

    Parameters
    ----------
    level:
        ``"error"`` or ``"warning"``.
    message:
        Human-readable message body.
    """
    _emit_vso("task.logissue", message, type=level)


def publish_artifact(
    path: Path,
    *,
    artifact_name: str = _DEFAULT_ARTIFACT_NAME,
) -> None:
    """Emit ``##vso[artifact.upload artifactname=…;]path`` for each file.

    Parameters
    ----------
    path:
        Local file path to attach to the pipeline run.
    artifact_name:
        ADO artifact container name; defaults to ``"squash-attestation"``.
    """
    _emit_vso("artifact.upload", str(path), artifactname=artifact_name)


def complete_task(passed: bool, *, message: str = "") -> None:
    """Emit ``##vso[task.complete result=Succeeded/Failed;]message``.

    Parameters
    ----------
    passed:
        If *True*, the task result is ``Succeeded``; otherwise ``Failed``.
    message:
        Summary appended to the logging command.
    """
    result_str = "Succeeded" if passed else "Failed"
    _emit_vso("task.complete", message, result=result_str)


# ---------------------------------------------------------------------------
# High-level adapter
# ---------------------------------------------------------------------------


class AzureDevOpsSquash:
    """Attach Squash attestation to an Azure DevOps pipeline job.

    The class method :meth:`attest` is the primary entry point.  It:

    1. Runs the :class:`~squish.squash.attest.AttestPipeline` against the
       supplied model directory.
    2. Emits ADO ``##vso[task.setvariable…]`` commands for all five standard
       output variables so downstream pipeline tasks can consume the results.
    3. Optionally publishes all attestation artifacts to the pipeline run via
       ``##vso[artifact.upload]``.
    4. Emits ``##vso[task.complete result=…]`` to surface the overall result
       in the Azure Pipelines UI.
    5. Raises :class:`SystemExit` (exit code 1) if *fail_on_violation=True*
       and attestation did not pass — the ADO agent interprets this as a
       task failure.
    """

    @staticmethod
    def attest(
        model_path: Path,
        policies: list[str] | None = None,
        *,
        sign: bool = False,
        fail_on_violation: bool = True,
        output_dir: Path | None = None,
        publish_artifacts: bool = True,
        artifact_name: str = _DEFAULT_ARTIFACT_NAME,
        **attest_kwargs,
    ) -> AttestResult:
        """Run Squash attestation and emit ADO pipeline logging commands.

        Parameters
        ----------
        model_path:
            Local path to the model directory or file being attested.
        policies:
            Policy templates to evaluate; defaults to ``["enterprise-strict"]``.
        sign:
            Sign the CycloneDX BOM with Sigstore.
        fail_on_violation:
            If *True* (default), call ``sys.exit(1)`` when attestation fails so
            the ADO task is marked as failed.  Set to *False* for non-blocking
            attestation that only sets output variables.
        output_dir:
            Directory for Squash output artifacts; defaults to
            ``model_path.parent / "squash"``.
        publish_artifacts:
            If *True* (default), emit ``##vso[artifact.upload…]`` for every
            Squash artifact so they appear in the pipeline run's *Artifacts* tab.
        artifact_name:
            ADO artifact container name; defaults to ``"squash-attestation"``.
        **attest_kwargs:
            Additional keyword arguments forwarded to :class:`AttestConfig`.

        Returns
        -------
        AttestResult
        """
        out = output_dir if output_dir is not None else model_path.parent / "squash"

        config = AttestConfig(
            model_path=model_path,
            output_dir=out,
            policies=policies if policies is not None else ["enterprise-strict"],
            sign=sign,
            fail_on_violation=False,  # We handle the gate ourselves below.
            **attest_kwargs,
        )

        result: AttestResult = AttestPipeline.run(config)

        # ── Set pipeline output variables ─────────────────────────────────────
        scan_status = (
            result.scan_result.status if result.scan_result is not None else "skipped"
        )

        set_variable("SQUASH_PASSED", str(result.passed).lower())
        set_variable("SQUASH_SCAN_STATUS", scan_status)

        artifacts = getattr(result, "artifacts", {}) or {}
        cyclonedx    = artifacts.get("cyclonedx", "")
        spdx_json    = artifacts.get("spdx_json", "")
        master_record = artifacts.get("master_record", "")

        if cyclonedx:
            set_variable("SQUASH_CYCLONEDX_PATH", cyclonedx)
        if spdx_json:
            set_variable("SQUASH_SPDX_JSON_PATH", spdx_json)
        if master_record:
            set_variable("SQUASH_MASTER_RECORD_PATH", master_record)

        # ── Emit policy summary to task log ──────────────────────────────────
        print()
        for policy_name, pr in result.policy_results.items():
            status_tag = "PASS" if pr.passed else "FAIL"
            print(
                f"  [{status_tag}] {policy_name} : "
                f"{pr.error_count} error(s), {pr.warning_count} warning(s)"
            )
        print()

        # ── Publish artifact files ────────────────────────────────────────────
        if publish_artifacts:
            for artifact_path_str in [cyclonedx, spdx_json, master_record]:
                if artifact_path_str:
                    p = Path(artifact_path_str)
                    if p.is_file():
                        publish_artifact(p, artifact_name=artifact_name)

        # ── Emit policy violations as log errors ──────────────────────────────
        if not result.passed:
            log_issue(
                "error",
                f"Squash attestation FAILED for '{model_path}'. "
                f"Scan: {scan_status}.",
            )

        # ── Complete the ADO task ─────────────────────────────────────────────
        msg = (
            f"Squash attestation {'passed' if result.passed else 'FAILED'} "
            f"for '{model_path}'. Scan: {scan_status}."
        )
        complete_task(result.passed, message=msg)

        log.info(
            "AzureDevOps: squash_passed=%s scan_status=%s model_path=%s",
            result.passed,
            scan_status,
            model_path,
        )

        if fail_on_violation and not result.passed:
            sys.exit(1)

        return result
