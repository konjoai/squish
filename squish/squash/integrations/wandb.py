"""squish.squash.integrations.wandb — Weights & Biases adapter for Squash.

Attests a model artifact after it has been logged to W&B, then adds Squash
attestation files as additional artifact files and logs compliance metrics as
W&B summary values and artifact metadata.

Usage::

    import wandb
    from squish.squash.integrations.wandb import WandbSquash

    with wandb.init(project="my-llm") as run:
        artifact = wandb.Artifact("llama-3.1-8b-int4", type="model")
        artifact.add_dir("./output/llama-3.1-8b")
        run.log_artifact(artifact)

        result = WandbSquash.attest_artifact(
            artifact=artifact,
            model_path=Path("./output/llama-3.1-8b"),
            policies=["eu-ai-act"],
        )
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import wandb as _wandb_mod

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult

log = logging.getLogger(__name__)


class WandbSquash:
    """Attach Squash attestation to a W&B artifact."""

    @staticmethod
    def attest_artifact(
        artifact: "_wandb_mod.Artifact",
        model_path: Path,
        *,
        output_dir: Path | None = None,
        policies: list[str] | None = None,
        sign: bool = False,
        fail_on_violation: bool = False,
        **attest_kwargs,
    ) -> AttestResult:
        """Run attestation and attach results to *artifact*.

        The attestation artifacts are added as files to the W&B artifact so
        they travel with the model through all downstream usages.  Compliance
        metrics are also logged to the active run's summary.

        Parameters
        ----------
        artifact:
            An already-created :class:`wandb.Artifact` (before or after
            ``log_artifact`` — W&B allows late file additions).
        model_path:
            Local path to the model directory or file.
        output_dir:
            Where Squash writes artifacts; defaults to ``model_path.parent/squash``.
        policies:
            Policy templates to evaluate.
        sign:
            Sign via Sigstore.
        fail_on_violation:
            Raise on compliance failure.
        """
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required. Install with: pip install wandb"
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

        # Add every generated file to the artifact
        for p in out.glob("squash-*"):
            artifact.add_file(str(p), name=f"squash/{p.name}")
        if result.cyclonedx_path and result.cyclonedx_path.exists():
            artifact.add_file(
                str(result.cyclonedx_path), name="squash/cyclonedx-mlbom.json"
            )

        # Log compliance metrics in run summary
        run = wandb.run
        if run is not None:
            run.summary.update(
                {
                    "squash/passed": result.passed,
                    "squash/scan_status": (
                        result.scan_result.status if result.scan_result else "skipped"
                    ),
                    **{
                        f"squash/policy/{name}/passed": pr.passed
                        for name, pr in result.policy_results.items()
                    },
                }
            )

        return result
