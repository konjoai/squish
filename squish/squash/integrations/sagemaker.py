"""squish.squash.integrations.sagemaker — SageMaker adapter for Squash attestation.

Attests a local model directory within a SageMaker pipeline and then:

1. Uploads every Squash artifact (BOM, SPDX, policy reports, …) to S3 alongside
   the model artefacts.
2. Tags the SageMaker Model or ModelPackage with ``squash:passed``,
   ``squash:scan_status``, and per-policy results so downstream Model Registry
   quality gates can query them.

Usage::

    from squish.squash.integrations.sagemaker import SageMakerSquash

    result = SageMakerSquash.attach_attestation(
        model_path=Path("./output/llama-3.1-8b"),
        model_package_arn="arn:aws:sagemaker:us-east-1:123456789012:model-package/my-model/1",
        s3_upload_prefix="s3://my-bucket/squash-boms/llama-3.1-8b/",
        policies=["eu-ai-act", "nist-ai-rmf"],
    )
    # Tags the ModelPackage with squash:passed=true/false, squash:scan_status=clean, …

For use inside a SageMaker Pipeline Step, pass ``attach_attestation`` as a
``ProcessingStep`` script or invoke it directly from a custom ``PythonScriptStep``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # boto3 / sagemaker imported lazily below

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult

log = logging.getLogger(__name__)

# SageMaker tag key namespace — colon separator matches AWS tag conventions
_TAG_PREFIX = "squash:"


class SageMakerSquash:
    """Attach Squash attestation artifacts and tags to a SageMaker model."""

    @staticmethod
    def attach_attestation(
        model_path: Path,
        *,
        model_package_arn: str | None = None,
        s3_upload_prefix: str | None = None,
        policies: list[str] | None = None,
        sign: bool = False,
        fail_on_violation: bool = False,
        tag_prefix: str = _TAG_PREFIX,
        **attest_kwargs,
    ) -> AttestResult:
        """Attest *model_path* and optionally tag the SageMaker ModelPackage.

        Parameters
        ----------
        model_path:
            Local path to the model directory or file being attested.
        model_package_arn:
            ARN of the SageMaker Model or ModelPackage to tag.  If *None*, no
            tags are written (useful for dry-run or pre-registration workflows).
        s3_upload_prefix:
            S3 URI prefix where Squash artifacts are uploaded, e.g.
            ``"s3://my-bucket/squash-boms/llama-3.1-8b/"``.  If *None*, no S3
            upload is performed.
        policies:
            Policy templates to evaluate; defaults to ``["enterprise-strict"]``.
        sign:
            Sign the CycloneDX BOM with Sigstore.
        fail_on_violation:
            Raise on policy/scan failure.
        tag_prefix:
            AWS tag key prefix, defaults to ``"squash:"``.
        **attest_kwargs:
            Additional keyword arguments forwarded to :class:`AttestConfig`.

        Returns
        -------
        AttestResult
        """
        try:
            import boto3 as _boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for SageMakerSquash. "
                "Install with: pip install boto3"
            ) from e

        out = model_path.parent / "squash"

        config = AttestConfig(
            model_path=model_path,
            output_dir=out,
            policies=policies if policies is not None else ["enterprise-strict"],
            sign=sign,
            fail_on_violation=fail_on_violation,
            **attest_kwargs,
        )
        result = AttestPipeline.run(config)

        # Upload artifacts to S3 if a prefix was supplied
        if s3_upload_prefix:
            SageMakerSquash._upload_to_s3(_boto3, out, s3_upload_prefix)

        # Tag ModelPackage / Model with attestation results
        if model_package_arn:
            SageMakerSquash.tag_model_package(
                model_package_arn=model_package_arn,
                result=result,
                tag_prefix=tag_prefix,
            )

        log.info(
            "SageMaker: squash.passed=%s for model_path=%s",
            result.passed,
            model_path,
        )
        return result

    @staticmethod
    def tag_model_package(
        model_package_arn: str,
        result: AttestResult,
        *,
        tag_prefix: str = _TAG_PREFIX,
    ) -> None:
        """Write Squash attestation results as AWS tags on a SageMaker resource.

        Parameters
        ----------
        model_package_arn:
            Full ARN of the SageMaker Model or ModelPackage to tag.
        result:
            Attestation result returned by :meth:`attach_attestation`.
        tag_prefix:
            Tag key prefix, defaults to ``"squash:"``.
        """
        try:
            import boto3 as _boto3
        except ImportError as e:
            raise ImportError(
                "boto3 is required for SageMakerSquash. "
                "Install with: pip install boto3"
            ) from e

        sm = _boto3.client("sagemaker")

        tags: list[dict[str, str]] = [
            {"Key": f"{tag_prefix}passed", "Value": str(result.passed).lower()},
            {
                "Key": f"{tag_prefix}scan_status",
                "Value": result.scan_result.status if result.scan_result else "skipped",
            },
        ]
        for policy_name, pr in result.policy_results.items():
            tags.append(
                {"Key": f"{tag_prefix}policy.{policy_name}.passed", "Value": str(pr.passed).lower()}
            )
            tags.append(
                {"Key": f"{tag_prefix}policy.{policy_name}.errors", "Value": str(pr.error_count)}
            )

        sm.add_tags(ResourceArn=model_package_arn, Tags=tags)
        log.debug("SageMaker: tagged %s with %d squash tags", model_package_arn, len(tags))

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _upload_to_s3(boto3_module, local_dir: Path, s3_prefix: str) -> None:
        """Upload every file in *local_dir* to *s3_prefix* using boto3.

        Parameters
        ----------
        boto3_module:
            The already-imported boto3 module.
        local_dir:
            Local directory whose contents will be uploaded.
        s3_prefix:
            Destination S3 URI, e.g. ``"s3://bucket/prefix/"`` — trailing
            slash is optional.
        """
        if not local_dir.exists():
            log.debug("SageMaker: no output dir %s — skipping S3 upload", local_dir)
            return

        # Parse s3://bucket/key-prefix
        s3_prefix = s3_prefix.rstrip("/")
        assert s3_prefix.startswith("s3://"), f"s3_upload_prefix must start with 's3://': {s3_prefix}"
        _, _, rest = s3_prefix.partition("//")
        bucket, _, key_prefix = rest.partition("/")

        s3 = boto3_module.client("s3")
        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(local_dir)
            s3_key = f"{key_prefix}/{rel}" if key_prefix else str(rel)
            s3.upload_file(str(file_path), bucket, s3_key)
            log.debug("SageMaker: uploaded s3://%s/%s", bucket, s3_key)
