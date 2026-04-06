"""squish.squash.integrations.vertex_ai — Vertex AI adapter for Squash attestation.

Attests a local model directory within a Google Cloud / Vertex AI workflow and then:

1. Uploads every Squash artifact (BOM, SPDX, policy reports, …) to a GCS bucket
   alongside the model artefacts.
2. Labels the Vertex AI Model resource with ``squash_passed``, ``squash_scan_status``,
   and per-policy results so Vertex AI Model Registry governance queries can filter on
   them.

Usage::

    from squish.squash.integrations.vertex_ai import VertexAISquash

    result = VertexAISquash.attach_attestation(
        model_path=Path("./output/gemma-3-4b"),
        model_resource_name="projects/my-project/locations/us-central1/models/12345",
        gcs_upload_prefix="gs://my-bucket/squash-boms/gemma-3-4b/",
        policies=["eu-ai-act", "nist-ai-rmf"],
    )
    # Labels the Model with squash_passed=true/false, squash_scan_status=clean, …

For use inside a Vertex AI Pipeline component, import this module inside the component
function body so the google-cloud-aiplatform SDK is loaded lazily.

GCP label key/value constraints (applied automatically by :func:`_sanitize_label`):

- Lowercase letters, digits, underscores, hyphens only.
- Keys must start with a letter.
- Maximum 63 characters per key; 63 characters per value.
- Colons and dots in Squash policy names are replaced with underscores.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # google-cloud-aiplatform / google-cloud-storage imported lazily below

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult

log = logging.getLogger(__name__)

# GCP label key prefix — underscores (colons violate GCP label grammar)
_LABEL_PREFIX = "squash_"

# GCP label constraints
_MAX_LABEL_LEN = 63
_LABEL_KEY_RE = re.compile(r"[^a-z0-9_\-]")


def _sanitize_label(value: str, *, max_len: int = _MAX_LABEL_LEN) -> str:
    """Return *value* as a valid GCP label key or value.

    Rules applied:
    - Lower-case the string.
    - Replace every character that is not ``[a-z0-9_-]`` with ``_``.
    - If the result starts with a digit, prepend ``sq``.
    - Truncate to *max_len* characters.
    """
    sanitized = _LABEL_KEY_RE.sub("_", value.lower())
    if sanitized and sanitized[0].isdigit():
        sanitized = "sq" + sanitized
    return sanitized[:max_len]


class VertexAISquash:
    """Attach Squash attestation artifacts and labels to a Vertex AI Model resource."""

    @staticmethod
    def attach_attestation(
        model_path: Path,
        *,
        model_resource_name: str | None = None,
        gcs_upload_prefix: str | None = None,
        policies: list[str] | None = None,
        sign: bool = False,
        fail_on_violation: bool = False,
        label_prefix: str = _LABEL_PREFIX,
        **attest_kwargs,
    ) -> AttestResult:
        """Attest *model_path* and optionally label the Vertex AI Model resource.

        Parameters
        ----------
        model_path:
            Local path to the model directory or file being attested.
        model_resource_name:
            Full Vertex AI Model resource name, e.g.
            ``"projects/my-project/locations/us-central1/models/12345"``.
            If *None*, no labels are written (useful for dry-run workflows).
        gcs_upload_prefix:
            GCS URI prefix where Squash artifacts are uploaded, e.g.
            ``"gs://my-bucket/squash-boms/gemma-3-4b/"``.  If *None*, no GCS
            upload is performed.
        policies:
            Policy templates to evaluate; defaults to ``["enterprise-strict"]``.
        sign:
            Sign the CycloneDX BOM with Sigstore.
        fail_on_violation:
            Raise on policy/scan failure.
        label_prefix:
            GCP label key prefix, defaults to ``"squash_"``.
        **attest_kwargs:
            Additional keyword arguments forwarded to :class:`AttestConfig`.

        Returns
        -------
        AttestResult
        """
        try:
            from google.cloud import aiplatform as _aiplatform  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform is required for VertexAISquash. "
                "Install with: pip install google-cloud-aiplatform"
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

        # Upload artifacts to GCS if a prefix was supplied
        if gcs_upload_prefix:
            VertexAISquash._upload_to_gcs(out, gcs_upload_prefix)

        # Label the Vertex AI Model with attestation results
        if model_resource_name:
            VertexAISquash.label_model(
                model_resource_name=model_resource_name,
                result=result,
                label_prefix=label_prefix,
            )

        log.info(
            "VertexAI: squash_passed=%s for model_path=%s",
            result.passed,
            model_path,
        )
        return result

    @staticmethod
    def label_model(
        model_resource_name: str,
        result: AttestResult,
        *,
        label_prefix: str = _LABEL_PREFIX,
    ) -> None:
        """Write Squash attestation results as GCP labels on a Vertex AI Model.

        Parameters
        ----------
        model_resource_name:
            Full resource name, e.g.
            ``"projects/my-project/locations/us-central1/models/12345"``.
        result:
            Attestation result returned by :meth:`attach_attestation`.
        label_prefix:
            GCP label key prefix, defaults to ``"squash_"``.
        """
        try:
            from google.cloud import aiplatform as _aiplatform
        except ImportError as e:
            raise ImportError(
                "google-cloud-aiplatform is required for VertexAISquash. "
                "Install with: pip install google-cloud-aiplatform"
            ) from e

        model = _aiplatform.Model(model_name=model_resource_name)

        labels: dict[str, str] = {
            _sanitize_label(f"{label_prefix}passed"): str(result.passed).lower(),
            _sanitize_label(f"{label_prefix}scan_status"): _sanitize_label(
                result.scan_result.status if result.scan_result else "skipped"
            ),
        }
        for policy_name, pr in result.policy_results.items():
            key_passed = _sanitize_label(f"{label_prefix}policy_{policy_name}_passed")
            key_errors = _sanitize_label(f"{label_prefix}policy_{policy_name}_errors")
            labels[key_passed] = str(pr.passed).lower()
            labels[key_errors] = str(pr.error_count)

        model.update(labels=labels)
        log.debug(
            "VertexAI: labelled %s with %d squash labels",
            model_resource_name,
            len(labels),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _upload_to_gcs(local_dir: Path, gcs_prefix: str) -> None:
        """Upload every file in *local_dir* to *gcs_prefix*.

        Parameters
        ----------
        local_dir:
            Local directory whose contents will be uploaded.
        gcs_prefix:
            Destination GCS URI, e.g. ``"gs://bucket/prefix/"`` — trailing
            slash is optional.
        """
        try:
            from google.cloud import storage as _storage
        except ImportError as e:
            raise ImportError(
                "google-cloud-storage is required for GCS upload in VertexAISquash. "
                "Install with: pip install google-cloud-storage"
            ) from e

        if not local_dir.exists():
            log.debug("VertexAI: no output dir %s — skipping GCS upload", local_dir)
            return

        # Parse gs://bucket/key-prefix
        gcs_prefix = gcs_prefix.rstrip("/")
        assert gcs_prefix.startswith("gs://"), (
            f"gcs_upload_prefix must start with 'gs://': {gcs_prefix}"
        )
        _, _, rest = gcs_prefix.partition("//")
        bucket_name, _, key_prefix = rest.partition("/")

        client = _storage.Client()
        bucket = client.bucket(bucket_name)

        for file_path in local_dir.rglob("*"):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(local_dir)
            blob_name = f"{key_prefix}/{rel}" if key_prefix else str(rel)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(file_path))
            log.debug("VertexAI: uploaded gs://%s/%s", bucket_name, blob_name)
