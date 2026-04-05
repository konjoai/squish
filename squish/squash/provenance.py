"""squish/squash/provenance.py — Training data provenance for AI models.

Provides cryptographic binding of training dataset fingerprints to model
attestations.  Supports three input forms:

1. **HuggingFace dataset IDs** — resolved via HF Hub API to get the repo's
   latest commit SHA as the dataset fingerprint.
2. **S3 URIs** — manifest CSV/JSON listing files with their ETags/checksums.
3. **Local datasheet JSON** — a structured document following the Datasheets
   for Datasets format (Gebru et al. 2018).

The dataset provenance is embedded in the CycloneDX SBOM as ``formulation``
entries (CycloneDX 1.7 workflow components) and as ``GENERATED_FROM``
relationships in SPDX output.

Usage::

    prov = ProvenanceCollector.from_hf_datasets(
        ["wikipedia", "c4", "the-pile"],
        token=os.environ.get("HF_TOKEN"),
    )
    prov.bind_to_sbom(Path("./model/cyclonedx-mlbom.json"))

The bind operation is *atomic* (write to tmp, rename) to prevent partial writes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class DatasetRecord:
    """Provenance record for a single training dataset.

    Parameters
    ----------
    dataset_id:
        Human-readable identifier.  For HuggingFace: ``"org/repo"`` or
        ``"repo"``.  For S3: the bucket/prefix.  For local: filename stem.
    source_type:
        One of ``"huggingface"`` | ``"s3"`` | ``"local"`` | ``"unknown"``.
    sha256:
        SHA-256 fingerprint of the dataset.  For HuggingFace: hash of the
        latest commit SHA.  For S3: hash of the manifest file.  For local:
        hash of the datasheet JSON.
    uri:
        Canonical URI for the dataset.
    version:
        Dataset version or commit SHA.
    license:
        SPDX license expression, e.g. ``"CC-BY-4.0"`` or ``"NOASSERTION"``.
    contains_pii:
        Whether the dataset contains personally identifiable information.
    """

    dataset_id: str
    source_type: str = "unknown"
    sha256: str = ""
    uri: str = ""
    version: str = ""
    license: str = "NOASSERTION"
    contains_pii: bool = False
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_cdx_formulation(self) -> dict[str, Any]:
        """Convert to a CycloneDX 1.7 formulation-style workflow dict."""
        comp: dict[str, Any] = {
            "type": "data",
            "bom-ref": f"dataset/{self.dataset_id}",
            "name": self.dataset_id,
            "version": self.version or "unknown",
            "downloadLocation": self.uri,
            "externalReferences": (
                [{"type": "vcs", "url": self.uri}] if self.uri else []
            ),
            "properties": [
                {"name": "squash:datasetSourceType", "value": self.source_type},
                {"name": "squash:containsPII", "value": str(self.contains_pii).lower()},
                {"name": "squash:datasetLicense", "value": self.license},
            ],
        }
        if self.sha256:
            comp["hashes"] = [{"alg": "SHA-256", "content": self.sha256}]
        return comp


@dataclass
class ProvenanceManifest:
    """Collection of dataset records for a model's training provenance."""

    datasets: list[DatasetRecord] = field(default_factory=list)
    composite_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.composite_sha256 and self.datasets:
            self.composite_sha256 = self._compute_composite()

    def _compute_composite(self) -> str:
        """Hash all dataset SHA-256s into one composite fingerprint."""
        combined = "".join(sorted(d.sha256 for d in self.datasets if d.sha256))
        return hashlib.sha256(combined.encode()).hexdigest()

    def bind_to_sbom(self, bom_path: Path) -> None:
        """Atomically inject dataset provenance into an existing CycloneDX BOM.

        Adds / replaces:
        - ``formulation`` array at the document level (dataset components)
        - ``squash:trainingDataComposite`` property on the model component

        Parameters
        ----------
        bom_path:
            Path to an existing ``cyclonedx-mlbom.json``.

        Raises
        ------
        FileNotFoundError
            If *bom_path* does not exist.
        """
        if not bom_path.exists():
            raise FileNotFoundError(f"SBOM not found: {bom_path}")

        bom: dict = json.loads(bom_path.read_text())

        # Build formulation entries
        formulation = [d.to_cdx_formulation() for d in self.datasets]
        bom["formulation"] = formulation

        # Annotate the model component
        components = bom.get("components", [])
        if components:
            props = components[0].setdefault("properties", [])
            # Remove existing training data properties to avoid duplication
            props[:] = [
                p for p in props
                if not p.get("name", "").startswith("squash:trainingData")
            ]
            props.append(
                {
                    "name": "squash:trainingDataComposite",
                    "value": self.composite_sha256,
                }
            )
            props.extend(
                {"name": "squash:trainingDataset", "value": d.dataset_id}
                for d in self.datasets
            )

        # Atomic write
        tmp = bom_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(bom, indent=2))
        tmp.replace(bom_path)
        log.info(
            "Bound %d dataset(s) to SBOM %s (composite: %s…)",
            len(self.datasets),
            bom_path,
            self.composite_sha256[:16],
        )


class ProvenanceCollector:
    """Collect training data provenance from multiple sources.

    All class methods return a :class:`ProvenanceManifest`.
    """

    @staticmethod
    def from_hf_datasets(
        dataset_ids: list[str],
        token: str | None = None,
    ) -> ProvenanceManifest:
        """Resolve HuggingFace dataset IDs to provenance records.

        Parameters
        ----------
        dataset_ids:
            List of HF dataset repository IDs, e.g. ``["wikipedia", "c4"]``.
        token:
            HuggingFace API token for private datasets.  When ``None``,
            reads ``HF_TOKEN`` from the environment.
        """
        hf_token = token or os.environ.get("HF_TOKEN", "")
        records: list[DatasetRecord] = []

        for ds_id in dataset_ids:
            record = ProvenanceCollector._resolve_hf_dataset(ds_id, hf_token)
            records.append(record)

        return ProvenanceManifest(datasets=records)

    @staticmethod
    def from_s3_manifest(
        manifest_uri: str,
        access_key: str | None = None,
        secret_key: str | None = None,
    ) -> ProvenanceManifest:
        """Parse an S3 dataset manifest (CSV or JSON) into provenance records.

        The manifest is expected to be a JSON file or CSV with columns:
        ``dataset_id, file_path, sha256``.

        Parameters
        ----------
        manifest_uri:
            S3 URI of the manifest file, e.g. ``"s3://my-bucket/manifest.json"``.
        access_key / secret_key:
            AWS credentials.  When ``None``, uses the default credential chain.
        """
        # Resolve via boto3 if available; fall back to a warning
        try:
            import boto3  # noqa: F401
        except ImportError:
            log.warning(
                "boto3 not installed — S3 manifest resolution unavailable. "
                "Install with: pip install boto3"
            )
            return ProvenanceManifest(
                datasets=[
                    DatasetRecord(
                        dataset_id=manifest_uri,
                        source_type="s3",
                        uri=manifest_uri,
                        sha256="",
                    )
                ]
            )

        import boto3

        parsed = _parse_s3_uri(manifest_uri)
        bucket, key = parsed

        session = boto3.Session(
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        s3 = session.client("s3")

        try:
            resp = s3.get_object(Bucket=bucket, Key=key)
            manifest_bytes = resp["Body"].read()
            etag = resp.get("ETag", "").strip('"')
        except Exception as e:
            log.error("Failed to fetch S3 manifest %s: %s", manifest_uri, e)
            return ProvenanceManifest(datasets=[])

        # Hash the manifest itself as the dataset fingerprint
        sha256 = hashlib.sha256(manifest_bytes).hexdigest()

        return ProvenanceManifest(
            datasets=[
                DatasetRecord(
                    dataset_id=key.rsplit("/", 1)[-1],
                    source_type="s3",
                    uri=manifest_uri,
                    sha256=sha256,
                    version=etag,
                )
            ]
        )

    @staticmethod
    def from_datasheet(datasheet_path: Path) -> ProvenanceManifest:
        """Parse a Datasheets-for-Datasets JSON file into a provenance record.

        Expects a JSON document with at least:
        ``{"name": "...", "license": "...", "version": "...", "pii": bool}``

        Parameters
        ----------
        datasheet_path:
            Local path to the datasheet JSON.
        """
        raw_bytes = datasheet_path.read_bytes()
        sha256 = hashlib.sha256(raw_bytes).hexdigest()

        try:
            ds: dict = json.loads(raw_bytes)
        except json.JSONDecodeError as e:
            log.error("Malformed datasheet JSON %s: %s", datasheet_path, e)
            return ProvenanceManifest(datasets=[])

        record = DatasetRecord(
            dataset_id=ds.get("name", datasheet_path.stem),
            source_type="local",
            sha256=sha256,
            uri=str(datasheet_path.resolve()),
            version=str(ds.get("version", "")),
            license=ds.get("license", "NOASSERTION"),
            contains_pii=bool(ds.get("pii", False)),
            raw_metadata=ds,
        )
        return ProvenanceManifest(datasets=[record])

    @staticmethod
    def _resolve_hf_dataset(
        dataset_id: str, token: str
    ) -> DatasetRecord:
        """Resolve a single HuggingFace dataset ID via the Hub API."""
        url = f"https://huggingface.co/api/datasets/{dataset_id}"
        headers: dict[str, str] = {"Accept": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw: dict = json.loads(resp.read().decode())
        except OSError as e:
            log.warning("HF Hub API unreachable for %s: %s", dataset_id, e)
            return DatasetRecord(
                dataset_id=dataset_id,
                source_type="huggingface",
                uri=f"https://huggingface.co/datasets/{dataset_id}",
            )

        sha = raw.get("sha", raw.get("id", ""))
        sha256 = hashlib.sha256(sha.encode()).hexdigest() if sha else ""
        tags: list[str] = raw.get("tags", [])
        pii_tags = {"personal-data", "pii", "contains-pii", "private"}
        contains_pii = bool(pii_tags.intersection(t.lower() for t in tags))

        # Find license in cardData
        license_expr = "NOASSERTION"
        card_data = raw.get("cardData") or {}
        if isinstance(card_data, dict):
            lic = card_data.get("license")
            if lic:
                license_expr = lic if isinstance(lic, str) else str(lic)

        return DatasetRecord(
            dataset_id=dataset_id,
            source_type="huggingface",
            sha256=sha256,
            uri=f"https://huggingface.co/datasets/{dataset_id}",
            version=sha[:16] if sha else "",
            license=license_expr,
            contains_pii=contains_pii,
            raw_metadata=raw,
        )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key`` → ``(bucket, key)``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"Not an S3 URI: {uri!r}")
    path = uri[5:]
    bucket, _, key = path.partition("/")
    return bucket, key
