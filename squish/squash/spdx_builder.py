"""squish/squash/spdx_builder.py — SPDX 2.3 + SPDX 3.0 AI Profile output.

Generates SPDX 2.3 tag-value and JSON documents from the same
:class:`~squish.squash.sbom_builder.CompressRunMeta` data structure used by
:class:`~squish.squash.sbom_builder.CycloneDXBuilder`.

SPDX 3.0 "AI Profile" fields (draft, but gaining fast enterprise adoption)
are included when present — they map directly onto model card metadata:
- ``ai:typeOfModel``
- ``ai:modelDataPreprocessing``
- ``ai:informationAboutTraining``
- ``ai:sensitivePersonalInformation``
- ``ai:safetyRiskAssessment``

Output files written alongside the CycloneDX sidecar:
- ``spdx-mlbom.json``    — SPDX 2.3 JSON (machine-readable, CI-friendly)
- ``spdx-mlbom.spdx``    — SPDX 2.3 tag-value (human-readable, audit-friendly)

References
----------
- SPDX 2.3 spec: https://spdx.github.io/spdx-spec/v2.3/
- SPDX 3.0 AI Profile: https://spdx.github.io/spdx-spec/v3.0/
"""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from squish.squash.sbom_builder import CompressRunMeta, CycloneDXBuilder

_SPDX_VERSION = "SPDX-2.3"
_DATA_LICENSE = "CC0-1.0"


@dataclass
class SpdxOptions:
    """Optional metadata to enrich the SPDX AI Profile fields.

    All fields default to suitable "not provided" placeholders so that
    omitting this object still produces a schema-valid SPDX document.
    """

    # SPDX 3.0 AI Profile fields ─────────────────────────────────────────
    type_of_model: str = "text-generation"
    """AI task type per SPDX AI Profile: text-generation, text-classification, …"""

    information_about_training: str = "see-model-card"
    """Free-text pointer to training procedure documentation."""

    sensitive_personal_information: str = "absent"
    """Whether the model was trained on SPI.  Values: absent | present | unknown."""

    safety_risk_assessment: str = "unspecified"
    """Safety risk tier per SPDX AI Profile.  Values: high | medium | low | unspecified."""

    # Training data provenance (Phase 1 optional) ──────────────────────
    dataset_ids: list[str] = field(default_factory=list)
    """HuggingFace dataset IDs or free-form URIs, e.g. ``["wikipedia", "c4"]``."""

    dataset_hashes: dict[str, str] = field(default_factory=dict)
    """Map of dataset identifier → SHA-256 fingerprint (optional)."""


class SpdxBuilder:
    """Build SPDX 2.3 JSON and tag-value documents for a squish compress run.

    Usage::

        spdx = SpdxBuilder.from_compress_run(meta)
        # Returns SpdxArtifacts(json_path=..., tagvalue_path=...)
    """

    @staticmethod
    def from_compress_run(
        meta: CompressRunMeta,
        options: SpdxOptions | None = None,
    ) -> "SpdxArtifacts":
        """Build and write both SPDX output files.

        Parameters
        ----------
        meta:
            Metadata from the compress run (same object used by CycloneDXBuilder).
        options:
            Optional AI Profile and provenance enrichment.  Pass ``None`` to
            use sensible defaults.

        Returns
        -------
        SpdxArtifacts
            Paths to the written JSON and tag-value files.
        """
        opts = options or SpdxOptions()
        doc = SpdxBuilder._build_doc(meta, opts)

        json_path = meta.output_dir / "spdx-mlbom.json"
        json_path.write_text(json.dumps(doc, indent=2))

        tv_path = meta.output_dir / "spdx-mlbom.spdx"
        tv_path.write_text(SpdxBuilder._to_tag_value(doc))

        return SpdxArtifacts(json_path=json_path, tagvalue_path=tv_path)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _purl(meta: CompressRunMeta) -> str:
        return f"pkg:huggingface/{meta.hf_mlx_repo}"

    @staticmethod
    def _doc_namespace(serial: str) -> str:
        return f"https://squish.konjo.ai/spdx/{serial}"

    @staticmethod
    def _build_doc(meta: CompressRunMeta, opts: SpdxOptions) -> dict[str, Any]:
        """Return the full SPDX 2.3 document as a plain Python dict."""
        import squish  # always installed

        serial = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        # Component hash — reuse CycloneDXBuilder logic to avoid duplication
        file_hashes = CycloneDXBuilder._hash_weight_files(meta.output_dir)
        composite = (
            CycloneDXBuilder._composite_hash(file_hashes) if file_hashes else ""
        )

        # Per-file checksums in SPDX snippet format
        file_spdx_entries = [
            {
                "SPDXID": f"SPDXRef-WeightFile-{i}",
                "fileName": h["path"],
                "checksums": [{"algorithm": "SHA256", "checksumValue": h["content"]}],
                "copyrightText": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "licenseInfoInFiles": ["NOASSERTION"],
            }
            for i, h in enumerate(file_hashes)
        ]

        # Relationships: document → DESCRIBES → model; model → CONTAINS → each file
        relationships = [
            {
                "spdxElementId": "SPDXRef-DOCUMENT",
                "relationshipType": "DESCRIBES",
                "relatedSpdxElement": "SPDXRef-Model",
            }
        ] + [
            {
                "spdxElementId": "SPDXRef-Model",
                "relationshipType": "CONTAINS",
                "relatedSpdxElement": f"SPDXRef-WeightFile-{i}",
            }
            for i in range(len(file_hashes))
        ]

        # Training dataset relationships
        dataset_entries = []
        for j, ds_id in enumerate(opts.dataset_ids):
            ds_ref = f"SPDXRef-TrainingData-{j}"
            ds_hash = opts.dataset_hashes.get(ds_id, "")
            entry: dict[str, Any] = {
                "SPDXID": ds_ref,
                "name": ds_id,
                "externalRefs": [
                    {
                        "referenceCategory": "PACKAGE-MANAGER",
                        "referenceType": "purl",
                        "referenceLocator": f"pkg:huggingface/{ds_id}",
                    }
                ],
                "copyrightText": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "versionInfo": "NOASSERTION",
                "downloadLocation": f"https://huggingface.co/datasets/{ds_id}",
            }
            if ds_hash:
                entry["checksums"] = [
                    {"algorithm": "SHA256", "checksumValue": ds_hash}
                ]
            dataset_entries.append(entry)
            relationships.append(
                {
                    "spdxElementId": "SPDXRef-Model",
                    "relationshipType": "GENERATED_FROM",
                    "relatedSpdxElement": ds_ref,
                }
            )

        model_package: dict[str, Any] = {
            "SPDXID": "SPDXRef-Model",
            "name": meta.model_id,
            "versionInfo": meta.quant_format,
            "packageVersion": meta.quant_format,
            "downloadLocation": f"https://huggingface.co/{meta.hf_mlx_repo}",
            "filesAnalyzed": True,
            "externalRefs": [
                {
                    "referenceCategory": "PACKAGE-MANAGER",
                    "referenceType": "purl",
                    "referenceLocator": SpdxBuilder._purl(meta),
                }
            ],
            "checksums": (
                [{"algorithm": "SHA256", "checksumValue": composite}]
                if composite
                else []
            ),
            "copyrightText": "NOASSERTION",
            "licenseConcluded": "NOASSERTION",
            # SPDX 3.0 AI Profile annotations embedded as SPDX 2.3 annotations
            "annotations": [
                _annotation(
                    "SPDXRef-Model",
                    "ai:typeOfModel",
                    opts.type_of_model,
                    now,
                    f"squish/{squish.__version__}",
                ),
                _annotation(
                    "SPDXRef-Model",
                    "ai:informationAboutTraining",
                    opts.information_about_training,
                    now,
                    f"squish/{squish.__version__}",
                ),
                _annotation(
                    "SPDXRef-Model",
                    "ai:sensitivePersonalInformation",
                    opts.sensitive_personal_information,
                    now,
                    f"squish/{squish.__version__}",
                ),
                _annotation(
                    "SPDXRef-Model",
                    "ai:safetyRiskAssessment",
                    opts.safety_risk_assessment,
                    now,
                    f"squish/{squish.__version__}",
                ),
                _annotation(
                    "SPDXRef-Model",
                    "squish:quantizationLevel",
                    meta.quant_format,
                    now,
                    f"squish/{squish.__version__}",
                ),
                _annotation(
                    "SPDXRef-Model",
                    "squish:architectureFamily",
                    meta.model_family or "unknown",
                    now,
                    f"squish/{squish.__version__}",
                ),
            ],
        }

        if meta.awq_alpha is not None:
            model_package["annotations"].append(
                _annotation(
                    "SPDXRef-Model",
                    "squish:awqAlpha",
                    str(meta.awq_alpha),
                    now,
                    f"squish/{squish.__version__}",
                )
            )

        all_packages = [model_package] + dataset_entries
        all_files = file_spdx_entries

        doc: dict[str, Any] = {
            "SPDXID": "SPDXRef-DOCUMENT",
            "spdxVersion": _SPDX_VERSION,
            "dataLicense": _DATA_LICENSE,
            "name": f"squish-{meta.model_id}-{meta.quant_format}",
            "documentNamespace": SpdxBuilder._doc_namespace(serial),
            "creationInfo": {
                "created": now,
                "creators": [
                    f"Tool: squish-{squish.__version__}",
                    "Tool: squish-squash",
                ],
                "licenseListVersion": "3.22",
            },
            "packages": all_packages,
            "files": all_files,
            "relationships": relationships,
        }
        return doc

    @staticmethod
    def _to_tag_value(doc: dict[str, Any]) -> str:
        """Convert the SPDX JSON doc to SPDX 2.3 tag-value text format."""
        lines: list[str] = []
        ci = doc.get("creationInfo", {})

        lines += [
            f"SPDXVersion: {doc['spdxVersion']}",
            f"DataLicense: {doc['dataLicense']}",
            f"SPDXID: {doc['SPDXID']}",
            f"DocumentName: {doc['name']}",
            f"DocumentNamespace: {doc['documentNamespace']}",
        ]
        for creator in ci.get("creators", []):
            lines.append(f"Creator: {creator}")
        lines.append(f"Created: {ci.get('created', '')}")
        lines.append("")

        for pkg in doc.get("packages", []):
            lines += [
                f"PackageName: {pkg['name']}",
                f"SPDXID: {pkg['SPDXID']}",
                f"PackageVersion: {pkg.get('versionInfo', 'NOASSERTION')}",
                f"PackageDownloadLocation: {pkg.get('downloadLocation', 'NOASSERTION')}",
                f"FilesAnalyzed: {str(pkg.get('filesAnalyzed', False)).lower()}",
                f"PackageLicenseConcluded: {pkg.get('licenseConcluded', 'NOASSERTION')}",
                f"PackageCopyrightText: {pkg.get('copyrightText', 'NOASSERTION')}",
            ]
            for cs in pkg.get("checksums", []):
                lines.append(f"PackageChecksum: {cs['algorithm']}: {cs['checksumValue']}")
            for ref in pkg.get("externalRefs", []):
                lines.append(
                    f"ExternalRef: {ref['referenceCategory']} "
                    f"{ref['referenceType']} {ref['referenceLocator']}"
                )
            for ann in pkg.get("annotations", []):
                lines.append(
                    f"Annotator: {ann.get('annotator', 'Tool: squish-squash')}"
                )
                lines.append(f"AnnotationDate: {ann.get('annotationDate', '')}")
                lines.append(f"AnnotationType: OTHER")
                lines.append(f"AnnotationComment: {ann.get('comment', '')}")
            lines.append("")

        for f in doc.get("files", []):
            lines += [
                f"FileName: {f['fileName']}",
                f"SPDXID: {f['SPDXID']}",
                f"LicenseConcluded: {f.get('licenseConcluded', 'NOASSERTION')}",
                f"CopyrightText: {f.get('copyrightText', 'NOASSERTION')}",
            ]
            for cs in f.get("checksums", []):
                lines.append(f"FileChecksum: {cs['algorithm']}: {cs['checksumValue']}")
            lines.append("")

        for rel in doc.get("relationships", []):
            lines.append(
                f"Relationship: {rel['spdxElementId']} "
                f"{rel['relationshipType']} {rel['relatedSpdxElement']}"
            )
        lines.append("")

        return "\n".join(lines)


@dataclass
class SpdxArtifacts:
    """Paths to the SPDX output files written by :class:`SpdxBuilder`."""

    json_path: Path
    tagvalue_path: Path


def _annotation(
    element: str,
    key: str,
    value: str,
    date: str,
    tool: str,
) -> dict[str, str]:
    return {
        "annotationType": "OTHER",
        "annotator": f"Tool: {tool}",
        "annotationDate": date,
        "annotatedElement": element,
        "comment": f"{key}: {value}",
    }
