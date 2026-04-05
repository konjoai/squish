"""squish/squash/sbom_builder.py — CycloneDX 1.7 ML-BOM sidecar writer.

Builds a valid CycloneDX 1.7 JSON document from a completed ``squish compress``
run and writes it as ``cyclonedx-mlbom.json`` alongside the compressed model.

The document uses the ``machine-learning-model`` component type introduced in
CycloneDX 1.5 and is valid against the CycloneDX 1.7 JSON schema.

Integration path
----------------
    Called as a post-hook from ``squish/cli.py::_cmd_compress_inner()``.
    Populated further by ``squish/squash/eval_binder.py`` (Phase 2) and
    verified at boot time by ``squish/squash/governor.py`` (Phase 3).

No heavy external dependencies at import time — ``cyclonedx-python-lib`` is
only imported when actually building a BOM (inside ``_build_bom``), so missing
the optional extra does not break the rest of squish.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Weight-file extensions to hash.  Covers npy-dir (tensors/*.npy),
# native MLX safetensors dirs, and GGUF single-file downloads.
_WEIGHT_EXTENSIONS: frozenset[str] = frozenset(
    {".npy", ".safetensors", ".gguf", ".npz"}
)

_CDEX_SPEC_VERSION = "1.7"
_CDEX_SCHEMA_URL = "http://cyclonedx.org/schema/bom-1.7.schema.json"


@dataclass
class CompressRunMeta:
    """Bag-of-facts the CLI passes to :class:`CycloneDXBuilder` after compression.

    All fields come from data already available at the end of
    ``_cmd_compress_inner``; nothing requires a second model load.

    Parameters
    ----------
    model_id:
        Human-readable model identifier (CLI argument), e.g. ``"qwen2.5:1.5b"``.
    hf_mlx_repo:
        HuggingFace repository path, e.g.
        ``"mlx-community/Qwen2.5-1.5B-Instruct-bf16"``.
    model_family:
        Architecture family string from ``detect_model_family()``, e.g.
        ``"qwen2"``, ``"llama"``, ``"gemma3"``, or ``None`` if unknown.
    quant_format:
        Quantization level label written into the BOM, e.g. ``"INT4"`` or
        ``"INT3"``.
    awq_alpha:
        AWQ smooth-quant alpha used during calibration, or ``None`` when AWQ
        was not run.
    awq_group_size:
        AWQ group size (scales granularity), or ``None`` when AWQ was not run.
    output_dir:
        Directory that received the compressed model weights.  The sidecar
        ``cyclonedx-mlbom.json`` is written here.
    """

    model_id: str
    hf_mlx_repo: str
    model_family: str | None
    quant_format: str
    awq_alpha: float | None
    awq_group_size: int | None
    output_dir: Path


class CycloneDXBuilder:
    """Build and write a CycloneDX 1.7 ML-BOM sidecar for a squish compress run.

    The document is serialised as ``cyclonedx-mlbom.json`` in
    ``meta.output_dir``.  A new ``serialNumber`` UUID is generated on every
    call, so repeated calls to :meth:`from_compress_run` are safe (idempotent
    in effect, non-idempotent in UUID).

    Phase 2 (``eval_binder.py``) can later append
    ``quantitativeAnalysis.performanceMetrics`` entries to the written file.
    Phase 3 (``governor.py``) reads the sidecar and verifies weight hashes at
    server boot time.
    """

    @staticmethod
    def from_compress_run(meta: CompressRunMeta) -> Path:
        """Build and write the BOM sidecar, returning its path.

        Parameters
        ----------
        meta:
            Metadata captured at the end of the compress pipeline.

        Returns
        -------
        Path
            Absolute path to ``cyclonedx-mlbom.json`` inside
            ``meta.output_dir``.

        Raises
        ------
        PermissionError / OSError
            If the sidecar cannot be written.  Callers should catch this and
            emit a non-fatal warning rather than aborting the compress run.
        """
        bom = CycloneDXBuilder._build_bom(meta)
        sidecar = meta.output_dir / "cyclonedx-mlbom.json"
        sidecar.write_text(json.dumps(bom, indent=2))
        return sidecar

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_weight_files(output_dir: Path) -> list[dict[str, str]]:
        """Return SHA-256 hashes for every weight file in ``output_dir``.

        Returns a list of dicts ``{path: str (relative), content: str (hex)}``,
        sorted by path for deterministic ordering.
        """
        results: list[dict[str, str]] = []
        for p in sorted(output_dir.rglob("*")):
            if p.is_file() and p.suffix in _WEIGHT_EXTENSIONS:
                sha256 = hashlib.sha256()
                with p.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        sha256.update(chunk)
                results.append(
                    {
                        "path": str(p.relative_to(output_dir)),
                        "content": sha256.hexdigest(),
                    }
                )
        return results

    @staticmethod
    def _composite_hash(file_hashes: list[dict[str, str]]) -> str:
        """Compute a single SHA-256 from the sorted per-file digests.

        This gives one canonical content hash for the whole model artefact
        suitable for the CycloneDX ``component.hashes`` array, while the
        individual per-file hashes live in ``properties`` for file-level
        tamper detection by the governor.
        """
        concatenated = "".join(h["content"] for h in file_hashes)
        return hashlib.sha256(concatenated.encode()).hexdigest()

    @staticmethod
    def _build_bom(meta: CompressRunMeta) -> dict[str, Any]:
        """Return the full CycloneDX 1.7 BOM as a plain Python dict."""
        import squish  # local import — squish is always installed

        try:
            import mlx_lm  # optional at import time; core dep in practice

            mlx_lm_version: str = getattr(mlx_lm, "__version__", "unknown")
        except ImportError:
            mlx_lm_version = "unknown"

        file_hashes = CycloneDXBuilder._hash_weight_files(meta.output_dir)
        composite = CycloneDXBuilder._composite_hash(file_hashes) if file_hashes else ""

        # ── PURL (pkg:huggingface/<namespace>/<name>) ──────────────────────
        purl = f"pkg:huggingface/{meta.hf_mlx_repo}"

        # ── Formulation properties — calibration parameters ──────────────
        properties: list[dict[str, str]] = [
            {"name": "squish:quant_format", "value": meta.quant_format},
        ]
        if meta.awq_alpha is not None:
            properties.append(
                {"name": "squish:awq_alpha", "value": str(meta.awq_alpha)}
            )
        if meta.awq_group_size is not None:
            properties.append(
                {"name": "squish:awq_group_size", "value": str(meta.awq_group_size)}
            )
        # Per-file hashes in properties — used by governor.py for file-level
        # tamper detection.  Prefixed with "squish:weight_hash:" so they are
        # easy to filter without colliding with other property namespaces.
        properties.extend(
            {"name": f"squish:weight_hash:{h['path']}", "value": h["content"]}
            for h in file_hashes
        )

        bom: dict[str, Any] = {
            "bomFormat": "CycloneDX",
            "specVersion": _CDEX_SPEC_VERSION,
            "$schema": _CDEX_SCHEMA_URL,
            "version": 1,
            "serialNumber": f"urn:uuid:{uuid.uuid4()}",
            "metadata": {
                "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "tools": [
                    {
                        "vendor": "squishai",
                        "name": "squish",
                        "version": squish.__version__,
                    },
                    {
                        "vendor": "mlx-community",
                        "name": "mlx-lm",
                        "version": mlx_lm_version,
                    },
                ],
            },
            "components": [
                {
                    "type": "machine-learning-model",
                    "bom-ref": f"model/{meta.model_id}",
                    "name": meta.model_id,
                    "purl": purl,
                    # Composite SHA-256 over all weight-file digests.
                    # Empty list when no weight files are present (e.g. tests).
                    "hashes": (
                        [{"alg": "SHA-256", "content": composite}]
                        if composite
                        else []
                    ),
                    "externalReferences": [
                        {
                            "type": "distribution",
                            "url": f"https://huggingface.co/{meta.hf_mlx_repo}",
                        }
                    ],
                    "pedigree": {
                        "ancestors": [
                            {
                                "type": "machine-learning-model",
                                "name": meta.hf_mlx_repo,
                                "externalReferences": [
                                    {
                                        "type": "vcs",
                                        "url": f"https://huggingface.co/{meta.hf_mlx_repo}",
                                    }
                                ],
                            }
                        ]
                    },
                    "modelCard": {
                        "modelParameters": {
                            "task": "text-generation",
                            "architectureFamily": meta.model_family or "unknown",
                            "quantizationLevel": meta.quant_format,
                        },
                        # Placeholder populated by eval_binder.py (Phase 2).
                        # The empty list is valid CycloneDX 1.7 — it signals
                        # the field is intentionally present, just not yet scored.
                        "quantitativeAnalysis": {
                            "performanceMetrics": [],
                        },
                    },
                    "properties": properties,
                }
            ],
        }
        return bom


@dataclass
class SbomDiff:
    """Difference between two CycloneDX SBOM snapshots.

    Produced by :meth:`compare`.  All lists contain human-readable summary
    strings suitable for display or logging.
    """

    hash_changed: bool
    score_delta: float | None
    policy_status_changed: bool
    new_findings: list[str] = field(default_factory=list)
    resolved_findings: list[str] = field(default_factory=list)
    metadata_changes: dict[str, tuple[Any, Any]] = field(default_factory=dict)

    @property
    def has_regressions(self) -> bool:
        """True when any findings were introduced or policy status worsened."""
        return bool(self.new_findings) or (
            self.policy_status_changed and self.score_delta is not None and self.score_delta < 0
        )

    @staticmethod
    def compare(bom_a: dict[str, Any], bom_b: dict[str, Any]) -> "SbomDiff":
        """Compare two CycloneDX BOM dicts and return a :class:`SbomDiff`.

        *bom_a* is the older (baseline) snapshot; *bom_b* is the newer one.
        Both must be dicts as produced by :meth:`CycloneDXBuilder.build`.
        """

        def _component(bom: dict[str, Any]) -> dict[str, Any]:
            comps = bom.get("components", [])
            return comps[0] if comps else {}

        comp_a = _component(bom_a)
        comp_b = _component(bom_b)

        # ── hash changed? ──────────────────────────────────────────────────
        def _hash(comp: dict[str, Any]) -> str:
            for h in comp.get("hashes", []):
                if h.get("alg") in ("SHA-256", "SHA-512"):
                    return h.get("content", "")
            return ""

        hash_changed = _hash(comp_a) != _hash(comp_b)

        # ── performance score delta ────────────────────────────────────────
        def _score(comp: dict[str, Any]) -> float | None:
            qa = comp.get("modelCard", {}).get("quantitativeAnalysis", {})
            metrics = qa.get("performanceMetrics", [])
            for m in metrics:
                if m.get("type") in ("arc_easy", "accuracy"):
                    v = m.get("value")
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            return None

        score_a, score_b = _score(comp_a), _score(comp_b)
        score_delta: float | None = None
        if score_a is not None and score_b is not None:
            score_delta = round(score_b - score_a, 4)

        # ── vulnerabilities ────────────────────────────────────────────────
        def _vuln_ids(bom: dict[str, Any]) -> set[str]:
            return {v.get("id", "") for v in bom.get("vulnerabilities", [])}

        ids_a, ids_b = _vuln_ids(bom_a), _vuln_ids(bom_b)
        new_findings = sorted(ids_b - ids_a)
        resolved_findings = sorted(ids_a - ids_b)

        # ── policy status changed? ─────────────────────────────────────────
        def _policy_status(bom: dict[str, Any]) -> str:
            for prop in bom.get("metadata", {}).get("properties", []):
                if prop.get("name") == "squash:policy_result":
                    return prop.get("value", "")
            return ""

        ps_a, ps_b = _policy_status(bom_a), _policy_status(bom_b)
        policy_status_changed = ps_a != ps_b

        # ── other metadata changes ─────────────────────────────────────────
        meta_a = bom_a.get("metadata", {})
        meta_b = bom_b.get("metadata", {})
        metadata_changes: dict[str, tuple[Any, Any]] = {}
        for key in ("timestamp", "version"):
            va, vb = meta_a.get(key), meta_b.get(key)
            if va != vb:
                metadata_changes[key] = (va, vb)

        return SbomDiff(
            hash_changed=hash_changed,
            score_delta=score_delta,
            policy_status_changed=policy_status_changed,
            new_findings=new_findings,
            resolved_findings=resolved_findings,
            metadata_changes=metadata_changes,
        )
