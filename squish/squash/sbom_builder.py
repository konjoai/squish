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
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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


# ─────────────────────────────────────────────────────────────────────────────
# Wave 19 — EvalBinder (moved from eval_binder.py; shim retained for compat)
# ─────────────────────────────────────────────────────────────────────────────

class EvalBinder:
    """Mutate an existing CycloneDX ML-BOM sidecar with lm_eval task scores.

    All methods are static — the class is a namespace, not a stateful object.

    Previously lived in ``squish.squash.eval_binder``; that module is now a
    backward-compatible re-export shim.
    """

    @staticmethod
    def bind(
        bom_path: Path,
        lmeval_json_path: Path,
        baseline_path: Path | None = None,
    ) -> None:
        """Add or replace ``performanceMetrics`` entries in *bom_path*.

        Parameters
        ----------
        bom_path:
            Path to ``cyclonedx-mlbom.json`` written by Phase 1.
        lmeval_json_path:
            Path to a squish lmeval JSON result file.
            Expected schema::

                {
                  "scores": {"arc_easy": 70.6, ...},
                  "raw_results": {"arc_easy": {"acc_norm_stderr,none": 0.02, ...}}
                }

        baseline_path:
            Optional second lmeval JSON for a higher-precision reference.
            Adds ``deltaFromBaseline`` keys when present.

        Raises
        ------
        FileNotFoundError
            If *bom_path* or *lmeval_json_path* does not exist.
        json.JSONDecodeError
            If any of the JSON files is malformed.
        """
        bom: dict = json.loads(bom_path.read_text())
        lmeval: dict = json.loads(lmeval_json_path.read_text())

        baseline_scores: dict[str, float] | None = None
        if baseline_path is not None:
            baseline_scores = json.loads(baseline_path.read_text()).get("scores", {})

        scores: dict[str, float] = lmeval.get("scores", {})
        raw_results: dict = lmeval.get("raw_results", {})

        metrics: list[dict] = []
        for task, score in scores.items():
            entry: dict = {
                "type": "accuracy",
                "value": str(round(score, 1)),
                "slice": task,
            }
            raw_task: dict = raw_results.get(task, {})
            stderr_frac: float | None = raw_task.get("acc_norm_stderr,none")
            if stderr_frac is not None:
                half = round(1.96 * stderr_frac * 100, 1)
                entry["confidenceInterval"] = {
                    "lowerBound": str(round(score - half, 1)),
                    "upperBound": str(round(score + half, 1)),
                }
            if baseline_scores is not None and task in baseline_scores:
                delta = round(score - baseline_scores[task], 1)
                sign = "+" if delta >= 0 else ""
                entry["deltaFromBaseline"] = f"{sign}{delta}"
            metrics.append(entry)

        component: dict = bom["components"][0]
        component["modelCard"]["quantitativeAnalysis"]["performanceMetrics"] = metrics

        tmp = bom_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(bom, indent=2))
            tmp.rename(bom_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

        log.debug(
            "EvalBinder: wrote %d performanceMetrics entries to %s",
            len(metrics),
            bom_path,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Wave 19 — SbomRegistry: push CycloneDX BOMs to external registries
# ─────────────────────────────────────────────────────────────────────────────

class SbomRegistry:
    """Push CycloneDX BOM documents to SBOM registries via stdlib urllib.

    Three registry protocols are supported:

    * **Dependency-Track** (``dtrack``) — REST API v4, ``PUT /api/v1/bom``
    * **GUAC** (``guac``) — HTTP POST to the GUAC ingest endpoint
    * **Squash registry** (``squash``) — generic authenticated POST endpoint

    All methods are static — the class is a namespace, not a stateful object.
    """

    TIMEOUT_SECONDS: int = 30

    @staticmethod
    def push_dtrack(
        bom_path: Path,
        base_url: str,
        api_key: str,
        project_name: str = "squash",
    ) -> str:
        """Upload *bom_path* to a Dependency-Track instance.

        Parameters
        ----------
        bom_path:
            Path to ``cyclonedx-mlbom.json`` (or any CycloneDX JSON).
        base_url:
            Root URL of the DTrack instance, e.g. ``https://dtrack.example.com``.
        api_key:
            DTrack API key with BOM:UPLOAD permission.
        project_name:
            Optional project name override (default: ``"squash"``).

        Returns
        -------
        str
            The URL where the BOM is visible in Dependency-Track.

        Raises
        ------
        RuntimeError
            On HTTP errors or network failures.
        """
        import base64
        import urllib.error
        import urllib.request

        bom_data = bom_path.read_bytes()
        b64_bom = base64.b64encode(bom_data).decode("ascii")

        import json as _json
        payload = _json.dumps({
            "projectName": project_name,
            "autoCreate": True,
            "bom": b64_bom,
        }).encode("utf-8")

        url = base_url.rstrip("/") + "/api/v1/bom"
        req = urllib.request.Request(
            url,
            data=payload,
            method="PUT",
            headers={
                "X-Api-Key": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=SbomRegistry.TIMEOUT_SECONDS) as resp:  # noqa: S310
                if not (200 <= resp.status < 300):
                    raise RuntimeError(f"DTrack returned HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"DTrack HTTP error {e.code}: {e.reason}") from e

        return f"{base_url.rstrip('/')}/projects?name={project_name}"

    @staticmethod
    def push_guac(
        bom_path: Path,
        endpoint_url: str,
    ) -> str:
        """POST *bom_path* to a GUAC ingest HTTP endpoint.

        Parameters
        ----------
        bom_path:
            Path to ``cyclonedx-mlbom.json``.
        endpoint_url:
            GUAC ingest endpoint, e.g. ``http://guac.internal/api/v1/upload``.

        Returns
        -------
        str
            The endpoint URL (GUAC does not return a canonical BOM URL).

        Raises
        ------
        RuntimeError
            On HTTP errors or network failures.
        """
        import urllib.error
        import urllib.request

        bom_data = bom_path.read_bytes()
        req = urllib.request.Request(
            endpoint_url,
            data=bom_data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=SbomRegistry.TIMEOUT_SECONDS) as resp:  # noqa: S310
                if not (200 <= resp.status < 300):
                    raise RuntimeError(f"GUAC returned HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"GUAC HTTP error {e.code}: {e.reason}") from e

        return endpoint_url

    @staticmethod
    def push_squash(
        bom_path: Path,
        registry_url: str,
        token: str,
    ) -> str:
        """POST *bom_path* to a Squash-compatible BOM registry.

        Parameters
        ----------
        bom_path:
            Path to ``cyclonedx-mlbom.json``.
        registry_url:
            Squash registry endpoint.
        token:
            Bearer token for ``Authorization: Bearer <token>`` header.

        Returns
        -------
        str
            *registry_url* (echo-back; response body is not parsed).

        Raises
        ------
        RuntimeError
            On HTTP errors or network failures.
        """
        import urllib.error
        import urllib.request

        bom_data = bom_path.read_bytes()
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        req = urllib.request.Request(
            registry_url,
            data=bom_data,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=SbomRegistry.TIMEOUT_SECONDS) as resp:  # noqa: S310
                if not (200 <= resp.status < 300):
                    raise RuntimeError(f"Squash registry returned HTTP {resp.status}")
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Squash registry HTTP error {e.code}: {e.reason}") from e

        return registry_url


# ── Wave 22 — BOM Merge & Composition ─────────────────────────────────────────

class BomMerger:
    """Merge multiple CycloneDX 1.5 BOM files into one canonical BOM.

    * Components are deduplicated by ``purl`` (exact match).  When the same
      PURL appears in multiple BOMs the entry with the highest vulnerability
      severity is kept.
    * Vulnerabilities are unioned; entries with the same ``id`` are merged
      (worst-case ``analysis.state``).
    * A ``compositions`` array is appended describing which component sets were
      merged, per CycloneDX 1.5 spec.

    Example::

        merged_bom = BomMerger.merge(
            [Path("model-a/cyclonedx-mlbom.json"),
             Path("model-b/cyclonedx-mlbom.json")],
            output_path=Path("merged/cyclonedx-mlbom.json"),
        )
    """

    _SEVERITY_ORDER = {
        "critical": 5,
        "high": 4,
        "medium": 3,
        "moderate": 3,
        "low": 2,
        "info": 1,
        "informational": 1,
        "none": 0,
        "unknown": 0,
        "": 0,
    }

    @classmethod
    def merge(
        cls,
        bom_paths: list,
        output_path,
        metadata: dict | None = None,
    ) -> dict:
        """Merge *bom_paths* into a single CycloneDX 1.5 BOM.

        Parameters
        ----------
        bom_paths:
            Iterable of :class:`~pathlib.Path` (or path strings) pointing to
            CycloneDX BOM files.
        output_path:
            Destination path for the merged BOM.
        metadata:
            Optional dict to *update* the merged BOM's ``metadata`` section.

        Returns
        -------
        dict
            The merged BOM dict (also written to *output_path*).
        """
        import pathlib as _pl
        bom_paths = [_pl.Path(p) for p in bom_paths]
        output_path = _pl.Path(output_path)

        boms: list[dict] = []
        for p in bom_paths:
            try:
                boms.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                boms.append({})

        # Deduplicate components by PURL
        purl_map: dict[str, dict] = {}
        no_purl: list[dict] = []
        for bom in boms:
            for comp in bom.get("components", []):
                purl = comp.get("purl", "")
                if not purl:
                    no_purl.append(comp)
                    continue
                if purl not in purl_map:
                    purl_map[purl] = comp
                else:
                    # Keep higher-severity vulnerabilities on collision
                    existing = purl_map[purl]
                    if cls._max_severity(comp) > cls._max_severity(existing):
                        purl_map[purl] = comp

        merged_components = list(purl_map.values()) + no_purl

        # Union vulnerabilities by ID
        vuln_map: dict[str, dict] = {}
        for bom in boms:
            for vuln in bom.get("vulnerabilities", []):
                vid = vuln.get("id", "")
                if vid not in vuln_map:
                    vuln_map[vid] = vuln
                else:
                    # Merge: keep worst analysis state
                    existing_state = (
                        vuln_map[vid].get("analysis", {}).get("state", "") or ""
                    )
                    new_state = vuln.get("analysis", {}).get("state", "") or ""
                    if cls._state_severity(new_state) > cls._state_severity(
                        existing_state
                    ):
                        vuln_map[vid] = vuln

        # Build composition entries
        compositions: list[dict] = []
        for i, (bom, path) in enumerate(zip(boms, bom_paths)):
            comp_refs = [
                c.get("purl") or c.get("bom-ref") or f"comp-{j}"
                for j, c in enumerate(bom.get("components", []))
            ]
            if comp_refs:
                compositions.append({
                    "aggregate": "complete",
                    "assemblies": comp_refs[:20],  # cap for readability
                    "source": str(path),
                })

        # Assemble base metadata from first non-empty BOM
        base_metadata: dict = {}
        for bom in boms:
            if bom.get("metadata"):
                base_metadata = dict(bom["metadata"])
                break
        if metadata:
            base_metadata.update(metadata)

        # Use CycloneDX schema version from first BOM or default to 1.5
        schema_version = boms[0].get("specVersion", "1.5") if boms else "1.5"

        merged: dict = {
            "bomFormat": "CycloneDX",
            "specVersion": schema_version,
            "metadata": base_metadata,
            "components": merged_components,
            "vulnerabilities": list(vuln_map.values()),
            "compositions": compositions,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        return merged

    # ── Internal helpers ───────────────────────────────────────────────────────

    @classmethod
    def _max_severity(cls, comp: dict) -> int:
        """Return the maximum vulnerability severity score for a component."""
        score = 0
        for vuln in comp.get("vulnerabilities", []):
            ratings = vuln.get("ratings", [])
            for r in ratings:
                sev = str(r.get("severity", "")).lower()
                score = max(score, cls._SEVERITY_ORDER.get(sev, 0))
        return score

    @classmethod
    def _state_severity(cls, state: str) -> int:
        """Numeric severity for analysis state strings."""
        _map = {
            "exploitable": 5,
            "in_triage": 4,
            "not_affected": 1,
            "fixed": 0,
            "false_positive": 0,
            "": 0,
        }
        return _map.get(state.lower(), 2)
