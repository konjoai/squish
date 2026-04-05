"""squish/squash/attest.py — Unified attestation orchestrator.

:class:`AttestPipeline` is the single entry point that CI/CD integrations,
the REST API, the CLI, and platform SDKs all call.  It orchestrates:

1. Security scan (pickle, GGUF, optionally ProtectAI ModelScan)
2. CycloneDX 1.7 ML-BOM generation (dual SHA-256 + SHA-512 hashes)
3. SPDX 2.3 + AI Profile generation
4. Policy evaluation (one or more named policies)
5. Sigstore keyless signing (optional)
6. Training data provenance binding (optional)
7. VEX evaluation against a feed (optional)

Output directory structure written alongside the model::

    ./model-dir/
        cyclonedx-mlbom.json        — CycloneDX 1.7
        spdx-mlbom.json             — SPDX 2.3 JSON
        spdx-mlbom.spdx             — SPDX 2.3 tag-value
        squash-scan.json            — Security scan results
        squash-policy-<name>.json   — Per-policy evaluation result
        squash-attest.json          — Master attestation record
        squash-vex-report.json      — VEX evaluation result (if fed)
        cyclonedx-mlbom.json.sig.json — Sigstore bundle (if signed)

Usage::

    result = AttestPipeline.run(AttestConfig(
        model_path=Path("./llama-3.1-8b-q4.gguf"),
        output_dir=Path("./attestation"),
        policies=["eu-ai-act", "enterprise-strict"],
        sign=True,
        fail_on_violation=True,
    ))
    if not result.passed:
        sys.exit(1)
"""

from __future__ import annotations

import datetime
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from squish.squash.sbom_builder import CompressRunMeta, CycloneDXBuilder
from squish.squash.spdx_builder import SpdxBuilder, SpdxOptions
from squish.squash.policy import PolicyEngine, PolicyResult, AVAILABLE_POLICIES
from squish.squash.scanner import ModelScanner, ScanResult
from squish.squash.oms_signer import OmsSigner

log = logging.getLogger(__name__)


@dataclass
class AttestConfig:
    """Configuration for a single attestation run.

    Parameters
    ----------
    model_path:
        Path to the model directory or single model file (GGUF, safetensors, …).
        When a directory is passed, all weight files within it are hashed and
        scanned.
    output_dir:
        Directory where all attestation artifacts are written.  Created if it
        does not exist.  Defaults to *model_path* when *model_path* is a dir,
        or ``model_path.parent`` when it is a file.
    model_id:
        Human-readable model identifier, e.g. ``"llama-3.1-8b"``.  Defaults to
        the directory/file stem.
    hf_repo:
        HuggingFace repository ID for provenance, e.g.
        ``"meta-llama/Llama-3.1-8B-Instruct"``.  Use ``""`` when not from HF.
    model_family:
        Architecture family string (``"llama"``, ``"qwen2"``, …).  ``None`` →
        left as ``"unknown"``.
    quant_format:
        Quantization label, e.g. ``"INT4"`` or ``"BF16"``.
    policies:
        List of policy names to evaluate.  Accepts any key from
        :data:`~squish.squash.policy.AVAILABLE_POLICIES`.
        Empty list → skip policy evaluation.
    sign:
        Whether to sign the CycloneDX BOM via Sigstore keyless signing.
    fail_on_violation:
        When ``True``, :meth:`AttestPipeline.run` raises
        :class:`AttestationViolationError` if any policy evaluation has
        error-severity failures or the security scan is ``unsafe``.
    skip_scan:
        Skip the security scanner (not recommended for production).
    spdx_options:
        Optional SPDX AI Profile enrichment.
    vex_feed_path:
        Optional path to a local VEX feed directory.
    vex_feed_url:
        Optional HTTPS URL to a remote VEX feed.
    training_dataset_ids:
        Optional list of HuggingFace dataset IDs to embed as training data
        provenance.
    awq_alpha:
        AWQ smooth-quant alpha (for SBOM metadata completeness).
    awq_group_size:
        AWQ group size.
    """

    model_path: Path
    output_dir: Path | None = None
    model_id: str = ""
    hf_repo: str = ""
    model_family: str | None = None
    quant_format: str = "unknown"
    policies: list[str] = field(default_factory=lambda: ["enterprise-strict"])
    sign: bool = False
    fail_on_violation: bool = True
    skip_scan: bool = False
    spdx_options: SpdxOptions | None = None
    vex_feed_path: Path | None = None
    vex_feed_url: str = ""
    training_dataset_ids: list[str] = field(default_factory=list)
    awq_alpha: float | None = None
    awq_group_size: int | None = None


@dataclass
class AttestResult:
    """Result of a complete attestation pipeline run."""

    model_id: str
    output_dir: Path
    passed: bool  # True iff scan safe AND all error-policies passed
    scan_result: ScanResult | None = None
    policy_results: dict[str, PolicyResult] = field(default_factory=dict)
    cyclonedx_path: Path | None = None
    spdx_json_path: Path | None = None
    spdx_tv_path: Path | None = None
    signature_path: Path | None = None
    master_record_path: Path | None = None
    vex_report_path: Path | None = None
    error: str = ""

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        policy_summary = "; ".join(
            r.summary() for r in self.policy_results.values()
        )
        return f"[{status}] {self.model_id}: {policy_summary}"


class AttestationViolationError(Exception):
    """Raised when :attr:`AttestConfig.fail_on_violation` is True and a check fails."""


class AttestPipeline:
    """Orchestrate a full attestation run for a model artifact.

    All work is done in :meth:`run`.  The class itself is stateless — every
    call creates a fresh pipeline.
    """

    @staticmethod
    def run(config: AttestConfig) -> AttestResult:
        """Execute the attestation pipeline.

        Parameters
        ----------
        config:
            :class:`AttestConfig` describing what to attest and how.

        Returns
        -------
        AttestResult
            Aggregate result with paths to all written artifacts.

        Raises
        ------
        AttestationViolationError
            If ``fail_on_violation=True`` and any error-severity rule fails
            or the security scan status is ``"unsafe"``.
        """
        model_path = config.model_path.resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        # Resolve output directory
        if config.output_dir is not None:
            out_dir = config.output_dir.resolve()
        elif model_path.is_dir():
            out_dir = model_path
        else:
            out_dir = model_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resolve model_id
        model_id = config.model_id or (
            model_path.name if model_path.is_dir() else model_path.stem
        )

        # Determine weight directory for hashing
        weight_dir = model_path if model_path.is_dir() else model_path.parent

        result = AttestResult(
            model_id=model_id,
            output_dir=out_dir,
            passed=True,
        )

        # ── Step 1: Security scan ──────────────────────────────────────────
        scan_result: ScanResult | None = None
        if not config.skip_scan:
            log.info("Running security scan on %s …", weight_dir)
            scan_result = ModelScanner.scan_directory(weight_dir)
            result.scan_result = scan_result

            scan_out = out_dir / "squash-scan.json"
            _write_json(scan_out, _scan_to_dict(scan_result))
            log.info("  Scan result: %s", scan_result.summary())

            if scan_result.status == "unsafe":
                result.passed = False
                if config.fail_on_violation:
                    raise AttestationViolationError(
                        f"Security scan UNSAFE for {model_id}: "
                        f"{scan_result.critical_count} critical finding(s). "
                        f"See {scan_out}"
                    )

        # ── Step 2: CycloneDX BOM ─────────────────────────────────────────
        log.info("Building CycloneDX 1.7 ML-BOM …")
        meta = CompressRunMeta(
            model_id=model_id,
            hf_mlx_repo=config.hf_repo or f"unknown/{model_id}",
            model_family=config.model_family,
            quant_format=config.quant_format,
            awq_alpha=config.awq_alpha,
            awq_group_size=config.awq_group_size,
            output_dir=out_dir,
        )
        try:
            cdx_path = CycloneDXBuilder.from_compress_run(meta)
            result.cyclonedx_path = cdx_path
        except OSError as e:
            result.error = str(e)
            result.passed = False
            if config.fail_on_violation:
                raise
            log.error("CycloneDX BOM write failed: %s", e)

        # Annotate BOM with scan result
        if scan_result is not None and cdx_path is not None:
            _annotate_bom_with_scan(cdx_path, scan_result)

        # ── Step 3: SPDX output ───────────────────────────────────────────
        log.info("Building SPDX 2.3 + AI Profile …")
        try:
            spdx_opts = config.spdx_options or SpdxOptions()
            if config.training_dataset_ids:
                spdx_opts.dataset_ids = list(config.training_dataset_ids)
            spdx_artifacts = SpdxBuilder.from_compress_run(meta, spdx_opts)
            result.spdx_json_path = spdx_artifacts.json_path
            result.spdx_tv_path = spdx_artifacts.tagvalue_path
        except OSError as e:
            log.warning("SPDX write failed (non-fatal): %s", e)

        # ── Step 4: Training data provenance ──────────────────────────────
        if config.training_dataset_ids and result.cyclonedx_path:
            log.info(
                "Resolving training data provenance for %d dataset(s) …",
                len(config.training_dataset_ids),
            )
            _bind_training_provenance(result.cyclonedx_path, config.training_dataset_ids)

        # ── Step 5: Policy evaluation ──────────────────────────────────────
        if config.policies and result.cyclonedx_path:
            log.info("Evaluating policies: %s …", config.policies)
            sbom_dict = json.loads(result.cyclonedx_path.read_text())

            # Inject scan result as a top-level squash: key for policy checks
            if scan_result is not None:
                sbom_dict["squash:scan_result"] = scan_result.status

            for policy_name in config.policies:
                if policy_name not in AVAILABLE_POLICIES:
                    log.warning("Unknown policy '%s' — skipping", policy_name)
                    continue
                pr = PolicyEngine.evaluate(sbom_dict, policy_name)
                result.policy_results[policy_name] = pr
                log.info("  %s", pr.summary())

                policy_out = out_dir / f"squash-policy-{policy_name}.json"
                _write_json(policy_out, _policy_result_to_dict(pr))

                if not pr.passed:
                    result.passed = False
                    if config.fail_on_violation:
                        raise AttestationViolationError(
                            f"Policy {policy_name!r} FAILED for {model_id}: "
                            f"{pr.error_count} error(s). See {policy_out}"
                        )

        # ── Step 6: VEX evaluation ────────────────────────────────────────
        if (config.vex_feed_path or config.vex_feed_url) and result.cyclonedx_path:
            log.info("Evaluating VEX feed …")
            try:
                from squish.squash.vex import (
                    VexFeed,
                    VexEvaluator,
                    ModelInventory,
                    ModelInventoryEntry,
                )

                feed = (
                    VexFeed.from_directory(config.vex_feed_path)
                    if config.vex_feed_path
                    else VexFeed.from_url(config.vex_feed_url)
                )
                bom = json.loads(result.cyclonedx_path.read_text())
                purl = bom.get("components", [{}])[0].get("purl", "")
                hashes = bom.get("components", [{}])[0].get("hashes", [])
                sha256 = next(
                    (h["content"] for h in hashes if h.get("alg") == "SHA-256"), ""
                )
                inv = ModelInventory(
                    entries=[
                        ModelInventoryEntry(
                            model_id=model_id,
                            purl=purl,
                            sbom_path=result.cyclonedx_path,
                            composite_sha256=sha256,
                        )
                    ]
                )
                vex_report = VexEvaluator.evaluate(feed, inv)
                vex_out = out_dir / "squash-vex-report.json"
                _write_json(vex_out, vex_report.to_dict())
                result.vex_report_path = vex_out
                log.info("  VEX: %s", vex_report.summary())

                if not vex_report.is_clean:
                    result.passed = False
                    if config.fail_on_violation:
                        raise AttestationViolationError(
                            f"VEX evaluation found {len(vex_report.affected_models)} "
                            f"affected model(s) for CVEs. See {vex_out}"
                        )
            except ImportError:
                log.warning("VEX engine unavailable — skipping (import error)")

        # ── Step 7: Sigstore signing ──────────────────────────────────────
        if config.sign and result.cyclonedx_path:
            log.info("Signing CycloneDX BOM via Sigstore …")
            sig_path = OmsSigner.sign(result.cyclonedx_path)
            result.signature_path = sig_path
            if sig_path:
                log.info("  Signed → %s", sig_path)
            else:
                log.warning("  Signing skipped (sigstore not installed or no OIDC)")

        # ── Step 8: Master attestation record ─────────────────────────────
        master = _build_master_record(config, result)
        master_out = out_dir / "squash-attest.json"
        _write_json(master_out, master)
        result.master_record_path = master_out

        log.info(
            "Attestation complete for %s → %s [%s]",
            model_id,
            out_dir,
            "PASS" if result.passed else "FAIL",
        )
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Private helpers
# ──────────────────────────────────────────────────────────────────────────────


def _write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


def _scan_to_dict(r: ScanResult) -> dict[str, Any]:
    return {
        "scanned_path": r.scanned_path,
        "status": r.status,
        "scanner_version": r.scanner_version,
        "critical": r.critical_count,
        "high": r.high_count,
        "findings": [
            {
                "severity": f.severity,
                "id": f.finding_id,
                "title": f.title,
                "detail": f.detail,
                "file": f.file_path,
                "cve": f.cve,
            }
            for f in r.findings
        ],
    }


def _policy_result_to_dict(r: PolicyResult) -> dict[str, Any]:
    return {
        "policy": r.policy_name,
        "passed": r.passed,
        "error_count": r.error_count,
        "warning_count": r.warning_count,
        "pass_count": r.pass_count,
        "findings": [
            {
                "rule_id": f.rule_id,
                "severity": f.severity,
                "passed": f.passed,
                "field": f.field,
                "rationale": f.rationale,
                "remediation": f.remediation,
            }
            for f in r.findings
        ],
    }


def _annotate_bom_with_scan(bom_path: Path, scan: ScanResult) -> None:
    """Inject scan findings as CycloneDX vulnerabilities into an existing BOM."""
    try:
        bom: dict = json.loads(bom_path.read_text())
        vulns = scan.to_cdx_vulnerabilities()
        if vulns:
            bom["vulnerabilities"] = vulns
        bom["squash:scan_result"] = scan.status
        # Atomic write
        tmp = bom_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(bom, indent=2))
        tmp.replace(bom_path)
    except OSError as e:
        log.warning("Could not annotate BOM with scan result: %s", e)


def _bind_training_provenance(bom_path: Path, dataset_ids: list[str]) -> None:
    """Resolve HF dataset provenance and bind to BOM (best-effort)."""
    try:
        from squish.squash.provenance import ProvenanceCollector
        manifest = ProvenanceCollector.from_hf_datasets(dataset_ids)
        manifest.bind_to_sbom(bom_path)
    except Exception as e:  # broad catch — provenance is enrichment, not gating
        log.warning("Training data provenance binding failed (non-fatal): %s", e)


def _build_master_record(config: AttestConfig, result: AttestResult) -> dict[str, Any]:
    """Build the squash-attest.json master record."""
    import squish

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    policies_summary = {
        name: {
            "passed": pr.passed,
            "errors": pr.error_count,
            "warnings": pr.warning_count,
        }
        for name, pr in result.policy_results.items()
    }
    return {
        "squash_version": squish.__version__,
        "attested_at": now,
        "model_id": result.model_id,
        "model_path": str(config.model_path),
        "output_dir": str(result.output_dir),
        "passed": result.passed,
        "scan_status": result.scan_result.status if result.scan_result else "skipped",
        "policies_evaluated": list(config.policies),
        "policy_results": policies_summary,
        "artifacts": {
            "cyclonedx": str(result.cyclonedx_path) if result.cyclonedx_path else None,
            "spdx_json": str(result.spdx_json_path) if result.spdx_json_path else None,
            "spdx_tv": str(result.spdx_tv_path) if result.spdx_tv_path else None,
            "signature": str(result.signature_path) if result.signature_path else None,
            "vex_report": str(result.vex_report_path) if result.vex_report_path else None,
        },
        "platform": {
            "python": sys.version,
            "os": platform.platform(),
        },
    }
