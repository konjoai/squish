"""squish/squash/api.py — FastAPI REST microservice for Squash attestation.

Exposes the full Squash engine over HTTP with five core endpoints:

    POST /attest           — full attestation pipeline
    POST /scan             — security scan only
    POST /policy/evaluate  — evaluate policy against a submitted SBOM
    POST /vex/evaluate     — VEX feed evaluation against a model inventory
    GET  /policies         — list available policy templates

The server is zero-config: it discovers the squish model store automatically.
Intended to run embedded alongside the squish inference server or as a
standalone compliance sidecar.

Start from CLI::

    uvicorn squish.squash.api:app --host 0.0.0.0 --port 4444

Or programmatically::

    from squish.squash.api import app
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=4444)

All endpoints accept and return ``application/json``.  Heavy operations
(attestation, signing) run inside a thread-pool executor so the async
event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError as _e:
    raise ImportError(
        "FastAPI is required for squash.api. "
        "Install with: pip install 'squish[squash-api]'"
    ) from _e

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult
from squish.squash.policy import AVAILABLE_POLICIES, PolicyEngine
from squish.squash.scanner import ModelScanner

log = logging.getLogger(__name__)

# Thread-pool for blocking attestation work
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("SQUASH_WORKERS", "4")))

app = FastAPI(
    title="Squash — AI-SBOM Attestation API",
    description=(
        "Compliance attestation for AI/ML models. "
        "Generates CycloneDX + SPDX SBOMs, evaluates policy templates, "
        "performs security scanning, and handles Sigstore signing."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────


class AttestRequest(BaseModel):
    model_path: str = Field(description="Absolute path to model dir or file")
    output_dir: str | None = Field(
        default=None,
        description="Destination for artifacts; defaults to model_path dir",
    )
    model_id: str = Field(default="", description="Human-readable model name")
    hf_repo: str = Field(default="", description="HuggingFace repo ID")
    model_family: str | None = Field(default=None, description="Architecture family")
    quant_format: str = Field(default="unknown", description="Quantization format")
    policies: list[str] = Field(default_factory=lambda: ["enterprise-strict"])
    sign: bool = Field(default=False)
    fail_on_violation: bool = Field(default=False)
    skip_scan: bool = Field(default=False)
    training_dataset_ids: list[str] = Field(default_factory=list)
    vex_feed_path: str | None = Field(default=None)
    vex_feed_url: str = Field(default="")


class ScanRequest(BaseModel):
    model_path: str = Field(description="Absolute path to model dir or file")


class PolicyEvaluateRequest(BaseModel):
    sbom: dict[str, Any] = Field(description="CycloneDX BOM as JSON object")
    policy: str = Field(description="Policy name to evaluate")


class VexEvaluateRequest(BaseModel):
    sbom_path: str = Field(description="Absolute path to CycloneDX BOM JSON")
    vex_feed_path: str | None = Field(
        default=None, description="Path to local VEX feed directory"
    )
    vex_feed_url: str = Field(default="", description="HTTPS URL to remote VEX feed")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/policies")
async def list_policies() -> dict[str, list[str]]:
    """Return the names of all built-in policy templates."""
    return {"policies": sorted(AVAILABLE_POLICIES)}


@app.post("/attest")
async def attest(req: AttestRequest) -> JSONResponse:
    """Run the full attestation pipeline for a model artifact.

    This is the primary endpoint for CI/CD integrations.  The operation is
    CPU-bound (file hashing, scanning) so it runs in a thread-pool executor.

    Returns the squash-attest.json master record plus paths to all generated
    artifacts.
    """
    _require_path(req.model_path)
    config = AttestConfig(
        model_path=Path(req.model_path),
        output_dir=Path(req.output_dir) if req.output_dir else None,
        model_id=req.model_id,
        hf_repo=req.hf_repo,
        model_family=req.model_family,
        quant_format=req.quant_format,
        policies=req.policies,
        sign=req.sign,
        fail_on_violation=False,  # never raise from HTTP handler; return 422 instead
        skip_scan=req.skip_scan,
        training_dataset_ids=list(req.training_dataset_ids),
        vex_feed_path=Path(req.vex_feed_path) if req.vex_feed_path else None,
        vex_feed_url=req.vex_feed_url,
    )

    loop = asyncio.get_running_loop()
    result: AttestResult = await loop.run_in_executor(
        _executor, AttestPipeline.run, config
    )

    master_data = _result_to_dict(result)

    if req.fail_on_violation and not result.passed:
        return JSONResponse(status_code=422, content=master_data)

    return JSONResponse(content=master_data)


@app.post("/scan")
async def scan(req: ScanRequest) -> JSONResponse:
    """Run the security scanner on a model artifact without full attestation.

    Checks for pickle exploits, GGUF metadata injection, and optionally
    delegates to ProtectAI ModelScan if installed.
    """
    _require_path(req.model_path)
    model_path = Path(req.model_path)
    scan_dir = model_path if model_path.is_dir() else model_path.parent

    loop = asyncio.get_running_loop()
    scan_result = await loop.run_in_executor(
        _executor, ModelScanner.scan_directory, scan_dir
    )

    return JSONResponse(
        content={
            "status": scan_result.status,
            "is_safe": scan_result.is_safe,
            "critical": scan_result.critical_count,
            "high": scan_result.high_count,
            "findings": [
                {
                    "severity": f.severity,
                    "id": f.finding_id,
                    "title": f.title,
                    "detail": f.detail,
                    "file": f.file_path,
                    "cve": f.cve,
                }
                for f in scan_result.findings
            ],
        }
    )


@app.post("/policy/evaluate")
async def evaluate_policy(req: PolicyEvaluateRequest) -> JSONResponse:
    """Evaluate a submitted CycloneDX BOM against a named policy template.

    The SBOM is posted as a JSON body; no files are read from disk.
    """
    if req.policy not in AVAILABLE_POLICIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown policy '{req.policy}'. Available: {sorted(AVAILABLE_POLICIES)}",
        )

    policy_result = PolicyEngine.evaluate(req.sbom, req.policy)

    status_code = 200 if policy_result.passed else 422
    return JSONResponse(
        status_code=status_code,
        content={
            "policy": req.policy,
            "passed": policy_result.passed,
            "error_count": policy_result.error_count,
            "warning_count": policy_result.warning_count,
            "pass_count": policy_result.pass_count,
            "findings": [
                {
                    "rule_id": f.rule_id,
                    "severity": f.severity,
                    "passed": f.passed,
                    "field": f.field,
                    "rationale": f.rationale,
                    "remediation": f.remediation,
                    "actual_value": str(f.actual_value) if f.actual_value else None,
                }
                for f in policy_result.findings
            ],
        },
    )


@app.post("/vex/evaluate")
async def evaluate_vex(req: VexEvaluateRequest) -> JSONResponse:
    """Evaluate a VEX feed against a single model's SBOM.

    Loads the CycloneDX BOM from disk, constructs a single-model inventory,
    and runs VexEvaluator.  Returns a VEX report with affected CVEs.
    """
    try:
        from squish.squash.vex import (
            VexFeed,
            VexEvaluator,
            ModelInventory,
            ModelInventoryEntry,
        )
    except ImportError:
        raise HTTPException(500, "VEX engine not available (import error)")

    _require_path(req.sbom_path)
    bom_path = Path(req.sbom_path)

    try:
        bom: dict = json.loads(bom_path.read_text())
    except OSError as e:
        raise HTTPException(400, f"Cannot read SBOM: {e}") from e

    if not req.vex_feed_path and not req.vex_feed_url:
        raise HTTPException(400, "Provide either vex_feed_path or vex_feed_url")

    loop = asyncio.get_running_loop()

    def _run_vex() -> dict:
        feed = (
            VexFeed.from_directory(Path(req.vex_feed_path))
            if req.vex_feed_path
            else VexFeed.from_url(req.vex_feed_url)
        )
        purl = bom.get("components", [{}])[0].get("purl", "")
        hashes = bom.get("components", [{}])[0].get("hashes", [])
        sha256 = next(
            (h["content"] for h in hashes if h.get("alg") == "SHA-256"), ""
        )
        inv = ModelInventory(
            entries=[
                ModelInventoryEntry(
                    model_id=bom_path.stem,
                    purl=purl,
                    sbom_path=bom_path,
                    composite_sha256=sha256,
                )
            ]
        )
        report = VexEvaluator.evaluate(feed, inv)
        return report.to_dict()

    report_dict = await loop.run_in_executor(_executor, _run_vex)
    return JSONResponse(content=report_dict)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def _require_path(p: str) -> None:
    if not Path(p).exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {p}")


def _result_to_dict(r: AttestResult) -> dict[str, Any]:
    return {
        "model_id": r.model_id,
        "passed": r.passed,
        "output_dir": str(r.output_dir),
        "scan_status": r.scan_result.status if r.scan_result else "skipped",
        "policy_results": {
            name: {
                "passed": pr.passed,
                "error_count": pr.error_count,
                "warning_count": pr.warning_count,
                "pass_count": pr.pass_count,
            }
            for name, pr in r.policy_results.items()
        },
        "artifacts": {
            "cyclonedx": str(r.cyclonedx_path) if r.cyclonedx_path else None,
            "spdx_json": str(r.spdx_json_path) if r.spdx_json_path else None,
            "spdx_tv": str(r.spdx_tv_path) if r.spdx_tv_path else None,
            "signature": str(r.signature_path) if r.signature_path else None,
            "vex_report": str(r.vex_report_path) if r.vex_report_path else None,
            "master_record": str(r.master_record_path) if r.master_record_path else None,
        },
        "error": r.error,
    }
