"""squish/squash/api.py — FastAPI REST microservice for Squash attestation.

Exposes the full Squash engine over HTTP.  Core endpoints:

    POST /attest               — full attestation pipeline
    POST /scan                 — security scan only
    POST /policy/evaluate      — evaluate policy against a submitted SBOM
    POST /vex/evaluate         — VEX feed evaluation against a model inventory
    GET  /policies             — list available policy templates
    POST /vex/publish          — generate an OpenVEX 0.2.0 document
    GET  /vex/status           — current VEX cache metadata (url, age, statement count, stale flag)
    POST /vex/update           — force-refresh the local VEX feed cache from a remote URL
    POST /attest/mlflow        — offline attestation for MLflow artifacts
    POST /attest/wandb         — offline attestation for W&B artifacts
    POST /attest/huggingface   — attestation with optional HuggingFace Hub push
    POST /attest/langchain     — pre-deployment attestation for LangChain pipelines
    POST /attest/mcp           — MCP tool manifest supply-chain attestation

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
import base64
import hashlib
import hmac
import json
import logging
import os
import tempfile
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as _e:
    raise ImportError(
        "FastAPI is required for squash.api. "
        "Install with: pip install 'squish[squash-api]'"
    ) from _e

from squish.squash.attest import AttestConfig, AttestPipeline, AttestResult
from squish.squash.policy import AVAILABLE_POLICIES, PolicyEngine, PolicyRegistry
from squish.squash.scanner import ModelScanner

log = logging.getLogger(__name__)

# Thread-pool for blocking attestation work
_executor = ThreadPoolExecutor(max_workers=int(os.getenv("SQUASH_WORKERS", "4")))

# In-memory scan job store: job_id → {"status": "pending"|"done"|"error", "result": dict}
# Bounded to 1000 entries — oldest evicted first (LRU via OrderedDict).
_SCAN_JOB_LIMIT = int(os.getenv("SQUASH_SCAN_JOB_LIMIT", "1000"))
_scan_jobs: OrderedDict[str, dict[str, Any]] = OrderedDict()

# ── Auth ──────────────────────────────────────────────────────────────────────
# Set SQUASH_API_TOKEN in the environment to enable bearer token auth.
# Paths listed here bypass auth (health probes, OpenAPI schema, metrics).
_UNAUTHED_PATHS = frozenset({"/health", "/docs", "/redoc", "/openapi.json", "/metrics"})

# ── Rate limiter ──────────────────────────────────────────────────────────────
# SQUASH_RATE_LIMIT requests per 60-second sliding window per client IP.
_RATE_LIMIT = int(os.environ.get("SQUASH_RATE_LIMIT", "60"))
_rate_window: dict[str, deque] = defaultdict(deque)

# ── Prometheus-style counters ─────────────────────────────────────────────────
_COUNTERS: dict[str, int] = {
    "squash_attest_total": 0,
    "squash_scan_total": 0,
    "squash_policy_evaluate_total": 0,
    "squash_vex_evaluate_total": 0,
    "squash_vex_update_total": 0,
    "squash_vex_status_total": 0,
    "squash_sbom_diff_total": 0,
    "squash_policy_violations_total": 0,
}

# ── Cloud dashboard stores (W52-55) ───────────────────────────────────────────
# All stores are in-process / in-memory; a production deployment backs these
# with a database by replacing the helpers below.  The API contract is stable.

# Tenant registry: tenant_id → {name, plan, contact_email, created_at}
_TENANT_LIMIT = int(os.getenv("SQUASH_TENANT_LIMIT", "1000"))
_tenants: dict[str, dict[str, Any]] = {}

# Model inventory: tenant_id → list[record dict], total capped at _INVENTORY_LIMIT
_INVENTORY_PER_TENANT = int(os.getenv("SQUASH_INVENTORY_PER_TENANT", "500"))
_inventory: defaultdict[str, deque] = defaultdict(
    lambda: deque(maxlen=_INVENTORY_PER_TENANT)
)

# VEX alert feed: tenant_id → deque[alert dict]
_VEX_ALERTS_PER_TENANT = int(os.getenv("SQUASH_VEX_ALERTS_PER_TENANT", "500"))
_vex_alerts: defaultdict[str, deque] = defaultdict(
    lambda: deque(maxlen=_VEX_ALERTS_PER_TENANT)
)

# Drift event stream: tenant_id → deque[event dict]
_DRIFT_EVENTS_PER_TENANT = int(os.getenv("SQUASH_DRIFT_EVENTS_PER_TENANT", "500"))
_drift_events: defaultdict[str, deque] = defaultdict(
    lambda: deque(maxlen=_DRIFT_EVENTS_PER_TENANT)
)

# Policy dashboard aggregates: tenant_id → policy_name → {"passed": int, "failed": int}
_policy_stats: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
    lambda: defaultdict(lambda: {"passed": 0, "failed": 0})
)

# ── SQLite persistence (W57) ─────────────────────────────────────────────────
# Optional write-through to a SQLite file.  Set SQUASH_CLOUD_DB=/path/to/db to
# enable durability.  When absent (default) all stores remain in-memory only.
try:
    from squish.squash.cloud_db import CloudDB as _CloudDB, _make_db as _make_cloud_db
    _db: "_CloudDB | None" = _make_cloud_db()
except Exception:  # pragma: no cover
    _db = None  # type: ignore[assignment]


def _db_write_tenant(tenant_id: str, record: dict) -> None:
    """Upsert a tenant record into the in-memory dict and (if active) SQLite."""
    _tenants[tenant_id] = record
    if _db is not None:
        _db.upsert_tenant(tenant_id, record)


def _db_write_inventory(tenant_id: str, record: dict) -> None:
    """Append an inventory record to the deque and (if active) SQLite."""
    _inventory[tenant_id].append(record)
    if _db is not None:
        _db.append_record("inventory", tenant_id, record)


def _db_write_vex_alert(tenant_id: str, record: dict) -> None:
    """Append a VEX alert to the deque and (if active) SQLite."""
    _vex_alerts[tenant_id].append(record)
    if _db is not None:
        _db.append_record("vex_alerts", tenant_id, record)


def _db_write_drift_event(tenant_id: str, record: dict) -> None:
    """Append a drift event to the deque and (if active) SQLite."""
    _drift_events[tenant_id].append(record)
    if _db is not None:
        _db.append_record("drift_events", tenant_id, record)


def _db_inc_policy_stat(tenant_id: str, policy_name: str, *, passed: bool) -> None:
    """Increment in-memory policy bucket and (if active) SQLite counter."""
    bucket = _policy_stats[tenant_id][policy_name]
    if passed:
        bucket["passed"] += 1
    else:
        bucket["failed"] += 1
    if _db is not None:
        _db.inc_policy_stat(tenant_id, policy_name, passed=passed)


# ── W58: CloudDB read helpers ────────────────────────────────────────────────


def _db_read_inventory(tenant_id: str) -> list[dict[str, Any]]:
    """Read inventory from SQLite when active, else fall back to in-memory deque."""
    if _db is not None:
        return _db.read_inventory(tenant_id)
    return list(_inventory[tenant_id])


def _db_read_vex_alerts(tenant_id: str) -> list[dict[str, Any]]:
    """Read VEX alerts from SQLite when active, else fall back to in-memory deque."""
    if _db is not None:
        return _db.read_vex_alerts(tenant_id)
    return list(_vex_alerts[tenant_id])


def _db_read_policy_stats() -> dict[str, dict[str, int]]:
    """Cross-tenant policy aggregate — SQLite when active, else fold in-memory dict."""
    if _db is not None:
        return _db.read_policy_stats()
    result: dict[str, dict[str, int]] = {}
    for tenant_stats in _policy_stats.values():
        for policy_name, counts in tenant_stats.items():
            if policy_name not in result:
                result[policy_name] = {"passed": 0, "failed": 0}
            result[policy_name]["passed"] += counts.get("passed", 0)
            result[policy_name]["failed"] += counts.get("failed", 0)
    return result


# ── Cloud auth helpers (W52-55) ───────────────────────────────────────────────

def _verify_jwt_hs256(token: str, secret: str) -> dict[str, Any]:
    """Minimal stdlib HS256 JWT verifier — no external dependencies.

    Raises ``ValueError`` on invalid structure, bad signature, or expiry.
    """
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("invalid JWT structure")
    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(
        secret.encode(), signing_input, hashlib.sha256
    ).digest()
    pad = "=" * (4 - len(sig_b64) % 4)
    actual_sig = base64.urlsafe_b64decode(sig_b64 + pad)
    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("invalid JWT signature")
    pad2 = "=" * (4 - len(payload_b64) % 4)
    claims: dict[str, Any] = json.loads(
        base64.urlsafe_b64decode(payload_b64 + pad2)
    )
    if "exp" in claims and claims["exp"] < time.time():
        raise ValueError("JWT expired")
    return claims


def _resolve_tenant_id(request: Request) -> str:
    """Return the tenant_id for the current request.

    Resolution order:
    1. If ``SQUASH_JWT_SECRET`` is set: decode ``Authorization: Bearer <jwt>``
       and return the ``tenant_id`` claim (or ``sub`` as fallback).
    2. Otherwise: return the ``X-Tenant-ID`` header value (may be empty string
       for single-tenant deployments).
    """
    secret = os.environ.get("SQUASH_JWT_SECRET", "")
    if secret:
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="JWT required for cloud endpoints")
        try:
            claims = _verify_jwt_hs256(auth[7:], secret)
        except ValueError as exc:
            raise HTTPException(status_code=401, detail=f"Invalid JWT: {exc}") from exc
        tid = claims.get("tenant_id") or claims.get("sub", "")
        if not tid:
            raise HTTPException(status_code=401, detail="JWT missing tenant_id claim")
        return str(tid)
    return request.headers.get("X-Tenant-ID", "")


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
# Middleware — auth + rate limiter
# ──────────────────────────────────────────────────────────────────────────────


@app.middleware("http")
async def _security_middleware(request: Request, call_next):
    """Bearer token auth and per-IP sliding-window rate limiter."""
    path = request.url.path

    # ── Rate limit (applied before auth) ──────────────────────────────────────
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window = _rate_window[client_ip]
    cutoff = now - 60.0
    while window and window[0] < cutoff:
        window.popleft()
    if len(window) >= _RATE_LIMIT:
        retry_after = int(60 - (now - window[0])) + 1
        return JSONResponse(
            {"detail": "Rate limit exceeded"},
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )
    window.append(now)

    # ── Bearer token auth ─────────────────────────────────────────────────────
    token = os.environ.get("SQUASH_API_TOKEN", "")
    if token and path not in _UNAUTHED_PATHS:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    return await call_next(request)


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
    offline: bool = Field(
        default=False,
        description="Air-gapped mode — skip all OIDC/network calls (also set by SQUASH_OFFLINE=1)",
    )
    local_signing_key: str | None = Field(
        default=None,
        description="Absolute path to an Ed25519 .priv.pem for offline signing (requires sign=True, offline=True)",
    )
    fail_on_violation: bool = Field(default=False)
    skip_scan: bool = Field(default=False)
    training_dataset_ids: list[str] = Field(default_factory=list)
    vex_feed_path: str | None = Field(default=None)
    vex_feed_url: str = Field(default="")
    # ── SPDX AI Profile enrichment (mirrors squash attest --spdx-* CLI flags) ──
    spdx_type: str | None = Field(
        default=None,
        description=(
            "SPDX AI Profile type_of_model "
            "(e.g. text-generation, text-classification, summarization). Default: text-generation"
        ),
    )
    spdx_safety_risk: str | None = Field(
        default=None,
        description="SPDX AI Profile safetyRiskAssessment: high | medium | low | unspecified",
    )
    spdx_datasets: list[str] = Field(
        default_factory=list,
        description=(
            "HuggingFace dataset IDs or URIs for the SPDX AI Profile "
            "(merged with training_dataset_ids; e.g. [\"wikipedia\", \"c4\"])"
        ),
    )
    spdx_training_info: str | None = Field(
        default=None,
        description="SPDX AI Profile informationAboutTraining free-text. Default: see-model-card",
    )
    spdx_sensitive_data: str | None = Field(
        default=None,
        description="SPDX AI Profile sensitivePIIInTrainingData: absent | present | unknown",
    )
    tenant_id: str = Field(
        default="",
        description=(
            "Optional tenant identifier.  When set the attestation result is "
            "automatically registered in the cloud model inventory under this tenant."
        ),
    )


class ScanRequest(BaseModel):
    model_path: str = Field(description="Absolute path to model dir or file")


class PolicyEvaluateRequest(BaseModel):
    sbom: dict[str, Any] = Field(description="CycloneDX BOM as JSON object")
    policy: str = Field(description="Policy name to evaluate (ignored when custom_rules is set)")
    custom_rules: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Ad-hoc rule list. When provided, evaluated via PolicyEngine.evaluate_custom "
            "and the 'policy' field is used only as the result label."
        ),
    )


class VexUpdateRequest(BaseModel):
    url: str | None = Field(
        default=None,
        description="VEX feed URL. Falls back to $SQUASH_VEX_URL then VexCache.DEFAULT_URL.",
    )
    timeout: float = Field(default=30.0, description="HTTP timeout in seconds.")


class VexEvaluateRequest(BaseModel):
    sbom_path: str = Field(description="Absolute path to CycloneDX BOM JSON")
    vex_feed_path: str | None = Field(
        default=None, description="Path to local VEX feed directory"
    )
    vex_feed_url: str = Field(default="", description="HTTPS URL to remote VEX feed")


class SbomDiffRequest(BaseModel):
    """Two SBOM file paths to compare."""
    sbom_a_path: str = Field(description="Path to the older (baseline) CycloneDX BOM JSON")
    sbom_b_path: str = Field(description="Path to the newer CycloneDX BOM JSON")


class VerifyRequest(BaseModel):
    """Verify a Sigstore bundle for a model's CycloneDX BOM."""
    model_path: str = Field(description="Absolute path to model dir (contains cyclonedx-mlbom.json)")
    bundle_path: str | None = Field(
        default=None,
        description="Explicit .sig.json bundle path; defaults to <bom>.sig.json",
    )
    strict: bool = Field(
        default=False,
        description="When True, treat a missing bundle as a verification failure",
    )


class WebhookTestRequest(BaseModel):
    """Send a test event to the configured webhook URL."""
    webhook_url: str | None = Field(
        default=None,
        description="Override URL (uses SQUASH_WEBHOOK_URL env if omitted)",
    )


class PushRequest(BaseModel):
    """Push a CycloneDX BOM to an SBOM registry."""
    model_path: str = Field(description="Absolute path to model directory")
    registry_url: str = Field(description="Base URL of the SBOM registry")
    api_key: str = Field(default="", description="API key / bearer token for the registry")
    registry_type: str = Field(
        default="squash",
        description="Registry type: 'dtrack' (Dependency-Track), 'guac', or 'squash'",
    )


class ComposedAttestRequest(BaseModel):
    """Composite multi-model attestation request."""
    model_paths: list[str] = Field(description="Absolute paths to component model directories")
    output_dir: str | None = Field(default=None, description="Destination for artifacts")
    policies: list[str] = Field(default_factory=lambda: ["enterprise-strict"])
    sign: bool = Field(default=False)


# ── W49: offline / air-gapped request models ─────────────────────────────────

class KeygenRequest(BaseModel):
    """Generate a local Ed25519 keypair for offline BOM signing."""
    key_name: str = Field(description="Base filename for the keypair (no extension)")
    key_dir: str = Field(
        default=".",
        description="Directory to write <key_name>.priv.pem and <key_name>.pub.pem",
    )


class VerifyLocalRequest(BaseModel):
    """Verify a CycloneDX BOM against a local Ed25519 offline signature."""
    bom_path: str = Field(description="Absolute path to the CycloneDX BOM to verify")
    pub_key_path: str = Field(description="Absolute path to the Ed25519 .pub.pem file")
    sig_path: str | None = Field(
        default=None,
        description="Explicit .sig file path; defaults to <bom_path with .sig extension>",
    )


class PackOfflineRequest(BaseModel):
    """Bundle a model directory into a portable .squash-bundle.tar.gz archive."""
    model_dir: str = Field(description="Absolute path to the model directory to bundle")
    output_path: str | None = Field(
        default=None,
        description=(
            "Output .squash-bundle.tar.gz path; "
            "auto-generated as <model_dir>-<timestamp>.squash-bundle.tar.gz if omitted"
        ),
    )


# ── W52-55: Cloud dashboard request/response models ───────────────────────────


class TenantCreateRequest(BaseModel):
    """Register a new tenant in the Squash Cloud dashboard."""
    tenant_id: str = Field(description="Unique tenant identifier (slug, max 64 chars)")
    name: str = Field(description="Human-readable tenant / organisation name")
    plan: str = Field(default="community", description="Subscription plan: community | pro | enterprise")
    contact_email: str = Field(default="", description="Primary contact e-mail")


class InventoryRegisterRequest(BaseModel):
    """Register an attestation result in the cloud model inventory.

    Called by CI/CD pipelines after a successful ``POST /attest``.
    """
    tenant_id: str = Field(description="Tenant that owns this model")
    model_id: str = Field(description="Human-readable model identifier")
    model_path: str = Field(description="Filesystem path where the model resides")
    bom_path: str = Field(default="", description="Path to the generated CycloneDX BOM")
    attestation_passed: bool = Field(description="Whether all policies passed")
    policy_results: dict[str, Any] = Field(
        default_factory=dict,
        description="Map of policy_name → {passed, error_count, warning_count}",
    )
    vex_cves: list[str] = Field(
        default_factory=list,
        description="CVE IDs surfaced by VEX evaluation for this model",
    )
    timestamp: str = Field(default="", description="ISO-8601 attestation timestamp; auto-set if empty")


class VexAlertRequest(BaseModel):
    """Ingest a VEX alert event into the cloud dashboard feed."""
    tenant_id: str = Field(description="Tenant to scope this alert to")
    cve_id: str = Field(description="CVE identifier (e.g. CVE-2024-12345)")
    severity: str = Field(default="unknown", description="critical | high | medium | low | unknown")
    model_id: str = Field(default="", description="Affected model identifier")
    status: str = Field(default="open", description="open | acknowledged | resolved")
    detail: str = Field(default="", description="Human-readable detail / remediation note")


class DriftEventRequest(BaseModel):
    """Ingest a drift event into the cloud dashboard stream."""
    tenant_id: str = Field(description="Tenant to scope this event to")
    model_id: str = Field(description="Model that changed")
    bom_a: str = Field(description="Path or identifier of the baseline BOM")
    bom_b: str = Field(description="Path or identifier of the new BOM")
    added: list[str] = Field(default_factory=list, description="Component hashes / IDs added")
    removed: list[str] = Field(default_factory=list, description="Component hashes / IDs removed")
    changed: list[str] = Field(default_factory=list, description="Component hashes / IDs changed")
    severity: str = Field(default="info", description="info | warning | critical")
    timestamp: str = Field(default="", description="ISO-8601 event timestamp; auto-set if empty")


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> PlainTextResponse:
    """Prometheus-compatible counter export."""
    lines: list[str] = []
    for name, value in _COUNTERS.items():
        lines.append(f"# TYPE {name} counter")
        lines.append(f"{name} {value}")
    return PlainTextResponse("\n".join(lines) + "\n")


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

    # Merge spdx_datasets with training_dataset_ids (deduplicated, order preserved).
    all_datasets: list[str] = list(req.training_dataset_ids) + [
        d for d in req.spdx_datasets if d not in req.training_dataset_ids
    ]

    # Construct SpdxOptions when any SPDX enrichment field is populated.
    spdx_options = None
    if any([req.spdx_type, req.spdx_safety_risk, req.spdx_training_info,
            req.spdx_sensitive_data, req.spdx_datasets]):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_options = SpdxOptions(
            type_of_model=req.spdx_type or "text-generation",
            safety_risk_assessment=req.spdx_safety_risk or "unspecified",
            dataset_ids=all_datasets,
            information_about_training=req.spdx_training_info or "see-model-card",
            sensitive_personal_information=req.spdx_sensitive_data or "absent",
        )

    config = AttestConfig(
        model_path=Path(req.model_path),
        output_dir=Path(req.output_dir) if req.output_dir else None,
        model_id=req.model_id,
        hf_repo=req.hf_repo,
        model_family=req.model_family,
        quant_format=req.quant_format,
        policies=req.policies,
        sign=req.sign,
        offline=req.offline,
        local_signing_key=Path(req.local_signing_key) if req.local_signing_key else None,
        fail_on_violation=False,  # never raise from HTTP handler; return 422 instead
        skip_scan=req.skip_scan,
        training_dataset_ids=all_datasets,
        vex_feed_path=Path(req.vex_feed_path) if req.vex_feed_path else None,
        vex_feed_url=req.vex_feed_url,
        spdx_options=spdx_options,
    )

    loop = asyncio.get_running_loop()
    result: AttestResult = await loop.run_in_executor(
        _executor, AttestPipeline.run, config
    )

    master_data = _result_to_dict(result)

    _COUNTERS["squash_attest_total"] += 1

    # ── Auto-register in cloud inventory when tenant_id is provided ───────────
    if req.tenant_id:
        _ts = _ts_now()
        _rec: dict[str, Any] = {
            "model_id": result.model_id or req.model_id,
            "model_path": req.model_path,
            "bom_path": str(result.cyclonedx_path) if result.cyclonedx_path else "",
            "attestation_passed": result.passed,
            "policy_results": master_data.get("policy_results", {}),
            "vex_cves": [],
            "timestamp": _ts,
            "record_id": str(uuid.uuid4()),
        }
        _db_write_inventory(req.tenant_id, _rec)
        for policy_name, pr in _rec["policy_results"].items():
            _db_inc_policy_stat(req.tenant_id, policy_name, passed=bool(pr.get("passed")))

    if req.fail_on_violation and not result.passed:
        return JSONResponse(status_code=422, content=master_data)

    return JSONResponse(content=master_data)


@app.post("/scan")
async def scan(req: ScanRequest) -> JSONResponse:
    """Queue a security scan job.  Returns ``{"job_id": "..."}`` immediately.

    The scan runs asynchronously in the thread-pool executor.  Poll
    ``GET /scan/{job_id}`` for results.

    Why async queuing?  Scanning a large GGUF or ONNX model can take several
    seconds.  Blocking the HTTP response for that duration breaks load-balanced
    deployments.  The queue is in-memory and bounded to
    ``SQUASH_SCAN_JOB_LIMIT`` entries (default 1000); the oldest job is evicted
    when the limit is hit.
    """
    _require_path(req.model_path)
    model_path = Path(req.model_path)
    scan_dir = model_path if model_path.is_dir() else model_path.parent

    job_id = str(uuid.uuid4())

    # Evict oldest if at capacity
    if len(_scan_jobs) >= _SCAN_JOB_LIMIT:
        _scan_jobs.popitem(last=False)

    _scan_jobs[job_id] = {"status": "pending", "result": None}

    loop = asyncio.get_running_loop()

    def _do_scan() -> None:
        try:
            result = ModelScanner.scan_directory(scan_dir)
            payload = {
                "status": result.status,
                "is_safe": result.is_safe,
                "critical": result.critical_count,
                "high": result.high_count,
                "findings": [
                    {
                        "severity": f.severity,
                        "id": f.finding_id,
                        "title": f.title,
                        "detail": f.detail,
                        "file": f.file_path,
                        "cve": f.cve,
                    }
                    for f in result.findings
                ],
            }
            _scan_jobs[job_id] = {"status": "done", "result": payload}
        except Exception as exc:  # noqa: BLE001
            log.warning("scan job %s failed: %s", job_id, exc)
            _scan_jobs[job_id] = {"status": "error", "result": {"error": str(exc)}}

    _COUNTERS["squash_scan_total"] += 1
    loop.run_in_executor(_executor, _do_scan)
    return JSONResponse(status_code=202, content={"job_id": job_id})


@app.get("/scan/{job_id}")
async def scan_result(job_id: str) -> JSONResponse:
    """Return the result of a previously queued scan job.

    Responses:
    - ``200``  ``{"status": "done",    "result": {...}}``
    - ``200``  ``{"status": "error",   "result": {"error": "..."}}``
    - ``202``  ``{"status": "pending"}``  — scan still running
    - ``404``  job_id unknown
    """
    job = _scan_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    if job["status"] == "pending":
        return JSONResponse(status_code=202, content={"status": "pending"})

    return JSONResponse(content={"status": job["status"], "result": job["result"]})


@app.get("/scan/{job_id}/sarif")
async def scan_result_sarif(job_id: str) -> JSONResponse:
    """Return the SARIF 2.1.0 representation of a completed scan job.

    Responses:
    - ``200``  SARIF document (``application/json``)
    - ``202``  scan still pending
    - ``404``  job_id unknown
    - ``400``  job ended with an error (no scan result to convert)
    """
    from squish.squash.sarif import SarifBuilder  # noqa: PLC0415

    job = _scan_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")
    if job["status"] == "pending":
        return JSONResponse(status_code=202, content={"status": "pending"})
    if job["status"] == "error":
        raise HTTPException(
            status_code=400,
            detail=f"Scan job {job_id} ended with an error: {job['result'].get('error', 'unknown')}",
        )
    sarif = SarifBuilder.from_payload(job["result"])
    return JSONResponse(content=sarif)


@app.post("/policy/evaluate")
async def evaluate_policy(req: PolicyEvaluateRequest) -> JSONResponse:
    """Evaluate a submitted CycloneDX BOM against a named policy template or custom rules.

    - When ``custom_rules`` is provided, evaluates the ad-hoc rule list.
      The ``policy`` field is used only as the label in the response.
    - When ``custom_rules`` is absent, evaluates the named built-in policy.

    The SBOM is posted as a JSON body; no files are read from disk.
    """
    if req.custom_rules is not None:
        # Validate custom rules before running
        errors = PolicyRegistry.validate_rules(req.custom_rules)
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"message": "Invalid custom rules", "errors": errors},
            )
        policy_result = PolicyEngine.evaluate_custom(
            req.sbom,
            req.custom_rules,
            policy_name=req.policy or "custom",
        )
    else:
        if req.policy not in AVAILABLE_POLICIES:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown policy '{req.policy}'. Available: {sorted(AVAILABLE_POLICIES)}",
            )
        policy_result = PolicyEngine.evaluate(req.sbom, req.policy)

    _COUNTERS["squash_policy_evaluate_total"] += 1
    if not policy_result.passed:
        _COUNTERS["squash_policy_violations_total"] += policy_result.error_count
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
                    "remediation_link": f.remediation_link,
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

    _COUNTERS["squash_vex_evaluate_total"] += 1
    report_dict = await loop.run_in_executor(_executor, _run_vex)
    return JSONResponse(content=report_dict)


@app.get("/vex/status")
async def vex_status() -> JSONResponse:
    """Return metadata about the local VEX feed cache.

    Reads the on-disk manifest — no network I/O.  Returns 200 in all cases;
    callers should check the ``empty`` field before relying on other fields.
    """
    try:
        from squish.squash.vex import VexCache  # noqa: PLC0415
    except ImportError:
        raise HTTPException(500, "VEX engine not available (import error)")

    cache = VexCache()
    manifest = cache.manifest()

    _COUNTERS["squash_vex_status_total"] += 1

    if not manifest:
        return JSONResponse(content={"empty": True})

    return JSONResponse(
        content={
            "empty": False,
            "url": manifest.get("url", ""),
            "last_fetched": manifest.get("last_fetched", ""),
            "statement_count": manifest.get("statement_count", 0),
            "stale": cache.is_stale(),
        }
    )


@app.post("/vex/update")
async def vex_update(req: VexUpdateRequest) -> JSONResponse:
    """Force-refresh the local VEX feed cache from a remote URL.

    Equivalent to ``squash vex update``.  The URL resolves via:
    1. ``req.url`` if provided
    2. ``$SQUASH_VEX_URL`` environment variable
    3. :attr:`VexCache.DEFAULT_URL` (canonical community feed)

    Returns 200 on success, 502 on network failure.
    """
    try:
        from squish.squash.vex import VexCache  # noqa: PLC0415
    except ImportError:
        raise HTTPException(500, "VEX engine not available (import error)")

    url = req.url or os.environ.get("SQUASH_VEX_URL", VexCache.DEFAULT_URL)

    loop = asyncio.get_running_loop()

    def _do_update() -> dict:
        cache = VexCache()
        feed = cache.load_or_fetch(url, timeout=req.timeout, force=True)
        return {
            "url": url,
            "statement_count": sum(len(d.statements) for d in feed.documents),
            "updated": True,
        }

    try:
        _COUNTERS["squash_vex_update_total"] += 1
        result = await loop.run_in_executor(_executor, _do_update)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return JSONResponse(content=result)


@app.post("/sbom/diff")
async def sbom_diff(req: SbomDiffRequest) -> JSONResponse:
    """Compare two CycloneDX BOMs and return a diff summary."""
    from squish.squash.sbom_builder import SbomDiff

    _require_path(req.sbom_a_path)
    _require_path(req.sbom_b_path)

    def _run() -> dict:
        with open(req.sbom_a_path) as fh:
            bom_a = json.load(fh)
        with open(req.sbom_b_path) as fh:
            bom_b = json.load(fh)
        diff = SbomDiff.compare(bom_a, bom_b)
        return {
            "hash_changed": diff.hash_changed,
            "score_delta": diff.score_delta,
            "policy_status_changed": diff.policy_status_changed,
            "new_findings": diff.new_findings,
            "resolved_findings": diff.resolved_findings,
            "metadata_changes": {k: list(v) for k, v in diff.metadata_changes.items()},
            "has_regressions": diff.has_regressions,
        }

    _COUNTERS["squash_sbom_diff_total"] += 1
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    return JSONResponse(content=result)


@app.post("/attest/verify")
async def attest_verify(req: VerifyRequest) -> JSONResponse:
    """Verify the Sigstore bundle for a model's CycloneDX BOM.

    Returns ``{"verified": bool, "skipped": bool, "reason": str}``.
    """
    from squish.squash.oms_signer import OmsVerifier

    model_path = Path(req.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model path not found: {req.model_path}")

    bom_path = model_path / "cyclonedx-mlbom.json" if model_path.is_dir() else model_path
    if not bom_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"CycloneDX BOM not found: {bom_path}",
        )

    bundle_path = Path(req.bundle_path) if req.bundle_path else None

    def _run() -> dict:
        res = OmsVerifier.verify(bom_path, bundle_path)
        if res is None:
            if req.strict:
                return {"verified": False, "skipped": False, "reason": "no bundle (strict)"}
            return {"verified": False, "skipped": True, "reason": "no bundle found"}
        if res:
            return {"verified": True, "skipped": False, "reason": ""}
        return {"verified": False, "skipped": False, "reason": "bundle verification failed"}

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


@app.get("/report")
async def get_report(model_path: str) -> Any:
    """Generate a human-readable HTML compliance report for a model.

    Query parameter: ``model_path`` — absolute path to the model directory.
    Returns ``text/html``.
    """
    from fastapi.responses import HTMLResponse
    from squish.squash.report import ComplianceReporter

    if not Path(model_path).exists():
        raise HTTPException(status_code=404, detail=f"Model path not found: {model_path}")

    def _run() -> str:
        return ComplianceReporter.generate_html(Path(model_path))

    loop = asyncio.get_running_loop()
    html = await loop.run_in_executor(_executor, _run)
    return HTMLResponse(content=html)


@app.post("/webhooks/test")
async def webhooks_test(req: WebhookTestRequest) -> JSONResponse:
    """Send a synthetic test event to the configured webhook URL."""
    from squish.squash.policy import PolicyWebhook

    url = req.webhook_url or os.environ.get("SQUASH_WEBHOOK_URL", "")
    if not url:
        raise HTTPException(
            status_code=400,
            detail="No webhook URL provided; set SQUASH_WEBHOOK_URL or pass webhook_url",
        )

    def _run() -> dict:
        fired = PolicyWebhook.notify_raw(
            {
                "event": "squash_webhook_test",
                "message": "This is a test event from Squash.",
            },
            url,
        )
        return {"sent": fired, "url": url}

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


@app.post("/attest/composed")
async def attest_composed(req: ComposedAttestRequest) -> JSONResponse:
    """Run composite multi-model attestation and return the parent BOM path."""
    from squish.squash.attest import CompositeAttestConfig, CompositeAttestPipeline

    for mp in req.model_paths:
        if not Path(mp).exists():
            raise HTTPException(status_code=404, detail=f"Model path not found: {mp}")

    def _run() -> dict:
        cfg = CompositeAttestConfig(
            model_paths=[Path(p) for p in req.model_paths],
            output_dir=Path(req.output_dir) if req.output_dir else None,
            policies=req.policies,
            sign=req.sign,
        )
        result = CompositeAttestPipeline.run(cfg)
        return {
            "passed": result.passed,
            "component_count": len(result.component_results),
            "parent_bom_path": str(result.parent_bom_path) if result.parent_bom_path else None,
            "output_dir": str(result.output_dir),
            "error": result.error,
        }

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


@app.post("/sbom/push")
async def sbom_push(req: PushRequest) -> JSONResponse:
    """Push a CycloneDX BOM to an SBOM registry (Dependency-Track, GUAC, or squash)."""
    from squish.squash.sbom_builder import SbomRegistry

    model_path = Path(req.model_path)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model path not found: {req.model_path}")

    bom_path = model_path / "cyclonedx-mlbom.json"
    if not bom_path.exists():
        raise HTTPException(status_code=404, detail=f"CycloneDX BOM not found: {bom_path}")

    valid_types = {"dtrack", "guac", "squash"}
    if req.registry_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"registry_type must be one of {sorted(valid_types)}",
        )

    def _run() -> dict:
        if req.registry_type == "dtrack":
            url = SbomRegistry.push_dtrack(bom_path, req.registry_url, req.api_key)
        elif req.registry_type == "guac":
            url = SbomRegistry.push_guac(bom_path, req.registry_url)
        else:
            url = SbomRegistry.push_squash(bom_path, req.registry_url, req.api_key)
        return {"pushed": True, "registry_url": url, "bom_path": str(bom_path)}

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 20 — NTIA minimum elements validation ────────────────────────────────

class NtiaRequest(BaseModel):
    bom_path: str
    strict: bool = False


@app.post("/ntia/validate")
async def ntia_validate(req: NtiaRequest) -> JSONResponse:
    """Validate NTIA minimum elements in a CycloneDX BOM.

    Returns ``passed``, ``completeness_score`` (0–1), ``missing_fields``,
    and ``present_fields``.
    """
    _require_path(req.bom_path)
    from squish.squash.policy import NtiaValidator

    def _run() -> dict:
        result = NtiaValidator.check(Path(req.bom_path), strict=req.strict)
        return {
            "passed": result.passed,
            "completeness_score": result.completeness_score,
            "missing_fields": result.missing_fields,
            "present_fields": result.present_fields,
            "bom_path": str(result.bom_path) if result.bom_path else req.bom_path,
        }

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 21 — SLSA provenance attestation ─────────────────────────────────────

class SlsaAttestRequest(BaseModel):
    model_dir: str
    level: int = 1
    builder_id: str = "https://squish.local/squash/builder"
    sign: bool = False


@app.post("/slsa/attest")
async def slsa_attest(req: SlsaAttestRequest) -> JSONResponse:
    """Generate a SLSA 1.0 provenance statement for the given model directory.

    Returns ``output_path``, ``level``, ``subject_sha256``, and ``subject_name``.
    """
    _require_path(req.model_dir)
    from squish.squash.slsa import SlsaLevel, SlsaProvenanceBuilder

    def _run() -> dict:
        level = SlsaLevel(req.level)
        attest = SlsaProvenanceBuilder.build(
            Path(req.model_dir),
            level=level,
            builder_id=req.builder_id,
        )
        return {
            "output_path": str(attest.output_path),
            "level": attest.level.value,
            "subject_name": attest.subject_name,
            "subject_sha256": attest.subject_sha256,
            "signed": req.level >= 2,
            "invocation_id": attest.invocation_id,
        }

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 22 — BOM merge ────────────────────────────────────────────────────────

class BomMergeRequest(BaseModel):
    bom_paths: list[str]
    output_path: str
    metadata: dict = {}  # noqa: RUF012


@app.post("/sbom/merge")
async def sbom_merge(req: BomMergeRequest) -> JSONResponse:
    """Merge multiple CycloneDX BOMs into one canonical BOM.

    Deduplicates by PURL and unions vulnerabilities.
    """
    for p in req.bom_paths:
        _require_path(p)
    from squish.squash.sbom_builder import BomMerger

    def _run() -> dict:
        merged = BomMerger.merge(
            [Path(p) for p in req.bom_paths],
            Path(req.output_path),
            metadata=req.metadata or None,
        )
        return {
            "merged": True,
            "output_path": req.output_path,
            "component_count": len(merged.get("components", [])),
            "vulnerability_count": len(merged.get("vulnerabilities", [])),
        }

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 23 — AI risk assessment ──────────────────────────────────────────────

class RiskAssessRequest(BaseModel):
    model_dir: str
    framework: str = "both"


@app.post("/risk/assess")
async def risk_assess(req: RiskAssessRequest) -> JSONResponse:
    """Assess AI risk per EU AI Act and/or NIST AI RMF.

    ``framework`` must be ``"eu-ai-act"``, ``"nist-rmf"``, or ``"both"``
    (default).
    """
    _require_path(req.model_dir)
    from squish.squash.risk import AiRiskAssessor

    def _run() -> dict:
        bom_path = Path(req.model_dir) / "cyclonedx-mlbom.json"
        result: dict = {}
        if req.framework in ("eu-ai-act", "both"):
            eu = AiRiskAssessor.assess_eu_ai_act(bom_path)
            result["eu_ai_act"] = {
                "passed": eu.passed,
                "category": eu.category.value,
                "rationale": eu.rationale,
                "mitigation_required": eu.mitigation_required,
            }
        if req.framework in ("nist-rmf", "both"):
            rmf = AiRiskAssessor.assess_nist_rmf(bom_path)
            result["nist_rmf"] = {
                "passed": rmf.passed,
                "category": rmf.category.value,
                "rationale": rmf.rationale,
                "mitigation_required": rmf.mitigation_required,
            }
        return result

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 24 — Drift monitoring ─────────────────────────────────────────────────

class MonitorSnapshotRequest(BaseModel):
    model_dir: str


class MonitorCompareRequest(BaseModel):
    model_dir: str
    baseline_snapshot: str


@app.post("/monitor/snapshot")
async def monitor_snapshot(req: MonitorSnapshotRequest) -> JSONResponse:
    """Return a SHA-256 snapshot fingerprint of *model_dir*."""
    _require_path(req.model_dir)
    from squish.squash.governor import DriftMonitor

    def _run() -> dict:
        snap = DriftMonitor.snapshot(Path(req.model_dir))
        return {"snapshot": snap, "model_dir": req.model_dir}

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


@app.post("/monitor/compare")
async def monitor_compare(req: MonitorCompareRequest) -> JSONResponse:
    """Compare current *model_dir* state against *baseline_snapshot*.

    Returns a list of drift events.
    """
    _require_path(req.model_dir)
    from squish.squash.governor import DriftMonitor

    def _run() -> dict:
        events = DriftMonitor.compare(Path(req.model_dir), req.baseline_snapshot)
        return {
            "drift_detected": len(events) > 0,
            "event_count": len(events),
            "events": [
                {
                    "event_type": e.event_type,
                    "component": e.component,
                    "old_value": e.old_value,
                    "new_value": e.new_value,
                    "detected_at": e.detected_at,
                }
                for e in events
            ],
        }

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 25 — CI/CD integration ────────────────────────────────────────────────

class CiRunRequest(BaseModel):
    model_dir: str
    report_format: str = "text"


@app.post("/cicd/report")
async def cicd_report(req: CiRunRequest) -> JSONResponse:
    """Run the full squash check pipeline and return a CI report.

    Runs NTIA validation, AI risk assessment, and drift detection.
    """
    _require_path(req.model_dir)
    from squish.squash.cicd import CicdAdapter

    def _run() -> dict:
        report = CicdAdapter.run_pipeline(
            Path(req.model_dir), report_format=req.report_format
        )
        return {
            "passed": report.passed,
            "env": {
                "env_name": report.env.env_name,
                "job_id": report.env.job_id,
                "repo": report.env.repo,
                "branch": report.env.branch,
            },
            "ntia": report.ntia,
            "risk": report.risk,
            "drift_events": report.drift_events,
            "summary": CicdAdapter.job_summary(report),
        }

    loop = asyncio.get_running_loop()
    return JSONResponse(content=await loop.run_in_executor(_executor, _run))


# ── Wave 30 — VEX publish + integration attestation REST endpoints ─────────────


class VexPublishRequest(BaseModel):
    """Request body for POST /vex/publish."""

    entries: list[dict] = Field(default_factory=list)
    author: str = Field(default="squash")
    doc_id: str | None = Field(default=None)


@app.post("/vex/publish")
async def vex_publish(req: VexPublishRequest) -> JSONResponse:
    """Generate an OpenVEX 0.2.0 document from a list of statement entries.

    Returns the full OpenVEX document JSON.  Validates the document before
    returning; responds with 422 if validation errors are found.
    """
    from squish.squash.vex import VexFeedManifest

    def _run() -> dict:
        doc = VexFeedManifest.generate(
            req.entries,
            author=req.author,
            doc_id=req.doc_id or None,
        )
        errors = VexFeedManifest.validate(doc)
        if errors:
            raise ValueError("; ".join(errors))
        return doc

    loop = asyncio.get_running_loop()
    try:
        doc = await loop.run_in_executor(_executor, _run)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return JSONResponse(content=doc)


class AttestIntegrationRequest(BaseModel):
    """Shared request body for offline integration attestation endpoints."""

    model_path: str
    policies: list[str] | None = Field(default=None)
    sign: bool = False
    fail_on_violation: bool = False


class McpAttestRequest(BaseModel):
    """Request body for POST /attest/mcp — MCP tool manifest attestation."""

    catalog_path: str = Field(description="Absolute path to MCP tool catalog JSON file")
    policy: str = Field(default="mcp-strict", description="Policy to apply (default: mcp-strict)")
    sign: bool = Field(default=False, description="Sign catalog with Sigstore after scanning")
    fail_on_violation: bool = Field(
        default=False,
        description="Return HTTP 422 if any error-severity finding is present",
    )


# ── Wave 47 — RAG KB integrity ────────────────────────────────────────────────


class RagIndexRequest(BaseModel):
    """Request body for POST /rag/index."""

    corpus_dir: str = Field(description="Absolute path to the corpus directory to index")
    glob: str = Field(default="**/*", description='File glob pattern (default "**/*")')


class RagVerifyRequest(BaseModel):
    """Request body for POST /rag/verify."""

    corpus_dir: str = Field(description="Absolute path to the corpus directory to verify")


# ── Wave 48 — Model transformation lineage ────────────────────────────────────


class LineageRecordRequest(BaseModel):
    """Request body for POST /lineage/record."""

    model_dir: str = Field(description="Absolute path to the model artefact directory")
    operation: str = Field(description='Operation label, e.g. "compress", "sign", "export"')
    model_id: str = Field(default="", description="Model identifier (default: directory name)")
    input_dir: str = Field(default="", description="Source directory (default: model_dir)")
    params: dict = Field(default_factory=dict, description="Operation-specific key/value params")


class LineageVerifyRequest(BaseModel):
    """Request body for POST /lineage/verify."""

    model_dir: str = Field(description="Absolute path to the model artefact directory")


@app.post("/attest/mlflow")
async def attest_mlflow(req: AttestIntegrationRequest) -> JSONResponse:
    """Run an offline AttestPipeline for an MLflow model artifact.

    Equivalent to ``squash attest-mlflow <model_path>``.
    """
    _require_path(req.model_path)

    def _run() -> dict:
        config = AttestConfig(
            model_path=Path(req.model_path),
            policies=(req.policies if req.policies is not None else ["enterprise-strict"]),
            sign=req.sign,
            fail_on_violation=False,  # never raise from HTTP handler
            output_dir=Path(req.model_path).parent / "squash",
        )
        result = AttestPipeline.run(config)
        return _result_to_dict(result)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    if req.fail_on_violation and not result.get("passed"):
        return JSONResponse(content=result, status_code=422)
    status = 200 if result.get("passed") else 400
    return JSONResponse(content=result, status_code=status)


@app.post("/attest/wandb")
async def attest_wandb(req: AttestIntegrationRequest) -> JSONResponse:
    """Run an offline AttestPipeline for a W&B artifact directory.

    Equivalent to ``squash attest-wandb <model_path>``.
    """
    _require_path(req.model_path)

    def _run() -> dict:
        config = AttestConfig(
            model_path=Path(req.model_path),
            policies=(req.policies if req.policies is not None else ["enterprise-strict"]),
            sign=req.sign,
            fail_on_violation=False,  # never raise from HTTP handler
            output_dir=Path(req.model_path).parent / "squash",
        )
        result = AttestPipeline.run(config)
        return _result_to_dict(result)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    if req.fail_on_violation and not result.get("passed"):
        return JSONResponse(content=result, status_code=422)
    status = 200 if result.get("passed") else 400
    return JSONResponse(content=result, status_code=status)


class AttestHuggingFaceRequest(BaseModel):
    """Request body for POST /attest/huggingface."""

    model_path: str
    repo_id: str | None = Field(default=None)
    hf_token: str | None = Field(default=None)
    policies: list[str] | None = Field(default=None)
    sign: bool = False
    fail_on_violation: bool = False


@app.post("/attest/huggingface")
async def attest_huggingface(req: AttestHuggingFaceRequest) -> JSONResponse:
    """Attest a HuggingFace model — offline or with Hub push.

    If ``repo_id`` is provided the attestation artefacts are pushed to the Hub
    via ``HFSquash.attest_and_push()``.  Otherwise an offline
    ``AttestPipeline.run()`` is executed.

    Equivalent to ``squash attest-huggingface <model_path> [--repo-id ...]``.
    """
    _require_path(req.model_path)

    def _run() -> dict:
        if req.repo_id:
            from squish.squash.integrations.huggingface import HFSquash

            result = HFSquash.attest_and_push(
                req.repo_id,
                Path(req.model_path),
                hf_token=req.hf_token or "",
                policies=(req.policies if req.policies is not None else ["enterprise-strict"]),
                sign=req.sign,
                fail_on_violation=False,  # never raise from HTTP handler
                repo_prefix="squash-attestations",
            )
        else:
            config = AttestConfig(
                model_path=Path(req.model_path),
                policies=(req.policies if req.policies is not None else ["enterprise-strict"]),
                sign=req.sign,
                fail_on_violation=False,  # never raise from HTTP handler
                output_dir=Path(req.model_path).parent / "squash",
            )
            result = AttestPipeline.run(config)
        return _result_to_dict(result)

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(_executor, _run)
    except Exception as exc:  # HF push failures (auth, network, missing dep)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if req.fail_on_violation and not result.get("passed"):
        return JSONResponse(content=result, status_code=422)
    status = 200 if result.get("passed") else 400
    return JSONResponse(content=result, status_code=status)


@app.post("/attest/langchain")
async def attest_langchain(req: AttestIntegrationRequest) -> JSONResponse:
    """Run a pre-deployment attestation for a LangChain pipeline artifact.

    Equivalent to ``squash attest-langchain <model_path>``.
    """
    _require_path(req.model_path)

    def _run() -> dict:
        config = AttestConfig(
            model_path=Path(req.model_path),
            policies=(req.policies if req.policies is not None else ["enterprise-strict"]),
            sign=req.sign,
            fail_on_violation=False,  # never raise from HTTP handler
            output_dir=Path(req.model_path).parent / "squash",
        )
        result = AttestPipeline.run(config)
        return _result_to_dict(result)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    if req.fail_on_violation and not result.get("passed"):
        return JSONResponse(content=result, status_code=422)
    status = 200 if result.get("passed") else 400
    return JSONResponse(content=result, status_code=status)


@app.post("/attest/mcp")
async def attest_mcp(req: McpAttestRequest) -> JSONResponse:
    """Scan an MCP tool manifest catalog for supply-chain threats.

    Equivalent to ``squash attest-mcp <catalog_path> --policy mcp-strict``.

    Addresses EU AI Act Art. 9(2)(d): adversarial input resilience for
    agentic AI systems that invoke MCP tools at runtime.
    """
    _require_path(req.catalog_path)

    from squish.squash.mcp import McpScanner, McpSigner  # lazy — keeps api.py import-fast

    def _run() -> dict:
        result = McpScanner.scan_file(Path(req.catalog_path), req.policy)
        if req.sign:
            McpSigner.sign(Path(req.catalog_path))
        return result.to_dict()

    loop = asyncio.get_running_loop()
    mcp_result = await loop.run_in_executor(_executor, _run)
    _COUNTERS["squash_attest_total"] += 1
    if req.fail_on_violation and mcp_result.get("status") == "unsafe":
        return JSONResponse(content=mcp_result, status_code=422)
    return JSONResponse(content=mcp_result)


def _require_path(p: str) -> None:
    if not Path(p).exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {p}")


# ── Wave 46 — Agent audit trail endpoint ─────────────────────────────────────

@app.get("/audit/trail")
async def get_audit_trail(
    limit: int = 100,
    log: str | None = None,
) -> JSONResponse:
    """Return the last *limit* entries from the agent audit trail.

    The audit trail is an append-only JSONL file maintained by
    :class:`~squish.squash.governor.AgentAuditLogger`.  Each entry contains a
    SHA-256 hash of the LLM input/output, the event type, session ID, and a
    forward hash-chain link for tamper evidence.

    Addresses EU AI Act Art. 12: mandatory logging for high-risk AI systems.

    Query parameters
    ----------------
    limit:
        Maximum number of most-recent entries to return (default 100, max 1000).
    log:
        Override the audit log file path (default: ``SQUASH_AUDIT_LOG`` env or
        ``~/.squash/audit.jsonl``).
    """
    from squish.squash.governor import AgentAuditLogger

    limit = max(1, min(limit, 1000))
    logger = AgentAuditLogger(log_path=log)
    entries = logger.read_tail(limit)
    return JSONResponse(
        content={
            "count": len(entries),
            "log_path": str(logger.path),
            "entries": entries,
        }
    )


# ── Wave 47 — RAG KB integrity ────────────────────────────────────────────────


@app.post("/rag/index")
async def post_rag_index(req: RagIndexRequest) -> JSONResponse:
    """Hash every document in *corpus_dir* and write ``.rag_manifest.json``.

    Equivalent to ``squash scan-rag index <corpus_dir>``.

    Addresses the #1 enterprise RAG failure: silently poisoned or drifted
    knowledge bases.  The returned ``manifest_hash`` is a deterministic
    content fingerprint suitable for CI/CD gating.
    """
    from squish.squash.rag import RagScanner

    corpus = Path(req.corpus_dir)
    if not corpus.is_dir():
        raise HTTPException(status_code=404, detail=f"corpus_dir not found: {req.corpus_dir}")

    def _run() -> dict:
        manifest = RagScanner.index(corpus, glob=req.glob)
        return {
            "corpus_dir": manifest.corpus_dir,
            "file_count": manifest.file_count,
            "manifest_path": str(Path(manifest.corpus_dir) / RagScanner.MANIFEST_FILENAME),
            "manifest_hash": manifest.manifest_hash,
            "indexed_at": manifest.indexed_at,
        }

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    return JSONResponse(content=result)


@app.post("/rag/verify")
async def post_rag_verify(req: RagVerifyRequest) -> JSONResponse:
    """Compare live corpus against the stored manifest.

    Equivalent to ``squash scan-rag verify <corpus_dir>``.

    Returns HTTP 200 with ``"ok": true`` when no drift is detected.
    Returns HTTP 200 with ``"ok": false`` when drift exists — callers
    should gate deployments on the ``ok`` field, not the status code.
    """
    from squish.squash.rag import RagScanner

    corpus = Path(req.corpus_dir)
    if not corpus.exists():
        raise HTTPException(status_code=404, detail=f"corpus_dir not found: {req.corpus_dir}")

    def _run() -> dict:
        result = RagScanner.verify(corpus)
        return result.to_dict()

    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(_executor, _run)
    return JSONResponse(content=content)


# ── Wave 48 — Model transformation lineage ────────────────────────────────────


@app.post("/lineage/record")
async def post_lineage_record(req: LineageRecordRequest) -> JSONResponse:
    """Record a model transformation event into the lineage chain.

    Equivalent to ``squash lineage record <model_dir> --operation <op>``.
    """
    from squish.squash.lineage import LineageChain

    model_dir = Path(req.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    def _run() -> dict:
        model_id = req.model_id or model_dir.name
        input_dir = req.input_dir or str(model_dir)
        evt = LineageChain.create_event(
            operation=req.operation,
            model_id=model_id,
            input_dir=input_dir,
            output_dir=str(model_dir),
            params=req.params,
        )
        event_hash = LineageChain.record(model_dir, evt)
        return {"event_hash": event_hash, "event_id": evt.event_id, "model_dir": str(model_dir)}

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(_executor, _run)
    return JSONResponse(content=result)


@app.get("/lineage/show")
async def get_lineage_show(model_dir: str) -> JSONResponse:
    """Return all recorded lineage events for a model directory.

    Equivalent to ``squash lineage show <model_dir>``.
    """
    from squish.squash.lineage import LineageChain

    mdir = Path(model_dir)
    if not mdir.exists():
        raise HTTPException(status_code=404, detail=f"model_dir not found: {model_dir}")

    def _run() -> list[dict]:
        events = LineageChain.load(mdir)
        return [e.to_dict() for e in events]

    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(_executor, _run)
    return JSONResponse(content={"model_dir": str(mdir), "event_count": len(events), "events": events})


@app.post("/lineage/verify")
async def post_lineage_verify(req: LineageVerifyRequest) -> JSONResponse:
    """Verify the integrity of a model's lineage chain.

    Equivalent to ``squash lineage verify <model_dir>``.
    Returns HTTP 200 with ``"ok": true`` when the chain is intact.
    Returns HTTP 200 with ``"ok": false`` on tampering or missing chain.
    """
    from squish.squash.lineage import LineageChain

    mdir = Path(req.model_dir)
    if not mdir.exists():
        raise HTTPException(status_code=404, detail=f"model_dir not found: {req.model_dir}")

    def _run() -> dict:
        result = LineageChain.verify(mdir)
        return result.to_dict()

    loop = asyncio.get_running_loop()
    content = await loop.run_in_executor(_executor, _run)
    return JSONResponse(content=content)


# ── W49: offline / air-gapped endpoints ──────────────────────────────────────


@app.post("/keygen")
async def keygen(req: KeygenRequest) -> JSONResponse:
    """Generate an Ed25519 keypair for offline BOM signing.

    Equivalent to ``squash keygen <name> --key-dir <dir>``.
    Returns the absolute paths to the generated ``.priv.pem`` and ``.pub.pem`` files.
    """
    from squish.squash.oms_signer import OmsSigner

    def _run() -> tuple[str, str]:
        priv, pub = OmsSigner.keygen(req.key_name, req.key_dir)
        return str(priv), str(pub)

    loop = asyncio.get_running_loop()
    try:
        priv_path, pub_path = await loop.run_in_executor(_executor, _run)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return JSONResponse(content={"priv_path": priv_path, "pub_path": pub_path})


@app.post("/attest/verify-local")
async def verify_local(req: VerifyLocalRequest) -> JSONResponse:
    """Verify a CycloneDX BOM against a local Ed25519 offline signature.

    Equivalent to ``squash verify-local <bom_path> --key <pub_key_path>``.
    Returns ``{"ok": true}`` on valid signature, ``{"ok": false}`` on failure.
    """
    from squish.squash.oms_signer import OmsVerifier

    bom = Path(req.bom_path)
    if not bom.exists():
        raise HTTPException(status_code=404, detail=f"bom_path not found: {req.bom_path}")

    pub = Path(req.pub_key_path)
    if not pub.exists():
        raise HTTPException(status_code=404, detail=f"pub_key_path not found: {req.pub_key_path}")

    sig = Path(req.sig_path) if req.sig_path else None

    def _run() -> bool:
        return OmsVerifier.verify_local(bom, pub, sig)

    loop = asyncio.get_running_loop()
    try:
        ok = await loop.run_in_executor(_executor, _run)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return JSONResponse(content={"ok": ok, "bom_path": str(bom)})


@app.post("/pack/offline")
async def pack_offline(req: PackOfflineRequest) -> JSONResponse:
    """Bundle a model directory into a portable .squash-bundle.tar.gz archive.

    Equivalent to ``squash pack-offline <model_dir>``.
    Returns the bundle path and its size in bytes.
    """
    from squish.squash.oms_signer import OmsSigner

    mdir = Path(req.model_dir)
    if not mdir.exists():
        raise HTTPException(status_code=404, detail=f"model_dir not found: {req.model_dir}")

    out = Path(req.output_path) if req.output_path else None

    def _run() -> str:
        return str(OmsSigner.pack_offline(mdir, out))

    loop = asyncio.get_running_loop()
    try:
        bundle_path = await loop.run_in_executor(_executor, _run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    size_bytes = Path(bundle_path).stat().st_size
    return JSONResponse(content={
        "bundle_path": bundle_path,
        "size_bytes": size_bytes,
        "model_dir": str(mdir),
    })


# ──────────────────────────────────────────────────────────────────────────────
# W52-55: Squash Cloud — tenant management + dashboard endpoints
# ──────────────────────────────────────────────────────────────────────────────
# All /cloud/* endpoints are multi-tenant.  Tenant identity is resolved via:
#   • SQUASH_JWT_SECRET set → JWT Bearer auth (tenant_id claim)
#   • No SQUASH_JWT_SECRET  → X-Tenant-ID header (single-tenant or trusted proxy)
#
# For the hosted Squash Cloud product the Next.js dashboard authenticates with
# the SSO provider (OIDC), receives a signed JWT, and forwards it here.  The
# squish API validates the JWT signature with the shared SQUASH_JWT_SECRET
# (HS256) and extracts the tenant_id claim.
# ──────────────────────────────────────────────────────────────────────────────


def _ts_now() -> str:
    """Return current UTC time as ISO-8601 string."""
    import datetime
    return datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z")


# ── Tenant management ─────────────────────────────────────────────────────────

@app.post("/cloud/tenant", status_code=201)
async def cloud_create_tenant(req: TenantCreateRequest) -> JSONResponse:
    """Register a new tenant in the Squash Cloud dashboard.

    Idempotent: re-posting the same ``tenant_id`` updates the record.

    Used by the Next.js onboarding flow and by enterprise provisioning scripts.
    """
    if len(req.tenant_id) > 64 or not req.tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id must be 1–64 chars")
    if len(_tenants) >= _TENANT_LIMIT:
        raise HTTPException(status_code=507, detail="Tenant limit reached")
    record: dict[str, Any] = {
        "tenant_id": req.tenant_id,
        "name": req.name,
        "plan": req.plan,
        "contact_email": req.contact_email,
        "created_at": _tenants.get(req.tenant_id, {}).get("created_at", _ts_now()),
        "updated_at": _ts_now(),
    }
    _db_write_tenant(req.tenant_id, record)
    return JSONResponse(status_code=201, content=record)


@app.get("/cloud/tenant/{tenant_id}")
async def cloud_get_tenant(tenant_id: str) -> JSONResponse:
    """Return metadata for a registered tenant.

    Used by the dashboard header / org switcher to display plan and quota info.
    """
    record = _tenants.get(tenant_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")
    return JSONResponse(content=record)


@app.get("/cloud/tenants")
async def cloud_list_tenants(request: Request) -> JSONResponse:
    """List all registered tenants.  Admin-only — requires SQUASH_API_TOKEN auth."""
    return JSONResponse(content={
        "count": len(_tenants),
        "tenants": list(_tenants.values()),
    })


# ── Model inventory ───────────────────────────────────────────────────────────

@app.post("/cloud/inventory/register", status_code=201)
async def cloud_register_inventory(
    req: InventoryRegisterRequest, request: Request
) -> JSONResponse:
    """Register an attestation result in the cloud model inventory.

    Called automatically by ``POST /attest`` when ``tenant_id`` is set, or
    explicitly by CI/CD scripts for out-of-band registrations.
    """
    if not req.tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id required")
    ts = req.timestamp or _ts_now()
    record: dict[str, Any] = {
        "model_id": req.model_id,
        "model_path": req.model_path,
        "bom_path": req.bom_path,
        "attestation_passed": req.attestation_passed,
        "policy_results": req.policy_results,
        "vex_cves": req.vex_cves,
        "timestamp": ts,
        "record_id": str(uuid.uuid4()),
    }
    _db_write_inventory(req.tenant_id, record)
    # Update policy stats aggregates
    for policy_name, pr in req.policy_results.items():
        _db_inc_policy_stat(req.tenant_id, policy_name, passed=bool(pr.get("passed")))
    return JSONResponse(status_code=201, content=record)


@app.get("/cloud/inventory")
async def cloud_get_inventory(
    request: Request,
    limit: int = 50,
    passed: bool | None = None,
) -> JSONResponse:
    """Return the model inventory for the resolved tenant.

    Query parameters
    ----------------
    limit:
        Maximum number of most-recent records to return (default 50, max 500).
    passed:
        If ``true``, return only models where all policies passed.
        If ``false``, return only models with policy failures.
        Omit to return all.
    """
    tenant_id = _resolve_tenant_id(request)
    limit = max(1, min(limit, 500))
    records: list[dict[str, Any]] = list(_inventory[tenant_id])
    if passed is not None:
        records = [r for r in records if r["attestation_passed"] is passed]
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "count": len(records[-limit:]),
        "total": len(records),
        "models": records[-limit:],
    })


# ── VEX alert feed ────────────────────────────────────────────────────────────

@app.post("/cloud/vex/alert", status_code=201)
async def cloud_post_vex_alert(
    req: VexAlertRequest, request: Request
) -> JSONResponse:
    """Ingest a VEX alert event into the cloud dashboard feed.

    Called by the VEX poller / ``squash vex subscribe`` daemon when a new
    CVE is detected in the feed for a subscribed tenant.
    """
    if not req.tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id required")
    record: dict[str, Any] = {
        "alert_id": str(uuid.uuid4()),
        "cve_id": req.cve_id,
        "severity": req.severity,
        "model_id": req.model_id,
        "status": req.status,
        "detail": req.detail,
        "tenant_id": req.tenant_id,
        "created_at": _ts_now(),
    }
    _db_write_vex_alert(req.tenant_id, record)
    return JSONResponse(status_code=201, content=record)


@app.get("/cloud/vex/alerts")
async def cloud_get_vex_alerts(
    request: Request,
    limit: int = 50,
    status: str | None = None,
    severity: str | None = None,
) -> JSONResponse:
    """Return the VEX alert feed for the resolved tenant.

    Addresses CISO-level requirement: real-time CVE exposure dashboard.

    Query parameters
    ----------------
    limit:
        Maximum recent alerts (default 50, max 500).
    status:
        Filter by alert status: ``open`` | ``acknowledged`` | ``resolved``.
    severity:
        Filter by severity: ``critical`` | ``high`` | ``medium`` | ``low`` | ``unknown``.
    """
    tenant_id = _resolve_tenant_id(request)
    limit = max(1, min(limit, 500))
    alerts: list[dict[str, Any]] = list(_vex_alerts[tenant_id])
    if status:
        alerts = [a for a in alerts if a["status"] == status]
    if severity:
        alerts = [a for a in alerts if a["severity"] == severity]
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "count": len(alerts[-limit:]),
        "total": len(alerts),
        "alerts": alerts[-limit:],
    })


# ── Drift event stream ────────────────────────────────────────────────────────

@app.post("/cloud/drift/event", status_code=201)
async def cloud_post_drift_event(
    req: DriftEventRequest, request: Request
) -> JSONResponse:
    """Ingest a drift event into the cloud dashboard stream.

    Called by CI/CD after ``squash drift-check`` detects BOM divergence.
    Enables boardroom-level "supply chain integrity over time" charts.
    """
    if not req.tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id required")
    record: dict[str, Any] = {
        "event_id": str(uuid.uuid4()),
        "model_id": req.model_id,
        "bom_a": req.bom_a,
        "bom_b": req.bom_b,
        "added": req.added,
        "removed": req.removed,
        "changed": req.changed,
        "severity": req.severity,
        "tenant_id": req.tenant_id,
        "timestamp": req.timestamp or _ts_now(),
    }
    _db_write_drift_event(req.tenant_id, record)
    return JSONResponse(status_code=201, content=record)


@app.get("/cloud/drift/events")
async def cloud_get_drift_events(
    request: Request,
    limit: int = 50,
    model_id: str | None = None,
    severity: str | None = None,
) -> JSONResponse:
    """Return the drift event stream for the resolved tenant.

    Addresses EU AI Act Art. 16(d): ongoing monitoring of model changes.

    Query parameters
    ----------------
    limit:
        Maximum recent events (default 50, max 500).
    model_id:
        Filter to a specific model.
    severity:
        Filter by: ``info`` | ``warning`` | ``critical``.
    """
    tenant_id = _resolve_tenant_id(request)
    limit = max(1, min(limit, 500))
    events: list[dict[str, Any]] = list(_drift_events[tenant_id])
    if model_id:
        events = [e for e in events if e["model_id"] == model_id]
    if severity:
        events = [e for e in events if e["severity"] == severity]
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "count": len(events[-limit:]),
        "total": len(events),
        "events": events[-limit:],
    })


# ── Policy dashboard ──────────────────────────────────────────────────────────

@app.get("/cloud/policy/dashboard")
async def cloud_policy_dashboard(request: Request) -> JSONResponse:
    """Return policy pass/fail aggregates for the resolved tenant.

    Feeds the boardroom compliance reporting dashboard:
    "What % of model deployments are policy-compliant across our fleet?"

    Addresses CMMC Level 2/3, NIST AI RMF, and ISO/IEC 42001 audit requirements.
    """
    tenant_id = _resolve_tenant_id(request)
    stats = _policy_stats[tenant_id]
    dashboard: list[dict[str, Any]] = []
    for policy_name, counts in stats.items():
        total = counts["passed"] + counts["failed"]
        rate = round(counts["passed"] / total, 4) if total else 0.0
        dashboard.append({
            "policy": policy_name,
            "passed": counts["passed"],
            "failed": counts["failed"],
            "total": total,
            "pass_rate": rate,
        })
    # Overall aggregate — model-level (did the deployment pass attestation?)
    models = list(_inventory[tenant_id])
    m_passed = sum(1 for m in models if m.get("attestation_passed"))
    m_failed = len(models) - m_passed
    m_total = len(models)
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "overall": {
            "passed": m_passed,
            "failed": m_failed,
            "total": m_total,
            "pass_rate": round(m_passed / m_total, 4) if m_total else 0.0,
        },
        "by_policy": dashboard,
    })


# ── Tenant-scoped audit log ───────────────────────────────────────────────────

@app.get("/cloud/audit")
async def cloud_get_audit(
    request: Request,
    limit: int = 100,
    log: str | None = None,
) -> JSONResponse:
    """Return the last *limit* audit trail entries, scoped to the resolved tenant.

    Delegates to the existing AgentAuditLogger hash-chain log.  When
    ``tenant_id`` is non-empty, entries are filtered to those whose
    ``session_id`` starts with the tenant prefix (CI pipelines should set
    ``SQUASH_AUDIT_SESSION_PREFIX=<tenant_id>-``).

    Addresses EU AI Act Art. 12 and SEC cybersecurity disclosure requirements.
    """
    from squish.squash.governor import AgentAuditLogger

    tenant_id = _resolve_tenant_id(request)
    limit = max(1, min(limit, 1000))
    logger = AgentAuditLogger(log_path=log)
    entries = logger.read_tail(limit * 10 if tenant_id else limit)

    if tenant_id:
        entries = [
            e for e in entries
            if str(e.get("session_id", "")).startswith(tenant_id)
        ][:limit]
    else:
        entries = entries[:limit]

    return JSONResponse(content={
        "tenant_id": tenant_id,
        "count": len(entries),
        "log_path": str(logger.path),
        "entries": entries,
    })


# ── W58: CloudDB-backed read endpoints ───────────────────────────────────────


@app.get("/cloud/tenants/{tenant_id}/inventory")
async def cloud_get_tenant_inventory_db(tenant_id: str) -> JSONResponse:
    """Return inventory for *tenant_id* — reads from CloudDB when active.

    Unlike ``GET /cloud/inventory`` (resolved via JWT/header), this endpoint
    accepts the tenant_id explicitly in the URL path (suitable for admin
    dashboards and cross-tenant tooling).
    """
    if tenant_id not in _tenants:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")
    records = _db_read_inventory(tenant_id)
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "count": len(records),
        "models": records,
    })


@app.get("/cloud/tenants/{tenant_id}/vex-alerts")
async def cloud_get_tenant_vex_alerts_db(tenant_id: str) -> JSONResponse:
    """Return VEX alerts for *tenant_id* — reads from CloudDB when active."""
    if tenant_id not in _tenants:
        raise HTTPException(status_code=404, detail=f"Tenant not found: {tenant_id}")
    alerts = _db_read_vex_alerts(tenant_id)
    return JSONResponse(content={
        "tenant_id": tenant_id,
        "count": len(alerts),
        "alerts": alerts,
    })


@app.get("/cloud/policy-stats")
async def cloud_get_policy_stats() -> JSONResponse:
    """Return cross-tenant policy pass/fail aggregates — reads from CloudDB when active.

    Aggregates across ALL tenants (global fleet view).  Does not require
    tenant auth — intended for admin dashboards and SLA reporting.
    """
    stats = _db_read_policy_stats()
    return JSONResponse(content={
        "count": len(stats),
        "stats": stats,
    })


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


# ── On-demand drift check ─────────────────────────────────────────────────────


class DriftCheckRequest(BaseModel):
    """Request body for ``POST /drift-check``."""

    model_dir: str = Field(
        description="Absolute path (on the squash server) to the compressed model directory"
    )
    bom_path: str = Field(
        description="Absolute path (on the squash server) to the CycloneDX BOM JSON sidecar"
    )


@app.post("/drift-check")
async def post_drift_check(req: DriftCheckRequest) -> JSONResponse:
    """Compare BOM-attested SHA-256 digests against the model directory on disk.

    Drives CMMC / EU AI Act Art. 9 supply-chain integrity verification from
    CI/CD pipelines.  Both ``model_dir`` and ``bom_path`` must be absolute
    paths accessible to the squash server process.

    Returns
    -------
    JSON with keys ``ok`` (bool), ``files_checked`` (int), ``hits`` (list),
    and ``summary`` (str).  Each hit has ``path``, ``expected_digest``,
    ``actual_digest``, ``missing``, and ``tampered`` fields.
    """
    from squish.squash.drift import DriftConfig, DriftResult, check_drift

    model_path = Path(req.model_dir).resolve()
    bom = Path(req.bom_path).resolve()

    if not model_path.is_dir():
        raise HTTPException(status_code=400, detail=f"model_dir not found: {model_path}")
    if not bom.is_file():
        raise HTTPException(status_code=400, detail=f"bom_path not found: {bom}")

    try:
        result: DriftResult = await asyncio.get_event_loop().run_in_executor(
            None, check_drift, DriftConfig(bom_path=bom, model_dir=model_path)
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"I/O error reading drift inputs: {exc}")

    return JSONResponse(content={
        "ok": result.ok,
        "files_checked": result.files_checked,
        "summary": result.summary,
        "hits": [
            {
                "path": h.path,
                "expected_digest": h.expected_digest,
                "actual_digest": h.actual_digest,
                "missing": h.missing,
                "tampered": h.tampered,
            }
            for h in result.hits
        ],
    })
