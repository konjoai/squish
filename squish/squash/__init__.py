"""squish.squash — AI-SBOM governance layer (Squish Squash).

Phase 1: CycloneDX 1.7 ML-BOM sidecar written at compress time.
Phase 2: lm_eval delta binding + OpenSSF Model Signing (eval_binder, oms_signer).
Phase 3: Runtime governor middleware + /sbom and /health/model endpoints (governor).
Phase 7: Standalone attestation engine — SPDX 2.3 dual output, policy templates,
         security scanner, VEX engine, training data provenance, REST API,
         MLOps integrations (MLflow, W&B, HF Hub, LangChain), CI/CD adapters.

Install the optional dependency group to enable:
    pip install "squish[squash]"

For the REST microservice:
    pip install "squish[squash-api]"
    uvicorn squish.squash.api:app --host 0.0.0.0 --port 4444
"""

from squish.squash.sbom_builder import CompressRunMeta, CycloneDXBuilder, SbomDiff, SbomRegistry, EvalBinder
from squish.squash.oms_signer import OmsSigner, OmsVerifier
from squish.squash.governor import SquashGovernor

# Phase 7 exports — lazy-guarded; raise ImportError at access time if cyclonedx absent
from squish.squash.spdx_builder import SpdxArtifacts, SpdxBuilder, SpdxOptions
from squish.squash.policy import (
    AVAILABLE_POLICIES,
    PolicyEngine,
    PolicyFinding,
    PolicyHistory,
    PolicyResult,
    PolicyRegistry,
    PolicyWebhook,
)
from squish.squash.scanner import ModelScanner, ScanFinding, ScanResult
from squish.squash.vex import (
    ModelInventory,
    ModelInventoryEntry,
    VexCache,
    VexDocument,
    VexEvaluator,
    VexFeed,
    VexReport,
    VexStatement,
)
from squish.squash.provenance import (
    DatasetRecord,
    ProvenanceCollector,
    ProvenanceManifest,
)
from squish.squash.attest import (
    AttestConfig,
    AttestPipeline,
    AttestResult,
    AttestationViolationError,
    CompositeAttestConfig,
    CompositeAttestPipeline,
    CompositeAttestResult,
)
from squish.squash.sarif import SarifBuilder
from squish.squash.report import ComplianceReporter

__all__ = [
    # Phase 1–3 (existing)
    "CycloneDXBuilder",
    "CompressRunMeta",
    "EvalBinder",
    "OmsSigner",
    "SquashGovernor",
    # Phase 7: SPDX
    "SpdxBuilder",
    "SpdxOptions",
    "SpdxArtifacts",
    # Phase 7: Policy
    "PolicyEngine",
    "PolicyResult",
    "PolicyFinding",
    "AVAILABLE_POLICIES",
    # Phase 7: Scanner
    "ModelScanner",
    "ScanResult",
    "ScanFinding",
    # Phase 7: VEX
    "VexFeed",
    "VexEvaluator",
    "VexReport",
    "ModelInventory",
    "ModelInventoryEntry",
    "VexDocument",
    "VexStatement",
    # Phase 7: Provenance
    "ProvenanceCollector",
    "ProvenanceManifest",
    "DatasetRecord",
    # Phase 7: Attestation pipeline
    "AttestPipeline",
    "AttestConfig",
    "AttestResult",
    "AttestationViolationError",
    # Wave 11: SARIF export
    "SarifBuilder",
    # Wave 12: SBOM diff + policy history
    "SbomDiff",
    "PolicyHistory",
    # Wave 14: Sigstore verify
    "OmsVerifier",
    # Wave 15: HTML compliance report
    "ComplianceReporter",
    # Wave 16: VEX cache
    "VexCache",
    # Wave 17: Policy webhooks
    "PolicyWebhook",
    # Wave 18: Composite attestation
    "CompositeAttestConfig",
    "CompositeAttestPipeline",
    "CompositeAttestResult",
    # Wave 19: SBOM registry push
    "SbomRegistry",
]
