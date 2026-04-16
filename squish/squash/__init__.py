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
from squish.squash.policy import NtiaResult, NtiaValidator  # noqa: F401 (Wave 20)
from squish.squash.slsa import SlsaLevel, SlsaAttestation, SlsaProvenanceBuilder  # noqa: F401 (Wave 21)
from squish.squash.sbom_builder import BomMerger  # noqa: F401 (Wave 22)
from squish.squash.risk import (  # noqa: F401 (Wave 23)
    RiskCategory,
    EuAiActCategory,
    NistRmfCategory,
    RiskAssessmentResult,
    AiRiskAssessor,
)
from squish.squash.governor import DriftEvent, DriftMonitor  # noqa: F401 (Wave 24)
from squish.squash.cicd import CiEnvironment, CicdAdapter, CicdReport  # noqa: F401 (Wave 25)
from squish.squash.integrations.sagemaker import SageMakerSquash  # noqa: F401 (Wave 26)
from squish.squash.sbom_builder import OrasAdapter  # noqa: F401 (Wave 26)
from squish.squash.vex import VexFeedManifest, SQUASH_VEX_FEED_URL, SQUASH_VEX_FEED_FALLBACK_URL  # noqa: F401 (Wave 26)
from squish.squash.integrations.ray import (  # noqa: F401 (Wave 28)
    SquashServeConfig,
    SquashServeDeployment,
    squash_serve,
)
from squish.squash.integrations.kubernetes import (  # noqa: F401 (Wave 27)
    KubernetesWebhookHandler,
    WebhookConfig,
)
from squish.squash.remediate import (  # noqa: F401 (Wave 54)
    Remediator,
    RemediateResult,
    ConvertedFile,
    FailedFile,
)
from squish.squash.evaluator import (  # noqa: F401 (Wave 55)
    EvalEngine,
    EvalReport,
    ProbeResult,
)
from squish.squash.edge_formats import (  # noqa: F401 (Wave 56)
    TFLiteParser,
    TFLiteMetadata,
    CoreMLParser,
    CoreMLMetadata,
    EdgeSecurityScanner,
    EdgeFinding,
    TensorDescriptor,
)
from squish.squash.chat import ChatSession  # noqa: F401 (Wave 56)
from squish.squash.model_card import (  # noqa: F401 (Wave 57)
    ModelCard,
    ModelCardConfig,
    ModelCardGenerator,
    ModelCardSection,
    KNOWN_FORMATS as MODEL_CARD_KNOWN_FORMATS,
)
from squish.squash.nist_rmf import (  # noqa: F401 (Wave 83)
    NistRmfFunction,
    NistControlStatus,
    NistRmfControl,
    NistRmfPosture,
    NistRmfReport,
    NistRmfScanner,
)

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
    # Wave 20: NTIA minimum elements
    "NtiaResult",
    "NtiaValidator",
    # Wave 21: SLSA provenance
    "SlsaLevel",
    "SlsaAttestation",
    "SlsaProvenanceBuilder",
    # Wave 22: BOM merge
    "BomMerger",
    # Wave 23: AI risk assessment
    "RiskCategory",
    "EuAiActCategory",
    "NistRmfCategory",
    "RiskAssessmentResult",
    "AiRiskAssessor",
    # Wave 24: Drift detection
    "DriftEvent",
    "DriftMonitor",
    # Wave 25: CI/CD integration
    "CiEnvironment",
    "CicdAdapter",
    "CicdReport",
    # Wave 26: SageMaker, ORAS, VEX feed
    "SageMakerSquash",
    "OrasAdapter",
    "VexFeedManifest",
    "SQUASH_VEX_FEED_URL",
    "SQUASH_VEX_FEED_FALLBACK_URL",
    # Wave 27: Kubernetes Admission Webhook
    "KubernetesWebhookHandler",
    "WebhookConfig",
    # Wave 28: Ray Serve decorator
    "SquashServeConfig",
    "SquashServeDeployment",
    "squash_serve",
    # Wave 54: Remediate (pickle → safetensors)
    "Remediator",
    "RemediateResult",
    "ConvertedFile",
    "FailedFile",
    # Wave 55: Dynamic evaluation / red-teaming
    "EvalEngine",
    "EvalReport",
    "ProbeResult",
    # Wave 56: Edge AI format support
    "TFLiteParser",
    "TFLiteMetadata",
    "CoreMLParser",
    "CoreMLMetadata",
    "EdgeSecurityScanner",
    "EdgeFinding",
    "TensorDescriptor",
    # Wave 56: RAG compliance chat
    "ChatSession",
    # Wave 57: Model card generator
    "ModelCard",
    "ModelCardConfig",
    "ModelCardGenerator",
    "ModelCardSection",
    "MODEL_CARD_KNOWN_FORMATS",
    # Wave 83: NIST AI RMF 1.0 controls scanner
    "NistRmfFunction",
    "NistControlStatus",
    "NistRmfControl",
    "NistRmfPosture",
    "NistRmfReport",
    "NistRmfScanner",
]
