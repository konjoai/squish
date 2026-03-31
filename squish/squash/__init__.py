"""squish.squash — AI-SBOM governance layer (Squish Squash).

Phase 1: CycloneDX 1.7 ML-BOM sidecar written at compress time.
Phase 2: lm_eval delta binding + OpenSSF Model Signing (eval_binder, oms_signer).
Phase 3: Runtime governor middleware + /sbom and /health/model endpoints (governor).

Install the optional dependency group to enable:
    pip install "squish[squash]"
"""

from squish.squash.sbom_builder import CompressRunMeta, CycloneDXBuilder
from squish.squash.eval_binder import EvalBinder
from squish.squash.oms_signer import OmsSigner
from squish.squash.governor import SquashGovernor

__all__ = [
    "CycloneDXBuilder",
    "CompressRunMeta",
    "EvalBinder",
    "OmsSigner",
    "SquashGovernor",
]
