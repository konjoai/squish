"""risk.py — AI risk assessment aligned with EU AI Act and NIST AI RMF (Wave 23).

Usage::

    from squish.squash.risk import AiRiskAssessor

    result_eu  = AiRiskAssessor.assess_eu_ai_act(Path("my-model/cyclonedx-mlbom.json"))
    result_rmf = AiRiskAssessor.assess_nist_rmf(Path("my-model/cyclonedx-mlbom.json"))

The assessment logic is heuristics-based, derived from the `ML BOM
<https://cyclonedx.org/capabilities/mlbom/>`_ ``modelCard`` structure embedded
in the CycloneDX 1.5 BOM.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ── EU AI Act risk categories ──────────────────────────────────────────────────

class EuAiActCategory(Enum):
    """EU AI Act risk tiers (Regulation 2024/1689)."""

    UNACCEPTABLE = "unacceptable"  # Art. 5 — prohibited practices
    HIGH = "high"                  # Annex III
    LIMITED = "limited"            # Transparency obligations
    MINIMAL = "minimal"            # All other cases


class NistRmfCategory(Enum):
    """NIST AI RMF (NIST AI 100-1) risk levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


# Keep backward-compatible generic alias
RiskCategory = EuAiActCategory


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class RiskAssessmentResult:
    """Result of a single framework risk assessment.

    Attributes
    ----------
    framework:
        ``"eu-ai-act"`` or ``"nist-rmf"``.
    category:
        Enum value from :class:`EuAiActCategory` or :class:`NistRmfCategory`.
    rationale:
        List of human-readable strings explaining why each signal was raised.
    mitigation_required:
        List of recommended mitigations.
    passed:
        ``True`` if the risk level is acceptable (MINIMAL / LOW), else ``False``.
    """

    framework: str
    category: EuAiActCategory | NistRmfCategory
    rationale: list[str] = field(default_factory=list)
    mitigation_required: list[str] = field(default_factory=list)
    passed: bool = True


# ── Risk signals ───────────────────────────────────────────────────────────────

_EU_HIGH_RISK_USE_CASES = {
    "biometric identification",
    "critical infrastructure",
    "education",
    "employment",
    "migration",
    "administration of justice",
    "law enforcement",
    "border control",
    "essential services",
}

_EU_PROHIBITED_SIGNALS = {
    "subliminal manipulation",
    "exploitation of vulnerabilities",
    "social scoring",
    "real-time biometric surveillance",
}

_HIGH_SENSITIVITY_SIGNALS = {
    "health",
    "medical",
    "finance",
    "financial",
    "criminal",
    "surveillance",
    "automated decision",
    "autonomous",
    "facial recognition",
}


def _load_bom(bom_path: Path) -> dict:
    """Load and return a CycloneDX BOM dict, or return an empty dict."""
    p = Path(bom_path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _collect_signals(bom: dict) -> dict[str, list[str]]:
    """Extract risk signals from a BOM.

    Returns a dict with keys:
    ``use_cases`` — list of use-case strings,
    ``sensitivity_flags`` — list of sensitivity strings,
    ``has_provenance`` — bool as str,
    ``component_count`` — str.
    """
    model_card: dict = {}
    for comp in bom.get("components", []):
        mc = comp.get("modelCard", {})
        if mc:
            model_card = mc
            break

    considerations: dict = model_card.get("considerations", {})
    use_cases: list[str] = [
        uc.get("description", uc) if isinstance(uc, dict) else str(uc)
        for uc in considerations.get("useCases", [])
    ]
    sensitivity_flags: list[str] = [
        str(f) for f in considerations.get("fairnessAssessments", [])
    ]
    # Also collect from properties
    for prop in bom.get("metadata", {}).get("properties", []):
        name = str(prop.get("name", "")).lower()
        val = str(prop.get("value", ""))
        if "sensitivity" in name or "use_case" in name or "usecase" in name:
            use_cases.append(val)

    has_provenance = any(
        ref.get("type") == "build-meta"
        for ref in bom.get("externalReferences", [])
    )
    component_count = len(bom.get("components", []))

    return {
        "use_cases": [u.lower() for u in use_cases if u],
        "sensitivity_flags": [f.lower() for f in sensitivity_flags if f],
        "has_provenance": str(has_provenance),
        "component_count": str(component_count),
    }


# ── Assessor class ─────────────────────────────────────────────────────────────

class AiRiskAssessor:
    """Heuristic AI risk assessor for EU AI Act and NIST AI RMF.

    All methods are static; no instantiation needed.
    """

    @staticmethod
    def assess_eu_ai_act(bom_path: Path) -> RiskAssessmentResult:
        """Run EU AI Act risk assessment against *bom_path*.

        Returns
        -------
        RiskAssessmentResult
            ``framework == "eu-ai-act"`` with the appropriate
            :class:`EuAiActCategory`.
        """
        bom = _load_bom(bom_path)
        signals = _collect_signals(bom)
        use_cases = signals["use_cases"]
        rationale: list[str] = []
        mitigations: list[str] = []

        # Check prohibited practices (Art. 5)
        for uc in use_cases:
            for sig in _EU_PROHIBITED_SIGNALS:
                if sig in uc:
                    rationale.append(f"Prohibited practice detected in use-case: '{sig}'")
                    mitigations.append("Remove or redesign the prohibited capability")
                    return RiskAssessmentResult(
                        framework="eu-ai-act",
                        category=EuAiActCategory.UNACCEPTABLE,
                        rationale=rationale,
                        mitigation_required=mitigations,
                        passed=False,
                    )

        # Check Annex III high-risk use cases
        high_risk_found = False
        for uc in use_cases:
            for sig in _EU_HIGH_RISK_USE_CASES:
                if sig in uc:
                    high_risk_found = True
                    rationale.append(f"High-risk domain detected: '{sig}'")
                    mitigations.append(
                        "Implement conformity assessment per EU AI Act Art. 43"
                    )
                    mitigations.append("Register in EU database per Art. 71")

        has_sensitivity = any(
            sig in flags
            for flags in signals["sensitivity_flags"]
            for sig in _HIGH_SENSITIVITY_SIGNALS
        )
        for uc in use_cases:
            for sig in _HIGH_SENSITIVITY_SIGNALS:
                if sig in uc:
                    has_sensitivity = True

        if high_risk_found:
            mitigations.append("Ensure technical documentation is complete and current")
            return RiskAssessmentResult(
                framework="eu-ai-act",
                category=EuAiActCategory.HIGH,
                rationale=rationale,
                mitigation_required=mitigations,
                passed=False,
            )

        if has_sensitivity:
            rationale.append("Sensitive domain signals detected — limited risk")
            mitigations.append(
                "Provide transparency notice per EU AI Act Art. 52"
            )
            return RiskAssessmentResult(
                framework="eu-ai-act",
                category=EuAiActCategory.LIMITED,
                rationale=rationale,
                mitigation_required=mitigations,
                passed=True,
            )

        rationale.append("No high-risk or prohibited-use signals detected")
        return RiskAssessmentResult(
            framework="eu-ai-act",
            category=EuAiActCategory.MINIMAL,
            rationale=rationale,
            mitigation_required=[],
            passed=True,
        )

    @staticmethod
    def assess_nist_rmf(bom_path: Path) -> RiskAssessmentResult:
        """Run NIST AI RMF risk assessment against *bom_path*.

        Returns
        -------
        RiskAssessmentResult
            ``framework == "nist-rmf"`` with a :class:`NistRmfCategory`.
        """
        bom = _load_bom(bom_path)
        signals = _collect_signals(bom)
        use_cases = signals["use_cases"]
        has_provenance = signals["has_provenance"] == "True"
        component_count = int(signals["component_count"])
        rationale: list[str] = []
        mitigations: list[str] = []

        # Critical: prohibited or autonomous decision-making
        for uc in use_cases:
            if "autonomous" in uc or "automated decision" in uc:
                rationale.append(
                    f"Autonomous / automated decision signal: '{uc[:60]}'"
                )
                mitigations.append(
                    "Implement human oversight per NIST AI RMF GOVERN 1.1"
                )

        critical_signals = any("autonomous" in uc for uc in use_cases)
        if critical_signals and not has_provenance:
            rationale.append("No provenance attestation found for autonomous system")
            mitigations.append("Attach SLSA provenance (squash slsa-attest)")
            return RiskAssessmentResult(
                framework="nist-rmf",
                category=NistRmfCategory.CRITICAL,
                rationale=rationale,
                mitigation_required=mitigations,
                passed=False,
            )

        # High: sensitive domains with no provenance
        has_sensitivity = any(
            sig in uc for uc in use_cases for sig in _HIGH_SENSITIVITY_SIGNALS
        )
        if has_sensitivity and not has_provenance:
            rationale.append("Sensitive domain without provenance attestation")
            mitigations.append("Attach SLSA provenance (squash slsa-attest)")
            mitigations.append("Implement MAP 1.6 measurement plan")
            return RiskAssessmentResult(
                framework="nist-rmf",
                category=NistRmfCategory.HIGH,
                rationale=rationale,
                mitigation_required=mitigations,
                passed=False,
            )

        # Moderate: many components or no SBOM
        if component_count == 0:
            rationale.append("BOM contains no components")
            mitigations.append("Generate full SBOM with squash scan")
            return RiskAssessmentResult(
                framework="nist-rmf",
                category=NistRmfCategory.MODERATE,
                rationale=rationale,
                mitigation_required=mitigations,
                passed=True,
            )

        if component_count > 50:
            rationale.append(
                f"Large component graph ({component_count} components) increases supply chain risk"
            )
            mitigations.append("Pin all dependency digests and enable drift monitoring")

        if rationale:
            return RiskAssessmentResult(
                framework="nist-rmf",
                category=NistRmfCategory.MODERATE,
                rationale=rationale,
                mitigation_required=mitigations,
                passed=True,
            )

        rationale.append("No elevated risk signals detected")
        return RiskAssessmentResult(
            framework="nist-rmf",
            category=NistRmfCategory.LOW,
            rationale=rationale,
            mitigation_required=[],
            passed=True,
        )


# ── W81 — Remediation plan generator ──────────────────────────────────────────

@dataclass
class RemediationStep:
    """A single actionable remediation step for a model at risk.

    Attributes
    ----------
    id:
        Unique slug identifying the step (e.g. ``"obtain_attestation"``).
    priority:
        1 = critical (blocks compliance), 2 = high, 3 = medium.
    action:
        Short imperative label for the step.
    description:
        Human-readable explanation of what to do and why.
    evidence_required:
        What artefact or proof satisfies this step.
    estimated_effort:
        Rough estimate: ``"1d"``, ``"1w"``, ``"2w"``.
    """

    id: str
    priority: int
    action: str
    description: str
    evidence_required: str
    estimated_effort: str


def generate_remediation_plan(
    risk_tier: str,
    policy_results: dict,
    open_vex: int,
    attestation_passed: bool = True,
) -> list[RemediationStep]:
    """Generate a prioritised remediation plan for a model.

    Rules (evaluated in order; a step is added once regardless of how many
    triggers fire for the same ``id``):

    * ``attestation_failed`` — priority 1 when ``not attestation_passed``
    * ``remediate_policy_failures`` — priority 1 when policy failure rate > 50 %
    * ``close_vex_alerts`` — priority 2 when ``open_vex > 0``
    * ``sign_model_artifact`` — priority 2 when no signature evidence found in
      policy_results (key contains "sign")

    UNACCEPTABLE tiers always surface all critical steps; MINIMAL returns an
    empty list.

    Parameters
    ----------
    risk_tier:
        One of ``"UNACCEPTABLE"``, ``"HIGH"``, ``"LIMITED"``, ``"MINIMAL"``.
    policy_results:
        Mapping of policy-name → ``{"passed": bool, "error_count": int, ...}``.
    open_vex:
        Number of open VEX alert records for this model.
    attestation_passed:
        Whether the model's attestation check passed.

    Returns
    -------
    list[RemediationStep]
        Steps sorted by ``priority`` ascending (most critical first).
    """
    if risk_tier == "MINIMAL" and attestation_passed and open_vex == 0:
        return []

    steps: dict[str, RemediationStep] = {}

    # ── Priority 1: attestation failure ──────────────────────────────────────
    if not attestation_passed:
        steps["obtain_attestation"] = RemediationStep(
            id="obtain_attestation",
            priority=1,
            action="Obtain attestation",
            description=(
                "The model has not passed attestation.  Run `squash attest` "
                "to generate a signed attestation record and store it alongside "
                "the model artefact."
            ),
            evidence_required="Signed attestation record (squash-attestation.json)",
            estimated_effort="1d",
        )

    # ── Priority 1: high policy failure rate ─────────────────────────────────
    if policy_results:
        total = len(policy_results)
        failed = sum(
            1 for v in policy_results.values() if not v.get("passed", True)
        )
        if total > 0 and (failed / total) > 0.5:
            steps["remediate_policy_failures"] = RemediationStep(
                id="remediate_policy_failures",
                priority=1,
                action="Remediate policy failures",
                description=(
                    f"{failed}/{total} policy checks are failing.  Review each "
                    "failing policy with `squash scan --policy <name>` and address "
                    "the reported violations before re-running attestation."
                ),
                evidence_required=(
                    "All policy checks passing (error_count == 0 for each policy)"
                ),
                estimated_effort="1w",
            )

    # ── Priority 2: open VEX alerts ───────────────────────────────────────────
    if open_vex > 0:
        steps["close_vex_alerts"] = RemediationStep(
            id="close_vex_alerts",
            priority=2,
            action="Close open VEX alerts",
            description=(
                f"{open_vex} open VEX alert(s) exist for this model.  Triage "
                "each CVE using `squash vex`, apply patches or document "
                "accepted risk, and mark alerts resolved."
            ),
            evidence_required=(
                "All VEX alerts in 'resolved', 'not_affected', or 'fixed' state"
            ),
            estimated_effort="1w",
        )

    # ── Priority 2: no signing evidence ──────────────────────────────────────
    has_signing_policy = any(
        "sign" in k.lower() for k in (policy_results or {})
    )
    if not has_signing_policy and policy_results is not None and len(policy_results) > 0:
        steps["sign_model_artifact"] = RemediationStep(
            id="sign_model_artifact",
            priority=2,
            action="Sign model artefact",
            description=(
                "No signing policy check was found in the policy results.  "
                "Add a signature policy and run `squash attest --sign` to "
                "produce a verifiable artefact signature."
            ),
            evidence_required="Cosign or SLSA signature present and verifiable",
            estimated_effort="1d",
        )

    return sorted(steps.values(), key=lambda s: s.priority)
