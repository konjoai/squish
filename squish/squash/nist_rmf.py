"""squish/squash/nist_rmf.py — NIST AI Risk Management Framework (AI RMF 1.0) scanner.

Maps squash attestation artifacts to the four core functions of the NIST AI
Risk Management Framework (NIST AI 100-1, January 2023):

    GOVERN  — Policies, roles, accountability, and culture
    MAP     — Context, categorisation, and risk identification
    MEASURE — Evaluation, testing, and metrics
    MANAGE  — Response, monitoring, and incident handling

The scanner inspects squash-produced artifacts (SBOM, attestation, scan
reports, model cards, provenance manifests) against a curated set of
controls drawn from NIST AI RMF 1.0 subcategories and returns a
:class:`NistRmfReport` that summarises which controls pass, which fail, and
the overall posture.

Usage::

    from squish.squash.nist_rmf import NistRmfScanner
    from pathlib import Path

    report = NistRmfScanner.scan(Path("./my-model-dir"))
    print(report.posture)          # "COMPLIANT" / "PARTIAL" / "NON_COMPLIANT"
    print(report.summary())        # human-readable multi-line text

References
----------
* NIST AI 100-1 (2023) — https://doi.org/10.6028/NIST.AI.100-1
* NIST AI RMF Playbook — https://airc.nist.gov/Docs/1
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class NistRmfFunction(str, Enum):
    """The four core functions of the NIST AI RMF (AI 100-1 §3)."""

    GOVERN = "GOVERN"
    MAP = "MAP"
    MEASURE = "MEASURE"
    MANAGE = "MANAGE"


class NistControlStatus(str, Enum):
    """Evaluation result for a single NIST AI RMF control."""

    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "N/A"
    UNKNOWN = "UNKNOWN"


class NistRmfPosture(str, Enum):
    """Aggregate compliance posture for a NIST AI RMF evaluation."""

    COMPLIANT = "COMPLIANT"      # all applicable controls pass
    PARTIAL = "PARTIAL"          # ≥50% of applicable controls pass
    NON_COMPLIANT = "NON_COMPLIANT"  # <50% of applicable controls pass


# ---------------------------------------------------------------------------
# Control dataclass
# ---------------------------------------------------------------------------

@dataclass
class NistRmfControl:
    """A single NIST AI RMF control and its evaluation result.

    Attributes
    ----------
    control_id:
        Short identifier, e.g. ``"GOV-1.1"``.
    function:
        Which of the four RMF functions this belongs to.
    title:
        One-line description of the control.
    description:
        Extended description / rationale.
    status:
        Evaluation result (:class:`NistControlStatus`).
    evidence:
        List of artifact paths or fact strings that justify the status.
    recommendations:
        Actionable remediation steps when ``status == FAIL``.
    """

    control_id: str
    function: NistRmfFunction
    title: str
    description: str
    status: NistControlStatus = NistControlStatus.UNKNOWN
    evidence: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "function": self.function.value,
            "title": self.title,
            "status": self.status.value,
            "evidence": self.evidence,
            "recommendations": self.recommendations,
        }


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class NistRmfReport:
    """Full NIST AI RMF evaluation report for one model or tenant.

    Attributes
    ----------
    model_path:
        The directory that was evaluated (as a string).
    controls:
        Ordered list of all evaluated :class:`NistRmfControl` objects.
    posture:
        Overall compliance posture.
    pass_count:
        Number of controls that passed.
    fail_count:
        Number of controls that failed.
    na_count:
        Number of controls marked not-applicable.
    """

    model_path: str
    controls: list[NistRmfControl] = field(default_factory=list)
    posture: NistRmfPosture = field(default=NistRmfPosture.NON_COMPLIANT)
    pass_count: int = 0
    fail_count: int = 0
    na_count: int = 0

    def summary(self) -> str:
        """Return a human-readable multi-line summary."""
        lines = [
            f"NIST AI RMF 1.0 Report — {self.model_path}",
            f"Posture  : {self.posture.value}",
            f"Controls : {self.pass_count} PASS / {self.fail_count} FAIL"
            f" / {self.na_count} N/A",
            "",
        ]
        for fn in NistRmfFunction:
            section = [c for c in self.controls if c.function == fn]
            lines.append(f"  [{fn.value}]")
            for c in section:
                icon = "✓" if c.status == NistControlStatus.PASS else (
                    "✗" if c.status == NistControlStatus.FAIL else "—"
                )
                lines.append(f"    {icon} {c.control_id}: {c.title}")
                if c.status == NistControlStatus.FAIL and c.recommendations:
                    for r in c.recommendations:
                        lines.append(f"        → {r}")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_path": self.model_path,
            "posture": self.posture.value,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "na_count": self.na_count,
            "controls": [c.to_dict() for c in self.controls],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_json(directory: Path, *stems: str) -> dict[str, Any] | None:
    """Search for the first matching JSON file and parse it."""
    for stem in stems:
        for candidate in directory.glob(f"**/{stem}.json"):
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
        for candidate in directory.glob(f"**/{stem}.cdx.json"):
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                pass
    return None


def _find_file(directory: Path, *patterns: str) -> Path | None:
    """Return first matching path or None."""
    for pattern in patterns:
        matches = sorted(directory.glob(f"**/{pattern}"))
        if matches:
            return matches[0]
    return None


def _has_key_deep(obj: Any, *keys: str) -> bool:
    """Return True if all keys exist somewhere in the nested JSON structure."""
    if not isinstance(obj, dict):
        return False
    for key in keys:
        if key in obj:
            return True
        for v in obj.values():
            if _has_key_deep(v, key):
                return True
    return False


def _get_deep(obj: Any, *path: str) -> Any:
    """Navigate a dot-path; return None if any step is missing."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class NistRmfScanner:
    """Evaluate a model directory against NIST AI RMF 1.0 controls.

    The scanner looks for artefacts written by the squash attestation pipeline
    and maps findings to the 13 representative controls below.

    Controls implemented
    --------------------
    GOVERN: GOV-1.1, GOV-2.1, GOV-3.1, GOV-4.1
    MAP:    MAP-1.1, MAP-2.1, MAP-3.1
    MEASURE:MS-1.1, MS-2.1, MS-3.1, MS-4.1
    MANAGE: MG-1.1, MG-2.1

    All check methods are public so they can be unit-tested individually.
    """

    # ------------------------------------------------------------------ #
    # Public entry points                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def scan(model_path: Path) -> NistRmfReport:
        """Scan *model_path* and return a :class:`NistRmfReport`."""
        scanner = NistRmfScanner(model_path)
        return scanner._run()

    @staticmethod
    def from_dict(artifacts: dict[str, Any], model_path: str = "<in-memory>") -> NistRmfReport:
        """Run against a pre-loaded artifacts dict (useful for API callers).

        *artifacts* is a flat dict whose keys may include:

        ``sbom``, ``attestation``, ``scan``, ``model_card``,
        ``provenance``, ``policy``, ``vex``, ``drift``
        """
        scanner = NistRmfScanner.__new__(NistRmfScanner)
        scanner._path = Path(model_path)
        scanner._sbom = artifacts.get("sbom") or {}
        scanner._attest = artifacts.get("attestation") or {}
        scanner._scan = artifacts.get("scan") or {}
        scanner._card = artifacts.get("model_card") or {}
        scanner._provenance = artifacts.get("provenance") or {}
        scanner._policy = artifacts.get("policy") or {}
        scanner._vex = artifacts.get("vex") or {}
        scanner._drift = artifacts.get("drift") or {}
        return scanner._run()

    # ------------------------------------------------------------------ #
    # Initialisation                                                       #
    # ------------------------------------------------------------------ #

    def __init__(self, model_path: Path) -> None:
        self._path = model_path
        self._sbom = _find_json(model_path, "bom", "sbom", "mlbom") or {}
        self._attest = _find_json(model_path, "attestation", "attest") or {}
        self._scan = _find_json(model_path, "scan_result", "scan") or {}
        self._card = _find_json(model_path, "model_card", "modelcard") or {}
        self._provenance = _find_json(model_path, "provenance", "data_provenance") or {}
        self._policy = _find_json(model_path, "policy_result", "policy") or {}
        self._vex = _find_json(model_path, "vex", "openvex") or {}
        self._drift = _find_json(model_path, "drift_report", "drift") or {}

    # ------------------------------------------------------------------ #
    # Orchestration                                                         #
    # ------------------------------------------------------------------ #

    def _run(self) -> NistRmfReport:
        controls: list[NistRmfControl] = [
            self._gov_1_1(),
            self._gov_2_1(),
            self._gov_3_1(),
            self._gov_4_1(),
            self._map_1_1(),
            self._map_2_1(),
            self._map_3_1(),
            self._ms_1_1(),
            self._ms_2_1(),
            self._ms_3_1(),
            self._ms_4_1(),
            self._mg_1_1(),
            self._mg_2_1(),
        ]

        pass_count = sum(1 for c in controls if c.status == NistControlStatus.PASS)
        fail_count = sum(1 for c in controls if c.status == NistControlStatus.FAIL)
        na_count = sum(1 for c in controls if c.status == NistControlStatus.NOT_APPLICABLE)
        applicable = pass_count + fail_count

        if applicable == 0:
            posture = NistRmfPosture.NON_COMPLIANT
        elif fail_count == 0:
            posture = NistRmfPosture.COMPLIANT
        elif pass_count / applicable >= 0.5:
            posture = NistRmfPosture.PARTIAL
        else:
            posture = NistRmfPosture.NON_COMPLIANT

        return NistRmfReport(
            model_path=str(self._path),
            controls=controls,
            posture=posture,
            pass_count=pass_count,
            fail_count=fail_count,
            na_count=na_count,
        )

    # ------------------------------------------------------------------ #
    # GOVERN controls                                                       #
    # ------------------------------------------------------------------ #

    def _gov_1_1(self) -> NistRmfControl:
        """GOV-1.1 — AI risk policies, processes, and procedures are documented."""
        c = NistRmfControl(
            control_id="GOV-1.1",
            function=NistRmfFunction.GOVERN,
            title="AI risk policies and procedures are documented",
            description=(
                "The organisation has formal written policies that address AI risk "
                "management, including processes for model approval, review cadence, "
                "and escalation paths (NIST AI RMF §2.2)."
            ),
        )
        if self._policy and self._policy.get("template"):
            c.status = NistControlStatus.PASS
            c.evidence.append(f"policy template: {self._policy['template']}")
        elif self._policy and self._policy.get("status") in ("pass", "PASS", True):
            c.status = NistControlStatus.PASS
            c.evidence.append("policy evaluation recorded in policy artifact")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Run `squash policy evaluate` and commit the resulting "
                "policy_result.json alongside the model."
            )
        return c

    def _gov_2_1(self) -> NistRmfControl:
        """GOV-2.1 — Roles and responsibilities for AI risk are defined."""
        c = NistRmfControl(
            control_id="GOV-2.1",
            function=NistRmfFunction.GOVERN,
            title="AI risk roles and responsibilities are defined",
            description=(
                "SBOM supplier and author fields identify accountable parties. "
                "At minimum the model supplier must be named (NIST AI RMF §2.2 GOV-2)."
            ),
        )
        # Check SBOM metadata.supplier or component.supplier
        supplier = (
            _get_deep(self._sbom, "metadata", "supplier", "name")
            or _get_deep(self._sbom, "metadata", "authors")
            or _get_deep(self._sbom, "metadata", "component", "supplier", "name")
        )
        if supplier:
            c.status = NistControlStatus.PASS
            c.evidence.append(f"SBOM supplier/author: {supplier!r}")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Add a `supplier` field to the CycloneDX BOM metadata when running "
                "`squash attest` to identify the accountable party."
            )
        return c

    def _gov_3_1(self) -> NistRmfControl:
        """GOV-3.1 — Organisational accountability for AI outcomes is assigned."""
        c = NistRmfControl(
            control_id="GOV-3.1",
            function=NistRmfFunction.GOVERN,
            title="Organisational accountability is assigned for AI outcomes",
            description=(
                "A named attestation author or signing identity confirms executive "
                "accountability (NIST AI RMF §2.2 GOV-3)."
            ),
        )
        signer = (
            _get_deep(self._attest, "signer")
            or _get_deep(self._attest, "signed_by")
            or _get_deep(self._attest, "attestation", "signer")
        )
        if signer:
            c.status = NistControlStatus.PASS
            c.evidence.append(f"attestation signer: {signer!r}")
        elif self._attest and "digest" in str(self._attest):
            # Signed attestation exists even if field name varies
            c.status = NistControlStatus.PASS
            c.evidence.append("signed attestation artifact present")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Use `squash attest --sign` (OMS / Sigstore) to attach a "
                "verifiable signing identity to the model attestation."
            )
        return c

    def _gov_4_1(self) -> NistRmfControl:
        """GOV-4.1 — AI risk tolerance levels are established and communicated."""
        c = NistRmfControl(
            control_id="GOV-4.1",
            function=NistRmfFunction.GOVERN,
            title="AI risk tolerance levels are established",
            description=(
                "The model card or attestation includes declared intended use, "
                "out-of-scope uses, and known limitations, reflecting organisational "
                "risk appetite (NIST AI RMF §2.2 GOV-4)."
            ),
        )
        limitations = (
            _get_deep(self._card, "limitations")
            or _get_deep(self._card, "model_card", "limitations")
            or _get_deep(self._card, "intended_use")
            or _get_deep(self._card, "out_of_scope")
        )
        if limitations:
            c.status = NistControlStatus.PASS
            c.evidence.append("model card includes limitations/intended-use section")
        elif _has_key_deep(self._sbom, "limitations"):
            c.status = NistControlStatus.PASS
            c.evidence.append("SBOM component modelCard includes limitations field")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Generate a model card with `squash model-card` and include "
                "`limitations` and `intended_use` sections."
            )
        return c

    # ------------------------------------------------------------------ #
    # MAP controls                                                          #
    # ------------------------------------------------------------------ #

    def _map_1_1(self) -> NistRmfControl:
        """MAP-1.1 — Context for AI deployment is established."""
        c = NistRmfControl(
            control_id="MAP-1.1",
            function=NistRmfFunction.MAP,
            title="Deployment context and use-case are documented",
            description=(
                "The AI system's purpose, deployment environment, and target users "
                "are documented, enabling context-appropriate risk evaluation "
                "(NIST AI RMF MAP-1)."
            ),
        )
        context = (
            _get_deep(self._card, "model_details")
            or _get_deep(self._card, "description")
            or _get_deep(self._sbom, "metadata", "component", "description")
        )
        if context:
            c.status = NistControlStatus.PASS
            c.evidence.append("model card / SBOM component description present")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Add a `description` field to the SBOM component or generate a "
                "model card with `squash model-card`."
            )
        return c

    def _map_2_1(self) -> NistRmfControl:
        """MAP-2.1 — AI risk categories have been identified."""
        c = NistRmfControl(
            control_id="MAP-2.1",
            function=NistRmfFunction.MAP,
            title="AI risk categories are identified and documented",
            description=(
                "The system has been evaluated against at least one AI risk taxonomy "
                "(e.g. EU AI Act risk tiers, NIST AI RMF categories) and findings "
                "are recorded (NIST AI RMF MAP-2)."
            ),
        )
        # risk.py assess_nist_rmf or assess_eu_ai_act output
        risk_present = (
            _has_key_deep(self._attest, "nist_rmf")
            or _has_key_deep(self._attest, "eu_ai_act")
            or _has_key_deep(self._attest, "risk_assessment")
            or bool(self._policy)
        )
        if risk_present:
            c.status = NistControlStatus.PASS
            c.evidence.append("risk assessment or policy evaluation present in artifacts")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Run `squash risk-assess` or `squash policy evaluate` and embed "
                "the result in the attestation payload."
            )
        return c

    def _map_3_1(self) -> NistRmfControl:
        """MAP-3.1 — AI lifecycle stages are understood and documented."""
        c = NistRmfControl(
            control_id="MAP-3.1",
            function=NistRmfFunction.MAP,
            title="AI lifecycle stages are documented",
            description=(
                "Training data provenance, training parameters, and the model "
                "lineage are captured, enabling lifecycle stage tracking "
                "(NIST AI RMF MAP-3)."
            ),
        )
        has_lineage = bool(self._provenance) or _has_key_deep(self._sbom, "formulation")
        if has_lineage:
            c.status = NistControlStatus.PASS
            c.evidence.append("training provenance or formulation data present")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Run `squash provenance collect` to capture training data lineage "
                "and embed it in the SBOM `formulation` block."
            )
        return c

    # ------------------------------------------------------------------ #
    # MEASURE controls                                                      #
    # ------------------------------------------------------------------ #

    def _ms_1_1(self) -> NistRmfControl:
        """MS-1.1 — AI risk assessment methods are selected and applied."""
        c = NistRmfControl(
            control_id="MS-1.1",
            function=NistRmfFunction.MEASURE,
            title="AI risk assessment methods are applied",
            description=(
                "At least one formal risk assessment method (security scan, policy "
                "evaluation, or structured red-teaming) has been applied and its "
                "results recorded (NIST AI RMF MS-1)."
            ),
        )
        has_assessment = bool(self._scan) or bool(self._policy)
        if has_assessment:
            c.status = NistControlStatus.PASS
            if self._scan:
                c.evidence.append(
                    f"security scan status: {self._scan.get('status', 'present')}"
                )
            if self._policy:
                c.evidence.append("policy evaluation artifact present")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Run `squash scan` and `squash policy evaluate` before deploying "
                "the model and store the results in the model directory."
            )
        return c

    def _ms_2_1(self) -> NistRmfControl:
        """MS-2.1 — Quantitative evaluation metrics are defined and measured."""
        c = NistRmfControl(
            control_id="MS-2.1",
            function=NistRmfFunction.MEASURE,
            title="Quantitative accuracy/performance metrics are measured",
            description=(
                "The model has been evaluated on a benchmark dataset with reported "
                "numeric accuracy metrics (e.g. arc_easy, MMLU, lm_eval results) "
                "(NIST AI RMF MS-2)."
            ),
        )
        # Look for eval_delta or benchmark fields in SBOM / attestation / model card
        eval_present = (
            _has_key_deep(self._sbom, "eval_delta")
            or _has_key_deep(self._sbom, "lm_eval")
            or _has_key_deep(self._sbom, "accuracy")
            or _has_key_deep(self._attest, "eval_delta")
            or _has_key_deep(self._attest, "accuracy")
            or _has_key_deep(self._card, "metrics")
            or _has_key_deep(self._card, "results")
        )
        if eval_present:
            c.status = NistControlStatus.PASS
            c.evidence.append("quantitative evaluation metrics present in artifacts")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Bind lm_eval benchmark results to the SBOM with `squash eval-bind` "
                "before deploying the model."
            )
        return c

    def _ms_3_1(self) -> NistRmfControl:
        """MS-3.1 — Bias and fairness testing has been performed."""
        c = NistRmfControl(
            control_id="MS-3.1",
            function=NistRmfFunction.MEASURE,
            title="Bias and fairness evaluation has been performed",
            description=(
                "The model has been tested for demographic or representational bias "
                "and results are documented (NIST AI RMF MS-3)."
            ),
        )
        bias_present = (
            _has_key_deep(self._card, "bias")
            or _has_key_deep(self._card, "fairness")
            or _has_key_deep(self._card, "caveats")
            or _has_key_deep(self._attest, "bias_assessment")
        )
        if bias_present:
            c.status = NistControlStatus.PASS
            c.evidence.append("bias/fairness documentation present in model card")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Add a `bias_and_limitations` section to the model card via "
                "`squash model-card`, documenting known demographic biases."
            )
        return c

    def _ms_4_1(self) -> NistRmfControl:
        """MS-4.1 — Security and robustness testing has been performed."""
        c = NistRmfControl(
            control_id="MS-4.1",
            function=NistRmfFunction.MEASURE,
            title="Security scan and robustness testing performed",
            description=(
                "The model artifacts have been scanned for supply-chain threats "
                "(pickle injection, safetensors header tampering, GGUF ACE) and "
                "results are clean or documented (NIST AI RMF MS-4)."
            ),
        )
        if not self._scan:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Run `squash scan <model-dir>` to check for unsafe deserialization, "
                "GGUF ACE, and ONNX path traversal vulnerabilities."
            )
            return c

        status = self._scan.get("status", "")
        if status in ("safe", "SAFE", "clean", "pass", "PASS"):
            c.status = NistControlStatus.PASS
            c.evidence.append(f"scan status: {status!r}")
        elif status in ("unsafe", "UNSAFE"):
            c.status = NistControlStatus.FAIL
            findings = self._scan.get("findings", [])
            c.evidence.append(f"scan status: {status!r}; {len(findings)} finding(s)")
            c.recommendations.append(
                "Remediate all UNSAFE scan findings before deployment. "
                "Use `squash remediate` to convert pickle files to safetensors."
            )
        else:
            c.status = NistControlStatus.PASS
            c.evidence.append(f"scan artifact present (status={status!r})")
        return c

    # ------------------------------------------------------------------ #
    # MANAGE controls                                                       #
    # ------------------------------------------------------------------ #

    def _mg_1_1(self) -> NistRmfControl:
        """MG-1.1 — Risk responses are planned and documented."""
        c = NistRmfControl(
            control_id="MG-1.1",
            function=NistRmfFunction.MANAGE,
            title="Risk response plans are documented",
            description=(
                "A VEX document or remediation plan documents how known "
                "vulnerabilities and identified risks will be addressed "
                "(NIST AI RMF MG-1)."
            ),
        )
        has_vex = bool(self._vex)
        if has_vex:
            n_statements = len(self._vex.get("statements", []))
            c.status = NistControlStatus.PASS
            c.evidence.append(f"VEX document present with {n_statements} statement(s)")
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Generate a VEX document with `squash vex publish` to declare the "
                "exploitability status of known CVEs affecting this model."
            )
        return c

    def _mg_2_1(self) -> NistRmfControl:
        """MG-2.1 — Residual risks are monitored continuously."""
        c = NistRmfControl(
            control_id="MG-2.1",
            function=NistRmfFunction.MANAGE,
            title="Residual risks are monitored at runtime",
            description=(
                "A runtime drift monitor is configured to detect distribution shift "
                "and concept drift, enabling continuous residual risk assessment "
                "(NIST AI RMF MG-2)."
            ),
        )
        has_drift = bool(self._drift)
        if has_drift:
            drift_status = self._drift.get("status") or self._drift.get("drift_status")
            c.status = NistControlStatus.PASS
            c.evidence.append(
                f"drift monitoring report present"
                + (f" (status={drift_status!r})" if drift_status else "")
            )
        else:
            c.status = NistControlStatus.FAIL
            c.recommendations.append(
                "Enable the squash DriftMonitor middleware and publish drift "
                "reports with `squash drift report`."
            )
        return c
