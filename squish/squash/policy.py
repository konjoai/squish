"""squish/squash/policy.py — Built-in compliance policy templates.

Policy templates ship embedded in the squish package — zero external config
required to get named framework alignment out of the box:

    squash attest --model ./model.gguf --policy eu-ai-act
    squash attest --model ./model.gguf --policy nist-ai-rmf
    squash attest --model ./model.gguf --policy owasp-llm-top10
    squash attest --model ./model.gguf --policy iso-42001
    squash attest --model ./model.gguf --policy enterprise-strict

Each policy is a dict of named *rules*, each with:
  - ``field``:    dot-path into the SBOM document to check (CycloneDX or SPDX)
  - ``check``:    one of ``present | non_empty | equals | min_value | not_equals``
  - ``value``:    expected value (for ``equals`` / ``not_equals`` / ``min_value``)
  - ``severity``: ``error`` (hard fail) | ``warning`` (soft warn)
  - ``rationale``: human-readable explanation linking back to the framework
  - ``remediation``: short remediation hint

:class:`PolicyEngine` evaluates a policy dict against a raw SBOM dict and
returns a :class:`PolicyResult` with pass/fail/warning counts and individual
finding objects.

Policies are intentionally kept as pure Python dicts — no YAML parser needed
at runtime.  The ``enterprise-strict`` policy is the opinionated superset used
as the default for Enterprise customers.
"""

from __future__ import annotations

import logging
import re
import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


# ── Policy definitions ───────────────────────────────────────────────────────


_POLICIES: dict[str, list[dict[str, Any]]] = {
    # ── EU AI Act (risk-based, high-risk AI systems) ─────────────────────
    "eu-ai-act": [
        {
            "id": "EU-AIA-001",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU AI Act Art. 11: Technical documentation must identify the AI system.",
            "remediation": "Ensure model_id is set in CompressRunMeta.",
        },
        {
            "id": "EU-AIA-002",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU AI Act Art. 12: Logs and audit trail require artifact integrity hashes.",
            "remediation": "Run squish compress on a non-empty model directory.",
        },
        {
            "id": "EU-AIA-003",
            "field": "components[0].modelCard.modelParameters.quantizationLevel",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU AI Act Annex IV: Technical docs must describe system capabilities and limitations.",
            "remediation": "Ensure quant_format is set in CompressRunMeta.",
        },
        {
            "id": "EU-AIA-004",
            "field": "metadata.timestamp",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU AI Act Art. 12: Audit records require timestamps.",
            "remediation": "SBOM is generated automatically at compress time — never override timestamp.",
        },
        {
            "id": "EU-AIA-005",
            "field": "components[0].externalReferences",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "EU AI Act Art. 13: Transparency requires traceability back to the upstream model.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta.",
        },
        {
            "id": "EU-AIA-006",
            "field": "components[0].pedigree.ancestors",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "EU AI Act Art. 17: Quality management requires provenance chain documentation.",
            "remediation": "Ensure hf_mlx_repo maps to a real upstream model.",
        },
    ],
    # ── NIST AI Risk Management Framework (AI RMF 1.0) ──────────────────
    "nist-ai-rmf": [
        {
            "id": "NIST-RMF-001",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "NIST AI RMF MAP-1.1: AI systems must be clearly identified and scoped.",
            "remediation": "Set model_id in CompressRunMeta.",
        },
        {
            "id": "NIST-RMF-002",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "NIST AI RMF MEASURE-2.5: Integrity of AI artifacts must be verifiable.",
            "remediation": "Ensure model directory contains weight files before attesting.",
        },
        {
            "id": "NIST-RMF-003",
            "field": "components[0].modelCard.modelParameters.architectureFamily",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "NIST AI RMF MAP-1.5: Model characteristics must be documented.",
            "remediation": "Ensure detect_model_family() resolves for this model.",
        },
        {
            "id": "NIST-RMF-004",
            "field": "metadata.tools",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "NIST AI RMF GOVERN-1.3: Provenance of toolchain must be captured.",
            "remediation": "SBOM metadata.tools is populated automatically — do not strip it.",
        },
        {
            "id": "NIST-RMF-005",
            "field": "components[0].modelCard.quantitativeAnalysis.performanceMetrics",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "NIST AI RMF MEASURE-2.3: Quantitative performance evidence must be documented.",
            "remediation": "Run squish eval after compress to bind lm_eval scores.",
        },
    ],
    # ── OWASP LLM Top 10 (2025 edition) ──────────────────────────────────
    "owasp-llm-top10": [
        {
            "id": "OWASP-LLM-001",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "OWASP LLM01 Prompt Injection / LLM03 Supply Chain: Artifact hashes detect tampered weights.",
            "remediation": "All model weight files must be hashed at compress time.",
        },
        {
            "id": "OWASP-LLM-002",
            "field": "squash:scan_result",
            "check": "equals",
            "value": "clean",
            "severity": "error",
            "rationale": "OWASP LLM03 Supply Chain: Models must be scanned for pickle exploits and ACE payloads.",
            "remediation": "Run squash scan before attestation. Fix or reject flagged models.",
        },
        {
            "id": "OWASP-LLM-003",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "OWASP LLM09 Overreliance: Model identity must be auditable.",
            "remediation": "Set model_id in CompressRunMeta.",
        },
        {
            "id": "OWASP-LLM-004",
            "field": "metadata.timestamp",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "OWASP LLM03 Supply Chain: Attestation timestamp provides temporal proof-of-compliance.",
            "remediation": "SBOM timestamp is auto-generated — do not strip it.",
        },
    ],
    # ── ISO/IEC 42001:2023 AI Management System ───────────────────────────
    "iso-42001": [
        {
            "id": "ISO-42001-001",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "ISO 42001 §8.4: AI system documentation must uniquely identify the system.",
            "remediation": "Set model_id in CompressRunMeta.",
        },
        {
            "id": "ISO-42001-002",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "ISO 42001 §8.5: Integrity verification requires cryptographic artifact binding.",
            "remediation": "Ensure model directory contains weight files before attesting.",
        },
        {
            "id": "ISO-42001-003",
            "field": "components[0].purl",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "ISO 42001 §8.7: Supply chain documentation requires source provenance.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta.",
        },
        {
            "id": "ISO-42001-004",
            "field": "components[0].modelCard.modelParameters",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "ISO 42001 §9.1: Performance monitoring requires documented model parameters.",
            "remediation": "Ensure model card fields are populated by the compress pipeline.",
        },
    ],
    # ── Enterprise Strict (opinionated superset) ──────────────────────────
    "enterprise-strict": [
        # Inherits all EU AI Act + NIST + OWASP errors
        {
            "id": "ENT-001",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "Enterprise policy: all attested artifacts must be identifiable.",
            "remediation": "Set model_id.",
        },
        {
            "id": "ENT-002",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "Enterprise policy: integrity hashes required on all production models.",
            "remediation": "Ensure weight files exist and are hashed.",
        },
        {
            "id": "ENT-003",
            "field": "squash:scan_result",
            "check": "equals",
            "value": "clean",
            "severity": "error",
            "rationale": "Enterprise policy: security scan must pass before deployment.",
            "remediation": "Run squash scan. Reject models with pickle exploits or ACE payloads.",
        },
        {
            "id": "ENT-004",
            "field": "components[0].modelCard.quantitativeAnalysis.performanceMetrics",
            "check": "non_empty",
            "severity": "error",
            "rationale": "Enterprise policy: performance must be validated before production.",
            "remediation": "Run squish eval to bind lm_eval scores.",
        },
        {
            "id": "ENT-005",
            "field": "components[0].pedigree.ancestors",
            "check": "non_empty",
            "severity": "error",
            "rationale": "Enterprise policy: provenance chain required for all production models.",
            "remediation": "Set hf_mlx_repo to the upstream model.",
        },
        {
            "id": "ENT-006",
            "field": "components[0].purl",
            "check": "non_empty",
            "severity": "error",
            "rationale": "Enterprise policy: PURL required for software composition analysis.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta.",
        },
        {
            "id": "ENT-007",
            "field": "metadata.timestamp",
            "check": "non_empty",
            "severity": "error",
            "rationale": "Enterprise policy: audit timestamp required on all records.",
            "remediation": "SBOM timestamp is auto-generated.",
        },
        {
            "id": "ENT-008",
            "field": "components[0].modelCard.modelParameters.architectureFamily",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "Enterprise policy: architecture family improves incident response triage.",
            "remediation": "Ensure detect_model_family() resolves for this model.",
        },
    ],
    # ── EU Cyber Resilience Act (effective September 2026) ────────────────
    #
    # The CRA imposes mandatory cybersecurity requirements on products with
    # digital elements placed on the EU market.  AI models distributed as
    # software components fall under its scope.  Key obligations for ML
    # artifacts map to squish's SBOM + attestation pipeline as follows.
    "eu-cra": [
        {
            "id": "CRA-001",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU CRA Art. 13 §3: Technical documentation must uniquely identify the product with digital elements.",
            "remediation": "Set model_id in CompressRunMeta.",
        },
        {
            "id": "CRA-002",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU CRA Annex I Part II §2: SBOM must include components with cryptographic artifact integrity verification.",
            "remediation": "Ensure model directory contains weight files before attesting.",
        },
        {
            "id": "CRA-003",
            "field": "squash:scan_result",
            "check": "equals",
            "value": "clean",
            "severity": "error",
            "rationale": "EU CRA Annex I Part I §2: Products must be delivered without known exploitable vulnerabilities.",
            "remediation": "Run squash scan before attestation. Reject models with pickle exploits or ACE payloads.",
        },
        {
            "id": "CRA-004",
            "field": "components[0].purl",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU CRA Annex I Part II §2: SBOM must include package identifiers for all third-party components.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta to populate the package URL.",
        },
        {
            "id": "CRA-005",
            "field": "components[0].pedigree.ancestors",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU CRA Art. 13 §3(a): Technical documentation must trace the product's upstream supply chain.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta.",
        },
        {
            "id": "CRA-006",
            "field": "metadata.timestamp",
            "check": "non_empty",
            "severity": "error",
            "rationale": "EU CRA Art. 14 §1: Vulnerability handling obligations require a dated audit record for each release.",
            "remediation": "SBOM timestamp is auto-generated at compress time — do not strip it.",
        },
        {
            "id": "CRA-007",
            "field": "components[0].modelCard.modelParameters.quantizationLevel",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "EU CRA Annex I Part I §3: Products must minimise the attack surface — active capabilities must be documented.",
            "remediation": "Ensure quant_format is set in CompressRunMeta.",
        },
        {
            "id": "CRA-008",
            "field": "metadata.tools",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "EU CRA Art. 13 §3(b): Technical documentation must describe the development and manufacturing process.",
            "remediation": "SBOM metadata.tools is populated automatically — do not strip it.",
        },
    ],
    # ── FedRAMP Moderate / CMMC Level 2 (NIST SP 800-53 / 800-171) ───────
    #
    # FedRAMP Moderate authorisation requires compliance with NIST SP 800-53
    # Rev 5 controls.  AI model components must satisfy the subset of controls
    # that apply to software supply chain integrity, configuration management,
    # vulnerability management, and audit logging.  CMMC Level 2 maps to the
    # same NIST 800-171 control families for organisations handling CUI.
    "fedramp": [
        {
            "id": "FEDRAMP-CM-8",
            "field": "components[0].name",
            "check": "non_empty",
            "severity": "error",
            "rationale": "FedRAMP CM-8: System component inventory requires unique identification of each component.",
            "remediation": "Set model_id in CompressRunMeta.",
        },
        {
            "id": "FEDRAMP-SI-7",
            "field": "components[0].hashes",
            "check": "non_empty",
            "severity": "error",
            "rationale": "FedRAMP SI-7: Software integrity verification requires cryptographic hashes on all system components.",
            "remediation": "Ensure model directory contains weight files before attesting.",
        },
        {
            "id": "FEDRAMP-RA-5",
            "field": "squash:scan_result",
            "check": "equals",
            "value": "clean",
            "severity": "error",
            "rationale": "FedRAMP RA-5: Vulnerability scanning must be performed on all system components before deployment.",
            "remediation": "Run squash scan before attestation.",
        },
        {
            "id": "FEDRAMP-SA-12",
            "field": "components[0].pedigree.ancestors",
            "check": "non_empty",
            "severity": "error",
            "rationale": "FedRAMP SA-12: Supply chain risk management requires documented provenance for all acquired components.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta.",
        },
        {
            "id": "FEDRAMP-SA-10",
            "field": "metadata.tools",
            "check": "non_empty",
            "severity": "error",
            "rationale": "FedRAMP SA-10: Developer configuration management requires documentation of development tools and toolchain.",
            "remediation": "SBOM metadata.tools is populated automatically — do not strip it.",
        },
        {
            "id": "FEDRAMP-CM-8-PURL",
            "field": "components[0].purl",
            "check": "non_empty",
            "severity": "error",
            "rationale": "FedRAMP CM-8: Component inventory must include package identifiers for third-party software components.",
            "remediation": "Set hf_mlx_repo in CompressRunMeta.",
        },
        {
            "id": "FEDRAMP-AU-9",
            "field": "metadata.timestamp",
            "check": "non_empty",
            "severity": "error",
            "rationale": "FedRAMP AU-9: Audit record protection requires timestamps on all compliance attestation records.",
            "remediation": "SBOM timestamp is auto-generated at compress time — do not strip it.",
        },
        {
            "id": "FEDRAMP-SA-11",
            "field": "components[0].modelCard.quantitativeAnalysis.performanceMetrics",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "FedRAMP SA-11: Developer security testing requires documented test results for system components.",
            "remediation": "Run squish eval to bind lm_eval performance scores.",
        },
        {
            "id": "FEDRAMP-CM-6",
            "field": "components[0].modelCard.modelParameters",
            "check": "non_empty",
            "severity": "warning",
            "rationale": "FedRAMP CM-6: Configuration settings must be documented and controlled for all system components.",
            "remediation": "Ensure model card parameters are populated by the compress pipeline.",
        },
    ],
}

# Allow "strict" as an alias for "enterprise-strict"
_POLICIES["strict"] = _POLICIES["enterprise-strict"]
# CMMC Level 2 (NIST SP 800-171) shares the same AI-component control
# mapping as FedRAMP Moderate (NIST SP 800-53) for SBOM / attestation.
_POLICIES["cmmc"] = _POLICIES["fedramp"]

# ── MCP server attestation policy (Wave 45) ───────────────────────────────────
# Targets the MCP tools/list catalog rather than a CycloneDX SBOM.
# Rules are evaluated by McpScanner.scan() — field paths are MCP-specific.
_POLICIES["mcp-strict"] = [
    {
        "id": "MCP-001",
        "field": "mcp:tool_descriptions",
        "check": "non_empty",
        "severity": "error",
        "rationale": (
            "EU AI Act Art. 9(2)(d): adversarial input resilience requires that "
            "MCP tool descriptions contain no prompt-injection phrases that could "
            "hijack an agentic LLM's system prompt."
        ),
        "remediation": "Remove or sanitize prompt-injection phrases from tool descriptions.",
    },
    {
        "id": "MCP-002",
        "field": "mcp:tool_params",
        "check": "non_empty",
        "severity": "error",
        "rationale": (
            "EU AI Act Art. 9(2)(d): SSRF and path-traversal vectors in parameter "
            "defaults or descriptions can route model-generated requests to internal "
            "infrastructure, violating network isolation requirements."
        ),
        "remediation": "Remove localhost, RFC1918, and cloud-metadata URL references.",
    },
    {
        "id": "MCP-003",
        "field": "mcp:tool_names",
        "check": "non_empty",
        "severity": "error",
        "rationale": (
            "Tool shadowing: a tool whose name matches a privileged OS command "
            "(exec, shell, eval, sudo, etc.) may cause an agentic LLM to confuse "
            "MCP invocation with direct system-level execution."
        ),
        "remediation": "Rename the tool to avoid collision with OS/runtime commands.",
    },
    {
        "id": "MCP-004",
        "field": "mcp:tool_entries",
        "check": "non_empty",
        "severity": "error",
        "rationale": (
            "MCP protocol integrity: every tool entry must declare name, description, "
            "and inputSchema.  Missing fields prevent safe attestation and may cause "
            "unpredictable agentic behavior."
        ),
        "remediation": "Ensure every tool object includes name, description, and inputSchema.",
    },
    {
        "id": "MCP-005",
        "field": "mcp:tool_params",
        "check": "non_empty",
        "severity": "warning",
        "rationale": (
            "HTTP endpoints resembling OOB beacons (path indicators /upload, /collect, "
            "/beacon, /track; OAST domains) in parameter text may indicate a data "
            "exfiltration vector. Review for legitimacy."
        ),
        "remediation": "Remove or document external HTTP endpoints in tool schema.",
    },
    {
        "id": "MCP-006",
        "field": "mcp:tool_descriptions",
        "check": "non_empty",
        "severity": "warning",
        "rationale": (
            "EU AI Act Art. 9: permission over-reach claims (admin, root, sudo, system "
            "access) in tool metadata should be scrutinised to ensure least-privilege "
            "design for the agentic system."
        ),
        "remediation": "Revise tool descriptions to reflect actual minimum required permissions.",
    },
]

AVAILABLE_POLICIES: frozenset[str] = frozenset(_POLICIES.keys())


# ── Finding dataclass ────────────────────────────────────────────────────────


@dataclass
class PolicyFinding:
    """A single policy rule evaluation result."""

    rule_id: str
    severity: str  # "error" | "warning"
    passed: bool
    field: str
    rationale: str
    remediation: str
    actual_value: Any = None
    remediation_link: str = ""


@dataclass
class PolicyResult:
    """Aggregate result of evaluating a policy against an SBOM document."""

    policy_name: str
    passed: bool  # True only if zero error-severity findings failed
    findings: list[PolicyFinding] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "error" and not f.passed)

    @property
    def warning_count(self) -> int:
        return sum(
            1 for f in self.findings if f.severity == "warning" and not f.passed
        )

    @property
    def pass_count(self) -> int:
        return sum(1 for f in self.findings if f.passed)

    def summary(self) -> str:
        total = len(self.findings)
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.policy_name}: "
            f"{self.pass_count}/{total} rules passed, "
            f"{self.error_count} errors, {self.warning_count} warnings"
        )


# ── Policy engine ────────────────────────────────────────────────────────────


class PolicyEngine:
    """Evaluate a named policy against a CycloneDX SBOM dict.

    The SBOM dict is the plain Python structure produced by
    :class:`~squish.squash.sbom_builder.CycloneDXBuilder`, optionally
    annotated with extra top-level ``squash:*`` keys by the attest pipeline
    (e.g. ``squash:scan_result = "clean"``).
    """

    @staticmethod
    def evaluate(sbom: dict[str, Any], policy_name: str) -> PolicyResult:
        """Evaluate *policy_name* rules against *sbom*.

        Parameters
        ----------
        sbom:
            CycloneDX 1.7 dict, optionally enriched with ``squash:*`` keys.
        policy_name:
            One of the keys in :data:`AVAILABLE_POLICIES`.

        Returns
        -------
        PolicyResult

        Raises
        ------
        KeyError
            If *policy_name* is not in :data:`AVAILABLE_POLICIES`.
        """
        if policy_name not in _POLICIES:
            available = ", ".join(sorted(_POLICIES))
            raise KeyError(
                f"Unknown policy '{policy_name}'. Available: {available}"
            )

        rules = _POLICIES[policy_name]
        findings: list[PolicyFinding] = []

        for rule in rules:
            actual = _resolve_field(sbom, rule["field"])
            passed = _check(actual, rule["check"], rule.get("value"), rule.get("pattern"), rule.get("allowed"))
            findings.append(
                PolicyFinding(
                    rule_id=rule["id"],
                    severity=rule["severity"],
                    passed=passed,
                    field=rule["field"],
                    rationale=rule["rationale"],
                    remediation=rule["remediation"],
                    actual_value=actual,
                    remediation_link=rule.get("remediation_link", ""),
                )
            )

        all_errors_passed = all(
            f.passed for f in findings if f.severity == "error"
        )
        return PolicyResult(
            policy_name=policy_name,
            passed=all_errors_passed,
            findings=findings,
        )

    @staticmethod
    def evaluate_all(
        sbom: dict[str, Any], policy_names: list[str]
    ) -> dict[str, PolicyResult]:
        """Evaluate multiple policies, returning a dict keyed by policy name."""
        return {name: PolicyEngine.evaluate(sbom, name) for name in policy_names}

    @staticmethod
    def evaluate_custom(
        sbom: dict[str, Any], rules: list[dict[str, Any]], policy_name: str = "custom"
    ) -> PolicyResult:
        """Evaluate an ad-hoc list of rule dicts against *sbom*.

        Rules use the same schema as built-in policies but are not stored in
        :data:`_POLICIES`.  Validation errors in individual rules cause that
        rule to be skipped with a logged warning rather than raising.

        Parameters
        ----------
        sbom:
            CycloneDX 1.7 dict, optionally enriched with ``squash:*`` keys.
        rules:
            List of rule dicts.  Each must have: ``id``, ``field``, ``check``,
            ``severity``.  Optional: ``value``, ``pattern``, ``allowed``,
            ``rationale``, ``remediation``, ``remediation_link``.
        policy_name:
            Label for the returned :class:`PolicyResult`.
        """
        validated = PolicyRegistry.validate_rules(rules)
        if validated:
            log.warning("custom rules have schema errors (skipping %d): %s", len(validated), validated)

        findings: list[PolicyFinding] = []
        for rule in rules:
            if not rule.get("id") or not rule.get("field") or not rule.get("check"):
                continue  # skip invalid rules
            actual = _resolve_field(sbom, rule["field"])
            passed = _check(actual, rule["check"], rule.get("value"), rule.get("pattern"), rule.get("allowed"))
            findings.append(
                PolicyFinding(
                    rule_id=rule["id"],
                    severity=rule.get("severity", "error"),
                    passed=passed,
                    field=rule["field"],
                    rationale=rule.get("rationale", ""),
                    remediation=rule.get("remediation", ""),
                    actual_value=actual,
                    remediation_link=rule.get("remediation_link", ""),
                )
            )

        all_errors_passed = all(f.passed for f in findings if f.severity == "error")
        return PolicyResult(
            policy_name=policy_name,
            passed=all_errors_passed,
            findings=findings,
        )


# ── Policy registry ──────────────────────────────────────────────────────────


# Required fields per rule; values are plain strings or special types noted.
_RULE_REQUIRED_KEYS: frozenset[str] = frozenset({"id", "field", "check", "severity"})
_VALID_CHECK_TYPES: frozenset[str] = frozenset(
    {"present", "non_empty", "equals", "not_equals", "min_value", "regex_match", "in_list"}
)
_VALID_SEVERITIES: frozenset[str] = frozenset({"error", "warning"})


class PolicyRegistry:
    """Load and validate custom policy rule sets from YAML files or plain dicts.

    Custom rules follow the same schema as built-in policies and are evaluated
    via :meth:`PolicyEngine.evaluate_custom`.

    Example YAML rule file::

        - id: CUSTOM-001
          field: components[0].name
          check: non_empty
          severity: error
          rationale: Our policy requires a named model.
          remediation: Set model_id in CompressRunMeta.
          remediation_link: https://docs.example.com/model-id

        - id: CUSTOM-002
          field: components[0].supplier.name
          check: regex_match
          pattern: "^Acme"
          severity: warning
          rationale: All models must come from Acme Corp.
          remediation: Update supplier in the SBOM.
    """

    @staticmethod
    def load_rules_from_yaml(path: Path) -> list[dict[str, Any]]:
        """Load rule dicts from a YAML file.

        Requires ``PyYAML`` (``pip install pyyaml``).

        Parameters
        ----------
        path:
            Path to a ``.yaml`` / ``.yml`` file containing a list of rule dicts.

        Returns
        -------
        list[dict]
            Validated rule dicts (invalid rules are filtered and logged).

        Raises
        ------
        ImportError  if PyYAML is not installed.
        OSError      if the file cannot be read.
        ValueError   if the YAML top-level is not a list.
        """
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML rules. "
                "Install with: pip install pyyaml"
            ) from e

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError(
                f"YAML rules file must contain a top-level list of rules, got {type(raw).__name__}"
            )
        return PolicyRegistry.load_rules_from_dict(raw)

    @staticmethod
    def load_rules_from_dict(rules: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalise and validate a list of rule dicts.

        Invalid rules (missing required keys, unknown check type, bad severity)
        are filtered out and logged as warnings.  Returns only valid rules.
        """
        valid: list[dict[str, Any]] = []
        for i, rule in enumerate(rules):
            if not isinstance(rule, dict):
                log.warning("rules[%d] is not a dict — skipped", i)
                continue
            errors = PolicyRegistry.validate_rules([rule])
            if errors:
                log.warning("rules[%d] invalid (%s) — skipped", i, "; ".join(errors))
                continue
            valid.append(rule)
        return valid

    @staticmethod
    def validate_rules(rules: list[dict[str, Any]]) -> list[str]:
        """Return a list of human-readable validation errors for *rules*.

        Returns an empty list when all rules are valid.
        """
        errors: list[str] = []
        for i, rule in enumerate(rules):
            prefix = f"rules[{i}] (id={rule.get('id', '?')!r})"
            if not isinstance(rule, dict):
                errors.append(f"{prefix}: not a dict")
                continue
            for key in _RULE_REQUIRED_KEYS:
                if not rule.get(key):
                    errors.append(f"{prefix}: missing required field '{key}'")
            check = rule.get("check", "")
            if check and check not in _VALID_CHECK_TYPES:
                errors.append(
                    f"{prefix}: unknown check type '{check}'. "
                    f"Valid: {sorted(_VALID_CHECK_TYPES)}"
                )
            sev = rule.get("severity", "")
            if sev and sev not in _VALID_SEVERITIES:
                errors.append(
                    f"{prefix}: unknown severity '{sev}'. Valid: {sorted(_VALID_SEVERITIES)}"
                )
            if check == "regex_match" and not rule.get("pattern"):
                errors.append(f"{prefix}: 'regex_match' requires a 'pattern' key")
            if check == "in_list" and not rule.get("allowed"):
                errors.append(f"{prefix}: 'in_list' requires an 'allowed' key (list)")
            if check in ("equals", "not_equals", "min_value") and rule.get("value") is None:
                errors.append(f"{prefix}: '{check}' requires a 'value' key")
        return errors


# ── Field resolution helpers ─────────────────────────────────────────────────


def _resolve_field(doc: dict[str, Any], path: str) -> Any:
    """Resolve a dot-path with optional array indexing into *doc*.

    Supports paths like:
    - ``"components[0].name"``
    - ``"metadata.timestamp"``
    - ``"squash:scan_result"`` (top-level key with colon — treated as flat key)
    """
    # Top-level flat keys with colons (squash:key notation)
    if ":" in path and "." not in path and "[" not in path:
        return doc.get(path)

    parts = _split_path(path)
    current: Any = doc
    for part in parts:
        if current is None:
            return None
        if isinstance(part, int):
            if isinstance(current, list) and len(current) > part:
                current = current[part]
            else:
                return None
        else:
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return None
    return current


def _split_path(path: str) -> list[str | int]:
    """Split a dot-path with array indices into tokens."""
    tokens: list[str | int] = []
    for segment in path.split("."):
        if "[" in segment:
            key, rest = segment.split("[", 1)
            idx_str = rest.split("]", 1)[0]
            if key:
                tokens.append(key)
            tokens.append(int(idx_str))
        else:
            tokens.append(segment)
    return tokens


def _check(
    actual: Any,
    check_type: str,
    expected: Any = None,
    pattern: str | None = None,
    allowed: list[Any] | None = None,
) -> bool:
    """Evaluate a single rule check."""
    if check_type == "present":
        return actual is not None
    if check_type == "non_empty":
        if actual is None:
            return False
        if isinstance(actual, (list, dict, str)):
            return len(actual) > 0
        return actual is not None
    if check_type == "equals":
        return actual == expected
    if check_type == "not_equals":
        return actual != expected
    if check_type == "min_value":
        try:
            return float(actual) >= float(expected)
        except (TypeError, ValueError):
            return False
    if check_type == "regex_match":
        if pattern is None or actual is None:
            return False
        try:
            return bool(re.search(pattern, str(actual)))
        except re.error as e:
            log.warning("regex_match: invalid pattern %r — %s", pattern, e)
            return False
    if check_type == "in_list":
        if allowed is None:
            return False
        return actual in allowed
    log.warning("Unknown check type '%s' — treating as failed", check_type)
    return False


# ── Policy history ────────────────────────────────────────────────────────────


class PolicyHistory:
    """Persistent log of :class:`PolicyResult` evaluations tied to a model.

    Results are persisted as newline-delimited JSON (ndjson) at *log_path*.
    Each line is a JSON object with the fields:
    ``{"ts": "<iso8601>", "model": "<path>", "policy": "<name>",
       "passed": bool, "errors": int, "warnings": int}``
    """

    def __init__(self, log_path: Path) -> None:
        self._path = Path(log_path)

    # ------------------------------------------------------------------ write

    def append(self, result: "PolicyResult", model_path: str) -> None:
        """Append *result* for *model_path* to the history log."""
        record = {
            "ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model": model_path,
            "policy": result.policy_name,
            "passed": result.passed,
            "errors": result.error_count,
            "warnings": result.warning_count,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ read

    def _iter_records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        records: list[dict[str, Any]] = []
        for raw in self._path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if raw:
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError:
                    pass
        return records

    def regressions_since(
        self, since: datetime.datetime
    ) -> list[dict[str, Any]]:
        """Return all failed records written after *since*.

        *since* must be timezone-aware.  The comparison is done in UTC.
        """
        since_utc = since.astimezone(datetime.timezone.utc)
        return [
            r
            for r in self._iter_records()
            if not r.get("passed", True)
            and datetime.datetime.fromisoformat(r["ts"]) >= since_utc
        ]

    def latest(self, model_path: str) -> "dict[str, Any] | None":
        """Return the most recent record for *model_path*, or ``None``."""
        matches = [r for r in self._iter_records() if r.get("model") == model_path]
        return matches[-1] if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# Wave 17 — PolicyWebhook: outbound HTTP alerting
# ─────────────────────────────────────────────────────────────────────────────

class PolicyWebhook:
    """Send policy evaluation results as JSON POST webhooks.

    Uses stdlib ``urllib.request`` — zero external dependencies.

    Two entry points:

    * :meth:`notify` — structured payload from a :class:`PolicyResult`.
    * :meth:`notify_raw` — send an arbitrary dict to any URL (used by the API).

    The payload format follows a simple envelope::

        {
          "event": "squash.policy.result",
          "model_path": "...",
          "policy": "enterprise-strict",
          "passed": false,
          "error_count": 3,
          "warning_count": 1,
          "timestamp": "2026-01-01T00:00:00+00:00"
        }

    The webhook URL can be set via the ``SQUASH_WEBHOOK_URL`` environment
    variable, or passed explicitly.
    """

    TIMEOUT_SECONDS: int = 5
    _ENV_KEY: str = "SQUASH_WEBHOOK_URL"

    @staticmethod
    def notify(
        result: "PolicyResult",
        model_path: str,
        webhook_url: str | None = None,
    ) -> bool:
        """Send a structured webhook for *result*.

        Parameters
        ----------
        result:
            The :class:`PolicyResult` to report.
        model_path:
            Human-readable model path / identifier to include in the payload.
        webhook_url:
            Explicit URL.  Falls back to ``SQUASH_WEBHOOK_URL`` env var.
            Returns ``False`` immediately if no URL is available.

        Returns
        -------
        bool
            ``True`` if the webhook was delivered (2xx response), ``False``
            otherwise.  Never raises.
        """
        import datetime as _dt
        import os

        url = webhook_url or os.environ.get(PolicyWebhook._ENV_KEY, "")
        if not url:
            return False

        payload: dict[str, Any] = {
            "event": "squash.policy.result",
            "model_path": str(model_path),
            "policy": result.policy_name,
            "passed": result.passed,
            "error_count": result.error_count,
            "warning_count": result.warning_count,
            "pass_count": result.pass_count,
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        }
        return PolicyWebhook.notify_raw(payload, url)

    @staticmethod
    def notify_raw(payload: dict[str, Any], webhook_url: str) -> bool:
        """POST *payload* as JSON to *webhook_url*.

        Returns ``True`` on a 2xx response, ``False`` on any error (network,
        timeout, non-2xx status).  Never raises.
        """
        import urllib.error
        import urllib.request

        if not webhook_url:
            return False

        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                webhook_url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=PolicyWebhook.TIMEOUT_SECONDS) as resp:  # noqa: S310
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as e:
            log.warning("PolicyWebhook: HTTP %s for %s", e.code, webhook_url)
            return False
        except Exception as e:
            log.warning("PolicyWebhook: delivery failed for %s — %s", webhook_url, e)
            return False


# ── Wave 20 — NTIA Minimum Elements Validator ─────────────────────────────────

# NTIA minimum elements per the 2021 "Minimum Elements For a Software Bill of
# Materials" guidance.  The CycloneDX field paths are mapped to the seven
# required elements as closely as the ML-BOM schema allows.
_NTIA_REQUIRED_FIELD_PATHS: dict[str, str] = {
    "supplier_name":           "metadata.supplier.name",
    "component_name":          "components[0].name",
    "component_version":       "components[0].version",
    "unique_identifier":       "components[0].purl",
    "dependency_relationship": "dependencies[0].ref",
    "author_of_sbom":          "metadata.authors[0].name",
    "timestamp":               "metadata.timestamp",
}


@dataclass
class NtiaResult:
    """Result of an NTIA minimum elements check against a CycloneDX BOM.

    Attributes
    ----------
    passed:
        ``True`` when all seven required NTIA elements are present.
    present_fields:
        Element names that were found in the BOM.
    missing_fields:
        Element names that were absent or empty.
    completeness_score:
        Fraction of required elements present (0.0–1.0).
    bom_path:
        Path that was checked (may be ``None`` when the BOM was supplied
        as a raw dict).
    """

    passed: bool
    present_fields: list[str]
    missing_fields: list[str]
    completeness_score: float
    bom_path: "Path | None" = None


class NtiaValidator:
    """Validate a CycloneDX ML-BOM against the NTIA minimum elements spec.

    Usage::

        result = NtiaValidator.check(Path("./model/cyclonedx-mlbom.json"))
        if not result.passed:
            print("Missing:", result.missing_fields)
    """

    @staticmethod
    def check(
        bom: "Path | dict[str, Any]",
        *,
        strict: bool = False,
    ) -> "NtiaResult":
        """Check *bom* for NTIA minimum elements.

        Parameters
        ----------
        bom:
            Either a :class:`~pathlib.Path` to a CycloneDX JSON file or
            an already-parsed BOM dict.
        strict:
            If ``True`` the *dependency_relationship* check requires at
            least one dependency entry with a non-empty ``dependsOn``
            list (not just the presence of the ``dependencies`` key).

        Returns
        -------
        NtiaResult
        """
        import pathlib

        bom_path: "Path | None" = None
        if isinstance(bom, pathlib.Path):
            bom_path = bom
            if not bom.exists():
                raise FileNotFoundError(f"BOM file not found: {bom}")
            try:
                doc: dict[str, Any] = json.loads(bom.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, ValueError, OSError):
                return NtiaResult(
                    passed=False,
                    present_fields=[],
                    missing_fields=list(_NTIA_REQUIRED_FIELD_PATHS.keys()),
                    completeness_score=0.0,
                    bom_path=bom_path,
                )
        else:
            doc = bom

        present: list[str] = []
        missing: list[str] = []

        for element_name, field_path in _NTIA_REQUIRED_FIELD_PATHS.items():
            value = _resolve_field(doc, field_path)
            if element_name == "dependency_relationship" and strict:
                # strict: require at least one non-empty dependsOn list
                deps = doc.get("dependencies", [])
                has_real_dep = any(
                    isinstance(d.get("dependsOn"), list) and len(d["dependsOn"]) > 0
                    for d in deps
                )
                if has_real_dep:
                    present.append(element_name)
                else:
                    missing.append(element_name)
            elif value is not None and value != "" and value != [] and value != {}:
                present.append(element_name)
            else:
                missing.append(element_name)

        total = len(_NTIA_REQUIRED_FIELD_PATHS)
        score = round(len(present) / total, 4) if total else 1.0
        return NtiaResult(
            passed=len(missing) == 0,
            present_fields=present,
            missing_fields=missing,
            completeness_score=score,
            bom_path=bom_path,
        )
