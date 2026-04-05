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
}

# Allow "strict" as an alias for "enterprise-strict"
_POLICIES["strict"] = _POLICIES["enterprise-strict"]

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
