"""tests/test_squash_wave34.py — Wave 34: EU CRA + FedRAMP policy templates.

Covers:
  Policy registration:
    - "eu-cra", "fedramp", and "cmmc" are present in AVAILABLE_POLICIES
    - "cmmc" maps to identical rules as "fedramp" (alias)
    - PolicyEngine raises KeyError on an unknown policy (regression guard)

  EU CRA policy — structure:
    - All expected rule IDs present (CRA-001 … CRA-008)
    - Each rule has required fields: id, field, check, severity, rationale, remediation
    - Error-severity rules: CRA-001 … CRA-006
    - Warning-severity rules: CRA-007, CRA-008
    - All check types are from the valid set

  EU CRA policy — behaviour:
    - Fully populated SBOM → PolicyResult.passed = True
    - Missing name → CRA-001 fails
    - Missing hashes → CRA-002 fails
    - scan_result != "clean" → CRA-003 fails
    - Missing purl → CRA-004 fails
    - Missing pedigree.ancestors → CRA-005 fails
    - Missing timestamp → CRA-006 fails
    - Missing quantizationLevel → CRA-007 fails (warning, policy still passes)
    - Missing metadata.tools → CRA-008 fails (warning, policy still passes)
    - Empty SBOM → all error rules fail, result is Policy.passed = False

  FedRAMP policy — structure:
    - All expected rule IDs present (FEDRAMP-*)
    - Error-severity rules: CM-8, SI-7, RA-5, SA-12, SA-10, CM-8-PURL, AU-9
    - Warning-severity rules: SA-11, CM-6
    - All check types are valid

  FedRAMP policy — behaviour:
    - Fully populated SBOM → PolicyResult.passed = True
    - Missing hashes → FEDRAMP-SI-7 fails
    - scan_result = "flagged" → FEDRAMP-RA-5 fails
    - Missing pedigree.ancestors → FEDRAMP-SA-12 fails
    - Missing timestamp → FEDRAMP-AU-9 fails
    - Empty SBOM → all error rules fail

  CMMC alias:
    - Evaluating "cmmc" and "fedramp" on identical SBOMs returns identical pass/fail
    - Number of findings is equal

  PolicyResult summary:
    - summary() on a passing eu-cra result contains "PASS"
    - summary() on a failing eu-cra result contains "FAIL"

Test taxonomy: Pure unit — no I/O, no network, no process-state mutation.
Uses only in-memory SBOM dicts and the imported policy engine.
"""

from __future__ import annotations

from typing import Any

import pytest

from squish.squash.policy import (
    AVAILABLE_POLICIES,
    PolicyEngine,
    _POLICIES,
    _VALID_CHECK_TYPES,
    _VALID_SEVERITIES,
)


# ── SBOM helpers ──────────────────────────────────────────────────────────────


def _full_bom(scan_result: str = "clean") -> dict[str, Any]:
    """Fully-populated SBOM that should pass all eu-cra and fedramp rules."""
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "metadata": {
            "timestamp": "2026-04-06T00:00:00Z",
            "tools": ["squash/9.13.0"],
        },
        "components": [
            {
                "type": "ml-model",
                "name": "Qwen2.5-1.5B-instruct-int4",
                "purl": "pkg:huggingface/Qwen/Qwen2.5-1.5B-Instruct@main",
                "hashes": [{"alg": "SHA-256", "content": "abc123"}],
                "pedigree": {"ancestors": [{"name": "Qwen/Qwen2.5-1.5B-Instruct"}]},
                "externalReferences": [
                    {"type": "website", "url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"}
                ],
                "modelCard": {
                    "modelParameters": {
                        "architectureFamily": "qwen",
                        "quantizationLevel": "INT4",
                    },
                    "quantitativeAnalysis": {
                        "performanceMetrics": [{"type": "arc_easy", "value": 0.706}]
                    },
                },
            }
        ],
        "squash:scan_result": scan_result,
    }


def _strip(bom: dict[str, Any], path: str) -> dict[str, Any]:
    """Return a shallow copy of bom with the dot-path value deleted.
    Supports one level of component array indexing (components[0].field).
    """
    import copy
    bom = copy.deepcopy(bom)
    if path.startswith("components[0]."):
        key = path[len("components[0]."):]
        parts = key.split(".", 1)
        comp = bom["components"][0]
        if len(parts) == 1:
            comp.pop(parts[0], None)
        else:
            # nested: e.g. pedigree.ancestors
            outer, inner = parts
            if outer in comp:
                comp[outer].pop(inner, None)
    elif path.startswith("metadata."):
        key = path[len("metadata."):]
        bom["metadata"].pop(key, None)
    elif path == "squash:scan_result":
        bom.pop("squash:scan_result", None)
    return bom


# ── Policy registration ───────────────────────────────────────────────────────


class TestPolicyRegistration:
    def test_eu_cra_in_available_policies(self):
        assert "eu-cra" in AVAILABLE_POLICIES

    def test_fedramp_in_available_policies(self):
        assert "fedramp" in AVAILABLE_POLICIES

    def test_cmmc_in_available_policies(self):
        assert "cmmc" in AVAILABLE_POLICIES

    def test_cmmc_rules_identical_to_fedramp(self):
        assert _POLICIES["cmmc"] is _POLICIES["fedramp"]

    def test_strict_alias_still_present(self):
        # Regression guard — existing aliases must not be broken
        assert "strict" in AVAILABLE_POLICIES

    def test_unknown_policy_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown policy"):
            PolicyEngine.evaluate({}, "not-a-real-policy")

    def test_available_policies_is_frozenset(self):
        assert isinstance(AVAILABLE_POLICIES, frozenset)


# ── EU CRA — rule structure ───────────────────────────────────────────────────


_EU_CRA_EXPECTED_RULE_IDS = {
    "CRA-001", "CRA-002", "CRA-003", "CRA-004",
    "CRA-005", "CRA-006", "CRA-007", "CRA-008",
}
_EU_CRA_ERROR_IDS = {"CRA-001", "CRA-002", "CRA-003", "CRA-004", "CRA-005", "CRA-006"}
_EU_CRA_WARNING_IDS = {"CRA-007", "CRA-008"}


class TestEuCraStructure:
    def _rules(self):
        return _POLICIES["eu-cra"]

    def _rule_by_id(self, rule_id: str):
        for r in self._rules():
            if r["id"] == rule_id:
                return r
        pytest.fail(f"Rule {rule_id!r} not found in eu-cra policy")

    def test_rule_count_is_eight(self):
        assert len(self._rules()) == 8

    def test_all_expected_rule_ids_present(self):
        ids = {r["id"] for r in self._rules()}
        assert _EU_CRA_EXPECTED_RULE_IDS == ids

    def test_all_rules_have_required_fields(self):
        required = {"id", "field", "check", "severity", "rationale", "remediation"}
        for rule in self._rules():
            missing = required - rule.keys()
            assert not missing, f"Rule {rule.get('id')} missing: {missing}"

    def test_all_check_types_are_valid(self):
        for rule in self._rules():
            assert rule["check"] in _VALID_CHECK_TYPES, (
                f"Rule {rule['id']}: unknown check type {rule['check']!r}"
            )

    def test_all_severities_are_valid(self):
        for rule in self._rules():
            assert rule["severity"] in _VALID_SEVERITIES

    def test_error_severity_rule_ids(self):
        error_ids = {r["id"] for r in self._rules() if r["severity"] == "error"}
        assert error_ids == _EU_CRA_ERROR_IDS

    def test_warning_severity_rule_ids(self):
        warning_ids = {r["id"] for r in self._rules() if r["severity"] == "warning"}
        assert warning_ids == _EU_CRA_WARNING_IDS

    def test_cra_003_checks_scan_result_equals_clean(self):
        rule = self._rule_by_id("CRA-003")
        assert rule["check"] == "equals"
        assert rule["value"] == "clean"

    def test_cra_001_targets_component_name(self):
        rule = self._rule_by_id("CRA-001")
        assert "components[0].name" in rule["field"]

    def test_cra_002_targets_hashes(self):
        rule = self._rule_by_id("CRA-002")
        assert "hashes" in rule["field"]

    def test_rationale_cites_eu_cra_reference(self):
        for rule in self._rules():
            assert "EU CRA" in rule["rationale"], (
                f"Rule {rule['id']} rationale does not cite EU CRA: {rule['rationale']!r}"
            )


# ── EU CRA — evaluation behaviour ────────────────────────────────────────────


class TestEuCraBehaviour:
    def test_full_bom_passes(self):
        result = PolicyEngine.evaluate(_full_bom(), "eu-cra")
        assert result.passed

    def test_full_bom_has_zero_errors(self):
        result = PolicyEngine.evaluate(_full_bom(), "eu-cra")
        assert result.error_count == 0

    def test_missing_name_fails_cra_001(self):
        bom = _strip(_full_bom(), "components[0].name")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-001" in ids

    def test_missing_hashes_fails_cra_002(self):
        bom = _strip(_full_bom(), "components[0].hashes")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-002" in ids

    def test_dirty_scan_fails_cra_003(self):
        bom = _full_bom(scan_result="flagged")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-003" in ids

    def test_missing_purl_fails_cra_004(self):
        bom = _strip(_full_bom(), "components[0].purl")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-004" in ids

    def test_missing_pedigree_fails_cra_005(self):
        bom = _strip(_full_bom(), "components[0].pedigree.ancestors")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-005" in ids

    def test_missing_timestamp_fails_cra_006(self):
        bom = _strip(_full_bom(), "metadata.timestamp")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-006" in ids

    def test_missing_quantization_fails_cra_007_but_policy_still_passes(self):
        import copy
        bom = copy.deepcopy(_full_bom())
        bom["components"][0]["modelCard"]["modelParameters"].pop("quantizationLevel", None)
        result = PolicyEngine.evaluate(bom, "eu-cra")
        # CRA-007 is a warning — policy.passed must still be True
        assert result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-007" in ids

    def test_missing_tools_fails_cra_008_but_policy_still_passes(self):
        bom = _strip(_full_bom(), "metadata.tools")
        result = PolicyEngine.evaluate(bom, "eu-cra")
        assert result.passed  # CRA-008 is warning
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "CRA-008" in ids

    def test_empty_bom_fails(self):
        result = PolicyEngine.evaluate({}, "eu-cra")
        assert not result.passed
        assert result.error_count > 0

    def test_result_policy_name_is_eu_cra(self):
        result = PolicyEngine.evaluate(_full_bom(), "eu-cra")
        assert result.policy_name == "eu-cra"

    def test_findings_count_matches_rule_count(self):
        result = PolicyEngine.evaluate(_full_bom(), "eu-cra")
        assert len(result.findings) == len(_POLICIES["eu-cra"])

    def test_pass_count_on_full_bom(self):
        result = PolicyEngine.evaluate(_full_bom(), "eu-cra")
        assert result.pass_count == len(_POLICIES["eu-cra"])

    def test_summary_contains_pass_on_full_bom(self):
        result = PolicyEngine.evaluate(_full_bom(), "eu-cra")
        assert "PASS" in result.summary()

    def test_summary_contains_fail_on_empty_bom(self):
        result = PolicyEngine.evaluate({}, "eu-cra")
        assert "FAIL" in result.summary()


# ── FedRAMP — rule structure ──────────────────────────────────────────────────


_FEDRAMP_ERROR_IDS = {
    "FEDRAMP-CM-8",
    "FEDRAMP-SI-7",
    "FEDRAMP-RA-5",
    "FEDRAMP-SA-12",
    "FEDRAMP-SA-10",
    "FEDRAMP-CM-8-PURL",
    "FEDRAMP-AU-9",
}
_FEDRAMP_WARNING_IDS = {"FEDRAMP-SA-11", "FEDRAMP-CM-6"}
_FEDRAMP_ALL_IDS = _FEDRAMP_ERROR_IDS | _FEDRAMP_WARNING_IDS


class TestFedRAMPStructure:
    def _rules(self):
        return _POLICIES["fedramp"]

    def _rule_by_id(self, rule_id: str):
        for r in self._rules():
            if r["id"] == rule_id:
                return r
        pytest.fail(f"Rule {rule_id!r} not found in fedramp policy")

    def test_rule_count_is_nine(self):
        assert len(self._rules()) == 9

    def test_all_expected_rule_ids_present(self):
        ids = {r["id"] for r in self._rules()}
        assert _FEDRAMP_ALL_IDS == ids

    def test_all_rules_have_required_fields(self):
        required = {"id", "field", "check", "severity", "rationale", "remediation"}
        for rule in self._rules():
            missing = required - rule.keys()
            assert not missing, f"Rule {rule.get('id')} missing: {missing}"

    def test_all_check_types_are_valid(self):
        for rule in self._rules():
            assert rule["check"] in _VALID_CHECK_TYPES

    def test_all_severities_are_valid(self):
        for rule in self._rules():
            assert rule["severity"] in _VALID_SEVERITIES

    def test_error_severity_rule_ids(self):
        error_ids = {r["id"] for r in self._rules() if r["severity"] == "error"}
        assert error_ids == _FEDRAMP_ERROR_IDS

    def test_warning_severity_rule_ids(self):
        warning_ids = {r["id"] for r in self._rules() if r["severity"] == "warning"}
        assert warning_ids == _FEDRAMP_WARNING_IDS

    def test_ra_5_checks_scan_result_equals_clean(self):
        rule = self._rule_by_id("FEDRAMP-RA-5")
        assert rule["check"] == "equals"
        assert rule["value"] == "clean"

    def test_rationale_cites_fedramp_control_id(self):
        for rule in self._rules():
            assert "FedRAMP" in rule["rationale"], (
                f"Rule {rule['id']} rationale does not cite FedRAMP: {rule['rationale']!r}"
            )

    def test_rationale_contains_nist_control_family(self):
        # Every rationale must reference its NIST control family (CM, SI, RA, SA, AU …)
        for rule in self._rules():
            assert any(
                ctrl in rule["rationale"]
                for ctrl in ("CM-", "SI-", "RA-", "SA-", "AU-")
            ), f"Rule {rule['id']} rationale missing NIST control: {rule['rationale']!r}"


# ── FedRAMP — evaluation behaviour ───────────────────────────────────────────


class TestFedRAMPBehaviour:
    def test_full_bom_passes(self):
        result = PolicyEngine.evaluate(_full_bom(), "fedramp")
        assert result.passed

    def test_full_bom_has_zero_errors(self):
        result = PolicyEngine.evaluate(_full_bom(), "fedramp")
        assert result.error_count == 0

    def test_missing_hashes_fails_si_7(self):
        bom = _strip(_full_bom(), "components[0].hashes")
        result = PolicyEngine.evaluate(bom, "fedramp")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "FEDRAMP-SI-7" in ids

    def test_dirty_scan_fails_ra_5(self):
        bom = _full_bom(scan_result="flagged")
        result = PolicyEngine.evaluate(bom, "fedramp")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "FEDRAMP-RA-5" in ids

    def test_missing_pedigree_fails_sa_12(self):
        bom = _strip(_full_bom(), "components[0].pedigree.ancestors")
        result = PolicyEngine.evaluate(bom, "fedramp")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "FEDRAMP-SA-12" in ids

    def test_missing_timestamp_fails_au_9(self):
        bom = _strip(_full_bom(), "metadata.timestamp")
        result = PolicyEngine.evaluate(bom, "fedramp")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "FEDRAMP-AU-9" in ids

    def test_missing_tools_fails_sa_10(self):
        bom = _strip(_full_bom(), "metadata.tools")
        result = PolicyEngine.evaluate(bom, "fedramp")
        assert not result.passed
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "FEDRAMP-SA-10" in ids

    def test_missing_performance_metrics_fails_sa_11_warning_only(self):
        import copy
        bom = copy.deepcopy(_full_bom())
        bom["components"][0]["modelCard"]["quantitativeAnalysis"].pop(
            "performanceMetrics", None
        )
        result = PolicyEngine.evaluate(bom, "fedramp")
        assert result.passed  # SA-11 is warning
        ids = {f.rule_id for f in result.findings if not f.passed}
        assert "FEDRAMP-SA-11" in ids

    def test_empty_bom_fails(self):
        result = PolicyEngine.evaluate({}, "fedramp")
        assert not result.passed
        assert result.error_count > 0

    def test_result_policy_name_is_fedramp(self):
        result = PolicyEngine.evaluate(_full_bom(), "fedramp")
        assert result.policy_name == "fedramp"

    def test_findings_count_matches_rule_count(self):
        result = PolicyEngine.evaluate(_full_bom(), "fedramp")
        assert len(result.findings) == len(_POLICIES["fedramp"])


# ── CMMC alias ────────────────────────────────────────────────────────────────


class TestCmmcAlias:
    def test_cmmc_passes_when_fedramp_passes(self):
        bom = _full_bom()
        r_fedramp = PolicyEngine.evaluate(bom, "fedramp")
        r_cmmc = PolicyEngine.evaluate(bom, "cmmc")
        assert r_fedramp.passed == r_cmmc.passed

    def test_cmmc_fails_when_fedramp_fails(self):
        bom = _full_bom(scan_result="flagged")
        r_fedramp = PolicyEngine.evaluate(bom, "fedramp")
        r_cmmc = PolicyEngine.evaluate(bom, "cmmc")
        assert not r_fedramp.passed
        assert not r_cmmc.passed

    def test_cmmc_and_fedramp_have_same_finding_count(self):
        bom = _full_bom()
        r_fedramp = PolicyEngine.evaluate(bom, "fedramp")
        r_cmmc = PolicyEngine.evaluate(bom, "cmmc")
        assert len(r_fedramp.findings) == len(r_cmmc.findings)

    def test_cmmc_result_policy_name_is_cmmc(self):
        result = PolicyEngine.evaluate(_full_bom(), "cmmc")
        assert result.policy_name == "cmmc"


# ── evaluate_all cross-policy ─────────────────────────────────────────────────


class TestEvaluateAll:
    def test_evaluate_all_returns_results_for_both_new_policies(self):
        results = PolicyEngine.evaluate_all(_full_bom(), ["eu-cra", "fedramp"])
        assert "eu-cra" in results
        assert "fedramp" in results

    def test_evaluate_all_both_pass_on_full_bom(self):
        results = PolicyEngine.evaluate_all(_full_bom(), ["eu-cra", "fedramp", "cmmc"])
        for name, result in results.items():
            assert result.passed, f"{name} failed on a full SBOM"

    def test_evaluate_all_includes_cumulative_error_from_single_field_strip(self):
        # Removing hashes should fail both eu-cra (CRA-002) and fedramp (FEDRAMP-SI-7)
        bom = _strip(_full_bom(), "components[0].hashes")
        results = PolicyEngine.evaluate_all(bom, ["eu-cra", "fedramp"])
        assert not results["eu-cra"].passed
        assert not results["fedramp"].passed
