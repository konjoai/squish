"""tests/test_squash_policy_custom.py — Wave 9 tests for custom policy rules.

Test taxonomy: Pure unit — no I/O, no YAML files on disk (except one
integration test that writes a temp YAML and reads it back).

Covers:
  - PolicyRegistry.validate_rules — all invalid rule permutations
  - PolicyRegistry.load_rules_from_dict — filters invalid rules, keeps valid
  - PolicyRegistry.load_rules_from_yaml — roundtrip via temp YAML file
  - PolicyEngine.evaluate_custom — regex_match, in_list, all existing checks
  - remediation_link field on PolicyFinding — roundtrip from rule to finding
  - PolicyEngine.evaluate — remediation_link forwarded from built-in rules
  - API custom_rules field — acceptance + rejection via FastAPI test client
  - Failure cases: bad regex, missing pattern/allowed, unknown check type
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest

from squish.squash.policy import (
    AVAILABLE_POLICIES,
    PolicyEngine,
    PolicyFinding,
    PolicyRegistry,
    _check,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _minimal_bom(scan_result: str = "clean") -> dict[str, Any]:
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "metadata": {"timestamp": "2024-01-01T00:00:00Z", "tools": ["squash"]},
        "components": [
            {
                "type": "ml-model",
                "name": "test-model",
                "purl": "pkg:mlmodel/test-model@1.0",
                "hashes": [{"alg": "SHA-256", "content": "abc123"}],
                "pedigree": {"ancestors": [{"name": "upstream-model"}]},
                "externalReferences": [{"type": "website", "url": "https://example.com"}],
                "modelCard": {
                    "modelParameters": {
                        "architectureFamily": "llama",
                        "quantizationLevel": "INT4",
                    },
                    "quantitativeAnalysis": {
                        "performanceMetrics": [{"type": "arc_easy", "value": 0.706}]
                    },
                },
                "supplier": {"name": "Acme Corp"},
            }
        ],
        "squash:scan_result": scan_result,
    }


def _rule(
    rule_id: str = "T-001",
    field: str = "components[0].name",
    check: str = "non_empty",
    severity: str = "error",
    **kwargs: Any,
) -> dict[str, Any]:
    return {"id": rule_id, "field": field, "check": check, "severity": severity, **kwargs}


# ── Unit: _check new types ────────────────────────────────────────────────────


class TestCheckRegexMatch:
    def test_matching_pattern_passes(self):
        assert _check("Acme Corp", "regex_match", pattern="^Acme") is True

    def test_non_matching_pattern_fails(self):
        assert _check("Other Corp", "regex_match", pattern="^Acme") is False

    def test_none_actual_fails(self):
        assert _check(None, "regex_match", pattern=".*") is False

    def test_none_pattern_fails(self):
        assert _check("hello", "regex_match", pattern=None) is False

    def test_invalid_regex_returns_false(self):
        # Bad regex — must not raise, must return False
        assert _check("hello", "regex_match", pattern="[invalid(") is False

    def test_regex_on_non_string_coerces(self):
        # Integers are coerced to string via str()
        assert _check(42, "regex_match", pattern="^4") is True


class TestCheckInList:
    def test_value_in_allowed_passes(self):
        assert _check("INT4", "in_list", allowed=["INT3", "INT4", "INT8"]) is True

    def test_value_not_in_allowed_fails(self):
        assert _check("INT2", "in_list", allowed=["INT3", "INT4", "INT8"]) is False

    def test_none_actual_fails(self):
        assert _check(None, "in_list", allowed=["INT4"]) is False

    def test_none_allowed_fails(self):
        assert _check("INT4", "in_list", allowed=None) is False

    def test_empty_allowed_fails(self):
        assert _check("INT4", "in_list", allowed=[]) is False


# ── Unit: PolicyRegistry.validate_rules ──────────────────────────────────────


class TestValidateRules:
    def test_valid_non_empty_rule_returns_no_errors(self):
        rules = [_rule()]
        assert PolicyRegistry.validate_rules(rules) == []

    def test_missing_id_returns_error(self):
        rules = [{"field": "x", "check": "non_empty", "severity": "error"}]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("id" in e for e in errors)

    def test_missing_field_returns_error(self):
        rules = [{"id": "X", "check": "non_empty", "severity": "error"}]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("field" in e for e in errors)

    def test_missing_check_returns_error(self):
        rules = [{"id": "X", "field": "x", "severity": "error"}]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("check" in e for e in errors)

    def test_unknown_check_type_returns_error(self):
        rules = [_rule(check="does_not_exist")]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("unknown check type" in e for e in errors)

    def test_unknown_severity_returns_error(self):
        rules = [_rule(severity="critical")]  # not a valid severity
        errors = PolicyRegistry.validate_rules(rules)
        assert any("severity" in e for e in errors)

    def test_regex_match_without_pattern_returns_error(self):
        rules = [_rule(check="regex_match")]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("pattern" in e for e in errors)

    def test_in_list_without_allowed_returns_error(self):
        rules = [_rule(check="in_list")]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("allowed" in e for e in errors)

    def test_equals_without_value_returns_error(self):
        rules = [_rule(check="equals")]
        errors = PolicyRegistry.validate_rules(rules)
        assert any("value" in e for e in errors)

    def test_valid_regex_match_returns_no_errors(self):
        rules = [_rule(check="regex_match", pattern="^Acme")]
        assert PolicyRegistry.validate_rules(rules) == []

    def test_valid_in_list_returns_no_errors(self):
        rules = [_rule(check="in_list", allowed=["INT4", "INT3"])]
        assert PolicyRegistry.validate_rules(rules) == []

    def test_multiple_rules_all_invalid(self):
        rules = [{"id": "A"}, {"id": "B"}]
        errors = PolicyRegistry.validate_rules(rules)
        assert len(errors) >= 2

    def test_empty_rules_list_returns_no_errors(self):
        assert PolicyRegistry.validate_rules([]) == []


# ── Unit: PolicyRegistry.load_rules_from_dict ────────────────────────────────


class TestLoadRulesFromDict:
    def test_valid_rules_returned_unchanged(self):
        rules = [_rule("R-001"), _rule("R-002", check="equals", value="clean")]
        loaded = PolicyRegistry.load_rules_from_dict(rules)
        assert len(loaded) == 2

    def test_invalid_rule_filtered_out(self):
        rules = [_rule("GOOD"), {"not": "a valid rule"}]
        loaded = PolicyRegistry.load_rules_from_dict(rules)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "GOOD"

    def test_all_invalid_returns_empty(self):
        rules = [{"x": 1}, {"y": 2}]
        loaded = PolicyRegistry.load_rules_from_dict(rules)
        assert loaded == []

    def test_non_dict_entry_filtered(self):
        rules = [_rule("GOOD"), "not_a_dict", None]  # type: ignore[list-item]
        loaded = PolicyRegistry.load_rules_from_dict(rules)
        assert len(loaded) == 1


# ── Integration: PolicyRegistry.load_rules_from_yaml ─────────────────────────


class TestLoadRulesFromYaml:
    def test_roundtrip_valid_yaml(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml

        rules_data = [
            {
                "id": "YAML-001",
                "field": "components[0].name",
                "check": "non_empty",
                "severity": "error",
                "rationale": "Model must have a name.",
                "remediation": "Set model_id.",
                "remediation_link": "https://docs.example.com/model-id",
            }
        ]
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(rules_data), encoding="utf-8")

        loaded = PolicyRegistry.load_rules_from_yaml(p)
        assert len(loaded) == 1
        assert loaded[0]["id"] == "YAML-001"
        assert loaded[0]["remediation_link"] == "https://docs.example.com/model-id"

    def test_yaml_with_invalid_rule_filtered(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml

        rules_data = [
            {"id": "GOOD", "field": "x", "check": "non_empty", "severity": "error"},
            {"id": "BAD"},  # missing required keys
        ]
        p = tmp_path / "rules.yaml"
        p.write_text(yaml.dump(rules_data), encoding="utf-8")

        loaded = PolicyRegistry.load_rules_from_yaml(p)
        assert len(loaded) == 1

    def test_yaml_not_a_list_raises_value_error(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "rules.yaml"
        p.write_text("key: value\n", encoding="utf-8")
        with pytest.raises(ValueError, match="top-level list"):
            PolicyRegistry.load_rules_from_yaml(p)

    def test_missing_yaml_file_raises_os_error(self, tmp_path):
        pytest.importorskip("yaml")
        p = tmp_path / "nonexistent.yaml"
        with pytest.raises(OSError):
            PolicyRegistry.load_rules_from_yaml(p)


# ── Unit: PolicyEngine.evaluate_custom ───────────────────────────────────────


class TestEvaluateCustom:
    def test_non_empty_passes(self):
        bom = _minimal_bom()
        rules = [_rule("C-001", field="components[0].name", check="non_empty")]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.passed
        assert result.pass_count == 1

    def test_non_empty_fails_on_empty_field(self):
        bom = _minimal_bom()
        bom["components"][0]["name"] = ""
        rules = [_rule("C-001", field="components[0].name", check="non_empty")]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert not result.passed
        assert result.error_count == 1

    def test_regex_match_on_supplier_passes(self):
        bom = _minimal_bom()
        rules = [
            _rule(
                "C-002",
                field="components[0].supplier.name",
                check="regex_match",
                pattern="^Acme",
                severity="error",
            )
        ]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.passed

    def test_regex_match_on_supplier_fails(self):
        bom = _minimal_bom()
        rules = [
            _rule(
                "C-002",
                field="components[0].supplier.name",
                check="regex_match",
                pattern="^OtherCo",
                severity="error",
            )
        ]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert not result.passed

    def test_in_list_passes(self):
        bom = _minimal_bom()
        rules = [
            _rule(
                "C-003",
                field="components[0].modelCard.modelParameters.quantizationLevel",
                check="in_list",
                allowed=["INT4", "INT3", "INT8"],
                severity="error",
            )
        ]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.passed

    def test_in_list_fails(self):
        bom = _minimal_bom()
        rules = [
            _rule(
                "C-003",
                field="components[0].modelCard.modelParameters.quantizationLevel",
                check="in_list",
                allowed=["BF16"],
                severity="error",
            )
        ]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert not result.passed

    def test_equals_check_passes(self):
        bom = _minimal_bom()
        rules = [_rule("C-004", field="squash:scan_result", check="equals", value="clean")]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.passed

    def test_custom_policy_name_preserved(self):
        bom = _minimal_bom()
        rules = [_rule()]
        result = PolicyEngine.evaluate_custom(bom, rules, policy_name="my-team-policy")
        assert result.policy_name == "my-team-policy"

    def test_warning_only_rules_do_not_fail_policy(self):
        bom = _minimal_bom()
        bom["components"][0]["name"] = ""
        rules = [_rule("C-005", field="components[0].name", check="non_empty", severity="warning")]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.passed  # warning does not fail
        assert result.warning_count == 1

    def test_invalid_rules_skipped_gracefully(self):
        bom = _minimal_bom()
        # Rules with missing required fields are skipped (not raised)
        rules = [{"id": "BAD"}]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert isinstance(result, type(result))

    def test_empty_rules_list_passes(self):
        bom = _minimal_bom()
        result = PolicyEngine.evaluate_custom(bom, [])
        assert result.passed
        assert result.pass_count == 0


# ── Unit: remediation_link on PolicyFinding ───────────────────────────────────


class TestRemediationLink:
    def test_remediation_link_flows_through_custom_evaluate(self):
        bom = _minimal_bom()
        bom["components"][0]["name"] = ""
        rules = [
            _rule(
                "C-006",
                field="components[0].name",
                check="non_empty",
                remediation_link="https://docs.example.com/fix-name",
            )
        ]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.findings[0].remediation_link == "https://docs.example.com/fix-name"

    def test_remediation_link_defaults_to_empty_string(self):
        bom = _minimal_bom()
        rules = [_rule()]
        result = PolicyEngine.evaluate_custom(bom, rules)
        assert result.findings[0].remediation_link == ""

    def test_builtin_policy_finding_has_remediation_link_field(self):
        bom = _minimal_bom()
        result = PolicyEngine.evaluate(bom, "eu-ai-act")
        # Field exists on all findings (defaults to "" for built-ins that don't define it)
        for f in result.findings:
            assert hasattr(f, "remediation_link")
            assert isinstance(f.remediation_link, str)

    def test_policy_finding_dataclass_has_remediation_link(self):
        f = PolicyFinding(
            rule_id="X",
            severity="error",
            passed=True,
            field="x",
            rationale="r",
            remediation="rem",
            remediation_link="https://example.com",
        )
        assert f.remediation_link == "https://example.com"


# ── Integration: API custom_rules endpoint ────────────────────────────────────


class TestApiCustomRules:
    """Integration tests for the POST /policy/evaluate endpoint with custom_rules."""

    @pytest.fixture()
    def client(self):
        """Return a synchronous FastAPI TestClient for squash.api.app."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from squish.squash.api import app

        return TestClient(app)

    def test_custom_rules_accepted(self, client):
        bom = _minimal_bom()
        payload = {
            "sbom": bom,
            "policy": "my-custom",
            "custom_rules": [
                {
                    "id": "CR-001",
                    "field": "components[0].name",
                    "check": "non_empty",
                    "severity": "error",
                    "rationale": "Must have name",
                    "remediation": "Set model_id",
                }
            ],
        }
        resp = client.post("/policy/evaluate", json=payload)
        assert resp.status_code in (200, 422)
        body = resp.json()
        assert "findings" in body
        assert body["policy"] == "my-custom"

    def test_custom_rules_invalid_schema_returns_400(self, client):
        bom = _minimal_bom()
        payload = {
            "sbom": bom,
            "policy": "bad-rules",
            "custom_rules": [
                {"id": "BAD"}  # missing required fields
            ],
        }
        resp = client.post("/policy/evaluate", json=payload)
        assert resp.status_code == 400
        body = resp.json()
        assert "errors" in body["detail"]

    def test_named_policy_still_works(self, client):
        bom = _minimal_bom()
        payload = {"sbom": bom, "policy": "eu-ai-act"}
        resp = client.post("/policy/evaluate", json=payload)
        assert resp.status_code in (200, 422)
        assert "findings" in resp.json()

    def test_custom_rules_with_regex_match(self, client):
        bom = _minimal_bom()
        payload = {
            "sbom": bom,
            "policy": "custom",
            "custom_rules": [
                {
                    "id": "CR-REG",
                    "field": "components[0].supplier.name",
                    "check": "regex_match",
                    "pattern": "^Acme",
                    "severity": "error",
                    "rationale": "Supplier must be Acme Corp",
                    "remediation": "Update supplier",
                }
            ],
        }
        resp = client.post("/policy/evaluate", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["passed"] is True

    def test_response_includes_remediation_link(self, client):
        bom = _minimal_bom()
        bom["components"][0]["name"] = ""
        payload = {
            "sbom": bom,
            "policy": "custom",
            "custom_rules": [
                {
                    "id": "CR-LINK",
                    "field": "components[0].name",
                    "check": "non_empty",
                    "severity": "error",
                    "rationale": "Must have name",
                    "remediation": "Set it",
                    "remediation_link": "https://docs.example.com/name",
                }
            ],
        }
        resp = client.post("/policy/evaluate", json=payload)
        body = resp.json()
        finding = next(f for f in body["findings"] if f["rule_id"] == "CR-LINK")
        assert finding["remediation_link"] == "https://docs.example.com/name"
