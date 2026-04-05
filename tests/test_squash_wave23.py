"""Tests for Wave 23 — AI risk assessment (EU AI Act / NIST AI RMF)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squish.squash.risk import (
    AiRiskAssessor,
    EuAiActCategory,
    NistRmfCategory,
    RiskAssessmentResult,
    RiskCategory,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_bom(tmp_path: Path, bom: dict) -> Path:
    p = tmp_path / "cyclonedx-mlbom.json"
    p.write_text(json.dumps(bom), encoding="utf-8")
    return p


def _bom_with_use_case(tmp_path: Path, use_case: str) -> Path:
    bom = {
        "components": [
            {
                "type": "machine-learning-model",
                "name": "test-model",
                "modelCard": {
                    "considerations": {
                        "useCases": [{"description": use_case}]
                    }
                },
            }
        ]
    }
    return _write_bom(tmp_path, bom)


def _empty_bom(tmp_path: Path) -> Path:
    return _write_bom(tmp_path, {})


# ── Shape / dtype contract tests ──────────────────────────────────────────────

def test_risk_assessment_result_fields():
    r = RiskAssessmentResult(
        framework="eu-ai-act",
        category=EuAiActCategory.MINIMAL,
        rationale=["no signals"],
        mitigation_required=[],
        passed=True,
    )
    assert isinstance(r.framework, str)
    assert isinstance(r.rationale, list)
    assert isinstance(r.mitigation_required, list)
    assert isinstance(r.passed, bool)


def test_risk_category_alias():
    """RiskCategory is an alias for EuAiActCategory."""
    assert RiskCategory is EuAiActCategory


def test_eu_ai_act_returns_result(tmp_path):
    r = AiRiskAssessor.assess_eu_ai_act(_empty_bom(tmp_path))
    assert isinstance(r, RiskAssessmentResult)
    assert r.framework == "eu-ai-act"


def test_nist_rmf_returns_result(tmp_path):
    r = AiRiskAssessor.assess_nist_rmf(_empty_bom(tmp_path))
    assert isinstance(r, RiskAssessmentResult)
    assert r.framework == "nist-rmf"


# ── EU AI Act correctness tests ───────────────────────────────────────────────

def test_empty_bom_eu_minimal(tmp_path):
    r = AiRiskAssessor.assess_eu_ai_act(_empty_bom(tmp_path))
    assert r.category == EuAiActCategory.MINIMAL
    assert r.passed


def test_social_scoring_eu_unacceptable(tmp_path):
    p = _bom_with_use_case(tmp_path, "citizen social scoring and behavioural profiling")
    r = AiRiskAssessor.assess_eu_ai_act(p)
    assert r.category == EuAiActCategory.UNACCEPTABLE
    assert not r.passed


def test_law_enforcement_eu_high(tmp_path):
    p = _bom_with_use_case(tmp_path, "law enforcement facial recognition system")
    r = AiRiskAssessor.assess_eu_ai_act(p)
    assert r.category in (EuAiActCategory.HIGH, EuAiActCategory.UNACCEPTABLE)
    assert not r.passed


def test_eu_high_has_mitigation(tmp_path):
    p = _bom_with_use_case(tmp_path, "law enforcement investigation")
    r = AiRiskAssessor.assess_eu_ai_act(p)
    assert len(r.mitigation_required) > 0


# ── NIST RMF correctness tests ─────────────────────────────────────────────────

def test_empty_bom_nist_moderate_or_low(tmp_path):
    r = AiRiskAssessor.assess_nist_rmf(_empty_bom(tmp_path))
    assert r.category in (NistRmfCategory.LOW, NistRmfCategory.MODERATE)


def test_autonomous_system_no_provenance_nist_critical(tmp_path):
    p = _bom_with_use_case(tmp_path, "fully autonomous decision making in production")
    r = AiRiskAssessor.assess_nist_rmf(p)
    assert r.category in (NistRmfCategory.CRITICAL, NistRmfCategory.HIGH)
    assert not r.passed


def test_nist_passes_for_low_risk(tmp_path):
    bom = {
        "components": [
            {
                "type": "library",
                "name": "torch",
                "modelCard": {
                    "considerations": {
                        "useCases": [{"description": "image compression demo"}]
                    }
                },
            }
            for _ in range(3)  # few components, benign use case
        ],
        "externalReferences": [{"type": "build-meta", "url": "provenance.json"}],
    }
    p = _write_bom(tmp_path, bom)
    r = AiRiskAssessor.assess_nist_rmf(p)
    assert r.passed


def test_eu_and_nist_independently_callable(tmp_path):
    p = _empty_bom(tmp_path)
    eu = AiRiskAssessor.assess_eu_ai_act(p)
    nist = AiRiskAssessor.assess_nist_rmf(p)
    assert eu.framework == "eu-ai-act"
    assert nist.framework == "nist-rmf"


def test_rationale_is_non_empty(tmp_path):
    p = _empty_bom(tmp_path)
    r = AiRiskAssessor.assess_eu_ai_act(p)
    assert len(r.rationale) > 0
