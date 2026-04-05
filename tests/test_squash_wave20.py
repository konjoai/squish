"""Tests for Wave 20 — NTIA minimum elements validator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from squish.squash.policy import NtiaResult, NtiaValidator


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _write_bom(tmp_path: Path, bom: dict) -> Path:
    p = tmp_path / "cyclonedx-mlbom.json"
    p.write_text(json.dumps(bom), encoding="utf-8")
    return p


def _full_bom(tmp_path: Path) -> Path:
    """BOM with all 7 NTIA elements populated."""
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": "urn:uuid:abc123",
        "metadata": {
            "timestamp": "2024-01-01T00:00:00Z",
            "supplier": {"name": "Konjo AI"},
            "authors": [{"name": "Wesley Scholl"}],
            "tools": [{"name": "squash", "version": "1.0"}],
        },
        "components": [
            {
                "type": "library",
                "name": "torch",
                "version": "2.0.0",
                "purl": "pkg:pypi/torch@2.0.0",
                "supplier": {"name": "Meta AI"},
            }
        ],
        "dependencies": [
            {
                "ref": "pkg:pypi/torch@2.0.0",
                "dependsOn": ["pkg:pypi/numpy@1.24.0"],
            }
        ],
    }
    return _write_bom(tmp_path, bom)


def _minimal_bom(tmp_path: Path) -> Path:
    """BOM with only partial fields."""
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": [
            {"type": "library", "name": "torch", "version": "2.0.0"}
        ],
    }
    return _write_bom(tmp_path, bom)


def _empty_bom(tmp_path: Path) -> Path:
    return _write_bom(tmp_path, {})


# ── Shape / dtype contract tests ──────────────────────────────────────────────

def test_ntia_result_fields():
    """NtiaResult has the correct field names and types."""
    r = NtiaResult(
        passed=True,
        present_fields=["a"],
        missing_fields=[],
        completeness_score=1.0,
        bom_path=None,
    )
    assert isinstance(r.passed, bool)
    assert isinstance(r.present_fields, list)
    assert isinstance(r.missing_fields, list)
    assert isinstance(r.completeness_score, float)


def test_ntia_check_returns_ntia_result(tmp_path):
    r = NtiaValidator.check(_full_bom(tmp_path))
    assert isinstance(r, NtiaResult)


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_full_bom_passes(tmp_path):
    r = NtiaValidator.check(_full_bom(tmp_path))
    assert r.passed
    assert r.completeness_score == pytest.approx(1.0, abs=1e-4)
    assert not r.missing_fields


def test_empty_bom_fails(tmp_path):
    r = NtiaValidator.check(_empty_bom(tmp_path))
    assert not r.passed
    assert r.completeness_score < 1.0
    assert len(r.missing_fields) > 0


def test_partial_bom_reports_missing(tmp_path):
    r = NtiaValidator.check(_minimal_bom(tmp_path))
    assert not r.passed
    assert r.completeness_score < 1.0
    assert len(r.missing_fields) >= 3


def test_check_accepts_dict_input():
    bom = {
        "serialNumber": "urn:uuid:abc",
        "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        "components": [
            {
                "name": "torch",
                "version": "2.0",
                "purl": "pkg:pypi/torch@2.0",
                "supplier": {"name": "Meta"},
                "dependsOn": ["pkg:pypi/numpy@1.0"],
            }
        ],
        "dependencies": [{"ref": "pkg:pypi/torch@2.0", "dependsOn": ["pkg:pypi/numpy@1.0"]}],
    }
    r = NtiaValidator.check(bom)
    assert isinstance(r, NtiaResult)


def test_completeness_score_math(tmp_path):
    """Score must be n_present / 7 rounded to 4 decimals."""
    r = NtiaValidator.check(_minimal_bom(tmp_path))
    expected = round(len(r.present_fields) / 7, 4)
    assert r.completeness_score == pytest.approx(expected, abs=1e-4)


def test_strict_mode_fails_on_empty_dependsOn(tmp_path):
    """Strict mode requires non-empty dependsOn."""
    bom = {
        "serialNumber": "urn:uuid:abc",
        "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        "components": [
            {
                "name": "torch",
                "version": "2.0",
                "purl": "pkg:pypi/torch@2.0",
                "supplier": {"name": "Meta"},
            }
        ],
        "dependencies": [{"ref": "pkg:pypi/torch@2.0", "dependsOn": []}],
    }
    r = NtiaValidator.check(bom, strict=True)
    assert not r.passed
    assert "relationships" in r.missing_fields or len(r.missing_fields) > 0


def test_bom_path_preserved(tmp_path):
    p = _full_bom(tmp_path)
    r = NtiaValidator.check(p)
    assert r.bom_path == p


def test_file_not_found_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NtiaValidator.check(Path("/nonexistent/path/bom.json"))


def test_missing_fields_are_strings(tmp_path):
    r = NtiaValidator.check(_empty_bom(tmp_path))
    for f in r.missing_fields:
        assert isinstance(f, str)


def test_present_fields_are_strings(tmp_path):
    r = NtiaValidator.check(_full_bom(tmp_path))
    for f in r.present_fields:
        assert isinstance(f, str)


def test_non_strict_empty_dependsOn_still_counts(tmp_path):
    """Without strict=True, missing dependsOn should still score lower."""
    bom = {
        "serialNumber": "urn:uuid:abc",
        "metadata": {"timestamp": "2024-01-01T00:00:00Z"},
        "components": [
            {
                "name": "torch",
                "version": "2.0",
                "purl": "pkg:pypi/torch@2.0",
                "supplier": {"name": "Meta"},
            }
        ],
    }
    r_non_strict = NtiaValidator.check(bom, strict=False)
    r_strict = NtiaValidator.check(bom, strict=True)
    # Strict should be no more lenient than non-strict
    assert r_strict.completeness_score <= r_non_strict.completeness_score + 1e-4
