"""Tests for Wave 22 — BOM Merge & Composition."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squish.squash.sbom_builder import BomMerger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bom(tmp_path: Path, name: str, components: list[dict],
         vulns: list[dict] | None = None) -> Path:
    p = tmp_path / name
    data: dict = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": components,
    }
    if vulns is not None:
        data["vulnerabilities"] = vulns
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def _comp(name: str, version: str = "1.0", purl: str | None = None) -> dict:
    c: dict = {"type": "library", "name": name, "version": version}
    if purl:
        c["purl"] = purl
    return c


# ── Shape / dtype contract tests ──────────────────────────────────────────────

def test_merge_returns_dict(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("torch", purl="pkg:pypi/torch@2.0")])
    b = _bom(tmp_path, "b.json", [_comp("numpy", purl="pkg:pypi/numpy@1.0")])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    assert isinstance(result, dict)


def test_output_file_created(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("torch", purl="pkg:pypi/torch@2.0")])
    b = _bom(tmp_path, "b.json", [_comp("numpy", purl="pkg:pypi/numpy@1.0")])
    out = tmp_path / "merged.json"
    BomMerger.merge([a, b], out)
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["bomFormat"] == "CycloneDX"


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_two_empty_boms_merge(tmp_path):
    a = _bom(tmp_path, "a.json", [])
    b = _bom(tmp_path, "b.json", [])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    assert result["components"] == []


def test_dup_purl_deduplicated(tmp_path):
    torch_purl = "pkg:pypi/torch@2.0"
    a = _bom(tmp_path, "a.json", [_comp("torch", purl=torch_purl)])
    b = _bom(tmp_path, "b.json", [_comp("torch", purl=torch_purl)])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    assert len([c for c in result["components"] if c.get("purl") == torch_purl]) == 1


def test_different_purls_both_kept(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("torch", purl="pkg:pypi/torch@2.0")])
    b = _bom(tmp_path, "b.json", [_comp("numpy", purl="pkg:pypi/numpy@1.0")])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    assert len(result["components"]) == 2


def test_no_purl_components_both_kept(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("libA")])
    b = _bom(tmp_path, "b.json", [_comp("libB")])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    assert len(result["components"]) == 2


def test_vulnerabilities_unioned(tmp_path):
    vuln_a = {"id": "CVE-2024-0001", "analysis": {"state": "exploitable"}}
    vuln_b = {"id": "CVE-2024-0002", "analysis": {"state": "in_triage"}}
    a = _bom(tmp_path, "a.json", [], vulns=[vuln_a])
    b = _bom(tmp_path, "b.json", [], vulns=[vuln_b])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    ids = {v["id"] for v in result.get("vulnerabilities", [])}
    assert "CVE-2024-0001" in ids
    assert "CVE-2024-0002" in ids


def test_dup_vuln_keeps_worst_state(tmp_path):
    vuln_low = {"id": "CVE-2024-0001", "analysis": {"state": "fixed"}}
    vuln_high = {"id": "CVE-2024-0001", "analysis": {"state": "exploitable"}}
    a = _bom(tmp_path, "a.json", [], vulns=[vuln_low])
    b = _bom(tmp_path, "b.json", [], vulns=[vuln_high])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    vulns = result.get("vulnerabilities", [])
    cve = next(v for v in vulns if v["id"] == "CVE-2024-0001")
    assert cve["analysis"]["state"] == "exploitable"


def test_compositions_array_present(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("torch", purl="pkg:pypi/torch@2.0")])
    b = _bom(tmp_path, "b.json", [_comp("numpy", purl="pkg:pypi/numpy@1.0")])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a, b], out)
    assert "compositions" in result
    assert isinstance(result["compositions"], list)


def test_metadata_applied(tmp_path):
    a = _bom(tmp_path, "a.json", [])
    b = _bom(tmp_path, "b.json", [])
    out = tmp_path / "merged.json"
    meta = {"custom_key": "custom_value"}
    result = BomMerger.merge([a, b], out, metadata=meta)
    assert result["metadata"].get("custom_key") == "custom_value"


def test_single_bom_input(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("torch", purl="pkg:pypi/torch@2.0")])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a], out)
    assert len(result["components"]) == 1


def test_valid_cyclonedx_output(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("x", purl="pkg:generic/x@1")])
    out = tmp_path / "merged.json"
    result = BomMerger.merge([a], out)
    assert result["bomFormat"] == "CycloneDX"
    assert "specVersion" in result


def test_output_is_readable_json(tmp_path):
    a = _bom(tmp_path, "a.json", [_comp("x", purl="pkg:generic/x@1")])
    out = tmp_path / "merged.json"
    BomMerger.merge([a], out)
    data = json.loads(out.read_text())
    assert isinstance(data, dict)
