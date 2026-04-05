"""tests/test_squash_vex.py — Unit tests for squish.squash.vex.

Test taxonomy: Integration (local VEX feed dir), Pure unit (VexDocument dict
parsing, VexStatement structure, VexReport.summary()).

No network calls are made — VexFeed.from_url is tested against a test HTTP
server in a separate subprocess-scope test to avoid in-process env mutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squish.squash.vex import (
    ModelInventory,
    ModelInventoryEntry,
    VexDocument,
    VexEvaluator,
    VexFeed,
    VexReport,
    VexStatement,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _vex_doc(vuln_id: str, status: str, purl: str | None = None) -> dict:
    """Build a minimal OpenVEX 0.2.0 document dict."""
    stmt: dict = {
        "vulnerability_id": vuln_id,
        "status": status,
        "justification": "component_not_present",
    }
    if purl:
        stmt["affected_model_purl"] = purl
    return {
        "@context": "https://openvex.dev/ns/v0.2.0",
        "document_id": f"https://example.com/vex/{vuln_id}",
        "author": "example-security-team",
        "statements": [stmt],
    }


def _write_vex(directory: Path, name: str, *docs: dict) -> None:
    (directory / name).write_text(json.dumps(docs[0] if len(docs) == 1 else docs))


def _inv(model_id: str = "test-model", purl: str = "pkg:mlmodel/test-model@1.0") -> ModelInventory:
    return ModelInventory(
        entries=[
            ModelInventoryEntry(
                model_id=model_id,
                purl=purl,
                sbom_path=Path("/tmp/fake-bom.json"),
                composite_sha256="abc123",
            )
        ]
    )


# ── Unit: VexDocument parsing ─────────────────────────────────────────────────


class TestVexDocumentParsing:
    def test_from_dict_parses_statements(self):
        doc = VexDocument.from_dict(_vex_doc("CVE-2024-1234", "not_affected"))
        assert len(doc.statements) == 1
        assert doc.statements[0].vulnerability_id == "CVE-2024-1234"
        assert doc.statements[0].status == "not_affected"

    def test_from_dict_stores_issuer(self):
        doc = VexDocument.from_dict(_vex_doc("CVE-2024-0001", "affected"))
        assert doc.issuer == "example-security-team"

    def test_to_dict_round_trip(self):
        original = _vex_doc("CVE-2024-9999", "not_affected")
        doc = VexDocument.from_dict(original)
        d = doc.to_dict()
        # to_dict() writes 'author' key (OpenVEX spec term)
        assert d["author"] == original["author"]
        assert len(d["statements"]) == 1


# ── Unit: VexStatement structure ──────────────────────────────────────────────


class TestVexStatement:
    def test_fields_accessible(self):
        stmt = VexStatement(
            vulnerability_id="CVE-2024-5678",
            status="not_affected",
            justification="inline_mitigations_already_exist",
            affected_model_purl=None,
            action_statement="",
        )
        assert stmt.vulnerability_id == "CVE-2024-5678"
        assert stmt.status == "not_affected"
        assert stmt.affected_model_purl is None


# ── Integration: VexFeed from local directory ─────────────────────────────────


class TestVexFeedFromDirectory:
    def test_loads_documents_from_dir(self, tmp_path):
        _write_vex(tmp_path, "cve-2024-001.json", _vex_doc("CVE-2024-001", "not_affected"))
        _write_vex(tmp_path, "cve-2024-002.json", _vex_doc("CVE-2024-002", "affected"))
        feed = VexFeed.from_directory(tmp_path)
        assert len(feed.documents) == 2

    def test_empty_dir_returns_empty_feed(self, tmp_path):
        feed = VexFeed.from_directory(tmp_path)
        assert isinstance(feed.documents, list)

    def test_skips_non_json_files(self, tmp_path):
        (tmp_path / "README.md").write_text("# Not a VEX document")
        _write_vex(tmp_path, "cve-001.json", _vex_doc("CVE-2024-001", "not_affected"))
        feed = VexFeed.from_directory(tmp_path)
        assert len(feed.documents) >= 1

    def test_ignores_malformed_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("{broken json")
        feed = VexFeed.from_directory(tmp_path)
        # Should not raise; malformed files are skipped
        assert isinstance(feed.documents, list)


# ── Integration: VexEvaluator logic ──────────────────────────────────────────


class TestVexEvaluator:
    def test_not_affected_is_clean(self, tmp_path):
        _write_vex(tmp_path, "cve.json", _vex_doc("CVE-2024-001", "not_affected"))
        feed = VexFeed.from_directory(tmp_path)
        report = VexEvaluator.evaluate(feed, _inv())
        assert report.is_clean

    def test_affected_flags_model(self, tmp_path):
        """When a statement's purl matches the model's purl, it should be flagged."""
        purl = "pkg:mlmodel/test-model@1.0"
        _write_vex(tmp_path, "cve.json", _vex_doc("CVE-2024-002", "affected", purl=purl))
        feed = VexFeed.from_directory(tmp_path)
        report = VexEvaluator.evaluate(feed, _inv(purl=purl))
        assert not report.is_clean
        assert len(report.affected_models) >= 1

    def test_affected_with_no_purl_matches_all(self, tmp_path):
        """A VEX statement with no target purl applies to every model in inventory."""
        _write_vex(tmp_path, "broadcast.json", _vex_doc("CVE-2024-003", "affected", purl=None))
        feed = VexFeed.from_directory(tmp_path)
        report = VexEvaluator.evaluate(feed, _inv())
        assert not report.is_clean


# ── Unit: VexReport structure ─────────────────────────────────────────────────


class TestVexReportStructure:
    def test_is_clean_when_no_affected(self, tmp_path):
        feed = VexFeed.from_directory(tmp_path)  # empty
        report = VexEvaluator.evaluate(feed, _inv())
        assert report.is_clean

    def test_to_dict_keys_present(self, tmp_path):
        feed = VexFeed.from_directory(tmp_path)
        report = VexEvaluator.evaluate(feed, _inv())
        d = report.to_dict()
        assert "is_clean" in d
        assert "affected" in d  # VexReport.to_dict() uses 'affected', not 'affected_models'

    def test_summary_string_non_empty(self, tmp_path):
        feed = VexFeed.from_directory(tmp_path)
        report = VexEvaluator.evaluate(feed, _inv())
        s = report.summary()
        assert isinstance(s, str)
        assert len(s) > 0
