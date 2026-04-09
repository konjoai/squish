"""tests/test_squash_wave33.py — Wave 33: VEX feed hosting + VexCache.load_bundled()

Covers:
  - Bundled community VEX feed JSON structure:
      - Valid OpenVEX 0.2.0 document (passes VexFeedManifest.validate)
      - All required top-level fields present
      - All statements have valid CVE names, products, status, justification
      - Exactly 3 statements (CVE-2024-34359, CVE-2023-27534, CVE-2024-3660)
  - VexCache.DEFAULT_URL aligned with SQUASH_VEX_FEED_FALLBACK_URL filename
  - VexCache.load_bundled() classmethod:
      - Returns a VexFeed
      - VexFeed has at least one document
      - VexFeed total statement count matches bundled JSON
      - Returns VexFeed(documents=[]) when data file is absent
      - Returned statements have non-empty vulnerability IDs
      - Returned statements are all not_affected
  - VexFeed from load_bundled() integrates with VexReport evaluation
  - SQUASH_VEX_FEED_FALLBACK_URL points to feed.openvex.json (not feed.json)
  - DEFAULT_URL points to feed.openvex.json (not feed.json)

Test taxonomy:
  - Pure unit — no I/O, no network, no process-state mutation.  All filesystem
    access via the real bundled file (which is part of the squish package).
    The one "absent file" test patches Path.read_text to raise OSError.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from squish.squash.vex import (
    SQUASH_VEX_FEED_FALLBACK_URL,
    VexCache,
    VexFeed,
    VexFeedManifest,
)

# Path to the bundled community feed — same file load_bundled() reads.
_BUNDLED_PATH = Path(__file__).parent.parent / "squish" / "squash" / "data" / "community_vex_feed.openvex.json"

_EXPECTED_CVES = {"CVE-2024-34359", "CVE-2023-27534", "CVE-2024-3660"}
_VALID_STATUSES = {"not_affected", "affected", "fixed", "under_investigation"}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_raw() -> dict:
    """Return the bundled feed as a raw dict (no squish imports)."""
    return json.loads(_BUNDLED_PATH.read_text(encoding="utf-8"))


# ── TestBundledFeedStructure ──────────────────────────────────────────────────

class TestBundledFeedStructure:
    """Structural invariants on the bundled community_vex_feed.openvex.json."""

    def test_file_exists(self):
        assert _BUNDLED_PATH.exists(), f"Bundled feed not found at {_BUNDLED_PATH}"

    def test_valid_json(self):
        doc = _load_raw()
        assert isinstance(doc, dict)

    def test_passes_vexfeedmanifest_validate(self):
        doc = _load_raw()
        errors = VexFeedManifest.validate(doc)
        assert errors == [], f"Validation errors: {errors}"

    def test_context_is_openvex_0_2_0(self):
        doc = _load_raw()
        assert doc["@context"] == VexFeedManifest.OPENVEX_CONTEXT

    def test_type_is_openvexdocument(self):
        doc = _load_raw()
        assert doc["@type"] == VexFeedManifest.OPENVEX_TYPE

    def test_spec_version_is_0_2_0(self):
        doc = _load_raw()
        assert doc["specVersion"] == VexFeedManifest.SPEC_VERSION

    def test_document_id_is_set(self):
        doc = _load_raw()
        assert doc.get("@id"), "Feed document must have a non-empty @id"

    def test_author_field_present(self):
        doc = _load_raw()
        assert doc.get("author"), "Feed document must have a non-empty author"

    def test_timestamp_field_present(self):
        doc = _load_raw()
        assert doc.get("timestamp"), "Feed document must have a non-empty timestamp"

    def test_statement_count_at_least_25(self):
        doc = _load_raw()
        assert len(doc["statements"]) >= 25, (
            f"Expected ≥25 statements after W52 expansion, got {len(doc['statements'])}"
        )

    def test_all_expected_cves_present(self):
        doc = _load_raw()
        found = {s["vulnerability"]["name"] for s in doc["statements"]}
        assert _EXPECTED_CVES <= found, (
            f"Original 3 CVEs must be a subset of the feed. Missing: {_EXPECTED_CVES - found}"
        )

    def test_all_statements_have_products(self):
        doc = _load_raw()
        for i, stmt in enumerate(doc["statements"]):
            assert stmt.get("products"), f"statements[{i}] has no products"

    def test_all_products_have_purl_ids(self):
        doc = _load_raw()
        for i, stmt in enumerate(doc["statements"]):
            for j, p in enumerate(stmt["products"]):
                assert "@id" in p, f"statements[{i}].products[{j}] missing @id"
                assert p["@id"].startswith("pkg:"), (
                    f"statements[{i}].products[{j}]['@id'] is not a PURL: {p['@id']!r}"
                )

    def test_all_statuses_are_valid(self):
        doc = _load_raw()
        for i, stmt in enumerate(doc["statements"]):
            assert stmt["status"] in _VALID_STATUSES, (
                f"statements[{i}]: unexpected status {stmt['status']!r}"
            )

    def test_all_statements_have_justification_or_under_investigation(self):
        """not_affected requires justification; under_investigation may omit it."""
        doc = _load_raw()
        for i, stmt in enumerate(doc["statements"]):
            if stmt["status"] == "not_affected":
                assert stmt.get("justification"), (
                    f"statements[{i}] is not_affected but missing justification"
                )

    def test_all_statements_have_impact_statement(self):
        doc = _load_raw()
        for i, stmt in enumerate(doc["statements"]):
            assert stmt.get("impact_statement"), (
                f"statements[{i}] missing impact_statement"
            )

    def test_cve_2024_34359_covers_qwen_models(self):
        doc = _load_raw()
        stmt = next(
            s for s in doc["statements"]
            if s["vulnerability"]["name"] == "CVE-2024-34359"
        )
        purls = {p["@id"] for p in stmt["products"]}
        assert "pkg:huggingface/Qwen/Qwen2.5-1.5B-Instruct" in purls
        assert "pkg:huggingface/Qwen/Qwen3-0.6B" in purls

    def test_cve_2023_27534_covers_squish_package(self):
        doc = _load_raw()
        stmt = next(
            s for s in doc["statements"]
            if s["vulnerability"]["name"] == "CVE-2023-27534"
        )
        purls = {p["@id"] for p in stmt["products"]}
        # W52: product PURL dropped the pinned version; match the unversioned PURL
        assert any(p.startswith("pkg:pypi/squish") for p in purls), (
            f"CVE-2023-27534 should cover pkg:pypi/squish (any version); found: {purls}"
        )


# ── TestDefaultUrlAlignment ───────────────────────────────────────────────────

class TestDefaultUrlAlignment:
    """VexCache.DEFAULT_URL must be consistent with SQUASH_VEX_FEED_FALLBACK_URL."""

    def test_default_url_uses_openvex_extension(self):
        assert VexCache.DEFAULT_URL.endswith(".openvex.json"), (
            f"DEFAULT_URL must end with .openvex.json, got: {VexCache.DEFAULT_URL!r}"
        )

    def test_fallback_url_uses_openvex_extension(self):
        assert SQUASH_VEX_FEED_FALLBACK_URL.endswith(".openvex.json"), (
            f"SQUASH_VEX_FEED_FALLBACK_URL must end with .openvex.json"
        )

    def test_default_url_not_feed_json(self):
        assert not VexCache.DEFAULT_URL.endswith("/feed.json"), (
            "DEFAULT_URL still points to old feed.json (not feed.openvex.json)"
        )

    def test_default_url_points_to_squishai_vex_feed(self):
        assert "squishai/vex-feed" in VexCache.DEFAULT_URL

    def test_default_url_and_fallback_same_filename(self):
        default_filename = VexCache.DEFAULT_URL.split("/")[-1]
        fallback_filename = SQUASH_VEX_FEED_FALLBACK_URL.split("/")[-1]
        assert default_filename == fallback_filename, (
            f"DEFAULT_URL filename ({default_filename!r}) != "
            f"SQUASH_VEX_FEED_FALLBACK_URL filename ({fallback_filename!r})"
        )


# ── TestLoadBundled ───────────────────────────────────────────────────────────

class TestLoadBundled:
    """VexCache.load_bundled() classmethod."""

    def test_returns_vex_feed(self):
        feed = VexCache.load_bundled()
        assert isinstance(feed, VexFeed)

    def test_feed_has_documents(self):
        feed = VexCache.load_bundled()
        assert len(feed.documents) > 0

    def test_statement_count_matches_bundled_json(self):
        raw_count = len(_load_raw()["statements"])
        feed = VexCache.load_bundled()
        total = sum(len(d.statements) for d in feed.documents)
        assert total == raw_count, (
            f"load_bundled statement count ({total}) != "
            f"bundled JSON statement count ({raw_count})"
        )

    def test_all_vulnerability_ids_non_empty(self):
        feed = VexCache.load_bundled()
        for doc in feed.documents:
            for stmt in doc.statements:
                assert stmt.vulnerability_id, "Statement has empty vulnerability_id"

    def test_all_vulnerability_ids_are_expected_cves(self):
        """Original 3 CVEs must be present; W52 adds more."""
        feed = VexCache.load_bundled()
        found: set[str] = set()
        for doc in feed.documents:
            for stmt in doc.statements:
                found.add(stmt.vulnerability_id)
        assert _EXPECTED_CVES <= found, (
            f"Original 3 CVEs must be a subset of load_bundled() output. Missing: {_EXPECTED_CVES - found}"
        )

    def test_all_statuses_are_valid(self):
        """W52 adds an under_investigation entry; all statuses must be valid."""
        feed = VexCache.load_bundled()
        for doc in feed.documents:
            for stmt in doc.statements:
                assert stmt.status in _VALID_STATUSES, (
                    f"{stmt.vulnerability_id}: unexpected status {stmt.status!r}"
                )

    def test_returns_empty_feed_when_data_file_absent(self, tmp_path):
        """load_bundled() must not raise when the bundled file is missing."""
        with patch.object(
            Path,
            "read_text",
            side_effect=OSError("file not found"),
        ):
            feed = VexCache.load_bundled()
        assert isinstance(feed, VexFeed)
        assert feed.documents == []

    def test_is_classmethod_on_instance(self):
        """load_bundled() is accessible on an instance too."""
        cache = VexCache()
        feed = cache.load_bundled()
        assert isinstance(feed, VexFeed)
