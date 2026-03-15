"""tests/test_catalog_hash.py — Unit tests for Phase 16A hf_sha256 + verify_hash."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from squish.catalog import (
    SQUISH_HASH_SENTINEL,
    CatalogEntry,
    verify_hash,
    write_hash_sentinel,
)


def _make_entry(sha256: str | None = None) -> CatalogEntry:
    return CatalogEntry(
        id="qwen3:8b",
        name="Qwen3-8B",
        hf_mlx_repo="mlx-community/Qwen3-8B-bf16",
        size_gb=16.0,
        squished_size_gb=4.2,
        params="8B",
        context=32768,
        hf_sha256=sha256,
    )


# ---------------------------------------------------------------------------
# CatalogEntry.hf_sha256 field
# ---------------------------------------------------------------------------

class TestCatalogEntryHfSha256:
    def test_default_is_none(self):
        entry = _make_entry()
        assert entry.hf_sha256 is None

    def test_field_set_correctly(self):
        entry = _make_entry(sha256="abc123def456")
        assert entry.hf_sha256 == "abc123def456"

    def test_field_in_entry_from_dict(self):
        from squish.catalog import _entry_from_dict
        d = {
            "id": "qwen3:8b",
            "name": "Qwen3-8B",
            "hf_mlx_repo": "mlx-community/Qwen3-8B-bf16",
            "size_gb": 16.0,
            "params": "8B",
            "context": 32768,
            "hf_sha256": "deadbeef1234",
        }
        entry = _entry_from_dict(d)
        assert entry.hf_sha256 == "deadbeef1234"

    def test_field_missing_in_dict_defaults_to_none(self):
        from squish.catalog import _entry_from_dict
        d = {
            "id": "qwen3:8b",
            "name": "Qwen3-8B",
            "hf_mlx_repo": "mlx-community/Qwen3-8B-bf16",
            "size_gb": 16.0,
            "params": "8B",
            "context": 32768,
        }
        entry = _entry_from_dict(d)
        assert entry.hf_sha256 is None


# ---------------------------------------------------------------------------
# write_hash_sentinel
# ---------------------------------------------------------------------------

class TestWriteHashSentinel:
    def test_creates_sentinel_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "abc123")
            assert (p / SQUISH_HASH_SENTINEL).exists()

    def test_file_content_matches_sha256(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "deadbeef")
            assert (p / SQUISH_HASH_SENTINEL).read_text() == "deadbeef"

    def test_strips_trailing_whitespace(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "  abc  ")
            # write_hash_sentinel strips the input
            assert (p / SQUISH_HASH_SENTINEL).read_text() == "abc"

    def test_sentinel_constant_value(self):
        assert SQUISH_HASH_SENTINEL == ".squish_hash"


# ---------------------------------------------------------------------------
# verify_hash
# ---------------------------------------------------------------------------

class TestVerifyHash:
    def test_no_hash_in_entry_returns_ok(self):
        entry = _make_entry(sha256=None)
        with tempfile.TemporaryDirectory() as td:
            ok, msg = verify_hash(entry, Path(td))
        assert ok is True
        assert msg == ""

    def test_hash_matches_returns_ok_empty_msg(self):
        entry = _make_entry(sha256="abc123")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "abc123")
            ok, msg = verify_hash(entry, p)
        assert ok is True
        assert msg == ""

    def test_hash_mismatch_returns_false_and_warning(self):
        entry = _make_entry(sha256="expected_hash")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "wrong_hash")
            ok, msg = verify_hash(entry, p)
        assert ok is False
        assert "mismatch" in msg.lower() or "WARNING" in msg
        assert "expected_hash" in msg
        assert "wrong_hash" in msg

    def test_no_sentinel_file_returns_ok_with_info_msg(self):
        """When catalog has a hash but no sentinel exists, still pass (manually compressed)."""
        entry = _make_entry(sha256="some_hash")
        with tempfile.TemporaryDirectory() as td:
            ok, msg = verify_hash(entry, Path(td))
        assert ok is True
        assert msg != ""
        assert SQUISH_HASH_SENTINEL in msg or "skipping" in msg.lower()

    def test_warning_message_contains_model_id(self):
        entry = _make_entry(sha256="expected_hash")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "wrong_hash")
            _, msg = verify_hash(entry, p)
        assert "qwen3:8b" in msg

    def test_warning_message_contains_re_download_hint(self):
        entry = _make_entry(sha256="expected")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, "actual")
            _, msg = verify_hash(entry, p)
        assert "squish pull" in msg

    def test_full_sha256_hex_string(self):
        sha = "a" * 64  # 64-char hex string like a real SHA-256
        entry = _make_entry(sha256=sha)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            write_hash_sentinel(p, sha)
            ok, msg = verify_hash(entry, p)
        assert ok is True

    def test_empty_sha256_treated_as_none(self):
        """Empty string hf_sha256 should behave like None (no verification)."""
        entry = _make_entry(sha256="")
        with tempfile.TemporaryDirectory() as td:
            ok, msg = verify_hash(entry, Path(td))
        assert ok is True
        assert msg == ""
