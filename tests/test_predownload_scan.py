"""tests/test_predownload_scan.py — W100: Pre-download safety scan.

Tests for ``scan_before_load()``, ``scan_hf_repo_metadata()``, and the
``_pull_from_hf`` abort-on-unsafe path.

Taxonomy: unit — real file I/O via tmp_path; no network calls; no GPU.
All HuggingFace API calls are mocked via unittest.mock to keep the suite
hermetic and fast.
"""
from __future__ import annotations

import io
import json
import pickle
import struct
import sys
import types
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.serving.local_model_scanner import (
    HFFileSummary,
    HFRepoScanResult,
    PreDownloadScanResult,
    _classify_hf_siblings,
    scan_before_load,
    scan_hf_repo_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_clean_pickle(path: Path) -> None:
    """Write a benign pickle (list of ints — no dangerous opcodes)."""
    path.write_bytes(pickle.dumps([1, 2, 3]))


def _write_dangerous_pickle(path: Path) -> None:
    """Write a pickle containing a REDUCE opcode (arbitrary execution)."""
    # Craft a minimal pickle that triggers REDUCE without actually running code.
    # Protocol 2, GLOBAL + REDUCE sequence (bytes won't actually execute here
    # because we never unpickle — we only scan raw opcodes).
    payload = (
        b"\x80\x02"       # PROTO 2
        b"c__builtin__\neval\n"  # GLOBAL — dangerous
        b"q\x00"          # BINPUT
        b"."              # STOP
    )
    path.write_bytes(payload)


def _write_clean_gguf(path: Path) -> None:
    path.write_bytes(b"GGUF" + b"\x00" * 100)


def _write_bad_gguf(path: Path) -> None:
    path.write_bytes(b"BADM" + b"\x00" * 100)


def _write_clean_safetensors(path: Path) -> None:
    header_json = b'{"__metadata__": {}}'
    header_len = struct.pack("<Q", len(header_json))
    path.write_bytes(header_len + header_json)


def _write_bad_safetensors_truncated(path: Path) -> None:
    """Write a safetensors file with only 3 bytes — too short for header."""
    path.write_bytes(b"\x01\x02\x03")


def _write_bad_safetensors_oversize_header(path: Path) -> None:
    """Write a safetensors file where header_len exceeds file size."""
    bogus_len = struct.pack("<Q", 10_000_000)
    path.write_bytes(bogus_len + b"x" * 8)


# ---------------------------------------------------------------------------
# PreDownloadScanResult shape
# ---------------------------------------------------------------------------

class TestPreDownloadScanResultShape:
    def test_status_field_exists(self, tmp_path):
        r = scan_before_load(tmp_path)
        assert hasattr(r, "status")

    def test_findings_field_is_list(self, tmp_path):
        r = scan_before_load(tmp_path)
        assert isinstance(r.findings, list)

    def test_scanned_field_is_int(self, tmp_path):
        r = scan_before_load(tmp_path)
        assert isinstance(r.scanned, int)

    def test_empty_dir_is_clean(self, tmp_path):
        r = scan_before_load(tmp_path)
        assert r.status == "clean"
        assert r.findings == []
        assert r.scanned == 0

    def test_missing_dir_returns_error(self, tmp_path):
        r = scan_before_load(tmp_path / "nonexistent")
        assert r.status == "error"


# ---------------------------------------------------------------------------
# Pickle scanning
# ---------------------------------------------------------------------------

class TestPickleScan:
    def test_clean_bin_file_passes(self, tmp_path):
        _write_clean_pickle(tmp_path / "model.bin")
        r = scan_before_load(tmp_path)
        assert r.status == "clean"
        assert r.scanned == 1

    def test_dangerous_bin_file_fails(self, tmp_path):
        _write_dangerous_pickle(tmp_path / "model.bin")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"
        assert any("GLOBAL" in f or "REDUCE" in f or "UNSAFE" in f for f in r.findings)

    def test_dangerous_pt_file_fails(self, tmp_path):
        _write_dangerous_pickle(tmp_path / "weights.pt")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"

    def test_dangerous_pkl_file_fails(self, tmp_path):
        _write_dangerous_pickle(tmp_path / "archive.pkl")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"

    def test_finding_contains_filename(self, tmp_path):
        _write_dangerous_pickle(tmp_path / "corrupt.bin")
        r = scan_before_load(tmp_path)
        assert any("corrupt.bin" in f for f in r.findings)


# ---------------------------------------------------------------------------
# GGUF scanning
# ---------------------------------------------------------------------------

class TestGgufScan:
    def test_clean_gguf_passes(self, tmp_path):
        _write_clean_gguf(tmp_path / "model.gguf")
        r = scan_before_load(tmp_path)
        assert r.status == "clean"
        assert r.scanned == 1

    def test_bad_magic_fails(self, tmp_path):
        _write_bad_gguf(tmp_path / "bad.gguf")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"
        assert any("magic" in f.lower() or "UNSAFE" in f for f in r.findings)


# ---------------------------------------------------------------------------
# safetensors scanning
# ---------------------------------------------------------------------------

class TestSafetensorsScan:
    def test_clean_safetensors_passes(self, tmp_path):
        _write_clean_safetensors(tmp_path / "model.safetensors")
        r = scan_before_load(tmp_path)
        assert r.status == "clean"
        assert r.scanned == 1

    def test_truncated_safetensors_fails(self, tmp_path):
        _write_bad_safetensors_truncated(tmp_path / "model.safetensors")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"

    def test_oversize_header_fails(self, tmp_path):
        _write_bad_safetensors_oversize_header(tmp_path / "model.safetensors")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"


# ---------------------------------------------------------------------------
# Mixed directory
# ---------------------------------------------------------------------------

class TestMixedDirectory:
    def test_mixed_clean_directory_passes(self, tmp_path):
        _write_clean_pickle(tmp_path / "model.bin")
        _write_clean_gguf(tmp_path / "model.gguf")
        _write_clean_safetensors(tmp_path / "model.safetensors")
        r = scan_before_load(tmp_path)
        assert r.status == "clean"
        assert r.scanned == 3

    def test_one_bad_file_fails_whole_dir(self, tmp_path):
        _write_clean_safetensors(tmp_path / "model.safetensors")
        _write_dangerous_pickle(tmp_path / "weights.bin")
        r = scan_before_load(tmp_path)
        assert r.status == "unsafe"

    def test_unknown_extensions_not_counted(self, tmp_path):
        (tmp_path / "README.md").write_text("readme")
        (tmp_path / "config.json").write_text("{}")
        r = scan_before_load(tmp_path)
        assert r.scanned == 0
        assert r.status == "clean"


# ---------------------------------------------------------------------------
# W100 — HFRepoScanResult shape
# ---------------------------------------------------------------------------

class TestHFRepoScanResultShape:
    """Verify the dataclass fields exist and have the right types."""

    def test_status_field_exists(self):
        r = HFRepoScanResult(status="safe", repo_id="owner/repo")
        assert hasattr(r, "status")

    def test_repo_id_field_exists(self):
        r = HFRepoScanResult(status="safe", repo_id="owner/repo")
        assert r.repo_id == "owner/repo"

    def test_findings_defaults_to_empty_list(self):
        r = HFRepoScanResult(status="safe", repo_id="owner/repo")
        assert r.findings == []

    def test_file_summary_defaults_to_empty_list(self):
        r = HFRepoScanResult(status="safe", repo_id="owner/repo")
        assert isinstance(r.file_summary, list)

    def test_count_fields_default_to_zero(self):
        r = HFRepoScanResult(status="safe", repo_id="owner/repo")
        assert r.total_files == 0
        assert r.safe_weight_count == 0
        assert r.dangerous_count == 0
        assert r.potentially_unsafe_count == 0

    def test_hf_file_summary_fields(self):
        s = HFFileSummary(filename="model.safetensors", size_bytes=1024)
        assert s.filename == "model.safetensors"
        assert s.size_bytes == 1024
        assert s.flagged is False
        assert s.flag_reason == ""


# ---------------------------------------------------------------------------
# W100 — _classify_hf_siblings (pure unit tests, no HTTP)
# ---------------------------------------------------------------------------

class TestClassifyHFSiblings:
    """Tests for the internal classifier — no network calls."""

    def _siblings(self, files):
        """Build a siblings list from (filename, size) tuples."""
        return [{"rfilename": f, "size": s} for f, s in files]

    def test_safetensors_only_is_safe(self):
        siblings = self._siblings([
            ("model.safetensors", 4_000_000_000),
            ("config.json", 1024),
            ("tokenizer.json", 2048),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "safe"
        assert r.findings == []
        assert r.safe_weight_count == 1

    def test_gguf_only_is_safe(self):
        siblings = self._siblings([("model.gguf", 3_000_000_000)])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "safe"

    def test_pkl_file_is_unsafe(self):
        siblings = self._siblings([
            ("model.safetensors", 1_000_000),
            ("exploit.pkl", 512),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "unsafe"
        assert r.dangerous_count == 1
        assert any("exploit.pkl" in f for f in r.findings)

    def test_pickle_extension_is_unsafe(self):
        siblings = self._siblings([("payload.pickle", 256)])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "unsafe"
        assert r.dangerous_count == 1

    def test_bin_without_safetensors_is_unsafe(self):
        """A .bin file with no .safetensors counterpart should block download."""
        siblings = self._siblings([
            ("pytorch_model.bin", 5_000_000_000),
            ("config.json", 1024),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "unsafe"
        assert r.potentially_unsafe_count == 1

    def test_bin_alongside_safetensors_is_warning(self):
        """Legacy .bin present but safe format available → warn, not block."""
        siblings = self._siblings([
            ("model.safetensors", 4_000_000_000),
            ("pytorch_model.bin", 4_000_000_000),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "warning"
        assert r.safe_weight_count == 1
        assert r.potentially_unsafe_count == 1
        assert any("WARNING" in f for f in r.findings)

    def test_pt_without_safetensors_is_unsafe(self):
        siblings = self._siblings([("weights.pt", 2_000_000_000)])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "unsafe"

    def test_total_size_bytes_accumulates(self):
        siblings = self._siblings([
            ("model.safetensors", 1_000),
            ("config.json", 500),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.total_size_bytes == 1_500

    def test_total_files_reflects_input_length(self):
        siblings = self._siblings([
            ("a.safetensors", 1),
            ("b.safetensors", 2),
            ("c.safetensors", 3),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.total_files == 3

    def test_empty_siblings_is_safe(self):
        r = _classify_hf_siblings("owner/empty-repo", [])
        assert r.status == "safe"
        assert r.total_files == 0

    def test_finding_includes_repo_id_for_bin_no_safe(self):
        siblings = self._siblings([("pytorch_model.bin", 1_000_000)])
        r = _classify_hf_siblings("myorg/myrepo", siblings)
        assert any("myorg/myrepo" in f for f in r.findings)

    def test_multiple_pkl_all_counted(self):
        siblings = self._siblings([
            ("a.pkl", 100),
            ("b.pkl", 200),
            ("model.safetensors", 1_000_000),
        ])
        r = _classify_hf_siblings("owner/repo", siblings)
        assert r.status == "unsafe"
        assert r.dangerous_count == 2


# ---------------------------------------------------------------------------
# W100 — scan_hf_repo_metadata (HTTP mocked)
# ---------------------------------------------------------------------------

def _make_hf_response(siblings):
    """Return a mock urllib.request.urlopen context manager with given siblings."""
    body = json.dumps({"siblings": siblings}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestScanHFRepoMetadata:
    """Tests for scan_hf_repo_metadata() — HTTP layer is mocked."""

    def _siblings(self, files):
        return [{"rfilename": f, "size": s} for f, s in files]

    def test_safe_repo_returns_safe_status(self):
        siblings = self._siblings([("model.safetensors", 4_000_000_000)])
        with patch("urllib.request.urlopen", return_value=_make_hf_response(siblings)):
            r = scan_hf_repo_metadata("owner/safe-model")
        assert r.status == "safe"
        assert r.repo_id == "owner/safe-model"

    def test_pkl_repo_returns_unsafe_status(self):
        siblings = self._siblings([
            ("model.safetensors", 1_000_000),
            ("exploit.pkl", 512),
        ])
        with patch("urllib.request.urlopen", return_value=_make_hf_response(siblings)):
            r = scan_hf_repo_metadata("owner/evil-model")
        assert r.status == "unsafe"
        assert r.dangerous_count == 1

    def test_bin_no_safe_returns_unsafe(self):
        siblings = self._siblings([("pytorch_model.bin", 5_000_000_000)])
        with patch("urllib.request.urlopen", return_value=_make_hf_response(siblings)):
            r = scan_hf_repo_metadata("owner/legacy-model")
        assert r.status == "unsafe"

    def test_warning_when_bin_and_safetensors(self):
        siblings = self._siblings([
            ("model.safetensors", 4_000_000_000),
            ("pytorch_model.bin", 4_000_000_000),
        ])
        with patch("urllib.request.urlopen", return_value=_make_hf_response(siblings)):
            r = scan_hf_repo_metadata("owner/dual-format-model")
        assert r.status == "warning"

    def test_http_404_returns_error(self):
        exc = urllib.error.HTTPError(
            url="http://x", code=404, msg="Not Found", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            r = scan_hf_repo_metadata("owner/missing-repo")
        assert r.status == "error"
        assert any("404" in f for f in r.findings)

    def test_http_401_returns_error_with_hint(self):
        exc = urllib.error.HTTPError(
            url="http://x", code=401, msg="Unauthorized", hdrs=None, fp=None
        )
        with patch("urllib.request.urlopen", side_effect=exc):
            r = scan_hf_repo_metadata("owner/private-repo")
        assert r.status == "error"
        assert any("401" in f or "token" in f.lower() for f in r.findings)

    def test_network_error_returns_error(self):
        exc = urllib.error.URLError(reason="Name or service not known")
        with patch("urllib.request.urlopen", side_effect=exc):
            r = scan_hf_repo_metadata("owner/any-model")
        assert r.status == "error"
        assert r.findings

    def test_token_passed_as_auth_header(self):
        siblings = self._siblings([("model.safetensors", 1_000)])
        captured_headers = {}

        def _fake_urlopen(req, timeout=None):
            captured_headers.update(dict(req.headers))
            return _make_hf_response(siblings)

        with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
            scan_hf_repo_metadata("owner/private-model", token="hf_abc123")

        assert "Authorization" in captured_headers
        assert "hf_abc123" in captured_headers["Authorization"]

    def test_file_summary_populated(self):
        siblings = self._siblings([
            ("model.safetensors", 4_000_000_000),
            ("config.json", 1024),
        ])
        with patch("urllib.request.urlopen", return_value=_make_hf_response(siblings)):
            r = scan_hf_repo_metadata("owner/model")
        assert len(r.file_summary) == 2
        filenames = [f.filename for f in r.file_summary]
        assert "model.safetensors" in filenames

    def test_safe_weight_count_correct(self):
        siblings = self._siblings([
            ("shard-1.safetensors", 2_000_000_000),
            ("shard-2.safetensors", 2_000_000_000),
            ("model.gguf", 4_000_000_000),
        ])
        with patch("urllib.request.urlopen", return_value=_make_hf_response(siblings)):
            r = scan_hf_repo_metadata("owner/model")
        assert r.safe_weight_count == 3

    def test_unexpected_api_structure_returns_error(self):
        """If 'siblings' is not a list the scan must return error, not crash."""
        body = json.dumps({"siblings": "not-a-list"}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = body
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_resp):
            r = scan_hf_repo_metadata("owner/broken-api")
        assert r.status == "error"

    def test_empty_repo_is_safe(self):
        with patch("urllib.request.urlopen", return_value=_make_hf_response([])):
            r = scan_hf_repo_metadata("owner/empty-repo")
        assert r.status == "safe"
        assert r.total_files == 0
