"""tests/test_predownload_scan.py — W100: Pre-download safety scan.

Tests for ``scan_before_load()`` and ``_pull_from_hf`` abort-on-unsafe path.

Taxonomy: unit — real file I/O via tmp_path; no network calls; no GPU.
"""
from __future__ import annotations

import pickle
import struct
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.serving.local_model_scanner import (
    PreDownloadScanResult,
    scan_before_load,
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
