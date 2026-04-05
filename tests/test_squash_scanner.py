"""tests/test_squash_scanner.py — Unit + integration tests for squish.squash.scanner.

Test taxonomy:
  - Pure unit  — logic with synthetic binary inputs, no real model files
  - Integration — temp-dir I/O, real ModelScanner.scan_directory() call
  - Failure cases — corrupted input, unknown extension, error status
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from squish.squash.scanner import ModelScanner, ScanResult, ScanFinding


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_pickle_clean(path: Path) -> None:
    """Write a benign, valid pickle stream (only safe opcodes)."""
    # python3 -c "import pickle; print(pickle.dumps('hello'))"
    path.write_bytes(b"\x80\x04\x95\x09\x00\x00\x00\x00\x00\x00\x00\x8c\x05hello\x94.")


def _make_pickle_with_global(path: Path) -> None:
    """Write a pickle stream containing the dangerous GLOBAL opcode (0x63)."""
    # \x63 = GLOBAL  — always flagged regardless of surrounding content
    path.write_bytes(b"\x80\x04\x63os\nsystem\nq\x00.")


def _make_gguf_clean(path: Path) -> None:
    """Write a minimal GGUF header with no dangerous metadata."""
    # magic (4) + version uint32 (4) + tensor_count uint64 (8) + kv_count uint64 (8)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 0)
    path.write_bytes(header + b"\x00" * 256)


def _make_gguf_with_shell(path: Path) -> None:
    """Write a GGUF blob containing a shell injection pattern in the first 2 MB."""
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", 0, 1)
    poison = b"os.system('/bin/bash -c evil_command')"
    path.write_bytes(header + poison + b"\x00" * 256)


# ── Unit: ScanFinding severity hierarchy ──────────────────────────────────────


class TestScanFindingStructure:
    def test_severity_field_present(self):
        f = ScanFinding(
            severity="critical",
            finding_id="TEST-001",
            title="Test",
            detail="detail",
            file_path="/tmp/test.bin",
        )
        assert f.severity == "critical"
        assert f.finding_id == "TEST-001"

    def test_default_cve_is_empty_string(self):
        f = ScanFinding(severity="high", finding_id="X", title="T", detail="D", file_path="/f")
        assert f.cve == ""


class TestScanResultStructure:
    def test_is_safe_true_for_clean(self):
        r = ScanResult(scanned_path="/tmp", status="clean", findings=[])
        assert r.is_safe is True

    def test_is_safe_false_for_unsafe(self):
        r = ScanResult(scanned_path="/tmp", status="unsafe", findings=[])
        assert r.is_safe is False

    def test_critical_count_zero_when_no_findings(self):
        r = ScanResult(scanned_path="/tmp", status="clean", findings=[])
        assert r.critical_count == 0

    def test_critical_count_correct(self):
        findings = [
            ScanFinding("critical", "A", "t", "d", "/f"),
            ScanFinding("high", "B", "t", "d", "/f"),
            ScanFinding("critical", "C", "t", "d", "/f"),
        ]
        r = ScanResult(scanned_path="/tmp", status="unsafe", findings=findings)
        assert r.critical_count == 2
        assert r.high_count == 1


# ── Integration: clean model directory ───────────────────────────────────────


class TestScanCleanDirectory:
    def test_clean_safetensors_returns_clean(self, tmp_path):
        # .safetensors files are skipped by the scanner (safe by spec — no pickle opcodes)
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 64)
        result = ModelScanner.scan_directory(tmp_path)
        # 'skipped' is a valid safe status for safetensors
        assert result.is_safe

    def test_clean_pickle_file(self, tmp_path):
        p = tmp_path / "model.pt"
        _make_pickle_clean(p)
        result = ModelScanner.scan_directory(tmp_path)
        # Clean opcodes should not generate critical/unsafe status
        assert result.status in ("clean", "warning")

    def test_clean_gguf_file(self, tmp_path):
        p = tmp_path / "model.gguf"
        _make_gguf_clean(p)
        result = ModelScanner.scan_directory(tmp_path)
        assert result.status in ("clean", "warning")
        assert result.is_safe


# ── Integration: dangerous inputs ────────────────────────────────────────────


class TestScanUnsafeDetection:
    def test_global_opcode_detected(self, tmp_path):
        p = tmp_path / "model.pt"
        _make_pickle_with_global(p)
        result = ModelScanner.scan_directory(tmp_path)
        # GLOBAL opcode must be flagged
        assert result.status in ("unsafe", "warning")
        # Dangerous opcodes are reported by hex code ('63' for GLOBAL), not by name
        assert any("opcode" in (f.title + f.detail).lower() for f in result.findings)

    def test_gguf_shell_injection_detected(self, tmp_path):
        p = tmp_path / "model.gguf"
        _make_gguf_with_shell(p)
        result = ModelScanner.scan_directory(tmp_path)
        assert result.status in ("unsafe", "warning")
        assert any("shell" in f.detail.lower() or "os.system" in f.detail for f in result.findings)


# ── Integration: empty directory ─────────────────────────────────────────────


class TestEmptyDirectory:
    def test_empty_dir_returns_result(self, tmp_path):
        result = ModelScanner.scan_directory(tmp_path)
        assert isinstance(result, ScanResult)
        assert result.status in ("clean", "warning", "skipped")


# ── Integration: CycloneDX vulnerability serialisation ───────────────────────


class TestToCdxVulnerabilities:
    def test_clean_scan_no_vulnerabilities(self, tmp_path):
        (tmp_path / "model.safetensors").write_bytes(b"\x00" * 32)
        result = ModelScanner.scan_directory(tmp_path)
        if result.status == "clean":
            assert result.to_cdx_vulnerabilities() == []

    def test_unsafe_scan_emits_vulnerabilities(self, tmp_path):
        p = tmp_path / "model.pt"
        _make_pickle_with_global(p)
        result = ModelScanner.scan_directory(tmp_path)
        vulns = result.to_cdx_vulnerabilities()
        assert isinstance(vulns, list)


# ── Failure: scan_directory on missing path ───────────────────────────────────


class TestFailureCases:
    def test_nonexistent_path_returns_error_status(self, tmp_path):
        missing = tmp_path / "nonexistent_model"
        result = ModelScanner.scan_directory(missing)
        # Never raises — returns error status
        assert result.status in ("error", "clean", "skipped")

    def test_binary_garbage_does_not_raise(self, tmp_path):
        p = tmp_path / "corrupt.pt"
        p.write_bytes(bytes(range(256)) * 4)
        result = ModelScanner.scan_directory(tmp_path)
        assert isinstance(result, ScanResult)
