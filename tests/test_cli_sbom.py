"""tests/test_cli_sbom.py — Unit tests for `squish sbom` CLI subcommand (Phase 5).

All tests are pure-unit: no real model weights, no network I/O, no mutation of
sys.modules or environment variables.  The squash stack (sbom_builder,
oms_signer) is imported for real; only EvalBinder.bind and
OmsSigner.sign are patched where the action under test delegates to them.

Patch targets use source-module paths (squish.squash.sbom_builder.EvalBinder.bind
etc.) because cmd_sbom performs local imports inside the function body — patching
cli.EvalBinder.bind would miss the real call site.

W45: eval_binder.py shim deleted — EvalBinder canonical location is sbom_builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from squish.cli import cmd_sbom


# ── helpers ───────────────────────────────────────────────────────────────────

def _ns(**kwargs) -> argparse.Namespace:
    """Build a Namespace with sbom defaults."""
    defaults = {"sbom_action": "show", "model_dir": "/tmp", "result": None, "baseline": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _write_minimal_sidecar(bom_path: Path, composite_hash: str = "ab" * 32) -> None:
    """Write the minimal valid CycloneDX sidecar fixture to *bom_path*."""
    sidecar = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "serialNumber": "urn:uuid:test-serial-0001-aabbccddeeff",
        "components": [
            {
                "type": "machine-learning-model",
                "name": "test-model",
                "modelCard": {
                    "modelParameters": {"quantizationLevel": "INT4"},
                    "quantitativeAnalysis": {
                        "performanceMetrics": [
                            {"type": "arc_easy", "value": 70.6, "slice": "arc_easy"}
                        ]
                    },
                },
                "hashes": [{"alg": "SHA-256", "content": composite_hash}],
            }
        ],
    }
    bom_path.write_text(json.dumps(sidecar))


def _compute_composite(file_path: Path) -> str:
    """Replicate CycloneDXBuilder logic: per-file sha256, then sha256(concat).

    For a single file this equals sha256(sha256(file_bytes).hexdigest().encode()).
    """
    sha = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            sha.update(chunk)
    file_hex = sha.hexdigest()
    return hashlib.sha256(file_hex.encode()).hexdigest()


# ── tests ─────────────────────────────────────────────────────────────────────

def test_show_prints_metrics(capsys):
    """show prints component name and at least one metrics line."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_sidecar(Path(tmp) / "cyclonedx-mlbom.json")
        cmd_sbom(_ns(sbom_action="show", model_dir=tmp))
    out = capsys.readouterr().out
    assert "test-model" in out
    assert "arc_easy" in out


def test_show_exits_1_no_sidecar():
    """show exits 1 when no sidecar exists."""
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(SystemExit) as exc_info:
            cmd_sbom(_ns(sbom_action="show", model_dir=tmp))
    assert exc_info.value.code == 1


def test_verify_ok(capsys):
    """verify exits 0 and prints ✓ when weights match sidecar hash."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        weight_file = tmp_path / "tensors.npy"
        weight_file.write_bytes(b"\x93NUMPY\x01\x00" + b"\x00" * 64)
        composite = _compute_composite(weight_file)
        _write_minimal_sidecar(tmp_path / "cyclonedx-mlbom.json", composite_hash=composite)
        cmd_sbom(_ns(sbom_action="verify", model_dir=tmp))
    out = capsys.readouterr().out
    assert "✓" in out


def test_verify_fail_mismatch(capsys):
    """verify exits 1 and prints stderr 'mismatch' when hashes differ."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        weight_file = tmp_path / "tensors.npy"
        weight_file.write_bytes(b"\x93NUMPY\x01\x00" + b"\x00" * 64)
        # Deliberately wrong hash
        _write_minimal_sidecar(tmp_path / "cyclonedx-mlbom.json", composite_hash="a" * 64)
        with pytest.raises(SystemExit) as exc_info:
            cmd_sbom(_ns(sbom_action="verify", model_dir=tmp))
    assert exc_info.value.code == 1
    assert "mismatch" in capsys.readouterr().err


def test_bind_calls_eval_binder():
    """bind delegates to EvalBinder.bind with correct Path arguments."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bom_path = tmp_path / "cyclonedx-mlbom.json"
        _write_minimal_sidecar(bom_path)
        result_file = tmp_path / "lmeval.json"
        result_file.write_text('{"scores": {"arc_easy": 70.6}, "raw_results": {}}')

        with patch("squish.squash.sbom_builder.EvalBinder.bind") as mock_bind:
            cmd_sbom(_ns(sbom_action="bind", model_dir=tmp, result=str(result_file)))

    mock_bind.assert_called_once_with(bom_path, result_file, None)


def test_bind_exits_1_no_sidecar():
    """bind exits 1 when no sidecar exists."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        result_file = tmp_path / "lmeval.json"
        result_file.write_text('{"scores": {}, "raw_results": {}}')
        with pytest.raises(SystemExit) as exc_info:
            cmd_sbom(_ns(sbom_action="bind", model_dir=tmp, result=str(result_file)))
    assert exc_info.value.code == 1


def test_sign_no_sigstore(capsys):
    """sign exits 0 and warns about sigstore when OmsSigner.sign returns None."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_minimal_sidecar(Path(tmp) / "cyclonedx-mlbom.json")
        with patch("squish.squash.oms_signer.OmsSigner.sign", return_value=None):
            cmd_sbom(_ns(sbom_action="sign", model_dir=tmp))
    out = capsys.readouterr().out
    assert "⚠" in out
    assert "sigstore" in out
