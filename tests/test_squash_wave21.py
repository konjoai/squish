"""Tests for Wave 21 — SLSA provenance attestation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from squish.squash.slsa import SlsaAttestation, SlsaLevel, SlsaProvenanceBuilder


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _model_dir_with_bom(tmp_path: Path) -> Path:
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "components": [{"name": "torch", "version": "2.0", "purl": "pkg:pypi/torch@2.0"}],
    }
    (tmp_path / "cyclonedx-mlbom.json").write_text(json.dumps(bom), encoding="utf-8")
    return tmp_path


def _empty_model_dir(tmp_path: Path) -> Path:
    return tmp_path


# ── Shape / dtype contract tests ──────────────────────────────────────────────

def test_slsa_attestation_fields():
    """SlsaAttestation has correct field names and types."""
    a = SlsaAttestation(
        subject_name="my-model",
        subject_sha256="abc123",
        builder_id="https://example.com",
        level=SlsaLevel.L1,
    )
    assert isinstance(a.subject_name, str)
    assert isinstance(a.subject_sha256, str)
    assert isinstance(a.builder_id, str)
    assert isinstance(a.level, SlsaLevel)
    assert isinstance(a.invocation_id, str)
    assert isinstance(a.build_finished_on, str)
    assert isinstance(a.materials, list)


def test_slsa_level_values():
    assert SlsaLevel.L1.value == 1
    assert SlsaLevel.L2.value == 2
    assert SlsaLevel.L3.value == 3


# ── Correctness tests ─────────────────────────────────────────────────────────

def test_l1_creates_provenance_file(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    assert attest.output_path is not None
    assert attest.output_path.exists()
    stmt = json.loads(attest.output_path.read_text())
    assert stmt["predicateType"] == SlsaProvenanceBuilder._PREDICATE_TYPE


def test_l1_subject_sha256_is_hex(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    assert len(attest.subject_sha256) == 64
    int(attest.subject_sha256, 16)  # must be valid hex


def test_l1_materials_populated(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    assert len(attest.materials) >= 1
    assert "uri" in attest.materials[0]
    assert "digest" in attest.materials[0]


def test_l2_calls_oms_signer(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    with patch("squish.squash.slsa.SlsaProvenanceBuilder._sign") as mock_sign:
        SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L2)
    mock_sign.assert_called_once()


def test_l3_calls_oms_verifier(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    with patch("squish.squash.slsa.SlsaProvenanceBuilder._sign"), \
         patch("squish.squash.slsa.SlsaProvenanceBuilder._verify") as mock_verify:
        SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L3)
    mock_verify.assert_called_once()


def test_l1_does_not_call_signer(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    with patch("squish.squash.slsa.SlsaProvenanceBuilder._sign") as mock_sign:
        SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    mock_sign.assert_not_called()


def test_output_path_returned(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    assert attest.output_path is not None
    assert str(attest.output_path).endswith("squash-slsa-provenance.json")


def test_builder_id_in_statement(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(
        model_dir,
        level=SlsaLevel.L1,
        builder_id="https://custom.builder/v1",
    )
    stmt = json.loads(attest.output_path.read_text())
    assert stmt["predicate"]["runDetails"]["builder"]["id"] == "https://custom.builder/v1"


def test_empty_dir_does_not_raise(tmp_path):
    """build() must not raise even when no BOM file is present."""
    attest = SlsaProvenanceBuilder.build(tmp_path, level=SlsaLevel.L1)
    assert attest.subject_sha256 != ""


def test_provenance_attaches_to_bom(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    bom = json.loads((model_dir / "cyclonedx-mlbom.json").read_text())
    ext_refs = bom.get("externalReferences", [])
    build_meta = [r for r in ext_refs if r.get("type") == "build-meta"]
    assert len(build_meta) >= 1


def test_invocation_id_is_uuid(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L1)
    import uuid
    uuid.UUID(attest.invocation_id)  # raises if not a valid UUID


def test_level_stored_on_attestation(tmp_path):
    model_dir = _model_dir_with_bom(tmp_path)
    attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L2,
                                          builder_id="https://x")
    assert attest.level == SlsaLevel.L2


def test_sign_error_does_not_propagate(tmp_path):
    """_sign errors must be swallowed (best-effort signing)."""
    model_dir = _model_dir_with_bom(tmp_path)
    with patch("squish.squash.slsa.SlsaProvenanceBuilder._sign",
               side_effect=RuntimeError("signer unavailable")):
        # No exception expected
        attest = SlsaProvenanceBuilder.build(model_dir, level=SlsaLevel.L2)
    assert attest.output_path.exists()
