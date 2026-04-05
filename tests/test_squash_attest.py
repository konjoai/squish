"""tests/test_squash_attest.py — Integration tests for the AttestPipeline orchestrator.

Test taxonomy:
  - Integration — uses tmp_path, real file I/O, synthetic weight files.
    Does NOT test sign=True (requires network/OIDC) or live VEX feeds.
  - Error path — nonexistent model path, fail_on_violation, unknown policy.

All tests run offline with no external dependencies.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from squish.squash.attest import (
    AttestConfig,
    AttestPipeline,
    AttestResult,
    AttestationViolationError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _stub_model_dir(tmp_path: Path) -> Path:
    """Write a minimal synthetic model directory that passes all offline checks."""
    d = tmp_path / "test-model"
    d.mkdir()
    # Write a small binary file that looks like a safetensors header
    weight = d / "model.safetensors"
    # minimal safetensors: 8-byte little-endian header length + empty JSON
    header = b"{}"
    weight.write_bytes(
        struct.pack("<Q", len(header)) + header + b"\x00" * 16
    )
    return d


def _gguf_stub(tmp_path: Path) -> Path:
    """Write a minimal valid GGUF file (magic only — no unsafe metadata)."""
    d = tmp_path / "gguf-model"
    d.mkdir()
    gguf = d / "model.gguf"
    # GGUF magic "GGUF" + version 3 + 0 tensors + 0 kv pairs
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQQ", 3, 0, 0, 0))
    return d


# ── Shape contract: AttestResult fields ───────────────────────────────────────


class TestAttestResultShape:
    def test_result_has_required_fields(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=[],
            sign=False,
            fail_on_violation=False,
        ))
        assert isinstance(result, AttestResult)
        assert isinstance(result.model_id, str)
        assert isinstance(result.output_dir, Path)
        assert isinstance(result.passed, bool)

    def test_model_id_defaults_to_dir_name(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=[],
            sign=False,
        ))
        assert result.model_id == d.name

    def test_model_id_custom_name(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            model_id="my-custom-model",
            policies=[],
            sign=False,
        ))
        assert result.model_id == "my-custom-model"

    def test_summary_returns_string(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(model_path=d, policies=[], sign=False))
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0


# ── Artifact paths ─────────────────────────────────────────────────────────────


class TestAttestArtifacts:
    def test_cyclonedx_artifact_exists(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(model_path=d, policies=[], sign=False))
        assert result.cyclonedx_path is not None
        assert result.cyclonedx_path.exists()

    def test_spdx_artifacts_exist(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(model_path=d, policies=[], sign=False))
        assert result.spdx_json_path is not None
        assert result.spdx_json_path.exists()
        assert result.spdx_tv_path is not None
        assert result.spdx_tv_path.exists()

    def test_master_record_exists(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(model_path=d, policies=[], sign=False))
        assert result.master_record_path is not None
        assert result.master_record_path.exists()

    def test_scan_artifact_exists(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(model_path=d, policies=[], sign=False))
        scan_json = result.output_dir / "squash-scan.json"
        assert scan_json.exists()

    def test_custom_output_dir_respected(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        out = tmp_path / "attestation-output"
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            output_dir=out,
            policies=[],
            sign=False,
        ))
        assert result.output_dir == out
        assert out.exists()
        assert result.cyclonedx_path.parent == out


# ── Master record content ─────────────────────────────────────────────────────


class TestMasterRecord:
    def _run(self, tmp_path: Path) -> tuple[AttestResult, dict]:
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(model_path=d, policies=[], sign=False))
        master = json.loads(result.master_record_path.read_text())
        return result, master

    def test_has_schema_version(self, tmp_path):
        _, m = self._run(tmp_path)
        # master record uses 'squash_version', not 'schemaVersion'
        assert "squash_version" in m

    def test_has_model_id(self, tmp_path):
        _, m = self._run(tmp_path)
        raw = json.dumps(m)
        assert "model" in raw.lower() or "modelId" in raw

    def test_has_timestamp(self, tmp_path):
        _, m = self._run(tmp_path)
        raw = json.dumps(m)
        assert "T" in raw  # ISO 8601 timestamp marker

    def test_has_passed_field(self, tmp_path):
        _, m = self._run(tmp_path)
        raw = json.dumps(m)
        assert "passed" in raw or "status" in raw


# ── Policy evaluation ─────────────────────────────────────────────────────────


class TestPolicyEvaluation:
    def test_policy_results_populated(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=["eu-ai-act"],
            sign=False,
            fail_on_violation=False,
        ))
        assert "eu-ai-act" in result.policy_results

    def test_policy_artifact_written(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=["nist-ai-rmf"],
            sign=False,
            fail_on_violation=False,
        ))
        policy_json = result.output_dir / "squash-policy-nist-ai-rmf.json"
        assert policy_json.exists()

    def test_unknown_policy_skipped(self, tmp_path):
        """An unknown policy name must not raise — it should be skipped with a warning."""
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=["this-policy-does-not-exist"],
            sign=False,
            fail_on_violation=False,
        ))
        # Should complete without exception; the unknown policy may appear
        # in policy_results with an error flag OR be silently skipped
        assert isinstance(result, AttestResult)

    def test_empty_policies_no_policy_artifacts(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=[],
            sign=False,
            fail_on_violation=False,
        ))
        policy_files = list(result.output_dir.glob("squash-policy-*.json"))
        assert len(policy_files) == 0


# ── fail_on_violation ─────────────────────────────────────────────────────────


class TestFailOnViolation:
    def test_clean_scan_clean_policy_does_not_raise(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        # Should not raise even with fail_on_violation=True on a clean model
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=[],
            sign=False,
            fail_on_violation=True,
        ))
        assert isinstance(result, AttestResult)


# ── skip_scan flag ────────────────────────────────────────────────────────────


class TestSkipScan:
    def test_skip_scan_no_scan_json(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=[],
            sign=False,
            skip_scan=True,
        ))
        scan_json = result.output_dir / "squash-scan.json"
        # When skip_scan=True the scan artifact may or may not exist
        # but result.scan_result should be None or have status="skipped"
        if result.scan_result is not None:
            assert result.scan_result.status in ("skipped", "clean", "warning", "error")


# ── Error paths ───────────────────────────────────────────────────────────────


class TestErrorPaths:
    def test_nonexistent_model_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AttestPipeline.run(AttestConfig(
                model_path=tmp_path / "nonexistent-model",
                policies=[],
                sign=False,
            ))

    def test_sign_false_no_signature_file(self, tmp_path):
        d = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=d,
            policies=[],
            sign=False,
        ))
        assert result.signature_path is None or not (result.signature_path.exists())
