"""Tests for Wave 26 — SageMaker integration, ORAS OCI push, VEX feed MVP.

Covers:
  - SageMakerSquash.attach_attestation() and tag_model_package()
  - OrasAdapter.push(), build_manifest(), library + subprocess fallback paths
  - VexFeedManifest.generate(), validate(), and VexCache.fetch_squash_feed()
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from squish.squash.integrations.sagemaker import SageMakerSquash, _TAG_PREFIX
from squish.squash.sbom_builder import OrasAdapter
from squish.squash.vex import (
    SQUASH_VEX_FEED_FALLBACK_URL,
    SQUASH_VEX_FEED_URL,
    VexCache,
    VexFeedManifest,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text(
        json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}),
        encoding="utf-8",
    )
    (d / "model.safetensors").write_text("fake-weights", encoding="utf-8")
    return d


def _make_bom_file(tmp_path: Path, name: str = "sbom.cdx.json") -> Path:
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "metadata": {"component": {"name": "test-model"}},
        "components": [],
    }
    p = tmp_path / name
    p.write_text(json.dumps(bom), encoding="utf-8")
    return p


def _make_attest_result(passed: bool = True) -> Any:
    """Build a minimal AttestResult-like mock."""
    scan_mock = MagicMock()
    scan_mock.status = "clean"
    policy_a = MagicMock()
    policy_a.passed = True
    policy_a.error_count = 0
    result = MagicMock()
    result.passed = passed
    result.scan_result = scan_mock
    result.policy_results = {"enterprise-strict": policy_a}
    result.cyclonedx_path = Path("/tmp/fake.json")
    return result


# ── SageMakerSquash tests ─────────────────────────────────────────────────────


def test_sagemaker_import_error_has_helpful_message(tmp_path):
    """ImportError should name the missing package and suggest pip install."""
    model_dir = _make_model_dir(tmp_path)

    with patch.dict("sys.modules", {"boto3": None}):
        with pytest.raises(ImportError) as exc_info:
            SageMakerSquash.attach_attestation(model_dir)
    msg = str(exc_info.value)
    assert "boto3" in msg
    assert "pip install boto3" in msg


def test_attach_attestation_returns_attest_result(tmp_path):
    """attach_attestation should return the AttestResult from the pipeline."""
    model_dir = _make_model_dir(tmp_path)
    expected_result = _make_attest_result(passed=True)

    boto3_mock = MagicMock()
    boto3_mock.client.return_value = MagicMock()

    with patch("squish.squash.integrations.sagemaker.AttestPipeline") as pipeline_mock, \
         patch.dict("sys.modules", {"boto3": boto3_mock}):
        pipeline_mock.run.return_value = expected_result
        result = SageMakerSquash.attach_attestation(model_dir)

    assert result is expected_result
    pipeline_mock.run.assert_called_once()


def test_tags_include_squash_passed(tmp_path):
    """tag_model_package should set squash:passed tag on the ModelPackage."""
    expected_result = _make_attest_result(passed=True)
    arn = "arn:aws:sagemaker:us-east-1:123456789:model-package/test/1"

    boto3_mock = MagicMock()
    sm_client = MagicMock()
    boto3_mock.client.return_value = sm_client

    with patch.dict("sys.modules", {"boto3": boto3_mock}):
        SageMakerSquash.tag_model_package(arn, expected_result)

    sm_client.add_tags.assert_called_once()
    call_kwargs = sm_client.add_tags.call_args[1]
    assert call_kwargs["ResourceArn"] == arn
    tag_keys = {t["Key"] for t in call_kwargs["Tags"]}
    assert "squash:passed" in tag_keys
    assert "squash:scan_status" in tag_keys


def test_s3_upload_called_when_prefix_given(tmp_path):
    """When s3_upload_prefix is supplied, _upload_to_s3 is called."""
    model_dir = _make_model_dir(tmp_path)
    # Create a squash output dir (as attach_attestation would)
    squash_out = model_dir.parent / "squash"
    squash_out.mkdir()
    (squash_out / "sbom.cdx.json").write_text("{}", encoding="utf-8")

    boto3_mock = MagicMock()
    s3_client = MagicMock()
    boto3_mock.client.return_value = s3_client

    with patch("squish.squash.integrations.sagemaker.AttestPipeline") as pipeline_mock, \
         patch.dict("sys.modules", {"boto3": boto3_mock}):
        pipeline_mock.run.return_value = _make_attest_result()
        SageMakerSquash.attach_attestation(
            model_dir, s3_upload_prefix="s3://my-bucket/boms/"
        )

    # boto3.client was called for s3 upload
    assert boto3_mock.client.call_count >= 1


def test_no_s3_upload_when_prefix_none(tmp_path):
    """No S3 call when s3_upload_prefix is None."""
    model_dir = _make_model_dir(tmp_path)

    boto3_mock = MagicMock()
    s3_client = MagicMock()
    boto3_mock.client.return_value = s3_client

    with patch("squish.squash.integrations.sagemaker.AttestPipeline") as pipeline_mock, \
         patch.dict("sys.modules", {"boto3": boto3_mock}):
        pipeline_mock.run.return_value = _make_attest_result()
        SageMakerSquash.attach_attestation(model_dir, s3_upload_prefix=None)

    # s3.upload_file must NOT have been called
    s3_client.upload_file.assert_not_called()


def test_no_tag_when_no_model_package_arn(tmp_path):
    """When model_package_arn is None, add_tags must not be called."""
    model_dir = _make_model_dir(tmp_path)

    boto3_mock = MagicMock()
    sm_client = MagicMock()
    boto3_mock.client.return_value = sm_client

    with patch("squish.squash.integrations.sagemaker.AttestPipeline") as pipeline_mock, \
         patch.dict("sys.modules", {"boto3": boto3_mock}):
        pipeline_mock.run.return_value = _make_attest_result()
        SageMakerSquash.attach_attestation(model_dir, model_package_arn=None)

    sm_client.add_tags.assert_not_called()


def test_fail_on_violation_propagates(tmp_path):
    """fail_on_violation=True is passed through to AttestConfig."""
    model_dir = _make_model_dir(tmp_path)

    boto3_mock = MagicMock()
    boto3_mock.client.return_value = MagicMock()

    with patch("squish.squash.integrations.sagemaker.AttestPipeline") as pipeline_mock, \
         patch.dict("sys.modules", {"boto3": boto3_mock}):
        pipeline_mock.run.return_value = _make_attest_result()
        SageMakerSquash.attach_attestation(model_dir, fail_on_violation=True)

    config_passed = pipeline_mock.run.call_args[0][0]
    assert config_passed.fail_on_violation is True


def test_model_package_tagging_uses_correct_arn(tmp_path):
    """The ARN supplied by the caller must be forwarded to add_tags."""
    expected_result = _make_attest_result()
    arn = "arn:aws:sagemaker:eu-west-1:999999999:model-package/prod/42"

    boto3_mock = MagicMock()
    sm_client = MagicMock()
    boto3_mock.client.return_value = sm_client

    with patch.dict("sys.modules", {"boto3": boto3_mock}):
        SageMakerSquash.tag_model_package(arn, expected_result)

    sm_client.add_tags.assert_called_once()
    assert sm_client.add_tags.call_args[1]["ResourceArn"] == arn


# ── OrasAdapter tests ─────────────────────────────────────────────────────────


def test_build_oci_manifest_structure(tmp_path):
    """build_manifest returns a well-formed OCI manifest dict."""
    bom_path = _make_bom_file(tmp_path)
    manifest = OrasAdapter.build_manifest(bom_path)

    assert manifest["schemaVersion"] == 2
    assert "mediaType" in manifest
    assert manifest["artifactType"] == OrasAdapter.SBOM_MEDIA_TYPE
    assert len(manifest["layers"]) == 1
    layer = manifest["layers"][0]
    assert layer["mediaType"] == OrasAdapter.SBOM_MEDIA_TYPE
    assert layer["digest"].startswith("sha256:")
    assert layer["size"] == bom_path.stat().st_size


def test_build_manifest_digest_is_correct(tmp_path):
    """build_manifest layer digest must match sha256 of the file."""
    bom_path = _make_bom_file(tmp_path)
    expected_digest = "sha256:" + hashlib.sha256(bom_path.read_bytes()).hexdigest()
    manifest = OrasAdapter.build_manifest(bom_path)
    assert manifest["layers"][0]["digest"] == expected_digest


def test_build_manifest_spdx_media_type(tmp_path):
    """build_manifest detects SPDX files by filename."""
    spdx = tmp_path / "sbom.spdx.json"
    spdx.write_text('{"spdxVersion": "SPDX-2.3"}', encoding="utf-8")
    manifest = OrasAdapter.build_manifest(spdx)
    assert manifest["artifactType"] == OrasAdapter.SPDX_MEDIA_TYPE
    assert manifest["layers"][0]["mediaType"] == OrasAdapter.SPDX_MEDIA_TYPE


def test_push_raises_when_file_not_found(tmp_path):
    """push raises FileNotFoundError for a missing BOM path."""
    with pytest.raises(FileNotFoundError):
        OrasAdapter.push(tmp_path / "no_such_file.json", "reg.example.com/img:v1")


def test_push_falls_back_to_subprocess_when_no_library(tmp_path):
    """When the oras library is absent, push attempts subprocess CLI."""
    bom = _make_bom_file(tmp_path)

    with patch.dict("sys.modules", {"oras": None, "oras.client": None}):
        with patch("shutil.which", return_value="/usr/local/bin/oras"):
            proc_mock = MagicMock()
            proc_mock.returncode = 0
            proc_mock.stdout = "Digest: sha256:abc123\n"
            proc_mock.stderr = ""
            with patch("subprocess.run", return_value=proc_mock) as run_mock:
                digest = OrasAdapter.push(bom, "reg.example.com/img:v1")

    assert "sha256" in digest or digest.startswith("pushed:")
    run_mock.assert_called_once()


def test_push_raises_when_neither_library_nor_cli(tmp_path):
    """RuntimeError raised when neither oras library nor oras CLI is available."""
    bom = _make_bom_file(tmp_path)

    with patch.dict("sys.modules", {"oras": None, "oras.client": None}):
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="oras"):
                OrasAdapter.push(bom, "reg.example.com/img:v1")


def test_push_via_subprocess_includes_auth_username(tmp_path):
    """Subprocess push forwards username and password flags to oras CLI."""
    bom = _make_bom_file(tmp_path)

    with patch.dict("sys.modules", {"oras": None, "oras.client": None}):
        with patch("shutil.which", return_value="/usr/bin/oras"):
            proc_mock = MagicMock()
            proc_mock.returncode = 0
            proc_mock.stdout = "sha256:deadbeef\n"
            proc_mock.stderr = ""
            with patch("subprocess.run", return_value=proc_mock) as run_mock:
                OrasAdapter.push(
                    bom,
                    "reg.example.com/img:v1",
                    username="alice",
                    password="secret",
                )

    cmd = run_mock.call_args[0][0]
    assert "--username" in cmd
    assert "alice" in cmd
    assert "--password" in cmd
    assert "secret" in cmd


def test_push_subprocess_raises_on_nonzero_exit(tmp_path):
    """RuntimeError raised when oras CLI exits non-zero."""
    bom = _make_bom_file(tmp_path)

    with patch.dict("sys.modules", {"oras": None, "oras.client": None}):
        with patch("shutil.which", return_value="/usr/bin/oras"):
            proc_mock = MagicMock()
            proc_mock.returncode = 1
            proc_mock.stdout = ""
            proc_mock.stderr = "unauthorized: authentication required"
            with patch("subprocess.run", return_value=proc_mock):
                with pytest.raises(RuntimeError, match="oras CLI"):
                    OrasAdapter.push(bom, "reg.example.com/img:v1")


def test_push_uses_library_when_available(tmp_path):
    """push prefers the oras Python library over subprocess."""
    bom = _make_bom_file(tmp_path)

    # Use a unified mock so sys.modules["oras"].client == sys.modules["oras.client"]
    oras_mock = MagicMock()
    oras_mock.client.OrasClient.return_value.push.return_value = {"digest": "sha256:cafebabe"}

    with patch.dict("sys.modules", {"oras": oras_mock, "oras.client": oras_mock.client}):
        with patch("subprocess.run") as run_mock:
            digest = OrasAdapter.push(bom, "reg.example.com/img:v1")

    # subprocess.run must NOT have been called since library succeeded
    run_mock.assert_not_called()
    assert digest == "sha256:cafebabe"


# ── VexFeedManifest tests ─────────────────────────────────────────────────────


def _entry(cve: str = "CVE-2024-12345", status: str = "not_affected") -> dict:
    return {
        "vulnerability": {"name": cve},
        "products": [{"@id": f"pkg:pypi/numpy@1.24.0"}],
        "status": status,
    }


def test_generate_returns_openvex_document():
    doc = VexFeedManifest.generate([_entry()])
    assert doc["@context"] == VexFeedManifest.OPENVEX_CONTEXT
    assert doc["@type"] == VexFeedManifest.OPENVEX_TYPE
    assert doc["specVersion"] == VexFeedManifest.SPEC_VERSION


def test_generate_sets_correct_context_url():
    doc = VexFeedManifest.generate([_entry()])
    assert doc["@context"] == "https://openvex.dev/ns/v0.2.0"


def test_generate_includes_all_entries():
    entries = [_entry(f"CVE-2024-{i:05d}") for i in range(5)]
    doc = VexFeedManifest.generate(entries)
    assert len(doc["statements"]) == 5


def test_generate_uses_provided_author():
    doc = VexFeedManifest.generate([_entry()], author="konjo-ai")
    assert doc["author"] == "konjo-ai"


def test_generate_uses_provided_doc_id():
    doc = VexFeedManifest.generate([_entry()], doc_id="https://example.com/vex-1")
    assert doc["@id"] == "https://example.com/vex-1"


def test_generate_creates_unique_doc_id_when_not_provided():
    doc_a = VexFeedManifest.generate([_entry()])
    doc_b = VexFeedManifest.generate([_entry()])
    assert doc_a["@id"] != doc_b["@id"]


def test_generate_preserves_justification():
    entry = _entry()
    entry["justification"] = "vulnerable_code_not_in_execute_path"
    doc = VexFeedManifest.generate([entry])
    assert doc["statements"][0]["justification"] == "vulnerable_code_not_in_execute_path"


def test_validate_accepts_valid_doc():
    doc = VexFeedManifest.generate([_entry()])
    errors = VexFeedManifest.validate(doc)
    assert errors == []


def test_validate_rejects_missing_context():
    doc = VexFeedManifest.generate([_entry()])
    del doc["@context"]
    errors = VexFeedManifest.validate(doc)
    assert any("@context" in e for e in errors)


def test_validate_rejects_missing_statements():
    doc = VexFeedManifest.generate([_entry()])
    del doc["statements"]
    errors = VexFeedManifest.validate(doc)
    assert any("statements" in e for e in errors)


def test_validate_rejects_missing_id():
    doc = VexFeedManifest.generate([_entry()])
    del doc["@id"]
    errors = VexFeedManifest.validate(doc)
    assert any("@id" in e for e in errors)


def test_validate_rejects_unknown_status():
    doc = VexFeedManifest.generate([_entry(status="unknown_status_xyz")])
    errors = VexFeedManifest.validate(doc)
    assert any("status" in e for e in errors)


def test_validate_accepts_all_known_statuses():
    for status in ("not_affected", "affected", "fixed", "under_investigation"):
        doc = VexFeedManifest.generate([_entry(status=status)])
        errors = VexFeedManifest.validate(doc)
        assert errors == [], f"Expected no errors for status={status!r}, got: {errors}"


def test_vex_feed_url_constants():
    """SQUASH_VEX_FEED_URL and FALLBACK must be non-empty HTTPS URLs."""
    assert SQUASH_VEX_FEED_URL.startswith("https://")
    assert SQUASH_VEX_FEED_FALLBACK_URL.startswith("https://")
    assert SQUASH_VEX_FEED_URL != SQUASH_VEX_FEED_FALLBACK_URL


def test_vex_cache_fetch_squash_feed_uses_canonical_url(tmp_path):
    """VexCache.fetch_squash_feed() calls load_or_fetch with SQUASH_VEX_FEED_URL."""
    cache = VexCache(cache_dir=tmp_path)

    with patch.object(cache, "load_or_fetch") as lof:
        feed_mock = MagicMock()
        lof.return_value = feed_mock
        result = cache.fetch_squash_feed()

    lof.assert_called_once_with(SQUASH_VEX_FEED_URL, force=False)
    assert result is feed_mock


def test_vex_cache_fetch_squash_feed_force(tmp_path):
    """VexCache.fetch_squash_feed(force=True) forwards force=True."""
    cache = VexCache(cache_dir=tmp_path)

    with patch.object(cache, "load_or_fetch") as lof:
        lof.return_value = MagicMock()
        cache.fetch_squash_feed(force=True)

    lof.assert_called_once_with(SQUASH_VEX_FEED_URL, force=True)


# ── Integration / E2E style tests ─────────────────────────────────────────────


def test_oras_manifest_roundtrip(tmp_path):
    """Write a BOM, build a manifest, verify it is parseable and digest matches."""
    bom = _make_bom_file(tmp_path)
    manifest = OrasAdapter.build_manifest(bom)

    # Can be re-serialised to JSON without error
    manifest_json = json.dumps(manifest)
    parsed = json.loads(manifest_json)

    # Digest must match the file contents
    expected_digest = "sha256:" + hashlib.sha256(bom.read_bytes()).hexdigest()
    assert parsed["layers"][0]["digest"] == expected_digest


def test_vex_feed_generate_and_validate_roundtrip():
    """generate() then validate() must always produce zero errors."""
    entries = [
        _entry("CVE-2024-00001", "not_affected"),
        _entry("CVE-2024-00002", "affected"),
        _entry("CVE-2024-00003", "fixed"),
        _entry("CVE-2024-00004", "under_investigation"),
    ]
    doc = VexFeedManifest.generate(entries, author="test-suite", doc_id="https://test.example.com/vex/1")
    errors = VexFeedManifest.validate(doc)
    assert errors == []
    assert len(doc["statements"]) == 4


def test_sagemaker_full_pipeline_with_real_bom(tmp_path):
    """Real BOM file + mocked boto3 → valid tags written to SageMaker."""
    model_dir = _make_model_dir(tmp_path)
    arn = "arn:aws:sagemaker:us-east-1:000000000001:model-package/test-model/1"

    boto3_mock = MagicMock()
    sm_client = MagicMock()
    boto3_mock.client.return_value = sm_client

    mock_result = _make_attest_result(passed=True)

    with patch("squish.squash.integrations.sagemaker.AttestPipeline") as pipeline_mock, \
         patch.dict("sys.modules", {"boto3": boto3_mock}):
        pipeline_mock.run.return_value = mock_result
        result = SageMakerSquash.attach_attestation(
            model_dir,
            model_package_arn=arn,
            policies=["enterprise-strict"],
        )

    assert result.passed is True
    sm_client.add_tags.assert_called_once()
    tags = {t["Key"]: t["Value"] for t in sm_client.add_tags.call_args[1]["Tags"]}
    assert tags.get("squash:passed") == "true"
    assert "squash:scan_status" in tags
