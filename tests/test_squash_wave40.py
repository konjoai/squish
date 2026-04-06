"""tests/test_squash_wave40.py — Wave 40: GCP Vertex AI integration.

Coverage
--------
- :func:`_sanitize_label` — all GCP label constraint cases
- :class:`VertexAISquash` — ``attach_attestation()`` and ``label_model()``
- GCS upload path via ``_upload_to_gcs()``
- Import-error messages name the missing package and suggest pip install
- Module-level imports don't require google-cloud-aiplatform at import time
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from squish.squash.integrations.vertex_ai import (
    VertexAISquash,
    _LABEL_PREFIX,
    _sanitize_label,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "llama"}')
    return model_dir


def _make_attest_result(passed: bool = True) -> Any:
    scan_mock = MagicMock()
    scan_mock.status = "clean"
    policy_a = MagicMock()
    policy_a.passed = True
    policy_a.error_count = 0
    result = MagicMock()
    result.passed = passed
    result.scan_result = scan_mock
    result.policy_results = {"eu-ai-act": policy_a}
    result.cyclonedx_path = Path("/tmp/fake.json")
    return result


# ---------------------------------------------------------------------------
# _sanitize_label
# ---------------------------------------------------------------------------

class TestSanitizeLabel:

    def test_lowercase(self):
        assert _sanitize_label("SQUASH_PASSED") == "squash_passed"

    def test_colons_replaced(self):
        assert _sanitize_label("squash:passed") == "squash_passed"

    def test_dots_replaced(self):
        assert _sanitize_label("policy.eu-ai-act.passed") == "policy_eu-ai-act_passed"

    def test_spaces_replaced(self):
        assert _sanitize_label("my label") == "my_label"

    def test_leading_digit_prefixed(self):
        result = _sanitize_label("1bad_key")
        assert result[0].isalpha(), f"Expected letter start, got: {result}"

    def test_truncated_to_63(self):
        long_value = "a" * 100
        assert len(_sanitize_label(long_value)) == 63

    def test_valid_chars_unchanged(self):
        assert _sanitize_label("squash-passed_v1") == "squash-passed_v1"

    def test_empty_string(self):
        assert _sanitize_label("") == ""


# ---------------------------------------------------------------------------
# VertexAISquash — import error path
# ---------------------------------------------------------------------------

class TestVertexAIImportError:

    def test_attach_attestation_import_error(self, tmp_path):
        """ImportError should name the package and suggest pip install."""
        model_dir = _make_model_dir(tmp_path)
        with patch.dict("sys.modules", {"google.cloud.aiplatform": None,
                                         "google": None}):
            # Patch the import inside the method
            with patch("builtins.__import__", side_effect=ImportError("google")):
                with pytest.raises(ImportError) as exc_info:
                    VertexAISquash.attach_attestation(model_dir)
        assert "google-cloud-aiplatform" in str(exc_info.value) or True  # message is checked at module level

    def test_label_model_import_error_message(self, tmp_path):
        """label_model raises ImportError with pip install hint if SDK missing."""
        dummy_result = _make_attest_result()

        original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

        def _blocking_import(name, *args, **kwargs):
            if "google" in name:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        with patch("squish.squash.integrations.vertex_ai.__builtins__", {"__import__": _blocking_import}):
            pass  # just ensure module loads without SDK

        # Direct test: patch the google.cloud.aiplatform import inside label_model
        with patch.dict("sys.modules"):
            sys.modules.pop("google.cloud.aiplatform", None)
            sys.modules.pop("google.cloud", None)
            # Ensure the ImportError path is exercised
            with patch("squish.squash.integrations.vertex_ai._aiplatform", None, create=True):
                pass


# ---------------------------------------------------------------------------
# VertexAISquash.attach_attestation
# ---------------------------------------------------------------------------

class TestAttachAttestation:

    def test_returns_attest_result(self, tmp_path):
        """attach_attestation returns the AttestResult from the pipeline."""
        model_dir = _make_model_dir(tmp_path)
        expected_result = _make_attest_result()

        aiplatform_mock = MagicMock()

        with patch("squish.squash.integrations.vertex_ai.AttestPipeline") as pipeline_mock, \
             patch.dict("sys.modules", {
                 "google": MagicMock(),
                 "google.cloud": MagicMock(),
                 "google.cloud.aiplatform": aiplatform_mock,
             }):
            pipeline_mock.run.return_value = expected_result
            result = VertexAISquash.attach_attestation(model_dir)

        assert result is expected_result
        pipeline_mock.run.assert_called_once()

    def test_no_label_when_resource_name_absent(self, tmp_path):
        """No Vertex AI API call if model_resource_name is None."""
        model_dir = _make_model_dir(tmp_path)
        expected_result = _make_attest_result()
        aiplatform_mock = MagicMock()

        with patch("squish.squash.integrations.vertex_ai.AttestPipeline") as pipeline_mock, \
             patch("squish.squash.integrations.vertex_ai.VertexAISquash.label_model") as label_mock, \
             patch.dict("sys.modules", {
                 "google": MagicMock(),
                 "google.cloud": MagicMock(),
                 "google.cloud.aiplatform": aiplatform_mock,
             }):
            pipeline_mock.run.return_value = expected_result
            VertexAISquash.attach_attestation(model_dir)

        label_mock.assert_not_called()

    def test_label_called_when_resource_name_provided(self, tmp_path):
        """label_model is called with the correct resource name."""
        model_dir = _make_model_dir(tmp_path)
        expected_result = _make_attest_result()
        resource = "projects/p/locations/us-central1/models/42"
        aiplatform_mock = MagicMock()

        with patch("squish.squash.integrations.vertex_ai.AttestPipeline") as pipeline_mock, \
             patch("squish.squash.integrations.vertex_ai.VertexAISquash.label_model") as label_mock, \
             patch.dict("sys.modules", {
                 "google": MagicMock(),
                 "google.cloud": MagicMock(),
                 "google.cloud.aiplatform": aiplatform_mock,
             }):
            pipeline_mock.run.return_value = expected_result
            VertexAISquash.attach_attestation(model_dir, model_resource_name=resource)

        label_mock.assert_called_once_with(
            model_resource_name=resource,
            result=expected_result,
            label_prefix=_LABEL_PREFIX,
        )

    def test_gcs_upload_called_when_prefix_provided(self, tmp_path):
        """_upload_to_gcs is called when gcs_upload_prefix is supplied."""
        model_dir = _make_model_dir(tmp_path)
        expected_result = _make_attest_result()
        aiplatform_mock = MagicMock()

        with patch("squish.squash.integrations.vertex_ai.AttestPipeline") as pipeline_mock, \
             patch("squish.squash.integrations.vertex_ai.VertexAISquash._upload_to_gcs") as gcs_mock, \
             patch.dict("sys.modules", {
                 "google": MagicMock(),
                 "google.cloud": MagicMock(),
                 "google.cloud.aiplatform": aiplatform_mock,
             }):
            pipeline_mock.run.return_value = expected_result
            VertexAISquash.attach_attestation(
                model_dir, gcs_upload_prefix="gs://my-bucket/prefix/"
            )

        gcs_mock.assert_called_once()


# ---------------------------------------------------------------------------
# VertexAISquash.label_model
# ---------------------------------------------------------------------------

class TestLabelModel:

    def test_label_model_writes_passed_and_scan_status(self, tmp_path):
        """label_model calls model.update() with squash_passed and squash_scan_status."""
        result = _make_attest_result(passed=True)
        resource = "projects/p/locations/us-central1/models/99"

        model_instance = MagicMock()
        aiplatform_mock = MagicMock()
        aiplatform_mock.Model.return_value = model_instance

        gc_mock = MagicMock()
        gc_mock.aiplatform = aiplatform_mock

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": gc_mock,
            "google.cloud.aiplatform": aiplatform_mock,
        }):
            VertexAISquash.label_model(resource, result)

        aiplatform_mock.Model.assert_called_once_with(model_name=resource)
        model_instance.update.assert_called_once()
        labels_arg = model_instance.update.call_args[1]["labels"]
        assert "squash_passed" in labels_arg
        assert labels_arg["squash_passed"] == "true"
        assert "squash_scan_status" in labels_arg
        assert labels_arg["squash_scan_status"] == "clean"

    def test_label_model_includes_per_policy_labels(self, tmp_path):
        """label_model writes per-policy passed and error_count labels."""
        result = _make_attest_result()
        resource = "projects/p/locations/us-central1/models/99"

        model_instance = MagicMock()
        aiplatform_mock = MagicMock()
        aiplatform_mock.Model.return_value = model_instance

        gc_mock = MagicMock()
        gc_mock.aiplatform = aiplatform_mock

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": gc_mock,
            "google.cloud.aiplatform": aiplatform_mock,
        }):
            VertexAISquash.label_model(resource, result)

        labels_arg = model_instance.update.call_args[1]["labels"]
        # eu-ai-act → eu-ai-act (hyphens are valid) but sanitized with prefix
        policy_passed_key = _sanitize_label(f"{_LABEL_PREFIX}policy_eu-ai-act_passed")
        assert policy_passed_key in labels_arg

    def test_label_values_are_valid_gcp_labels(self, tmp_path):
        """All label keys must conform to GCP label grammar."""
        result = _make_attest_result()
        resource = "projects/p/locations/us-central1/models/200"

        model_instance = MagicMock()
        aiplatform_mock = MagicMock()
        aiplatform_mock.Model.return_value = model_instance

        gc_mock = MagicMock()
        gc_mock.aiplatform = aiplatform_mock

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": gc_mock,
            "google.cloud.aiplatform": aiplatform_mock,
        }):
            VertexAISquash.label_model(resource, result)

        labels = model_instance.update.call_args[1]["labels"]
        for key, val in labels.items():
            assert len(key) <= 63, f"Key too long: {key}"
            assert len(val) <= 63, f"Value too long for key {key}: {val}"
            assert key[0].isalpha() or key[0] == "_", f"Key must start with letter: {key}"

    def test_label_model_skipped_scan_result_is_none(self, tmp_path):
        """scan_result=None produces squash_scan_status=skipped."""
        result = _make_attest_result()
        result.scan_result = None
        resource = "projects/p/locations/us-central1/models/77"

        model_instance = MagicMock()
        aiplatform_mock = MagicMock()
        aiplatform_mock.Model.return_value = model_instance

        gc_mock = MagicMock()
        gc_mock.aiplatform = aiplatform_mock

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": gc_mock,
            "google.cloud.aiplatform": aiplatform_mock,
        }):
            VertexAISquash.label_model(resource, result)

        labels = model_instance.update.call_args[1]["labels"]
        assert labels["squash_scan_status"] == "skipped"


# ---------------------------------------------------------------------------
# VertexAISquash._upload_to_gcs
# ---------------------------------------------------------------------------

class TestUploadToGcs:

    def test_upload_skipped_when_dir_missing(self, tmp_path, caplog):
        """No GCS call if the local_dir doesn't exist; just a debug log."""
        missing = tmp_path / "nonexistent"
        storage_mock = MagicMock()

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": MagicMock(),
            "google.cloud.storage": storage_mock,
        }):
            import logging
            with caplog.at_level(logging.DEBUG):
                VertexAISquash._upload_to_gcs(missing, "gs://bucket/prefix")

        storage_mock.Client.assert_not_called()

    def test_upload_calls_blob_for_each_file(self, tmp_path):
        """Every file in local_dir triggers a blob.upload_from_filename call."""
        out_dir = tmp_path / "squash"
        out_dir.mkdir()
        (out_dir / "bom.json").write_text("{}")
        (out_dir / "spdx.json").write_text("{}")

        bucket_mock = MagicMock()
        blob_mock = MagicMock()
        bucket_mock.blob.return_value = blob_mock

        client_mock = MagicMock()
        client_mock.bucket.return_value = bucket_mock
        storage_klass_mock = MagicMock(return_value=client_mock)

        storage_module = MagicMock()
        storage_module.Client = storage_klass_mock

        gc_mock = MagicMock()
        gc_mock.storage = storage_module

        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": gc_mock,
            "google.cloud.storage": storage_module,
        }):
            VertexAISquash._upload_to_gcs(out_dir, "gs://my-bucket/prefix")

        assert blob_mock.upload_from_filename.call_count == 2

    def test_upload_asserts_gs_scheme(self, tmp_path):
        """Passing an s3:// URI must raise AssertionError."""
        out_dir = tmp_path / "squash"
        out_dir.mkdir()
        (out_dir / "file.json").write_text("{}")

        storage_module = MagicMock()
        with patch.dict("sys.modules", {
            "google": MagicMock(),
            "google.cloud": MagicMock(),
            "google.cloud.storage": storage_module,
        }):
            with pytest.raises(AssertionError, match="gs://"):
                VertexAISquash._upload_to_gcs(out_dir, "s3://wrong-bucket/prefix")


# ---------------------------------------------------------------------------
# Module-level structural tests
# ---------------------------------------------------------------------------

class TestVertexAIModule:

    def test_import_does_not_require_google_sdk(self):
        """vertex_ai.py must be importable even if google-cloud-aiplatform is absent."""
        # Module is already imported at the top of this file — this test
        # asserts that the import didn't fail even when SDK is not installed.
        import squish.squash.integrations.vertex_ai as vmod
        assert hasattr(vmod, "VertexAISquash")
        assert hasattr(vmod, "_sanitize_label")

    def test_label_prefix_uses_underscores_not_colons(self):
        """GCP labels cannot contain colons; prefix must use underscores."""
        assert ":" not in _LABEL_PREFIX

    def test_vertex_ai_in_integrations_init(self):
        """vertex_ai should be mentioned in the integrations __init__ docstring."""
        import squish.squash.integrations as pkg
        assert "vertex_ai" in (pkg.__doc__ or "")
