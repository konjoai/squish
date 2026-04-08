"""tests/test_squash_wave44.py — Wave 44: Azure DevOps SquashAttest@1 extension.

Coverage
--------
- :func:`_emit_vso`        — output format for all ADO logging commands
- :func:`set_variable`     — ``##vso[task.setvariable…]`` format
- :func:`log_issue`        — ``##vso[task.logissue type=…]`` format
- :func:`publish_artifact` — ``##vso[artifact.upload artifactname=…]`` format
- :func:`complete_task`    — ``##vso[task.complete result=…]`` format
- :func:`is_ado_context`   — ``TF_BUILD`` env-var detection
- :class:`AzureDevOpsSquash` — ``attest()`` happy path, fail_on_violation, variable setting
- ``integrations/azure-devops/vss-extension.json`` — manifest structure validation
- ``integrations/azure-devops/SquashAttestTask/task.json`` — task definition validation
- ``integrations/azure-devops/SquashAttestTask/run_squash.ps1`` — script structure markers
- ``integrations/azure-devops/SquashAttestTask/run_squash.sh`` — script structure markers
- Import of ``azure_devops`` module succeeds without azure SDK installed
"""

from __future__ import annotations

import json
import re
import sys
from io import StringIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from squish.squash.integrations.azure_devops import (
    AzureDevOpsSquash,
    _DEFAULT_ARTIFACT_NAME,
    _emit_vso,
    complete_task,
    is_ado_context,
    log_issue,
    publish_artifact,
    set_variable,
)

# ---------------------------------------------------------------------------
# Repo-relative paths to integration files
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).parent.parent
_ADO_DIR = _REPO_ROOT / "integrations" / "azure-devops"
_TASK_DIR = _ADO_DIR / "SquashAttestTask"
_VSS_EXT = _ADO_DIR / "vss-extension.json"
_TASK_JSON = _TASK_DIR / "task.json"
_PS1 = _TASK_DIR / "run_squash.ps1"
_SH = _TASK_DIR / "run_squash.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type": "qwen3"}')
    return model_dir


def _make_attest_result(passed: bool = True) -> Any:
    scan_mock = MagicMock()
    scan_mock.status = "clean"

    policy_a = MagicMock()
    policy_a.passed = passed
    policy_a.error_count = 0
    policy_a.warning_count = 0

    result = MagicMock()
    result.passed = passed
    result.scan_result = scan_mock
    result.policy_results = {"eu-ai-act": policy_a}
    result.artifacts = {
        "cyclonedx": "/tmp/model-sbom.cdx.json",
        "spdx_json": "/tmp/model-sbom.spdx.json",
        "master_record": "/tmp/squash-master.json",
    }
    return result


# ---------------------------------------------------------------------------
# _emit_vso — output format
# ---------------------------------------------------------------------------


class TestEmitVso:

    def test_simple_command_no_props(self, capsys):
        _emit_vso("task.debug", "hello world")
        out = capsys.readouterr().out
        assert out.strip() == "##vso[task.debug]hello world"

    def test_single_property(self, capsys):
        _emit_vso("task.logissue", "bad thing", type="error")
        out = capsys.readouterr().out
        assert out.strip() == "##vso[task.logissue type=error;]bad thing"

    def test_multiple_properties(self, capsys):
        _emit_vso("task.setvariable", "true", variable="MY_VAR", isOutput="true")
        out = capsys.readouterr().out
        assert "##vso[task.setvariable" in out
        assert "variable=MY_VAR" in out
        assert "isOutput=true" in out
        assert "]true" in out

    def test_no_message(self, capsys):
        _emit_vso("task.complete", result="Succeeded")
        out = capsys.readouterr().out
        assert "##vso[task.complete result=Succeeded;]" in out

    def test_output_flushed(self, capsys):
        # Verify no buffering issues — capsys captures it immediately.
        _emit_vso("task.debug", "flush-test")
        assert "flush-test" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# set_variable
# ---------------------------------------------------------------------------


class TestSetVariable:

    def test_basic_output(self, capsys):
        set_variable("FOO", "bar")
        out = capsys.readouterr().out
        assert "##vso[task.setvariable" in out
        assert "variable=FOO" in out
        assert "isOutput=true" in out
        assert "]bar" in out

    def test_not_output(self, capsys):
        set_variable("INTERNAL", "val", is_output=False)
        out = capsys.readouterr().out
        assert "isOutput=false" in out

    def test_secret_flag(self, capsys):
        set_variable("TOKEN", "xyz", is_secret=True)
        out = capsys.readouterr().out
        assert "isSecret=true" in out

    def test_squash_passed_format(self, capsys):
        set_variable("SQUASH_PASSED", "true")
        out = capsys.readouterr().out
        assert "SQUASH_PASSED" in out
        assert "]true" in out


# ---------------------------------------------------------------------------
# log_issue
# ---------------------------------------------------------------------------


class TestLogIssue:

    def test_error(self, capsys):
        log_issue("error", "Something went wrong")
        out = capsys.readouterr().out
        assert "##vso[task.logissue type=error;]Something went wrong" in out

    def test_warning(self, capsys):
        log_issue("warning", "Mild concern")
        out = capsys.readouterr().out
        assert "##vso[task.logissue type=warning;]Mild concern" in out


# ---------------------------------------------------------------------------
# publish_artifact
# ---------------------------------------------------------------------------


class TestPublishArtifact:

    def test_default_artifact_name(self, tmp_path, capsys):
        f = tmp_path / "sbom.cdx.json"
        f.write_text("{}")
        publish_artifact(f)
        out = capsys.readouterr().out
        assert f"artifactname={_DEFAULT_ARTIFACT_NAME}" in out
        assert str(f) in out

    def test_custom_artifact_name(self, tmp_path, capsys):
        f = tmp_path / "sbom.spdx.json"
        f.write_text("{}")
        publish_artifact(f, artifact_name="custom-bom")
        out = capsys.readouterr().out
        assert "artifactname=custom-bom" in out

    def test_command_format(self, tmp_path, capsys):
        f = tmp_path / "master.json"
        f.write_text("{}")
        publish_artifact(f)
        out = capsys.readouterr().out
        assert "##vso[artifact.upload" in out


# ---------------------------------------------------------------------------
# complete_task
# ---------------------------------------------------------------------------


class TestCompleteTask:

    def test_succeeded(self, capsys):
        complete_task(True, message="All good")
        out = capsys.readouterr().out
        assert "##vso[task.complete result=Succeeded;]All good" in out

    def test_failed(self, capsys):
        complete_task(False, message="Attestation failed")
        out = capsys.readouterr().out
        assert "##vso[task.complete result=Failed;]Attestation failed" in out

    def test_result_field_present(self, capsys):
        complete_task(True)
        out = capsys.readouterr().out
        assert "result=" in out


# ---------------------------------------------------------------------------
# is_ado_context
# ---------------------------------------------------------------------------


class TestIsAdoContext:

    def test_true_when_tf_build_set(self, monkeypatch):
        monkeypatch.setenv("TF_BUILD", "True")
        assert is_ado_context() is True

    def test_false_when_tf_build_absent(self, monkeypatch):
        monkeypatch.delenv("TF_BUILD", raising=False)
        assert is_ado_context() is False

    def test_false_when_tf_build_wrong_value(self, monkeypatch):
        monkeypatch.setenv("TF_BUILD", "1")
        assert is_ado_context() is False


# ---------------------------------------------------------------------------
# AzureDevOpsSquash.attest — happy path
# ---------------------------------------------------------------------------


class TestAzureDevOpsSquashAttest:

    def test_attest_sets_output_variables(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=True)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            returned = AzureDevOpsSquash.attest(
                model_path=model_dir,
                policies=["eu-ai-act"],
                fail_on_violation=False,
            )

        out = capsys.readouterr().out
        assert "SQUASH_PASSED" in out
        assert "SQUASH_SCAN_STATUS" in out
        assert returned is result

    def test_passed_true_emits_succeeded(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=True)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            AzureDevOpsSquash.attest(model_path=model_dir, fail_on_violation=False)
        out = capsys.readouterr().out
        assert "result=Succeeded" in out

    def test_passed_false_emits_failed(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=False)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            AzureDevOpsSquash.attest(model_path=model_dir, fail_on_violation=False)
        out = capsys.readouterr().out
        assert "result=Failed" in out

    def test_fail_on_violation_true_exits(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=False)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            with pytest.raises(SystemExit) as exc_info:
                AzureDevOpsSquash.attest(
                    model_path=model_dir,
                    fail_on_violation=True,
                )
        assert exc_info.value.code == 1

    def test_fail_on_violation_false_no_exit(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=False)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            # Should not raise
            AzureDevOpsSquash.attest(model_path=model_dir, fail_on_violation=False)

    def test_scan_status_in_output(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=True)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            AzureDevOpsSquash.attest(model_path=model_dir, fail_on_violation=False)
        out = capsys.readouterr().out
        assert "clean" in out  # scan_result.status = "clean" in mock

    def test_artifact_paths_set_when_present(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=True)
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            AzureDevOpsSquash.attest(
                model_path=model_dir,
                fail_on_violation=False,
                publish_artifacts=False,  # no upload attempted (files don't exist)
            )
        out = capsys.readouterr().out
        assert "SQUASH_CYCLONEDX_PATH" in out
        assert "SQUASH_SPDX_JSON_PATH" in out
        assert "SQUASH_MASTER_RECORD_PATH" in out

    def test_none_scan_result_handled(self, tmp_path, capsys):
        model_dir = _make_model_dir(tmp_path)
        result    = _make_attest_result(passed=True)
        result.scan_result = None  # Edge case: scan skipped
        with patch("squish.squash.integrations.azure_devops.AttestPipeline") as mock_pipeline:
            mock_pipeline.run.return_value = result
            AzureDevOpsSquash.attest(model_path=model_dir, fail_on_violation=False)
        out = capsys.readouterr().out
        assert "SQUASH_SCAN_STATUS" in out
        assert "]skipped" in out


# ---------------------------------------------------------------------------
# Import isolation — no Azure SDK required
# ---------------------------------------------------------------------------


class TestAdoImportNoSdk:

    def test_module_importable_without_azure_sdk(self):
        """azure_devops.py must import cleanly even with no azure-* packages."""
        # The module is already imported; verify no azure SDK is in sys.modules.
        azure_sdk_modules = [
            k for k in sys.modules
            if k.startswith("azure.devops") or k.startswith("azure.pipelines")
        ]
        assert azure_sdk_modules == [], (
            f"azure_devops.py imported Azure SDK modules: {azure_sdk_modules}"
        )

    def test_all_public_symbols_present(self):
        import squish.squash.integrations.azure_devops as ado

        for sym in [
            "_emit_vso",
            "set_variable",
            "log_issue",
            "publish_artifact",
            "complete_task",
            "is_ado_context",
            "AzureDevOpsSquash",
        ]:
            assert hasattr(ado, sym), f"Missing expected symbol: {sym}"


# ---------------------------------------------------------------------------
# vss-extension.json — manifest structure
# ---------------------------------------------------------------------------


class TestVssExtensionJson:

    @pytest.fixture(scope="class")
    def manifest(self):
        assert _VSS_EXT.exists(), f"vss-extension.json not found at {_VSS_EXT}"
        return json.loads(_VSS_EXT.read_text())

    def test_manifest_version(self, manifest):
        assert manifest.get("manifestVersion") == 1

    def test_publisher_field(self, manifest):
        assert "publisher" in manifest
        assert manifest["publisher"] == "squishai"

    def test_extension_id(self, manifest):
        assert manifest.get("id") == "squash-attest"

    def test_version_present(self, manifest):
        assert "version" in manifest
        # Should be semver x.y.z
        version = manifest["version"]
        assert re.match(r"^\d+\.\d+\.\d+$", version), (
            f"version '{version}' is not semver x.y.z"
        )

    def test_categories_azure_pipelines(self, manifest):
        categories = manifest.get("categories", [])
        assert "Azure Pipelines" in categories

    def test_contributions_present(self, manifest):
        contributions = manifest.get("contributions", [])
        assert len(contributions) >= 1

    def test_contribution_type_is_task(self, manifest):
        contribution = manifest["contributions"][0]
        assert contribution.get("type") == "ms.vss-distributed-task.task"

    def test_contribution_properties_name(self, manifest):
        props = manifest["contributions"][0].get("properties", {})
        assert "name" in props and props["name"] == "SquashAttestTask"

    def test_files_include_task_dir(self, manifest):
        files = manifest.get("files", [])
        paths = [f.get("path", "") for f in files]
        assert "SquashAttestTask" in paths


# ---------------------------------------------------------------------------
# SquashAttestTask/task.json — task definition
# ---------------------------------------------------------------------------


class TestTaskJson:

    @pytest.fixture(scope="class")
    def task(self):
        assert _TASK_JSON.exists(), f"task.json not found at {_TASK_JSON}"
        return json.loads(_TASK_JSON.read_text())

    def test_id_is_valid_uuid(self, task):
        task_id = task.get("id", "")
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            task_id,
        ), f"task id '{task_id}' is not a valid UUID"

    def test_name_is_squash_attest(self, task):
        assert task.get("name") == "SquashAttest"

    def test_version_major_is_one(self, task):
        assert task.get("version", {}).get("Major") == 1

    def test_required_inputs_present(self, task):
        input_names = {i["name"] for i in task.get("inputs", [])}
        required = {"modelPath", "policies", "sign", "failOnViolation", "outputDir"}
        missing = required - input_names
        assert not missing, f"task.json missing inputs: {missing}"

    def test_model_path_is_required(self, task):
        inputs = {i["name"]: i for i in task.get("inputs", [])}
        model_path_input = inputs.get("modelPath", {})
        assert model_path_input.get("required") is True

    def test_output_variables_present(self, task):
        output_vars = {v["name"] for v in task.get("outputVariables", [])}
        expected = {
            "SQUASH_PASSED",
            "SQUASH_SCAN_STATUS",
            "SQUASH_CYCLONEDX_PATH",
            "SQUASH_SPDX_JSON_PATH",
            "SQUASH_MASTER_RECORD_PATH",
        }
        missing = expected - output_vars
        assert not missing, f"task.json missing outputVariables: {missing}"

    def test_execution_powershell3_specified(self, task):
        execution = task.get("execution", {})
        assert "PowerShell3" in execution

    def test_powershell3_target_is_ps1(self, task):
        target = task["execution"]["PowerShell3"].get("target", "")
        assert target.endswith(".ps1"), (
            f"PowerShell3 target should be a .ps1 file: '{target}'"
        )

    def test_platforms_declared(self, task):
        platforms = task["execution"]["PowerShell3"].get("platforms", [])
        assert "windows" in platforms
        assert "linux" in platforms
        assert "osx" in platforms


# ---------------------------------------------------------------------------
# run_squash.ps1 — structural markers
# ---------------------------------------------------------------------------


class TestRunSquashPs1:

    @pytest.fixture(scope="class")
    def content(self):
        assert _PS1.exists(), f"run_squash.ps1 not found at {_PS1}"
        return _PS1.read_text()

    def test_file_is_non_empty(self, content):
        assert len(content.strip()) > 200

    def test_reads_modelpath_env(self, content):
        # PS1 uses Get-TaskInput helper which reads INPUT_<UPPER> — check the helper call
        assert "modelPath" in content and ("Get-TaskInput" in content or "INPUT_" in content)

    def test_reads_policies_env(self, content):
        assert "policies" in content or "Policies" in content

    def test_emits_setvariable(self, content):
        assert "##vso[task.setvariable" in content

    def test_emits_task_complete(self, content):
        assert "##vso[task.complete" in content

    def test_squash_passed_variable(self, content):
        assert "SQUASH_PASSED" in content

    def test_install_squish(self, content):
        assert "squish" in content.lower() and "pip" in content.lower()

    def test_result_json_parsed(self, content):
        # The script must parse the JSON result file
        assert "ConvertFrom-Json" in content or "json" in content.lower()

    def test_fail_on_violation_handled(self, content):
        assert "FailOnViolation" in content or "FAILONVIOLATION" in content


# ---------------------------------------------------------------------------
# run_squash.sh — structural markers
# ---------------------------------------------------------------------------


class TestRunSquashSh:

    @pytest.fixture(scope="class")
    def content(self):
        assert _SH.exists(), f"run_squash.sh not found at {_SH}"
        return _SH.read_text()

    def test_file_is_non_empty(self, content):
        assert len(content.strip()) > 200

    def test_shebang_bash(self, content):
        assert content.startswith("#!/usr/bin/env bash") or content.startswith("#!/bin/bash")

    def test_reads_model_path_env(self, content):
        assert "INPUT_MODELPATH" in content

    def test_reads_policies_env(self, content):
        assert "INPUT_POLICIES" in content

    def test_emits_setvariable(self, content):
        assert "##vso[task.setvariable" in content

    def test_emits_task_complete(self, content):
        assert "##vso[task.complete" in content

    def test_squash_passed_variable(self, content):
        assert "SQUASH_PASSED" in content

    def test_install_squish(self, content):
        assert "squish" in content.lower() and "pip" in content.lower()

    def test_errexit_set(self, content):
        assert "set -e" in content

    def test_fail_on_violation_handled(self, content):
        assert "FAIL_ON_VIOLATION" in content
