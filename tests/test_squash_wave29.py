"""Wave 29 — VEX publish CLI + integration CLI shims (attest-mlflow/wandb/hf/langchain).

Test taxonomy:
- TestVexPublishCli        — subprocess / integration: parser, --help, real invocation
- TestVexPublishHandler    — unit: _cmd_vex_publish() with mocked VexFeedManifest
- TestAttestMlflowCli      — unit: parser exit-0 help, missing-path → exit 1
- TestAttestWandbCli       — unit: parser exit-0 help, missing-path → exit 1
- TestAttestHuggingFaceCli — unit: parser exit-0 help, missing-path → exit 1
- TestAttestLangchainCli   — unit: parser exit-0 help, missing-path → exit 1
- TestIntegrationShimHandlers — unit: each handler delegates to AttestPipeline.run
- TestVexPublishJson       — unit: generated JSON structure is valid OpenVEX
- TestModuleCount          — no new Python modules added by wave 29
- TestCliSubcommandList    — all 5 new subcommands present in squash --help
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─── helpers ─────────────────────────────────────────────────────────────────

CLI_MODULE = "squish.squash.cli"


def _run_cli(*args) -> subprocess.CompletedProcess:
    """Run the squash CLI as a subprocess and return the CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", CLI_MODULE, *args],
        capture_output=True,
        text=True,
    )


# Minimal AttestResult stub shared across handler unit tests
class _FakeResult:
    passed = True
    bom_path = Path("/tmp/bom.json")
    output_dir = Path("/tmp/squash")

    def to_dict(self):
        return {"passed": self.passed, "bom_path": str(self.bom_path)}


# ─── TestVexPublishCli ────────────────────────────────────────────────────────

class TestVexPublishCli:
    def test_help_exits_zero(self):
        proc = _run_cli("vex-publish", "--help")
        assert proc.returncode == 0, proc.stderr

    def test_help_shows_output_flag(self):
        proc = _run_cli("vex-publish", "--help")
        assert "--output" in proc.stdout

    def test_missing_output_exits_nonzero(self):
        proc = _run_cli("vex-publish")
        assert proc.returncode != 0

    def test_empty_entries_writes_valid_vex(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]", "--quiet")
        assert proc.returncode == 0, proc.stderr
        assert out.exists()
        doc = json.loads(out.read_text())
        assert doc["@context"] == "https://openvex.dev/ns/v0.2.0"
        assert doc["@type"] == "OpenVEXDocument"
        assert doc["statements"] == []

    def test_inline_entry_written(self, tmp_path):
        entries = json.dumps([
            {
                "vulnerability": {"name": "CVE-2024-99999"},
                "products": [{"@id": "pkg:pypi/numpy@1.24.0"}],
                "status": "not_affected",
            }
        ])
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", entries, "--quiet")
        assert proc.returncode == 0, proc.stderr
        doc = json.loads(out.read_text())
        assert len(doc["statements"]) == 1
        assert doc["statements"][0]["vulnerability"] == {"name": "CVE-2024-99999"}

    def test_entries_from_file(self, tmp_path):
        entries = [{"vulnerability": {"name": "CVE-0000-0001"}, "products": [], "status": "fixed"}]
        entries_file = tmp_path / "entries.json"
        entries_file.write_text(json.dumps(entries))
        out = tmp_path / "result.json"
        proc = _run_cli(
            "vex-publish",
            "--output", str(out),
            "--entries", str(entries_file),
            "--quiet",
        )
        assert proc.returncode == 0, proc.stderr
        doc = json.loads(out.read_text())
        assert len(doc["statements"]) == 1

    def test_author_override(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli(
            "vex-publish", "--output", str(out),
            "--entries", "[]",
            "--author", "acme-security",
            "--quiet",
        )
        assert proc.returncode == 0, proc.stderr
        doc = json.loads(out.read_text())
        assert doc["author"] == "acme-security"

    def test_doc_id_override(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli(
            "vex-publish", "--output", str(out),
            "--entries", "[]",
            "--doc-id", "https://example.com/vex/v1",
            "--quiet",
        )
        assert proc.returncode == 0, proc.stderr
        doc = json.loads(out.read_text())
        assert doc["@id"] == "https://example.com/vex/v1"

    def test_invalid_json_entries_exits_1(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "{not valid json}")
        assert proc.returncode == 1

    def test_non_list_entries_exits_1(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", '{"key": "val"}')
        assert proc.returncode == 1

    def test_output_dir_created_if_missing(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]", "--quiet")
        assert proc.returncode == 0, proc.stderr
        assert out.exists()

    def test_quiet_suppresses_output(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]", "--quiet")
        assert proc.returncode == 0
        assert proc.stdout == ""

    def test_non_quiet_prints_confirmation(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]")
        assert proc.returncode == 0
        assert "VEX feed written" in proc.stdout


# ─── TestVexPublishHandler ────────────────────────────────────────────────────

class TestVexPublishHandler:
    def _make_args(self, **kwargs):
        import argparse
        ns = argparse.Namespace(
            output=str(kwargs.get("output", "/tmp/test_feed.json")),
            entries=kwargs.get("entries", "[]"),
            author=kwargs.get("author", "squash"),
            doc_id=kwargs.get("doc_id", None),
            quiet=kwargs.get("quiet", False),
        )
        return ns

    def test_returns_zero_on_success(self, tmp_path):
        from squish.squash.cli import _cmd_vex_publish

        args = self._make_args(output=str(tmp_path / "out.json"), entries="[]", quiet=True)
        rc = _cmd_vex_publish(args, quiet=True)
        assert rc == 0

    def test_output_file_created(self, tmp_path):
        from squish.squash.cli import _cmd_vex_publish

        out = tmp_path / "feed.json"
        args = self._make_args(output=str(out), entries="[]", quiet=True)
        _cmd_vex_publish(args, quiet=True)
        assert out.exists()

    def test_returns_one_on_invalid_json(self, tmp_path):
        from squish.squash.cli import _cmd_vex_publish

        args = self._make_args(output=str(tmp_path / "f.json"), entries="not-json")
        rc = _cmd_vex_publish(args, quiet=True)
        assert rc == 1

    def test_mocked_generate_called(self, tmp_path):
        from squish.squash.cli import _cmd_vex_publish

        out = tmp_path / "out.json"
        args = self._make_args(output=str(out), entries="[]", quiet=True)

        fake_doc = {
            "@context": "https://openvex.dev/ns/v0.2.0",
            "@type": "OpenVEXDocument",
            "specVersion": "0.2.0",
            "@id": "https://test/1",
            "author": "squash",
            "timestamp": "2024-01-01T00:00:00Z",
            "statements": [],
        }
        with patch("squish.squash.vex.VexFeedManifest.generate", return_value=fake_doc):
            with patch("squish.squash.vex.VexFeedManifest.validate", return_value=[]):
                rc = _cmd_vex_publish(args, quiet=True)
        assert rc == 0

    def test_validate_error_returns_one(self, tmp_path):
        from squish.squash.cli import _cmd_vex_publish

        out = tmp_path / "out.json"
        args = self._make_args(output=str(out), entries="[]", quiet=True)

        with patch("squish.squash.vex.VexFeedManifest.generate", return_value={}):
            with patch(
                "squish.squash.vex.VexFeedManifest.validate",
                return_value=["missing @context"],
            ):
                rc = _cmd_vex_publish(args, quiet=True)
        assert rc == 1


# ─── TestVexPublishJson ───────────────────────────────────────────────────────

class TestVexPublishJson:
    """Unit: generated JSON conforms to OpenVEX 0.2.0 spec."""

    def test_generated_doc_context(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]", "--quiet")
        assert proc.returncode == 0
        doc = json.loads(out.read_text())
        assert doc["@context"] == "https://openvex.dev/ns/v0.2.0"

    def test_generated_doc_has_timestamp(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]", "--quiet")
        assert proc.returncode == 0
        doc = json.loads(out.read_text())
        assert "timestamp" in doc
        assert doc["timestamp"].endswith("Z")

    def test_generated_doc_has_id(self, tmp_path):
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", "[]", "--quiet")
        assert proc.returncode == 0
        doc = json.loads(out.read_text())
        assert "@id" in doc
        assert doc["@id"].startswith("https://")

    def test_statement_fields_preserved(self, tmp_path):
        entries = json.dumps([{
            "vulnerability": {"name": "CVE-2024-12345"},
            "products": [{"@id": "pkg:pypi/test@1.0"}],
            "status": "not_affected",
            "justification": "vulnerable_code_not_in_execute_path",
        }])
        out = tmp_path / "feed.json"
        proc = _run_cli("vex-publish", "--output", str(out), "--entries", entries, "--quiet")
        assert proc.returncode == 0
        doc = json.loads(out.read_text())
        stmt = doc["statements"][0]
        assert stmt["justification"] == "vulnerable_code_not_in_execute_path"
        assert stmt["status"] == "not_affected"


# ─── TestAttestMlflowCli ─────────────────────────────────────────────────────

class TestAttestMlflowCli:
    def test_help_exits_zero(self):
        proc = _run_cli("attest-mlflow", "--help")
        assert proc.returncode == 0

    def test_help_shows_model_path(self):
        proc = _run_cli("attest-mlflow", "--help")
        assert "model_path" in proc.stdout

    def test_nonexistent_path_exits_1(self, tmp_path):
        proc = _run_cli("attest-mlflow", str(tmp_path / "no_such_model"))
        assert proc.returncode == 1

    def test_policies_flag_accepted(self, tmp_path):
        """Parser accepts --policies without model even if handler errors on missing path."""
        proc = _run_cli("attest-mlflow", "--help")
        assert "--policies" in proc.stdout

    def test_sign_flag_accepted(self):
        proc = _run_cli("attest-mlflow", "--help")
        assert "--sign" in proc.stdout

    def test_fail_on_violation_flag(self):
        proc = _run_cli("attest-mlflow", "--help")
        assert "--fail-on-violation" in proc.stdout

    def test_handler_calls_attest_pipeline(self, tmp_path):
        from squish.squash.cli import _cmd_attest_mlflow
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=["enterprise-strict"],
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        with patch("squish.squash.attest.AttestPipeline.run", return_value=fake_result):
            rc = _cmd_attest_mlflow(args, quiet=True)
        assert rc == 0


# ─── TestAttestWandbCli ───────────────────────────────────────────────────────

class TestAttestWandbCli:
    def test_help_exits_zero(self):
        proc = _run_cli("attest-wandb", "--help")
        assert proc.returncode == 0

    def test_help_shows_model_path(self):
        proc = _run_cli("attest-wandb", "--help")
        assert "model_path" in proc.stdout

    def test_nonexistent_path_exits_1(self, tmp_path):
        proc = _run_cli("attest-wandb", str(tmp_path / "no_such_model"))
        assert proc.returncode == 1

    def test_handler_calls_attest_pipeline(self, tmp_path):
        from squish.squash.cli import _cmd_attest_wandb
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        with patch("squish.squash.attest.AttestPipeline.run", return_value=fake_result):
            rc = _cmd_attest_wandb(args, quiet=True)
        assert rc == 0

    def test_failed_result_exits_1(self, tmp_path):
        from squish.squash.cli import _cmd_attest_wandb
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        fake_result.passed = False
        with patch("squish.squash.attest.AttestPipeline.run", return_value=fake_result):
            rc = _cmd_attest_wandb(args, quiet=True)
        assert rc == 1


# ─── TestAttestHuggingFaceCli ─────────────────────────────────────────────────

class TestAttestHuggingFaceCli:
    def test_help_exits_zero(self):
        proc = _run_cli("attest-huggingface", "--help")
        assert proc.returncode == 0

    def test_help_shows_repo_id_flag(self):
        proc = _run_cli("attest-huggingface", "--help")
        assert "--repo-id" in proc.stdout

    def test_help_shows_hf_token_flag(self):
        proc = _run_cli("attest-huggingface", "--help")
        assert "--hf-token" in proc.stdout

    def test_nonexistent_path_exits_1(self, tmp_path):
        proc = _run_cli("attest-huggingface", str(tmp_path / "no_such_model"))
        assert proc.returncode == 1

    def test_offline_path_no_repo(self, tmp_path):
        """Without --repo-id, falls back to offline AttestPipeline.run."""
        from squish.squash.cli import _cmd_attest_huggingface
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            repo_id=None,
            hf_token=None,
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        with patch("squish.squash.attest.AttestPipeline.run", return_value=fake_result):
            rc = _cmd_attest_huggingface(args, quiet=True)
        assert rc == 0

    def test_hf_push_path_calls_hfsquash(self, tmp_path):
        """With --repo-id, delegates to HFSquash.attest_and_push."""
        from squish.squash.cli import _cmd_attest_huggingface
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            repo_id="myorg/test-model",
            hf_token="fake-token",
            output_dir=None,
            policies=["enterprise-strict"],
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        with patch(
            "squish.squash.integrations.huggingface.HFSquash.attest_and_push",
            return_value=fake_result,
        ):
            rc = _cmd_attest_huggingface(args, quiet=True)
        assert rc == 0

    def test_failed_hf_push_exits_1(self, tmp_path):
        from squish.squash.cli import _cmd_attest_huggingface
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            repo_id="myorg/test-model",
            hf_token=None,
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        fake_result.passed = False
        with patch(
            "squish.squash.integrations.huggingface.HFSquash.attest_and_push",
            return_value=fake_result,
        ):
            rc = _cmd_attest_huggingface(args, quiet=True)
        assert rc == 1


# ─── TestAttestLangchainCli ───────────────────────────────────────────────────

class TestAttestLangchainCli:
    def test_help_exits_zero(self):
        proc = _run_cli("attest-langchain", "--help")
        assert proc.returncode == 0

    def test_help_shows_model_path(self):
        proc = _run_cli("attest-langchain", "--help")
        assert "model_path" in proc.stdout

    def test_nonexistent_path_exits_1(self, tmp_path):
        proc = _run_cli("attest-langchain", str(tmp_path / "no_such_model"))
        assert proc.returncode == 1

    def test_handler_calls_attest_pipeline(self, tmp_path):
        from squish.squash.cli import _cmd_attest_langchain
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=["enterprise-strict"],
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        with patch("squish.squash.attest.AttestPipeline.run", return_value=fake_result):
            rc = _cmd_attest_langchain(args, quiet=True)
        assert rc == 0

    def test_failed_result_exits_1(self, tmp_path):
        from squish.squash.cli import _cmd_attest_langchain
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        fake_result.passed = False
        with patch("squish.squash.attest.AttestPipeline.run", return_value=fake_result):
            rc = _cmd_attest_langchain(args, quiet=True)
        assert rc == 1

    def test_pipeline_exception_exits_2(self, tmp_path):
        from squish.squash.cli import _cmd_attest_langchain
        import argparse

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        with patch("squish.squash.attest.AttestPipeline.run", side_effect=RuntimeError("boom")):
            rc = _cmd_attest_langchain(args, quiet=True)
        assert rc == 2


# ─── TestIntegrationShimHandlers ─────────────────────────────────────────────

class TestIntegrationShimHandlers:
    """Cross-shim: default-policy fall-back and output_dir defaulting."""

    def test_mlflow_defaults_enterprise_strict(self, tmp_path):
        from squish.squash.cli import _cmd_attest_mlflow
        import argparse
        from squish.squash.attest import AttestConfig

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=None,  # not supplied — should default
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        captured_config: list[AttestConfig] = []

        def _capture(config):
            captured_config.append(config)
            return fake_result

        with patch("squish.squash.attest.AttestPipeline.run", side_effect=_capture):
            _cmd_attest_mlflow(args, quiet=True)

        assert captured_config[0].policies == ["enterprise-strict"]

    def test_wandb_output_dir_defaults_to_squash_subdir(self, tmp_path):
        from squish.squash.cli import _cmd_attest_wandb
        import argparse
        from squish.squash.attest import AttestConfig

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=None,
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        captured_config: list[AttestConfig] = []

        def _capture(config):
            captured_config.append(config)
            return fake_result

        with patch("squish.squash.attest.AttestPipeline.run", side_effect=_capture):
            _cmd_attest_wandb(args, quiet=True)

        assert captured_config[0].output_dir == model_path.parent / "squash"

    def test_langchain_passes_policies_through(self, tmp_path):
        from squish.squash.cli import _cmd_attest_langchain
        import argparse
        from squish.squash.attest import AttestConfig

        model_path = tmp_path / "model"
        model_path.mkdir()
        args = argparse.Namespace(
            model_path=str(model_path),
            output_dir=None,
            policies=["custom-policy"],
            sign=False,
            fail_on_violation=False,
            quiet=True,
        )
        fake_result = _FakeResult()
        captured: list[AttestConfig] = []

        with patch("squish.squash.attest.AttestPipeline.run", side_effect=lambda c: (captured.append(c), fake_result)[1]):
            _cmd_attest_langchain(args, quiet=True)

        assert captured[0].policies == ["custom-policy"]


# ─── TestModuleCount ─────────────────────────────────────────────────────────

class TestModuleCount:
    def test_no_new_python_modules_added(self):
        """Wave 29 adds only CLI functions to cli.py — no new .py modules."""
        squash_root = Path(__file__).parent.parent / "squish" / "squash"
        py_files = [
            p for p in squash_root.rglob("*.py")
            if "experimental" not in str(p)
            and "__pycache__" not in str(p)
        ]
        # Hard limit: 106 (as documented in CHANGELOG for waves 1–28)
        assert len(py_files) <= 106, (
            f"Module count exceeded 106: {len(py_files)} files found. "
            "Add justification to CHANGELOG before adding new modules."
        )


# ─── TestCliSubcommandList ────────────────────────────────────────────────────

class TestCliSubcommandList:
    """All 5 new Wave 29 subcommands appear in squash --help."""

    def test_vex_publish_in_help(self):
        proc = _run_cli("--help")
        assert "vex-publish" in proc.stdout

    def test_attest_mlflow_in_help(self):
        proc = _run_cli("--help")
        assert "attest-mlflow" in proc.stdout

    def test_attest_wandb_in_help(self):
        proc = _run_cli("--help")
        assert "attest-wandb" in proc.stdout

    def test_attest_huggingface_in_help(self):
        proc = _run_cli("--help")
        assert "attest-huggingface" in proc.stdout

    def test_attest_langchain_in_help(self):
        proc = _run_cli("--help")
        assert "attest-langchain" in proc.stdout

    def test_existing_webhook_still_present(self):
        proc = _run_cli("--help")
        assert "webhook" in proc.stdout

    def test_existing_vex_still_present(self):
        proc = _run_cli("--help")
        assert "vex" in proc.stdout
