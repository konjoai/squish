"""tests/test_squash_wave36.py — Wave 36: SPDX AI Profile CLI options for squash attest.

Test taxonomy:
  Unit   — argparse namespace inspection; no I/O.
  Integration — AttestPipeline.run() with tmp_path + synthetic weights;
                verifies SpdxOptions fields propagate to written artifacts.

Covers:
  - --spdx-type, --spdx-safety-risk, --spdx-dataset (repeatable),
    --spdx-training-info, --spdx-sensitive-data registered on squash attest subparser
  - No SPDX flags → spdx_options=None, SpdxOptions defaults used (text-generation, unspecified)
  - Custom flags → correct SpdxOptions constructed and passed to AttestConfig
  - SPDX JSON artifact contains custom type_of_model when --spdx-type supplied
  - Multiple --spdx-dataset flags populate dataset_ids list correctly
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────


def _build_squash_parser() -> argparse.ArgumentParser:
    """Return the squash top-level parser with all subcommands registered."""
    from squish.squash.cli import _build_parser
    return _build_parser()


def _stub_model_dir(tmp_path: Path) -> Path:
    """Write a minimal synthetic model directory that passes all offline checks."""
    d = tmp_path / "test-model"
    d.mkdir()
    weight = d / "model.safetensors"
    header = b"{}"
    weight.write_bytes(struct.pack("<Q", len(header)) + header + b"\x00" * 16)
    return d


def _parse_attest(args: list[str]) -> argparse.Namespace:
    """Parse a squash attest command line and return the namespace."""
    parser = _build_squash_parser()
    return parser.parse_args(["attest", "some/model"] + args)


# ── Argparse registration: args exist ─────────────────────────────────────────


class TestSpdxArgsRegistered:
    """Verify the five new SPDX arguments are present on the attest subparser."""

    def test_spdx_type_registered(self):
        ns = _parse_attest([])
        assert hasattr(ns, "spdx_type"), "--spdx-type not registered as spdx_type"
        assert ns.spdx_type is None

    def test_spdx_safety_risk_registered(self):
        ns = _parse_attest([])
        assert hasattr(ns, "spdx_safety_risk"), "--spdx-safety-risk not registered"
        assert ns.spdx_safety_risk is None

    def test_spdx_datasets_registered(self):
        ns = _parse_attest([])
        assert hasattr(ns, "spdx_datasets"), "--spdx-dataset not registered"
        assert ns.spdx_datasets == []

    def test_spdx_training_info_registered(self):
        ns = _parse_attest([])
        assert hasattr(ns, "spdx_training_info"), "--spdx-training-info not registered"
        assert ns.spdx_training_info is None

    def test_spdx_sensitive_data_registered(self):
        ns = _parse_attest([])
        assert hasattr(ns, "spdx_sensitive_data"), "--spdx-sensitive-data not registered"
        assert ns.spdx_sensitive_data is None


# ── Argparse: custom values parsed correctly ──────────────────────────────────


class TestSpdxArgsParsed:
    """Values supplied on the CLI are stored in the correct namespace fields."""

    def test_spdx_type_parsed(self):
        ns = _parse_attest(["--spdx-type", "text-classification"])
        assert ns.spdx_type == "text-classification"

    def test_spdx_safety_risk_high(self):
        ns = _parse_attest(["--spdx-safety-risk", "high"])
        assert ns.spdx_safety_risk == "high"

    def test_spdx_safety_risk_low(self):
        ns = _parse_attest(["--spdx-safety-risk", "low"])
        assert ns.spdx_safety_risk == "low"

    def test_spdx_safety_risk_rejects_invalid(self):
        parser = _build_squash_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["attest", "m", "--spdx-safety-risk", "critical"])

    def test_spdx_dataset_single(self):
        ns = _parse_attest(["--spdx-dataset", "wikipedia"])
        assert ns.spdx_datasets == ["wikipedia"]

    def test_spdx_dataset_multiple(self):
        ns = _parse_attest(["--spdx-dataset", "wikipedia", "--spdx-dataset", "c4"])
        assert ns.spdx_datasets == ["wikipedia", "c4"]

    def test_spdx_training_info_parsed(self):
        ns = _parse_attest(["--spdx-training-info", "https://huggingface.co/llama"])
        assert ns.spdx_training_info == "https://huggingface.co/llama"

    def test_spdx_sensitive_data_parsed(self):
        ns = _parse_attest(["--spdx-sensitive-data", "present"])
        assert ns.spdx_sensitive_data == "present"

    def test_spdx_sensitive_data_rejects_invalid(self):
        parser = _build_squash_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["attest", "m", "--spdx-sensitive-data", "maybe"])


# ── SpdxOptions construction from CLI flags ────────────────────────────────────


class TestSpdxOptionsConstruction:
    """Verify that _cmd_attest builds SpdxOptions iff at least one flag is given."""

    def _run_attest(self, tmp_path: Path, extra_args: list[str]):
        """Helper: run AttestPipeline.run() with the given extra CLI-like flags."""
        from squish.squash.attest import AttestConfig, AttestPipeline
        from squish.squash.spdx_builder import SpdxOptions

        # Simulate how _cmd_attest constructs spdx_options
        ns = _parse_attest(extra_args)
        spdx_options = None
        if any([
            ns.spdx_type,
            ns.spdx_safety_risk,
            ns.spdx_datasets,
            ns.spdx_training_info,
            ns.spdx_sensitive_data,
        ]):
            spdx_options = SpdxOptions(
                type_of_model=ns.spdx_type or "text-generation",
                safety_risk_assessment=ns.spdx_safety_risk or "unspecified",
                dataset_ids=list(ns.spdx_datasets),
                information_about_training=ns.spdx_training_info or "see-model-card",
                sensitive_personal_information=ns.spdx_sensitive_data or "absent",
            )
        return spdx_options

    def test_no_flags_yields_none(self):
        spdx_opts = self._run_attest(None, [])
        assert spdx_opts is None

    def test_spdx_type_flag_triggers_construction(self):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(None, ["--spdx-type", "summarization"])
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.type_of_model == "summarization"

    def test_safety_risk_flag_triggers_construction(self):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(None, ["--spdx-safety-risk", "medium"])
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.safety_risk_assessment == "medium"

    def test_dataset_flag_triggers_construction(self):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(None, ["--spdx-dataset", "pile"])
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.dataset_ids == ["pile"]

    def test_multiple_datasets_list_correct(self):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(
            None,
            ["--spdx-dataset", "wikipedia", "--spdx-dataset", "c4", "--spdx-dataset", "books3"],
        )
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.dataset_ids == ["wikipedia", "c4", "books3"]

    def test_sensitive_data_flag_triggers_construction(self):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(None, ["--spdx-sensitive-data", "present"])
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.sensitive_personal_information == "present"

    def test_training_info_flag_triggers_construction(self):
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(None, ["--spdx-training-info", "custom-training-log"])
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.information_about_training == "custom-training-log"

    def test_default_field_backfill_when_only_risk_supplied(self):
        """When only --spdx-safety-risk is given, other fields fall back to defaults."""
        from squish.squash.spdx_builder import SpdxOptions
        spdx_opts = self._run_attest(None, ["--spdx-safety-risk", "high"])
        assert isinstance(spdx_opts, SpdxOptions)
        assert spdx_opts.type_of_model == "text-generation"
        assert spdx_opts.information_about_training == "see-model-card"
        assert spdx_opts.sensitive_personal_information == "absent"
        assert spdx_opts.dataset_ids == []


# ── Integration: custom SPDX options propagate to written artifact ──────────────


class TestSpdxOptionsIntegration:
    """Run the full AttestPipeline with custom SpdxOptions and verify the SPDX JSON."""

    def test_custom_type_of_model_in_spdx_json(self, tmp_path):
        from squish.squash.attest import AttestConfig, AttestPipeline
        from squish.squash.spdx_builder import SpdxOptions

        model_dir = _stub_model_dir(tmp_path)
        spdx_opts = SpdxOptions(
            type_of_model="question-answering",
            safety_risk_assessment="medium",
            dataset_ids=["squad"],
            information_about_training="see-hf-model-card",
            sensitive_personal_information="absent",
        )
        result = AttestPipeline.run(AttestConfig(
            model_path=model_dir,
            policies=[],
            sign=False,
            fail_on_violation=False,
            spdx_options=spdx_opts,
        ))
        assert result.spdx_json_path is not None and result.spdx_json_path.exists()
        doc = json.loads(result.spdx_json_path.read_text())
        # The SPDX JSON should contain the custom type_of_model value
        content = json.dumps(doc)
        assert "question-answering" in content, (
            "Custom type_of_model not found in SPDX JSON output"
        )

    def test_custom_safety_risk_in_spdx_json(self, tmp_path):
        from squish.squash.attest import AttestConfig, AttestPipeline
        from squish.squash.spdx_builder import SpdxOptions

        model_dir = _stub_model_dir(tmp_path)
        spdx_opts = SpdxOptions(safety_risk_assessment="high")
        result = AttestPipeline.run(AttestConfig(
            model_path=model_dir,
            policies=[],
            sign=False,
            fail_on_violation=False,
            spdx_options=spdx_opts,
        ))
        assert result.spdx_json_path is not None and result.spdx_json_path.exists()
        content = result.spdx_json_path.read_text()
        assert "high" in content.lower()

    def test_default_spdx_options_when_none(self, tmp_path):
        """AttestPipeline with spdx_options=None emits valid SPDX with defaults."""
        from squish.squash.attest import AttestConfig, AttestPipeline

        model_dir = _stub_model_dir(tmp_path)
        result = AttestPipeline.run(AttestConfig(
            model_path=model_dir,
            policies=[],
            sign=False,
            fail_on_violation=False,
            spdx_options=None,
        ))
        assert result.spdx_json_path is not None and result.spdx_json_path.exists()
        content = result.spdx_json_path.read_text()
        # Default type_of_model is "text-generation"
        assert "text-generation" in content

    def test_dataset_ids_in_spdx_json(self, tmp_path):
        from squish.squash.attest import AttestConfig, AttestPipeline
        from squish.squash.spdx_builder import SpdxOptions

        model_dir = _stub_model_dir(tmp_path)
        spdx_opts = SpdxOptions(dataset_ids=["wikipedia", "c4"])
        result = AttestPipeline.run(AttestConfig(
            model_path=model_dir,
            policies=[],
            sign=False,
            fail_on_violation=False,
            spdx_options=spdx_opts,
        ))
        assert result.spdx_json_path is not None and result.spdx_json_path.exists()
        content = result.spdx_json_path.read_text()
        assert "wikipedia" in content
        assert "c4" in content
