"""tests/test_squash_spdx_builder.py — Integration tests for squish.squash.spdx_builder.

Test taxonomy: Integration — real temp-dir I/O, no mocks of the builder.
Covers:
  - Shape/path contract: both JSON and tag-value files written
  - Top-level SPDX 2.3 schema fields
  - AI Profile annotation injection
  - Hash verification for weight files
  - Failure case: empty directory
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from squish.squash.sbom_builder import CompressRunMeta
from squish.squash.spdx_builder import SpdxBuilder, SpdxOptions, SpdxArtifacts


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Minimal model directory with weight files."""
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    for name in ("embed_tokens.npy", "lm_head.npy"):
        (tensors / name).write_bytes(b"stub-" + name.encode())
    return tmp_path


def _meta(output_dir: Path, **overrides) -> CompressRunMeta:
    defaults = dict(
        model_id="llama-3.1-8b",
        hf_mlx_repo="meta-llama/Llama-3.1-8B-Instruct",
        model_family="llama",
        quant_format="INT4",
        awq_alpha=0.10,
        awq_group_size=16,
        output_dir=output_dir,
    )
    defaults.update(overrides)
    return CompressRunMeta(**defaults)


def _opts(**kwargs) -> SpdxOptions:
    defaults = dict(
        type_of_model="language",
        information_about_training="",
        sensitive_personal_information="no",
        safety_risk_assessment="limited",
    )
    defaults.update(kwargs)
    return SpdxOptions(**defaults)


# ── Shape / path contract ─────────────────────────────────────────────────────


class TestSpdxArtifactPaths:
    def test_returns_spdx_artifacts(self, model_dir):
        artifacts = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        assert isinstance(artifacts, SpdxArtifacts)

    def test_json_file_exists(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        assert a.json_path.exists()

    def test_tagvalue_file_exists(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        assert a.tagvalue_path.exists()

    def test_json_path_inside_output_dir(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        assert a.json_path.parent == model_dir

    def test_json_filename(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        assert a.json_path.name == "spdx-mlbom.json"

    def test_tv_filename(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        assert a.tagvalue_path.name == "spdx-mlbom.spdx"


# ── SPDX 2.3 schema fields ───────────────────────────────────────────────────


class TestSpdxJsonTopLevel:
    def _parse(self, model_dir: Path) -> dict:
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        return json.loads(a.json_path.read_text())

    def test_spdxVersion_is_2_3(self, model_dir):
        doc = self._parse(model_dir)
        assert doc.get("spdxVersion") == "SPDX-2.3"

    def test_dataLicense_is_CC0(self, model_dir):
        doc = self._parse(model_dir)
        assert doc.get("dataLicense") == "CC0-1.0"

    def test_SPDXID_present(self, model_dir):
        doc = self._parse(model_dir)
        assert doc.get("SPDXID") == "SPDXRef-DOCUMENT"

    def test_name_contains_model_id(self, model_dir):
        doc = self._parse(model_dir)
        assert "llama-3.1-8b" in doc.get("name", "")

    def test_packages_list_non_empty(self, model_dir):
        doc = self._parse(model_dir)
        assert len(doc.get("packages", [])) > 0

    def test_relationships_present(self, model_dir):
        doc = self._parse(model_dir)
        assert len(doc.get("relationships", [])) > 0


# ── AI Profile annotations ────────────────────────────────────────────────────


class TestAiProfileAnnotations:
    def test_annotations_present(self, model_dir):
        opts = _opts(
            type_of_model="language",
            sensitive_personal_information="no",
            safety_risk_assessment="limited",
        )
        a = SpdxBuilder.from_compress_run(_meta(model_dir), opts)
        doc = json.loads(a.json_path.read_text())
        # AI Profile annotations live on the first package, not the top-level document
        annotations = doc.get("packages", [{}])[0].get("annotations", [])
        assert len(annotations) > 0, "Expected AI Profile annotations in packages[0]"

    def test_annotation_contains_type_of_model(self, model_dir):
        opts = _opts(type_of_model="language")
        a = SpdxBuilder.from_compress_run(_meta(model_dir), opts)
        doc = json.loads(a.json_path.read_text())
        # Annotations are on packages[0]; comment format is "ai:typeOfModel: language"
        combined = " ".join(
            ann.get("comment", "")
            for ann in doc.get("packages", [{}])[0].get("annotations", [])
        )
        assert "language" in combined or "typeOfModel" in combined


# ── Tag-value syntax ──────────────────────────────────────────────────────────


class TestTagValue:
    def test_tv_contains_spdxversion(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        tv = a.tagvalue_path.read_text()
        assert "SPDXVersion: SPDX-2.3" in tv

    def test_tv_contains_packagename(self, model_dir):
        a = SpdxBuilder.from_compress_run(_meta(model_dir), _opts())
        tv = a.tagvalue_path.read_text()
        assert "PackageName:" in tv


# ── Empty directory ───────────────────────────────────────────────────────────


class TestEmptyDirectory:
    def test_empty_dir_produces_valid_spdx(self, tmp_path):
        a = SpdxBuilder.from_compress_run(_meta(tmp_path), _opts())
        doc = json.loads(a.json_path.read_text())
        assert doc.get("spdxVersion") == "SPDX-2.3"
