"""tests/test_sbom_builder.py — Integration tests for squish.squash.sbom_builder.

Test taxonomy: Integration — real file I/O, temp dirs (cleaned up by pytest
``tmp_path`` fixture), real CycloneDXBuilder calls, no mocks of the builder
itself.  Covers:
    - Shape/dtype contract: sidecar path, top-level fields
    - Numerical correctness: SHA-256 hashes match actual file contents
    - Schema validation: required CycloneDX 1.7 fields present
    - Regression snapshot: field presence invariants
    - Failure cases: no weight files, unwritable dir, missing model family
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from squish.squash.sbom_builder import CompressRunMeta, CycloneDXBuilder


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def weight_dir(tmp_path: Path) -> Path:
    """Fake compressed model dir — three stub .npy weight files under tensors/."""
    tensors = tmp_path / "tensors"
    tensors.mkdir()
    for name in ("embed_tokens.npy", "lm_head.npy", "layers.0.attn.npy"):
        (tensors / name).write_bytes(b"stub-weight-" + name.encode())
    return tmp_path


def _meta(output_dir: Path, **overrides) -> CompressRunMeta:
    """Sensible default CompressRunMeta for testing."""
    defaults: dict = dict(
        model_id="qwen2.5:1.5b",
        hf_mlx_repo="mlx-community/Qwen2.5-1.5B-Instruct-bf16",
        model_family="qwen2",
        quant_format="INT4",
        awq_alpha=0.10,
        awq_group_size=16,
        output_dir=output_dir,
    )
    defaults.update(overrides)
    return CompressRunMeta(**defaults)


def _bom(output_dir: Path) -> dict:
    return json.loads((output_dir / "cyclonedx-mlbom.json").read_text())


# ── Shape / path contract ─────────────────────────────────────────────────────


class TestSidecarPath:
    def test_sidecar_written_inside_output_dir(self, weight_dir):
        result = CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        assert result == weight_dir / "cyclonedx-mlbom.json"
        assert result.exists()

    def test_returns_path_object(self, weight_dir):
        result = CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        assert isinstance(result, Path)


# ── Top-level CycloneDX schema fields ─────────────────────────────────────────


class TestTopLevelFields:
    def test_bom_format(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        assert _bom(weight_dir)["bomFormat"] == "CycloneDX"

    def test_spec_version(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        assert _bom(weight_dir)["specVersion"] == "1.7"

    def test_serial_number_urn_uuid(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        sn = _bom(weight_dir)["serialNumber"]
        assert sn.startswith("urn:uuid:")
        # 36-char UUID after the prefix
        assert len(sn) == len("urn:uuid:") + 36

    def test_metadata_timestamp_iso8601(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        ts = _bom(weight_dir)["metadata"]["timestamp"]
        assert ts.endswith("Z")
        assert "T" in ts

    def test_metadata_tools_include_squish(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        tools = _bom(weight_dir)["metadata"]["tools"]
        names = {t["name"] for t in tools}
        assert "squish" in names

    def test_exactly_one_component(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        assert len(_bom(weight_dir)["components"]) == 1


# ── Component contract ────────────────────────────────────────────────────────


class TestComponentFields:
    def test_component_type_machine_learning_model(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        assert _bom(weight_dir)["components"][0]["type"] == "machine-learning-model"

    def test_purl_format(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        purl = _bom(weight_dir)["components"][0]["purl"]
        assert purl == "pkg:huggingface/mlx-community/Qwen2.5-1.5B-Instruct-bf16"

    def test_external_reference_distribution_url(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        refs = _bom(weight_dir)["components"][0]["externalReferences"]
        types = {r["type"] for r in refs}
        assert "distribution" in types

    def test_pedigree_ancestor_present(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        ancestors = _bom(weight_dir)["components"][0]["pedigree"]["ancestors"]
        assert len(ancestors) == 1
        assert ancestors[0]["name"] == "mlx-community/Qwen2.5-1.5B-Instruct-bf16"


# ── Hashing correctness ───────────────────────────────────────────────────────


class TestHashing:
    def test_composite_hash_sha256_present(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        hashes = _bom(weight_dir)["components"][0]["hashes"]
        assert len(hashes) == 1
        assert hashes[0]["alg"] == "SHA-256"
        assert len(hashes[0]["content"]) == 64  # hex SHA-256

    def test_per_file_hashes_in_properties(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        props = {p["name"]: p["value"] for p in _bom(weight_dir)["components"][0]["properties"]}
        weight_hash_keys = [k for k in props if k.startswith("squish:weight_hash:")]
        assert len(weight_hash_keys) == 3  # three stub .npy files

    def test_per_file_hash_value_correct(self, weight_dir):
        """The stored per-file hash must match hashlib.sha256 of the actual file."""
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        props = {p["name"]: p["value"] for p in _bom(weight_dir)["components"][0]["properties"]}
        stub = weight_dir / "tensors" / "embed_tokens.npy"
        expected = hashlib.sha256(stub.read_bytes()).hexdigest()
        key = next(k for k in props if "embed_tokens.npy" in k)
        assert props[key] == expected

    def test_no_weight_files_empty_hashes(self, tmp_path):
        """Output dir with no weight files → empty hashes list, no properties."""
        CycloneDXBuilder.from_compress_run(_meta(tmp_path))
        bom = _bom(tmp_path)
        assert bom["components"][0]["hashes"] == []
        props = {p["name"]: p["value"] for p in bom["components"][0]["properties"]}
        assert not any(k.startswith("squish:weight_hash:") for k in props)


# ── modelCard / modelParameters ───────────────────────────────────────────────


class TestModelCard:
    def test_task_text_generation(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        params = _bom(weight_dir)["components"][0]["modelCard"]["modelParameters"]
        assert params["task"] == "text-generation"

    def test_quantization_level_int4(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        params = _bom(weight_dir)["components"][0]["modelCard"]["modelParameters"]
        assert params["quantizationLevel"] == "INT4"

    def test_quantization_level_int3(self, weight_dir):
        meta = _meta(weight_dir, quant_format="INT3", awq_alpha=None)
        CycloneDXBuilder.from_compress_run(meta)
        params = _bom(weight_dir)["components"][0]["modelCard"]["modelParameters"]
        assert params["quantizationLevel"] == "INT3"

    def test_architecture_family_qwen2(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        params = _bom(weight_dir)["components"][0]["modelCard"]["modelParameters"]
        assert params["architectureFamily"] == "qwen2"

    def test_architecture_family_none_defaults_to_unknown(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir, model_family=None))
        params = _bom(weight_dir)["components"][0]["modelCard"]["modelParameters"]
        assert params["architectureFamily"] == "unknown"

    def test_performance_metrics_empty_placeholder(self, weight_dir):
        """Phase 2 eval_binder populates this; Phase 1 must leave an empty list."""
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        metrics = (
            _bom(weight_dir)["components"][0]["modelCard"]["quantitativeAnalysis"][
                "performanceMetrics"
            ]
        )
        assert metrics == []


# ── AWQ properties ────────────────────────────────────────────────────────────


class TestAwqProperties:
    def test_awq_alpha_present_when_provided(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir, awq_alpha=0.07))
        props = {p["name"]: p["value"] for p in _bom(weight_dir)["components"][0]["properties"]}
        assert props["squish:awq_alpha"] == "0.07"

    def test_awq_group_size_present_when_provided(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir, awq_group_size=16))
        props = {p["name"]: p["value"] for p in _bom(weight_dir)["components"][0]["properties"]}
        assert props["squish:awq_group_size"] == "16"

    def test_awq_alpha_absent_when_none(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir, awq_alpha=None))
        prop_names = {p["name"] for p in _bom(weight_dir)["components"][0]["properties"]}
        assert "squish:awq_alpha" not in prop_names

    def test_awq_group_size_absent_when_none(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir, awq_group_size=None))
        prop_names = {p["name"] for p in _bom(weight_dir)["components"][0]["properties"]}
        assert "squish:awq_group_size" not in prop_names

    def test_quant_format_property_always_present(self, weight_dir):
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        props = {p["name"]: p["value"] for p in _bom(weight_dir)["components"][0]["properties"]}
        assert props["squish:quant_format"] == "INT4"


# ── Idempotency + regression ───────────────────────────────────────────────────


class TestIdempotency:
    def test_overwrite_changes_serial_number(self, weight_dir):
        """from_compress_run must be safe to call twice; new UUID each call."""
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        sn1 = _bom(weight_dir)["serialNumber"]
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        sn2 = _bom(weight_dir)["serialNumber"]
        assert sn1 != sn2

    def test_overwrite_preserves_hashes(self, weight_dir):
        """File hashes must be identical across two calls (files unchanged)."""
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        h1 = _bom(weight_dir)["components"][0]["hashes"]
        CycloneDXBuilder.from_compress_run(_meta(weight_dir))
        h2 = _bom(weight_dir)["components"][0]["hashes"]
        assert h1 == h2


# ── Failure cases ─────────────────────────────────────────────────────────────


class TestFailureCases:
    def test_unwritable_output_dir_raises(self, weight_dir, monkeypatch):
        """I/O failure must surface as an exception, not be silently swallowed."""
        original_write_text = Path.write_text

        def _raise(*args, **kwargs):
            raise PermissionError("no write access")

        monkeypatch.setattr(Path, "write_text", _raise)
        with pytest.raises(PermissionError):
            CycloneDXBuilder.from_compress_run(_meta(weight_dir))

    def test_safetensors_files_are_included(self, tmp_path):
        """Verify .safetensors extension is picked up alongside .npy."""
        (tmp_path / "model.safetensors").write_bytes(b"safetensors-stub")
        CycloneDXBuilder.from_compress_run(_meta(tmp_path))
        props = {
            p["name"]: p["value"]
            for p in _bom(tmp_path)["components"][0]["properties"]
        }
        weight_keys = [k for k in props if k.startswith("squish:weight_hash:")]
        assert any("model.safetensors" in k for k in weight_keys)

    def test_non_weight_files_excluded_from_hash(self, tmp_path):
        """config.json and tokenizer.json must NOT be hashed."""
        (tmp_path / "config.json").write_bytes(b"{}")
        (tmp_path / "tokenizer.json").write_bytes(b"{}")
        CycloneDXBuilder.from_compress_run(_meta(tmp_path))
        props = {
            p["name"]: p["value"]
            for p in _bom(tmp_path)["components"][0]["properties"]
        }
        assert not any(
            "config.json" in k or "tokenizer.json" in k
            for k in props
        )
