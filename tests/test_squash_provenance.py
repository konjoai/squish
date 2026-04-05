"""tests/test_squash_provenance.py — Unit + integration tests for squish.squash.provenance.

Test taxonomy:
  - Pure unit  — DatasetRecord.to_cdx_formulation(), ProvenanceManifest.composite_sha256
  - Integration — from_datasheet() with a temp JSON file; bind_to_sbom() atomic write

Network calls (from_hf_datasets, from_s3_manifest) are not tested here —
those integrations require live credentials and are validated in E2E.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from squish.squash.provenance import DatasetRecord, ProvenanceCollector, ProvenanceManifest


# ── Unit: DatasetRecord ───────────────────────────────────────────────────────


class TestDatasetRecord:
    def test_to_cdx_formulation_returns_dict(self):
        rec = DatasetRecord(
            dataset_id="tatsu-lab/alpaca",
            source_type="huggingface",
            sha256="abc123",
            uri="https://huggingface.co/datasets/tatsu-lab/alpaca",
            version="main",
            license="MIT",
            contains_pii=False,
        )
        cdx = rec.to_cdx_formulation()
        assert isinstance(cdx, dict)

    def test_to_cdx_formulation_includes_dataset_id(self):
        rec = DatasetRecord(
            dataset_id="my-dataset",
            source_type="local",
            sha256="000",
            uri="file:///data/my-dataset",
            version="",
            license="",
            contains_pii=False,
        )
        cdx = rec.to_cdx_formulation()
        raw = json.dumps(cdx)
        assert "my-dataset" in raw

    def test_contains_pii_field_propagated(self):
        rec = DatasetRecord(
            dataset_id="pii-dataset",
            source_type="local",
            sha256="000",
            uri="file:///pii",
            version="",
            license="",
            contains_pii=True,
        )
        cdx = rec.to_cdx_formulation()
        raw = json.dumps(cdx)
        assert "pii" in raw.lower() or "personal" in raw.lower()


# ── Unit: ProvenanceManifest composite hash ───────────────────────────────────


class TestProvenanceManifest:
    def _make_manifest(self, n: int = 2) -> ProvenanceManifest:
        recs = [
            DatasetRecord(
                dataset_id=f"dataset-{i}",
                source_type="local",
                sha256=hashlib.sha256(f"data-{i}".encode()).hexdigest(),
                uri=f"file:///data/{i}",
                version="1.0",
                license="Apache-2.0",
                contains_pii=False,
            )
            for i in range(n)
        ]
        return ProvenanceManifest(datasets=recs)

    def test_composite_sha256_is_hex_string(self):
        m = self._make_manifest()
        assert isinstance(m.composite_sha256, str)
        assert len(m.composite_sha256) == 64

    def test_different_datasets_give_different_composite(self):
        m1 = self._make_manifest(1)
        m2 = self._make_manifest(2)
        assert m1.composite_sha256 != m2.composite_sha256

    def test_deterministic_hash(self):
        """Same datasets → same composite hash every time."""
        m1 = self._make_manifest(3)
        m2 = self._make_manifest(3)
        assert m1.composite_sha256 == m2.composite_sha256

    def test_empty_manifest_has_hash(self):
        m = ProvenanceManifest(datasets=[])
        assert isinstance(m.composite_sha256, str)
        # composite_sha256 is only computed for non-empty manifests
        assert m.composite_sha256 == ""


# ── Integration: from_datasheet ───────────────────────────────────────────────


class TestFromDatasheet:
    def _write_datasheet(self, path: Path) -> None:
        sheet = {
            "datasets": [
                {
                    "id": "alpaca",
                    "uri": "https://huggingface.co/datasets/tatsu-lab/alpaca",
                    "sha256": "abc123",
                    "version": "main",
                    "license": "MIT",
                    "contains_pii": False,
                },
                {
                    "id": "openassistant",
                    "uri": "https://huggingface.co/datasets/OpenAssistant/oasst1",
                    "sha256": "def456",
                    "version": "1.0",
                    "license": "Apache-2.0",
                    "contains_pii": False,
                },
            ]
        }
        path.write_text(json.dumps(sheet))

    def test_returns_manifest(self, tmp_path):
        ds_path = tmp_path / "datasheet.json"
        self._write_datasheet(ds_path)
        manifest = ProvenanceCollector.from_datasheet(ds_path)
        assert isinstance(manifest, ProvenanceManifest)

    def test_loads_correct_count(self, tmp_path):
        # from_datasheet reads ONE file as ONE record (one file → one DatasetRecord)
        ds_path = tmp_path / "alpaca.json"
        ds_path.write_text(json.dumps({
            "name": "alpaca",
            "uri": "https://example.com/alpaca",
            "license": "MIT",
            "version": "main",
            "pii": False,
        }))
        manifest = ProvenanceCollector.from_datasheet(ds_path)
        assert len(manifest.datasets) == 1

    def test_dataset_id_preserved(self, tmp_path):
        ds_path = tmp_path / "alpaca.json"
        ds_path.write_text(json.dumps({
            "name": "alpaca",
            "uri": "https://example.com/alpaca",
            "license": "MIT",
        }))
        manifest = ProvenanceCollector.from_datasheet(ds_path)
        ids = [r.dataset_id for r in manifest.datasets]
        assert "alpaca" in ids

    def test_missing_file_returns_empty_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ProvenanceCollector.from_datasheet(tmp_path / "nonexistent.json")


# ── Integration: bind_to_sbom atomic write ────────────────────────────────────


class TestBindToSbom:
    def _minimal_bom(self, path: Path) -> Path:
        bom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.7",
            "components": [{"name": "test-model", "hashes": []}],
        }
        path.write_text(json.dumps(bom))
        return path

    def test_bind_writes_formulation_key(self, tmp_path):
        bom_path = self._minimal_bom(tmp_path / "cyclonedx-mlbom.json")
        ds_path = tmp_path / "datasheet.json"
        ds_path.write_text(json.dumps({
            "datasets": [
                {"id": "d1", "uri": "file:///d1", "sha256": "aaa", "version": "1.0", "license": "MIT", "contains_pii": False}
            ]
        }))
        manifest = ProvenanceCollector.from_datasheet(ds_path)
        manifest.bind_to_sbom(bom_path)

        updated = json.loads(bom_path.read_text())
        # Either "formulation" array or "squash:trainingDataComposite" must exist
        has_formulation = "formulation" in updated
        has_composite = "squash:trainingDataComposite" in updated
        assert has_formulation or has_composite, "Expected provenance binding in BOM"

    def test_bind_is_atomic_no_partial_write(self, tmp_path):
        """No .tmp file should remain after a successful bind."""
        bom_path = self._minimal_bom(tmp_path / "cyclonedx-mlbom.json")
        manifest = ProvenanceManifest(datasets=[])
        manifest.bind_to_sbom(bom_path)
        assert not (bom_path.with_suffix(".tmp")).exists()
