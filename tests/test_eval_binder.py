"""tests/test_eval_binder.py — Integration tests for squash.sbom_builder.EvalBinder.

W45: eval_binder.py shim deleted. EvalBinder canonical location is sbom_builder.py.
Test taxonomy: Integration — real temp dirs, real JSON files written and read.
No mocks of the binder itself.  Covers:
    - Shape contract: output metrics list length == number of score keys
    - Value contract: arc_easy entry fields are exactly correct
    - Confidence interval: lowerBound computed from acc_norm_stderr,none
    - Baseline delta: deltaFromBaseline key present and correct when given
    - No-baseline: deltaFromBaseline key absent when baseline_path is None
    - Idempotency: two bind() calls produce the same number of entries
    - Missing stderr: confidenceInterval absent, no crash
    - Atomic write: sidecar is valid JSON after bind
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

squash = pytest.importorskip("squash", reason="squash-ai not installed")
from squash.sbom_builder import EvalBinder


# ── Fixtures ──────────────────────────────────────────────────────────────────

# Minimal CycloneDX 1.7 BOM with an empty performanceMetrics list.
_MINIMAL_BOM: dict = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.7",
    "serialNumber": "urn:uuid:00000000-0000-0000-0000-000000000001",
    "components": [
        {
            "type": "machine-learning-model",
            "name": "qwen2.5:1.5b",
            "hashes": [],
            "modelCard": {
                "modelParameters": {
                    "task": "text-generation",
                    "architectureFamily": "qwen2",
                    "quantizationLevel": "INT4",
                },
                "quantitativeAnalysis": {
                    "performanceMetrics": [],
                },
            },
            "properties": [],
        }
    ],
}

# Minimal lmeval JSON mirroring results/lmeval_Qwen2.5-1.5B-int4_*.json schema.
_LMEVAL: dict = {
    "model": "Qwen2.5-1.5B-int4",
    "scores": {
        "arc_easy": 70.6,
        "arc_challenge": 43.6,
        "hellaswag": 54.8,
        "winogrande": 61.4,
        "piqa": 73.2,
        "openbookqa": 39.4,
    },
    "raw_results": {
        "arc_easy": {
            "acc_norm,none": 0.706,
            "acc_norm_stderr,none": 0.02039509548493655,
        },
        "arc_challenge": {
            "acc_norm,none": 0.436,
            "acc_norm_stderr,none": 0.022198954641476896,
        },
        "hellaswag": {
            "acc_norm,none": 0.548,
            "acc_norm_stderr,none": 0.02227969410784354,
        },
        "winogrande": {
            "acc,none": 0.614,
            "acc_norm_stderr,none": 0.021793529219281196,
        },
        "piqa": {
            "acc_norm,none": 0.732,
            "acc_norm_stderr,none": 0.020664,
        },
        "openbookqa": {
            "acc_norm,none": 0.394,
            "acc_norm_stderr,none": 0.021868,
        },
    },
}

_BASELINE_LMEVAL: dict = {
    "model": "Qwen2.5-1.5B-bf16",
    "scores": {
        "arc_easy": 70.6,
        "arc_challenge": 44.0,
        "hellaswag": 55.2,
        "winogrande": 62.0,
        "piqa": 73.8,
        "openbookqa": 40.2,
    },
    "raw_results": {},
}


def _write_bom(tmp_path: Path) -> Path:
    """Write a fresh minimal sidecar and return its path."""
    p = tmp_path / "cyclonedx-mlbom.json"
    p.write_text(json.dumps(_MINIMAL_BOM, indent=2))
    return p


def _write_lmeval(tmp_path: Path, data: dict | None = None) -> Path:
    p = tmp_path / "lmeval.json"
    p.write_text(json.dumps(data or _LMEVAL))
    return p


def _write_baseline(tmp_path: Path) -> Path:
    p = tmp_path / "baseline.json"
    p.write_text(json.dumps(_BASELINE_LMEVAL))
    return p


def _metrics(bom_path: Path) -> list[dict]:
    bom = json.loads(bom_path.read_text())
    return bom["components"][0]["modelCard"]["quantitativeAnalysis"]["performanceMetrics"]


def _arc_easy_entry(bom_path: Path) -> dict:
    return next(m for m in _metrics(bom_path) if m["slice"] == "arc_easy")


# ── Shape contract ────────────────────────────────────────────────────────────


class TestShape:
    def test_metrics_length_equals_score_keys(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        assert len(_metrics(bom_path)) == len(_LMEVAL["scores"])


# ── Value / field contract ────────────────────────────────────────────────────


class TestValues:
    def test_arc_easy_value(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        entry = _arc_easy_entry(bom_path)
        assert entry["value"] == "70.6"

    def test_arc_easy_slice(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        entry = _arc_easy_entry(bom_path)
        assert entry["slice"] == "arc_easy"

    def test_arc_easy_type(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        entry = _arc_easy_entry(bom_path)
        assert entry["type"] == "accuracy"


# ── Confidence interval ───────────────────────────────────────────────────────


class TestConfidenceInterval:
    def test_lower_bound_correct(self, tmp_path: Path) -> None:
        # stderr_frac = 0.02039509548493655
        # half = round(1.96 * 0.02039509548493655 * 100, 1) = round(3.997..., 1) = 4.0
        # lowerBound = round(70.6 - 4.0, 1) = 66.6
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        entry = _arc_easy_entry(bom_path)
        assert entry["confidenceInterval"]["lowerBound"] == "66.6"

    def test_upper_bound_correct(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        entry = _arc_easy_entry(bom_path)
        assert entry["confidenceInterval"]["upperBound"] == "74.6"


# ── Baseline delta ────────────────────────────────────────────────────────────


class TestBaselineDelta:
    def test_delta_present_with_baseline(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(
            bom_path,
            _write_lmeval(tmp_path),
            baseline_path=_write_baseline(tmp_path),
        )
        entry = _arc_easy_entry(bom_path)
        assert "deltaFromBaseline" in entry

    def test_delta_correct_with_baseline(self, tmp_path: Path) -> None:
        # arc_easy: 70.6 - 70.6 = 0.0
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(
            bom_path,
            _write_lmeval(tmp_path),
            baseline_path=_write_baseline(tmp_path),
        )
        entry = _arc_easy_entry(bom_path)
        assert entry["deltaFromBaseline"] == "+0.0"

    def test_delta_negative_format(self, tmp_path: Path) -> None:
        # arc_challenge: 43.6 - 44.0 = -0.4
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(
            bom_path,
            _write_lmeval(tmp_path),
            baseline_path=_write_baseline(tmp_path),
        )
        entry = next(m for m in _metrics(bom_path) if m["slice"] == "arc_challenge")
        assert entry["deltaFromBaseline"] == "-0.4"

    def test_delta_absent_without_baseline(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        entry = _arc_easy_entry(bom_path)
        assert "deltaFromBaseline" not in entry


# ── Idempotency ───────────────────────────────────────────────────────────────


class TestIdempotency:
    def test_double_bind_same_count(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        lmeval_path = _write_lmeval(tmp_path)
        EvalBinder.bind(bom_path, lmeval_path)
        count_first = len(_metrics(bom_path))
        EvalBinder.bind(bom_path, lmeval_path)
        count_second = len(_metrics(bom_path))
        assert count_first == count_second


# ── Missing stderr ────────────────────────────────────────────────────────────


class TestMissingStderr:
    def test_no_stderr_no_confidence_interval(self, tmp_path: Path) -> None:
        # Build lmeval with no stderr for arc_easy.
        lmeval_no_stderr = {
            "scores": {"arc_easy": 70.6},
            "raw_results": {"arc_easy": {"acc_norm,none": 0.706}},
        }
        bom_path = _write_bom(tmp_path)
        lmeval_path = tmp_path / "no_stderr.json"
        lmeval_path.write_text(json.dumps(lmeval_no_stderr))
        EvalBinder.bind(bom_path, lmeval_path)  # must not raise
        entry = _arc_easy_entry(bom_path)
        assert "confidenceInterval" not in entry

    def test_no_raw_results_entry_no_crash(self, tmp_path: Path) -> None:
        # raw_results entirely missing for a task.
        lmeval_sparse = {
            "scores": {"arc_easy": 70.6, "piqa": 73.2},
            "raw_results": {},  # no entries at all
        }
        bom_path = _write_bom(tmp_path)
        lmeval_path = tmp_path / "sparse.json"
        lmeval_path.write_text(json.dumps(lmeval_sparse))
        EvalBinder.bind(bom_path, lmeval_path)  # must not raise
        assert len(_metrics(bom_path)) == 2


# ── Atomic write ──────────────────────────────────────────────────────────────


class TestAtomicWrite:
    def test_sidecar_valid_json_after_bind(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        # Must parse without error.
        result = json.loads(bom_path.read_text())
        assert "components" in result

    def test_tmp_file_cleaned_up(self, tmp_path: Path) -> None:
        bom_path = _write_bom(tmp_path)
        EvalBinder.bind(bom_path, _write_lmeval(tmp_path))
        assert not (tmp_path / "cyclonedx-mlbom.tmp").exists()
