"""tests/test_squash_backfill.py — Unit tests for dev/squash_backfill.py.

Pure unit tests: all filesystem access uses TemporaryDirectory.
No real model loading, no Metal, no mlx.

Test taxonomy: Integration (uses temp dirs; cleaned up in setUp/tearDown).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add dev/ to path so we can import squash_backfill directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "dev"))
import squash_backfill as bf


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_model_dir(root: Path, dir_name: str, with_weights: bool = True) -> Path:
    """Create a minimal model dir with config.json and optionally a weight file."""
    d = root / dir_name
    d.mkdir(parents=True)
    config = {
        "model_type": "qwen3",
        "_name_or_path": f"mlx-community/{dir_name}",
        "hidden_size": 512,
    }
    (d / "config.json").write_text(json.dumps(config))
    if with_weights:
        (d / "model.safetensors").write_bytes(b"\x00" * 16)
    return d


def _make_lmeval_result(results_dir: Path, bench_name: str, scores: dict | None = None) -> Path:
    """Write a complete lmeval result JSON (6/6 tasks, 0 errors)."""
    if scores is None:
        scores = {
            "arc_easy": 35.0,
            "arc_challenge": 28.4,
            "hellaswag": 31.6,
            "piqa": 63.8,
            "winogrande": 51.2,
            "openbookqa": 30.0,
        }
    ts = "20260331T120000"
    payload = {
        "model": bench_name,
        "scores": scores,
        "raw_results": {k: {"acc,none": v / 100} for k, v in scores.items()},
        "errors": {},
    }
    p = results_dir / f"lmeval_{bench_name}_{ts}.json"
    p.write_text(json.dumps(payload))
    return p


def _minimal_bom(model_dir: Path) -> None:
    """Write a minimal CycloneDX BOM that EvalBinder.bind() can mutate."""
    bom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.7",
        "components": [
            {
                "type": "machine-learning-model",
                "modelCard": {
                    "quantitativeAnalysis": {
                        "performanceMetrics": []
                    }
                },
            }
        ],
    }
    (model_dir / "cyclonedx-mlbom.json").write_text(json.dumps(bom))


# ---------------------------------------------------------------------------
# Test 1 — sidecar written for a model dir with weight files
# ---------------------------------------------------------------------------

def test_sidecar_written_for_model_with_weights():
    with tempfile.TemporaryDirectory() as tmp:
        models_root = Path(tmp) / "models"
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()

        model_dir = _make_model_dir(models_root, "Qwen3-0.6B-int4", with_weights=True)

        build_calls = []

        def fake_from_compress_run(meta):
            build_calls.append(meta)
            _minimal_bom(meta.output_dir)
            return meta.output_dir / "cyclonedx-mlbom.json"

        with (
            patch("squish.squash.sbom_builder.CycloneDXBuilder.from_compress_run", side_effect=fake_from_compress_run),
            patch("squish.squash.sbom_builder.EvalBinder.bind"),
        ):
            status = bf.process_one(
                "Qwen3-0.6B-int4",
                model_dir,
                results_dir,
                no_overwrite=False,
                dry_run=False,
            )

    assert len(build_calls) == 1, "CycloneDXBuilder.from_compress_run must be called once"
    assert build_calls[0].model_id == "Qwen3-0.6B-int4"
    assert status == "SKIP-NORESULT"  # sidecar written but no result JSON


# ---------------------------------------------------------------------------
# Test 2 — EvalBinder.bind called when result JSON exists
# ---------------------------------------------------------------------------

def test_bind_called_when_result_exists():
    with tempfile.TemporaryDirectory() as tmp:
        models_root = Path(tmp) / "models"
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()

        model_dir = _make_model_dir(models_root, "Qwen3-0.6B-int4", with_weights=True)
        result_json = _make_lmeval_result(results_dir, "Qwen3-0.6B-int4")

        bind_calls: list[tuple] = []

        def fake_from_compress_run(meta):
            _minimal_bom(meta.output_dir)
            return meta.output_dir / "cyclonedx-mlbom.json"

        def fake_bind(bom_path, lmeval_path, baseline_path=None):
            bind_calls.append((bom_path, lmeval_path, baseline_path))

        with (
            patch("squish.squash.sbom_builder.CycloneDXBuilder.from_compress_run", side_effect=fake_from_compress_run),
            patch("squish.squash.sbom_builder.EvalBinder.bind", side_effect=fake_bind),
        ):
            status = bf.process_one(
                "Qwen3-0.6B-int4",
                model_dir,
                results_dir,
                no_overwrite=False,
                dry_run=False,
            )

    assert status == "OK"
    assert len(bind_calls) == 1
    bom_p, lmeval_p, baseline_p = bind_calls[0]
    assert lmeval_p == result_json
    assert baseline_p is None  # INT4 has no baseline


# ---------------------------------------------------------------------------
# Test 3 — existing sidecar not rewritten when --no-overwrite
# ---------------------------------------------------------------------------

def test_no_overwrite_skips_existing_sidecar():
    with tempfile.TemporaryDirectory() as tmp:
        models_root = Path(tmp) / "models"
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()

        model_dir = _make_model_dir(models_root, "Qwen3-0.6B-int3", with_weights=True)
        _minimal_bom(model_dir)  # sidecar already exists
        _make_lmeval_result(results_dir, "Qwen3-0.6B-int3")
        # INT4 baseline
        _make_lmeval_result(results_dir, "Qwen3-0.6B-int4", scores={
            "arc_easy": 35.0, "arc_challenge": 28.4, "hellaswag": 31.6,
            "piqa": 63.8, "winogrande": 51.2, "openbookqa": 30.0,
        })

        build_calls = []
        bind_calls: list[tuple] = []

        with (
            patch("squish.squash.sbom_builder.CycloneDXBuilder.from_compress_run", side_effect=lambda m: build_calls.append(m)),
            patch("squish.squash.sbom_builder.EvalBinder.bind", side_effect=lambda *a, **kw: bind_calls.append(a)),
        ):
            status = bf.process_one(
                "Qwen3-0.6B-int3",
                model_dir,
                results_dir,
                no_overwrite=True,
                dry_run=False,
            )

    assert len(build_calls) == 0, "from_compress_run must NOT be called when --no-overwrite and sidecar exists"
    # bind may still be called — the sidecar already exists, bind populates it
    assert status in ("OK", "SKIP-NORESULT")


# ---------------------------------------------------------------------------
# Test 4 — bind skipped gracefully when no result JSON
# ---------------------------------------------------------------------------

def test_bind_skipped_when_no_result_json():
    with tempfile.TemporaryDirectory() as tmp:
        models_root = Path(tmp) / "models"
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()

        model_dir = _make_model_dir(models_root, "Qwen3-0.6B-int4", with_weights=True)
        # No result JSON written.

        bind_calls = []

        def fake_from_compress_run(meta):
            _minimal_bom(meta.output_dir)
            return meta.output_dir / "cyclonedx-mlbom.py"

        with (
            patch("squish.squash.sbom_builder.CycloneDXBuilder.from_compress_run", side_effect=fake_from_compress_run),
            patch("squish.squash.sbom_builder.EvalBinder.bind", side_effect=lambda *a, **kw: bind_calls.append(a)),
        ):
            status = bf.process_one(
                "Qwen3-0.6B-int4",
                model_dir,
                results_dir,
                no_overwrite=False,
                dry_run=False,
            )

    assert status == "SKIP-NORESULT"
    assert len(bind_calls) == 0, "EvalBinder.bind must not be called when no result JSON exists"


# ---------------------------------------------------------------------------
# Test 5 — dry-run writes nothing
# ---------------------------------------------------------------------------

def test_dry_run_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        models_root = Path(tmp) / "models"
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()

        model_dir = _make_model_dir(models_root, "Qwen3-0.6B-int4", with_weights=True)
        _make_lmeval_result(results_dir, "Qwen3-0.6B-int4")

        with (
            patch("squish.squash.sbom_builder.CycloneDXBuilder.from_compress_run") as mock_build,
            patch("squish.squash.sbom_builder.EvalBinder.bind") as mock_bind,
        ):
            status = bf.process_one(
                "Qwen3-0.6B-int4",
                model_dir,
                results_dir,
                no_overwrite=False,
                dry_run=True,
            )
            mock_build.assert_not_called()
            mock_bind.assert_not_called()

    assert status == "WOULD"
    bom_path = model_dir / "cyclonedx-mlbom.json"
    assert not bom_path.exists(), "dry-run must not write any files"


# ---------------------------------------------------------------------------
# Test 6 — missing model dir returns MISSING without error
# ---------------------------------------------------------------------------

def test_missing_model_dir_returns_missing():
    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp) / "results"
        results_dir.mkdir()

        nonexistent = Path(tmp) / "models" / "ghost-int4"

        status = bf.process_one(
            "Qwen3-0.6B-int4",
            nonexistent,
            results_dir,
            no_overwrite=False,
            dry_run=False,
        )

    assert status == "MISSING"
