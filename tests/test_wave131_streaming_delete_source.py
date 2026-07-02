"""tests/test_wave131_streaming_delete_source.py

Wave 131 — delete-as-you-go raw shard cleanup.

`squish pull`/`squish compress` today require raw + compressed model on disk
simultaneously — the raw copy is never auto-deleted after quantization. This
wave adds an opt-in `--delete-source` flag to `process_weights_streaming()`
(threaded through `convert.py::main()`, `cli.py::cmd_compress`, and
`cli.py::cmd_pull` / `catalog.py::pull()`) that deletes each raw
`.safetensors` shard immediately after its tensors are quantized, written,
and committed to an incrementally-updated manifest — bounding peak disk to
~(compressed output + one shard) instead of ~(raw + compressed).

Covers:
- A raw shard is deleted only after its .npy outputs + manifest entry are
  committed and the per-shard disk check passes (mid-shard, the file is
  still present).
- delete_source=False (default) leaves every raw shard untouched — the
  existing behavior is unchanged.
- manifest.json / _compress_progress.json reflect completed shards
  incrementally, not just once at the very end of the run.
- A simulated mid-run failure does not silently wipe partial output, and
  main()'s diagnostic names the shards that were deleted and are
  unrecoverable, exiting non-zero.
- The pre-flight disk estimate uses the reduced-peak formula
  (compressed + largest shard) only when delete_source=True.
- Peak disk usage is measurably lower with delete_source=True than False,
  sampled via shutil.disk_usage during the run (not just size math).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import squish.convert as convert_mod
from squish.convert import process_weights_streaming


def _make_synthetic_model(
    model_dir: Path,
    n_shards: int = 3,
    tensors_per_shard: int = 4,
    tensor_shape: tuple = (64, 64),
    seed: int = 42,
) -> Path:
    """Write n_shards synthetic float32 .safetensors shards to model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n_shards):
        tensors = {
            f"layer.{i}.weight.{j}": rng.standard_normal(tensor_shape).astype(np.float32)
            for j in range(tensors_per_shard)
        }
        save_file(tensors, str(model_dir / f"model-{i:05d}-of-{n_shards:05d}.safetensors"))
    return model_dir


def _run(model_dir, output_dir, **kwargs):
    defaults = dict(
        passthrough_patterns=[],
        outlier_threshold=20.0,
        verbose=False,
        min_free_gb=0.0,
    )
    defaults.update(kwargs)
    return process_weights_streaming(model_dir, output_dir, **defaults)


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Shard deletion timing
# ──────────────────────────────────────────────────────────────────────────────


class TestShardDeletionTiming:
    def test_shard_deleted_only_after_commit(self, tmp_path, monkeypatch):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=3)
        shard_files = sorted(model_dir.glob("*.safetensors"))
        first_shard = shard_files[0]

        checked = {"seen": False}
        orig_quantize = convert_mod.quantize_tensor

        def spy_quantize(name, arr_f32, *args, **kwargs):
            if name.startswith("layer.0.") and not checked["seen"]:
                assert first_shard.exists(), "raw shard deleted before its tensors finished writing"
                checked["seen"] = True
            return orig_quantize(name, arr_f32, *args, **kwargs)

        monkeypatch.setattr(convert_mod, "quantize_tensor", spy_quantize)

        _run(model_dir, tmp_path / "out", delete_source=True)

        assert checked["seen"], "spy never observed shard-0 tensors"
        assert not first_shard.exists(), "raw shard should be deleted after processing"

    def test_delete_source_false_leaves_shards_untouched(self, tmp_path):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=3)
        shard_files = sorted(model_dir.glob("*.safetensors"))
        sizes_before = {p.name: p.stat().st_size for p in shard_files}

        _run(model_dir, tmp_path / "out", delete_source=False)

        shard_files_after = sorted(model_dir.glob("*.safetensors"))
        assert len(shard_files_after) == len(shard_files)
        for p in shard_files_after:
            assert p.stat().st_size == sizes_before[p.name]

    def test_delete_source_true_removes_all_shards(self, tmp_path):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=4)
        _run(model_dir, tmp_path / "out", delete_source=True)
        assert list(model_dir.glob("*.safetensors")) == []


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Incremental manifest / progress tracking
# ──────────────────────────────────────────────────────────────────────────────


class TestIncrementalProgress:
    def test_manifest_and_progress_updated_per_shard(self, tmp_path, monkeypatch):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=3, tensors_per_shard=2)
        output_dir = tmp_path / "out"

        observed = []
        orig_load = convert_mod.load_mlx_weights_shard

        def spy_load(shard_path):
            manifest_path = output_dir / "manifest.json"
            progress_path = output_dir / "_compress_progress.json"
            manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
            progress = (
                json.loads(progress_path.read_text())
                if progress_path.exists()
                else {"completed_shards": []}
            )
            observed.append((shard_path.name, len(manifest), len(progress["completed_shards"])))
            return orig_load(shard_path)

        monkeypatch.setattr(convert_mod, "load_mlx_weights_shard", spy_load)

        _run(model_dir, output_dir, delete_source=False)

        # Shard 0 starts with nothing committed yet; shard 1 starts with shard
        # 0's 2 tensors + 1 completed shard already on disk; shard 2 with 4/2.
        assert observed[0][1] == 0 and observed[0][2] == 0
        assert observed[1][1] == 2 and observed[1][2] == 1
        assert observed[2][1] == 4 and observed[2][2] == 2

        final_progress = json.loads((output_dir / "_compress_progress.json").read_text())
        assert final_progress["completed_shards"] == [
            "model-00000-of-00003.safetensors",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
        ]
        assert final_progress["total_shards"] == 3
        final_manifest = json.loads((output_dir / "manifest.json").read_text())
        assert len(final_manifest) == 6

    def test_progress_records_delete_source_and_freed_bytes(self, tmp_path):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=2)
        output_dir = tmp_path / "out"
        _run(model_dir, output_dir, delete_source=True)

        progress = json.loads((output_dir / "_compress_progress.json").read_text())
        assert progress["delete_source"] is True
        assert progress["freed_bytes"] > 0


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Mid-run failure handling
# ──────────────────────────────────────────────────────────────────────────────


class TestMidRunFailure:
    def test_partial_output_preserved_on_injected_failure(self, tmp_path, monkeypatch):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=5)
        shard_files = sorted(model_dir.glob("*.safetensors"))
        output_dir = tmp_path / "out"

        orig_load = convert_mod.load_mlx_weights_shard
        call_count = {"n": 0}

        def failing_load(shard_path):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("injected failure on shard 3")
            return orig_load(shard_path)

        monkeypatch.setattr(convert_mod, "load_mlx_weights_shard", failing_load)

        with pytest.raises(RuntimeError, match="injected failure"):
            _run(model_dir, output_dir, delete_source=True)

        # Shards 1-2 were fully committed and deleted before the injected
        # failure on shard 3's *load* — shard 3 itself was never touched.
        assert not shard_files[0].exists()
        assert not shard_files[1].exists()
        assert shard_files[2].exists()
        assert shard_files[3].exists()
        assert shard_files[4].exists()

        # Output not silently wiped — partial manifest/progress survive.
        assert (output_dir / "manifest.json").exists()
        progress = json.loads((output_dir / "_compress_progress.json").read_text())
        assert progress["completed_shards"] == [shard_files[0].name, shard_files[1].name]

    def test_main_reports_deleted_shards_and_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=4)
        shard_files = sorted(model_dir.glob("*.safetensors"))
        output_dir = tmp_path / "out"

        orig_load = convert_mod.load_mlx_weights_shard
        call_count = {"n": 0}

        def failing_load(shard_path):
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("boom")
            return orig_load(shard_path)

        monkeypatch.setattr(convert_mod, "load_mlx_weights_shard", failing_load)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "squish.convert",
                "--model-dir",
                str(model_dir),
                "--output",
                str(output_dir),
                "--format",
                "npy-dir",
                "--delete-source",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            convert_mod.main()

        assert exc_info.value.code != 0
        stderr = capsys.readouterr().err
        assert shard_files[0].name in stderr
        assert shard_files[1].name in stderr
        assert "re-download" in stderr.lower() or "pull" in stderr.lower()
        # Output must survive for diagnostics — no cleanup on delete_source failures.
        assert output_dir.exists()
        assert (output_dir / "manifest.json").exists()

    def test_main_default_still_cleans_up_partial_output_on_failure(self, tmp_path, monkeypatch):
        """Regression guard: delete_source=False keeps today's cleanup behavior."""
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=3)

        output_dir = tmp_path / "out"

        def failing_load(shard_path):
            raise RuntimeError("boom")

        monkeypatch.setattr(convert_mod, "load_mlx_weights_shard", failing_load)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "squish.convert",
                "--model-dir",
                str(model_dir),
                "--output",
                str(output_dir),
                "--format",
                "npy-dir",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            convert_mod.main()

        assert exc_info.value.code != 0
        assert not output_dir.exists()


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Pre-flight disk estimate
# ──────────────────────────────────────────────────────────────────────────────


class TestPreflightDiskEstimate:
    def test_delete_source_true_requires_headroom_for_largest_shard(self, tmp_path, monkeypatch):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=3, tensor_shape=(128, 128))
        shard_files = sorted(model_dir.glob("*.safetensors"))
        largest_shard_bytes = max(p.stat().st_size for p in shard_files)

        estimated_bytes = convert_mod._estimate_output_bytes(model_dir)
        # Just enough for the compressed output alone, nothing extra for an
        # in-flight raw shard — should pass without delete_source, fail with it.
        fixed_free = estimated_bytes + 1

        monkeypatch.setattr(convert_mod, "_get_free_bytes", lambda _p: fixed_free)

        # False path: only needs ~estimated_bytes — passes preflight (may still
        # fail later for unrelated reasons, but not on the initial disk check).
        try:
            _run(model_dir, tmp_path / "out_false", delete_source=False)
        except RuntimeError as exc:
            assert "Insufficient disk space" not in str(exc)

        assert largest_shard_bytes > 1  # sanity: the extra headroom is nonzero

        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            _run(model_dir, tmp_path / "out_true", delete_source=True)

    def test_delete_source_false_formula_unchanged(self, tmp_path, monkeypatch):
        model_dir = _make_synthetic_model(tmp_path / "raw", n_shards=2)
        estimated_bytes = convert_mod._estimate_output_bytes(model_dir)

        monkeypatch.setattr(convert_mod, "_get_free_bytes", lambda _p: estimated_bytes - 1)
        with pytest.raises(RuntimeError, match="Insufficient disk space"):
            _run(model_dir, tmp_path / "out", delete_source=False, min_free_gb=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Peak disk usage — sampled via shutil.disk_usage during the run
# ──────────────────────────────────────────────────────────────────────────────


class TestPeakDiskUsage:
    def test_peak_disk_usage_lower_with_delete_source(self, tmp_path, monkeypatch):
        root = tmp_path
        model_a = _make_synthetic_model(
            root / "model_a", n_shards=4, tensors_per_shard=6, tensor_shape=(256, 256), seed=1
        )
        model_b = _make_synthetic_model(
            root / "model_b", n_shards=4, tensors_per_shard=6, tensor_shape=(256, 256), seed=1
        )

        orig_load = convert_mod.load_mlx_weights_shard
        snapshots = {"target": None}

        def spy_load(shard_path):
            snapshots["target"].append(shutil.disk_usage(str(root)).used)
            result = orig_load(shard_path)
            snapshots["target"].append(shutil.disk_usage(str(root)).used)
            return result

        monkeypatch.setattr(convert_mod, "load_mlx_weights_shard", spy_load)

        baseline_a = shutil.disk_usage(str(root)).used
        snapshots["target"] = []
        _run(model_a, root / "out_a", delete_source=False)
        peak_delta_a = max(snapshots["target"]) - baseline_a

        baseline_b = shutil.disk_usage(str(root)).used
        snapshots["target"] = []
        _run(model_b, root / "out_b", delete_source=True)
        peak_delta_b = max(snapshots["target"]) - baseline_b

        assert list(model_b.glob("*.safetensors")) == []
        assert len(list(model_a.glob("*.safetensors"))) == 4

        assert peak_delta_b < peak_delta_a, (
            f"expected delete_source=True to peak lower: "
            f"peak_delta_a={peak_delta_a} peak_delta_b={peak_delta_b}"
        )
