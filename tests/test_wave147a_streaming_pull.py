"""tests/test_wave147a_streaming_pull.py

Wave 147a — true per-shard streaming pull
(squish/quant/streaming_pull.py): fetch one raw weight shard from
Hugging Face, quantize it, delete it, fetch the next.

Unlike `squish quantize-remote`'s AWQ paths (Wave 146), which still
download the entire raw model before quantizing anything, this is the
non-AWQ path that never has more than one raw weight shard resident on
local disk at once -- proven directly here by snapshotting the model
directory's `.safetensors` file count immediately before every simulated
download and asserting it's always zero (the previous shard, if any,
was already deleted).

`huggingface_hub`'s network calls (`snapshot_download`, `hf_hub_download`,
`list_repo_files`) are faked against a local "fake HF repo" directory
built with real `safetensors.numpy.save_file` shards, so the actual
quantization math (`squish.convert.quantize_tensor`,
`load_mlx_weights_shard`, `safe_key`) runs for real -- this is a
mechanism test (shard sequencing, deletion timing, manifest
correctness), not a mocked-everything wiring test.

Uses ``use_int4=False`` (the default INT8 path, `quantize_embeddings` /
`_quantize_numpy`) rather than INT4 -- INT4 asymmetric quantization
hard-requires the `squish_quant` Rust extension with no numpy fallback,
which CI's Linux test matrix doesn't build. The shard-sequencing
invariant this file pins doesn't depend on which quant format is used.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from safetensors.numpy import save_file

from squish.quant.streaming_pull import pull_and_quantize_shard_by_shard


def _build_fake_sharded_repo(repo_dir: Path) -> dict[str, np.ndarray]:
    """A minimal two-shard model laid out the way a real HF repo would be:
    config.json + model.safetensors.index.json + two .safetensors files."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "config.json").write_text('{"model_type": "llama"}')

    tensors_shard1 = {
        "model.embed_tokens.weight": np.random.randn(16, 8).astype(np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(8, 8).astype(np.float32),
    }
    tensors_shard2 = {
        "model.layers.0.mlp.down_proj.weight": np.random.randn(8, 16).astype(np.float32),
        "model.norm.weight": np.random.randn(8).astype(np.float32),
    }
    save_file(tensors_shard1, str(repo_dir / "model-00001-of-00002.safetensors"))
    save_file(tensors_shard2, str(repo_dir / "model-00002-of-00002.safetensors"))

    weight_map = {name: "model-00001-of-00002.safetensors" for name in tensors_shard1}
    weight_map.update({name: "model-00002-of-00002.safetensors" for name in tensors_shard2})
    (repo_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map})
    )
    all_tensors = dict(tensors_shard1)
    all_tensors.update(tensors_shard2)
    return all_tensors


def _build_fake_unsharded_repo(repo_dir: Path) -> dict[str, np.ndarray]:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "config.json").write_text('{"model_type": "llama"}')
    tensors = {
        "model.embed_tokens.weight": np.random.randn(16, 8).astype(np.float32),
        "model.norm.weight": np.random.randn(8).astype(np.float32),
    }
    save_file(tensors, str(repo_dir / "model.safetensors"))
    return tensors


class _FakeHfHub:
    """Fakes huggingface_hub's download functions against a local
    directory standing in for the remote repo -- copies files instead of
    making network calls, so the real quantization pipeline runs on real
    (small) tensor data."""

    def __init__(self, repo_dir: Path):
        self.repo_dir = repo_dir
        self.hf_hub_download_calls: list[str] = []

    def list_repo_files(self, repo_id, token=None):
        return sorted(p.name for p in self.repo_dir.iterdir() if p.is_file())

    def snapshot_download(self, repo_id, local_dir, token=None, ignore_patterns=None):
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        for p in self.repo_dir.iterdir():
            if p.suffix == ".safetensors":
                continue
            shutil.copy2(p, local_dir / p.name)
        return str(local_dir)

    def hf_hub_download(self, repo_id, filename, local_dir=None, token=None):
        self.hf_hub_download_calls.append(filename)
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        dest = local_dir / filename
        shutil.copy2(self.repo_dir / filename, dest)
        return str(dest)


def _safe_scan():
    import types
    meta = MagicMock(return_value=types.SimpleNamespace(status="safe", findings=[]))
    byte_scan = MagicMock(return_value=types.SimpleNamespace(status="clean", findings=[], scanned=1))
    return meta, byte_scan


class TestStreamingPullNeverHoldsMoreThanOneRawShard:
    def test_shard_count_on_disk_is_zero_before_every_fetch(self, tmp_path):
        """The core invariant: by the time the next shard is about to be
        fetched, the previous one has already been deleted."""
        repo_dir = tmp_path / "fake_repo"
        _build_fake_sharded_repo(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        output_dir = tmp_path / "local" / "Fake-Model-int4"

        fake_hub = _FakeHfHub(repo_dir)
        shard_counts_before_fetch = []

        def _tracking_hf_hub_download(repo_id, filename, local_dir=None, token=None):
            if filename.endswith(".safetensors"):
                shard_counts_before_fetch.append(
                    len(list(Path(local_dir).glob("*.safetensors")))
                )
            return fake_hub.hf_hub_download(repo_id, filename, local_dir=local_dir, token=token)

        meta_scan, byte_scan = _safe_scan()
        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
            patch("huggingface_hub.hf_hub_download", _tracking_hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            stats = pull_and_quantize_shard_by_shard(
                "fake/Fake-Model-bf16", model_dir, output_dir, use_int4=False, verbose=False,
            )

        assert shard_counts_before_fetch == [0, 0], (
            "a raw shard was still on disk when the next one was fetched — "
            "peak disk is no longer bounded to one shard at a time"
        )
        assert stats["shards_deleted"] == 2
        assert stats["source_bytes_reclaimed"] > 0
        assert list(model_dir.glob("*.safetensors")) == [], "raw shards must all be gone afterward"
        assert (model_dir / "config.json").exists(), (
            "config.json must survive -- it's the sibling base dir the "
            "compressed output's loader looks up for architecture config"
        )

    def test_output_manifest_covers_every_tensor_across_both_shards(self, tmp_path):
        repo_dir = tmp_path / "fake_repo"
        all_tensors = _build_fake_sharded_repo(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        output_dir = tmp_path / "local" / "Fake-Model-int4"
        fake_hub = _FakeHfHub(repo_dir)
        meta_scan, byte_scan = _safe_scan()

        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
            patch("huggingface_hub.hf_hub_download", fake_hub.hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            stats = pull_and_quantize_shard_by_shard(
                "fake/Fake-Model-bf16", model_dir, output_dir, use_int4=False, verbose=False,
            )

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert set(manifest.keys()) == set(all_tensors.keys())
        assert (output_dir / "tensors" / ".manifest_ready").exists()
        assert stats["n_quantized"] + stats["n_passthrough"] == len(all_tensors)

    def test_unsharded_single_file_repo_is_treated_as_one_shard(self, tmp_path):
        repo_dir = tmp_path / "fake_repo"
        all_tensors = _build_fake_unsharded_repo(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        output_dir = tmp_path / "local" / "Fake-Model-int4"
        fake_hub = _FakeHfHub(repo_dir)
        meta_scan, byte_scan = _safe_scan()

        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
            patch("huggingface_hub.hf_hub_download", fake_hub.hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            stats = pull_and_quantize_shard_by_shard(
                "fake/Fake-Model-bf16", model_dir, output_dir, use_int4=False, verbose=False,
            )

        assert stats["shards_deleted"] == 1
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert set(manifest.keys()) == set(all_tensors.keys())


class TestStreamingPullSecurityScans:
    def test_unsafe_metadata_scan_aborts_before_any_download(self, tmp_path):
        repo_dir = tmp_path / "fake_repo"
        _build_fake_sharded_repo(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        output_dir = tmp_path / "local" / "Fake-Model-int4"
        fake_hub = _FakeHfHub(repo_dir)

        import types
        meta_scan = MagicMock(return_value=types.SimpleNamespace(
            status="unsafe", findings=["fake dangerous pickle"],
        ))

        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
            patch("huggingface_hub.hf_hub_download", fake_hub.hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
        ):
            with pytest.raises(RuntimeError, match="security scan"):
                pull_and_quantize_shard_by_shard(
                    "fake/Fake-Model-bf16", model_dir, output_dir, use_int4=False, verbose=False,
                )

        assert not fake_hub.hf_hub_download_calls

    def test_unsafe_shard_byte_scan_aborts_that_shard(self, tmp_path):
        repo_dir = tmp_path / "fake_repo"
        _build_fake_sharded_repo(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        output_dir = tmp_path / "local" / "Fake-Model-int4"
        fake_hub = _FakeHfHub(repo_dir)

        import types
        meta_scan = MagicMock(return_value=types.SimpleNamespace(status="safe", findings=[]))
        byte_scan = MagicMock(return_value=types.SimpleNamespace(
            status="unsafe", findings=["fake dangerous opcode"], scanned=1,
        ))

        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
            patch("huggingface_hub.hf_hub_download", fake_hub.hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            with pytest.raises(RuntimeError, match="security scan"):
                pull_and_quantize_shard_by_shard(
                    "fake/Fake-Model-bf16", model_dir, output_dir, use_int4=False, verbose=False,
                )


class TestStreamingPullNoWeightsFound:
    def test_repo_with_no_safetensors_files_raises(self, tmp_path):
        repo_dir = tmp_path / "fake_repo"
        repo_dir.mkdir(parents=True)
        (repo_dir / "config.json").write_text("{}")
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        output_dir = tmp_path / "local" / "Fake-Model-int4"
        fake_hub = _FakeHfHub(repo_dir)
        meta_scan, byte_scan = _safe_scan()

        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
            patch("huggingface_hub.hf_hub_download", fake_hub.hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            with pytest.raises(RuntimeError, match="No .safetensors"):
                pull_and_quantize_shard_by_shard(
                    "fake/Fake-Model-bf16", model_dir, output_dir, use_int4=False, verbose=False,
                )
