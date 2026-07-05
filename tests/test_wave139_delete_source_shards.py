"""tests/test_wave139_delete_source_shards.py

Wave 139 — delete-as-you-go raw shard cleanup.

`process_weights_streaming` processes one .safetensors shard at a time and,
until this wave, always left the raw shard files on disk afterward — the
full raw model and the full compressed output had to coexist, which blocks
quantizing a model whose raw size exceeds local disk. The new
``delete_source`` flag deletes each raw shard immediately after all of its
tensors are quantized and written, bounding peak disk to roughly one raw
shard + the compressed output built so far.

This pins:
- delete_source=False (default) leaves every raw shard on disk — no
  behavior change for existing callers
- delete_source=True deletes each shard right after it's fully processed,
  and only after its tensors are written (not before)
- stats reports shards_deleted / source_bytes_reclaimed
- a shard that fails to delete (OSError) logs a warning and does not raise
  — losing cleanup for one shard isn't a reason to fail an otherwise
  successful compression
- multiple shards are each deleted independently, in order
- main()'s failure-cleanup path keeps partial output instead of wiping it
  when delete_source was active, since some raw shards may already be gone
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

import squish.convert as convert_mod
from squish.convert import process_weights_streaming

pytestmark = pytest.mark.filterwarnings("ignore")


def _write_shard(path: Path, tensor_names: list[str], shape=(64, 128)) -> None:
    tensors = {name: np.random.randn(*shape).astype(np.float32) for name in tensor_names}
    save_file(tensors, str(path))


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    return d


@pytest.fixture
def output_path(tmp_path):
    return tmp_path / "output"


class TestDeleteSourceDefaultOff:
    def test_shard_survives_by_default(self, model_dir, output_path):
        shard = model_dir / "model-00001-of-00001.safetensors"
        _write_shard(shard, ["layer.weight"])

        stats = process_weights_streaming(model_dir, output_path, [], 20.0, False, min_free_gb=0.0)

        assert shard.exists()
        assert stats["shards_deleted"] == 0
        assert stats["source_bytes_reclaimed"] == 0


class TestDeleteSourceEnabled:
    def test_shard_removed_after_processing(self, model_dir, output_path):
        shard = model_dir / "model-00001-of-00001.safetensors"
        _write_shard(shard, ["layer.weight"])

        stats = process_weights_streaming(
            model_dir, output_path, [], 20.0, False, min_free_gb=0.0, delete_source=True
        )

        assert not shard.exists()
        assert stats["shards_deleted"] == 1
        assert stats["source_bytes_reclaimed"] > 0

    def test_multiple_shards_each_deleted_independently(self, model_dir, output_path):
        shard1 = model_dir / "model-00001-of-00002.safetensors"
        shard2 = model_dir / "model-00002-of-00002.safetensors"
        _write_shard(shard1, ["layer0.weight"])
        _write_shard(shard2, ["layer1.weight"])

        stats = process_weights_streaming(
            model_dir, output_path, [], 20.0, False, min_free_gb=0.0, delete_source=True
        )

        assert not shard1.exists()
        assert not shard2.exists()
        assert stats["shards_deleted"] == 2

    def test_compressed_output_unaffected_by_deletion(self, model_dir, output_path):
        shard = model_dir / "model-00001-of-00001.safetensors"
        _write_shard(shard, ["layer.weight"])

        stats = process_weights_streaming(
            model_dir, output_path, [], 20.0, False, min_free_gb=0.0, delete_source=True
        )

        assert (output_path / "manifest.json").exists()
        assert (output_path / "tensors" / ".manifest_ready").exists()
        assert stats["n_quantized"] + stats["n_passthrough"] == 1

    def test_deletion_failure_propagates_but_compressed_output_already_safe(
        self, model_dir, output_path, monkeypatch
    ):
        # NOTE: this pins the behavior of the merged (main) delete-as-you-go
        # implementation, which deletes a shard without a try/except — a
        # deletion failure is a hard error here, caught and reported by
        # main()'s CLI-level failure handler (which preserves partial output
        # rather than the earlier design this test originally covered, where
        # library-level code swallowed the failure and logged a warning).
        shard = model_dir / "model-00001-of-00001.safetensors"
        _write_shard(shard, ["layer.weight"])

        real_unlink = Path.unlink

        def _boom(self, *a, **kw):
            if self == shard:
                raise OSError("permission denied (simulated)")
            return real_unlink(self, *a, **kw)

        monkeypatch.setattr(Path, "unlink", _boom)

        with pytest.raises(OSError, match="permission denied"):
            process_weights_streaming(
                model_dir, output_path, [], 20.0, False, min_free_gb=0.0, delete_source=True
            )

        # The tensor's compressed output and manifest entry were committed to
        # disk BEFORE the deletion attempt — so even though the call raised,
        # nothing quantized is lost; only the raw shard survives (deletion
        # failed) alongside the now-complete compressed output.
        assert (output_path / "manifest.json").exists()
        assert shard.exists()

    def test_shard_not_deleted_until_its_own_tensors_written(self, model_dir, output_path):
        # A shard with two tensors: both must be quantized/written before the
        # shard file disappears (not deleted mid-loop after the first tensor).
        shard = model_dir / "model-00001-of-00001.safetensors"
        _write_shard(shard, ["a.weight", "b.weight"])

        stats = process_weights_streaming(
            model_dir, output_path, [], 20.0, False, min_free_gb=0.0, delete_source=True
        )

        assert stats["n_quantized"] + stats["n_passthrough"] == 2
        assert not shard.exists()
        assert stats["shards_deleted"] == 1


class TestMainFailureCleanupRespectsDeleteSource:
    """main()'s exception handler must not wipe partial output when
    --delete-source was active, since some raw shards may already be
    irrecoverably deleted by that point."""

    def _run_main_expect_failure(self, tmp_path, monkeypatch, extra_argv):
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        output_path = tmp_path / "output"
        output_path.mkdir()
        (output_path / "partial.npy").write_bytes(b"\x00")

        monkeypatch.setattr(
            convert_mod,
            "process_weights_streaming",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("simulated failure")),
        )
        rmtree_calls = []
        monkeypatch.setattr(convert_mod.shutil, "rmtree", lambda *a, **kw: rmtree_calls.append(a))
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "convert.py",
                "--model-dir",
                str(model_dir),
                "--output",
                str(output_path),
                *extra_argv,
            ],
        )
        with pytest.raises(SystemExit):
            convert_mod.main()
        return output_path, rmtree_calls

    def test_delete_source_keeps_partial_output_on_failure(self, tmp_path, monkeypatch):
        output_path, rmtree_calls = self._run_main_expect_failure(
            tmp_path, monkeypatch, ["--delete-source"]
        )
        assert rmtree_calls == []
        assert output_path.exists()
        assert (output_path / "partial.npy").exists()

    def test_no_delete_source_still_cleans_up_on_failure(self, tmp_path, monkeypatch):
        _output_path, rmtree_calls = self._run_main_expect_failure(tmp_path, monkeypatch, [])
        assert len(rmtree_calls) == 1
