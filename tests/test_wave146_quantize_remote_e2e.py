"""tests/test_wave146_quantize_remote_e2e.py

Wave 146 — `squish quantize-remote <hf-repo>`: the end-to-end
download -> quantize -> (optionally) push command for models that may be
too large for local RAM.

This pins the core decision logic in `cmd_quantize_remote`
(squish/cli.py) with every network- and MLX-touching stage mocked at its
import boundary:

- when the model fits comfortably in RAM, full-load AWQ
  (`collect_activation_scales` via `mlx_lm.load`) is used
- when it doesn't (but the model is sharded), layer-at-a-time streaming
  AWQ (`collect_activation_scales_streaming`) is used instead, with
  `delete_source=False` during calibration (raw shards must survive
  calibration -- the quantization pass right after still needs them)
- `--no-awq` skips calibration entirely
- a model with no shard index that also doesn't fit in RAM has no path
  to AWQ at all and falls back to plain quantization rather than
  guessing or crashing
- regardless of which calibration path ran, `process_weights_streaming`
  is always invoked exactly once, with `delete_source=True` -- the
  caller (a human running the command) never needs to know which
  calibration strategy was used to get a working, space-bounded result
- `--push` delegates to the existing `cmd_push` machinery with the
  freshly quantized output directory

Downloads, real model loads, and the real quantization pass are all
mocked -- this is a control-flow/wiring test, not an accuracy test
(Wave 145 already covers calibration fidelity against a real model).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

from squish import cli  # noqa: E402


def _make_fake_model_dir(tmp_path: Path, name: str = "Fake-Model-bf16") -> Path:
    """A minimal on-disk model dir: config.json + one tiny fake shard.

    Never actually loaded by mlx_lm/safetensors in these tests (every
    stage that would read it is mocked), it just needs to exist so
    filesystem-touching helpers (glob, stat) don't blow up.
    """
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "config.json").write_text('{"model_type": "llama"}')
    (d / "model.safetensors").write_bytes(b"\x00" * 1024)
    return d


def _args(model="mlx-community/Fake-Model-bf16", **overrides):
    defaults = dict(
        model=model,
        int8=False,
        no_awq=False,
        awq_samples=4,
        push=None,
        private=False,
        commit_message=None,
        output=None,
        token="",
        models_dir="",
        verbose=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


class _PatchSet:
    """Bundles the mocks every scenario needs, so each test only overrides
    the handful that differ (RAM numbers, shard index presence, --no-awq)."""

    def __init__(self, tmp_path, monkeypatch, *, total_ram_gb, free_ram_gb, shard_index,
                 full_awq_peak_gb=None, dir_name="Fake-Model-bf16"):
        self.model_dir = _make_fake_model_dir(tmp_path, name=dir_name)
        self.out_dir = self.model_dir.parent / dir_name.replace("-bf16", "-int4")
        # A plain value (not object()) so two independently-constructed
        # _PatchSets compare equal by value -- needed by the test that checks
        # both AWQ strategies converge on the same downstream call shape.
        self.fake_scales = {"model.layers.0.self_attn.q_proj": "fake-scale-vector"}

        monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path)

        # Full-load AWQ's peak RAM estimate is driven purely by real on-disk
        # model size (see squish/cli.py:_peak_ram_estimate_gb) -- mocked
        # directly here so "fits" vs "doesn't fit" is deterministic and
        # independent of the tiny fixture file's actual byte size.
        if full_awq_peak_gb is None:
            full_awq_peak_gb = min(total_ram_gb, free_ram_gb) / 2  # comfortably fits by default
        self.peak_ram_estimate = MagicMock(return_value=(2.0, full_awq_peak_gb, 1.0))

        self.snapshot_download = MagicMock(return_value=str(self.model_dir))
        self.scan_meta = MagicMock(return_value=types.SimpleNamespace(
            status="safe", findings=[], total_files=2,
        ))
        self.scan_bytes = MagicMock(return_value=types.SimpleNamespace(
            status="safe", findings=[], scanned=2,
        ))
        self.ram = MagicMock(return_value=(total_ram_gb, free_ram_gb))
        self.shard_index = MagicMock(return_value=shard_index)
        self.estimate_output_bytes = MagicMock(return_value=1_000_000)
        self.get_free_bytes = MagicMock(return_value=int(1e12))  # plenty of disk

        self.mlx_load = MagicMock(return_value=(MagicMock(), MagicMock()))
        self.load_tokenizer = MagicMock(return_value=MagicMock())
        self.collect_full = MagicMock(return_value=dict(self.fake_scales))
        self.collect_streaming = MagicMock(return_value=dict(self.fake_scales))
        self.process_weights = MagicMock(return_value={})
        self.cmd_push = MagicMock()

        self._patches = [
            patch("huggingface_hub.snapshot_download", self.snapshot_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", self.scan_meta),
            patch("squish.serving.local_model_scanner.scan_before_load", self.scan_bytes),
            patch.object(cli, "_ram_available_gb", self.ram),
            patch.object(cli, "_peak_ram_estimate_gb", self.peak_ram_estimate),
            patch("squish.quant.shard_index.load_shard_index", self.shard_index),
            patch("squish.convert._estimate_output_bytes", self.estimate_output_bytes),
            patch("squish.convert._get_free_bytes", self.get_free_bytes),
            patch("mlx_lm.load", self.mlx_load),
            patch("mlx_lm.utils.load_tokenizer", self.load_tokenizer),
            patch("squish.quant.awq.collect_activation_scales", self.collect_full),
            patch("squish.quant.awq_streaming.collect_activation_scales_streaming", self.collect_streaming),
            patch("squish.convert.process_weights_streaming", self.process_weights),
            patch.object(cli, "cmd_push", self.cmd_push),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


class TestQuantizeRemoteAWQModeSelection:
    def test_model_fits_in_ram_uses_full_load_awq(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, shard_index=MagicMock(),
        ) as p:
            cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_called_once()
            p.collect_full.assert_called_once()
            p.collect_streaming.assert_not_called()
            p.process_weights.assert_called_once()
            _, kwargs = p.process_weights.call_args
            assert kwargs["awq_scales"] == p.fake_scales
            assert kwargs["delete_source"] is True

    def test_model_too_large_for_ram_uses_streaming_awq(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=16.0, free_ram_gb=4.0, shard_index=MagicMock(),
            full_awq_peak_gb=40.0,
        ) as p:
            cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.collect_streaming.assert_called_once()
            _, kwargs = p.collect_streaming.call_args
            assert kwargs["delete_source"] is False, (
                "streaming calibration must not delete shards -- the "
                "quantization pass right after still needs them"
            )
            p.process_weights.assert_called_once()
            _, kwargs = p.process_weights.call_args
            assert kwargs["awq_scales"] == p.fake_scales
            assert kwargs["delete_source"] is True

    def test_caller_does_not_need_to_know_which_path_ran(self, tmp_path, monkeypatch):
        """Both AWQ strategies converge on the same downstream call shape --
        the whole point of this command is that the caller shouldn't have
        to pick a strategy themselves."""
        results = []
        for i, (total_ram_gb, peak_gb) in enumerate([(64.0, 4.0), (16.0, 40.0)]):
            with _PatchSet(
                tmp_path, monkeypatch,
                total_ram_gb=total_ram_gb, free_ram_gb=4.0, shard_index=MagicMock(),
                full_awq_peak_gb=peak_gb, dir_name=f"Fake-Model-{i}-bf16",
            ) as p:
                cli.cmd_quantize_remote(_args(model=f"mlx-community/Fake-Model-{i}-bf16"))
                _, kwargs = p.process_weights.call_args
                results.append((kwargs["awq_scales"], kwargs["delete_source"], kwargs["use_int4"]))

        assert results[0] == results[1]

    def test_no_awq_flag_skips_calibration_entirely(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, shard_index=MagicMock(),
        ) as p:
            cli.cmd_quantize_remote(_args(no_awq=True))

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.collect_streaming.assert_not_called()
            p.process_weights.assert_called_once()
            _, kwargs = p.process_weights.call_args
            assert kwargs["awq_scales"] is None
            assert kwargs["delete_source"] is True

    def test_too_large_for_ram_and_no_shard_index_falls_back_to_plain(self, tmp_path, monkeypatch):
        """No shard index means a single-file checkpoint -- streaming AWQ
        has nothing to stream over, so this must degrade to plain
        quantization rather than crash or silently guess."""
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=16.0, free_ram_gb=4.0, shard_index=None,
            full_awq_peak_gb=40.0,
        ) as p:
            cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.collect_streaming.assert_not_called()
            p.process_weights.assert_called_once()
            _, kwargs = p.process_weights.call_args
            assert kwargs["awq_scales"] is None


class TestQuantizeRemotePushIntegration:
    def test_push_delegates_to_cmd_push_with_quantized_output(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, shard_index=MagicMock(),
        ) as p:
            cli.cmd_quantize_remote(_args(push="squishai/fake-model-int4"))

            p.cmd_push.assert_called_once()
            (push_args,), _ = p.cmd_push.call_args
            assert push_args.repo_id == "squishai/fake-model-int4"
            assert push_args.local_dir == str(p.out_dir)

    def test_no_push_flag_does_not_call_cmd_push(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, shard_index=MagicMock(),
        ) as p:
            cli.cmd_quantize_remote(_args(push=None))
            p.cmd_push.assert_not_called()


class TestQuantizeRemoteDiskPreflight:
    def test_insufficient_disk_aborts_before_any_calibration(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, shard_index=MagicMock(),
        ) as p:
            p.get_free_bytes.return_value = 0  # no disk at all
            with pytest.raises(SystemExit):
                cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.process_weights.assert_not_called()


class TestQuantizeRemoteSecurityScan:
    def test_unsafe_metadata_scan_aborts_before_download(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, shard_index=MagicMock(),
        ) as p:
            p.scan_meta.return_value = types.SimpleNamespace(
                status="unsafe", findings=["fake dangerous pickle"], total_files=1,
            )
            with pytest.raises(SystemExit):
                cli.cmd_quantize_remote(_args())

            p.snapshot_download.assert_not_called()
            p.process_weights.assert_not_called()
