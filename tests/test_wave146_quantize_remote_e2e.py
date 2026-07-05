"""tests/test_wave146_quantize_remote_e2e.py

Wave 146 — `squish quantize-remote <hf-repo>`: the end-to-end
download -> quantize -> (optionally) push command for models that may be
too large for local RAM.

This pins the core decision logic in `cmd_quantize_remote`
(squish/cli.py) with every network- and MLX-touching stage mocked at its
import boundary. As of Wave 147a/147b, there are three distinct modes
with three distinct downstream mechanisms:

- **full** (fits in RAM): `mlx_lm.load` needs every weight anyway, so
  the whole raw model is downloaded up front, then
  `process_weights_streaming(..., delete_source=True)` quantizes it.
- **streaming** (doesn't fit, but sharded): never downloads the full raw
  model. `collect_activation_scales_streaming` fetches shards on demand
  and deletes them as it goes (`delete_source=True`, `hf_repo`/`token`
  passed through) to compute AWQ scales, then
  `pull_and_quantize_shard_by_shard` fetches every shard *again* to
  actually quantize it with those scales applied -- the shard sequencing
  differs from `none` mode only in that it carries `awq_scales`.
- **none** (`--no-awq`, or too large for RAM with no shard index):
  `pull_and_quantize_shard_by_shard` alone, `awq_scales=None`.

`--push` delegates to the existing `cmd_push` machinery with the
freshly quantized output directory in every mode.

The AWQ-mode decision itself is made from Hugging Face API metadata
alone (`scan_hf_repo_metadata`'s total_size_bytes for the RAM-fit
estimate, `list_repo_files` for shard-index presence) -- no weight
bytes are downloaded before the decision is made.

Downloads, real model loads, and the real quantization pass are all
mocked -- this is a control-flow/wiring test, not an accuracy test
(Wave 145 already covers calibration fidelity against a real model, and
tests/test_wave147a_streaming_pull.py covers the streaming-pull
mechanism itself against a real synthetic model).
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
    the handful that differ (RAM/size numbers, sharded-ness, --no-awq).

    The AWQ-mode decision (Wave 147a) is driven entirely by HF metadata:
    `model_size_gb` controls `scan_hf_repo_metadata`'s `total_size_bytes`
    (peak full-load AWQ RAM = model_size_gb * 2.0 + 2.0), and
    `is_sharded` controls whether `list_repo_files` reports
    `model.safetensors.index.json`. `fetch_repo_metadata` and
    `load_shard_index` run for real (against the already-mocked
    `snapshot_download`/`scan_*`/`list_repo_files`) since they're cheap,
    real wiring, not the thing under test -- only the top-level
    calibration/quantization functions are mocked wholesale.
    """

    def __init__(self, tmp_path, monkeypatch, *, total_ram_gb, free_ram_gb, is_sharded,
                 model_size_gb=None, dir_name="Fake-Model-bf16"):
        self.model_dir = _make_fake_model_dir(tmp_path, name=dir_name)
        self.out_dir = self.model_dir.parent / dir_name.replace("-bf16", "-int4")
        # A plain value (not object()) so two independently-constructed
        # _PatchSets compare equal by value -- needed by the test that checks
        # multiple modes converge on the same downstream call shape.
        self.fake_scales = {"model.layers.0.self_attn.q_proj": "fake-scale-vector"}

        monkeypatch.setattr(cli, "_MODELS_DIR", tmp_path)

        if model_size_gb is None:
            # Comfortably fits by default: peak (model_size*2+2) well under total_ram.
            model_size_gb = max((total_ram_gb - 4.0) / 2.0, 0.1)
        self.model_size_gb = model_size_gb

        self.snapshot_download = MagicMock(return_value=str(self.model_dir))
        self.scan_meta = MagicMock(return_value=types.SimpleNamespace(
            status="safe", findings=[], total_files=2, total_size_bytes=model_size_gb * 1e9,
        ))
        self.scan_bytes = MagicMock(return_value=types.SimpleNamespace(
            status="safe", findings=[], scanned=2,
        ))
        self.ram = MagicMock(return_value=(total_ram_gb, free_ram_gb))
        repo_files = ["config.json", "model.safetensors"]
        if is_sharded:
            repo_files.append("model.safetensors.index.json")
        self.list_repo_files = MagicMock(return_value=repo_files)
        self.shard_index = MagicMock(return_value=MagicMock() if is_sharded else None)
        self.estimate_output_bytes = MagicMock(return_value=1_000_000)
        self.get_free_bytes = MagicMock(return_value=int(1e12))  # plenty of disk

        self.mlx_load = MagicMock(return_value=(MagicMock(), MagicMock()))
        self.load_tokenizer = MagicMock(return_value=MagicMock())
        self.collect_full = MagicMock(return_value=dict(self.fake_scales))
        self.collect_streaming = MagicMock(return_value=dict(self.fake_scales))
        self.process_weights = MagicMock(return_value={})
        self.streaming_pull = MagicMock(return_value={})
        self.cmd_push = MagicMock()

        self._patches = [
            patch("huggingface_hub.snapshot_download", self.snapshot_download),
            patch("huggingface_hub.list_repo_files", self.list_repo_files),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", self.scan_meta),
            patch("squish.serving.local_model_scanner.scan_before_load", self.scan_bytes),
            patch.object(cli, "_ram_available_gb", self.ram),
            patch("squish.quant.shard_index.load_shard_index", self.shard_index),
            patch("squish.convert._estimate_output_bytes", self.estimate_output_bytes),
            patch("squish.convert._get_free_bytes", self.get_free_bytes),
            patch("mlx_lm.load", self.mlx_load),
            patch("mlx_lm.utils.load_tokenizer", self.load_tokenizer),
            patch("squish.quant.awq.collect_activation_scales", self.collect_full),
            patch("squish.quant.awq_streaming.collect_activation_scales_streaming", self.collect_streaming),
            patch("squish.convert.process_weights_streaming", self.process_weights),
            patch("squish.quant.streaming_pull.pull_and_quantize_shard_by_shard", self.streaming_pull),
            patch.object(cli, "cmd_push", self.cmd_push),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()

    def snapshot_download_ever_did_a_full_fetch(self) -> bool:
        """True if any snapshot_download call would have fetched weight
        shards (i.e. wasn't restricted to metadata-only via
        ignore_patterns=["*.safetensors"])."""
        return any(
            call.kwargs.get("ignore_patterns") != ["*.safetensors"]
            for call in self.snapshot_download.call_args_list
        )


class TestQuantizeRemoteAWQModeSelection:
    def test_model_fits_in_ram_uses_full_load_awq(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, is_sharded=True,
        ) as p:
            cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_called_once()
            p.collect_full.assert_called_once()
            p.collect_streaming.assert_not_called()
            p.streaming_pull.assert_not_called()
            p.process_weights.assert_called_once()
            _, kwargs = p.process_weights.call_args
            assert kwargs["awq_scales"] == p.fake_scales
            assert kwargs["delete_source"] is True
            assert p.snapshot_download_ever_did_a_full_fetch(), (
                "full-load AWQ needs mlx_lm.load, which needs every shard resident"
            )

    def test_model_too_large_for_ram_uses_streaming_awq_never_fully_downloaded(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=16.0, free_ram_gb=4.0, is_sharded=True,
            model_size_gb=40.0,  # peak = 40*2+2 = 82 GB, far over 16 GB total
        ) as p:
            cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.process_weights.assert_not_called()

            p.collect_streaming.assert_called_once()
            _, kwargs = p.collect_streaming.call_args
            assert kwargs["delete_source"] is True, (
                "streaming AWQ (Wave 147b) fetches shards on demand and must "
                "delete them as it goes, or peak disk isn't bounded"
            )
            assert kwargs["hf_repo"] == "mlx-community/Fake-Model-bf16"

            p.streaming_pull.assert_called_once()
            _, kwargs = p.streaming_pull.call_args
            assert kwargs["awq_scales"] == p.fake_scales

            assert not p.snapshot_download_ever_did_a_full_fetch(), (
                "streaming AWQ (Wave 147b) must never fully download the raw model — "
                "every snapshot_download call must be metadata-only"
            )

    def test_none_and_streaming_modes_both_use_the_disk_bounded_streaming_pull(self, tmp_path, monkeypatch):
        """The two non-full-load modes converge on the same downstream
        mechanism (pull_and_quantize_shard_by_shard) -- streaming AWQ
        differs from no-AWQ only in the awq_scales it carries, not in
        which function does the actual quantizing."""
        results = []
        for i, (no_awq, model_size_gb) in enumerate([(True, 4.0), (False, 40.0)]):
            with _PatchSet(
                tmp_path, monkeypatch,
                total_ram_gb=16.0, free_ram_gb=4.0, is_sharded=True,
                model_size_gb=model_size_gb, dir_name=f"Fake-Model-{i}-bf16",
            ) as p:
                cli.cmd_quantize_remote(_args(model=f"mlx-community/Fake-Model-{i}-bf16", no_awq=no_awq))
                p.streaming_pull.assert_called_once()
                p.process_weights.assert_not_called()
                _, kwargs = p.streaming_pull.call_args
                results.append(kwargs["use_int4"])

        assert results[0] == results[1] is True

    def test_no_awq_flag_routes_to_streaming_pull_with_no_scales(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, is_sharded=True,
        ) as p:
            cli.cmd_quantize_remote(_args(no_awq=True))

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.collect_streaming.assert_not_called()
            p.process_weights.assert_not_called()
            p.streaming_pull.assert_called_once()
            _, kwargs = p.streaming_pull.call_args
            assert kwargs["use_int4"] is True
            assert kwargs["awq_scales"] is None
            assert not p.snapshot_download_ever_did_a_full_fetch()

    def test_too_large_for_ram_and_not_sharded_routes_to_streaming_pull_with_no_scales(self, tmp_path, monkeypatch):
        """No shard index means a single-file checkpoint -- streaming AWQ
        has nothing to stream over, so this must degrade to the no-AWQ
        streaming-pull path rather than crash or silently guess."""
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=16.0, free_ram_gb=4.0, is_sharded=False,
            model_size_gb=40.0,
        ) as p:
            cli.cmd_quantize_remote(_args())

            p.mlx_load.assert_not_called()
            p.collect_full.assert_not_called()
            p.collect_streaming.assert_not_called()
            p.process_weights.assert_not_called()
            p.streaming_pull.assert_called_once()
            _, kwargs = p.streaming_pull.call_args
            assert kwargs["awq_scales"] is None


class TestQuantizeRemotePushIntegration:
    def test_push_delegates_to_cmd_push_with_quantized_output(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, is_sharded=True,
        ) as p:
            cli.cmd_quantize_remote(_args(push="squishai/fake-model-int4"))

            p.cmd_push.assert_called_once()
            (push_args,), _ = p.cmd_push.call_args
            assert push_args.repo_id == "squishai/fake-model-int4"
            assert push_args.local_dir == str(p.out_dir)

    def test_no_push_flag_does_not_call_cmd_push(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, is_sharded=True,
        ) as p:
            cli.cmd_quantize_remote(_args(push=None))
            p.cmd_push.assert_not_called()


class TestQuantizeRemoteDiskPreflight:
    def test_insufficient_disk_aborts_before_any_calibration(self, tmp_path, monkeypatch):
        with _PatchSet(
            tmp_path, monkeypatch,
            total_ram_gb=64.0, free_ram_gb=40.0, is_sharded=True,
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
            total_ram_gb=64.0, free_ram_gb=40.0, is_sharded=True,
        ) as p:
            p.scan_meta.return_value = types.SimpleNamespace(
                status="unsafe", findings=["fake dangerous pickle"], total_files=1,
                total_size_bytes=0,
            )
            with pytest.raises(SystemExit):
                cli.cmd_quantize_remote(_args())

            p.snapshot_download.assert_not_called()
            p.list_repo_files.assert_not_called()
            p.process_weights.assert_not_called()
            p.streaming_pull.assert_not_called()
