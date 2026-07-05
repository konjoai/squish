"""tests/test_wave147b_awq_streaming_pull.py

Wave 147b — AWQ-integrated true per-shard streaming pull: fold
fetch-on-demand into the streaming AWQ calibration loop itself
(``squish.quant.awq_streaming.collect_activation_scales_streaming``'s new
``hf_repo``/``token`` params), and give the quantization pass
(``squish.quant.streaming_pull.pull_and_quantize_shard_by_shard``) the
ability to apply the resulting AWQ scales -- so the "doesn't fit in RAM"
AWQ path (Wave 146/147a's ``streaming`` mode) never needs the full raw
model downloaded up front either, matching Wave 147a's non-AWQ mode.

Reuses Wave 144's real synthetic Llama-shaped model construction (a real
``mlx_lm`` ``TransformerBlock`` is built and run per layer -- this is not
a mocked-forward-pass test), but fetches shards from a faked
``huggingface_hub`` layer instead of assuming they're already local,
mirroring Wave 147a's ``_FakeHfHub`` approach.

The trade this wave makes: peak disk stays bounded to one raw shard in
flight across *both* the calibration pass and the quantization pass, at
the cost of fetching every shard twice (once per pass). That trade-off
is inherent to needing calibration to complete (and its raw shards
released) before the quantization pass can even start -- verified here,
not just asserted.
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

pytest.importorskip(
    "mlx_lm.models.llama",
    reason="mlx_lm blocked in sandbox — set CI=1 to enable",
    exc_type=ImportError,
)

from squish.quant.awq import prepare_awq_application
from squish.quant.awq_streaming import collect_activation_scales_streaming
from squish.quant.shard_index import load_shard_index
from squish.quant.streaming_pull import pull_and_quantize_shard_by_shard

HIDDEN = 64
N_HEADS = 4
N_KV = 2
HEAD_DIM = 16
INTER = 128
VOCAB = 500
N_LAYERS = 2


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [(ord(c) % (VOCAB - 1)) + 1 for c in text[:12]]


def _build_fake_repo_two_shards(repo_dir: Path) -> dict[str, np.ndarray]:
    """A real (small) Llama-shaped 2-layer model, laid out as an HF repo
    would be: shard1 = embed_tokens + layer0, shard2 = layer1."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "llama",
        "hidden_size": HIDDEN,
        "num_hidden_layers": N_LAYERS,
        "intermediate_size": INTER,
        "num_attention_heads": N_HEADS,
        "num_key_value_heads": N_KV,
        "head_dim": HEAD_DIM,
        "rms_norm_eps": 1e-5,
        "vocab_size": VOCAB,
        "tie_word_embeddings": True,
    }
    (repo_dir / "config.json").write_text(json.dumps(config))

    rng = np.random.default_rng(0)
    tensors = {
        "model.embed_tokens.weight": (rng.standard_normal((VOCAB, HIDDEN)) * 0.02).astype(np.float32),
    }
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = np.ones((HIDDEN,), dtype=np.float32)
        tensors[p + "post_attention_layernorm.weight"] = np.ones((HIDDEN,), dtype=np.float32)
        tensors[p + "self_attn.q_proj.weight"] = (
            rng.standard_normal((N_HEADS * HEAD_DIM, HIDDEN)) * 0.02
        ).astype(np.float32)
        tensors[p + "self_attn.k_proj.weight"] = (
            rng.standard_normal((N_KV * HEAD_DIM, HIDDEN)) * 0.02
        ).astype(np.float32)
        tensors[p + "self_attn.v_proj.weight"] = (
            rng.standard_normal((N_KV * HEAD_DIM, HIDDEN)) * 0.02
        ).astype(np.float32)
        tensors[p + "self_attn.o_proj.weight"] = (
            rng.standard_normal((HIDDEN, N_HEADS * HEAD_DIM)) * 0.02
        ).astype(np.float32)
        tensors[p + "mlp.gate_proj.weight"] = (rng.standard_normal((INTER, HIDDEN)) * 0.02).astype(np.float32)
        tensors[p + "mlp.up_proj.weight"] = (rng.standard_normal((INTER, HIDDEN)) * 0.02).astype(np.float32)
        tensors[p + "mlp.down_proj.weight"] = (rng.standard_normal((HIDDEN, INTER)) * 0.02).astype(np.float32)

    shard1 = {k: v for k, v in tensors.items() if k.startswith("model.embed_tokens") or k.startswith("model.layers.0.")}
    shard2 = {k: v for k, v in tensors.items() if k not in shard1}
    save_file(shard1, str(repo_dir / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(repo_dir / "model-00002-of-00002.safetensors"))

    weight_map = {k: "model-00001-of-00002.safetensors" for k in shard1}
    weight_map.update({k: "model-00002-of-00002.safetensors" for k in shard2})
    (repo_dir / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return tensors


class _FakeHfHub:
    """Fakes huggingface_hub's download functions against a local
    directory standing in for the remote repo -- copies files instead of
    making network calls."""

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
    meta = MagicMock(return_value=MagicMock(status="safe", findings=[]))
    byte_scan = MagicMock(return_value=MagicMock(status="clean", findings=[], scanned=1))
    return meta, byte_scan


class TestFetchOnDemandCalibrationNeverHoldsMoreThanOneShard:
    def test_shard_count_on_disk_is_zero_before_every_fetch(self, tmp_path):
        repo_dir = tmp_path / "fake_repo"
        _build_fake_repo_two_shards(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        fake_hub = _FakeHfHub(repo_dir)
        shard_counts_before_fetch = []

        def _tracking_hf_hub_download(repo_id, filename, local_dir=None, token=None):
            if filename.endswith(".safetensors"):
                shard_counts_before_fetch.append(len(list(Path(local_dir).glob("*.safetensors"))))
            return fake_hub.hf_hub_download(repo_id, filename, local_dir=local_dir, token=token)

        meta_scan, byte_scan = _safe_scan()
        with (
            patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
            patch("huggingface_hub.hf_hub_download", _tracking_hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            from squish.quant.streaming_pull import fetch_repo_metadata

            fetch_repo_metadata("fake/Fake-Model-bf16", model_dir, verbose=False)
            shard_index = load_shard_index(model_dir)
            scales = collect_activation_scales_streaming(
                model_dir, _FakeTokenizer(), shard_index, n_samples=2, seq_len=8, verbose=False,
                delete_source=True, hf_repo="fake/Fake-Model-bf16",
            )

        assert scales is not None
        assert len(scales) == 7 * N_LAYERS
        for s in scales.values():
            assert not np.isnan(s).any()
        assert shard_counts_before_fetch == [0, 0], (
            "a raw shard was still on disk when the next one was fetched during "
            "calibration — peak disk is no longer bounded to one shard at a time"
        )
        assert list(model_dir.glob("*.safetensors")) == []
        assert (model_dir / "config.json").exists()

    def test_without_hf_repo_still_requires_local_shard_and_errors_clearly(self, tmp_path):
        """Backward compatibility: hf_repo=None (the default, and every
        pre-Wave-147b caller) must still behave exactly as before -- a
        missing shard is the caller's bug, not something to silently fetch."""
        repo_dir = tmp_path / "fake_repo"
        _build_fake_repo_two_shards(repo_dir)
        model_dir = tmp_path / "local" / "Fake-Model-bf16"
        model_dir.mkdir(parents=True)
        shutil.copy2(repo_dir / "config.json", model_dir / "config.json")
        shutil.copy2(repo_dir / "model.safetensors.index.json", model_dir / "model.safetensors.index.json")
        # Deliberately do NOT copy the .safetensors shards.

        shard_index = load_shard_index(model_dir)
        with pytest.raises(FileNotFoundError, match="no hf_repo was given"):
            collect_activation_scales_streaming(
                model_dir, _FakeTokenizer(), shard_index, n_samples=2, seq_len=8, verbose=False,
            )


class TestStreamingPullAppliesAwqScales:
    def test_awq_scaled_quantization_differs_from_unscaled(self, tmp_path):
        """Runs the real fetch-on-demand calibration pass to get real AWQ
        scales, then runs the real quantization pass twice (with and
        without those scales) against a fresh copy of the same repo --
        proving prepare_awq_application/_apply_awq_single actually changed
        the tensors that got quantized, not just that the code ran."""
        repo_dir = tmp_path / "fake_repo"
        _build_fake_repo_two_shards(repo_dir)
        meta_scan, byte_scan = _safe_scan()

        calib_model_dir = tmp_path / "calib" / "Fake-Model-bf16"
        fake_hub_calib = _FakeHfHub(repo_dir)
        with (
            patch("huggingface_hub.snapshot_download", fake_hub_calib.snapshot_download),
            patch("huggingface_hub.hf_hub_download", fake_hub_calib.hf_hub_download),
            patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
            patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
        ):
            from squish.quant.streaming_pull import fetch_repo_metadata

            fetch_repo_metadata("fake/Fake-Model-bf16", calib_model_dir, verbose=False)
            shard_index = load_shard_index(calib_model_dir)
            awq_scales = collect_activation_scales_streaming(
                calib_model_dir, _FakeTokenizer(), shard_index, n_samples=2, seq_len=8, verbose=False,
                delete_source=True, hf_repo="fake/Fake-Model-bf16",
            )
        assert awq_scales
        proj_apply, ln_apply = prepare_awq_application(awq_scales)
        assert proj_apply, "expected at least one projection to receive an AWQ scale"

        def _quantize(dest_name, scales):
            model_dir = tmp_path / dest_name / "Fake-Model-bf16"
            output_dir = tmp_path / dest_name / "Fake-Model-int4"
            fake_hub = _FakeHfHub(repo_dir)
            with (
                patch("huggingface_hub.snapshot_download", fake_hub.snapshot_download),
                patch("huggingface_hub.list_repo_files", fake_hub.list_repo_files),
                patch("huggingface_hub.hf_hub_download", fake_hub.hf_hub_download),
                patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan),
                patch("squish.serving.local_model_scanner.scan_before_load", byte_scan),
            ):
                pull_and_quantize_shard_by_shard(
                    "fake/Fake-Model-bf16", model_dir, output_dir,
                    use_int4=False, awq_scales=scales, verbose=False,
                )
            return output_dir

        out_unscaled = _quantize("unscaled", None)
        out_scaled = _quantize("scaled", awq_scales)

        q_key = next(iter(proj_apply))  # e.g. "model.layers.0.self_attn.q_proj"
        sk_name = None
        manifest_unscaled = json.loads((out_unscaled / "manifest.json").read_text())
        for name, sk in manifest_unscaled.items():
            if name == f"{q_key}.weight":
                sk_name = sk
                break
        assert sk_name is not None

        a = np.load(out_unscaled / "tensors" / f"{sk_name}__q.npy")
        b = np.load(out_scaled / "tensors" / f"{sk_name}__q.npy")
        assert a.shape == b.shape
        assert not np.array_equal(a, b), (
            "AWQ-scaled quantization produced byte-identical output to the "
            "unscaled run — the scale application had no effect"
        )


class TestFetchRepoMetadataAndEnsureShardLocal:
    def test_ensure_shard_local_returns_existing_file_without_fetching(self, tmp_path):
        from squish.quant.streaming_pull import ensure_shard_local

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        shard = model_dir / "model.safetensors"
        shard.write_bytes(b"already here")

        hf_hub_download = MagicMock()
        with patch("huggingface_hub.hf_hub_download", hf_hub_download):
            result = ensure_shard_local("model.safetensors", model_dir, hf_repo="fake/repo")

        assert result == shard
        hf_hub_download.assert_not_called()

    def test_fetch_repo_metadata_aborts_on_unsafe_scan(self, tmp_path):
        from squish.quant.streaming_pull import fetch_repo_metadata

        model_dir = tmp_path / "model"
        meta_scan = MagicMock(return_value=MagicMock(status="unsafe", findings=["bad file"]))
        with patch("squish.serving.local_model_scanner.scan_hf_repo_metadata", meta_scan):
            with pytest.raises(RuntimeError, match="security scan"):
                fetch_repo_metadata("fake/repo", model_dir, verbose=False)
