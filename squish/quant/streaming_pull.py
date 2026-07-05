"""squish/quant/streaming_pull.py — true per-shard streaming pull: fetch one
raw safetensors shard from Hugging Face, quantize it, delete it, fetch the
next. Never has more than one raw weight shard resident on local disk at
once, regardless of total model size — unlike ``squish quantize-remote``
(Wave 146), which downloads the entire raw model before quantizing anything
and only reclaims shard space *during* the quantization pass.

Wave 147a scope: the non-AWQ path only. AWQ calibration (Waves 143-145)
still needs a shard's weights loaded to run real forward passes across
*all* calibration samples before that shard can be safely discarded, and
today expects the shards it needs already resident locally (see
``squish/quant/awq_streaming.py``) — folding fetch-on-demand into that same
per-layer loop is Wave 147b's job, not this one's. This module is usable
standalone; it does not import anything from the AWQ modules.

Reuses, unchanged:
- ``squish/convert.py``'s ``quantize_tensor`` (the actual per-tensor
  quantization math), ``safe_key``, and ``load_mlx_weights_shard`` (the
  bf16-safe shard loader, Wave 141), so output written by this path is the
  same manifest/tensor-file shape as ``process_weights_streaming``'s.
- ``squish.serving.local_model_scanner``'s pre/post-download security
  scans (W100), run before any bytes are trusted — once for the repo's
  metadata, then again after every single shard lands, before its tensors
  are ever read.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

_LOG = logging.getLogger("squish.quant.streaming_pull")


def pull_and_quantize_shard_by_shard(
    hf_repo: str,
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    token: str | None = None,
    use_int4: bool = True,
    int4_group_size: int | None = None,
    outlier_threshold: float = 20.0,
    verbose: bool = True,
) -> dict:
    """Download and quantize *hf_repo* one raw weight shard at a time.

    ``model_dir`` ends up holding only the non-weight files (config,
    tokenizer, generation config, the shard index) — every ``.safetensors``
    shard is deleted immediately after its tensors are quantized and
    written to ``output_dir``. This matches the existing
    ``--delete-source`` convention (Waves 139/141): the base directory
    keeps existing (and keeps its config/tokenizer) so it can still serve
    as the config source for the compressed model, it just never holds
    more than one raw weight shard at a time.

    Returns the same stats dict shape as
    :func:`squish.convert.process_weights_streaming`: ``n_quantized``,
    ``n_passthrough``, ``orig_f32_bytes``, ``compressed_bytes``,
    ``shards_deleted``, ``source_bytes_reclaimed``.
    """
    from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

    from squish.convert import load_mlx_weights_shard, quantize_tensor, safe_key
    from squish.serving.local_model_scanner import scan_before_load, scan_hf_repo_metadata

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    tensor_dir = output_dir / "tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    meta_scan = scan_hf_repo_metadata(hf_repo, token=token)
    if meta_scan.status == "unsafe":
        raise RuntimeError(
            f"Pre-download security scan failed for {hf_repo}: {meta_scan.findings}"
        )

    if verbose:
        print(f"  [streaming-pull] fetching config/tokenizer for {hf_repo} …")
    snapshot_download(
        repo_id=hf_repo, local_dir=str(model_dir), token=token,
        ignore_patterns=["*.safetensors"],
    )
    scan_result = scan_before_load(model_dir)
    if scan_result.status == "unsafe":
        raise RuntimeError(
            f"Post-download security scan failed for {hf_repo} metadata: {scan_result.findings}"
        )

    # list_repo_files (not the shard index) is the authoritative shard list —
    # it's correct whether the model is sharded (model.safetensors.index.json
    # present) or a single unsharded checkpoint, with no extra round trip to
    # fetch and parse the index file just to enumerate shard names.
    shard_names = sorted(f for f in list_repo_files(hf_repo, token=token) if f.endswith(".safetensors"))
    if not shard_names:
        raise RuntimeError(f"No .safetensors weight files found in {hf_repo}")

    manifest: dict[str, str] = {}
    stats = {
        "n_quantized": 0,
        "n_passthrough": 0,
        "orig_f32_bytes": 0,
        "compressed_bytes": 0,
        "shards_deleted": 0,
        "source_bytes_reclaimed": 0,
    }

    if verbose:
        print(
            f"  [streaming-pull] {len(shard_names)} shard(s) — "
            "never more than one raw shard on disk at a time"
        )

    for shard_idx, shard_name in enumerate(shard_names, 1):
        if verbose:
            print(f"\n  [{shard_idx}/{len(shard_names)}] fetching {shard_name} …")
        shard_path = Path(
            hf_hub_download(repo_id=hf_repo, filename=shard_name, local_dir=str(model_dir), token=token)
        )

        scan_result = scan_before_load(model_dir)
        if scan_result.status == "unsafe":
            raise RuntimeError(
                f"Post-download security scan failed for shard {shard_name}: {scan_result.findings}"
            )

        shard_weights = load_mlx_weights_shard(shard_path)
        for name, arr_f32 in shard_weights.items():
            sk = safe_key(name)
            manifest[name] = sk
            sub = quantize_tensor(
                name, arr_f32, outlier_threshold, [],
                use_int4=use_int4, int4_group_size=int4_group_size,
            )
            for suffix, data in sub.items():
                out_path = tensor_dir / f"{sk}{suffix}.npy"
                np.save(str(out_path), data.astype(np.float16) if suffix == "__pt" else data)

            orig_bytes = arr_f32.nbytes
            comp_bytes = sum(
                (tensor_dir / f"{sk}{sfx}.npy").stat().st_size
                for sfx in sub if not sfx.endswith("__shape")
            )
            stats["orig_f32_bytes"] += orig_bytes
            stats["compressed_bytes"] += comp_bytes
            stats["n_passthrough" if "__pt" in sub else "n_quantized"] += 1
        del shard_weights

        with open(output_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        shard_bytes = shard_path.stat().st_size
        shard_path.unlink()
        stats["shards_deleted"] += 1
        stats["source_bytes_reclaimed"] += shard_bytes
        if verbose:
            print(f"    quantized + deleted {shard_name}  ({shard_bytes / 1e9:.2f} GB freed)")

    (tensor_dir / ".manifest_ready").touch()
    if verbose:
        print(
            f"\n  [streaming-pull] done — {stats['n_quantized']} quantized, "
            f"{stats['n_passthrough']} passthrough, "
            f"{stats['source_bytes_reclaimed'] / 1e9:.2f} GB of raw shards reclaimed"
        )
    return stats
