#!/usr/bin/env python3
"""
Upload P0 compressed models to the squishai HuggingFace organization.

Requires HF_TOKEN with write access to squishai org.

Usage:
    export HF_TOKEN=hf_...
    uv run python dev/scripts/upload_to_squishai.py

    # Dry run (no actual upload)
    uv run python dev/scripts/upload_to_squishai.py --dry-run

    # Single model only
    uv run python dev/scripts/upload_to_squishai.py --model 1.5b
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

# P0 models: (compressed_dir, tokenizer_dir, hf_repo, base_model_id)
P0_MODELS = [
    (
        Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16-compressed",
        Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16",
        "squishai/Qwen2.5-1.5B-Instruct-squished",
        "Qwen/Qwen2.5-1.5B-Instruct",
    ),
    (
        Path.home() / "models" / "Qwen3-8B-bf16-compressed",
        Path.home() / "models" / "Qwen3-8B-bf16",
        "squishai/Qwen3-8B-squished",
        "Qwen/Qwen3-8B",
    ),
]

TOKENIZER_FILES = [
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "config.json", "generation_config.json",
    "added_tokens.json",
]

MODEL_CARD = """\
---
license: apache-2.0
tags:
  - squish
  - mlx
  - apple-silicon
  - quantized
  - int8
base_model: {base_model}
---

# {model_name} — Squish format (Apple Silicon)

Pre-squished weights for [Squish](https://github.com/squishai/squish), a sub-second
model server for Apple Silicon.

## Measured performance (Apple M3, 16 GB, March 2026)

| Model | Load time | TTFT | Decode |
|-------|----------:|-----:|-------:|
| Qwen2.5-1.5B | 1.61 s | **148 ms** | 7.5 tok/s |
| Qwen2.5-7B | 3.41 s | **533 ms** | 2.0 tok/s |
| Qwen2.5-14B | 5.93 s | **1,008 ms** | 1.2 tok/s |

Accuracy vs reference model (0-shot, 200 samples):
ARC-Easy 73.5% | HellaSwag 63.0% | PIQA 76.5% | WinoGrande 66.0%

## Install & run

```bash
pip install squish
squish pull {repo}
squish serve {repo} --port 11435
```

Then call via OpenAI-compatible API:

```bash
curl http://localhost:11435/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer squish" \\
  -d '{{"model":"squish","messages":[{{"role":"user","content":"Hello!"}}],"stream":true}}'
```

## What is the squish format?

`squish_weights.safetensors` is BF16, Metal-native layout. Loading maps directly into
unified memory with no dtype conversion — **4.5× faster TTFT** vs v1, **~15× less RAM
during load** vs standard cold-start.

## Source

[github.com/squishai/squish](https://github.com/squishai/squish)
"""


def collect_files(compressed_dir: Path, tokenizer_dir: Path) -> list[tuple[Path, str]]:
    """Return (local_path, repo_path) pairs to upload."""
    uploads: list[tuple[Path, str]] = []

    weights = compressed_dir / "squish_weights.safetensors"
    if not weights.exists():
        raise FileNotFoundError(f"squish_weights.safetensors not found in {compressed_dir}")
    uploads.append((weights, "squish_weights.safetensors"))

    for name in TOKENIZER_FILES:
        p = tokenizer_dir / name
        if p.exists():
            uploads.append((p, name))

    manifest = compressed_dir / "manifest.json"
    if manifest.exists():
        uploads.append((manifest, "manifest.json"))

    return uploads


def upload_model(
    compressed_dir: Path,
    tokenizer_dir: Path,
    repo: str,
    base_model: str,
    hf_token: str,
    dry_run: bool = False,
) -> None:
    model_name = repo.split("/")[-1]
    resolved = compressed_dir.resolve()
    tok_resolved = tokenizer_dir.resolve()

    print(f"\n{'='*60}")
    print(f"  Repo:     https://huggingface.co/{repo}")
    print(f"  Weights:  {resolved}")
    print(f"  Tokenizer:{tok_resolved}")

    try:
        uploads = collect_files(resolved, tok_resolved)
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return

    total_gb = sum(p.stat().st_size for p, _ in uploads) / 1e9
    print(f"  Files ({len(uploads)}, {total_gb:.2f} GB total):")
    for local, remote in uploads:
        print(f"    {remote:<50} {local.stat().st_size/1e6:>8.1f} MB")

    if dry_run:
        print("  [DRY RUN] No files uploaded.")
        return

    from huggingface_hub import HfApi  # noqa: PLC0415
    api = HfApi(token=hf_token)

    print(f"\n  Creating/verifying repo...")
    api.create_repo(repo_id=repo, repo_type="model", private=False, exist_ok=True)
    print(f"  Repo ready.")

    card = MODEL_CARD.format(model_name=model_name, base_model=base_model, repo=repo)
    print("  Uploading README.md...")
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message="Add model card with benchmarks",
    )

    for local_path, repo_path in uploads:
        size_mb = local_path.stat().st_size / 1e6
        print(f"  Uploading {repo_path} ({size_mb:.0f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type="model",
            commit_message=f"Upload {repo_path}",
        )
        print(f"    ✓ {repo_path}")

    print(f"\n  Published → https://huggingface.co/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    ap.add_argument("--model", choices=["1.5b", "qwen3-8b", "all"], default="all")
    args = ap.parse_args()

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.dry_run:
        print("ERROR: HF_TOKEN not set.\n  Run: export HF_TOKEN=hf_<your_token>")
        sys.exit(1)

    models = P0_MODELS
    if args.model == "1.5b":
        models = [P0_MODELS[0]]
    elif args.model == "qwen3-8b":
        models = [P0_MODELS[1]]

    for compressed_dir, tokenizer_dir, repo, base_model in models:
        upload_model(compressed_dir, tokenizer_dir, repo, base_model, hf_token, args.dry_run)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
