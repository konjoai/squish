#!/usr/bin/env python3
"""
dev/publish_hf.py — Publish a squished model to the Hugging Face Hub.

Uploads the squish-format weights (squish_weights.safetensors + tokenizer files)
to a HuggingFace Model Hub repository so users can do:

    squish pull squish-community/<model>
    # rather than waiting for the one-time conversion

Authentication
──────────────
  export HF_TOKEN=hf_...   # write-access token from huggingface.co/settings/tokens
  # or pass --hf-token hf_...

Usage
─────
  # Publish after squishing a model locally
  squish compress Qwen/Qwen2.5-1.5B-Instruct
  python3 dev/publish_hf.py \\
      --model-dir ~/.cache/squish/Qwen2.5-1.5B-Instruct \\
      --repo squish-community/Qwen2.5-1.5B-Instruct-squished \\
      --base-model Qwen/Qwen2.5-1.5B-Instruct

  # Dry run — show what would be uploaded without pushing
  python3 dev/publish_hf.py --model-dir ... --repo ... --dry-run

  # Private repo
  python3 dev/publish_hf.py --model-dir ... --repo ... --private
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Uploaded in fixed priority order — critical files first.
_SAFETENSORS_NAME = "squish_weights.safetensors"

# Config-like files that should always be included when present.
_TOKENIZER_GLOBS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "config.json",
    "generation_config.json",
]

_MODEL_CARD_TEMPLATE = """\
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

# {model_name} (Squish INT8 / BF16 MLX format)

Pre-squished weights for use with [Squish](https://github.com/wesleyscholl/squish) —
a sub-second model loader for Apple Silicon.

## What is the squish format?

Squish stores weights as `squish_weights.safetensors` (BF16, MLX-native layout).
Loading skips all dtype conversion and maps directly into Metal unified memory,
giving **~54× faster cold-start vs standard `.safetensors`** and **~15× less RAM during load**.

## Usage

```bash
# Pull and serve in one step
squish pull {repo}
squish run {repo}
```

Or use the drop-in OpenAI/Ollama server:

```bash
squish serve {repo} --port 11435
curl http://localhost:11435/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"squish","messages":[{{"role":"user","content":"Hello!"}}]}}'
```

## Performance (Apple Silicon M-series)

| Metric | Value |
|--------|------:|
| Cold load time | ~0.33–0.53 s |
| RAM during load | ~160 MB |
| Peak RAM | ~402 MB |
| Original `.safetensors` needed? | No |

## Base model

[{base_model}](https://huggingface.co/{base_model})

## Squish

- GitHub: https://github.com/wesleyscholl/squish
- Docs: https://github.com/wesleyscholl/squish/tree/main/docs
"""


def _build_card(repo: str, base_model: str) -> str:
    model_name = repo.split("/")[-1]
    return _MODEL_CARD_TEMPLATE.format(
        repo=repo,
        base_model=base_model,
        model_name=model_name,
    )


def _build_card_from_eval(repo: str, base_model: str, eval_json_path: Path) -> str:
    """
    Generate a model card README.md enriched with accuracy benchmarks
    parsed from an eval JSON file produced by `squish bench`.

    The eval JSON is expected to contain a list of task result objects with at
    least ``{"task": str, "metric": str, "score": float}`` entries.  Any
    well-formed squish eval JSON (mmlu, hellaswag, humaneval …) is accepted;
    unknown keys are silently ignored.
    """
    model_name = repo.split("/")[-1]

    # ── Parse eval JSON ────────────────────────────────────────────────────
    try:
        with open(eval_json_path, encoding="utf-8") as fh:
            eval_data = json.load(fh)
    except Exception as exc:
        print(f"  WARN: could not parse eval JSON {eval_json_path}: {exc}",
              file=sys.stderr)
        # Fall back to generic card
        return _build_card(repo, base_model)

    # Normalise to a flat list of {"task", "metric", "score"} dicts
    results: list[dict] = []
    if isinstance(eval_data, list):
        results = eval_data
    elif isinstance(eval_data, dict):
        # Support {"results": [...]} or {"tasks": [...]} wrapper formats
        for key in ("results", "tasks", "benchmarks"):
            if key in eval_data and isinstance(eval_data[key], list):
                results = eval_data[key]
                break

    # Build benchmark table rows
    rows: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        task   = item.get("task") or item.get("name") or item.get("benchmark", "")
        metric = item.get("metric") or item.get("type", "score")
        score  = item.get("score") or item.get("value") or item.get("result")
        if task and score is not None:
            try:
                score_str = f"{float(score):.4f}"
            except (TypeError, ValueError):
                score_str = str(score)
            rows.append(f"| {task} | {metric} | {score_str} |")

    if rows:
        bench_section = (
            "## Accuracy Benchmarks\n\n"
            "Evaluated with `squish bench`.\n\n"
            "| Task | Metric | Score |\n"
            "|------|--------|------:|\n"
            + "\n".join(rows)
            + "\n"
        )
    else:
        bench_section = ""

    base_card = _build_card(repo, base_model)
    # Insert benchmark section just before the final "## Squish" section
    marker = "## Squish\n"
    if marker in base_card and bench_section:
        base_card = base_card.replace(marker, bench_section + "\n" + marker, 1)

    return base_card


def _collect_files(model_dir: Path) -> list[tuple[Path, str]]:
    """
    Return list of (local_path, path_in_repo) pairs to upload.

    Priority:
      1. squish_weights.safetensors  (required)
      2. tokenizer / config files    (best-effort)
    """
    uploads: list[tuple[Path, str]] = []

    # 1. Main weights file (required)
    weights = model_dir / _SAFETENSORS_NAME
    if not weights.exists():
        raise FileNotFoundError(
            f"{_SAFETENSORS_NAME} not found in {model_dir}. "
            "Run `squish compress <model>` first."
        )
    uploads.append((weights, _SAFETENSORS_NAME))

    # 2. Tokenizer + config files — check compressed dir first, then parent (base model dir)
    search_dirs = [model_dir, model_dir.parent]
    for name in _TOKENIZER_GLOBS:
        for search_dir in search_dirs:
            p = search_dir / name
            if p.exists():
                uploads.append((p, name))
                break  # found in first available dir, stop searching

    # 3. Any remaining .json files not already included
    included = {path_in_repo for _, path_in_repo in uploads}
    for json_file in sorted(model_dir.glob("*.json")):
        if json_file.name not in included:
            uploads.append((json_file, json_file.name))

    return uploads


def _sizeof_fmt(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} TB"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Publish a squished model to the Hugging Face Hub",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap.add_argument("--model-dir",  required=True,
                    help="Path to the squish model directory (contains squish_weights.safetensors)")
    ap.add_argument("--repo",       required=True,
                    help="HF repo id, e.g. squish-community/Qwen2.5-1.5B-Instruct-squished")
    ap.add_argument("--base-model", default="",
                    help="Original HF model id for the model card (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--hf-token",   default="",
                    help="HuggingFace write token (falls back to HF_TOKEN env var)")
    ap.add_argument("--private",    action="store_true", default=False,
                    help="Create a private repository")
    ap.add_argument("--dry-run",    action="store_true", default=False,
                    help="Show what would be uploaded without pushing to HF")
    ap.add_argument("--commit-message", default="",
                    help="Custom commit message for the HF push")
    ap.add_argument("--hf-model-card", action="store_true", default=False,
                    help="Auto-generate a model card README.md from eval JSON and upload it.\n"
                         "Looks for eval JSON in dev/results/ (most-recently modified *.json)\n"
                         "or uses the path supplied via --eval-json.")
    ap.add_argument("--eval-json", default="",
                    help="Path to a squish bench eval JSON file used to enrich the model card\n"
                         "with accuracy benchmarks when --hf-model-card is active.")
    args = ap.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        print(f"ERROR: --model-dir {model_dir} does not exist or is not a directory",
              file=sys.stderr)
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    if not hf_token and not args.dry_run:
        print(
            "ERROR: No HuggingFace token found.\n"
            "  Set HF_TOKEN environment variable or pass --hf-token.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Collect files ───────────────────────────────────────────────────────
    try:
        uploads = _collect_files(model_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    total_bytes = sum(p.stat().st_size for p, _ in uploads)

    print(f"\n  Repository : {args.repo}")
    print(f"  Model dir  : {model_dir}")
    print(f"  Files      : {len(uploads)}")
    print(f"  Total size : {_sizeof_fmt(total_bytes)}")
    print(f"  Private    : {args.private}")
    print()
    for local, remote in uploads:
        size = _sizeof_fmt(local.stat().st_size)
        print(f"    {remote:<50} {size:>10}")
    print()

    if args.dry_run:
        print("  DRY RUN — no files were uploaded.")
        return

    # ── Lazy-import huggingface_hub ─────────────────────────────────────────
    try:
        from huggingface_hub import HfApi, CommitOperationAdd  # noqa: PLC0415
    except ImportError:
        print(
            "ERROR: huggingface_hub not installed.\n"
            "  pip install huggingface-hub",
            file=sys.stderr,
        )
        sys.exit(1)

    api = HfApi(token=hf_token)

    # ── Create repo if it doesn't exist ────────────────────────────────────
    try:
        api.create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"  Repo ready: https://huggingface.co/{args.repo}")
    except Exception as exc:
        print(f"ERROR: could not create/access repo: {exc}", file=sys.stderr)
        sys.exit(1)

    # ── Upload model card ───────────────────────────────────────────────────
    if args.hf_model_card:
        # Resolve eval JSON path
        eval_json_path: Path | None = None
        if args.eval_json:
            eval_json_path = Path(args.eval_json).expanduser().resolve()
            if not eval_json_path.is_file():
                print(f"  WARN: --eval-json path not found: {eval_json_path}", file=sys.stderr)
                eval_json_path = None
        if eval_json_path is None:
            # Auto-discover: most recently modified *.json in dev/results/
            results_dir = Path(__file__).resolve().parent / "results"
            if results_dir.is_dir():
                candidates = sorted(results_dir.glob("*.json"),
                                    key=lambda p: p.stat().st_mtime, reverse=True)
                if candidates:
                    eval_json_path = candidates[0]
                    print(f"  Using eval JSON : {eval_json_path}")

        if eval_json_path is not None:
            card_text = _build_card_from_eval(
                args.repo,
                args.base_model or args.repo.split("/")[-1],
                eval_json_path,
            )
        else:
            card_text = _build_card(args.repo, args.base_model or args.repo.split("/")[-1])
    else:
        card_text = _build_card(args.repo, args.base_model or args.repo.split("/")[-1])
    try:
        api.upload_file(
            path_or_fileobj=card_text.encode(),
            path_in_repo="README.md",
            repo_id=args.repo,
            repo_type="model",
            commit_message="Add Squish model card",
        )
        print("  Uploaded   : README.md (model card)")
    except Exception as exc:
        print(f"  WARN: could not upload model card: {exc}", file=sys.stderr)

    # ── Upload files in a single commit ────────────────────────────────────
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
        for local, remote in uploads
    ]
    commit_msg = args.commit_message or f"Add squish-format weights ({_sizeof_fmt(total_bytes)})"

    try:
        commit = api.create_commit(
            repo_id=args.repo,
            repo_type="model",
            operations=operations,
            commit_message=commit_msg,
        )
        print(f"\n  Committed  : {commit.commit_url}")
    except Exception as exc:
        print(f"ERROR: commit failed: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Model page : https://huggingface.co/{args.repo}")
    print(f"  Pull with  : squish pull {args.repo}\n")


if __name__ == "__main__":
    main()
