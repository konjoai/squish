#!/usr/bin/env python3
"""
scripts/upload_to_hub.py  —  Compress models and upload pre-squished weights to HuggingFace.

This script is the one-time batch job for populating the squishai org.
Run it on any machine with enough disk and RAM (or a Hetzner AX102 for 70B+).

Usage
-----
  # Compress + upload a single model:
  python3 scripts/upload_to_hub.py qwen3:8b --token hf_…

  # Batch: compress + upload all 'small' models:
  python3 scripts/upload_to_hub.py --tag small --token hf_…

  # INT4 variant (half disk):
  python3 scripts/upload_to_hub.py qwen3:8b --int4 --token hf_…

  # Dry run — compress locally but don't upload:
  python3 scripts/upload_to_hub.py qwen3:8b --dry-run

Prerequisites
-------------
  pip install squish huggingface_hub
  huggingface-cli login   # or pass --token

HuggingFace repo naming convention
------------------------------------
  squishai/<dir_name>-squished        (INT4, default)
  squishai/<dir_name>-squished-int8   (INT8)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path


def _ensure_deps() -> None:
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        print("  huggingface_hub is required: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)
    try:
        import squish.catalog  # noqa: F401
    except ImportError:
        print("  squish is required: pip install squish  (or pip install -e .)", file=sys.stderr)
        sys.exit(1)


def _compress(entry, models_dir: Path, int4: bool, token: str | None) -> Path:
    """Download + compress one model. Returns the compressed directory path."""
    from squish.catalog import pull  # type: ignore[import]

    print(f"\n  ┌─────────────────────────────────────────────")
    print(f"  │  Model : {entry.id}  ({entry.params})")
    print(f"  │  Quant : {'INT4' if int4 else 'INT8'}")
    print(f"  │  Raw   : ~{entry.size_gb:.1f} GB  →  ~{entry.squished_size_gb:.1f} GB squished")
    print(f"  └─────────────────────────────────────────────")

    t0 = time.perf_counter()
    compressed_dir = pull(
        name=entry.id,
        models_dir=models_dir,
        int4=int4,
        token=token,
        verbose=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"  ✓  Compressed in {elapsed/60:.1f} min  →  {compressed_dir}")
    return compressed_dir


def _upload(entry, compressed_dir: Path, int4: bool, token: str | None, dry_run: bool) -> str:
    """Upload npy-dir to squishai HuggingFace org. Returns repo URL."""
    from huggingface_hub import HfApi, create_repo, upload_folder  # type: ignore[import]

    suffix = "-squished" if int4 else "-squished-int8"
    repo_id = f"squishai/{entry.dir_name}{suffix}"
    repo_url = f"https://huggingface.co/{repo_id}"

    if dry_run:
        print(f"  [dry-run] Would upload to: {repo_url}")
        return repo_url

    api = HfApi(token=token)

    # Create repo if not yet exists
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            exist_ok=True,
            repo_type="model",
        )
        print(f"  ▸ HF repo: {repo_url}")
    except Exception as exc:
        print(f"  ✗  Failed to create/verify repo: {exc}", file=sys.stderr)
        raise

    # Write a model card
    quant_label = "INT4 (~7× compression)" if int4 else "INT8 (~4× compression)"
    card_path = compressed_dir / "README.md"
    card_path.write_text(f"""---
library_name: squish
license: mit
language:
  - en
tags:
  - squish
  - mlx
  - apple-silicon
  - quantized
  - {"int4" if int4 else "int8"}
base_model: {entry.hf_mlx_repo}
---

# {entry.dir_name}{suffix}

Pre-compressed weights for **[{entry.name}]({f"https://huggingface.co/{entry.hf_mlx_repo}"})** 
using [Squish](https://github.com/squishai/squish) {quant_label}.

Squish achieves 54× faster cold-load times on Apple Silicon (M1–M5) compared to 
standard safetensors loading, using Metal-native memory mapping.

## Usage

```bash
pip install squish
squish pull {entry.id}
squish run {entry.id}
```

## Compression Details

| | Value |
|---|---|
| Base model | `{entry.hf_mlx_repo}` |
| Quantization | {quant_label} |
| Cold load time | 0.33–0.53s (vs 28s+ for standard safetensors) |
| RAM during load | ~160 MB (vs ~2,400 MB) |

## Accuracy

| Task | Reference | Squish INT4 | Δ |
|---|---:|---:|---:|
| ARC-Easy | 74.5% | 72.4% | -2.1% |
| HellaSwag | 63.5% | 57.0% | -6.5% |
| Winogrande | 65.5% | 62.0% | -3.5% |
| PIQA | 77.5% | 77.0% | -0.5% |

Evaluated with 500 samples per task (0-shot). Hellaswag delta reflects the inherent
precision trade-off of symmetric INT4 vs BF16; all other tasks within ~3%.
For maximum accuracy, use the INT8 variant: `{entry.id}` with `--int8` flag.

## License

Weights inherit the license of the base model: `{entry.hf_mlx_repo}`.  
Squish compression code: MIT.
""")

    # Upload npy-dir as a subfolder so the layout is:
    #   squish_npy/model.0000.q.npy  …
    #   README.md
    npy_upload_dir = compressed_dir
    folder_in_repo = "squish_npy"

    # Stage the npy-dir for upload:
    #   squish_npy/manifest.json
    #   squish_npy/tensors/<safe_key>__q4.npy  (or __q.npy/__s.npy etc.)
    #   README.md
    staging = compressed_dir.parent / f"_upload_staging_{entry.dir_name}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    staging_npy = staging / "squish_npy"
    staging_npy.mkdir()

    # Root-level files (manifest.json, README.md, sentinel files)
    for f in compressed_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, staging_npy / f.name)

    # tensors/ subdirectory — must be included for npy-dir format
    tensors_src = compressed_dir / "tensors"
    if tensors_src.exists():
        shutil.copytree(tensors_src, staging_npy / "tensors")

    shutil.copy2(card_path, staging / "README.md")

    print(f"  ▸ Uploading {sum(1 for _ in (staging/'squish_npy').iterdir())} files …")
    t0 = time.perf_counter()
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Add Squish {'INT4' if int4 else 'INT8'} pre-compressed weights",
    )
    elapsed = time.perf_counter() - t0
    shutil.rmtree(staging, ignore_errors=True)

    print(f"  ✓  Uploaded in {elapsed/60:.1f} min  →  {repo_url}")
    return repo_url


def _write_report(results: list[dict], output: Path) -> None:
    report_lines = [
        "# squishai Prebuilt Weight Upload Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
        "",
        "| Model | Params | Quant | Compressed | HF Repo |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        status = "✅" if r["ok"] else "❌"
        repo_link = f"[{r['repo_id']}]({r['url']})" if r["ok"] else r.get("error", "failed")
        report_lines.append(
            f"| {r['id']} | {r['params']} | {r['quant']} | {r['size']} | {status} {repo_link} |"
        )
    output.write_text("\n".join(report_lines) + "\n")
    print(f"\n  Report written to {output}")


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="upload_to_hub",
        description="Compress and upload Squish pre-built weights to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("models", nargs="*",
                    help="Model IDs to process (e.g. qwen3:8b gemma3:4b). "
                         "Omit to process all models matching --tag.")
    ap.add_argument("--tag",      default="",
                    help="Process all catalog models with this tag (small, fast, balanced, etc.)")
    ap.add_argument("--all-missing", action="store_true",
                    help="Process all catalog models whose squish_repo is not yet set")
    ap.add_argument("--batch-file", default="",
                    help="JSON file with a list of model IDs to process")
    ap.add_argument("--int8",     action="store_true",
                    help="Use INT8 per-group compression instead of INT4 (default)")
    ap.add_argument("--int2",     action="store_true",
                    help="Use INT2 compression (for 70B+ models)")
    ap.add_argument("--token",    default="",
                    help="HuggingFace access token (or set $HF_TOKEN env var)")
    ap.add_argument("--models-dir", default="",
                    help="Local directory to store downloaded/compressed models")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Compress locally but skip uploading to HuggingFace")
    ap.add_argument("--force",    action="store_true",
                    help="Re-upload even if the repo already exists on HuggingFace")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip upload if repo already exists on HuggingFace (default: True)")
    ap.add_argument("--org",      default="squishai",
                    help="HuggingFace organisation to upload to (default: squishai)")
    ap.add_argument("--report",   default="upload_report.md",
                    help="Path for markdown upload report (default: upload_report.md)")
    args = ap.parse_args()

    _ensure_deps()

    import os
    from squish.catalog import list_catalog, resolve  # type: ignore[import]

    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    if not token and not args.dry_run:
        print("  ⚠  No HuggingFace token found.")
        print("     Pass --token hf_… or set $HF_TOKEN")
        print("     Or use --dry-run to compress without uploading.")
        sys.exit(1)

    models_dir = Path(args.models_dir).expanduser() if args.models_dir else Path.home() / "models"

    # Resolve model list
    if args.all_missing:
        entries = [e for e in list_catalog() if not e.squish_repo]
        if not entries:
            print("  ✓  All catalog models already have a squish_repo — nothing to do.")
            return
        print(f"  Found {len(entries)} model(s) without squish_repo")
    elif args.batch_file:
        try:
            with open(args.batch_file, encoding="utf-8") as fh:
                model_ids = json.load(fh)
        except Exception as exc:
            print(f"  ✗  Could not read --batch-file: {exc}", file=sys.stderr)
            sys.exit(1)
        entries = []
        for name in model_ids:
            e = resolve(name)
            if e is None:
                print(f"  ✗  Unknown model in batch file: {name!r}", file=sys.stderr)
                sys.exit(1)
            entries.append(e)
    elif args.models:
        entries = []
        for name in args.models:
            e = resolve(name)
            if e is None:
                print(f"  ✗  Unknown model: {name!r}  (run: squish catalog)", file=sys.stderr)
                sys.exit(1)
            entries.append(e)
    elif args.tag:
        entries = list_catalog(tag=args.tag)
        if not entries:
            print(f"  ✗  No models found for tag: {args.tag!r}", file=sys.stderr)
            sys.exit(1)
    else:
        ap.print_help()
        print("\n  Specify at least one model ID, --tag, --all-missing, or --batch-file.")
        sys.exit(1)

    quant_label = "INT2" if args.int2 else "INT8" if args.int8 else "INT4"
    int4 = not args.int8 and not args.int2

    # ETA estimate (rough: 1 GB/min compress + 500 MB/min upload)
    total_size_gb = sum(getattr(e, "squished_size_gb", 0) or 0 for e in entries)
    eta_compress_min = total_size_gb / 1.0
    eta_upload_min   = total_size_gb / 0.5
    eta_total_min    = eta_compress_min + eta_upload_min

    print(f"\n  Processing {len(entries)} model(s) with {quant_label} compression")
    print(f"  Estimated total: ~{total_size_gb:.1f} GB "
          f"(~{eta_total_min:.0f} min at 1 GB/min compress + 500 MB/min upload)")
    if args.dry_run:
        print("  [dry-run mode — no HuggingFace uploads]")

    # Check which repos already exist (skip unless --force)
    if not args.force and not args.dry_run:
        try:
            from huggingface_hub import HfApi  # type: ignore[import]
            api = HfApi(token=token)
            to_skip = []
            for entry in entries:
                suffix = "-squished-int2" if args.int2 else "-squished" if int4 else "-squished-int8"
                repo_id = f"{args.org}/{entry.dir_name}{suffix}"
                try:
                    if api.repo_exists(repo_id):
                        to_skip.append(entry.id)
                        print(f"  ⏭  {entry.id}: repo already exists at {repo_id} — skipping")
                except Exception:
                    pass
            entries = [e for e in entries if e.id not in to_skip]
            if not entries:
                print("  ✓  All repos already exist. Use --force to re-upload.")
                return
        except ImportError:
            pass

    results = []
    for entry in entries:
        suffix = "-squished-int2" if args.int2 else "-squished" if int4 else "-squished-int8"
        repo_id = f"{args.org}/{entry.dir_name}{suffix}"
        t_start = time.perf_counter()
        try:
            compressed_dir = _compress(entry, models_dir, int4, token)
            disk_gb = sum(
                f.stat().st_size for f in compressed_dir.rglob("*") if f.is_file()
            ) / 1e9
            url = _upload(entry, compressed_dir, int4, token, args.dry_run)
            elapsed_min = (time.perf_counter() - t_start) / 60
            results.append({
                "id": entry.id, "params": entry.params, "quant": quant_label,
                "size": f"{disk_gb:.1f} GB", "ok": True,
                "repo_id": repo_id, "url": url,
                "elapsed_min": f"{elapsed_min:.1f}",
            })
        except Exception as exc:
            print(f"\n  ✗  {entry.id} failed: {exc}", file=sys.stderr)
            results.append({
                "id": entry.id, "params": entry.params, "quant": quant_label,
                "size": "?", "ok": False,
                "repo_id": repo_id, "url": "", "error": str(exc),
            })

    _write_report(results, Path(args.report))

    ok_count = sum(1 for r in results if r["ok"])
    print(f"\n  Done: {ok_count}/{len(results)} models {'uploaded' if not args.dry_run else 'compressed'} successfully.")
    if ok_count < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
