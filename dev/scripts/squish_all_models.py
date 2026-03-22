#!/usr/bin/env python3
"""
Compress all BF16 source models to INT4, INT3, and INT2 using `squish quantize`.

Runs from smallest to largest model to fail fast on small models and catch
issues before committing hours of compute to the large ones.

Usage:
  python3 dev/scripts/squish_all_models.py                # all models, all bits
  python3 dev/scripts/squish_all_models.py --bits 4       # INT4 only
  python3 dev/scripts/squish_all_models.py --models Qwen3-0.6B-bf16  # one model
  python3 dev/scripts/squish_all_models.py --dry-run      # print plan only
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

MODELS_DIR = Path.home() / "models"

# (name, approx_size_gb) — sorted smallest → largest
SOURCE_MODELS: list[tuple[str, float]] = [
    ("Qwen3-0.6B-bf16",               1.1),
    ("Llama-3.2-1B-Instruct-bf16",    2.3),
    ("gemma-3-1b-it-bf16",            2.5),
    ("Qwen2.5-1.5B-Instruct-bf16",    2.9),
    ("Llama-3.2-3B-Instruct-bf16",    6.0),
    ("Qwen3-4B-bf16",                 7.5),
    ("gemma-3-4b-it-bf16",            9.3),
    ("Mistral-7B-Instruct-v0.3-bf16", 8.8),
    ("Qwen2.5-7B-Instruct-bf16",      14.0),
    ("Qwen3-8B-bf16",                 15.0),
    ("Qwen3-14B-bf16",                28.0),
]

# FFN bits → (embed bits, output suffix)
BIT_CONFIGS: list[tuple[int, int, str]] = [
    (4, 8, "int4"),
    (3, 8, "int3"),
    (2, 8, "int2"),
]

def _model_base(name: str) -> str:
    """Strip -bf16 suffix to get the base model name."""
    return name.removesuffix("-bf16")

def _output_name(name: str, suffix: str) -> str:
    return f"{_model_base(name)}-{suffix}"

def _compress_one(
    source_path: Path,
    output_path: Path,
    ffn_bits: int,
    embed_bits: int,
    dry_run: bool,
) -> bool:
    """Run squish quantize for one model/bitwidth. Returns True on success."""
    if output_path.exists():
        print(f"  [skip] {output_path.name} already exists")
        return True

    cmd = [
        "squish", "quantize",
        "--source-path", str(source_path),
        "--output-path", str(output_path),
        "--ffn-bits", str(ffn_bits),
        "--embed-bits", str(embed_bits),
    ]
    if dry_run:
        cmd.append("--dry-run")

    print(f"  {'[dry-run] ' if dry_run else ''}squish quantize "
          f"{source_path.name} → {output_path.name} "
          f"(ffn={ffn_bits}b embed={embed_bits}b)")

    if dry_run:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0

    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"  [ERROR] {output_path.name} failed (exit {result.returncode})")
        # Clean up partial output
        if output_path.exists():
            shutil.rmtree(output_path)
        return False

    size_gb = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file()) / 1e9
    print(f"  [done] {output_path.name}  {size_gb:.1f} GB  ({elapsed/60:.1f} min)")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bits", type=int, nargs="+", choices=[2, 3, 4],
                        help="Bit widths to compress (default: 4 3 2)")
    parser.add_argument("--models", nargs="+",
                        help="Source model names to process (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without running compression")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip models that already have an output dir (default: on)")
    args = parser.parse_args()

    bits_to_run = set(args.bits) if args.bits else {4, 3, 2}
    configs = [(f, e, s) for f, e, s in BIT_CONFIGS if f in bits_to_run]

    models = SOURCE_MODELS
    if args.models:
        requested = set(args.models)
        models = [(n, s) for n, s in models if n in requested]
        if not models:
            print(f"No matching models. Available:\n  " +
                  "\n  ".join(n for n, _ in SOURCE_MODELS))
            sys.exit(1)

    total = len(models) * len(configs)
    done = 0
    failed: list[str] = []

    print(f"\nSquish compression plan: {len(models)} models × {len(configs)} bitwidths = {total} jobs")
    print(f"Bits: {[c[2] for c in configs]}")
    print(f"Order: smallest → largest\n")

    for model_name, approx_gb in models:
        source_path = MODELS_DIR / model_name
        if not source_path.exists():
            print(f"[SKIP] {model_name} — not found in {MODELS_DIR}")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {model_name}  (~{approx_gb:.1f} GB source)")
        print(f"{'='*60}")

        for ffn_bits, embed_bits, suffix in configs:
            output_path = MODELS_DIR / _output_name(model_name, suffix)
            done += 1
            print(f"\n[{done}/{total}]  {suffix.upper()}")
            ok = _compress_one(source_path, output_path, ffn_bits, embed_bits, args.dry_run)
            if not ok:
                failed.append(f"{model_name} → {suffix}")

    print(f"\n{'='*60}")
    print(f"Compression complete: {total - len(failed)}/{total} succeeded")
    if failed:
        print("Failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All done.")


if __name__ == "__main__":
    main()
