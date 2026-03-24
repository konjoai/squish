#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
dev/scripts/resquish_all_models.py — Re-squish INT2/INT3 models with fixed
three-tier mixed-precision recipe that eliminates garbage output (INT2) and
repetition loops (INT3).

Root cause of broken models
----------------------------
The previous recipe quantized ALL linear layers uniformly at 2-bit or 3-bit,
including Q/K/V/O attention projections.  At those bit depths the attention
mechanism has too few levels (4 for INT2) to represent distinct attention
patterns → garbage / looping output.

Fixed recipe
------------
  INT2:  attn=4-bit · ffn=2-bit · embed=8-bit · group_size=32
  INT3:  attn=4-bit · ffn=3-bit · embed=8-bit · group_size=32
  INT4:  attn=4-bit · ffn=4-bit · embed=8-bit · group_size=64  (unchanged)

This script
-----------
  1. Deletes all existing INT2/INT3 model directories (the broken ones).
  2. Re-squishes each INT2/INT3 model from its BF16 counterpart.
  3. Squishes missing Mistral-7B INT2/INT3/INT4 (BF16 already present).
  4. Skips families where the BF16 source is absent.

Usage
-----
  # Dry-run — no changes on disk:
  python3 dev/scripts/resquish_all_models.py --dry-run

  # Execute (interactive confirm before first deletion):
  python3 dev/scripts/resquish_all_models.py

  # Re-squish only specific families:
  python3 dev/scripts/resquish_all_models.py --families Qwen3-0.6B Llama-3.2-1B

  # Skip INT4 (already working) and only fix INT2/INT3:
  python3 dev/scripts/resquish_all_models.py --bits 2 3

Requirements
------------
  pip install mlx-lm
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT   = Path(__file__).resolve().parents[2]
_MODELS_ROOT = Path.home() / "models"

# ── colour codes ─────────────────────────────────────────────────────────────
G  = "\033[32m"; Y  = "\033[33m"; C  = "\033[36m"; W  = "\033[1;37m"
D  = "\033[2m";  NC = "\033[0m";  R  = "\033[31m"; B  = "\033[1m"


# ── model family registry ─────────────────────────────────────────────────────

class ModelFamily(NamedTuple):
    name:         str          # display key, e.g. "Qwen3-0.6B"
    bf16_dir:     str          # relative to _MODELS_ROOT
    int4_dir:     str          # output path, relative to _MODELS_ROOT
    int3_dir:     str
    int2_dir:     str
    hf_id:        str          # HuggingFace model ID (for remote squish if BF16 missing)
    resquish_int4: bool = False  # True only for Mistral where INT4 is also missing


MODEL_FAMILIES: list[ModelFamily] = [
    ModelFamily(
        name="Qwen3-0.6B",
        bf16_dir="Qwen3-0.6B-bf16",
        int4_dir="Qwen3-0.6B-int4",
        int3_dir="Qwen3-0.6B-int3",
        int2_dir="Qwen3-0.6B-int2",
        hf_id="Qwen/Qwen3-0.6B",
    ),
    ModelFamily(
        name="Llama-3.2-1B",
        bf16_dir="Llama-3.2-1B-Instruct-bf16",
        int4_dir="Llama-3.2-1B-Instruct-int4",
        int3_dir="Llama-3.2-1B-Instruct-int3",
        int2_dir="Llama-3.2-1B-Instruct-int2",
        hf_id="meta-llama/Llama-3.2-1B-Instruct",
    ),
    ModelFamily(
        name="gemma-3-1b",
        bf16_dir="gemma-3-1b-it-bf16",
        int4_dir="gemma-3-1b-it-int4",
        int3_dir="gemma-3-1b-it-int3",
        int2_dir="gemma-3-1b-it-int2",
        hf_id="google/gemma-3-1b-it",
    ),
    ModelFamily(
        name="Qwen2.5-1.5B",
        bf16_dir="Qwen2.5-1.5B-Instruct-bf16",
        int4_dir="Qwen2.5-1.5B-Instruct-int4",
        int3_dir="Qwen2.5-1.5B-Instruct-int3",
        int2_dir="Qwen2.5-1.5B-Instruct-int2",
        hf_id="Qwen/Qwen2.5-1.5B-Instruct",
    ),
    ModelFamily(
        name="Llama-3.2-3B",
        bf16_dir="Llama-3.2-3B-Instruct-bf16",
        int4_dir="Llama-3.2-3B-Instruct-int4",
        int3_dir="Llama-3.2-3B-Instruct-int3",
        int2_dir="Llama-3.2-3B-Instruct-int2",
        hf_id="meta-llama/Llama-3.2-3B-Instruct",
    ),
    ModelFamily(
        name="Qwen3-4B",
        bf16_dir="Qwen3-4B-bf16",
        int4_dir="Qwen3-4B-int4",
        int3_dir="Qwen3-4B-int3",
        int2_dir="Qwen3-4B-int2",
        hf_id="Qwen/Qwen3-4B",
    ),
    ModelFamily(
        name="gemma-3-4b",
        bf16_dir="gemma-3-4b-it-bf16",
        int4_dir="gemma-3-4b-it-int4",
        int3_dir="gemma-3-4b-it-int3",
        int2_dir="gemma-3-4b-it-int2",
        hf_id="google/gemma-3-4b-it",
    ),
    ModelFamily(
        # Mistral has no INT2/3/4 yet — all three need squishing
        name="Mistral-7B",
        bf16_dir="Mistral-7B-Instruct-v0.3-bf16",
        int4_dir="Mistral-7B-Instruct-v0.3-int4",
        int3_dir="Mistral-7B-Instruct-v0.3-int3",
        int2_dir="Mistral-7B-Instruct-v0.3-int2",
        hf_id="mistralai/Mistral-7B-Instruct-v0.3",
        resquish_int4=True,  # INT4 also missing for Mistral
    ),
    ModelFamily(
        name="Qwen2.5-7B",
        bf16_dir="Qwen2.5-7B-Instruct-bf16",
        int4_dir="Qwen2.5-7B-Instruct-int4",
        int3_dir="Qwen2.5-7B-Instruct-int3",
        int2_dir="Qwen2.5-7B-Instruct-int2",
        hf_id="Qwen/Qwen2.5-7B-Instruct",
    ),
    ModelFamily(
        name="Qwen3-8B",
        bf16_dir="Qwen3-8B-bf16",
        int4_dir="Qwen3-8B-int4",
        int3_dir="Qwen3-8B-int3",
        int2_dir="Qwen3-8B-int2",
        hf_id="Qwen/Qwen3-8B",
    ),
    ModelFamily(
        name="Qwen3-14B",
        # BF16 is ~28 GB — if absent we download via HF ID
        bf16_dir="Qwen3-14B-bf16",
        int4_dir="Qwen3-14B-int4",
        int3_dir="Qwen3-14B-int3",
        int2_dir="Qwen3-14B-int2",
        hf_id="Qwen/Qwen3-14B",
    ),
]

# ── quantization recipes ──────────────────────────────────────────────────────

# Per-bit recipe: (ffn_bits, attn_bits, embed_bits, group_size)
RECIPES: dict[int, tuple[int, int, int, int]] = {
    2: (2, 4, 8, 32),   # attn=4 fixes garbage output, gs=32 reduces error
    3: (3, 4, 8, 32),   # attn=4 fixes repetition loops
    4: (4, 4, 8, 64),   # attn=4 (same as ffn), standard gs=64
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _gb(path: Path) -> float:
    """Total size of a directory tree in GB (0.0 if path absent)."""
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / 1_073_741_824


def _free_gb() -> float:
    """Available disk space at the models root in GB."""
    stat = shutil.disk_usage(_MODELS_ROOT)
    return stat.free / 1_073_741_824


def _hdr(title: str, sub: str = "") -> None:
    print(f"\n{W}{'─' * 70}{NC}")
    print(f"{C}  {title}{NC}")
    if sub:
        print(f"{D}  {sub}{NC}")
    print(f"{W}{'─' * 70}{NC}")


def _delete_model(path: Path, dry_run: bool) -> float:
    """Delete a model directory.  Returns space freed in GB."""
    freed = _gb(path)
    if dry_run:
        print(f"  {Y}[dry-run]{NC} would delete  {path}  ({freed:.2f} GB)")
    else:
        print(f"  {R}✗ deleting{NC} {path}  ({freed:.2f} GB) …", end="", flush=True)
        shutil.rmtree(path)
        print(f"  {G}done{NC}")
    return freed


def _squish(
    source:      str | Path,
    output_path: Path,
    bits:        int,
    dry_run:     bool,
    cpu:         bool,
) -> bool:
    """Run `squish quantize` for one (source → output) pair.  Returns success."""
    ffn_bits, attn_bits, embed_bits, group_size = RECIPES[bits]

    cli = [
        sys.executable, "-m", "squish.cli", "quantize",
        "--source-path", str(source),
        "--output-path",  str(output_path),
        "--ffn-bits",    str(ffn_bits),
        "--attn-bits",   str(attn_bits),
        "--embed-bits",  str(embed_bits),
        "--group-size",  str(group_size),
    ]
    if cpu:
        cli.append("--cpu")

    if dry_run:
        print(f"  {Y}[dry-run]{NC} would run: {' '.join(cli[2:])}")
        return True

    print(f"  {C}→ squishing{NC}  {source}  →  {output_path.name}")
    print(f"  {D}recipe: ffn={ffn_bits}-bit · attn={attn_bits}-bit · "
          f"embed={embed_bits}-bit · gs={group_size}{NC}")

    t0   = time.time()
    proc = subprocess.run(cli, cwd=str(_REPO_ROOT))
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"  {R}✗ squish quantize failed (exit {proc.returncode}){NC}")
        return False

    size = _gb(output_path)
    print(f"  {G}✓ done{NC}  ({elapsed / 60:.1f} min · {size:.2f} GB)")
    return True


# ── main logic ────────────────────────────────────────────────────────────────

def _process_family(
    fam:      ModelFamily,
    bits_to_fix: list[int],
    models_root: Path,
    dry_run: bool,
    cpu: bool,
) -> dict:
    """Process one model family: delete old quantized dirs, re-squish."""
    report = {"deleted": [], "squished": [], "skipped": [], "errors": []}

    bf16_dir = models_root / fam.bf16_dir

    # Determine the quantization source: prefer local BF16, fall back to HF ID
    if bf16_dir.exists():
        source = bf16_dir
    else:
        source = fam.hf_id  # mlx_lm.convert accepts HF model IDs
        print(f"  {Y}WARN:{NC} BF16 not on disk — will download from HF: {source}")
        if not dry_run:
            free = _free_gb()
            print(f"  {D}Available disk: {free:.1f} GB{NC}")

    for bits in sorted(bits_to_fix):
        dir_key   = {2: "int2_dir", 3: "int3_dir", 4: "int4_dir"}[bits]
        quant_dir = models_root / getattr(fam, dir_key)

        # Delete existing (broken) quantized directory
        if quant_dir.exists():
            freed = _delete_model(quant_dir, dry_run)
            report["deleted"].append(str(quant_dir))
        else:
            print(f"  {D}(dir absent, skip delete){NC}  {quant_dir.name}")

        # Re-squish
        ok = _squish(
            source=source,
            output_path=quant_dir,
            bits=bits,
            dry_run=dry_run,
            cpu=cpu,
        )
        if ok:
            report["squished"].append(str(quant_dir))
        else:
            report["errors"].append(str(quant_dir))

    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--families",
        nargs="+",
        metavar="NAME",
        default=None,
        help=(
            "Model families to process (default: all). "
            f"Available: {', '.join(f.name for f in MODEL_FAMILIES)}"
        ),
    )
    ap.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=[2, 3],
        choices=[2, 3, 4],
        metavar="N",
        help="Quantization levels to fix (default: 2 3). Use '4' only for Mistral.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without changing anything on disk.",
    )
    ap.add_argument(
        "--cpu",
        action="store_true",
        help="Force MLX to CPU (avoids Metal OOM for large models on 16 GB machines).",
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=_MODELS_ROOT,
        help=f"Root directory containing model dirs (default: {_MODELS_ROOT})",
    )
    ap.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    args = ap.parse_args()

    # ── resolve family list ───────────────────────────────────────────────────
    if args.families:
        requested   = set(args.families)
        families    = [f for f in MODEL_FAMILIES if f.name in requested]
        missing_fam = requested - {f.name for f in families}
        if missing_fam:
            print(f"{R}ERROR:{NC} Unknown families: {', '.join(sorted(missing_fam))}")
            sys.exit(1)
    else:
        families = list(MODEL_FAMILIES)

    bits_to_fix = sorted(set(args.bits))

    # ── plan ─────────────────────────────────────────────────────────────────
    _hdr(
        "resquish_all_models — Three-Tier Mixed-Precision Re-Squish",
        f"{len(families)} families · bits={bits_to_fix} · "
        f"{'DRY RUN' if args.dry_run else 'LIVE RUN'}",
    )

    print(f"\n  Recipes:")
    for b in bits_to_fix:
        ffn, attn, embed, gs = RECIPES[b]
        print(f"    INT{b}: ffn={ffn}-bit · attn={attn}-bit · embed={embed}-bit · gs={gs}")

    print(f"\n  Free disk: {_free_gb():.1f} GB\n")
    print(f"  Models:")
    for fam in families:
        bf16_exists = (args.models_root / fam.bf16_dir).exists()
        src_note    = f"BF16 {G}✓{NC}" if bf16_exists else f"BF16 absent — will HF download {Y}⚠{NC}"
        print(f"    {C}{fam.name:<18}{NC}  {src_note}")
        for b in bits_to_fix:
            dir_key   = {2: "int2_dir", 3: "int3_dir", 4: "int4_dir"}[b]
            quant_dir = args.models_root / getattr(fam, dir_key)
            exists    = quant_dir.exists()
            gb_str    = f"({_gb(quant_dir):.2f} GB)" if exists else "(absent)"
            action    = f"{R}DELETE+resquish{NC}" if exists else f"squish new{NC}"
            print(f"      INT{b}: {action}  {quant_dir.name}  {gb_str}")

    print()

    if not args.dry_run and not args.yes:
        resp = input(
            f"  {Y}Proceed? This will DELETE the INT2/INT3 dirs listed above. [y/N]: {NC}"
        ).strip().lower()
        if resp not in ("y", "yes"):
            print("  Aborted.")
            sys.exit(0)

    # ── execute ───────────────────────────────────────────────────────────────
    suite_t0   = time.time()
    all_errors = []

    for fam in families:
        _hdr(f"Processing: {fam.name}", f"source: {fam.bf16_dir}")

        # Determine which bits to handle for this family
        bits_for_this = list(bits_to_fix)
        if 4 in bits_for_this and not fam.resquish_int4:
            # Only re-squish INT4 for families that need it (e.g. Mistral)
            bits_for_this = [b for b in bits_for_this if b != 4]

        report = _process_family(
            fam=fam,
            bits_to_fix=bits_for_this,
            models_root=args.models_root,
            dry_run=args.dry_run,
            cpu=args.cpu,
        )

        if report["errors"]:
            all_errors.extend(report["errors"])
            print(f"  {R}✗ errors: {report['errors']}{NC}")
        else:
            print(f"  {G}✓ done{NC}")

    total = time.time() - suite_t0
    _hdr(
        "Re-squish complete",
        f"Wall time: {total / 60:.1f} min · free disk: {_free_gb():.1f} GB",
    )

    if all_errors:
        print(f"\n{R}Failed models ({len(all_errors)}):{NC}")
        for e in all_errors:
            print(f"  {R}✗{NC} {e}")
        sys.exit(1)
    else:
        print(f"\n{G}All re-squished successfully.{NC}")
        print(f"\nNext step: run the benchmark to verify all models:")
        print(f"  python3 dev/benchmarks/bench_lmeval_all_models.py \\")
        print(f"    --bits 2 3 --skip-existing --gen-sanity --limit 500")


if __name__ == "__main__":
    main()
