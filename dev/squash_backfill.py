#!/usr/bin/env python3
"""dev/squash_backfill.py — Retroactive Squash Phase 4 sidecar backfill.

Generates ``cyclonedx-mlbom.json`` sidecars for every model dir under
``--models-root`` that contains weight files but no existing sidecar, then
calls ``EvalBinder.bind()`` for any model whose bench result already exists
in ``--results-dir``.

This is a dev utility, not a squish module.  Lives in ``dev/`` so it does not
consume module quota (max 100 Python files under ``squish/``).

Usage
-----
    python3 dev/squash_backfill.py [--models-root ~/models] [--results-dir results]
                                    [--no-overwrite] [--dry-run]

Exit codes
----------
    0  All processable models succeeded (skips are not failures).
    1  Argument / path error.
    2  One or more models failed to produce a sidecar.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bench name <-> model dir name mapping
# ---------------------------------------------------------------------------
# Keys:   short names used in results/lmeval_<name>_*.json filenames.
# Values: actual directory names under ~/models/.
# ---------------------------------------------------------------------------
_BENCH_TO_DIR: dict[str, str] = {
    "Qwen3-0.6B-int4":          "Qwen3-0.6B-int4",
    "Qwen3-0.6B-int3":          "Qwen3-0.6B-int3",
    "Qwen3-0.6B-int2":          "Qwen3-0.6B-int2",
    "Qwen3-4B-int4":            "Qwen3-4B-int4",
    "Qwen3-4B-int3":            "Qwen3-4B-int3",
    "Qwen3-4B-int2":            "Qwen3-4B-int2",
    "Qwen3-8B-int4":            "Qwen3-8B-int4",
    "Qwen3-8B-int3":            "Qwen3-8B-int3",
    "Qwen3-8B-int2":            "Qwen3-8B-int2",
    "Llama-3.2-1B-int4":        "Llama-3.2-1B-Instruct-int4",
    "Llama-3.2-1B-int3":        "Llama-3.2-1B-Instruct-int3",
    "Llama-3.2-1B-int2":        "Llama-3.2-1B-Instruct-int2",
    "Llama-3.2-3B-int4":        "Llama-3.2-3B-Instruct-int4",
    "Llama-3.2-3B-int3":        "Llama-3.2-3B-Instruct-int3",
    "Llama-3.2-3B-int2":        "Llama-3.2-3B-Instruct-int2",
    "gemma-3-1b-int4":          "gemma-3-1b-it-int4",
    "gemma-3-1b-int3":          "gemma-3-1b-it-int3",
    "gemma-3-1b-int2":          "gemma-3-1b-it-int2",
    "gemma-3-4b-int4":          "gemma-3-4b-it-int4",
    "gemma-3-4b-int3":          "gemma-3-4b-it-int3",
    "gemma-3-4b-int2":          "gemma-3-4b-it-int2",
    "Qwen2.5-1.5B-int4":        "Qwen2.5-1.5B-Instruct-int4",
    "Qwen2.5-1.5B-int3":        "Qwen2.5-1.5B-Instruct-int3",
    "Qwen2.5-1.5B-int2":        "Qwen2.5-1.5B-Instruct-int2",
    "Qwen2.5-7B-int4":          "Qwen2.5-7B-Instruct-int4",
    "Qwen2.5-7B-int3":          "Qwen2.5-7B-Instruct-int3",
    "Qwen2.5-7B-int2":          "Qwen2.5-7B-Instruct-int2",
}

_DIR_TO_BENCH: dict[str, str] = {v: k for k, v in _BENCH_TO_DIR.items()}

_WEIGHT_SUFFIXES: frozenset[str] = frozenset({".safetensors", ".npy", ".gguf", ".npz"})

# AWQ alpha per model family (mirrors squish/quant/awq.py default).
_FAMILY_AWQ_ALPHA: dict[str, float] = {
    "qwen3": 0.07,
}
_DEFAULT_AWQ_ALPHA = 0.10

_QUANT_DEFAULTS: dict[str, tuple[str, int | None]] = {
    # (quant_format, awq_group_size)
    "int4": ("INT4", 64),
    "int3": ("INT3", 32),
    "int2": ("INT2", 32),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_weight_files(model_dir: Path) -> bool:
    return any(p.suffix in _WEIGHT_SUFFIXES for p in model_dir.rglob("*") if p.is_file())


def _most_recent_result(bench_name: str, results_dir: Path) -> Path | None:
    """Return the most recently modified complete lmeval JSON for bench_name."""
    candidates = sorted(
        results_dir.glob(f"lmeval_{bench_name}_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    for p in reversed(candidates):
        try:
            data = json.loads(p.read_text())
            scores = data.get("scores", {})
            errors = data.get("errors", {})
            if len(scores) >= 6 and len(errors) == 0:
                return p
        except Exception:
            continue
    return None


def _baseline_result(bench_name: str, results_dir: Path) -> Path | None:
    """Return the INT4 baseline result for INT3/INT2 models, else None."""
    if not re.search(r"-int[23]$", bench_name, re.IGNORECASE):
        return None
    baseline_name = re.sub(r"-int[23]$", "-int4", bench_name, flags=re.IGNORECASE)
    return _most_recent_result(baseline_name, results_dir)


def _infer_meta(bench_name: str, model_dir: Path):
    """Build a CompressRunMeta from the model dir alone (no live compress run)."""
    from squish.squash.sbom_builder import CompressRunMeta  # noqa: PLC0415

    suffix_match = re.search(r"-(int[234])$", bench_name, re.IGNORECASE)
    quant_key = suffix_match.group(1).lower() if suffix_match else "int4"
    quant_format, awq_group_size = _QUANT_DEFAULTS.get(quant_key, ("INT4", 64))

    # Detect model family from config.json architecture field.
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
        arch = (cfg.get("model_type") or "").lower()
    except Exception:
        arch = ""

    family: str | None = arch if arch else None

    # AWQ alpha: Qwen3 uses 0.07, everything else 0.10.
    awq_alpha = _FAMILY_AWQ_ALPHA.get(arch, _DEFAULT_AWQ_ALPHA)

    # hf_mlx_repo: best-effort from config.json _name_or_path or id field.
    try:
        hf_repo = cfg.get("_name_or_path") or cfg.get("id") or bench_name
    except Exception:
        hf_repo = bench_name

    return CompressRunMeta(
        model_id=bench_name,
        hf_mlx_repo=hf_repo,
        model_family=family,
        quant_format=quant_format,
        awq_alpha=awq_alpha,
        awq_group_size=awq_group_size,
        output_dir=model_dir,
    )


def process_one(
    bench_name: str,
    model_dir: Path,
    results_dir: Path,
    *,
    no_overwrite: bool,
    dry_run: bool,
) -> str:
    """Process one model. Returns a status tag: OK / SKIP / FAIL / BOUND.

    This function is importable for unit tests.
    """
    from squish.squash.sbom_builder import CycloneDXBuilder, EvalBinder  # noqa: PLC0415

    if not model_dir.exists():
        return "MISSING"

    bom_path = model_dir / "cyclonedx-mlbom.json"

    # Build sidecar if needed.
    if bom_path.exists() and no_overwrite:
        sidecar_written = False  # already exists; skip but may still bind
    else:
        if not _has_weight_files(model_dir):
            return "SKIP-NOWEIGHTS"
        if dry_run:
            print(f"[DRY-RUN] WOULD write sidecar: {bom_path}")
        else:
            try:
                meta = _infer_meta(bench_name, model_dir)
                CycloneDXBuilder.from_compress_run(meta)
                sidecar_written = True
            except Exception as exc:
                print(f"  ERROR building sidecar for {bench_name}: {exc}", file=sys.stderr)
                return "FAIL"

    # Bind scores if a complete result exists.
    result_json = _most_recent_result(bench_name, results_dir)
    if result_json is None:
        if dry_run:
            print(f"[DRY-RUN] SKIP bind {bench_name} — no complete result JSON")
        return "SKIP-NORESULT"

    baseline_json = _baseline_result(bench_name, results_dir)

    if dry_run:
        print(f"[DRY-RUN] WOULD bind {result_json.name} → {bom_path.name}"
              + (f" (baseline: {baseline_json.name})" if baseline_json else ""))
        return "WOULD"

    try:
        EvalBinder.bind(bom_path, result_json, baseline_json)
    except Exception as exc:
        print(f"  ERROR binding scores for {bench_name}: {exc}", file=sys.stderr)
        return "FAIL-BIND"

    return "OK"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Retroactively generate CycloneDX sidecars and bind lmeval scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example:\n  python3 dev/squash_backfill.py --models-root ~/models --results-dir results",
    )
    p.add_argument(
        "--models-root",
        default="~/models",
        help="Root directory containing model dirs (default: ~/models)",
    )
    p.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing lmeval_*.json files (default: results)",
    )
    p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip model dirs that already have cyclonedx-mlbom.json",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen; write nothing",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress informational output (for scripting)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    models_root = Path(args.models_root).expanduser()
    results_dir = Path(args.results_dir).expanduser()

    if not models_root.is_dir():
        print(f"ERROR: --models-root {models_root} does not exist", file=sys.stderr)
        return 1
    if not results_dir.is_dir():
        print(f"ERROR: --results-dir {results_dir} does not exist", file=sys.stderr)
        return 1

    counts: dict[str, int] = {
        "written": 0,
        "skipped_exists": 0,
        "bound": 0,
        "no_result": 0,
        "missing_dir": 0,
        "failed": 0,
    }

    for bench_name, dir_name in sorted(_BENCH_TO_DIR.items()):
        model_dir = models_root / dir_name
        status = process_one(
            bench_name,
            model_dir,
            results_dir,
            no_overwrite=args.no_overwrite,
            dry_run=args.dry_run,
        )

        if not args.quiet:
            tag = f"[{status}]"
            print(f"{tag:<18} {bench_name}")

        if status == "MISSING":
            counts["missing_dir"] += 1
        elif status in ("SKIP-NOWEIGHTS", "SKIP-NORESULT"):
            if "NORESULT" in status:
                counts["no_result"] += 1
            else:
                counts["missing_dir"] += 1
        elif status.startswith("FAIL"):
            counts["failed"] += 1
        elif status == "WOULD":
            counts["written"] += 1
            counts["bound"] += 1
        elif status == "OK":
            counts["written"] += 1
            counts["bound"] += 1
        elif "NORESULT" in status:
            counts["written"] += 1
            counts["no_result"] += 1

    if not args.quiet:
        print()
        print(f"Sidecars written  : {counts['written']}")
        print(f"Scores bound      : {counts['bound']}")
        print(f"No result JSON    : {counts['no_result']}")
        print(f"Dir missing       : {counts['missing_dir']}")
        if counts["failed"]:
            print(f"FAILED            : {counts['failed']}", file=sys.stderr)

    return 2 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
