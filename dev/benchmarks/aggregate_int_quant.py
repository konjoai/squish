#!/usr/bin/env python3
"""
aggregate_int_quant.py — Aggregate all INT4/INT3/INT2 benchmark results into
a single Markdown report and combined JSON file.

Reads all JSON files matching dev/results/int_quant/*_<bits>bit.json,
computes deltas vs the BF16 baseline (loaded from benchmark_multi_model.json
or inferred), and writes:

  docs/benchmark_int_quant.md   — human-readable tables (throughput, PPL, accuracy)
  dev/results/int_quant/combined.json — all raw results merged

Usage
-----
  python3 dev/benchmarks/aggregate_int_quant.py
  python3 dev/benchmarks/aggregate_int_quant.py --results-dir dev/results/int_quant
  python3 dev/benchmarks/aggregate_int_quant.py --output docs/benchmark_int_quant.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── colour helpers ────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

# ── BF16 reference data (from benchmark_multi_model.json + known model sizes) ─
# Fallback reference when no BF16 result JSON exists.
# Format: model_id → {tps_bf16, ppl_bf16, arc_easy_bf16, hellaswag_bf16, bf16_gb}
_BF16_REFERENCE: dict[str, dict[str, Any]] = {
    "Qwen2.5-1.5B":            {"tps_bf16": 24.2, "ppl_bf16": None, "arc_easy_bf16": 0.715, "hellaswag_bf16": 0.560, "bf16_gb": 3.1},
    "Qwen2.5-7B":              {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 14.0},
    "Qwen2.5-14B":             {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 29.6},
    "Qwen3-8B":                {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 16.4},
    "Llama-3.2-3B":            {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 6.4},
    "Llama-3.1-8B":            {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 16.0},
    "Mistral-7B-v0.3":         {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 14.5},
    "Phi-4":                   {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 29.4},
    "Gemma-3-4B":              {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 8.6},
    "gemma-3-4b-it":           {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 10.0},
    "DeepSeek-R1-Distill-7B":  {"tps_bf16": None, "ppl_bf16": None, "arc_easy_bf16": None,  "hellaswag_bf16": None,  "bf16_gb": 15.2},
}

# Approximate compressed sizes by bits (based on measured compression ratios)
_SIZE_RATIOS = {4: 0.28, 3: 0.21, 2: 0.14}

# Bit-level display labels
_BIT_LABELS = {16: "BF16", 4: "INT4", 3: "INT3", 2: "INT2 ⚠"}


def _fmt_f(v: float | None, fmt: str = ".1f", fallback: str = "—") -> str:
    if v is None:
        return fallback
    return format(v, fmt)


def _delta_s(base: float | None, val: float | None, higher_better: bool = True) -> str:
    """Return a coloured delta string (Markdown-friendly plain text here)."""
    if base is None or val is None:
        return "—"
    d = val - base
    if higher_better:
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"
    else:
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"


# ── loading ───────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> dict[str, dict[int, dict]]:
    """
    Load all *_<N>bit.json files from results_dir.
    Returns {model_id: {bits: raw_dict}}.
    """
    data: dict[str, dict[int, dict]] = {}
    for path in sorted(results_dir.glob("*_?bit.json")):
        stem = path.stem  # e.g. "Qwen2.5-1.5B_4bit"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].endswith("bit"):
            continue
        model_id = parts[0]
        try:
            bits = int(parts[1].replace("bit", ""))
        except ValueError:
            continue
        try:
            raw = json.loads(path.read_text())
        except Exception as e:
            print(f"{Y}  ⚠ could not load {path.name}: {e}{NC}", file=sys.stderr)
            continue
        data.setdefault(model_id, {})[bits] = raw
    return data


# ── table builders ────────────────────────────────────────────────────────────

def build_compression_table(data: dict[str, dict[int, dict]]) -> str:
    lines = [
        "## Compression Metrics",
        "",
        "| Model | Bits | Method | BF16 GB | Compressed GB | Size Ratio | bpw | Compress time |",
        "|-------|-----:|--------|--------:|--------------:|----------:|----:|:-------------:|",
    ]
    # Model order: existing squish models first, then new models
    model_order = list(_BF16_REFERENCE.keys())
    extra_models = [m for m in data if m not in model_order]
    for model_id in model_order + extra_models:
        bits_map = data.get(model_id, {})
        ref = _BF16_REFERENCE.get(model_id, {})
        bf16_gb = ref.get("bf16_gb", "—")

        if not bits_map:
            # Model not yet benchmarked — show expected sizes from reference
            for bits in (4, 3, 2):
                ratio  = _SIZE_RATIOS[bits]
                est_gb = f"~{bf16_gb * ratio:.2f}" if isinstance(bf16_gb, float) else "—"
                lines.append(
                    f"| {model_id} | {bits} | — | "
                    f"{'—' if not isinstance(bf16_gb, float) else f'{bf16_gb:.1f}'} | "
                    f"{est_gb} | ~{ratio:.0%} | — | — |"
                )
            continue

        for bits in sorted(bits_map.keys(), reverse=True):
            r  = bits_map[bits]
            c  = r.get("compression", {})
            note = " ⚠" if bits == 2 else ""
            method_labels = {4: "INT4 nibble", 3: "MiLo INT3", 2: "AQLM INT2"}
            method = method_labels.get(bits, "?")

            if c.get("error"):
                lines.append(f"| {model_id} | {bits}{note} | {method} | — | — | — | — | ✗ {c['error'][:30]} |")
                continue

            orig_gb = _fmt_f(c.get("original_gb"), ".1f")
            comp_gb = _fmt_f(c.get("compressed_gb"), ".2f")
            ratio   = _fmt_f(c.get("size_ratio", 0) * 100, ".1f") + "%"
            bpw     = _fmt_f(c.get("bpw_approx"), ".2f")
            csec    = _fmt_f(c.get("compress_s"), ".0f") + "s" if c.get("compress_s") else "—"
            lines.append(
                f"| {model_id} | {bits}{note} | {method} | {orig_gb} | {comp_gb} | {ratio} | {bpw} | {csec} |"
            )

    return "\n".join(lines)


def build_throughput_table(data: dict[str, dict[int, dict]]) -> str:
    lines = [
        "## Throughput (T1 — tok/s)",
        "",
        "| Model | BF16 tok/s | INT4 tok/s | Δ INT4 | INT3 tok/s | Δ INT3 | INT2 tok/s | Δ INT2 |",
        "|-------|:----------:|:----------:|:------:|:----------:|:------:|:----------:|:------:|",
    ]
    model_order = list(_BF16_REFERENCE.keys())
    extra_models = [m for m in data if m not in model_order]
    for model_id in model_order + extra_models:
        bits_map = data.get(model_id, {})
        ref = _BF16_REFERENCE.get(model_id, {})
        bf16_tps = ref.get("tps_bf16")
        bf16_s = _fmt_f(bf16_tps, ".1f")

        def _tps(bits: int) -> float | None:
            r = bits_map.get(bits, {})
            t = r.get("throughput", {})
            if t.get("error"):
                return None
            return t.get("tps_mean")

        i4 = _tps(4); i3 = _tps(3); i2 = _tps(2)
        lines.append(
            f"| {model_id} | {bf16_s} "
            f"| {_fmt_f(i4, '.1f')} | {_delta_s(bf16_tps, i4)} "
            f"| {_fmt_f(i3, '.1f')} | {_delta_s(bf16_tps, i3)} "
            f"| {_fmt_f(i2, '.1f')} | {_delta_s(bf16_tps, i2)} |"
        )
    return "\n".join(lines)


def build_ppl_table(data: dict[str, dict[int, dict]]) -> str:
    lines = [
        "## Perplexity (T2 — wikitext-2, lower = better)",
        "",
        "| Model | BF16 PPL | INT4 PPL | Δ INT4 | INT3 PPL | Δ INT3 | INT2 PPL | Δ INT2 |",
        "|-------|:--------:|:--------:|:------:|:--------:|:------:|:--------:|:------:|",
    ]
    model_order = list(_BF16_REFERENCE.keys())
    extra_models = [m for m in data if m not in model_order]
    for model_id in model_order + extra_models:
        bits_map = data.get(model_id, {})
        ref = _BF16_REFERENCE.get(model_id, {})
        bf16_ppl = ref.get("ppl_bf16")
        bf16_s = _fmt_f(bf16_ppl, ".1f")

        def _ppl(bits: int) -> float | None:
            r = bits_map.get(bits, {})
            p = r.get("perplexity", {})
            if not p or p.get("error"):
                return None
            return p.get("ppl")

        i4 = _ppl(4); i3 = _ppl(3); i2 = _ppl(2)
        lines.append(
            f"| {model_id} | {bf16_s} "
            f"| {_fmt_f(i4, '.2f')} | {_delta_s(bf16_ppl, i4, higher_better=False)} "
            f"| {_fmt_f(i3, '.2f')} | {_delta_s(bf16_ppl, i3, higher_better=False)} "
            f"| {_fmt_f(i2, '.2f')} | {_delta_s(bf16_ppl, i2, higher_better=False)} |"
        )
    return "\n".join(lines)


def build_accuracy_table(data: dict[str, dict[int, dict]]) -> str:
    lines = [
        "## Accuracy (T3 — 0-shot, 200 samples)",
        "",
        "### ARC-Easy (acc_norm)",
        "",
        "| Model | BF16 | INT4 | Δ INT4 | INT3 | Δ INT3 |",
        "|-------|:----:|:----:|:------:|:----:|:------:|",
    ]
    model_order = list(_BF16_REFERENCE.keys())
    extra_models = [m for m in data if m not in model_order]
    for model_id in model_order + extra_models:
        bits_map = data.get(model_id, {})
        ref = _BF16_REFERENCE.get(model_id, {})
        bf16 = ref.get("arc_easy_bf16")
        bf16_s = f"{bf16:.1%}" if bf16 is not None else "—"

        def _arc(bits: int) -> float | None:
            r = bits_map.get(bits, {})
            a = r.get("accuracy", {})
            if not a or a.get("error"):
                return None
            return a.get("arc_easy")

        i4 = _arc(4); i3 = _arc(3)
        def _pct(v): return f"{v:.1%}" if v is not None else "—"
        def _d(base, val):
            if base is None or val is None:
                return "—"
            d = val - base
            return f"{d:+.1%}"
        lines.append(f"| {model_id} | {bf16_s} | {_pct(i4)} | {_d(bf16, i4)} | {_pct(i3)} | {_d(bf16, i3)} |")

    lines += [
        "",
        "### HellaSwag (acc_norm)",
        "",
        "| Model | BF16 | INT4 | Δ INT4 | INT3 | Δ INT3 |",
        "|-------|:----:|:----:|:------:|:----:|:------:|",
    ]
    for model_id in model_order + extra_models:
        bits_map = data.get(model_id, {})
        ref = _BF16_REFERENCE.get(model_id, {})
        bf16 = ref.get("hellaswag_bf16")
        bf16_s = f"{bf16:.1%}" if bf16 is not None else "—"

        def _hell(bits: int) -> float | None:
            r = bits_map.get(bits, {})
            a = r.get("accuracy", {})
            if not a or a.get("error"):
                return None
            return a.get("hellaswag")

        i4 = _hell(4); i3 = _hell(3)
        lines.append(f"| {model_id} | {bf16_s} | {_pct(i4)} | {_d(bf16, i4)} | {_pct(i3)} | {_d(bf16, i3)} |")

    return "\n".join(lines)


def build_size_reference_table() -> str:
    lines = [
        "## Model Size Reference",
        "",
        "| Model | Family | Params | BF16 disk | INT4 (~28%) | INT3 (~21%) | INT2 (~14%) |",
        "|-------|--------|-------:|----------:|------------:|------------:|------------:|",
    ]
    models = [
        ("Qwen2.5-1.5B",           "Qwen",      "1.5B",  3.1),
        ("Qwen2.5-7B",             "Qwen",      "7.2B",  14.0),
        ("Qwen2.5-14B",            "Qwen",      "14.2B", 29.6),
        ("Qwen3-8B",               "Qwen",      "8.2B",  16.4),
        ("Llama-3.2-3B",           "Llama",     "3.2B",  6.4),
        ("Llama-3.1-8B",           "Llama",     "8.0B",  16.0),
        ("Mistral-7B-v0.3",        "Mistral",   "7.25B", 14.5),
        ("Phi-4",                  "Phi",       "14.7B", 29.4),
        ("Gemma-3-4B",             "Gemma",     "4.3B",  8.6),
        ("DeepSeek-R1-Distill-7B", "DeepSeek",  "7.6B",  15.2),
    ]
    total_bf16 = sum(m[3] for m in models)
    for name, family, params, bf16 in models:
        i4 = bf16 * 0.28; i3 = bf16 * 0.21; i2 = bf16 * 0.14
        lines.append(
            f"| {name} | {family} | {params} | {bf16:.1f} GB | {i4:.2f} GB | {i3:.2f} GB | {i2:.2f} GB |"
        )
    # Totals row
    t4 = total_bf16 * 0.28; t3 = total_bf16 * 0.21; t2 = total_bf16 * 0.14
    lines.append(
        f"| **TOTAL** | — | — | **{total_bf16:.1f} GB** | "
        f"**{t4:.1f} GB** | **{t3:.1f} GB** | **{t2:.1f} GB** |"
    )
    return "\n".join(lines)


def build_status_table(data: dict[str, dict[int, dict]]) -> str:
    """Show which model × bit combinations have been run."""
    lines = [
        "## Benchmark Status",
        "",
        "| Model | INT4 T1 | INT4 T2 | INT4 T3 | INT3 T1 | INT3 T2 | INT3 T3 | INT2 T1 | INT2 T2 | INT2 T3 |",
        "|-------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|",
    ]
    model_order = list(_BF16_REFERENCE.keys())
    extra_models = [m for m in data if m not in model_order]

    def _status(model_id: str, bits: int) -> tuple[str, str, str]:
        bits_map = data.get(model_id, {})
        r = bits_map.get(bits, {})
        def _ok(key: str) -> str:
            sub = r.get(key, {})
            if not sub:
                return "✗"
            if sub.get("error"):
                return "⚠"
            # check there's actual data
            if key == "throughput" and sub.get("tps_mean", 0) > 0:
                return "✓"
            if key == "perplexity" and sub.get("ppl", 0) > 0:
                return "✓"
            if key == "accuracy" and (sub.get("arc_easy") or sub.get("hellaswag")):
                return "✓"
            return "✗"
        return _ok("throughput"), _ok("perplexity"), _ok("accuracy")

    for model_id in model_order + extra_models:
        t4a, t4b, t4c = _status(model_id, 4)
        t3a, t3b, t3c = _status(model_id, 3)
        t2a, t2b, t2c = _status(model_id, 2)
        lines.append(
            f"| {model_id} | {t4a} | {t4b} | {t4c} | {t3a} | {t3b} | {t3c} | {t2a} | {t2b} | {t2c} |"
        )
    lines += ["", "> ✓ = complete  ⚠ = ran with error  ✗ = not yet run"]
    return "\n".join(lines)


# ── full document ─────────────────────────────────────────────────────────────

def to_markdown(data: dict[str, dict[int, dict]]) -> str:
    n_complete = sum(
        1
        for bits_map in data.values()
        for r in bits_map.values()
        if r.get("compression") and not r["compression"].get("error")
    )
    total_runs = len(data) * 3  # 10 models × 3 bit levels

    sections = [
        "# Squish — INT4 / INT3 / INT2 Quantization Benchmark",
        "",
        f"> Last updated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"> Runs complete: {n_complete} / {total_runs} model×bit combinations",
        ">",
        "> **Test battery:** 3 tests per model per bit level (T1 throughput · T2 perplexity · T3 accuracy)",
        "> **Hardware:** Apple Silicon M-series (MLX backend)",
        "> **INT2 note:** 2-bit is included as a floor reference only — expect catastrophic quality loss.",
        "",
        "---",
        "",
        build_status_table(data),
        "",
        "---",
        "",
        build_size_reference_table(),
        "",
        "---",
        "",
        build_compression_table(data),
        "",
        "---",
        "",
        build_throughput_table(data),
        "",
        "---",
        "",
        build_ppl_table(data),
        "",
        "---",
        "",
        build_accuracy_table(data),
        "",
        "---",
        "",
        "## Methodology",
        "",
        "| Test | Tool | Config |",
        "|------|------|--------|",
        "| **T1 Throughput** | mlx_lm.stream_generate | 3 prompts × 3 runs × 128 max tokens |",
        "| **T2 Perplexity** | mlx token NLL | wikitext-2-raw-v1, 512 tokens, stride 512 |",
        "| **T3 Accuracy** | lm-eval harness | ARC-Easy + HellaSwag, 0-shot, 200 samples |",
        "",
        "**Compression methods:**",
        "",
        "| Level | Method | bpw | squish flag |",
        "|-------|--------|----:|-------------|",
        "| INT4 | Nibble-packed asymmetric INT4, group-32 | ~5.0 | `squish-convert --int4 --super-weight` |",
        "| INT3 | MiLo INT3 + low-rank compensator, group-128 | ~3.75 | Python API: `MiLoQuantizer` |",
        "| INT2 | AQLM 2-codebook additive VQ, group-8 | ~2.0 | Python API: `AQLMQuantizer` |",
        "",
        "BF16 reference data for existing squish models sourced from `dev/results/benchmark_multi_model.json`.",
        "New models (Llama, Mistral, Phi-4, Gemma, DeepSeek) have no prior squish benchmarks.",
        "",
        "Raw result JSON: `dev/results/int_quant/`",
        "Benchmark script: `dev/benchmarks/bench_int_quant.py`",
        "Run all models: `dev/scripts/run_all_int_quant.sh`",
    ]
    return "\n".join(sections) + "\n"


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate INT4/INT3/INT2 benchmark JSONs into a Markdown report.",
    )
    ap.add_argument(
        "--results-dir", default="dev/results/int_quant",
        help="Directory containing *_<N>bit.json result files (default: dev/results/int_quant/).",
    )
    ap.add_argument(
        "--output", default="docs/benchmark_int_quant.md",
        help="Markdown output path (default: docs/benchmark_int_quant.md).",
    )
    ap.add_argument(
        "--json-output", default="dev/results/int_quant/combined.json",
        help="Combined JSON output path.",
    )
    ap.add_argument(
        "--status-only", action="store_true",
        help="Print completion status table to stdout and exit without writing files.",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = _REPO_ROOT / results_dir

    print(f"\n{B}{C}  Squish INT Quant — Aggregating results{NC}")
    print(f"{D}  results dir: {results_dir}{NC}")

    data = load_results(results_dir)

    total_models = len(_BF16_REFERENCE)
    found_models = len(data)
    found_runs   = sum(len(bm) for bm in data.values())
    print(f"  Found {found_models}/{total_models} models · {found_runs} bit-level results")

    if args.status_only:
        # Just print the status table
        print()
        print(build_status_table(data))
        return

    # ── write combined JSON ───────────────────────────────────────────────────
    json_out = Path(args.json_output)
    if not json_out.is_absolute():
        json_out = _REPO_ROOT / json_out
    json_out.parent.mkdir(parents=True, exist_ok=True)

    combined = {}
    for model_id, bits_map in data.items():
        combined[model_id] = {}
        for bits, raw in bits_map.items():
            combined[model_id][str(bits)] = raw
    json_out.write_text(json.dumps(combined, indent=2))
    print(f"  {G}✓{NC} combined JSON → {json_out.relative_to(_REPO_ROOT)}")

    # ── write markdown ────────────────────────────────────────────────────────
    md     = to_markdown(data)
    md_out = Path(args.output)
    if not md_out.is_absolute():
        md_out = _REPO_ROOT / md_out
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(md)
    print(f"  {G}✓{NC} markdown report → {md_out.relative_to(_REPO_ROOT)}")

    # ── print quick status to stdout ──────────────────────────────────────────
    print()
    print(build_status_table(data))
    print(f"\n  {G}✓ Done.{NC}")


if __name__ == "__main__":
    main()
