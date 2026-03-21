#!/usr/bin/env python3
"""
bench_int3_int2.py — INT3 vs INT2 quantization benchmark for squish.

Measures compression quality (SNR), compression ratio, bits-per-weight (BPW),
and timing for MiLo INT3 and WeightOnlyInt2Quant INT2 across a suite of weight
matrix shapes that match real transformer layer dimensions.

No BF16 model download needed — uses synthetic float32 Gaussian weights at
transformer-realistic shapes.

Usage
-----
    python3 dev/benchmarks/bench_int3_int2.py
    python3 dev/benchmarks/bench_int3_int2.py --output dev/results/
    python3 dev/benchmarks/bench_int3_int2.py --output dev/results/ --markdown

Output
------
  Prints a formatted summary table to stdout.
  Writes JSON to <output>/int3_int2_bench.json if --output is given.
  Writes Markdown to <output>/int3_int2_bench.md if both --output and --markdown.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from squish.quant.milo_quant import MiLoConfig, MiLoQuantizer
from squish.quant.weight_only_int2 import Int2QuantConfig, WeightOnlyInt2Quant

# ── ANSI colours ──────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m"; NC = "\033[0m"; B = "\033[1m"

_RNG = np.random.default_rng(0xDEAD_BEEF)

# ── Layer shapes (rows × cols) representative of real transformer layers ──────
# SmolLM2-135M / 360M  — hs=576/960,  ffn=1536/2560
# Qwen3-0.6B / 1.7B   — hs=1024/2048, ffn=3072/8192
# LLaMA-3.1-8B         — hs=4096,      ffn=14336
LAYER_SHAPES: list[tuple[str, tuple[int, int]]] = [
    ("attn_proj  [576×576]",    (576,   576)),
    ("ffn_up     [576×1536]",   (576,   1536)),
    ("attn_proj  [1024×1024]",  (1024,  1024)),
    ("ffn_up     [1024×3072]",  (1024,  3072)),
    ("attn_proj  [2048×2048]",  (2048,  2048)),
    ("ffn_up     [2048×8192]",  (2048,  8192)),
    ("attn_proj  [4096×4096]",  (4096,  4096)),
    ("ffn_up     [4096×14336]", (4096, 14336)),
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class LayerResult:
    shape:             tuple[int, int]
    method:            str          # "INT3-MiLo" | "INT2-WOQ"
    bpw:               float        # bits per weight
    snr_db:            float        # signal-to-noise ratio of reconstruction
    compress_ratio:    float        # fp32_bytes / compressed_bytes
    compress_ms:       float        # time to quantize (ms)
    decompress_ms:     float        # time to dequantize (ms)
    compressed_mb:     float        # compressed size in MB
    original_mb:       float        # original fp32 size in MB


# ── SNR helper ────────────────────────────────────────────────────────────────

def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    sig = float(np.sum(original.astype(np.float64) ** 2))
    err = float(np.sum((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if err == 0.0:
        return float("inf")
    return 10.0 * np.log10(sig / (err + 1e-30))


# ── INT3 benchmark (MiLo) ─────────────────────────────────────────────────────

def _bench_int3(
    weight: np.ndarray,
    cfg: MiLoConfig,
) -> LayerResult:
    """Compress a single layer with MiLo INT3 and measure all metrics."""
    m, n = weight.shape
    original_bytes = weight.nbytes  # float32 → 4 bytes/element

    q = MiLoQuantizer(cfg)

    # Compression timing
    t0 = time.perf_counter()
    q_packed, scales, zeros, compensator = q.quantize(weight)
    compress_ms = (time.perf_counter() - t0) * 1000.0

    # Compressed size:
    #   q_packed: 3-bit packed uint8  → 3 bytes per 8 weights
    #   scales, zeros: float32 per group → 4 bytes × n_groups × 2
    #   compensator A, B: float32 low-rank factors
    n_weights = m * n
    packed_bytes = q_packed.nbytes
    scale_bytes = scales.nbytes + zeros.nbytes
    comp_bytes = compensator.a.nbytes + compensator.b.nbytes
    compressed_bytes = packed_bytes + scale_bytes + comp_bytes

    bpw = (compressed_bytes * 8) / n_weights
    compress_ratio = original_bytes / compressed_bytes

    # Decompression timing
    t0 = time.perf_counter()
    w_dq = q.dequantize(q_packed, scales, zeros, n_weights, weight.shape)
    w_rec = w_dq + compensator.a @ compensator.b
    decompress_ms = (time.perf_counter() - t0) * 1000.0

    snr = _snr_db(weight, w_rec)

    return LayerResult(
        shape=weight.shape,
        method="INT3-MiLo",
        bpw=bpw,
        snr_db=snr,
        compress_ratio=compress_ratio,
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        compressed_mb=compressed_bytes / 1e6,
        original_mb=original_bytes / 1e6,
    )


# ── INT2 benchmark (WeightOnlyInt2Quant) ──────────────────────────────────────

def _bench_int2(
    weight: np.ndarray,
    cfg: Int2QuantConfig,
) -> LayerResult:
    """Compress a single layer with WeightOnlyInt2Quant INT2 and measure all metrics."""
    m, n = weight.shape
    original_bytes = weight.nbytes  # float32

    q = WeightOnlyInt2Quant(cfg)

    # Compression timing
    t0 = time.perf_counter()
    packed, scale, zero = q.quantize(weight)
    compress_ms = (time.perf_counter() - t0) * 1000.0

    # Compressed size:
    #   packed: pack-4 → 2-bit per weight = 4 weights/byte
    #   scale, zero: float32 per group
    n_weights = m * n
    compressed_bytes = packed.nbytes + scale.nbytes + zero.nbytes
    bpw = (compressed_bytes * 8) / n_weights
    compress_ratio = original_bytes / compressed_bytes

    # Decompression timing
    t0 = time.perf_counter()
    w_rec = q.dequantize(packed, scale, zero)
    decompress_ms = (time.perf_counter() - t0) * 1000.0

    snr = _snr_db(weight, w_rec)

    return LayerResult(
        shape=weight.shape,
        method="INT2-WOQ",
        bpw=bpw,
        snr_db=snr,
        compress_ratio=compress_ratio,
        compress_ms=compress_ms,
        decompress_ms=decompress_ms,
        compressed_mb=compressed_bytes / 1e6,
        original_mb=original_bytes / 1e6,
    )


# ── Projection helper ─────────────────────────────────────────────────────────

def _project_model(
    results: list[LayerResult],
    n_layers: int,
    attn_layers_per_block: int = 4,
    ffn_layers_per_block: int = 2,
) -> dict:
    """Project total model memory savings from a single-layer sample.

    For a rough projection, we assume alternating attn/ffn shapes and multiply
    by the number of transformer blocks.  This is approximate but realistic.

    Parameters
    ----------
    results:
        All per-layer results for a given method.
    n_layers:
        Number of transformer blocks in the target model.
    """
    total_orig = sum(r.original_mb for r in results)
    total_comp = sum(r.compressed_mb for r in results)
    if total_orig == 0:
        return {}
    ratio = total_orig / total_comp
    projected_orig_gb = (total_orig * n_layers) / 1024.0
    projected_comp_gb = (total_comp * n_layers) / 1024.0
    return {
        "n_layers": n_layers,
        "projected_original_gb": round(projected_orig_gb, 2),
        "projected_compressed_gb": round(projected_comp_gb, 2),
        "memory_savings_pct": round((1.0 - total_comp / total_orig) * 100.0, 1),
        "overall_compress_ratio": round(ratio, 2),
    }


# ── Pretty printing ───────────────────────────────────────────────────────────

_COL = [28, 12, 8, 10, 10, 12, 14, 14]

def _header() -> str:
    hdrs = ["Layer", "Method", "BPW", "SNR (dB)", "CR", "Compress ms", "Decomp ms", "Size MB"]
    return "  ".join(h.ljust(_COL[i]) for i, h in enumerate(hdrs))


def _row(label: str, r: LayerResult) -> str:
    cols = [
        label,
        r.method,
        f"{r.bpw:.2f}",
        f"{r.snr_db:.1f}",
        f"{r.compress_ratio:.2f}×",
        f"{r.compress_ms:.1f}",
        f"{r.decompress_ms:.1f}",
        f"{r.compressed_mb:.2f}",
    ]
    return "  ".join(c.ljust(_COL[i]) for i, c in enumerate(cols))


def _md_table(rows: list[tuple[str, LayerResult]]) -> str:
    hdrs = ["Layer", "Method", "BPW", "SNR (dB)", "Compress ratio", "Compress (ms)", "Decomp (ms)", "Size (MB)"]
    sep = ["-" * 28, "-" * 10, "-" * 6, "-" * 9, "-" * 14, "-" * 13, "-" * 11, "-" * 10]
    lines: list[str] = []
    lines.append("| " + " | ".join(hdrs) + " |")
    lines.append("| " + " | ".join(sep) + " |")
    for label, r in rows:
        cols = [
            label,
            r.method,
            f"{r.bpw:.2f}",
            f"{r.snr_db:.1f}",
            f"{r.compress_ratio:.2f}×",
            f"{r.compress_ms:.1f}",
            f"{r.decompress_ms:.1f}",
            f"{r.compressed_mb:.2f}",
        ]
        lines.append("| " + " | ".join(cols) + " |")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_benchmark(
    shapes: list[tuple[str, tuple[int, int]]],
    int3_cfg: MiLoConfig,
    int2_cfg: Int2QuantConfig,
    verbose: bool = True,
) -> list[tuple[str, LayerResult]]:
    rows: list[tuple[str, LayerResult]] = []

    if verbose:
        print(f"\n{B}{C}squish INT3 vs INT2 Quantization Benchmark{NC}")
        print(f"{D}{'─' * 110}{NC}")
        print(f"{B}{_header()}{NC}")
        print(f"{D}{'─' * 110}{NC}")

    for label, (m, n) in shapes:
        # Synthetic Gaussian weight — matches distribution of real transformer weights
        weight = _RNG.standard_normal((m, n)).astype(np.float32) * 0.02

        r3 = _bench_int3(weight, int3_cfg)
        r2 = _bench_int2(weight, int2_cfg)

        rows.append((label, r3))
        rows.append((label, r2))

        if verbose:
            print(_row(label, r3))
            print(_row("", r2))
            print(f"{D}{'─' * 110}{NC}")

    return rows


def _averages(rows: list[tuple[str, LayerResult]], method: str) -> dict:
    subset = [r for _, r in rows if r.method == method]
    if not subset:
        return {}
    return {
        "method": method,
        "avg_bpw": round(sum(r.bpw for r in subset) / len(subset), 3),
        "avg_snr_db": round(sum(r.snr_db for r in subset) / len(subset), 2),
        "avg_compress_ratio": round(sum(r.compress_ratio for r in subset) / len(subset), 2),
        "avg_compress_ms": round(sum(r.compress_ms for r in subset) / len(subset), 2),
        "avg_decompress_ms": round(sum(r.decompress_ms for r in subset) / len(subset), 2),
        "n_layers": len(subset),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", metavar="DIR", help="Directory for JSON/Markdown output.")
    parser.add_argument("--markdown", action="store_true", help="Also write a Markdown report.")
    parser.add_argument("--int3-group", type=int, default=64, metavar="N", help="INT3 group size (default 64).")
    parser.add_argument("--int3-max-rank", type=int, default=8, metavar="R", help="INT3 MiLo max compensator rank (default 8).")
    parser.add_argument("--int2-group", type=int, default=64, metavar="N", help="INT2 group size (default 64).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-row output.")
    args = parser.parse_args(argv)

    int3_cfg = MiLoConfig(
        target_bits=3,
        max_rank=args.int3_max_rank,
        min_rank=2,
        group_size=args.int3_group,
        snr_threshold_db=35.0,
        adaptive_rank=True,
    )
    int2_cfg = Int2QuantConfig(
        group_size=args.int2_group,
        symmetric=False,
        clip_threshold=0.99,
    )

    rows = run_benchmark(LAYER_SHAPES, int3_cfg, int2_cfg, verbose=not args.quiet)

    int3_avg = _averages(rows, "INT3-MiLo")
    int2_avg = _averages(rows, "INT2-WOQ")

    # Model-scale projections (using Qwen3-1.7B as representative: 28 layers)
    int3_proj = _project_model([r for _, r in rows if r.method == "INT3-MiLo"], n_layers=28)
    int2_proj = _project_model([r for _, r in rows if r.method == "INT2-WOQ"], n_layers=28)

    if not args.quiet:
        print(f"\n{B}Averages across all {len(LAYER_SHAPES)} shapes:{NC}")
        print(f"  INT3-MiLo  — BPW: {int3_avg['avg_bpw']:.2f}  "
              f"SNR: {int3_avg['avg_snr_db']:.1f} dB  "
              f"CR: {int3_avg['avg_compress_ratio']:.2f}×  "
              f"Compress: {int3_avg['avg_compress_ms']:.0f} ms  "
              f"Decomp: {int3_avg['avg_decompress_ms']:.1f} ms")
        print(f"  INT2-WOQ   — BPW: {int2_avg['avg_bpw']:.2f}  "
              f"SNR: {int2_avg['avg_snr_db']:.1f} dB  "
              f"CR: {int2_avg['avg_compress_ratio']:.2f}×  "
              f"Compress: {int2_avg['avg_compress_ms']:.0f} ms  "
              f"Decomp: {int2_avg['avg_decompress_ms']:.1f} ms")

        print(f"\n{B}Projected model memory (28-layer Qwen3-1.7B scale, FP32 baseline):{NC}")
        print(f"  INT3-MiLo  — {int3_proj['projected_compressed_gb']:.2f} GB vs "
              f"{int3_proj['projected_original_gb']:.2f} GB FP32  "
              f"({int3_proj['memory_savings_pct']:.0f}% savings)")
        print(f"  INT2-WOQ   — {int2_proj['projected_compressed_gb']:.2f} GB vs "
              f"{int2_proj['projected_original_gb']:.2f} GB FP32  "
              f"({int2_proj['memory_savings_pct']:.0f}% savings)")

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON output
        payload = {
            "int3_config": {"target_bits": 3, "max_rank": args.int3_max_rank, "group_size": args.int3_group},
            "int2_config": {"bits": 2, "group_size": args.int2_group, "symmetric": False},
            "results": [
                {
                    "label": label,
                    "shape": list(r.shape),
                    **{k: v for k, v in asdict(r).items() if k != "shape"},
                }
                for label, r in rows
            ],
            "averages": {"INT3-MiLo": int3_avg, "INT2-WOQ": int2_avg},
            "model_projection_28layers": {"INT3-MiLo": int3_proj, "INT2-WOQ": int2_proj},
        }
        json_path = out_dir / "int3_int2_bench.json"
        json_path.write_text(json.dumps(payload, indent=2))
        print(f"\n{G}JSON results written → {json_path}{NC}")

        if args.markdown:
            table = _md_table(rows)
            md = (
                "# INT3 vs INT2 Quantization Benchmark\n\n"
                "Synthetic Gaussian weights (σ=0.02) at transformer-realistic shapes.  \n"
                f"INT3: MiLo (group_size={args.int3_group}, max_rank={args.int3_max_rank})  \n"
                f"INT2: WeightOnlyInt2Quant (group_size={args.int2_group}, asymmetric)  \n\n"
                "## Per-Layer Results\n\n"
                + table
                + "\n\n## Averages\n\n"
                f"| Method     | BPW   | SNR (dB) | Compress ratio | Compress (ms) | Decomp (ms) |\n"
                f"| ---------- | ----- | -------- | -------------- | ------------- | ----------- |\n"
                f"| INT3-MiLo  | {int3_avg['avg_bpw']:.2f} | {int3_avg['avg_snr_db']:.1f} "
                f"| {int3_avg['avg_compress_ratio']:.2f}× "
                f"| {int3_avg['avg_compress_ms']:.0f} | {int3_avg['avg_decompress_ms']:.1f} |\n"
                f"| INT2-WOQ   | {int2_avg['avg_bpw']:.2f} | {int2_avg['avg_snr_db']:.1f} "
                f"| {int2_avg['avg_compress_ratio']:.2f}× "
                f"| {int2_avg['avg_compress_ms']:.0f} | {int2_avg['avg_decompress_ms']:.1f} |\n\n"
                "## Model Projection (28-layer Qwen3-1.7B scale, FP32 baseline)\n\n"
                "| Method     | Compressed (GB) | Original FP32 (GB) | Memory Savings |\n"
                "| ---------- | --------------- | ------------------ | -------------- |\n"
                f"| INT3-MiLo  | {int3_proj['projected_compressed_gb']:.2f} "
                f"| {int3_proj['projected_original_gb']:.2f} "
                f"| {int3_proj['memory_savings_pct']:.0f}% |\n"
                f"| INT2-WOQ   | {int2_proj['projected_compressed_gb']:.2f} "
                f"| {int2_proj['projected_original_gb']:.2f} "
                f"| {int2_proj['memory_savings_pct']:.0f}% |\n"
            )
            md_path = out_dir / "int3_int2_bench.md"
            md_path.write_text(md)
            print(f"{G}Markdown results written → {md_path}{NC}")


if __name__ == "__main__":
    main()
