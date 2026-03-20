#!/usr/bin/env python3
"""
rebuild_benchmark_summary.py — Rebuild BENCHMARK_SUMMARY.md from individual
per-model bench .md files in the given results directory.

Usage:
    python3 scripts/rebuild_benchmark_summary.py results/benchmarks/<timestamp>/
"""
import re
import sys
from pathlib import Path


def extract_averages(md_file: Path) -> tuple[str, str]:
    """Return (avg_ttft_ms, avg_tps) from a squish bench markdown file."""
    text = md_file.read_text(errors="replace")
    avg_match = re.search(r"\|\s*\*\*Average\*\*\s*\|\s*\*\*([0-9]+)\*\*\s*\|\s*—\s*\|\s*\*\*([0-9.]+)\*\*", text)
    if avg_match:
        return avg_match.group(1), avg_match.group(2)
    return "?", "?"


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/rebuild_benchmark_summary.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.is_dir():
        print(f"ERROR: not a directory: {results_dir}")
        sys.exit(1)

    # Collect all per-model bench files (exclude BENCHMARK_SUMMARY.md)
    bench_files = sorted(
        f for f in results_dir.glob("*.md")
        if f.name != "BENCHMARK_SUMMARY.md"
    )

    if not bench_files:
        print("No per-model .md files found.")
        sys.exit(0)

    # Infer hardware info (may not always be parseable — just hardcode known values)
    hardware_line = "Apple M3 · 17 GB Unified RAM · MLX Metal backend"

    # Build the summary
    lines = [
        "# Squish — Full Model Benchmark Results",
        "",
        f"Platform: {hardware_line}  ",
        "Benchmark: 4 prompts × 256 max tokens · streams measured via OpenAI-compat API",
        "",
        "| Model | Avg TTFT (ms) | Avg Tok/s | Status |",
        "|-------|-------------:|----------:|--------|",
    ]

    for f in bench_files:
        model_name = f.stem  # filename without .md
        avg_ttft, avg_tps = extract_averages(f)
        status = "OK" if avg_ttft != "?" else "FAIL (no avg row)"
        lines.append(f"| `{model_name}` | {avg_ttft} | {avg_tps} | {status} |")

    lines += [
        "",
        "---",
        f"*Regenerated from individual per-model files in `{results_dir}`.*",
    ]

    summary_file = results_dir / "BENCHMARK_SUMMARY.md"
    summary_file.write_text("\n".join(lines) + "\n")
    print(f"Summary written to: {summary_file}")

    # Print to stdout as well
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
