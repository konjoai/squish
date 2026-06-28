"""Full matrix runner — gated behind explicit human approval (--i-have-approved).

Runs every (reuse x context) cell across all systems, writes raw per-run JSON,
per-metric summary tables, plots, the one-screen summary, and the post-flight
verification. Refuses to run without the approval flag, because the kill-test
must be reviewed first (it is expensive and one-way on machine time).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import systems as S
from .cell import CellRunner
from .corpus import Corpus
from .host import MLXTokenizer, detect_ram_bytes
from .matrix_spec import CONTEXT_LENGTHS, REUSE_LEVELS, all_cells, counterbalanced_order
from .report import metric_table, one_screen_summary, postflight, render_plots

OUT_ROOT = Path(__file__).resolve().parents[3] / "results" / "benchmark_matrix"


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - hardware orchestration
    ap = argparse.ArgumentParser(description="Full reuse x context benchmark matrix")
    ap.add_argument(
        "--i-have-approved",
        action="store_true",
        help="Confirm the kill-test was reviewed and approved.",
    )
    ap.add_argument("--n-runs", type=int, default=30, help="paired runs per cell (>=30)")
    args = ap.parse_args(argv)

    if not args.i_have_approved:
        print("REFUSING: run the kill-test first and pass --i-have-approved.")
        print("  python -m benchmarks.ollama_vs_squish.matrix.run_killtest")
        return 2
    if args.n_runs < 30:
        print("REFUSING: the methodology requires >=30 paired runs per cell.")
        return 2

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_ROOT / "matrix" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] output dir: {out_dir}")

    ram = detect_ram_bytes()
    tok = MLXTokenizer(S.SQUISH_MODEL_INT4)
    corpus = Corpus(tok, corpus_dir=Path(__file__).resolve().parent / "corpus_files")
    systems = S.build_systems()
    runner = CellRunner(corpus, out_dir, ram_bytes=ram, n_runs=args.n_runs)

    cells_out: list[dict] = []
    for ci, cell in enumerate(all_cells()):
        order = counterbalanced_order(list(systems.keys()), ci)
        result = runner.run(cell.reuse, cell.ctx_tokens, systems, order)
        cell_json = out_dir / f"{cell.cell_id}.json"
        cell_json.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        cells_out.append(result.to_dict())
        print(f"[matrix] cell {cell.cell_id}: status={result.status}")

    (out_dir / "all_cells.json").write_text(json.dumps(cells_out, indent=2, default=str))
    tables = []
    for metric in ("e2e_s", "decode_tps", "ttft_s"):
        tables.append(metric_table(cells_out, metric, REUSE_LEVELS, CONTEXT_LENGTHS))
    summary = one_screen_summary(cells_out, CONTEXT_LENGTHS)
    checks = postflight(cells_out, min_runs=args.n_runs)
    (out_dir / "summary.md").write_text("\n\n".join(tables) + "\n\n" + summary + "\n\n" + checks)
    for line in render_plots(cells_out, out_dir / "plots", REUSE_LEVELS, CONTEXT_LENGTHS):
        print(f"[matrix] plot: {line}")
    print("\n".join(tables))
    print(summary)
    print(checks)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
