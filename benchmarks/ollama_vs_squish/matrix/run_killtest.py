"""Kill-test: run ONE cell (8k @ 50% reuse, Squish INT4 vs Ollama Q4_K_M), then STOP.

The sprint mandates this gate before the expensive, one-way full matrix. It runs
the single cell end to end and verifies: measured cache-hit % ~= 50% on Squish
and is correctly characterised on Ollama; the thermal baseline held; >=30 paired
runs produced a Wilcoxon p and an effect size; OOM handling worked. It then
prints the cell and a WAIT banner — it does NOT proceed to the matrix.

    BENCH_SQUISH_INT4=~/models/Qwen2.5-7B-Instruct-int4 \\
    BENCH_OLLAMA_MODEL=qwen2.5:7b \\
    ~/squish/.venv/bin/python -m benchmarks.ollama_vs_squish.matrix.run_killtest
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import systems as S
from .cell import CellRunner
from .corpus import Corpus
from .host import MLXTokenizer, detect_ram_bytes
from .matrix_spec import counterbalanced_order, kill_test_cell
from .report import one_screen_summary, postflight

OUT_ROOT = Path(__file__).resolve().parents[3] / "results" / "benchmark_matrix"


def main() -> int:  # pragma: no cover - hardware orchestration
    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = OUT_ROOT / "killtest" / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[killtest] output dir: {out_dir}")

    ram = detect_ram_bytes()
    tok = MLXTokenizer(S.SQUISH_MODEL_INT4)
    corpus = Corpus(tok, corpus_dir=Path(__file__).resolve().parent / "corpus_files")
    systems = S.build_systems()
    # Kill-test is strictly the head-to-head pair.
    systems = {k: v for k, v in systems.items() if v.role == "head_to_head"}
    order = counterbalanced_order(list(systems.keys()), cell_index=0)

    cell = kill_test_cell()
    runner = CellRunner(corpus, out_dir, ram_bytes=ram, n_runs=30)
    result = runner.run(cell.reuse, cell.ctx_tokens, systems, order)

    raw = out_dir / "killtest_raw.json"
    raw.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"[killtest] wrote {raw}")

    cells = [result.to_dict()]
    print()
    print(one_screen_summary(cells, ctx_lengths=[cell.ctx_tokens]))
    print()
    print(postflight(cells, min_runs=30))
    print()
    print("=" * 72)
    print("KILL-TEST COMPLETE — review the single cell above.")
    print("DO NOT run the full matrix until these checks are green AND a human")
    print("has approved. The matrix is expensive and one-way on machine time.")
    print("To proceed after approval:")
    print("  python -m benchmarks.ollama_vs_squish.matrix.run_matrix --i-have-approved")
    print("=" * 72)
    return 0 if result.status in ("ok",) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
