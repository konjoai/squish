#!/usr/bin/env python3
"""
bench_lmeval_qwen3_14b.py — Full lm-evaluation-harness suite for Qwen3-14B.

On Apple Silicon M3 16 GB the 29 GB BF16 model will not fit in RAM.
This script uses mlx_lm's built-in lm-eval integration
(`python -m mlx_lm evaluate`) which runs the MLX INT4 model (~7.8 GB)
natively on Metal GPU under lm-evaluation-harness.

Tasks (industry-standard)
--------------------------
  arc_easy        ARC Easy,       25-shot
  arc_challenge   ARC Challenge,  25-shot
  hellaswag       HellaSwag,      10-shot
  winogrande      Winogrande,      5-shot
  piqa            PIQA,            0-shot
  openbookqa      OpenBookQA,      0-shot
  mmlu            MMLU,            5-shot
  truthfulqa_mc2  TruthfulQA MC2,  0-shot
  gsm8k           GSM8K,           5-shot

Usage
-----
  # Full suite (hours):
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py

  # Smoke test — 100 samples/task (minutes):
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py --limit 100

  # Selected tasks:
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py --tasks arc_easy hellaswag

Requirements
------------
  pip install lm-eval mlx-lm
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ── colour codes ─────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR  = Path.home() / "models" / "Qwen3-14B-mlx-int4"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results"

# (task_name, primary_lmeval_metric, standard_fewshots)
TASKS: list[tuple[str, str, int]] = [
    ("arc_easy",       "acc_norm,none",              25),
    ("arc_challenge",  "acc_norm,none",              25),
    ("hellaswag",      "acc_norm,none",              10),
    ("winogrande",     "acc,none",                    5),
    ("piqa",           "acc_norm,none",               0),
    ("openbookqa",     "acc_norm,none",               0),
    ("mmlu",           "acc,none",                    5),
    ("truthfulqa_mc2", "acc,none",                    0),
    ("gsm8k",          "exact_match,strict-match",    5),
]
_TASK_METRIC  = {t: m for t, m, _ in TASKS}
_TASK_FEWSHOT = {t: f for t, _, f in TASKS}


# ── helpers ───────────────────────────────────────────────────────────────────

def _hdr(title: str, sub: str = "") -> None:
    print(f"\n{W}{'─' * 72}{NC}")
    print(f"{C}  {title}{NC}")
    if sub:
        print(f"{D}  {sub}{NC}")
    print(f"{W}{'─' * 72}{NC}")


def _ok(label: str, val: str, extra: str = "") -> None:
    print(f"  {G}✓{NC}  {label:<52} {G}{val:>10}{NC}  {D}{extra}{NC}")


def _err(label: str, reason: str) -> None:
    print(f"  {R}✗{NC}  {label:<52} {D}{reason}{NC}")


def _platform_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "ram_gb": 16,
        "python": sys.version.split()[0],
        "mlx_lm": _mlx_lm_version(),
        "lm_eval": _lm_eval_version(),
    }


def _mlx_lm_version() -> str:
    try:
        import mlx_lm
        return mlx_lm.__version__
    except Exception:
        return "unknown"


def _lm_eval_version() -> str:
    try:
        import lm_eval
        return lm_eval.__version__
    except Exception:
        return "unknown"


def _extract_metric(task_result: dict, metric_key: str) -> float | None:
    """Extract primary metric value from a lm_eval 0.4.x task result dict."""
    primary = metric_key.split(",")[0]
    for k, v in task_result.items():
        if primary in k and isinstance(v, (int, float)):
            if "stderr" not in k and "std" not in k:
                return float(v)
    return None


# ── evaluation via mlx_lm evaluate ───────────────────────────────────────────

def _run_single_task(
    task: str,
    model_dir: Path,
    limit: int | None,
    num_fewshot_override: int | None,
    lmeval_out_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    """
    Run a single task in its own `python -m mlx_lm evaluate` subprocess.
    Running one task at a time releases GPU/Metal memory between tasks,
    preventing the kIOGPUCommandBufferCallbackErrorOutOfMemory crash that
    occurs when long fewshot prompts (e.g. MMLU 5-shot) exhaust RAM
    while the previous tasks' weights still occupy the Metal heap.
    """
    fewshot = num_fewshot_override if num_fewshot_override is not None else _TASK_FEWSHOT.get(task, 0)

    cmd = [
        sys.executable, "-m", "mlx_lm", "evaluate",
        "--model", str(model_dir),
        "--tasks", task,
        "--num-shots", str(fewshot),
        "--output-dir", str(lmeval_out_dir),
        "--batch-size", str(batch_size),
        "--trust-remote-code",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"  {D}{' '.join(cmd)}{NC}\n")

    t0 = time.time()
    proc = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {"error": f"exit code {proc.returncode}", "_elapsed_s": elapsed}

    # mlx_lm evaluate writes files named eval_* (no .json extension)
    all_files = sorted(
        (p for p in lmeval_out_dir.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    eval_files = [p for p in all_files if p.name.startswith("eval_")]
    candidates = eval_files if eval_files else all_files
    if not candidates:
        return {"error": "no output file written", "_elapsed_s": elapsed}

    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}", "_elapsed_s": elapsed}
    data["_elapsed_s"] = elapsed
    data["_raw_output_file"] = str(latest)
    return data


def run_mlx_lmeval(
    model_dir: Path,
    tasks: list[str],
    limit: int | None,
    num_fewshot_override: int | None,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    """
    Run each task sequentially in its own subprocess so Metal GPU memory is
    fully released between tasks.  Aggregates individual results into a single
    flat dict keyed by task name (same format _display_results expects).
    """
    lmeval_out_dir = output_dir / "_mlx_lmeval_raw"
    lmeval_out_dir.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, Any] = {}
    total_elapsed = 0.0
    errors: dict[str, str] = {}

    for i, task in enumerate(tasks, 1):
        fewshot = num_fewshot_override if num_fewshot_override is not None else _TASK_FEWSHOT.get(task, 0)
        print(f"\n  [{i}/{len(tasks)}] {W}{task}{NC}  ({fewshot}-shot"
              + (f", limit={limit}" if limit else "") + ")")

        result = _run_single_task(
            task=task,
            model_dir=model_dir,
            limit=limit,
            num_fewshot_override=num_fewshot_override,
            lmeval_out_dir=lmeval_out_dir,
            batch_size=batch_size,
        )

        elapsed = result.get("_elapsed_s", 0)
        total_elapsed += elapsed

        if "error" in result:
            errors[task] = result["error"]
            print(f"  {R}✗{NC}  {task}  {D}FAILED: {result['error']}{NC}")
            continue

        # The file contains {task_name: {metrics...}} — merge into aggregate
        task_data = {k: v for k, v in result.items() if not k.startswith("_")}
        aggregate.update(task_data)
        elapsed_min = elapsed / 60
        print(f"  {G}✓{NC}  {task}  done in {elapsed_min:.1f} min")

    aggregate["_elapsed_s"] = total_elapsed
    if errors:
        aggregate["_errors"] = errors
    return aggregate


# ── display & save ────────────────────────────────────────────────────────────

def _display_results(raw: dict, tasks: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    # mlx_lm evaluate writes a flat dict  {task_name: {...metrics...}}
    # lm_eval 0.4.x standard format wraps it under "results".
    # Support both.
    if "results" in raw and isinstance(raw["results"], dict):
        task_results: dict = raw["results"]
    else:
        task_results = {
            k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    if not task_results:
        _err("results", raw.get("error", "no results key in output"))
        return scores

    _hdr("lm-eval Results")

    for task in tasks:
        primary_metric = _TASK_METRIC.get(task, "acc,none")
        if task not in task_results:
            _err(task, "not in results (task may alias differently in output)")
            continue

        tr = task_results[task]
        score = _extract_metric(tr, primary_metric)
        if score is None:
            _err(task, f"metric {primary_metric!r} not found; keys: {list(tr.keys())[:5]}")
            continue

        pct = score * 100
        stderr_key = primary_metric.split(",")[0] + "_stderr,none"
        stderr = tr.get(stderr_key)
        extra = f"+-{stderr*100:.2f}%" if isinstance(stderr, float) else primary_metric
        _ok(task, f"{pct:.2f}%", extra)
        scores[task] = round(pct, 4)

    return scores


def _save_results(
    scores: dict[str, float],
    raw: dict,
    model_dir: Path,
    output_dir: Path,
    limit: int | None,
    platform_info: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_file = output_dir / f"lmeval_qwen3_14b_{ts}.json"
    if "results" in raw and isinstance(raw["results"], dict):
        raw_results = raw["results"]
    else:
        raw_results = {k: v for k, v in raw.items() if not k.startswith("_") and isinstance(v, dict)}
    payload = {
        "model": model_dir.name,
        "model_path": str(model_dir),
        "timestamp": ts,
        "limit": limit,
        "platform": platform_info,
        "scores": scores,
        "raw_results": raw_results,
        "elapsed_s": raw.get("_elapsed_s"),
    }
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    return out_file


def _print_markdown(
    scores: dict[str, float],
    model_name: str,
    platform_info: dict,
    limit: int | None,
) -> None:
    _hdr("Markdown Summary")
    limit_note = f" (limit={limit} samples/task)" if limit else " (full dataset)"
    print(f"\n## Qwen3-14B lm-eval Benchmark{limit_note}")
    print(
        f"\n*Model: `{model_name}` · "
        f"Platform: Apple M3 · {platform_info.get('ram_gb', 16)} GB RAM · "
        f"mlx-lm {platform_info.get('mlx_lm', '?')} · "
        f"lm-eval {platform_info.get('lm_eval', '?')}*"
    )
    print("\n| Task | Fewshot | Score |")
    print("|------|---------|-------|")
    for task in sorted(scores):
        fs = _TASK_FEWSHOT.get(task, 0)
        print(f"| {task} | {fs} | {scores[task]:.2f}% |")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"| **Average** | - | **{avg:.2f}%** |")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full lm-eval benchmark suite for Qwen3-14B on Apple M3 16 GB (MLX INT4)."
    )
    ap.add_argument(
        "--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
        help=f"Path to MLX INT4 model (default: {DEFAULT_MODEL_DIR})",
    )
    ap.add_argument(
        "--tasks", nargs="+", default=[t for t, _, _ in TASKS],
        help="Task names (default: all 9 standard tasks)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="Max samples per task for a fast subset (e.g. 100). Omit for full eval.",
    )
    ap.add_argument(
        "--num-fewshot", type=int, default=None, dest="num_fewshot",
        help="Override few-shot count for all tasks (default: standard per-task settings)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for inference (default: 1)",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON results (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--markdown", action="store_true",
        help="Print markdown summary table at the end",
    )
    args = ap.parse_args()

    if not args.model_dir.exists():
        print(f"{R}ERROR:{NC} Model directory not found: {args.model_dir}")
        sys.exit(1)

    pinfo = _platform_info()

    _hdr(
        "Qwen3-14B — Full lm-eval Benchmark Suite (MLX INT4 / M3 16 GB)",
        f"Model: {args.model_dir.name}  ·  Tasks: {len(args.tasks)}  ·  "
        f"Limit: {args.limit or 'full'}  ·  mlx-lm {pinfo['mlx_lm']}  ·  lm-eval {pinfo['lm_eval']}",
    )

    raw = run_mlx_lmeval(
        model_dir=args.model_dir,
        tasks=args.tasks,
        limit=args.limit,
        num_fewshot_override=args.num_fewshot,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )

    # Report any per-task failures (non-fatal — other tasks may still have results)
    if "_errors" in raw:
        _hdr("Task Failures")
        for task, err in raw["_errors"].items():
            _err(task, err)

    scores = _display_results(raw, args.tasks)

    if not scores:
        print(f"\n{R}ERROR:{NC} No tasks produced results.")
        sys.exit(1)

    out_file = _save_results(
        scores=scores,
        raw=raw,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        platform_info=pinfo,
    )
    print(f"\n  {G}.{NC}  Results saved -> {D}{out_file}{NC}")
    print(f"  {D}Total eval time: {raw.get('_elapsed_s', 0)/60:.1f} min{NC}")

    if args.markdown:
        _print_markdown(scores, args.model_dir.name, pinfo, args.limit)


if __name__ == "__main__":
    main()
