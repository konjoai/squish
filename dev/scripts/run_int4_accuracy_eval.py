#!/usr/bin/env python3
"""Run quick accuracy eval with more samples to confirm INT4 numbers."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import squish.squish_lm_eval   # noqa: F401  registers "squish" backend
import lm_eval

MODEL_DIR       = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
COMPRESSED_DIR  = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16-compressed")
MODEL_ARGS      = f"model_dir={MODEL_DIR},compressed_dir={COMPRESSED_DIR}"
RESULTS_DIR     = ROOT / "dev" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_and_save(tasks, limit, fewshot, out_path, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  tasks={tasks}  limit={limit}  fewshot={fewshot}")
    print(f"{'='*60}")
    results = lm_eval.simple_evaluate(
        model="squish",
        model_args=MODEL_ARGS,
        tasks=tasks,
        num_fewshot=fewshot,
        limit=limit,
        log_samples=False,
    )
    summary = {}
    for task, metrics in results["results"].items():
        acc_norm = metrics.get("acc_norm,none")
        acc      = metrics.get("acc,none")
        stderr   = metrics.get("acc_norm_stderr,none") or metrics.get("acc_stderr,none")
        val      = acc_norm if acc_norm is not None else acc
        metric   = "acc_norm" if acc_norm is not None else "acc"
        summary[task] = {"acc": round(val, 4), "stderr": round(stderr, 4) if stderr else None, "metric": metric}
        print(f"  {task:30s}: {val:.4f}  ±{stderr:.4f}")

    out = {
        "model": "Qwen2.5-1.5B-Instruct-bf16-INT4-group32",
        "hardware": "Apple M3",
        "limit": limit,
        "num_fewshot": fewshot,
        "results": summary,
    }
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(f"\n  Saved → {out_path}")
    return summary


# 500-sample eval for better statistical confidence
run_and_save(
    tasks=["arc_easy", "hellaswag", "piqa", "winogrande"],
    limit=500,
    fewshot=0,
    out_path=RESULTS_DIR / "accuracy_int4_group32_500.json",
    label="INT4 group-32 accuracy eval — 500 samples/task"
)
