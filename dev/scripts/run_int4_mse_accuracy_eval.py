#!/usr/bin/env python3
"""Run 500-sample accuracy eval against the INT4+MSE compressed model."""
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import squish.squish_lm_eval  # noqa: F401  registers "squish" backend
import lm_eval

MODEL_DIR      = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
COMPRESSED_DIR = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-int4-mse")
MODEL_ARGS     = f"model_dir={MODEL_DIR},compressed_dir={COMPRESSED_DIR}"
RESULTS_DIR    = ROOT / "dev" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print("  INT4 asymmetric+MSE — 500-sample accuracy eval")
print(f"  compressed_dir: {COMPRESSED_DIR}")
print(f"{'='*60}\n")

results = lm_eval.simple_evaluate(
    model="squish",
    model_args=MODEL_ARGS,
    tasks=["arc_easy", "hellaswag", "piqa", "winogrande"],
    num_fewshot=0,
    limit=500,
    log_samples=False,
)

print("\n=== Results ===")
summary = {}
for task, metrics in results["results"].items():
    acc_norm = metrics.get("acc_norm,none")
    acc      = metrics.get("acc,none")
    se       = metrics.get("acc_norm_stderr,none") or metrics.get("acc_stderr,none")
    val      = acc_norm if acc_norm is not None else acc
    metric   = "acc_norm" if acc_norm is not None else "acc"
    print(f"  {task:20s}: {val:.4f}  ±{se:.4f}")
    summary[task] = {
        "acc":    round(val, 4),
        "stderr": round(se,  4) if se else None,
        "metric": metric,
    }

out = {
    "model":          "Qwen2.5-1.5B-INT4-asymmetric-MSE",
    "compressed_dir": COMPRESSED_DIR,
    "limit":          500,
    "num_fewshot":    0,
    "results":        summary,
}
out_path = RESULTS_DIR / "accuracy_int4_mse_500.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {out_path}")
