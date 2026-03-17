#!/usr/bin/env python3
"""Establish a fresh BF16 reference baseline using the squish inference path.

This runs the original BF16 model through exactly the same:
  - squish SquishCompressedLM backend
  - loglikelihood batched forward pass
  - lm-eval tasks, num_fewshot=0, limit=500

By setting compressed_dir = model_dir (the BF16 directory), the squish
compressed_loader detects no manifest.json and falls through to Tier 0a:
    mlx_lm.load(bf16_dir) → squish loglikelihood path

This gives the true apples-to-apples reference for all INT4 comparisons.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import squish.squish_lm_eval  # noqa: F401
import lm_eval

MODEL_DIR      = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
# Point compressed_dir at the BF16 directory itself — triggers native mlx_lm.load()
COMPRESSED_DIR = MODEL_DIR
MODEL_ARGS     = f"model_dir={MODEL_DIR},compressed_dir={COMPRESSED_DIR}"
RESULTS_DIR    = ROOT / "dev" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print("  BF16 reference — squish inference path — 500-sample eval")
print(f"  model_dir: {MODEL_DIR}")
print(f"  (compressed_dir = model_dir → native mlx_lm.load path)")
print(f"{'='*60}\n")

results = lm_eval.simple_evaluate(
    model="squish",
    model_args=MODEL_ARGS,
    tasks=["arc_easy", "hellaswag", "piqa", "winogrande"],
    num_fewshot=0,
    limit=500,
    log_samples=False,
)

print("\n=== BF16 Reference Results (squish path) ===")
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
    "model":       "Qwen2.5-1.5B-Instruct-bf16",
    "loader":      "squish-native-mlx_lm (Tier0a)",
    "description": "BF16 reference measured through squish loglikelihood path",
    "limit":       500,
    "num_fewshot": 0,
    "results":     summary,
}
out_path = RESULTS_DIR / "accuracy_bf16_reference_squish_path_500.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {out_path}")
print("\n  Use these numbers as the true reference for INT4 comparisons.")
