#!/usr/bin/env python3
"""Eval for mixed-precision v3 (alpha=0.15, g=16, n=20)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import squish.squish_lm_eval  # noqa: F401
import lm_eval

MODEL_DIR      = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
COMPRESSED_DIR = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-mixed-v3")
MODEL_ARGS     = f"model_dir={MODEL_DIR},compressed_dir={COMPRESSED_DIR}"
RESULTS_DIR    = ROOT / "dev" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BF16_REF = {"arc_easy": 0.750, "hellaswag": 0.612, "piqa": 0.772, "winogrande": 0.630}
V1       = {"arc_easy": 0.746, "hellaswag": 0.606, "piqa": 0.776, "winogrande": 0.648}

print(f"\n{'='*60}")
print("  Mixed v3: FP16 attn + INT4 g=16 AWQ alpha=0.15 — 500 samples")
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
    ref      = BF16_REF[task]
    delta    = val - ref
    flag     = "✓" if delta >= 0 else "✗"
    print(f"  {task:20s}: {val:.4f}  ±{se:.4f}  (bf16={ref:.3f}, {delta:+.3f}) {flag}")
    summary[task] = {
        "acc": round(val, 4), "stderr": round(se, 4) if se else None,
        "metric": metric, "bf16_ref": ref, "delta_vs_bf16": round(delta, 4),
    }

all_beat = all(summary[t]["delta_vs_bf16"] >= 0 for t in BF16_REF)
print(f"\n  {'ALL TASKS BEAT BF16 ✓' if all_beat else 'Some tasks below BF16'}")

print("\n=== vs v1 (alpha=0.10) ===")
for task in BF16_REF:
    v3  = summary[task]["acc"]
    v1v = V1[task]
    delta = v3 - v1v
    print(f"  {task:20s}: v1={v1v:.4f}  v3={v3:.4f}  {delta:+.4f} {'↑' if delta > 0 else ('↓' if delta < 0 else '=')}")

out = {
    "model": "Qwen2.5-1.5B-mixed-v3-FP16attn-INT4g16-AWQa15",
    "compressed_dir": COMPRESSED_DIR,
    "config": {"group_size": 16, "awq_alpha": 0.15, "awq_n_samples": 20, "passthrough": "self_attn"},
    "limit": 500, "num_fewshot": 0, "results": summary, "all_beat_bf16": all_beat,
}
out_path = RESULTS_DIR / "accuracy_mixed_v3_500.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {out_path}")
