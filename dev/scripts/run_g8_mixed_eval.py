#!/usr/bin/env python3
"""Run 500-sample accuracy eval against the g=8 mixed-precision model.

Mixed precision: FP16 self_attn + INT4 g=8 AWQ MLP.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import squish.squish_lm_eval  # noqa: F401  registers "squish" backend
import lm_eval

MODEL_DIR      = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
COMPRESSED_DIR = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-squished-g8-mixed")
MODEL_ARGS     = f"model_dir={MODEL_DIR},compressed_dir={COMPRESSED_DIR}"
RESULTS_DIR    = ROOT / "dev" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print("  g=8 mixed (FP16 attn + INT4 g=8 AWQ MLP) — 500-sample eval")
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
BF16_REF = {"arc_easy": 0.745, "hellaswag": 0.635, "piqa": 0.775, "winogrande": 0.655}
summary = {}
for task, metrics in results["results"].items():
    acc_norm = metrics.get("acc_norm,none")
    acc      = metrics.get("acc,none")
    se       = metrics.get("acc_norm_stderr,none") or metrics.get("acc_stderr,none")
    val      = acc_norm if acc_norm is not None else acc
    metric   = "acc_norm" if acc_norm is not None else "acc"
    ref      = BF16_REF.get(task, 0.0)
    delta    = val - ref
    flag     = "✓" if delta >= 0 else "✗"
    print(f"  {task:20s}: {val:.4f}  ±{se:.4f}  (ref {ref:.3f}, {delta:+.3f}) {flag}")
    summary[task] = {
        "acc":    round(val, 4),
        "stderr": round(se,  4) if se else None,
        "metric": metric,
        "bf16_ref": ref,
        "delta_vs_bf16": round(delta, 4),
    }

all_beat = all(summary[t]["delta_vs_bf16"] >= 0 for t in BF16_REF)
print(f"\n  {'ALL TASKS BEAT BF16 REFERENCE ✓' if all_beat else 'Some tasks still below BF16 reference'}")

out = {
    "model":          "Qwen2.5-1.5B-g8-FP16attn-INT4g8MLP-AWQ",
    "compressed_dir": COMPRESSED_DIR,
    "limit":          500,
    "num_fewshot":    0,
    "results":        summary,
    "all_beat_bf16":  all_beat,
}
out_path = RESULTS_DIR / "accuracy_g8_mixed_500.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {out_path}")
