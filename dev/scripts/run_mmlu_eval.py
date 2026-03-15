#!/usr/bin/env python3
"""
Run MMLU accuracy evaluation for squish compressed model.
Standard 4-task eval already done (accuracy_v9_standard.json).
This script runs only the full MMLU (14,042 samples, 5-shot).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import squish.squish_lm_eval   # noqa: F401  registers "squish" backend
import lm_eval

MODEL_DIR       = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16")
COMPRESSED_DIR  = str(Path.home() / "models" / "Qwen2.5-1.5B-Instruct-bf16-compressed")
# batch_size=16 for faster batched inference
MODEL_ARGS      = f"model_dir={MODEL_DIR},compressed_dir={COMPRESSED_DIR},batch_size=16"
RESULTS_DIR     = ROOT / "dev" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


print("\n" + "="*60)
print("  MMLU evaluation — full 14,042 samples, 5-shot, batch_size=16")
print("="*60)

results = lm_eval.simple_evaluate(
    model="squish",
    model_args=MODEL_ARGS,
    tasks=["mmlu"],
    num_fewshot=5,
    limit=None,
    log_samples=False,
)

summary = {}
top_accs = []
for task, metrics in results["results"].items():
    acc_norm = metrics.get("acc_norm,none")
    acc      = metrics.get("acc,none")
    stderr   = metrics.get("acc_norm_stderr,none") or metrics.get("acc_stderr,none")
    val      = acc_norm if acc_norm is not None else acc
    metric   = "acc_norm" if acc_norm is not None else "acc"
    if val is not None:
        summary[task] = {"acc": round(val, 4), "stderr": round(stderr, 4) if stderr else None, "metric": metric}
        top_accs.append(val)

# Print results
print("\n=== RESULTS ===")
for task, r in sorted(summary.items()):
    print(f"  {task:45s}: {r['acc']:.4f}")

if top_accs:
    avg = sum(top_accs) / len(top_accs)
    print(f"\n  MMLU average (57 subjects): {avg:.4f}")
    summary["_average"] = {"acc": round(avg, 4)}

out = {
    "model": "Qwen2.5-1.5B-Instruct-bf16-compressed",
    "hardware": "Apple M3",
    "n_subjects": 57,
    "total_samples": 14042,
    "num_fewshot": 5,
    "batch_size": 16,
    "mmlu_avg": round(avg, 4) if top_accs else None,
    "results": summary,
}
out_path = RESULTS_DIR / "accuracy_v9_mmlu.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\n  Saved → {out_path}")
print(f"\n=== DONE === MMLU average: {avg:.4f}")
