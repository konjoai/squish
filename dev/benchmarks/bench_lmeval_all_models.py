#!/usr/bin/env python3
"""
bench_lmeval_all_models.py — Full lm-evaluation-harness suite across all
previously-benchmarked models on Apple Silicon (M3 16 GB).

Each model is evaluated on the standard 6-task lm-eval suite using
``python -m mlx_lm evaluate``, which runs models natively on the Metal GPU.
Tasks are executed one at a time in separate subprocesses to release Metal
heap memory between tasks and prevent kIOGPUCommandBufferCallbackErrorOutOfMemory.

Models evaluated
----------------
  11 source models × (INT4 + INT3 + INT2) = 33 squish-quantized variants.
  BF16 reference baselines are excluded by default; add --include-bf16 to
  run them alongside the quantized variants.

  Qwen3-0.6B       BF16(1.1 GB)  INT4  INT3  INT2
  Llama-3.2-1B     BF16(2.3 GB)  INT4  INT3  INT2
  gemma-3-1b       BF16(2.5 GB)  INT4  INT3  INT2
  Qwen2.5-1.5B     BF16(2.9 GB)  INT4  INT3  INT2
  Llama-3.2-3B     BF16(6.0 GB)  INT4  INT3  INT2
  Qwen3-4B         BF16(7.5 GB)  INT4  INT3  INT2
  gemma-3-4b       BF16(9.3 GB)  INT4  INT3  INT2
  Mistral-7B       BF16(8.8 GB)  INT4  INT3  INT2
  Qwen2.5-7B       BF16(14.0 GB) INT4  INT3  INT2
  Qwen3-8B         BF16(15.0 GB) INT4  INT3  INT2
  Qwen3-14B        BF16(28.0 GB) INT4  INT3  INT2  (BF16 swap-risk on 16 GB)

Tasks (industry-standard)
--------------------------
  arc_easy        ARC Easy,       25-shot
  arc_challenge   ARC Challenge,  25-shot
  hellaswag       HellaSwag,      10-shot
  winogrande      Winogrande,      5-shot
  piqa            PIQA,            0-shot
  openbookqa      OpenBookQA,      0-shot

Generation sanity check (--gen-sanity)
--------------------------------------
  Before running lmeval, three short chat prompts are sent to the model.
  A model is flagged BROKEN if any response:
    - Has more than 80 %% of tokens identical (repetition loop)
    - Contains fewer than 3 unique words (garbage / numeric spew)
  BROKEN models are logged but NOT skipped; lmeval will still run to
  capture their (low) accuracy scores for the comparison table.

Usage
-----
  # Resume benchmark from last stopping point (safe to re-run):
  python3 dev/benchmarks/bench_lmeval_all_models.py --skip-existing

  # Smoke test (fast — 500 samples per task, good accuracy estimate):
  python3 dev/benchmarks/bench_lmeval_all_models.py --limit 500 --skip-existing

  # Full suite with generation sanity pre-check:
  python3 dev/benchmarks/bench_lmeval_all_models.py --gen-sanity

  # Include BF16 reference baselines:
  python3 dev/benchmarks/bench_lmeval_all_models.py --include-bf16 --skip-existing

  # Specific bit depths only:
  python3 dev/benchmarks/bench_lmeval_all_models.py --bits 2 3 --skip-existing

  # Specific models only:
  python3 dev/benchmarks/bench_lmeval_all_models.py --models Qwen3-0.6B-int2 Qwen3-0.6B-int3

  # Force re-run even if results exist:
  python3 dev/benchmarks/bench_lmeval_all_models.py --force

Requirements
------------
  pip install lm-eval mlx-lm
"""
from __future__ import annotations

import argparse
import json
import platform
import re
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
G  = "\033[32m"; Y  = "\033[33m"; C  = "\033[36m"; W  = "\033[1;37m"
D  = "\033[2m";  NC = "\033[0m";  R  = "\033[31m"; B  = "\033[1m"
M  = "\033[35m"

# ── generation-sanity prompts ─────────────────────────────────────────────────
# Three short, unambiguous questions that a working instruct model should answer
# with distinct, coherent words.  Used by --gen-sanity to detect broken models.
_SANITY_PROMPTS: list[str] = [
    "What is the capital of France?",
    "What color is the sky on a clear day?",
    "Name one planet in our solar system.",
]
_SANITY_MAX_TOKENS = 40   # enough for a short answer
# A response is broken if >80 % of words are identical (loop) or <3 unique words
_REPETITION_THRESHOLD = 0.80
_MIN_UNIQUE_WORDS     = 3

# ── model registry ────────────────────────────────────────────────────────────
_MODELS_ROOT = Path.home() / "models"

# (display_name, path_relative_to_models_root, approx_gb, notes)
# Sorted smallest → largest; grouped by model family.
# BF16 source models are listed as reference baselines.
# Squish-quantized variants under test: INT4, INT3, INT2.
# Note: Qwen3-14B-int4 maps to the already-compressed Qwen3-14B-mlx-int4 dir.
MODEL_REGISTRY: list[tuple[str, str, float, str]] = [
    # ── 0.6 B ──────────────────────────────────────────────────────────────
    ("Qwen3-0.6B-bf16",   "Qwen3-0.6B-bf16",   1.1,  "BF16 reference"),
    ("Qwen3-0.6B-int4",   "Qwen3-0.6B-int4",   0.35, "squish INT4"),
    ("Qwen3-0.6B-int3",   "Qwen3-0.6B-int3",   0.27, "squish INT3"),
    ("Qwen3-0.6B-int2",   "Qwen3-0.6B-int2",   0.20, "squish INT2"),
    # ── 1 B (Llama) ────────────────────────────────────────────────────────
    ("Llama-3.2-1B-bf16",   "Llama-3.2-1B-Instruct-bf16",   2.3,  "BF16 reference"),
    ("Llama-3.2-1B-int4",   "Llama-3.2-1B-Instruct-int4",   0.55, "squish INT4"),
    ("Llama-3.2-1B-int3",   "Llama-3.2-1B-Instruct-int3",   0.43, "squish INT3"),
    ("Llama-3.2-1B-int2",   "Llama-3.2-1B-Instruct-int2",   0.32, "squish INT2"),
    # ── 1 B (Gemma) ────────────────────────────────────────────────────────
    ("gemma-3-1b-bf16",   "gemma-3-1b-it-bf16",   2.5,  "BF16 reference"),
    ("gemma-3-1b-int4",   "gemma-3-1b-it-int4",   0.60, "squish INT4"),
    ("gemma-3-1b-int3",   "gemma-3-1b-it-int3",   0.47, "squish INT3"),
    ("gemma-3-1b-int2",   "gemma-3-1b-it-int2",   0.35, "squish INT2"),
    # ── 1.5 B ──────────────────────────────────────────────────────────────
    ("Qwen2.5-1.5B-bf16",   "Qwen2.5-1.5B-Instruct-bf16",   2.9,  "BF16 reference"),
    ("Qwen2.5-1.5B-int4",   "Qwen2.5-1.5B-Instruct-int4",   0.70, "squish INT4"),
    ("Qwen2.5-1.5B-int3",   "Qwen2.5-1.5B-Instruct-int3",   0.55, "squish INT3"),
    ("Qwen2.5-1.5B-int2",   "Qwen2.5-1.5B-Instruct-int2",   0.41, "squish INT2"),
    # ── 3 B ────────────────────────────────────────────────────────────────
    ("Llama-3.2-3B-bf16",   "Llama-3.2-3B-Instruct-bf16",   6.0,  "BF16 reference"),
    ("Llama-3.2-3B-int4",   "Llama-3.2-3B-Instruct-int4",   1.5,  "squish INT4"),
    ("Llama-3.2-3B-int3",   "Llama-3.2-3B-Instruct-int3",   1.2,  "squish INT3"),
    ("Llama-3.2-3B-int2",   "Llama-3.2-3B-Instruct-int2",   0.90, "squish INT2"),
    # ── 4 B (Qwen3) ────────────────────────────────────────────────────────
    ("Qwen3-4B-bf16",   "Qwen3-4B-bf16",   7.5,  "BF16 reference"),
    ("Qwen3-4B-int4",   "Qwen3-4B-int4",   2.0,  "squish INT4"),
    ("Qwen3-4B-int3",   "Qwen3-4B-int3",   1.6,  "squish INT3"),
    ("Qwen3-4B-int2",   "Qwen3-4B-int2",   1.2,  "squish INT2"),
    # ── 4 B (Gemma) ────────────────────────────────────────────────────────
    ("gemma-3-4b-bf16",   "gemma-3-4b-it-bf16",   9.3,  "BF16 reference"),
    ("gemma-3-4b-int4",   "gemma-3-4b-it-int4",   2.4,  "squish INT4"),
    ("gemma-3-4b-int3",   "gemma-3-4b-it-int3",   1.9,  "squish INT3"),
    ("gemma-3-4b-int2",   "gemma-3-4b-it-int2",   1.4,  "squish INT2"),
    # ── 7 B (Mistral) ──────────────────────────────────────────────────────
    ("Mistral-7B-bf16",   "Mistral-7B-Instruct-v0.3-bf16",   8.8,  "BF16 reference"),
    ("Mistral-7B-int4",   "Mistral-7B-Instruct-v0.3-int4",   3.5,  "squish INT4"),
    ("Mistral-7B-int3",   "Mistral-7B-Instruct-v0.3-int3",   2.8,  "squish INT3"),
    ("Mistral-7B-int2",   "Mistral-7B-Instruct-v0.3-int2",   2.1,  "squish INT2"),
    # ── 7 B (Qwen2.5) ──────────────────────────────────────────────────────
    ("Qwen2.5-7B-bf16",   "Qwen2.5-7B-Instruct-bf16",   14.0, "BF16 reference"),
    ("Qwen2.5-7B-int4",   "Qwen2.5-7B-Instruct-int4",    3.5,  "squish INT4"),
    ("Qwen2.5-7B-int3",   "Qwen2.5-7B-Instruct-int3",    2.8,  "squish INT3"),
    ("Qwen2.5-7B-int2",   "Qwen2.5-7B-Instruct-int2",    2.1,  "squish INT2"),
    # ── 8 B ────────────────────────────────────────────────────────────────
    ("Qwen3-8B-bf16",   "Qwen3-8B-bf16",   15.0, "BF16 reference"),
    ("Qwen3-8B-int4",   "Qwen3-8B-int4",    4.0,  "squish INT4"),
    ("Qwen3-8B-int3",   "Qwen3-8B-int3",    3.2,  "squish INT3"),
    ("Qwen3-8B-int2",   "Qwen3-8B-int2",    2.4,  "squish INT2"),
    # ── 14 B ───────────────────────────────────────────────────────────────
    ("Qwen3-14B-bf16",   "Qwen3-14B-bf16",   28.0, "BF16 reference (swap risk)"),
    ("Qwen3-14B-int4",   "Qwen3-14B-int4",    9.1,  "squish INT4"),
    ("Qwen3-14B-int3",   "Qwen3-14B-int3",    7.4,  "squish INT3"),
    ("Qwen3-14B-int2",   "Qwen3-14B-int2",    5.8,  "squish INT2"),
]

# ── task definitions ──────────────────────────────────────────────────────────
# (task_name, primary_lmeval_metric, standard_fewshots)
TASKS: list[tuple[str, str, int]] = [
    ("arc_easy",      "acc_norm,none", 25),
    ("arc_challenge", "acc_norm,none", 25),
    ("hellaswag",     "acc_norm,none", 10),
    ("winogrande",    "acc,none",       5),
    ("piqa",          "acc_norm,none",  0),
    ("openbookqa",    "acc_norm,none",  0),
]
_TASK_METRIC  = {t: m for t, m, _ in TASKS}
_TASK_FEWSHOT = {t: f for t, _, f in TASKS}
_ALL_TASK_NAMES = [t for t, _, _ in TASKS]


# ── helpers ────────────────────────────────────────────────────────────────────

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


def _info(msg: str) -> None:
    print(f"  {C}ℹ{NC}  {msg}")


def _platform_info() -> dict[str, Any]:
    return {
        "platform":   platform.platform(),
        "processor":  platform.processor() or platform.machine(),
        "ram_gb":     16,
        "python":     sys.version.split()[0],
        "mlx_lm":     _mlx_lm_version(),
        "lm_eval":    _lm_eval_version(),
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
    """Extract primary metric value from a lm_eval 0.4.x flat task-result dict."""
    primary = metric_key.split(",")[0]
    for k, v in task_result.items():
        if primary in k and isinstance(v, (int, float)):
            if "stderr" not in k and "std" not in k:
                return float(v)
    return None


def _result_path(model_name: str, output_dir: Path) -> Path | None:
    """Return the most recent existing result file for a given model, or None."""
    candidates = sorted(output_dir.glob(f"lmeval_{model_name}_*.json"))
    return candidates[-1] if candidates else None


# ── generation sanity check ───────────────────────────────────────────────────

def _run_generation_sanity(model_dir: Path) -> dict[str, Any]:
    """Run a quick generation sanity check: load the model, generate 3 short
    answers, and report whether any look broken (repetition loop or garbage).

    Returns a dict with keys:
        passed     – bool: True if no issues detected
        issues     – list[str]: human-readable problem descriptions
        responses  – list[str]: raw generated text for each prompt
        elapsed_s  – float: wall time for the whole check
    """
    import mlx_lm  # local import to avoid cost when --gen-sanity not used

    t0 = time.time()
    issues: list[str]    = []
    responses: list[str] = []

    try:
        model, tok = mlx_lm.load(str(model_dir))
    except Exception as exc:
        return {
            "passed":    False,
            "issues":    [f"load failed: {exc}"],
            "responses": [],
            "elapsed_s": time.time() - t0,
        }

    has_chat_template = (
        hasattr(tok, "apply_chat_template")
        and tok.chat_template is not None
    )

    for user_msg in _SANITY_PROMPTS:
        if has_chat_template:
            msgs   = [{"role": "user", "content": user_msg}]
            prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        else:
            prompt = user_msg

        try:
            out = mlx_lm.generate(
                model, tok,
                prompt=prompt,
                max_tokens=_SANITY_MAX_TOKENS,
                verbose=False,
            )
        except Exception as exc:
            issues.append(f"generate failed: {exc}")
            responses.append("")
            continue

        responses.append(out)
        words = [w.lower() for w in re.findall(r"\w+", out)]
        if not words:
            issues.append(f"empty output for: {user_msg!r}")
            continue
        most_common_frac = max(words.count(w) for w in words) / len(words)
        unique_count     = len(set(words))
        if most_common_frac >= _REPETITION_THRESHOLD:
            issues.append(
                f"repetition loop (top-word {most_common_frac:.0%}): {out[:60]!r}"
            )
        elif unique_count < _MIN_UNIQUE_WORDS:
            issues.append(
                f"garbage/incoherent output ({unique_count} unique words): {out[:60]!r}"
            )

    del model  # release Metal memory before lmeval subprocess starts

    return {
        "passed":    len(issues) == 0,
        "issues":    issues,
        "responses": responses,
        "elapsed_s": time.time() - t0,
    }


# ── per-task subprocess runner ────────────────────────────────────────────────

def _run_single_task(
    task: str,
    model_dir: Path,
    limit: int | None,
    num_fewshot_override: int | None,
    lmeval_out_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    """
    Run one task in its own ``python -m mlx_lm evaluate`` subprocess.
    Separation prevents Metal OOM when long few-shot prompts (e.g. MMLU 5-shot)
    exhaust memory while previous tasks' weights still hold the Metal heap.
    """
    fewshot = (
        num_fewshot_override
        if num_fewshot_override is not None
        else _TASK_FEWSHOT.get(task, 0)
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "evaluate",
        "--model",       str(model_dir),
        "--tasks",       task,
        "--num-shots",   str(fewshot),
        "--output-dir",  str(lmeval_out_dir),
        "--batch-size",  str(batch_size),
        "--trust-remote-code",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print(f"  {D}{' '.join(cmd)}{NC}\n")

    t0    = time.time()
    proc  = subprocess.run(cmd, text=True)
    elapsed = time.time() - t0

    if proc.returncode != 0:
        return {"error": f"mlx_lm exit code {proc.returncode}", "_elapsed_s": elapsed}

    # mlx_lm evaluate writes files named eval_* (no .json extension)
    all_files = sorted(
        (p for p in lmeval_out_dir.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    eval_files = [p for p in all_files if p.name.startswith("eval_")]
    candidates = eval_files if eval_files else all_files
    if not candidates:
        return {"error": "no output file written by mlx_lm evaluate", "_elapsed_s": elapsed}

    latest = candidates[-1]
    try:
        data = json.loads(latest.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error: {exc}", "_elapsed_s": elapsed}

    data["_elapsed_s"]       = elapsed
    data["_raw_output_file"] = str(latest)
    return data


# ── per-model evaluator ────────────────────────────────────────────────────────

def _run_model_eval(
    model_name: str,
    model_dir: Path,
    tasks: list[str],
    limit: int | None,
    num_fewshot_override: int | None,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    """
    Run all requested tasks for a single model and return the aggregated raw dict.
    Each task runs in its own subprocess to free Metal memory between evaluations.
    """
    lmeval_out_dir = output_dir / "_mlx_lmeval_raw" / model_name
    lmeval_out_dir.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, Any] = {}
    total_elapsed = 0.0
    errors: dict[str, str] = {}

    for i, task in enumerate(tasks, 1):
        fewshot = (
            num_fewshot_override
            if num_fewshot_override is not None
            else _TASK_FEWSHOT.get(task, 0)
        )
        print(
            f"\n  [{i}/{len(tasks)}] {W}{task}{NC}  ({fewshot}-shot"
            + (f", limit={limit}" if limit else "")
            + ")"
        )

        result  = _run_single_task(
            task=task,
            model_dir=model_dir,
            limit=limit,
            num_fewshot_override=num_fewshot_override,
            lmeval_out_dir=lmeval_out_dir,
            batch_size=batch_size,
        )
        elapsed = result.get("_elapsed_s", 0.0)
        total_elapsed += elapsed

        if "error" in result:
            errors[task] = result["error"]
            print(f"  {R}✗{NC}  {task}  {D}FAILED: {result['error']}{NC}")
            continue

        task_data = {k: v for k, v in result.items() if not k.startswith("_")}
        aggregate.update(task_data)
        print(f"  {G}✓{NC}  {task}  done in {elapsed / 60:.1f} min")

    aggregate["_elapsed_s"] = total_elapsed
    if errors:
        aggregate["_errors"] = errors
    return aggregate


# ── score extraction & display ────────────────────────────────────────────────

def _extract_scores(raw: dict, tasks: list[str]) -> dict[str, float]:
    """Pull numeric scores out of the aggregated raw result dict."""
    # Support both flat dict (mlx_lm) and results-wrapped (lm_eval standard) formats
    if "results" in raw and isinstance(raw["results"], dict):
        task_results: dict = raw["results"]
    else:
        task_results = {
            k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    scores: dict[str, float] = {}
    for task in tasks:
        if task not in task_results:
            continue
        primary_metric = _TASK_METRIC.get(task, "acc,none")
        score = _extract_metric(task_results[task], primary_metric)
        if score is not None:
            scores[task] = round(score * 100, 4)
    return scores


def _display_model_results(
    model_name: str,
    raw: dict,
    tasks: list[str],
) -> dict[str, float]:
    _hdr(f"Results — {model_name}")

    scores = _extract_scores(raw, tasks)
    errors = raw.get("_errors", {})

    if "results" in raw and isinstance(raw["results"], dict):
        task_results = raw["results"]
    else:
        task_results = {
            k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    for task in tasks:
        primary_metric = _TASK_METRIC.get(task, "acc,none")
        if task in errors:
            _err(task, errors[task])
            continue
        if task not in task_results:
            _err(task, "not in results output")
            continue

        score = scores.get(task)
        if score is None:
            _err(task, f"metric {primary_metric!r} not found")
            continue

        tr = task_results[task]
        stderr_key = primary_metric.split(",")[0] + "_stderr,none"
        stderr = tr.get(stderr_key)
        extra  = f"±{stderr * 100:.2f}%" if isinstance(stderr, float) else primary_metric
        _ok(task, f"{score:.2f}%", extra)

    return scores


def _save_model_result(
    model_name: str,
    model_dir: Path,
    model_size_gb: float,
    scores: dict[str, float],
    raw: dict,
    output_dir: Path,
    limit: int | None,
    platform_info: dict,
) -> Path:
    """Save per-model result JSON to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_file = output_dir / f"lmeval_{model_name}_{ts}.json"

    if "results" in raw and isinstance(raw["results"], dict):
        raw_results = raw["results"]
    else:
        raw_results = {
            k: v for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, dict)
        }

    payload = {
        "model":          model_name,
        "model_path":     str(model_dir),
        "model_size_gb":  model_size_gb,
        "timestamp":      ts,
        "limit":          limit,
        "platform":       platform_info,
        "scores":         scores,
        "raw_results":    raw_results,
        "elapsed_s":      raw.get("_elapsed_s"),
        "errors":         raw.get("_errors", {}),
    }
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    return out_file


# ── comparison table ──────────────────────────────────────────────────────────

def _print_comparison_markdown(
    all_scores:  dict[str, dict[str, float]],
    tasks:       list[str],
    platform_info: dict,
    limit:       int | None,
) -> None:
    """Print a markdown comparison table across all evaluated models."""
    _hdr("Multi-Model Comparison (Markdown)")

    limit_note = f" (limit={limit} samples/task)" if limit else " (full dataset)"
    print(f"\n## lm-eval Multi-Model Benchmark{limit_note}")
    print(
        f"\n*Platform: Apple M3 · 16 GB RAM · "
        f"mlx-lm {platform_info.get('mlx_lm', '?')} · "
        f"lm-eval {platform_info.get('lm_eval', '?')} · "
        f"{datetime.now().strftime('%Y-%m-%d')}*"
    )

    # Header row
    task_cols = "".join(f" {t} |" for t in tasks)
    print(f"\n| Model |{task_cols} Avg |")
    sep_cols = "".join(" ------ |" for _ in tasks)
    print(f"| ----- |{sep_cols} --- |")

    for model_name, scores in sorted(all_scores.items()):
        row_scores = [scores.get(t) for t in tasks]
        cells      = "".join(
            f" {s:.1f}% |" if s is not None else " — |"
            for s in row_scores
        )
        valid = [s for s in row_scores if s is not None]
        avg   = f"{sum(valid) / len(valid):.1f}%" if valid else "—"
        print(f"| {model_name} |{cells} **{avg}** |")


def _save_comparison_json(
    all_scores:     dict[str, dict[str, float]],
    all_elapsed:    dict[str, float],
    tasks:          list[str],
    output_dir:     Path,
    limit:          int | None,
    platform_info:  dict,
) -> Path:
    """Save the cross-model comparison to a single JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_file = output_dir / f"lmeval_comparison_{ts}.json"
    payload  = {
        "timestamp":    ts,
        "limit":        limit,
        "tasks":        tasks,
        "platform":     platform_info,
        "scores":       all_scores,
        "elapsed_s":    all_elapsed,
    }
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    return out_file


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Full lm-eval benchmark suite across all previously-benchmarked models "
            "on Apple Silicon M3 16 GB (mlx_lm Metal backend)."
        )
    )
    ap.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help=(
            "Subset of model names to evaluate (default: all). "
            f"Available: {', '.join(n for n, *_ in MODEL_REGISTRY)}"
        ),
    )
    ap.add_argument(
        "--tasks",
        nargs="+",
        default=_ALL_TASK_NAMES,
        help="Task names (default: all 9 standard tasks)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per task — use 100 for a fast smoke test. Omit for full eval.",
    )
    ap.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        dest="num_fewshot",
        help="Override few-shot count for all tasks (default: standard per-task settings)",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for mlx_lm inference (default: 1)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Directory for JSON results (default: results/)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models that already have a result JSON in --output-dir",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-evaluate even if results already exist (overrides --skip-existing)",
    )
    ap.add_argument(
        "--gen-sanity",
        action="store_true",
        dest="gen_sanity",
        help=(
            "Run a 3-prompt generation sanity check before lmeval.  "
            "Detects repetition loops and garbage output (broken INT2/INT3 models).  "
            "Reports issues but does not skip the model so lmeval scores are still recorded."
        ),
    )
    ap.add_argument(
        "--include-bf16",
        action="store_true",
        dest="include_bf16",
        help=(
            "Include BF16 reference baselines in the evaluation.  "
            "Excluded by default because they are slow and rarely change."
        ),
    )
    ap.add_argument(
        "--bits",
        nargs="+",
        type=int,
        choices=[2, 3, 4],
        default=None,
        metavar="N",
        dest="bits",
        help=(
            "Only evaluate models quantised at these bit widths "
            "(e.g. --bits 2 3 skips INT4 and BF16 models).  "
            "BF16 models are only included when --include-bf16 is also given."
        ),
    )
    ap.add_argument(
        "--models-root",
        type=Path,
        default=_MODELS_ROOT,
        help=f"Root directory containing model subdirectories (default: {_MODELS_ROOT})",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Print per-model markdown summary tables",
    )
    args = ap.parse_args()

    # ── resolve model list ────────────────────────────────────────────────────
    if args.models:
        requested = set(args.models)
        registry  = [r for r in MODEL_REGISTRY if r[0] in requested]
        missing   = requested - {r[0] for r in registry}
        if missing:
            print(f"{R}ERROR:{NC} Unknown model(s): {', '.join(sorted(missing))}")
            print(f"Available: {', '.join(n for n, *_ in MODEL_REGISTRY)}")
            sys.exit(1)
    else:
        registry = list(MODEL_REGISTRY)

    # ── apply --include-bf16 / --bits filters ─────────────────────────────────
    if not args.include_bf16:
        registry = [r for r in registry if "bf16" not in r[0].lower()]

    if args.bits:
        def _model_bits(name: str) -> int | None:
            for b in [2, 3, 4]:
                if f"int{b}" in name.lower():
                    return b
            return None  # BF16 baseline

        registry = [r for r in registry if _model_bits(r[0]) in args.bits]

    # ── filter to models that actually exist on disk ──────────────────────────
    available = []
    for name, rel_path, size_gb, notes in registry:
        model_dir = args.models_root / rel_path
        if model_dir.exists():
            available.append((name, model_dir, size_gb, notes))
        else:
            print(f"{Y}WARN:{NC} Model not found on disk, skipping: {model_dir}")

    if not available:
        print(f"{R}ERROR:{NC} No models found in {args.models_root}")
        sys.exit(1)

    platform_info = _platform_info()

    # ── header ────────────────────────────────────────────────────────────────
    _hdr(
        "lm-eval Multi-Model Benchmark",
        f"{len(available)} models · {len(args.tasks)} tasks"
        + (f" · limit={args.limit}" if args.limit else " · full dataset"),
    )
    print(f"\n  {D}Platform : {platform_info['platform']}{NC}")
    print(f"  {D}mlx-lm   : {platform_info['mlx_lm']}{NC}")
    print(f"  {D}lm-eval  : {platform_info['lm_eval']}{NC}")
    print(f"\n  Models to evaluate:")
    for name, model_dir, size_gb, notes in available:
        flag = ""
        if _result_path(name, args.output_dir) is not None:
            if args.skip_existing and not args.force:
                flag = f"  {Y}[will skip — result exists]{NC}"
            else:
                flag = f"  {D}[result exists — will re-run]{NC}"
        print(f"    {C}{name:<40}{NC}  {size_gb:5.1f} GB  {D}{notes}{NC}{flag}")
    print()

    suite_t0     = time.time()
    all_scores:  dict[str, dict[str, float]] = {}
    all_elapsed: dict[str, float]            = {}

    # ── evaluate each model ───────────────────────────────────────────────────
    for model_idx, (name, model_dir, size_gb, notes) in enumerate(available, 1):
        _hdr(
            f"[{model_idx}/{len(available)}] {name}",
            f"{size_gb:.1f} GB  {notes}  →  {model_dir}",
        )

        # Skip logic
        existing = _result_path(name, args.output_dir)
        if existing and args.skip_existing and not args.force:
            _info(f"Skipping — existing result: {existing.name}")
            try:
                data   = json.loads(existing.read_text())
                scores = data.get("scores", {})
                all_scores[name]  = scores
                all_elapsed[name] = data.get("elapsed_s", 0.0)
                for task, score in scores.items():
                    _ok(task, f"{score:.2f}%", "(loaded from cache)")
            except Exception as exc:
                _err(name, f"could not load existing result: {exc}")
            continue

        # ── optional generation sanity check ──────────────────────────────────
        if args.gen_sanity:
            _info("gen-sanity", f"loading model for quick generation check …")
            sanity = _run_generation_sanity(model_dir)
            if sanity["passed"]:
                _ok("gen-sanity", f"PASS  (took {sanity['elapsed_s']:.1f}s)")
            else:
                for issue in sanity["issues"]:
                    _err("gen-sanity", f"BROKEN: {issue}")
                _info(
                    "gen-sanity",
                    "model appears broken — lmeval will still run for score record",
                )
        # ─────────────────────────────────────────────────────────────────────

        raw = _run_model_eval(
            model_name=name,
            model_dir=model_dir,
            tasks=args.tasks,
            limit=args.limit,
            num_fewshot_override=args.num_fewshot,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

        scores = _display_model_results(name, raw, args.tasks)
        all_scores[name]  = scores
        all_elapsed[name] = raw.get("_elapsed_s", 0.0)

        out_file = _save_model_result(
            model_name=name,
            model_dir=model_dir,
            model_size_gb=size_gb,
            scores=scores,
            raw=raw,
            output_dir=args.output_dir,
            limit=args.limit,
            platform_info=platform_info,
        )
        _info(f"Saved: {out_file.name}")

        if args.markdown and scores:
            print(f"\n## lm-eval — {name}")
            print(f"\n| Task | Fewshot | Score |")
            print("|------|---------|-------|")
            for task in sorted(scores):
                fs = _TASK_FEWSHOT.get(task, 0)
                print(f"| {task} | {fs} | {scores[task]:.2f}% |")
            avg = sum(scores.values()) / len(scores)
            print(f"| **Average** | - | **{avg:.2f}%** |")

    # ── final comparison ─────────────────────────────────────────────────────
    total_elapsed = time.time() - suite_t0
    _hdr(
        "Benchmark Complete",
        f"Total wall time: {total_elapsed / 3600:.2f} h  ({total_elapsed / 60:.0f} min)",
    )

    _print_comparison_markdown(
        all_scores=all_scores,
        tasks=args.tasks,
        platform_info=platform_info,
        limit=args.limit,
    )

    cmp_file = _save_comparison_json(
        all_scores=all_scores,
        all_elapsed=all_elapsed,
        tasks=args.tasks,
        output_dir=args.output_dir,
        limit=args.limit,
        platform_info=platform_info,
    )
    _info(f"Comparison saved: {cmp_file.name}")


if __name__ == "__main__":
    main()
