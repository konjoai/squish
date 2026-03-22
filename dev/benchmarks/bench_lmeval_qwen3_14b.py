#!/usr/bin/env python3
"""
bench_lmeval_qwen3_14b.py — Full lm-evaluation-harness suite for Qwen3-14B.

On Apple Silicon (M3 16 GB) the 29 GB BF16 model does not fit in RAM.
This script serves the MLX INT4 model (~7.8 GB) via mlx_lm.server on a local
OpenAI-compatible endpoint (default: http://localhost:8081) and runs lm-eval's
local-completions backend against it.

Tasks (industry-standard set)
------------------------------
  arc_easy         (ARC Easy)
  arc_challenge    (ARC Challenge)
  hellaswag        (HellaSwag)
  winogrande       (Winogrande)
  piqa             (PIQA)
  openbookqa       (OpenBookQA)
  mmlu             (MMLU — 57 subjects rolled up)
  truthfulqa_mc2   (TruthfulQA MC2)
  gsm8k            (GSM8K math reasoning, 5-shot)

Metrics reported per task
--------------------------
  acc / acc_norm / exact_match depending on the task

Usage
-----
  # Full suite — all tasks, full dataset (hours):
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py

  # Smoke test — 100 samples per task (minutes):
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py --limit 100

  # Specific tasks only:
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py --tasks arc_easy,hellaswag

  # Use a model server already running on a custom URL:
  python3 dev/benchmarks/bench_lmeval_qwen3_14b.py --server-url http://localhost:8081

Requirements
-----------
  pip install lm-eval
  pip install mlx-lm   (provides mlx_lm.server)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import signal
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
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8081

# Industry-standard tasks and their primary evaluation metric
TASKS: list[tuple[str, str]] = [
    ("arc_easy",       "acc_norm"),
    ("arc_challenge",  "acc_norm"),
    ("hellaswag",      "acc_norm"),
    ("winogrande",     "acc"),
    ("piqa",           "acc_norm"),
    ("openbookqa",     "acc_norm"),
    ("mmlu",           "acc"),
    ("truthfulqa_mc2", "acc"),
    ("gsm8k",          "exact_match,strict-match"),
]

# Standard published few-shot settings per task
FEWSHOT: dict[str, int] = {
    "arc_easy":       25,
    "arc_challenge":  25,
    "hellaswag":      10,
    "winogrande":      5,
    "piqa":            0,
    "openbookqa":      0,
    "mmlu":            5,
    "truthfulqa_mc2":  0,
    "gsm8k":           5,
}


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


def _detect_ram_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1e9
    except Exception:
        pass
    return 0.0


def _platform_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "ram_gb": 16,          # M3 16 GB — lm_eval/sysctl reports 17.18 (marketing vs actual)
        "python": sys.version.split()[0],
        "lm_eval": _lm_eval_version(),
    }


def _lm_eval_version() -> str:
    try:
        import lm_eval
        return lm_eval.__version__
    except Exception:
        return "unknown"


def _extract_metric(task_result: dict, metric_key: str) -> float | None:
    """Pull the primary metric from a task result dict (lm_eval 0.4.x format)."""
    primary = metric_key.split(",")[0]
    for k, v in task_result.items():
        if primary in k and isinstance(v, (int, float)):
            # Skip stderr / std keys
            if "stderr" not in k and "std" not in k:
                return float(v)
    return None


# ── server management ─────────────────────────────────────────────────────────

def _wait_for_server(url: str, timeout: int = 120) -> bool:
    """Poll the server's /health or / endpoint until it responds."""
    import urllib.request
    import urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{url}/v1/models", timeout=2)
            return True
        except Exception:
            time.sleep(2)
    return False


def _start_mlx_server(model_dir: Path, host: str, port: int) -> subprocess.Popen:
    """Launch mlx_lm.server in a subprocess and return the handle."""
    cmd = [
        sys.executable, "-m", "mlx_lm", "server",
        "--model", str(model_dir),
        "--host", host,
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"  Starting mlx_lm server: {D}{' '.join(cmd)}{NC}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc


# ── core evaluation ───────────────────────────────────────────────────────────

def run_lmeval(
    server_url: str,
    model_name: str,
    model_dir: Path,
    tasks: list[str],
    limit: int | None,
    num_fewshot_override: int | None,
) -> dict[str, Any]:
    """Run lm_eval with the local-completions backend and return the raw results."""
    try:
        import lm_eval
    except ImportError:
        print(f"{R}ERROR:{NC} lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)

    # Per-task few-shot counts
    num_fewshot: dict[str, int] | int
    if num_fewshot_override is not None:
        num_fewshot = num_fewshot_override
    else:
        num_fewshot = {t: FEWSHOT.get(t.split(",")[0], 0) for t in tasks}

    print(f"\n  Running lm_eval (local-completions → {server_url}) …")
    start = time.time()
    results = lm_eval.simple_evaluate(
        model="local-completions",
        model_args=(
            f"model={model_name},"
            f"base_url={server_url}/v1,"
            f"tokenizer={model_dir},"
            f"tokenizer_backend=huggingface,"
            f"tokenized_requests=False,"
            f"num_concurrent=1,"
            f"max_retries=3,"
            f"timeout=300"
        ),
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        log_samples=False,
    )
    elapsed = time.time() - start
    results["_elapsed_s"] = elapsed
    return results


# ── display & save ────────────────────────────────────────────────────────────

def _display_results(raw: dict, task_metrics: list[tuple[str, str]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    task_results: dict = raw.get("results", {})

    _hdr("lm-eval Results")

    for task, primary_metric in task_metrics:
        if task not in task_results:
            _err(task, "not in results")
            continue

        tr = task_results[task]
        score = _extract_metric(tr, primary_metric)
        if score is None:
            _err(task, f"metric '{primary_metric}' not found — keys: {list(tr.keys())[:4]}")
            continue

        pct = score * 100
        stderr_key = primary_metric.split(",")[0] + "_stderr,none"
        stderr = tr.get(stderr_key, None)
        extra = f"±{stderr*100:.2f}%" if isinstance(stderr, float) else primary_metric
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
    payload = {
        "model": model_dir.name,
        "model_path": str(model_dir),
        "timestamp": ts,
        "limit": limit,
        "platform": platform_info,
        "scores": scores,
        "raw_results": raw.get("results", {}),
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
        f"Platform: Apple M3 · {platform_info.get('ram_gb', 16)} GB RAM*"
    )
    print(f"\n| Task | Score |")
    print(f"|------|-------|")
    for task, score in sorted(scores.items()):
        print(f"| {task} | {score:.2f}% |")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"| **Average** | **{avg:.2f}%** |")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full lm-evaluation-harness suite for Qwen3-14B (Apple M3 / MLX INT4)."
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to MLX INT4 model (default: {DEFAULT_MODEL_DIR})",
    )
    ap.add_argument(
        "--server-url",
        type=str,
        default=None,
        help=(
            f"Use an already-running mlx_lm server at this URL "
            f"(default: start one automatically on port {DEFAULT_SERVER_PORT})"
        ),
    )
    ap.add_argument(
        "--server-port",
        type=int,
        default=DEFAULT_SERVER_PORT,
        help=f"Port for the auto-started mlx_lm server (default: {DEFAULT_SERVER_PORT})",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(t for t, _ in TASKS),
        help="Comma-separated task names (default: all standard tasks)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per task for a fast subset (e.g. 100). Omit for full eval.",
    )
    ap.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        dest="num_fewshot",
        help="Override few-shot count for all tasks (default: use standard published settings)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save JSON results (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown summary table at the end",
    )
    args = ap.parse_args()

    if not args.model_dir.exists():
        print(f"{R}ERROR:{NC} Model directory not found: {args.model_dir}")
        sys.exit(1)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    known = {t for t, _ in TASKS}
    task_metrics = [(t, m) for t, m in TASKS if t in task_names]
    for t in task_names:
        if t not in known:
            task_metrics.append((t, "acc"))

    pinfo = _platform_info()

    _hdr(
        "Qwen3-14B — Full lm-eval Benchmark Suite (MLX INT4 / M3 16 GB)",
        f"Model: {args.model_dir.name}  ·  Tasks: {len(task_names)}  ·  "
        f"Limit: {args.limit or 'full'}",
    )

    print(f"\n  Tasks to run ({len(task_names)}):")
    for t in task_names:
        fs = args.num_fewshot if args.num_fewshot is not None else FEWSHOT.get(t, 0)
        print(f"    {D}·{NC} {t}  ({fs}-shot)")

    # ── server setup ─────────────────────────────────────────────────────────
    server_proc: subprocess.Popen | None = None
    server_url = args.server_url

    if server_url is None:
        server_url = f"http://{DEFAULT_SERVER_HOST}:{args.server_port}"
        server_proc = _start_mlx_server(
            args.model_dir, DEFAULT_SERVER_HOST, args.server_port
        )
        print(f"  Waiting for server at {server_url} …")
        if not _wait_for_server(server_url, timeout=180):
            print(f"{R}ERROR:{NC} Server did not become ready in 180 s")
            if server_proc:
                server_proc.terminate()
            sys.exit(1)
        print(f"  {G}✓{NC} Server ready")
    else:
        print(f"  Using existing server at {server_url}")

    # ── run eval ─────────────────────────────────────────────────────────────
    try:
        raw = run_lmeval(
            server_url=server_url,
            model_name=args.model_dir.name,
            model_dir=args.model_dir,
            tasks=task_names,
            limit=args.limit,
            num_fewshot_override=args.num_fewshot,
        )
    finally:
        if server_proc is not None:
            print(f"\n  Stopping mlx_lm server (PID {server_proc.pid}) …")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    scores = _display_results(raw, task_metrics)

    out_file = _save_results(
        scores=scores,
        raw=raw,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        platform_info=pinfo,
    )
    print(f"\n  {G}✓{NC}  Results saved → {D}{out_file}{NC}")

    elapsed = raw.get("_elapsed_s", 0)
    print(f"  {D}Total eval time: {elapsed/60:.1f} min{NC}")

    if args.markdown:
        _print_markdown(scores, args.model_dir.name, pinfo, args.limit)


if __name__ == "__main__":
    main()


# Industry-standard tasks and the primary metric to display
TASKS: list[tuple[str, str]] = [
    ("arc_easy",       "acc_norm"),
    ("arc_challenge",  "acc_norm"),
    ("hellaswag",      "acc_norm"),
    ("winogrande",     "acc"),
    ("piqa",           "acc_norm"),
    ("openbookqa",     "acc_norm"),
    ("mmlu",           "acc"),
    ("truthfulqa_mc2", "acc"),
    ("gsm8k",          "exact_match,strict-match"),
]

# Few-shot settings per task (standard published settings)
FEWSHOT: dict[str, int] = {
    "arc_easy":       25,
    "arc_challenge":  25,
    "hellaswag":      10,
    "winogrande":      5,
    "piqa":            0,
    "openbookqa":      0,
    "mmlu":            5,
    "truthfulqa_mc2":  0,
    "gsm8k":           5,
}


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


def _detect_ram_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1e9
    except Exception:
        pass
    return 0.0


def _platform_info() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "ram_gb": round(_detect_ram_gb(), 1),
        "python": sys.version.split()[0],
        "lm_eval": _lm_eval_version(),
    }


def _lm_eval_version() -> str:
    try:
        import lm_eval
        return lm_eval.__version__
    except Exception:
        return "unknown"


def _extract_metric(task_result: dict, metric_key: str) -> float | None:
    """Pull the primary metric from a task result dict (lm_eval 0.4.x format)."""
    # In 0.4.x the results are under task_result['results'][task_name]
    # metric_key may be e.g. "acc_norm,none" or "acc_norm" — try both
    for k, v in task_result.items():
        if metric_key.split(",")[0] in k:
            if isinstance(v, (int, float)):
                return float(v)
    return None


# ── core evaluation ───────────────────────────────────────────────────────────

def run_lmeval(
    model_dir: Path,
    tasks: list[str],
    limit: int | None,
    batch_size: int,
    num_fewshot_override: int | None,
) -> dict[str, Any]:
    """Run lm_eval.simple_evaluate and return the raw results dict."""
    try:
        import lm_eval
    except ImportError:
        print(f"{R}ERROR:{NC} lm-eval not installed. Run: pip install lm-eval")
        sys.exit(1)

    from lm_eval.models.huggingface import HFLM

    # Detect device: MPS for Apple Silicon, CPU otherwise (no CUDA on M-series)
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"\n  Loading model from {D}{model_dir}{NC} on {device} …")
    lm = HFLM(
        pretrained=str(model_dir),
        dtype="bfloat16",
        device=device,
        batch_size=batch_size,
        trust_remote_code=True,
    )

    # Build per-task fewshot settings
    num_fewshot_map: dict[str, int] = {}
    for task in tasks:
        if num_fewshot_override is not None:
            num_fewshot_map[task] = num_fewshot_override
        else:
            # Use standard published few-shot counts
            base = task.split(",")[0]
            num_fewshot_map[task] = FEWSHOT.get(base, 0)

    start = time.time()
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=num_fewshot_map if num_fewshot_override is None else num_fewshot_override,
        limit=limit,
        batch_size=batch_size,
        log_samples=False,
    )
    elapsed = time.time() - start
    results["_elapsed_s"] = elapsed
    return results


# ── display & save ────────────────────────────────────────────────────────────

def _display_results(raw: dict, task_metrics: list[tuple[str, str]]) -> dict[str, float]:
    """Print per-task results table and return {task: score} dict."""
    scores: dict[str, float] = {}
    task_results: dict = raw.get("results", {})

    _hdr("lm-eval Results")

    for task, primary_metric in task_metrics:
        if task not in task_results:
            _err(task, "not in results")
            continue

        tr = task_results[task]
        score = _extract_metric(tr, primary_metric)
        if score is None:
            # Fallback: print all available metrics
            _err(task, f"metric '{primary_metric}' not found — keys: {list(tr.keys())[:4]}")
            continue

        pct = score * 100
        stderr_key = primary_metric.split(",")[0] + "_stderr,none"
        stderr = tr.get(stderr_key, tr.get(primary_metric + "_stderr", None))
        extra = f"±{stderr*100:.2f}%" if isinstance(stderr, float) else primary_metric
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

    payload = {
        "model": model_dir.name,
        "model_path": str(model_dir),
        "timestamp": ts,
        "limit": limit,
        "platform": platform_info,
        "scores": scores,
        "raw_results": raw.get("results", {}),
        "elapsed_s": raw.get("_elapsed_s"),
    }

    out_file.write_text(json.dumps(payload, indent=2, default=str))
    return out_file


def _print_markdown(scores: dict[str, float], model_name: str, platform_info: dict, limit: int | None) -> None:
    _hdr("Markdown Summary")
    limit_note = f" (limit={limit} samples/task)" if limit else " (full dataset)"
    print(f"\n## Qwen3-14B lm-eval Benchmark{limit_note}")
    print(f"\n*Model: `{model_name}` · Platform: {platform_info.get('processor', 'unknown')} · {platform_info.get('ram_gb', 0):.0f} GB RAM*")
    print(f"\n| Task | Score |")
    print(f"|------|-------|")
    for task, score in sorted(scores.items()):
        print(f"| {task} | {score:.2f}% |")
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"| **Average** | **{avg:.2f}%** |")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full lm-evaluation-harness suite for Qwen3-14B."
    )
    ap.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to model (default: {DEFAULT_MODEL_DIR})",
    )
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(t for t, _ in TASKS),
        help="Comma-separated task names (default: all standard tasks)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per task for a fast subset (e.g. 100). Omit for full eval.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1 — safe for large models on 16-17 GB RAM)",
    )
    ap.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        dest="num_fewshot",
        help="Override few-shot count for all tasks (default: use standard published settings)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save JSON results (default: {DEFAULT_OUTPUT_DIR})",
    )
    ap.add_argument(
        "--markdown",
        action="store_true",
        help="Print markdown summary table at the end",
    )
    args = ap.parse_args()

    if not args.model_dir.exists():
        print(f"{R}ERROR:{NC} Model directory not found: {args.model_dir}")
        sys.exit(1)

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    task_metrics = [(t, m) for t, m in TASKS if t in task_names]
    # For any user-supplied tasks not in our TASKS list, default to 'acc'
    known = {t for t, _ in TASKS}
    for t in task_names:
        if t not in known:
            task_metrics.append((t, "acc"))

    pinfo = _platform_info()

    _hdr(
        "Qwen3-14B — Full lm-eval Benchmark Suite",
        f"Model: {args.model_dir.name}  ·  Tasks: {len(task_names)}  ·  "
        f"Limit: {args.limit or 'full'}  ·  RAM: {pinfo.get('ram_gb', 0):.0f} GB",
    )

    print(f"\n  Tasks to run ({len(task_names)}):")
    for t in task_names:
        fs = args.num_fewshot if args.num_fewshot is not None else FEWSHOT.get(t, 0)
        print(f"    {D}·{NC} {t}  ({fs}-shot)")

    raw = run_lmeval(
        model_dir=args.model_dir,
        tasks=task_names,
        limit=args.limit,
        batch_size=args.batch_size,
        num_fewshot_override=args.num_fewshot,
    )

    scores = _display_results(raw, task_metrics)

    out_file = _save_results(
        scores=scores,
        raw=raw,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        limit=args.limit,
        platform_info=pinfo,
    )
    print(f"\n  {G}✓{NC}  Results saved → {D}{out_file}{NC}")

    elapsed = raw.get("_elapsed_s", 0)
    print(f"  {D}Total time: {elapsed/60:.1f} min{NC}")

    if args.markdown:
        _print_markdown(scores, args.model_dir.name, pinfo, args.limit)


if __name__ == "__main__":
    main()
