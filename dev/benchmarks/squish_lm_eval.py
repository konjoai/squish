#!/usr/bin/env python3
"""squish_lm_eval.py — squish-native lm_eval harness.

Evaluates models stored in squish **npy-dir format** (produced by
``squish compress --format int4``, ``--format mixed_attn``, etc.) on the
standard 6-task lm_eval benchmark suite.

Standard ``mlx_lm evaluate`` can only load models from native MLX safetensors
(config.json-based dirs).  Squish's npy-dir format is an intermediate compressed
representation that ``mlx_lm.load()`` cannot handle.  This harness bridges the
gap by:

1. Calling :func:`squish.quant.compressed_loader.load_from_npy_dir` — this
   triggers a **one-time cache build** that constructs a ``squish_4bit/`` (INT4
   AWQ) or ``squish_3bit/`` (INT3) subdirectory of valid MLX safetensors inside
   the npy-dir.  Subsequent calls hit the cache and skip the Vectro step.

2. Running ``python -m mlx_lm evaluate`` subprocesses on the cached MLX
   safetensors dir — one task per subprocess to prevent Metal OOM.

3. Saving results in the same JSON format as ``bench_lmeval_all_models.py`` so
   they slot directly into the accuracy table.

**Acceptance criterion (Wave 41):** arc_easy on Qwen2.5-1.5B-Instruct-int4 AWQ
must land within ±2pp of the validated mlx_lm.convert INT4 baseline of 70.6%
(SESSION.md, 2026-03-28).

Usage::

    # Evaluate a squish INT4 AWQ npy-dir on all 6 tasks (limit=500):
    python3 dev/benchmarks/squish_lm_eval.py \\
        --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --limit 500

    # Mixed-attention (FP16 attn + INT4 MLP):
    python3 dev/benchmarks/squish_lm_eval.py \\
        --npy-dir ~/models/Qwen2.5-1.5B-Instruct-mixed-attn \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --tasks arc_easy arc_challenge

    # Multiple npy-dirs in one run:
    python3 dev/benchmarks/squish_lm_eval.py \\
        --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq \\
                  ~/models/Qwen2.5-1.5B-Instruct-mixed-attn \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --limit 500

    # Skip cache build if squish_4bit/ already exists:
    python3 dev/benchmarks/squish_lm_eval.py \\
        --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --skip-cache-build

    # Compare result against validated baseline (prints PASS/FAIL):
    python3 dev/benchmarks/squish_lm_eval.py \\
        --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq \\
        --model-dir ~/models/Qwen2.5-1.5B-Instruct \\
        --baseline 70.6 --threshold 2.0 --tasks arc_easy

Output::

    Results written to results/squish_lmeval_<timestamp>/
    One JSON file per npy-dir, named by directory basename.

Requirements::

    pip install lm-eval mlx-lm
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── repo root on sys.path so squish.quant is importable ──────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ── colour codes (same palette as bench_lmeval_all_models.py) ────────────────
G  = "\033[32m"; Y  = "\033[33m"; C  = "\033[36m"; W  = "\033[1;37m"
D  = "\033[2m";  NC = "\033[0m";  R  = "\033[31m"; B  = "\033[1m"

# ── task definitions (copied from bench_lmeval_all_models.py — single source) ─
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

# ── Wave 41 acceptance criterion: arc_easy Qwen2.5-1.5B-Instruct-int4 ────────
_WAVE41_BASELINE_ARC_EASY = 70.6  # % — mlx_lm.convert INT4 g=64, limit=500
_WAVE41_THRESHOLD_PP      = 2.0   # ±2pp tolerance


# ---------------------------------------------------------------------------
# Pure helpers — fully testable without real models
# ---------------------------------------------------------------------------

def _validate_npy_dir(path: Path) -> tuple[bool, str]:
    """Return (is_valid, reason) for a candidate npy-dir or native MLX dir.

    A path is valid if it is either:
    - A squish npy-dir: has ``manifest.json`` **and** a ``tensors/`` subdirectory.
    - A native MLX safetensors dir: has ``config.json`` **without** ``manifest.json``
      (load_from_npy_dir handles this via its Tier 0a fast path).
    """
    if not path.exists():
        return False, f"path does not exist: {path}"
    if not path.is_dir():
        return False, f"path is not a directory: {path}"

    has_manifest = (path / "manifest.json").exists()
    has_tensors  = (path / "tensors").is_dir()
    has_config   = (path / "config.json").exists()

    if has_config and not has_manifest:
        return True, "native-mlx"
    if has_manifest and has_tensors:
        return True, "squish-npy-dir"
    if has_manifest and not has_tensors:
        return False, "manifest.json found but tensors/ directory missing"
    return False, "neither config.json (native MLX) nor manifest.json+tensors/ (squish npy-dir) found"


def _detect_eval_dir(npy_dir: Path) -> Path | None:
    """Return the MLX-loadable directory to pass to ``mlx_lm evaluate``.

    Precedence:
    1. ``npy_dir/squish_4bit/`` — INT4 AWQ cache (best quality, native INT4 Metal)
    2. ``npy_dir/squish_3bit/`` — INT3 cache
    3. ``npy_dir`` itself if it has ``config.json`` (native MLX dir)
    4. ``None`` — no loadable cache found (cache must be built first)
    """
    four_bit = npy_dir / "squish_4bit"
    three_bit = npy_dir / "squish_3bit"

    if (npy_dir / ".squish_4bit_ready").exists() and (four_bit / "config.json").exists():
        return four_bit
    if (npy_dir / ".squish_3bit_ready").exists() and (three_bit / "model.safetensors").exists():
        return three_bit
    if (npy_dir / "config.json").exists() and not (npy_dir / "manifest.json").exists():
        # Native MLX safetensors — usable directly
        return npy_dir
    return None


def _extract_metric(task_result: dict, metric_key: str) -> float | None:
    """Extract primary metric value from a lm_eval 0.4.x / mlx_lm flat result dict.

    Handles both the flat dict format produced by ``mlx_lm evaluate`` and the
    nested ``{"results": {"task": {...}}}`` format produced by lm_eval 0.4.x.
    """
    # Unwrap lm_eval 0.4.x nested format
    if "results" in task_result:
        for _task_name, task_data in task_result["results"].items():
            if isinstance(task_data, dict):
                val = _extract_metric(task_data, metric_key)
                if val is not None:
                    return val
        return None

    primary = metric_key.split(",")[0]
    for k, v in task_result.items():
        if primary in k and isinstance(v, (int, float)):
            if "stderr" not in k and "std" not in k:
                return float(v)
    return None


def _parse_result_file(result_path: Path) -> dict[str, Any]:
    """Parse a JSON file written by ``mlx_lm evaluate`` and return the dict.

    Returns ``{"error": "..."}`` on failure.
    """
    if not result_path.exists():
        return {"error": f"result file not found: {result_path}"}
    try:
        return json.loads(result_path.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"JSON parse error in {result_path}: {exc}"}


def compare_to_baseline(
    score_pct: float,
    baseline_pct: float = _WAVE41_BASELINE_ARC_EASY,
    threshold_pp: float = _WAVE41_THRESHOLD_PP,
) -> tuple[bool, float]:
    """Return (passed, delta_pp) where passed is True if within threshold.

    Args:
        score_pct:    Observed metric value as a percentage (e.g. 70.2).
        baseline_pct: Expected baseline (default: Wave 41 arc_easy 70.6%).
        threshold_pp: Maximum absolute deviation in percentage points.

    Returns:
        (passed, delta_pp) — delta_pp is positive when score > baseline.
    """
    delta = score_pct - baseline_pct
    return abs(delta) <= threshold_pp, delta


def _results_path(output_dir: Path, model_stem: str, ts: str) -> Path:
    """Return the path for saving a per-model JSON result file."""
    return output_dir / f"squish_lmeval_{ts}" / f"{model_stem}.json"


def _npy_dir_format_tag(npy_dir: Path) -> str:
    """Infer a short format tag from the npy-dir name or contents.

    Examples:
        Qwen2.5-1.5B-Instruct-int4-awq → "int4-awq"
        Qwen2.5-1.5B-Instruct-mixed-attn → "mixed-attn"
        Qwen3-0.6B-int3 → "int3"
        (unknown) → "squish-npy"
    """
    name = npy_dir.name.lower()
    for tag in ("mixed-attn", "mixed_attn", "int4-awq", "int4", "int3", "int2"):
        if tag.replace("_", "-") in name.replace("_", "-"):
            return tag
    return "squish-npy"


def _platform_info() -> dict[str, Any]:
    """Return hardware/software metadata for result provenance."""
    mlx_lm_ver = "unknown"
    lm_eval_ver = "unknown"
    try:
        import mlx_lm
        mlx_lm_ver = mlx_lm.__version__
    except Exception:
        pass
    try:
        import lm_eval
        lm_eval_ver = lm_eval.__version__
    except Exception:
        pass
    return {
        "platform":  platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python":    sys.version.split()[0],
        "mlx_lm":   mlx_lm_ver,
        "lm_eval":  lm_eval_ver,
    }


def _summarize_row(tag: str, task: str, score: float | None) -> str:
    """Return a single-line summary row for the console table."""
    if score is None:
        return f"  {R}✗{NC}  {tag:<45} {task:<20}  {R}{'N/A':>8}{NC}"
    return f"  {G}✓{NC}  {tag:<45} {task:<20}  {G}{score:>7.1f}%{NC}"


# ---------------------------------------------------------------------------
# Cache build
# ---------------------------------------------------------------------------

def _ensure_eval_dir(
    npy_dir: Path,
    model_dir: Path,
    skip_cache_build: bool = False,
    quiet: bool = False,
) -> Path:
    """Ensure that an mlx-loadable eval dir exists for *npy_dir*.

    If *skip_cache_build* is True, only checks for an existing cache.
    Raises ``FileNotFoundError`` if the cache does not exist and building
    is skipped.
    Raises ``RuntimeError`` if cache build fails.
    """
    eval_dir = _detect_eval_dir(npy_dir)
    if eval_dir is not None:
        if not quiet:
            print(f"  {C}ℹ{NC}  Using existing eval dir: {eval_dir}")
        return eval_dir

    if skip_cache_build:
        raise FileNotFoundError(
            f"No cached eval dir found under {npy_dir} and --skip-cache-build is set.\n"
            "Run without --skip-cache-build to trigger one-time cache construction."
        )

    # Trigger one-time cache build via load_from_npy_dir
    if not quiet:
        print(f"\n{C}  Building squish cache for {npy_dir.name}…{NC}")
        print(f"  {D}(first run only — subsequent evaluations skip this step){NC}\n")

    try:
        from squish.quant.compressed_loader import load_from_npy_dir  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "squish.quant.compressed_loader not importable. "
            f"Run from the repo root: python3 dev/benchmarks/squish_lm_eval.py …\n{exc}"
        ) from exc

    try:
        model, tokenizer, stats = load_from_npy_dir(
            str(npy_dir),
            str(model_dir),
            verbose=not quiet,
            return_stats=True,
        )
        del model, tokenizer  # release Metal memory; we only needed the cache-build side-effect
    except Exception as exc:
        raise RuntimeError(f"load_from_npy_dir failed for {npy_dir}: {exc}") from exc

    eval_dir = _detect_eval_dir(npy_dir)
    if eval_dir is None:
        raise RuntimeError(
            f"load_from_npy_dir completed but no cached eval dir detected under {npy_dir}.\n"
            "Check that squish_4bit/ or squish_3bit/ was written correctly."
        )

    if not quiet:
        loader = stats.get("loader", "?")
        delta  = stats.get("ram_delta_mb", 0)
        print(f"  {G}✓{NC}  Cache built  loader={loader}  RAM Δ {delta:+.0f} MB → {eval_dir.name}/")

    return eval_dir


# ---------------------------------------------------------------------------
# Per-task subprocess evaluation
# ---------------------------------------------------------------------------

def _run_single_task(
    task: str,
    eval_dir: Path,
    limit: int | None,
    lmeval_out_dir: Path,
    batch_size: int,
    disable_thinking: bool,
    quiet: bool,
) -> dict[str, Any]:
    """Run one lm_eval task via ``python -m mlx_lm evaluate`` subprocess.

    Mirrors the pattern in bench_lmeval_all_models.py — one process per task
    to release Metal heap between tasks.
    """
    fewshot = _TASK_FEWSHOT.get(task, 0)
    lmeval_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "evaluate",
        "--model",         str(eval_dir),
        "--tasks",         task,
        "--num-shots",     str(fewshot),
        "--output-dir",    str(lmeval_out_dir),
        "--batch-size",    str(batch_size),
        "--trust-remote-code",
    ]
    if disable_thinking:
        cmd += ["--apply-chat-template", "--chat-template-args", '{"enable_thinking": false}']
    if limit is not None:
        cmd += ["--limit", str(limit)]

    if not quiet:
        print(f"  {D}{' '.join(cmd)}{NC}\n")

    t0   = time.monotonic()
    proc = subprocess.run(cmd, text=True, capture_output=True, close_fds=True)
    elapsed = time.monotonic() - t0

    if not quiet and proc.stdout:
        print(proc.stdout, end="")

    if proc.returncode != 0:
        stderr_tail = (proc.stderr or "")[-600:].strip()
        msg = f"mlx_lm exit {proc.returncode}"
        if stderr_tail:
            msg += f" | stderr: {stderr_tail}"
        if not quiet:
            print(f"  {R}✗{NC}  {task}: {msg}", file=sys.stderr)
        return {"error": msg, "_elapsed_s": elapsed}

    # mlx_lm evaluate writes output files named eval_* (no .json extension usually)
    all_files = sorted(
        (p for p in lmeval_out_dir.rglob("*") if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )
    eval_files = [p for p in all_files if p.name.startswith("eval_")]
    candidates = eval_files if eval_files else all_files
    if not candidates:
        return {"error": "no output file written by mlx_lm evaluate", "_elapsed_s": elapsed}

    latest = candidates[-1]
    data = _parse_result_file(latest)
    data["_elapsed_s"]       = elapsed
    data["_raw_output_file"] = str(latest)
    return data


# ---------------------------------------------------------------------------
# Per-model evaluation orchestration
# ---------------------------------------------------------------------------

def _is_thinking_model(model_name: str) -> bool:
    """Return True for Qwen3-family models that emit <think>...</think> tokens."""
    # Same heuristic as bench_lmeval_all_models.py
    return any(model_name.startswith(p) for p in ("Qwen3",))


def evaluate_npy_dir(
    npy_dir: Path,
    model_dir: Path,
    tasks: list[str],
    *,
    limit: int | None = 500,
    skip_cache_build: bool = False,
    batch_size: int = 4,
    output_dir: Path,
    baseline_pct: float | None = None,
    threshold_pp: float = _WAVE41_THRESHOLD_PP,
    quiet: bool = False,
) -> dict[str, Any]:
    """Evaluate one npy-dir model on *tasks*, return aggregated result dict.

    Returns a dict with keys:
    - ``"model_name"``, ``"npy_dir"``, ``"eval_dir"``, ``"format_tag"``
    - ``"tasks"``: mapping task → {"score_pct": float|None, "raw": dict}
    - ``"platform"``: hardware/software metadata
    - ``"baseline_check"``: acceptance criterion result (if baseline_pct supplied)
    """
    if not quiet:
        hdr = "─" * 72
        print(f"\n{W}{hdr}{NC}")
        print(f"{C}  Evaluating: {npy_dir.name}{NC}")
        print(f"{W}{hdr}{NC}")

    # Validate input
    is_valid, reason = _validate_npy_dir(npy_dir)
    if not is_valid:
        return {
            "model_name": npy_dir.name,
            "npy_dir": str(npy_dir),
            "error": reason,
        }

    # Build / locate the mlx-loadable eval dir
    eval_dir = _ensure_eval_dir(npy_dir, model_dir,
                                skip_cache_build=skip_cache_build, quiet=quiet)

    format_tag    = _npy_dir_format_tag(npy_dir)
    disable_think = _is_thinking_model(npy_dir.name)
    ts            = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    # Temp output dir for this model's lmeval dumps
    lmeval_tmp = output_dir / f"_tmp_{npy_dir.name}"

    task_scores: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not quiet:
            print(f"\n  {B}Task:{NC} {task}")
        raw = _run_single_task(
            task=task,
            eval_dir=eval_dir,
            limit=limit,
            lmeval_out_dir=lmeval_tmp / task,
            batch_size=batch_size,
            disable_thinking=disable_think,
            quiet=quiet,
        )
        metric_key = _TASK_METRIC.get(task, "acc_norm,none")
        score_pct  = None
        if "error" not in raw:
            raw_val = _extract_metric(raw, metric_key)
            if raw_val is not None:
                # lm_eval returns fractions (0–1); multiply to percentage
                score_pct = raw_val * 100 if raw_val <= 1.0 else raw_val
        task_scores[task] = {"score_pct": score_pct, "raw": raw}
        if not quiet:
            print(_summarize_row(npy_dir.name, task, score_pct))

    # Baseline acceptance check
    baseline_check: dict[str, Any] = {}
    if baseline_pct is not None and "arc_easy" in task_scores:
        arc_score = task_scores["arc_easy"].get("score_pct")
        if arc_score is not None:
            passed, delta = compare_to_baseline(arc_score, baseline_pct, threshold_pp)
            baseline_check = {
                "task":         "arc_easy",
                "score_pct":    arc_score,
                "baseline_pct": baseline_pct,
                "threshold_pp": threshold_pp,
                "delta_pp":     round(delta, 2),
                "passed":       passed,
            }
            label = f"{G}PASS{NC}" if passed else f"{R}FAIL{NC}"
            print(f"\n  {label}  arc_easy {arc_score:.1f}%  "
                  f"(baseline {baseline_pct:.1f}% ± {threshold_pp:.1f}pp, "
                  f"delta {delta:+.2f}pp)")

    result: dict[str, Any] = {
        "model_name":     npy_dir.name,
        "npy_dir":        str(npy_dir),
        "eval_dir":       str(eval_dir),
        "format_tag":     format_tag,
        "tasks":          task_scores,
        "platform":       _platform_info(),
        "timestamp":      ts,
        "limit":          limit,
        "batch_size":     batch_size,
    }
    if baseline_check:
        result["baseline_check"] = baseline_check

    # Save per-model result
    out_path = _results_path(output_dir, npy_dir.name, ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    if not quiet:
        print(f"\n  {G}Saved:{NC} {out_path}")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="squish_lm_eval",
        description=(
            "Evaluate squish npy-dir compressed models on the standard lm_eval "
            "benchmark suite (arc_easy, arc_challenge, hellaswag, winogrande, piqa, "
            "openbookqa). Builds a squish_4bit/ or squish_3bit/ safetensors cache on "
            "first run, then delegates to mlx_lm evaluate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python3 dev/benchmarks/squish_lm_eval.py \\\n"
            "      --npy-dir ~/models/Qwen2.5-1.5B-Instruct-int4-awq \\\n"
            "      --model-dir ~/models/Qwen2.5-1.5B-Instruct \\\n"
            "      --limit 500\n"
        ),
    )
    p.add_argument(
        "--npy-dir", nargs="+", required=True, metavar="PATH",
        help="One or more squish npy-dir directories to evaluate.",
    )
    p.add_argument(
        "--model-dir", metavar="PATH", default=None,
        help=(
            "Path to the original HF model directory (contains config.json + tokenizer). "
            "Required for squish npy-dir format. Not required for native MLX dirs."
        ),
    )
    p.add_argument(
        "--tasks", nargs="+", default=_ALL_TASK_NAMES, metavar="TASK",
        choices=_ALL_TASK_NAMES,
        help=f"Tasks to run (default: all 6). Choices: {_ALL_TASK_NAMES}.",
    )
    p.add_argument(
        "--limit", type=int, default=500, metavar="N",
        help="lm_eval sample limit per task (default: 500 for a fast estimate).",
    )
    p.add_argument(
        "--batch-size", type=int, default=4, metavar="N",
        help="mlx_lm evaluate --batch-size (default: 4).",
    )
    p.add_argument(
        "--skip-cache-build", action="store_true",
        help=(
            "Do not call load_from_npy_dir; assume squish_4bit/ or squish_3bit/ "
            "already exists. Raises if no cache is found."
        ),
    )
    p.add_argument(
        "--output-dir", metavar="PATH",
        default=str(_REPO_ROOT / "results"),
        help="Directory where result JSON files are written (default: results/).",
    )
    p.add_argument(
        "--baseline", type=float, default=None, metavar="PCT",
        help=(
            "Expected arc_easy baseline in percent (e.g. 70.6). "
            "When set, prints PASS/FAIL against this baseline."
        ),
    )
    p.add_argument(
        "--threshold", type=float, default=_WAVE41_THRESHOLD_PP, metavar="PP",
        help=(
            f"Tolerance in percentage points for baseline check "
            f"(default: {_WAVE41_THRESHOLD_PP})."
        ),
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output; only print PASS/FAIL and the result path.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns exit code (0 success, 1 error, 2 runtime error)."""
    parser = _build_parser()
    args   = parser.parse_args(argv)

    npy_dirs   = [Path(d).expanduser().resolve() for d in args.npy_dir]
    output_dir = Path(args.output_dir).expanduser().resolve()

    # Validate --model-dir requirement
    model_dir: Path | None = None
    if args.model_dir:
        model_dir = Path(args.model_dir).expanduser().resolve()
        if not model_dir.exists():
            print(f"Error: --model-dir does not exist: {model_dir}", file=sys.stderr)
            return 1

    # If model_dir is not supplied, only native MLX dirs are supported
    if model_dir is None:
        for nd in npy_dirs:
            is_valid, reason = _validate_npy_dir(nd)
            if is_valid and reason == "squish-npy-dir":
                print(
                    f"Error: {nd.name} is a squish npy-dir but --model-dir was not specified.\n"
                    "Provide --model-dir pointing to the original HF model directory.",
                    file=sys.stderr,
                )
                return 1
            # For native MLX dirs, model_dir = npy_dir itself
            model_dir = nd

    failed = 0
    for npy_dir in npy_dirs:
        # For multiple npy-dirs sharing the same base model, model_dir stays constant.
        # If the npy_dir itself is native MLX (no manifest.json), override model_dir.
        if (npy_dir / "config.json").exists() and not (npy_dir / "manifest.json").exists():
            effective_model_dir = npy_dir
        else:
            if model_dir is None:
                print(f"Error: --model-dir required for squish npy-dir {npy_dir}", file=sys.stderr)
                return 1
            effective_model_dir = model_dir

        try:
            result = evaluate_npy_dir(
                npy_dir=npy_dir,
                model_dir=effective_model_dir,
                tasks=args.tasks,
                limit=args.limit,
                skip_cache_build=args.skip_cache_build,
                batch_size=args.batch_size,
                output_dir=output_dir,
                baseline_pct=args.baseline,
                threshold_pp=args.threshold,
                quiet=args.quiet,
            )
        except (FileNotFoundError, RuntimeError, ImportError) as exc:
            print(f"{R}Error{NC} evaluating {npy_dir.name}: {exc}", file=sys.stderr)
            failed += 1
            continue

        baseline_check = result.get("baseline_check", {})
        if baseline_check and not baseline_check.get("passed", True):
            failed += 1

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
