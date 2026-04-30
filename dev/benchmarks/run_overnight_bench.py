#!/usr/bin/env python3
"""
run_overnight_bench.py — Re-squish 5 models at INT4/INT3/INT2 then run
the full lm-eval accuracy suite.  Safe for overnight unattended execution
on Apple M3 16 GB.

Architecture
------------
  Step 1 — Quantize: Create INT4/INT3/INT2 model dirs (one at a time to
            keep Metal heap pressure low).  ALL bit widths use mlx_lm.convert
            (q_bits=N) — always outputs mlx safetensors format required by
            mlx_lm evaluate.  squish compress --int4 outputs its own .npy
            manifest format which lm_eval cannot load, so it is NOT used here.

  Step 2 — Evaluate: Delegates to bench_lmeval_all_models.py (the proven
            evaluation harness). Results are written to --output-dir as
            lmeval_<model>_<ts>.json files.

  Step 3 — Table: Reads all lmeval_*.json result files and builds
            BENCHMARK_TABLE.md with per-task scores and Δ-vs-BF16 rows.

Models
------
  Qwen3-0.6B      INT4  INT3  INT2  +BF16ref
  Llama-3.2-1B    INT4  INT3  INT2  +BF16ref
  gemma-3-1b      INT4  INT3  INT2  +BF16ref
  Qwen2.5-1.5B    INT4  INT3  INT2  +BF16ref
  Qwen3-4B        INT4  INT3  INT2  +BF16ref  (INT4 = 2.0 GB — safe on M3 16 GB)

Usage
-----
  # Full overnight run (re-squish + eval + table):
  python3 dev/benchmarks/run_overnight_bench.py

  # Smoke test — 50 samples per task:
  python3 dev/benchmarks/run_overnight_bench.py --limit 50

  # Squish only — no lm-eval:
  python3 dev/benchmarks/run_overnight_bench.py --squish-only

  # Eval only — models must exist on disk:
  python3 dev/benchmarks/run_overnight_bench.py --eval-only

  # Force re-squish even if output dir exists:
  python3 dev/benchmarks/run_overnight_bench.py --force-squish

  # Dry run — print plan, no execution:
  python3 dev/benchmarks/run_overnight_bench.py --dry-run

  # Target a subset of models:
  python3 dev/benchmarks/run_overnight_bench.py --models Qwen3-0.6B Qwen3-4B

  # Table-only — from existing result JSONs in results/:
  python3 dev/benchmarks/run_overnight_bench.py --table-only

Outputs
-------
  results/overnight_<timestamp>/
    squish_log.txt                 — per-model compress timing + disk sizes
    lmeval_<model>_<ts>.json       — per-model accuracy JSON (written by bench script)
    lmeval_comparison_<ts>.json    — cross-model comparison JSON
    BENCHMARK_TABLE.md             — final markdown table with scores + Δ vs BF16
"""
from __future__ import annotations

import argparse
import gc
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT    = Path(__file__).resolve().parents[2]
_BENCH_SCRIPT = Path(__file__).parent / "bench_lmeval_all_models.py"
_MODELS_ROOT  = Path.home() / "models"
_RESULTS_ROOT = _REPO_ROOT / "results"

# ── ANSI colours ───────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"

# ── OOM guard ────────────────────────────────────────────────────────────────
INT4_OOM_GB = 9.0   # INT4 dirs larger than this cap are known Metal OOM on M3 16 GB

# ── Model plan ───────────────────────────────────────────────────────────────
# (short_display_name, bf16_dir_name, bits_to_squish, include_bf16_eval)
# bits_to_squish: strictly the INT widths to CREATE. bench script handles eval.
# include_bf16_eval: pass to bench script via --include-bf16 flag per-model.
MODEL_PLAN: list[tuple[str, str, list[int | str], bool]] = [
    ("Qwen3-0.6B",   "Qwen3-0.6B-bf16",             [4, 3, 2], True),
    ("Llama-3.2-1B", "Llama-3.2-1B-Instruct-bf16",  [4, 3, 2], True),
    ("gemma-3-1b",   "gemma-3-1b-it-bf16",           [4, 3, 2], True),
    ("Qwen2.5-1.5B", "Qwen2.5-1.5B-Instruct-bf16",  [4, 3, 2], True),
    ("Qwen3-4B",     "Qwen3-4B-bf16",               [4, 3, 2], True),   # INT4 = 2.0 GB — safe on M3 16 GB
    # W103.4d ship gate: Qwen2.5-7B at SQINT2 (Hadamard + NF2 + low-rank residual).
    # bf16 source = ~14 GB; SQINT2 output = ~2 GB. Compress is CPU-bound (NumPy),
    # bypasses the INT4 Metal OOM guard. include_bf16=False because the bf16
    # reference for 7B does not fit in 16 GB at eval time — comparison should be
    # against the published Qwen2.5-7B-Instruct baselines instead.
    ("Qwen2.5-7B",   "Qwen2.5-7B-Instruct-bf16",    ["sqint2"], False),
]

# Model display names as registered in bench_lmeval_all_models.py MODEL_REGISTRY.
# Maps (short_name, bits) → registry_display_name for passing to --models.
_BENCH_MODEL_NAME: dict[tuple[str, int | str], str] = {
    ("Qwen3-0.6B",   "bf16"): "Qwen3-0.6B-bf16",
    ("Qwen3-0.6B",   4):      "Qwen3-0.6B-int4",
    ("Qwen3-0.6B",   3):      "Qwen3-0.6B-int3",
    ("Qwen3-0.6B",   2):      "Qwen3-0.6B-int2",
    ("Llama-3.2-1B", "bf16"): "Llama-3.2-1B-bf16",
    ("Llama-3.2-1B", 4):      "Llama-3.2-1B-int4",
    ("Llama-3.2-1B", 3):      "Llama-3.2-1B-int3",
    ("Llama-3.2-1B", 2):      "Llama-3.2-1B-int2",
    ("gemma-3-1b",   "bf16"): "gemma-3-1b-bf16",
    ("gemma-3-1b",   4):      "gemma-3-1b-int4",
    ("gemma-3-1b",   3):      "gemma-3-1b-int3",
    ("gemma-3-1b",   2):      "gemma-3-1b-int2",
    ("Qwen2.5-1.5B", "bf16"): "Qwen2.5-1.5B-bf16",
    ("Qwen2.5-1.5B", 4):      "Qwen2.5-1.5B-int4",
    ("Qwen2.5-1.5B", 3):      "Qwen2.5-1.5B-int3",
    ("Qwen2.5-1.5B", 2):      "Qwen2.5-1.5B-int2",
    ("Qwen3-4B",     "bf16"): "Qwen3-4B-bf16",
    ("Qwen3-4B",     4):      "Qwen3-4B-int4",
    ("Qwen3-4B",     3):      "Qwen3-4B-int3",
    ("Qwen3-4B",     2):      "Qwen3-4B-int2",
    # W103.4d ship gate
    ("Qwen2.5-7B",   "sqint2"): "Qwen2.5-7B-sqint2",
}

# Standard 6-task lm-eval suite (same as bench_lmeval_all_models.py)
TASK_NAMES = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa", "openbookqa"]

# Display order for the final markdown table
_TABLE_ORDER = [
    "Qwen3-0.6B-bf16",   "Qwen3-0.6B-int4",   "Qwen3-0.6B-int3",   "Qwen3-0.6B-int2",
    "Llama-3.2-1B-bf16", "Llama-3.2-1B-int4", "Llama-3.2-1B-int3", "Llama-3.2-1B-int2",
    "gemma-3-1b-bf16",   "gemma-3-1b-int4",   "gemma-3-1b-int3",   "gemma-3-1b-int2",
    "Qwen2.5-1.5B-bf16", "Qwen2.5-1.5B-int4", "Qwen2.5-1.5B-int3", "Qwen2.5-1.5B-int2",
    "Qwen3-4B-bf16",   "Qwen3-4B-int4",   "Qwen3-4B-int3",   "Qwen3-4B-int2",
    "Qwen2.5-7B-sqint2",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _hdr(msg: str) -> None:
    print(f"\n{W}{'═' * 70}{NC}")
    print(f"{C}  {msg}{NC}")
    print(f"{W}{'═' * 70}{NC}")


def _step(msg: str) -> None:
    print(f"  {C}→{NC}  {msg}")


def _ok(msg: str) -> None:
    print(f"  {G}✓{NC}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {Y}⚠{NC}  {msg}")


def _err(msg: str) -> None:
    print(f"  {R}✗{NC}  {msg}", file=sys.stderr)


def _dir_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def _infer_quant_dir(bf16_dir: Path, bits: int | str) -> Path:
    """Strip -bf16 / -fp16 suffix and append the tier suffix.

    For numeric ``bits`` we append ``-int{bits}``. For the SQINT2 ship-gate
    we use ``-sqint2`` instead (the npy-dir contains a mix of SQINT2 / INT3 /
    INT4 tiers per the W103.3 router; ``-sqint2`` is the dominant tier and
    matches the registry name used by ``bench_lmeval_all_models.py``).
    """
    base = re.sub(r"-(bf16|fp16|[0-9]+bit)(-mlx)?$", "", bf16_dir.name, flags=re.IGNORECASE)
    suffix = f"int{bits}" if isinstance(bits, int) else str(bits)
    return bf16_dir.parent / f"{base}-{suffix}"


def _platform_info() -> dict[str, Any]:
    try:
        import mlx_lm
        mlx_ver = mlx_lm.__version__
    except Exception:
        mlx_ver = "unknown"
    try:
        import lm_eval
        lm_ver = lm_eval.__version__
    except Exception:
        lm_ver = "unknown"
    return {
        "platform":  platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python":    sys.version.split()[0],
        "mlx_lm":    mlx_ver,
        "lm_eval":   lm_ver,
    }


# ── squish step ───────────────────────────────────────────────────────────────

def _is_mlx_format(model_dir: Path) -> bool:
    """True if model_dir contains standard mlx safetensors (loadable by mlx_lm evaluate)."""
    return any(
        (model_dir / f).exists()
        for f in ("model.safetensors", "model.safetensors.index.json")
    )


def squish_model(
    bf16_dir: Path,
    bits: int | str,
    force: bool,
    dry_run: bool,
    log_lines: list[str],
) -> Path | None:
    """
    Quantize bf16_dir to `bits` using mlx_lm.convert — always outputs mlx safetensors
    format, which is required for mlx_lm evaluate / lm_eval harness.

    NOTE  We do NOT use `squish compress --int4` here even though it supports AWQ
    calibration — squish INT4 outputs its own .npy manifest format which mlx_lm evaluate
    cannot load.  mlx_lm.convert(q_bits=4) produces standard safetensors instead.

    INT2 is research-only (4 discrete weight levels).

    The ``bits == "sqint2"`` branch is the W103.4d ship-gate path: it calls
    ``squish compress --format sqint2`` (W103.3) which writes a npy-dir with the
    layer-routed mix of SQINT2 / INT3 / INT4 / passthrough tiers. The npy-dir
    is loaded at eval time via ``compressed_loader._load_sqint2_npy_dir`` which
    replaces ``nn.Linear`` with ``SQINT2Linear`` (W103.4c) and ``INT3Linear``.
    No safetensors cache is materialised — the npy-dir is the canonical
    compressed format.

    Returns the output Path on success, None on failure or if OOM guard triggered.
    """
    out_dir = _infer_quant_dir(bf16_dir, bits)
    label   = (
        f"{bf16_dir.name} → SQINT2" if bits == "sqint2"
        else f"{bf16_dir.name} → INT{bits}"
    )

    # ── SQINT2 ship-gate path (W103.4d) ───────────────────────────────────────
    # Bypass the Metal OOM guard: the SQINT2 compress pipeline is CPU-bound
    # NumPy (Hadamard, randomised SVD, Lloyd-Max) and never materialises the
    # full BF16 weight in Metal. The output npy-dir is ~2 GB for Qwen2.5-7B
    # which is well within the 16 GB device budget.
    if bits == "sqint2":
        if out_dir.exists() and not force:
            manifest = out_dir / "manifest.json"
            tensors  = out_dir / "tensors"
            if manifest.exists() and tensors.exists():
                size_gb = _dir_gb(out_dir)
                _ok(f"{label}: SQINT2 npy-dir exists ({size_gb:.2f} GB) — skipping "
                    f"(use --force-squish to redo)")
                log_lines.append(
                    f"SKIP  SQINT2  {out_dir.name}  {size_gb:.2f}GB  (npy-dir exists)"
                )
                return out_dir
            _warn(f"{label}: dir exists but is missing manifest.json/tensors — overwriting")

        _step(f"Compressing {label}  →  {out_dir.name}")
        if out_dir.exists():
            _step(f"  Removing existing {out_dir.name}")
            if not dry_run:
                shutil.rmtree(str(out_dir), ignore_errors=True)

        if dry_run:
            _ok("  DRY-RUN: would `squish compress --format sqint2`")
            return out_dir

        # squish compress --format sqint2 is exposed by squish/cli.py (W103.3).
        # We invoke it as a subprocess so a SQINT2 compress crash (e.g. OOM
        # mid-Hadamard on a borderline machine) does not kill the eval phase.
        cmd = [
            sys.executable, "-m", "squish.cli",
            "compress",
            "--format", "sqint2",
            "--input",  str(bf16_dir),
            "--output", str(out_dir),
        ]
        _step(f"  $ {' '.join(cmd)}")
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, text=True)
            if proc.returncode != 0:
                _err(f"  squish compress --format sqint2 exited {proc.returncode}")
                log_lines.append(
                    f"FAIL  SQINT2  {out_dir.name}  (rc={proc.returncode})"
                )
                return None
        except Exception as exc:                          # noqa: BLE001
            _err(f"  squish compress failed: {exc}")
            log_lines.append(f"FAIL  SQINT2  {out_dir.name}  ({exc})")
            return None
        finally:
            gc.collect()

        elapsed = time.time() - t0
        if not (out_dir / "manifest.json").exists():
            _err(f"  manifest.json not produced at {out_dir}")
            log_lines.append(f"FAIL  SQINT2  {out_dir.name}  (no manifest.json)")
            return None
        size_gb = _dir_gb(out_dir)
        _ok(f"  Done in {elapsed / 60:.1f} min  →  {out_dir.name}  ({size_gb:.2f} GB)")
        log_lines.append(f"OK    SQINT2  {out_dir.name}  {size_gb:.2f}GB  {elapsed:.0f}s")
        return out_dir

    # OOM guard: skip INT4 for large models (BF16 source > 8 GB → Metal OOM risk on M3 16 GB)
    # Use the BF16 source size as a proxy (INT4 ≈ BF16/4 but Metal loads both during convert)
    if bits == 4:
        bf16_gb = _dir_gb(bf16_dir)
        # Models > 8 GB BF16 would produce >2 GB INT4 and require >12 GB peak during convert
        if bf16_gb > 8.0:
            _warn(f"{label}: BF16 source is {bf16_gb:.1f} GB — INT4 likely Metal OOM on M3 16 GB. SKIPPING.")
            log_lines.append(f"SKIP  INT4  {out_dir.name}  (OOM guard: bf16={bf16_gb:.1f}GB)")
            return None

    # Skip if already exists in mlx format (not squish format) and not forced
    if out_dir.exists() and not force:
        if _is_mlx_format(out_dir):
            size_gb = _dir_gb(out_dir)
            _ok(f"{label}: mlx-format dir exists ({size_gb:.2f} GB) — skipping (use --force-squish to redo)")
            log_lines.append(f"SKIP  INT{bits}  {out_dir.name}  {size_gb:.2f}GB  (mlx-format exists)")
            return out_dir
        else:
            _warn(f"{label}: dir exists but is NOT mlx-format (squish proprietary) — will overwrite with mlx-format for eval compatibility")

    _step(f"Quantizing {label}  →  {out_dir.name}")

    if out_dir.exists():
        _step(f"  Removing existing {out_dir.name}")
        if not dry_run:
            shutil.rmtree(str(out_dir), ignore_errors=True)

    if dry_run:
        _ok(f"  DRY-RUN: would mlx_lm.convert q_bits={bits}")
        return out_dir

    q_group_size = {4: 64, 3: 32, 2: 64}.get(bits, 64)

    if bits == 2:
        _warn(
            "  INT2: research-only — 4 discrete weight levels.  "
            "Expect incoherent output on models < ~30B params."
        )

    _step(f"  mlx_lm.convert(q_bits={bits}, q_group_size={q_group_size})")

    t0 = time.time()
    try:
        import mlx_lm
        mlx_lm.convert(
            hf_path=str(bf16_dir),
            mlx_path=str(out_dir),
            quantize=True,
            q_bits=bits,
            q_group_size=q_group_size,
        )
    except Exception as exc:
        _err(f"  mlx_lm.convert q_bits={bits} failed: {exc}")
        log_lines.append(f"FAIL  INT{bits}  {out_dir.name}  ({exc})")
        return None
    finally:
        gc.collect()  # release Metal buffers before next model

    elapsed = time.time() - t0

    if not out_dir.exists() or not _is_mlx_format(out_dir):
        _err(f"  mlx-format output not found after convert: {out_dir}")
        log_lines.append(f"FAIL  INT{bits}  {out_dir.name}  (mlx-format not created)")
        return None

    size_gb = _dir_gb(out_dir)
    _ok(f"  Done in {elapsed / 60:.1f} min  →  {out_dir.name}  ({size_gb:.2f} GB)")
    log_lines.append(f"OK    INT{bits}  {out_dir.name}  {size_gb:.2f}GB  {elapsed:.0f}s")
    return out_dir


# ── lm-eval delegation ────────────────────────────────────────────────────────

def run_lmeval_for_models(
    bench_model_names: list[str],
    limit: int,
    models_root: Path,
    out_dir: Path,
    skip_existing: bool,
    dry_run: bool,
    gen_sanity: bool,
) -> bool:
    """
    Delegate to bench_lmeval_all_models.py for evaluation.
    bench_model_names: registry names like ["Qwen3-0.6B-int4", "Qwen3-0.6B-bf16", ...]
    Returns True on success.
    """
    if not _BENCH_SCRIPT.exists():
        _err(f"bench_lmeval_all_models.py not found at {_BENCH_SCRIPT}")
        return False

    # Separate BF16 names (need --include-bf16) from quantised names
    bf16_names  = [n for n in bench_model_names if "bf16" in n]
    quant_names = [n for n in bench_model_names if "bf16" not in n]

    def _run_batch(names: list[str], extra_flags: list[str]) -> bool:
        if not names:
            return True
        cmd = [
            sys.executable, str(_BENCH_SCRIPT),
            "--models",      *names,
            "--limit",       str(limit),
            "--output-dir",  str(out_dir),
            "--models-root", str(models_root),
        ] + extra_flags

        if skip_existing:
            cmd.append("--skip-existing")
        if gen_sanity:
            cmd.append("--gen-sanity")

        print(f"\n  {D}$ {' '.join(cmd)}{NC}")
        if dry_run:
            _ok(f"  DRY-RUN: would run bench for {', '.join(names)}")
            return True

        proc = subprocess.run(cmd, text=True)
        if proc.returncode != 0:
            _err(f"  bench_lmeval_all_models.py exited {proc.returncode}")
            return False
        return True

    ok = True

    # Quantised models first (each runs in its own subprocess per-task inside the bench script)
    if quant_names:
        _step(f"Running lm-eval for {len(quant_names)} quantised model(s): {', '.join(quant_names)}")
        ok = _run_batch(quant_names, []) and ok

    # BF16 references (slow but accurate reference point)
    if bf16_names:
        _step(f"Running lm-eval for {len(bf16_names)} BF16 reference model(s): {', '.join(bf16_names)}")
        ok = _run_batch(bf16_names, ["--include-bf16"]) and ok

    return ok


# ── result loader ─────────────────────────────────────────────────────────────

def load_scores_from_dir(results_dir: Path) -> dict[str, dict[str, float]]:
    """
    Load all lmeval_*.json files from results_dir and extract scores dict.
    The bench script writes files with schema: {"model": str, "scores": {task: pct}}.
    Returns {model_name: {task: score_pct}}.
    """
    scores: dict[str, dict[str, float]] = {}
    for jf in sorted(results_dir.glob("lmeval_*.json")):
        try:
            d = json.loads(jf.read_text())
        except Exception:
            continue
        model = d.get("model")
        s     = d.get("scores")
        if model and isinstance(s, dict) and s:
            # Keep only the most recent file per model (glob is sorted by name → ts)
            scores[model] = s
    return scores


# ── table builder ─────────────────────────────────────────────────────────────

def build_comparison_table(
    all_scores:    dict[str, dict[str, float]],
    platform_info: dict[str, Any],
    limit:         int,
    out_dir:       Path,
) -> Path:
    """Build BENCHMARK_TABLE.md with per-task accuracy + Δ-vs-BF16 rows."""

    ordered   = [m for m in _TABLE_ORDER if m in all_scores]
    remaining = [m for m in sorted(all_scores) if m not in ordered]
    ordered  += remaining

    now        = datetime.now().strftime("%Y-%m-%d %H:%M")
    mlx_ver    = platform_info.get("mlx_lm", "?")
    lm_ver     = platform_info.get("lm_eval", "?")
    limit_note = f"limit={limit} samples/task"

    lines: list[str] = []
    lines += [
        f"# Squish Quantization Accuracy Benchmark ({limit_note})",
        "",
        f"*Platform: Apple M3 · 16 GB UMA · mlx-lm {mlx_ver} · lm-eval {lm_ver} · {now}*",
        "",
        "> **Key:** INT2 ⚠ = research-only (4 discrete weight levels; expect incoherent output on <30B params).  ",
        "> INT3 = experimental.  INT4 = production baseline.",
        "",
    ]

    # ── header ────────────────────────────────────────────────────────────────
    col_w = 13
    task_hdr = " | ".join(f"{t:<{col_w}}" for t in TASK_NAMES)
    sep_cols = " | ".join("-" * col_w for _ in TASK_NAMES)
    lines.append(f"| {'Model':<32} | {task_hdr} | {'Avg':>6} |")
    lines.append(f"| {'-'*32} | {sep_cols} | {'------':>6} |")

    # Pre-index BF16 scores for delta computation
    bf16_ref: dict[str, dict[str, float]] = {}
    for mname, sc in all_scores.items():
        if mname.endswith("-bf16"):
            family = mname[:-5]  # strip -bf16
            bf16_ref[family] = sc

    for mname in ordered:
        sc = all_scores[mname]
        row = [sc.get(t) for t in TASK_NAMES]
        valid = [v for v in row if v is not None]
        avg   = sum(valid) / len(valid) if valid else None

        cells = " | ".join(
            f"{v:.1f}%{'':<{col_w - 6}}" if v is not None else f"{'—':<{col_w}}"
            for v in row
        )
        avg_s = f"{avg:.1f}%" if avg is not None else "—"
        lines.append(f"| {mname:<32} | {cells} | {avg_s:>6} |")

        # Δ vs BF16 row (only for quantised variants)
        family   = re.sub(r"-int[0-9]+$", "", mname)
        ref      = bf16_ref.get(family)
        if ref and not mname.endswith("-bf16"):
            deltas = []
            for t in TASK_NAMES:
                s = sc.get(t)
                r = ref.get(t)
                if s is not None and r is not None:
                    deltas.append(f"({s - r:+.1f}%)")
                else:
                    deltas.append("(—)")
            valid_d = [sc.get(t, 0) - ref.get(t, 0)
                       for t in TASK_NAMES if sc.get(t) is not None and ref.get(t) is not None]
            avg_d   = sum(valid_d) / len(valid_d) if valid_d else None
            avg_ds  = f"({avg_d:+.1f}%)" if avg_d is not None else "(—)"

            dcells = " | ".join(f"{d:<{col_w}}" for d in deltas)
            lines.append(f"| {'  Δ vs BF16':<32} | {dcells} | {avg_ds:>6} |")

    lines += [
        "",
        "## Methodology",
        "- Tasks: arc_easy · arc_challenge · hellaswag · winogrande · piqa · openbookqa",
        "- Metrics: `acc_norm,none` for arc/hellaswag/piqa/openbookqa; `acc,none` for winogrande.",
        f"- Sample limit: {limit} samples per task (not full dataset; use for relative comparisons).",
        "- Backend: `mlx_lm evaluate` (Apple Metal, direct model load — not via squish server).",
        "- INT4: `mlx_lm.convert q_bits=4 q_group_size=64` (mlx safetensors format).",
        "- INT3: `mlx_lm.convert q_bits=3 q_group_size=32` (mlx safetensors format).",
        "- INT2: `mlx_lm.convert q_bits=2 q_group_size=64` (research only).",
        "",
    ]

    md_path = out_dir / "BENCHMARK_TABLE.md"
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines))
    return md_path


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overnight re-squish + lm-eval accuracy benchmark for 5 models × INT4/3/2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python3 dev/benchmarks/run_overnight_bench.py                    # full run\n"
            "  python3 dev/benchmarks/run_overnight_bench.py --limit 50         # smoke test\n"
            "  python3 dev/benchmarks/run_overnight_bench.py --squish-only      # quantize only\n"
            "  python3 dev/benchmarks/run_overnight_bench.py --eval-only        # eval only\n"
            "  python3 dev/benchmarks/run_overnight_bench.py --table-only       # rebuild table\n"
            "  python3 dev/benchmarks/run_overnight_bench.py --dry-run          # preview plan\n"
        ),
    )
    ap.add_argument("--limit",         type=int,  default=500,  help="lm-eval samples per task (default 500)")
    ap.add_argument("--squish-only",   action="store_true",     help="Quantize only — skip lm-eval")
    ap.add_argument("--eval-only",     action="store_true",     help="lm-eval only — skip quantization")
    ap.add_argument("--table-only",    action="store_true",     help="Rebuild table from existing result JSONs")
    ap.add_argument("--force-squish",  action="store_true",     help="Re-quantize even if output dir exists")
    ap.add_argument("--skip-existing", action="store_true",     help="Skip lm-eval for models with existing results")
    ap.add_argument("--gen-sanity",    action="store_true",     help="Run generation sanity check before lm-eval")
    ap.add_argument("--dry-run",       action="store_true",     help="Print plan without executing")
    ap.add_argument("--models",        nargs="+", default=None, help="Subset of model family names (e.g. Qwen3-0.6B Qwen3-4B)")
    ap.add_argument(
        "--bits",
        nargs="+",
        default=None,
        help="Subset of tiers: any of 2 3 4 sqint2 (mix freely; e.g. --bits 4 sqint2)",
    )
    ap.add_argument("--models-root",   type=Path, default=_MODELS_ROOT, help=f"Models root directory (default: {_MODELS_ROOT})")
    ap.add_argument("--results-dir",   type=Path, default=None, help="Output directory (default: results/overnight_<ts>/)")
    args = ap.parse_args()

    # ── validate & filter plan ────────────────────────────────────────────────
    plan = list(MODEL_PLAN)
    if args.models:
        valid_names = {n for n, *_ in MODEL_PLAN}
        unknown = set(args.models) - valid_names
        if unknown:
            print(f"{R}Unknown model(s): {', '.join(sorted(unknown))}{NC}")
            print(f"Available: {', '.join(n for n, *_ in MODEL_PLAN)}")
            sys.exit(1)
        plan = [row for row in plan if row[0] in args.models]

    if args.bits:
        # Accept "2"/"3"/"4" (legacy ints) and "sqint2" (W103.4d sentinel) freely.
        wanted: set = set()
        for raw in args.bits:
            if isinstance(raw, str) and raw.isdigit():
                wanted.add(int(raw))
            elif isinstance(raw, int):
                wanted.add(raw)
            else:
                wanted.add(raw)              # "sqint2" or anything future
        unknown = wanted - {2, 3, 4, "sqint2"}
        if unknown:
            print(f"{R}Unknown --bits value(s): {sorted(unknown, key=str)}{NC}")
            print("Allowed: 2 3 4 sqint2")
            sys.exit(1)
        plan = [
            (name, bf16, [b for b in bits if b in wanted], beval)
            for name, bf16, bits, beval in plan
        ]
        plan = [row for row in plan if row[2]]

    ts      = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = args.results_dir or (_RESULTS_ROOT / f"overnight_{ts}")

    plat = _platform_info()

    # ── table-only shortcut ───────────────────────────────────────────────────
    if args.table_only:
        search_dirs = [out_dir, _RESULTS_ROOT]
        all_scores  = {}
        for d in search_dirs:
            if d.exists():
                all_scores.update(load_scores_from_dir(d))
        if not all_scores:
            _err("No lmeval_*.json result files found. Run without --table-only first.")
            sys.exit(1)
        _hdr("Building comparison table from existing results")
        md_path = build_comparison_table(all_scores, plat, args.limit, out_dir)
        _ok(f"Table written: {md_path}")
        return

    # ── header ────────────────────────────────────────────────────────────────
    _hdr(f"Overnight Benchmark  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Models root:    {args.models_root}")
    print(f"  Results dir:    {out_dir}")
    print(f"  lm-eval limit:  {args.limit} samples/task")
    print(f"  DRY RUN:        {args.dry_run}")
    print(f"  mlx-lm:         {plat['mlx_lm']}")
    print(f"  lm-eval:        {plat['lm_eval']}")
    print(f"\n  Plan:")
    for name, bf16_name, bits_list, run_bf16 in plan:
        bits_s = " + ".join(
            (f"INT{b}" if isinstance(b, int) else b.upper()) for b in bits_list
        )
        bf16_s  = " + BF16" if run_bf16 else ""
        bf16_ok = "✓" if (args.models_root / bf16_name).exists() else "✗ MISSING"
        print(f"    {C}{name:<20}{NC}  {bits_s:<18}{bf16_s}   [{bf16_ok}]")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    suite_t0   = time.time()
    log_lines: list[str] = []

    # ── Step 1: Quantize ─────────────────────────────────────────────────────
    if not args.eval_only:
        _hdr("Step 1 — Quantize")
        for name, bf16_name, bits_list, _beval in plan:
            bf16_dir = args.models_root / bf16_name
            if not bf16_dir.exists():
                _warn(f"BF16 source missing: {bf16_dir} — SKIPPING {name}")
                log_lines.append(f"SKIP  bf16-source  {bf16_name}  (not found)")
                continue

            _step(f"{name}  ({_dir_gb(bf16_dir):.2f} GB BF16)")
            for bits in bits_list:
                squish_model(bf16_dir, bits, args.force_squish, args.dry_run, log_lines)

        if log_lines and not args.dry_run:
            (out_dir / "squish_log.txt").write_text("\n".join(log_lines) + "\n")
            _ok(f"Squish log: {out_dir / 'squish_log.txt'}")

    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    if not args.squish_only:
        _hdr("Step 2 — lm-eval (delegating to bench_lmeval_all_models.py)")

        bench_names: list[str] = []
        for name, bf16_name, bits_list, run_bf16 in plan:
            if run_bf16:
                bname = _BENCH_MODEL_NAME.get((name, "bf16"))
                if bname:
                    bench_names.append(bname)
            for bits in bits_list:
                bname = _BENCH_MODEL_NAME.get((name, bits))
                if bname:
                    # Only include if dir exists
                    bf16_dir = args.models_root / bf16_name
                    q_dir    = _infer_quant_dir(bf16_dir, bits)
                    if q_dir.exists() or args.dry_run:
                        bench_names.append(bname)
                    else:
                        _warn(f"  {bname}: no dir at {q_dir} — skipping eval")

        if bench_names:
            run_lmeval_for_models(
                bench_model_names = bench_names,
                limit             = args.limit,
                models_root       = args.models_root,
                out_dir           = out_dir,
                skip_existing     = args.skip_existing,
                dry_run           = args.dry_run,
                gen_sanity        = args.gen_sanity,
            )
        else:
            _warn("No models to evaluate.")

    # ── Step 3: Table ─────────────────────────────────────────────────────────
    if not args.squish_only and not args.dry_run:
        _hdr("Step 3 — Build comparison table")

        all_scores = load_scores_from_dir(out_dir)
        # Also pull in historical results from top-level results/ as supplementary
        for jf in sorted(_RESULTS_ROOT.glob("lmeval_*.json")):
            if jf.parent == out_dir:
                continue
            try:
                d = json.loads(jf.read_text())
                m = d.get("model"); s = d.get("scores")
                if m and s and m not in all_scores:
                    all_scores[m] = s
            except Exception:
                pass

        if all_scores:
            md_path = build_comparison_table(all_scores, plat, args.limit, out_dir)
            _ok(f"Table: {md_path}")
        else:
            _warn("No scores loaded — table not built.")

    # ── Done ──────────────────────────────────────────────────────────────────
    elapsed = time.time() - suite_t0
    _hdr(f"DONE — {elapsed / 3600:.2f} h  ({elapsed / 60:.0f} min) elapsed")
    if not args.dry_run:
        print(f"  Results: {out_dir}")
        all_scores = load_scores_from_dir(out_dir)
        if all_scores:
            print()
            for mname in [m for m in _TABLE_ORDER if m in all_scores] + \
                         [m for m in sorted(all_scores) if m not in _TABLE_ORDER]:
                sc    = all_scores[mname]
                valid = [v for v in sc.values() if v is not None]
                avg   = sum(valid) / len(valid) if valid else 0.0
                print(f"    {mname:<40}  avg {avg:.1f}%")


if __name__ == "__main__":
    main()
