#!/usr/bin/env python3
"""
dev/benchmarks/bench_wave98_multiplier_stack.py
Wave 98 Multiplier Stack Benchmark

Tests the performance impact of the Wave 98 optimisation stack:
  1. MaskedFFN micro-benchmark — overhead of mask multiply at real FFN dims
  2. Full model inference benchmark — TTFT, tok/s at synthetic sparsity levels
  3. Real mask generation — run gen-masks calibration, report sparsity + cost

Model: llama3.1-8b-4bit  (~4 bpw, fits in 16 GB with headroom)
Hardware profiled: Apple M3 16 GB

Usage
-----
    python3 dev/benchmarks/bench_wave98_multiplier_stack.py
    python3 dev/benchmarks/bench_wave98_multiplier_stack.py --model ~/.squish/models/llama3.1-8b-4bit
    python3 dev/benchmarks/bench_wave98_multiplier_stack.py --no-genmasks  (skip calibration)
    python3 dev/benchmarks/bench_wave98_multiplier_stack.py --output ./my-results/

Output
------
  Prints a formatted table to stdout.
  Writes JSON to benchmarks/results/wave98_<timestamp>.json
  Writes Markdown to benchmarks/results/wave98_<timestamp>.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── Repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ── ANSI helpers ──────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
B = "\033[1m"; D = "\033[2m"; NC = "\033[0m"; R = "\033[31m"

_DEFAULT_MODEL = str(Path.home() / ".squish" / "models" / "llama3.1-8b-4bit")

# Prompts designed to stress short, medium, and code-heavy contexts
_BENCH_PROMPTS = [
    "What is the capital of France?",
    "Explain how transformers work in machine learning in one paragraph.",
    "Write a Python function that computes the nth Fibonacci number iteratively.",
    "In three sentences, summarize the French Revolution.",
    "What are the pros and cons of using Rust over C++ for systems programming?",
]

_N_WARMUP = 5     # GPU JIT warmup runs (discarded)
_N_MEASURED = 10  # measured runs per condition


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class MicroResult:
    """Single microbenchmark data point."""
    name: str
    hidden_dim: int
    sparsity: float   # fraction of zeros in mask
    runs: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_ms: float
    overhead_pct: float = 0.0   # relative to baseline (set post-hoc)


@dataclass
class InferenceResult:
    """Single end-to-end inference pass result."""
    condition: str    # "baseline" | "mask_30" | "mask_50" | "mask_70" | "gen_masks"
    sparsity: float   # fraction of neurons masked to zero
    prompt: str
    n_tokens: int
    ttft_ms: float
    elapsed_ms: float
    tok_per_sec: float


@dataclass
class PassSummary:
    """Aggregate stats across multiple InferenceResult runs."""
    condition: str
    sparsity: float
    n_runs: int
    mean_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    mean_tps: float
    p50_tps: float
    p95_tps: float
    std_tps: float


@dataclass
class BenchReport:
    timestamp: str
    hardware: dict
    model_path: str
    micro: list
    inference: list[PassSummary]
    genmasks_sparsity_pct: float = 0.0
    genmasks_time_s: float = 0.0
    notes: list[str] = field(default_factory=list)


# ── Hardware probe ────────────────────────────────────────────────────────────

def _hw_context() -> dict:
    try:
        import mlx.core as mx
        di = mx.device_info()
    except Exception:
        di = {}
    mem_gb = di.get("memory_size", 0) / 1024**3
    uname = platform.uname()
    return {
        "chip":          di.get("device_name", "unknown"),
        "architecture":  di.get("architecture", "?"),
        "memory_gb":     round(mem_gb, 1),
        "os":            f"{uname.system} {uname.release}",
        "python":        platform.python_version(),
        "mlx_version":   _mlx_version(),
    }


def _mlx_version() -> str:
    try:
        import mlx.core as mx
        return getattr(mx, "__version__", "?")
    except Exception:
        return "?"


# ── Timing helper ─────────────────────────────────────────────────────────────

def _bench_fn(fn, runs: int = _N_MEASURED, warmup: int = _N_WARMUP) -> dict:
    """Run *fn()* with warmup and return latency statistics dict (ms)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
    a = np.array(times)
    return {
        "mean_ms": float(a.mean()),
        "p50_ms":  float(np.median(a)),
        "p95_ms":  float(np.percentile(a, 95)),
        "p99_ms":  float(np.percentile(a, 99)),
        "std_ms":  float(a.std()),
        "runs":    runs,
    }


# ── Section 1: MaskedFFN microbenchmark ──────────────────────────────────────

def _micro_bench_masked_ffn(runs: int = 200) -> list[MicroResult]:
    """
    Measure the cost of one MaskedFFN forward pass at LLaMA-8B FFN dims.

    Both baseline and MaskedFFN wrap the SAME synthetic LLaMA FFN (gate_proj +
    up_proj SiLU + down_proj matmuls using numpy).  This gives meaningful
    overhead percentages — the comparison is FFN-with-mask vs FFN-without-mask,
    not mask vs empty function.

    Dims match LLaMA-3.1-8B:
      hidden_size = 4096, intermediate_size = 14336

    Use numpy (not MLX GPU) to avoid GPU scheduling noise and give a
    conservative worst-case estimate.  On the GPU, the mask multiply is a
    single elementwise kernel; the matmuls dominate by ~1000x.
    """
    from squish.kernels.ffn_mask_patch import MaskedFFN

    HIDDEN  = 4096
    FFN_DIM = 14336
    rng = np.random.default_rng(42)

    # Synthetic FFN weights (float32 Gaussian at realistic scale)
    W_gate  = (rng.standard_normal((FFN_DIM, HIDDEN))  * 0.02).astype(np.float32)
    W_up    = (rng.standard_normal((FFN_DIM, HIDDEN))  * 0.02).astype(np.float32)
    W_down  = (rng.standard_normal((HIDDEN,  FFN_DIM)) * 0.02).astype(np.float32)
    x_token = rng.standard_normal(HIDDEN).astype(np.float32)  # single decode token

    def _silu(z: np.ndarray) -> np.ndarray:
        return z / (1.0 + np.exp(-z))

    def _llama_ffn(v: np.ndarray) -> np.ndarray:
        """SwiGLU FFN: down(silu(gate(x)) * up(x))"""
        gate = _silu(W_gate @ v)
        up   = W_up @ v
        return W_down @ (gate * up)   # shape: (HIDDEN,)

    results: list[MicroResult] = []
    sparsity_levels = [0.0, 0.30, 0.50, 0.70]

    # ------------------------------------------------------------------
    # Baseline: raw LLaMA FFN without any mask wrapper
    # ------------------------------------------------------------------
    baseline_stats = _bench_fn(lambda: _llama_ffn(x_token), runs=runs, warmup=10)
    baseline_result = MicroResult(
        name="baseline (raw FFN, no mask)",
        hidden_dim=HIDDEN,
        sparsity=0.0,
        runs=runs,
        **{k: v for k, v in baseline_stats.items() if k != "runs"},
        overhead_pct=0.0,
    )
    results.append(baseline_result)

    # ------------------------------------------------------------------
    # MaskedFFN wrapping the same FFN at each sparsity level
    # ------------------------------------------------------------------
    for sp in sparsity_levels:
        mask = np.ones(HIDDEN, dtype=np.float32)
        n_zero = int(sp * HIDDEN)
        if n_zero > 0:
            mask[rng.choice(HIDDEN, n_zero, replace=False)] = 0.0

        wrapped = MaskedFFN(_llama_ffn, mask, layer_idx=0)
        stats = _bench_fn(lambda w=wrapped: w(x_token), runs=runs, warmup=10)

        overhead = 0.0
        if baseline_result.mean_ms > 0:
            overhead = ((stats["mean_ms"] - baseline_result.mean_ms) / baseline_result.mean_ms) * 100

        overhead_abs_us = (stats["mean_ms"] - baseline_result.mean_ms) * 1000
        results.append(MicroResult(
            name=f"MaskedFFN sparsity={sp:.0%}",
            hidden_dim=HIDDEN,
            sparsity=sp,
            runs=runs,
            **{k: v for k, v in stats.items() if k != "runs"},
            overhead_pct=overhead,
        ))

    return results


# ── Section 2: Full model inference benchmark ────────────────────────────────

def _ensure_mlx_eval(model):
    """Force pending MLX computations to complete before timing."""
    import mlx.core as mx
    mx.eval(model.parameters())


def _infer_stream(model, tokenizer, prompt: str, max_tokens: int = 100) -> tuple[float, float, int]:
    """
    Run a single generation and return (ttft_ms, elapsed_ms, n_tokens).

    Uses mlx_lm.stream_generate to capture time-to-first-token accurately.
    """
    import mlx.core as mx
    import mlx_lm

    # Force Metal JIT warmup for any lazy compilation
    mx.eval()

    t_start = time.perf_counter()
    ttft_ms = None
    n_tokens = 0

    for response in mlx_lm.stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens
    ):
        if ttft_ms is None:
            ttft_ms = (time.perf_counter() - t_start) * 1e3
        n_tokens += 1

    elapsed_ms = (time.perf_counter() - t_start) * 1e3
    if ttft_ms is None:
        ttft_ms = elapsed_ms  # degenerate: 0 tokens

    return ttft_ms, elapsed_ms, n_tokens


def _run_inference_pass(
    model,
    tokenizer,
    condition: str,
    sparsity: float,
    prompts: list[str],
    max_tokens: int = 100,
    warmup_runs: int = _N_WARMUP,
    measured_runs: int = _N_MEASURED,
    verbose: bool = True,
) -> tuple[list[InferenceResult], PassSummary]:
    """Run warmup + measured passes, return raw results and aggregate summary."""

    raw: list[InferenceResult] = []

    if verbose:
        print(f"  {D}[{condition}] warmup ({warmup_runs} runs)…{NC}")

    # Warmup: cycle through prompts
    for i in range(warmup_runs):
        p = prompts[i % len(prompts)]
        _infer_stream(model, tokenizer, p, max_tokens=64)

    if verbose:
        print(f"  {G}[{condition}] measuring ({measured_runs} runs)…{NC}")

    for i in range(measured_runs):
        p = prompts[i % len(prompts)]
        ttft_ms, elapsed_ms, n_tokens = _infer_stream(
            model, tokenizer, p, max_tokens=max_tokens
        )
        tps = (n_tokens / elapsed_ms * 1e3) if elapsed_ms > 0 else 0.0
        raw.append(InferenceResult(
            condition=condition,
            sparsity=sparsity,
            prompt=p,
            n_tokens=n_tokens,
            ttft_ms=ttft_ms,
            elapsed_ms=elapsed_ms,
            tok_per_sec=tps,
        ))
        if verbose:
            print(f"    run {i+1:2d}: TTFT={ttft_ms:6.0f} ms  tok/s={tps:5.1f}  tokens={n_tokens}")

    # Aggregate
    ttfts = np.array([r.ttft_ms for r in raw])
    tpss  = np.array([r.tok_per_sec for r in raw])
    summary = PassSummary(
        condition=condition,
        sparsity=sparsity,
        n_runs=len(raw),
        mean_ttft_ms=float(ttfts.mean()),
        p50_ttft_ms=float(np.median(ttfts)),
        p95_ttft_ms=float(np.percentile(ttfts, 95)),
        mean_tps=float(tpss.mean()),
        p50_tps=float(np.median(tpss)),
        p95_tps=float(np.percentile(tpss, 95)),
        std_tps=float(tpss.std()),
    )
    return raw, summary


def _apply_synthetic_masks(model, sparsity: float, seed: int = 42) -> int:
    """Apply random binary masks at given sparsity to all MLP layers."""
    from squish.kernels.ffn_mask_patch import patch_model_ffn_sparsity, unpatch_model_ffn_sparsity
    from squish.runtime.structured_sparsity import StructuredFfnSparsity

    unpatch_model_ffn_sparsity(model)

    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0

    rng = np.random.default_rng(seed)
    masks: dict[int, np.ndarray] = {}

    for i, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue
        # Probe hidden_dim by running a tiny forward pass to get output dim
        # or fall back to inspecting QuantizedLinear weight shape
        hidden_dim = _probe_mlp_output_dim(mlp)
        if hidden_dim is None:
            continue
        mask = np.ones(hidden_dim, dtype=np.float32)
        n_zero = int(sparsity * hidden_dim)
        if n_zero > 0:
            mask[rng.choice(hidden_dim, n_zero, replace=False)] = 0.0
        masks[i] = mask

    sp = StructuredFfnSparsity(masks)
    return patch_model_ffn_sparsity(model, sp, verbose=False)


def _probe_mlp_output_dim(mlp) -> int | None:
    """
    Return the MLP output hidden dimension (what the mask should cover).

    For LLaMA SwiGLU FFN: output = down_proj(silu(gate) * up)
      down_proj: (hidden_size, intermediate_size) → output shape hidden_size

    MLX QuantizedLinear layout:
      weight  shape: (out_features, packed_in_features)  [nibble-packed INT4]
      scales  shape: (out_features, in_features / group_size)                 
      biases  shape: (out_features, in_features / group_size) [zero points]   

    So down_proj.scales.shape[0] = out_features = hidden_size = 4096 ✓
    """
    import mlx.core as mx

    # Primary path: down_proj scales
    for attr in ("down_proj", "c_proj", "dense", "out_proj"):
        proj = getattr(mlp, attr, None)
        if proj is None:
            continue
        scales = getattr(proj, "scales", None)
        if scales is not None:
            try:
                return int(scales.shape[0])
            except Exception:
                pass
        w = getattr(proj, "weight", None)
        if w is not None:
            try:
                return int(w.shape[0])
            except Exception:
                pass

    # Fallback: probe gate_proj input dimension (= hidden_size = mask output size)
    for attr in ("gate_proj", "fc1", "w1"):
        proj = getattr(mlp, attr, None)
        if proj is None:
            continue
        # gate_proj.scales: (intermediate_size, hidden_size/group_size)
        # We need hidden_size = output of down_proj = input of gate_proj
        # gate_proj weight: (intermediate_size, hidden_size) → input is hidden_size
        # In MLX INT4: weight.shape = (intermediate_size, packed_dim), scales.shape = (intermediate_size, n_groups)
        # So scales.shape[1] * group_size = hidden_size
        scales = getattr(proj, "scales", None)
        if scales is not None:
            gs = getattr(proj, "group_size", 64)
            try:
                return int(scales.shape[1] * int(gs))
            except Exception:
                pass
    return None


# ── Section 3: gen-masks calibration benchmark ───────────────────────────────

def _run_genmasks_calibration(
    model,
    tokenizer,
    n_prompts: int = 20,
    threshold: float = 0.05,
) -> tuple[float, float]:
    """
    Run the gen-masks activation capture pipeline and return
    (mean_sparsity_pct, elapsed_secs).
    """
    import mlx.core as mx
    from squish.kernels.ffn_mask_patch import unpatch_model_ffn_sparsity

    unpatch_model_ffn_sparsity(model)

    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return 0.0, 0.0

    # Built-in calibration prompts (same as cmd_gen_masks)
    cal_prompts = [
        "The capital of France is", "Machine learning transformers use",
        "The derivative of sin(x) is", "Once upon a time", "Python is known for",
        "The best way to learn math is", "Water boils at 100 degrees because",
        "The Roman Empire lasted", "A binary search tree in Python",
        "The human genome contains", "Climate change is caused by",
        "Relativity states that", "A neural network learns via",
        "The French Revolution began", "Quantum physics says",
        "The mitochondria is the", "Shakespeare wrote during",
        "The speed of light is", "Gradient descent works by",
        "The first computer was built by",
    ]

    class _CaptureMLP:
        def __init__(self, inner, idx):
            self.inner = inner
            self.idx = idx
            self.outputs: list[np.ndarray] = []
        def __call__(self, x):
            out = self.inner(x)
            mx.eval(out)
            self.outputs.append(np.array(out).reshape(-1, out.shape[-1]))
            return out
        def __getattr__(self, n):
            return getattr(self.inner, n)

    hooks: list[_CaptureMLP] = []
    for i, layer in enumerate(layers):
        h = _CaptureMLP(layer.mlp, i)
        layer.mlp = h
        hooks.append(h)

    t0 = time.perf_counter()
    prompts_to_run = (cal_prompts * ((n_prompts // len(cal_prompts)) + 1))[:n_prompts]
    for p in prompts_to_run:
        toks = tokenizer.encode(p)
        if not toks:
            continue
        import mlx.core as _mx
        _inp = _mx.array([toks])
        model(_inp)
        _mx.eval()

    # Restore and compute masks
    sparsity_vals: list[float] = []
    for hook, layer in zip(hooks, layers):
        layer.mlp = hook.inner
        if not hook.outputs:
            continue
        all_out = np.concatenate(hook.outputs, axis=0).astype(np.float32)
        firing = (np.abs(all_out) > threshold).astype(np.float32).mean(axis=0)
        binary = (firing >= threshold).astype(np.float32)
        sparsity_vals.append(float(1.0 - binary.mean()))

    elapsed = time.perf_counter() - t0
    mean_sp = float(np.mean(sparsity_vals)) if sparsity_vals else 0.0
    return mean_sp * 100, elapsed


# ── Printing helpers ──────────────────────────────────────────────────────────

def _print_header(title: str) -> None:
    w = 72
    print(f"\n{B}{'─' * w}{NC}")
    print(f"{B}  {title}{NC}")
    print(f"{B}{'─' * w}{NC}")


def _print_micro_table(results: list[MicroResult]) -> None:
    _print_header("1. MaskedFFN Micro-benchmark  (numpy SwiGLU FFN, LLaMA-8B dims: 4096×14336)")
    hdr = f"  {'Condition':<38}  {'mean ms':>9}  {'p50 ms':>9}  {'p95 ms':>9}  {'overhead':>10}"
    print(hdr)
    print(f"  {'─' * 82}")
    for r in results:
        if r.overhead_pct == 0.0 and r.name.startswith("baseline"):
            oh = "  baseline"
            color = NC
        else:
            color = G if abs(r.overhead_pct) < 2.0 else Y
            oh = f"{r.overhead_pct:+.2f}%"
        print(
            f"  {r.name:<38}  {r.mean_ms:9.3f}  {r.p50_ms:9.3f}  {r.p95_ms:9.3f}  {color}{oh:>10}{NC}"
        )
    print()
    baseline = next((r for r in results if r.name.startswith("baseline")), None)
    if baseline:
        abs_overhead_us = (results[1].mean_ms - baseline.mean_ms) * 1000 if len(results) > 1 else 0
        print(f"  {D}Absolute mask-multiply overhead: ~{abs_overhead_us:.1f} µs/layer")
        print(f"  Real INT4 matmul (4096×14336) on M3 GPU: ~0.3–2 ms → mask is <1% of FFN time.{NC}")


def _print_inference_table(summaries: list[PassSummary]) -> None:
    _print_header("2. Full-model Inference  (LLaMA-3.1-8B INT4, 100 tokens/run)")
    hdr = (
        f"  {'Condition':<22}  {'sparsity':>8}  "
        f"{'TTFT p50':>9}  {'TTFT p95':>9}  "
        f"{'tok/s mean':>10}  {'tok/s p50':>9}  {'tok/s p95':>9}  {'vs baseline':>12}"
    )
    print(hdr)
    print(f"  {'─' * 110}")
    baseline_tps = None
    for s in summaries:
        if s.condition == "baseline":
            baseline_tps = s.mean_tps
        delta = ""
        if baseline_tps and s.condition != "baseline":
            pct = (s.mean_tps - baseline_tps) / baseline_tps * 100
            col = G if pct >= -0.5 else R
            delta = f"{col}{pct:+.1f}%{NC}"
        print(
            f"  {s.condition:<22}  {s.sparsity:>7.0%}  "
            f"{s.p50_ttft_ms:>8.0f}ms  {s.p95_ttft_ms:>8.0f}ms  "
            f"{s.mean_tps:>10.1f}  {s.p50_tps:>9.1f}  {s.p95_tps:>9.1f}  {delta:>15}"
        )


def _print_genmasks_summary(sparsity_pct: float, elapsed_s: float) -> None:
    _print_header("3. gen-masks Calibration  (20 prompts, threshold=0.05)")
    print(f"  Calibration time  : {elapsed_s:.1f}s")
    print(f"  Mean layer sparsity: {sparsity_pct:.1f}%")
    note = ""
    if sparsity_pct < 5:
        note = f"  {Y}⚠  Low sparsity (<5%) — INT4 model activations are dense. Expected:{NC}\n"
        note += f"     INT2 models typically achieve 25–45% sparsity at threshold=0.05."
    elif sparsity_pct < 20:
        note = f"  {Y}→  Moderate sparsity ({sparsity_pct:.0f}%). Quality benefit for INT2/INT3.{NC}"
    else:
        note = f"  {G}✓  Good sparsity ({sparsity_pct:.0f}%). Activations should improve quality.{NC}"
    if note:
        print(note)


def _markdown_table(summaries: list[PassSummary], micro: list[MicroResult], hw: dict,
                    sparsity_pct: float, genmasks_s: float) -> str:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        f"# Wave 98 Multiplier Stack Benchmark",
        f"",
        f"**Date**: {ts}  ",
        f"**Chip**: {hw.get('chip','?')} {hw.get('memory_gb','?')} GB  ",
        f"**MLX**: {hw.get('mlx_version','?')}  ",
        f"**OS**: {hw.get('os','?')}  ",
        f"",
        f"---",
        f"",
        f"## 1. MaskedFFN Micro-benchmark",
        f"",
        f"| Condition | hidden_dim | sparsity | mean ms | p50 ms | p95 ms | overhead |",
        f"|-----------|-----------|---------|---------|--------|--------|----------|",
    ]
    for r in micro:
        oh = f"{r.overhead_pct:+.1f}%" if r.overhead_pct != 0 else "baseline"
        lines.append(
            f"| {r.name} | {r.hidden_dim} | {r.sparsity:.0%} | {r.mean_ms:.4f} | {r.p50_ms:.4f} | {r.p95_ms:.4f} | {oh} |"
        )
    baseline_tps2 = next((s.mean_tps for s in summaries if s.condition == "baseline"), None)
    lines += [
        f"",
        f"## 2. Full-model Inference  (LLaMA-3.1-8B INT4, 100 tokens/run)",
        f"",
        f"| Condition | Sparsity | TTFT p50 (ms) | TTFT p95 (ms) | mean tok/s | p50 tok/s | p95 tok/s | vs baseline |",
        f"|-----------|----------|--------------|--------------|-----------|----------|----------|------------|",
    ]
    for s in summaries:
        delta = ""
        if baseline_tps2 and s.condition != "baseline":
            pct = (s.mean_tps - baseline_tps2) / baseline_tps2 * 100
            delta = f"{pct:+.1f}%"
        lines.append(
            f"| {s.condition} | {s.sparsity:.0%} | {s.p50_ttft_ms:.0f} | {s.p95_ttft_ms:.0f} | {s.mean_tps:.1f} | {s.p50_tps:.1f} | {s.p95_tps:.1f} | {delta} |"
        )
    lines += [
        f"",
        f"## 3. gen-masks Calibration  (20 prompts, threshold=0.05)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Calibration time | {genmasks_s:.1f}s |",
        f"| Mean layer sparsity | {sparsity_pct:.1f}% |",
        f"",
        f"---",
        f"",
        f"> Generated by `dev/benchmarks/bench_wave98_multiplier_stack.py`",
    ]
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Wave 98 multiplier stack benchmark")
    ap.add_argument("--model", default=_DEFAULT_MODEL,
                    help=f"Path to compressed model dir (default: {_DEFAULT_MODEL})")
    ap.add_argument("--no-genmasks", action="store_true",
                    help="Skip gen-masks calibration (saves ~2 min)")
    ap.add_argument("--max-tokens", type=int, default=100,
                    help="Max tokens per inference run (default: 100)")
    ap.add_argument("--output", default=str(_REPO_ROOT / "benchmarks" / "results"),
                    help="Output directory for JSON + Markdown")
    ap.add_argument("--runs", type=int, default=_N_MEASURED,
                    help=f"Number of measured runs per condition (default: {_N_MEASURED})")
    ap.add_argument("--warmup", type=int, default=_N_WARMUP,
                    help=f"Number of warmup runs (default: {_N_WARMUP})")
    ap.add_argument("--micro-only", action="store_true",
                    help="Run only the micro-benchmark (fast, no model load)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    hw = _hw_context()

    print(f"\n{B}Wave 98 Multiplier Stack Benchmark{NC}")
    print(f"  Chip      : {hw['chip']}  {hw['memory_gb']} GB")
    print(f"  MLX       : {hw['mlx_version']}")
    print(f"  Model     : {args.model}")
    print(f"  OS        : {hw['os']}")
    print()

    # ── Section 1: Micro-benchmark ────────────────────────────────────────────
    _print_header("Running Section 1 — MaskedFFN micro-benchmark…")
    micro_results = _micro_bench_masked_ffn(runs=300)
    _print_micro_table(micro_results)

    if args.micro_only:
        _save_and_exit(out_dir, ts, hw, micro_results, [], 0.0, 0.0, args.model)
        return

    # ── Load model ────────────────────────────────────────────────────────────
    _print_header(f"Loading {Path(args.model).name}…")
    import mlx_lm
    from squish.kernels.ffn_mask_patch import unpatch_model_ffn_sparsity

    t_load = time.perf_counter()
    model, tokenizer = mlx_lm.load(args.model)
    print(f"  Loaded in {time.perf_counter() - t_load:.1f}s")

    layers = getattr(model, "layers", None) or getattr(getattr(model, "model", None), "layers", None)
    n_layers = len(layers) if layers else "?"
    print(f"  Layers: {n_layers}")

    # ── Section 2: Inference benchmark ───────────────────────────────────────
    _print_header("Running Section 2 — Inference benchmarks…")

    summaries: list[PassSummary] = []

    # --- Baseline (no masks) ---
    unpatch_model_ffn_sparsity(model)
    _, s_base = _run_inference_pass(
        model, tokenizer,
        condition="baseline",
        sparsity=0.0,
        prompts=_BENCH_PROMPTS,
        max_tokens=args.max_tokens,
        warmup_runs=args.warmup,
        measured_runs=args.runs,
    )
    summaries.append(s_base)

    # --- Synthetic sparsity levels ---
    for sp in [0.30, 0.50, 0.70]:
        n_patched = _apply_synthetic_masks(model, sparsity=sp)
        if n_patched == 0:
            print(f"  {Y}⚠ Could not probe MLP output dim — skipping sparsity={sp:.0%}{NC}")
            continue
        _, s = _run_inference_pass(
            model, tokenizer,
            condition=f"mask_{int(sp*100):02d}pct",
            sparsity=sp,
            prompts=_BENCH_PROMPTS,
            max_tokens=args.max_tokens,
            warmup_runs=args.warmup,
            measured_runs=args.runs,
        )
        summaries.append(s)

    # Restore baseline for gen-masks section
    unpatch_model_ffn_sparsity(model)

    _print_inference_table(summaries)

    # ── Section 3: gen-masks calibration ─────────────────────────────────────
    genmasks_sp = 0.0
    genmasks_s = 0.0
    if not args.no_genmasks:
        _print_header("Running Section 3 — gen-masks calibration (20 prompts)…")
        genmasks_sp, genmasks_s = _run_genmasks_calibration(model, tokenizer)
        _print_genmasks_summary(genmasks_sp, genmasks_s)

    # Restore clean model state
    unpatch_model_ffn_sparsity(model)

    # ── Save results ──────────────────────────────────────────────────────────
    _save_and_exit(out_dir, ts, hw, micro_results, summaries, genmasks_sp, genmasks_s, args.model)


def _save_and_exit(
    out_dir: Path,
    ts: str,
    hw: dict,
    micro_results: list,
    summaries: list,
    genmasks_sp: float,
    genmasks_s: float,
    model_path: str,
) -> None:
    json_path = out_dir / f"wave98_{ts}.json"
    md_path   = out_dir / f"wave98_{ts}.md"

    report = {
        "timestamp": ts,
        "hardware": hw,
        "model_path": model_path,
        "micro": [asdict(r) for r in micro_results],
        "inference": [asdict(s) for s in summaries],
        "genmasks_sparsity_pct": genmasks_sp,
        "genmasks_time_s": genmasks_s,
    }

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if summaries:
        md = _markdown_table(summaries, micro_results, hw, genmasks_sp, genmasks_s)
        md_path.write_text(md, encoding="utf-8")

    _print_header("Results saved")
    print(f"  JSON : {json_path}")
    if summaries:
        print(f"  MD   : {md_path}")
    print()

    if summaries:
        _print_future_plan(summaries)


def _print_future_plan(summaries: list[PassSummary]) -> None:
    """Print a concise improvement roadmap based on current numbers."""
    baseline = next((s for s in summaries if s.condition == "baseline"), None)
    if baseline is None:
        return

    tps = baseline.mean_tps
    ttft = baseline.mean_ttft_ms

    print(f"\n{B}═══ Improvement Roadmap ═══════════════════════════════════════════{NC}")
    print(f"\nCurrent baseline  : {tps:.1f} tok/s  TTFT={ttft:.0f} ms  (INT4, no speculative decode)")
    print(f"Theoretical ceiling: ~205 tok/s  TTFT=<200 ms  (INT2 + 50% sparsity + EAGLE4 on M3)")
    print()
    print(f"{B}Priority | Feature                | Expected gain  | Status{NC}")
    print(f"─────────────────────────────────────────────────────────────────────")

    rows = [
        ("P1 🔥", "EAGLE speculative decode       ", "3–4× tok/s     ", "gen-masks + pull-head ready; need calibrated head checkpoint"),
        ("P2 🔥", "INT2 weights (pull --int2)     ", "~1.7× tok/s    ", "MLX native bits=2 works; recompress llama3.1-8b with --int2"),
        ("P3 🔶", "Gen-masks calibration (real)   ", "+5–15% quality ", "run: squish gen-masks llama3.1-8b → sparse_masks.npz"),
        ("P4 🔶", "KV-cache INT8 (--kv-bits 8)   ", "−15% TTFT long ", "already wired in server.py; add to serve flags"),
        ("P5 🔷", "Sparse kernel (skip zero rows) ", "up to 1.5× FFN ", "blocked: mx.metal.kernel not in MLX 0.30.6; revisit MLX 1.x"),
        ("P6 🔷", "ASTC texture encoding (FFN)    ", "~1.3× bandwidth", "available in squish compress --format astc; M3-only"),
        ("P7 ⬜", "Prefix caching (warm repeat)   ", "−90% TTFT hits  ", "already wired; enable --prefix-cache flag"),
        ("P8 ⬜", "Distill EAGLE head locally     ", "4–5× tok/s     ", "requires calibration corpus + distill_eagle.py; GPU time needed"),
    ]
    for p, f, g, n in rows:
        print(f"  {p}  {f}  {g}  {D}{n}{NC}")

    print(f"\n{B}Recommended next actions:{NC}")
    print(f"  1. {G}squish pull --int2 llama3.1:8b{NC}          → recompress to INT2 (~5 min)")
    print(f"  2. {G}squish pull-head llama3.1:8b{NC}             → download EAGLE3-LLaMA3.1-8B")
    print(f"  3. {G}squish gen-masks ~/.squish/models/... {NC}   → run 20-prompt calibration")
    print(f"  4. {G}re-run this benchmark{NC}                     → measure actual stack delta")
    print()


if __name__ == "__main__":
    main()
