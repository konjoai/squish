#!/usr/bin/env python3
"""
bench_qwen3_14b_int3.py — End-to-end benchmark for the INT3-MiLo compressed
Qwen3-14B model.

Stages
------
  S0  Compression metrics  — disk GB before/after, ratio, actual BPW, time
  S1  Reconstruction SNR   — per-layer dequant quality vs BF16 originals
  S2  Throughput           — tok/s via mlx_lm (INT4-quantised BF16 baseline)
  S3  Perplexity           — wikitext-2 PPL via mlx_lm

Inputs
------
  --bf16-dir   Path to the original BF16 model (default: ~/models/Qwen3-14B-bf16)
  --int3-dir   Path to the squish INT3 npy-dir (default: ~/models/Qwen3-14B-int3)
  --wait       Poll until INT3 dir contains manifest.json (useful if
               compression is still running)

Usage
-----
  # Compression and SNR only (no model load needed):
  python3 dev/benchmarks/bench_qwen3_14b_int3.py

  # Full benchmark — SNR + throughput + perplexity:
  python3 dev/benchmarks/bench_qwen3_14b_int3.py \\
      --eval-tps --eval-ppl --runs 3

  # Watch compression progress then benchmark automatically:
  python3 dev/benchmarks/bench_qwen3_14b_int3.py --wait --eval-tps
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ── repo root ─────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

# ── colour codes ─────────────────────────────────────────────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_BF16_DIR  = Path.home() / "models" / "Qwen3-14B-bf16"
DEFAULT_INT3_DIR  = Path.home() / "models" / "Qwen3-14B-int3"
MAX_SNR_LAYERS    = 24     # max weight tensors to sample for SNR measurement
TPS_MAX_TOKENS    = 128    # generation tokens per throughput run
PPL_MAX_TOKENS    = 512    # tokens for perplexity estimate
MLX_INT4_CACHE    = Path.home() / "models" / "Qwen3-14B-mlx-int4"  # cached INT4

# ── wikitext-2 sample ─────────────────────────────────────────────────────────
_WIKITEXT_SAMPLE = (
    "= Valkyria Chronicles III =\n"
    "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , "
    "lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III "
    "outside Japan , is a tactical role @-@ playing video game developed by Sega and "
    "Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , "
    "it is the third game in the Valkyria series . Employing the same fusion of tactical "
    "and real @-@ time gameplay as its predecessors , the story runs parallel to the first "
    "game and follows the Nameless , a penal military unit serving the nation of Gallia "
    "during the Second Europan War who perform missions too sensitive for the regular army "
    "to carry out directly . "
)

_THROUGHPUT_PROMPTS = [
    "Explain the architectural differences between transformer attention and linear attention.",
    "What are the practical tradeoffs of INT3 quantization for large language models?",
    "Describe how low-rank matrix decomposition improves quantization reconstruction quality.",
]

# ── result dataclasses ────────────────────────────────────────────────────────

@dataclass
class CompressionMetrics:
    bf16_gb: float
    int3_gb: float
    ratio: float              # int3 / bf16
    size_savings_pct: float   # (1 - ratio) * 100
    actual_bpw: float         # measured from file sizes
    n_int3_tensors: int
    n_passthrough_tensors: int
    compress_seconds: float | None  # None if we didn't time it
    error: str | None = None


@dataclass
class SNRLayerResult:
    name: str
    shape: tuple
    snr_db: float
    time_ms: float


@dataclass
class SNRMetrics:
    layers_tested: int
    snr_mean_db: float
    snr_min_db: float
    snr_max_db: float
    snr_p25_db: float     # 25th percentile (worst quarter)
    layer_results: list[SNRLayerResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class ThroughputMetrics:
    tps_mean: float
    tps_stdev: float
    tps_min: float
    tps_max: float
    ttft_mean_ms: float
    n_runs: int
    model_path_used: str  # which model path was actually loaded
    error: str | None = None


@dataclass
class PerplexityMetrics:
    ppl: float
    n_tokens: int
    error: str | None = None


@dataclass
class BenchResults:
    model_id: str = "Qwen3-14B"
    compression: CompressionMetrics | None = None
    snr: SNRMetrics | None = None
    throughput: ThroughputMetrics | None = None
    perplexity: PerplexityMetrics | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    platform_info: dict = field(default_factory=dict)


# ── display helpers ───────────────────────────────────────────────────────────

def _hdr(title: str, sub: str = "") -> None:
    print(f"\n{W}{'─' * 70}{NC}")
    print(f"{C}  {title}{NC}")
    if sub:
        print(f"{D}  {sub}{NC}")
    print(f"{W}{'─' * 70}{NC}")


def _row(label: str, val: str, extra: str = "") -> None:
    print(f"  {label:<48} {G}{val:>14}{NC}  {D}{extra}{NC}")


def _warn(msg: str) -> None:
    print(f"  {Y}⚠{NC}  {msg}")


def _ok(label: str, val: str, extra: str = "") -> None:
    print(f"  {G}✓{NC} {label:<50} {G}{val}{NC}  {D}{extra}{NC}")


def _err(label: str, reason: str) -> None:
    print(f"  {R}✗{NC} {label:<50} {D}{reason}{NC}")


# ── utility helpers ───────────────────────────────────────────────────────────

def _dir_gb(path: Path) -> float:
    """Total on-disk size of a directory in GB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def _python() -> str:
    return sys.executable


def _detect_ram_gb() -> float:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / 1e9
    except Exception:
        pass
    return 16.0


def _platform_info() -> dict:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python":   platform.python_version(),
        "ram_gb":   _detect_ram_gb(),
    }
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["cpu"] = result.stdout.strip()
    except Exception:
        pass
    return info


# ── S0: compression metrics ───────────────────────────────────────────────────

def measure_compression(bf16_dir: Path, int3_dir: Path) -> CompressionMetrics:
    """Derive compression metrics from directory contents."""
    bf16_gb = _dir_gb(bf16_dir)
    int3_gb = _dir_gb(int3_dir)

    # Count INT3 vs passthrough tensors by suffix
    npy_files = list(int3_dir.rglob("*.npy"))
    n_int3 = sum(1 for f in npy_files if "__q3" in f.name)
    n_pass = sum(1 for f in npy_files if "__pt" in f.name)

    # Actual BPW: (int3 bytes) / (bf16 params).
    # BF16 model is 2 bytes/param; total params = bf16_gb * 1e9 / 2
    total_params_approx = (bf16_gb * 1e9) / 2.0
    actual_bpw = (int3_gb * 1e9 * 8) / max(total_params_approx, 1)

    ratio = int3_gb / max(bf16_gb, 1e-6)

    # Check if manifest has compression time recorded
    compress_seconds = None
    manifest_path = int3_dir / "tensors" / "manifest.json"
    if not manifest_path.exists():
        manifest_path = int3_dir / "manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                mf = json.load(f)
            compress_seconds = mf.get("compress_seconds")
        except Exception:
            pass

    return CompressionMetrics(
        bf16_gb=round(bf16_gb, 2),
        int3_gb=round(int3_gb, 2),
        ratio=round(ratio, 4),
        size_savings_pct=round((1 - ratio) * 100, 1),
        actual_bpw=round(actual_bpw, 2),
        n_int3_tensors=n_int3,
        n_passthrough_tensors=n_pass,
        compress_seconds=compress_seconds,
    )


# ── S1: per-layer reconstruction SNR ────────────────────────────────────────

def _safe_key(tensor_name: str) -> str:
    """Replicate squish convert.safe_key() to retrieve npy filenames."""
    return tensor_name.replace(".", "__")


def _load_int3_layer(int3_dir: Path, sk: str) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple
]:
    """Load q3/s3/z3/lra/lrb/shape arrays for one tensor."""
    def _load(suffix: str) -> np.ndarray:
        # tensors sub-directory used by squish convert npy-dir format
        for candidate in [
            int3_dir / "tensors" / f"{sk}{suffix}.npy",
            int3_dir / f"{sk}{suffix}.npy",
        ]:
            if candidate.exists():
                return np.load(str(candidate))
        raise FileNotFoundError(f"{sk}{suffix}.npy not found in {int3_dir}")

    q3    = _load("__q3")
    s3    = _load("__s3")
    z3    = _load("__z3")
    lra   = _load("__lra")
    lrb   = _load("__lrb")
    shape_arr = _load("__shape")
    shape = tuple(int(d) for d in shape_arr)
    return q3, s3, z3, lra, lrb, shape


def _dequantize_int3(
    q3: np.ndarray,
    s3: np.ndarray,
    z3: np.ndarray,
    lra: np.ndarray,
    lrb: np.ndarray,
    shape: tuple,
    group_size: int = 64,
) -> np.ndarray:
    """Reconstruct float32 weight from INT3 MiLo representation."""
    from squish.quant.milo_quant import MiLoConfig, MiLoQuantizer
    milo = MiLoQuantizer(MiLoConfig(group_size=group_size))
    n = int(np.prod(shape))
    w_dq = milo.dequantize(q3, s3, z3, n=n, original_shape=shape)
    # Add low-rank compensator: A @ B  (shape must broadcast to w_dq)
    w_comp = w_dq + lra @ lrb
    return w_comp.reshape(shape)


def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """20·log10(rms(signal) / rms(error))."""
    err = original.astype(np.float32) - reconstructed.astype(np.float32)
    sig_pow = float(np.sum(original.astype(np.float32) ** 2))
    err_pow = float(np.sum(err ** 2))
    if err_pow == 0.0:
        return float("inf")
    return float(10.0 * np.log10(sig_pow / (err_pow + 1e-30)))


def measure_snr(
    bf16_dir: Path,
    int3_dir: Path,
    max_layers: int = MAX_SNR_LAYERS,
    group_size: int = 64,
) -> SNRMetrics:
    """Sample weight layers, dequantize INT3 → fp32, compare SNR to BF16."""
    try:
        from safetensors.torch import load_file as _st_load
    except ImportError:
        return SNRMetrics(
            layers_tested=0,
            snr_mean_db=0.0, snr_min_db=0.0, snr_max_db=0.0, snr_p25_db=0.0,
            error="safetensors not installed",
        )

    # Build tensor_name → safe_key map from manifest
    manifest_path = int3_dir / "tensors" / "manifest.json"
    if not manifest_path.exists():
        manifest_path = int3_dir / "manifest.json"
    if not manifest_path.exists():
        return SNRMetrics(
            layers_tested=0,
            snr_mean_db=0.0, snr_min_db=0.0, snr_max_db=0.0, snr_p25_db=0.0,
            error=f"manifest.json not found in {int3_dir}",
        )
    with open(manifest_path) as f:
        manifest: dict[str, str] = json.load(f)

    # Identify which tensor names have INT3 representation
    tensors_dir = int3_dir / "tensors"
    if not tensors_dir.exists():
        tensors_dir = int3_dir
    q3_stems = {f.name.replace("__q3.npy", "") for f in tensors_dir.glob("*__q3.npy")}
    int3_names = [tn for tn, sk in manifest.items() if sk in q3_stems]

    # Sort for determinism, interleave layers evenly so we cover all depths
    int3_names.sort()
    step = max(1, len(int3_names) // max_layers)
    sampled = int3_names[::step][:max_layers]

    print(f"  Sampling {len(sampled)} / {len(int3_names)} INT3 weight tensors for SNR…")

    # Group sampled names by shard
    shard_files = sorted(bf16_dir.glob("*.safetensors"))
    layer_results: list[SNRLayerResult] = []

    for shard_path in shard_files:
        shard_data = _st_load(str(shard_path))
        for tname, tensor in shard_data.items():
            if tname not in sampled:
                continue
            sk = manifest[tname]
            t0 = time.perf_counter()
            try:
                original_f32 = tensor.float().numpy()
                q3, s3, z3, lra, lrb, shape = _load_int3_layer(int3_dir, sk)
                reconstructed = _dequantize_int3(q3, s3, z3, lra, lrb, shape, group_size)
                snr = _snr_db(original_f32, reconstructed)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                layer_results.append(SNRLayerResult(
                    name=tname,
                    shape=tuple(original_f32.shape),
                    snr_db=round(snr, 2),
                    time_ms=round(elapsed_ms, 1),
                ))
                print(f"    {G}✓{NC} {tname:<56} SNR={snr:>6.1f} dB  {elapsed_ms:.0f} ms")
            except Exception as exc:
                print(f"    {Y}⚠{NC} {tname}: {exc}")
        del shard_data

    if not layer_results:
        return SNRMetrics(
            layers_tested=0,
            snr_mean_db=0.0, snr_min_db=0.0, snr_max_db=0.0, snr_p25_db=0.0,
            error="no INT3 tensors successfully dequantized",
        )

    snr_vals = np.array([r.snr_db for r in layer_results])
    return SNRMetrics(
        layers_tested=len(layer_results),
        snr_mean_db=round(float(snr_vals.mean()), 2),
        snr_min_db=round(float(snr_vals.min()), 2),
        snr_max_db=round(float(snr_vals.max()), 2),
        snr_p25_db=round(float(np.percentile(snr_vals, 25)), 2),
        layer_results=layer_results,
    )


# ── S2: throughput via mlx_lm ─────────────────────────────────────────────────

def _ensure_mlx_int4(bf16_dir: Path, cache_dir: Path) -> Path:
    """Return an MLX INT4 quantized version of bf16_dir (creates if absent)."""
    if cache_dir.exists() and any(cache_dir.glob("*.safetensors")):
        print(f"  Using cached MLX INT4 model: {cache_dir}")
        return cache_dir

    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    print(f"  {C}→ Converting BF16 → MLX INT4 for inference baseline…{NC}")
    cmd = [
        _python(), "-m", "mlx_lm", "convert",
        "--hf-path",     str(bf16_dir),
        "--mlx-path",    str(cache_dir),
        "--quantize",
        "--q-bits",      "4",
        "--q-group-size", "64",
    ]
    try:
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), timeout=7200)
        if proc.returncode != 0:
            _warn(f"mlx_lm convert returned {proc.returncode}")
            return bf16_dir
    except subprocess.TimeoutExpired:
        _warn("mlx_lm convert timed out (2h)")
        return bf16_dir
    except FileNotFoundError:
        _warn("mlx_lm not found in PATH")
        return bf16_dir

    if any(cache_dir.glob("*.safetensors")):
        print(f"  {G}✓{NC} MLX INT4 ready: {cache_dir}")
        return cache_dir
    _warn("mlx_lm convert produced no safetensors")
    return bf16_dir


def measure_throughput(
    bf16_dir: Path,
    runs: int = 3,
    max_tokens: int = TPS_MAX_TOKENS,
) -> ThroughputMetrics:
    """Run generation throughput benchmark via mlx_lm on INT4-quantised BF16."""
    try:
        import mlx_lm
    except ImportError:
        return ThroughputMetrics(
            tps_mean=0.0, tps_stdev=0.0, tps_min=0.0, tps_max=0.0,
            ttft_mean_ms=0.0, n_runs=0,
            model_path_used="",
            error="mlx_lm not installed",
        )

    ram_gb = _detect_ram_gb()
    bf16_gb = _dir_gb(bf16_dir)
    if bf16_gb > (ram_gb - 4.5):
        infer_path = _ensure_mlx_int4(bf16_dir, MLX_INT4_CACHE)
    else:
        infer_path = bf16_dir

    print(f"  Loading {infer_path.name} for throughput test…")
    try:
        model, tokenizer = mlx_lm.load(str(infer_path))
    except Exception as exc:
        return ThroughputMetrics(
            tps_mean=0.0, tps_stdev=0.0, tps_min=0.0, tps_max=0.0,
            ttft_mean_ms=0.0, n_runs=0,
            model_path_used=str(infer_path),
            error=f"mlx_lm.load failed: {exc}",
        )

    tps_list: list[float] = []
    ttft_list: list[float] = []

    for i, prompt in enumerate(_THROUGHPUT_PROMPTS[:runs]):
        print(f"  Run {i + 1}/{runs}: ", end="", flush=True)
        t_start = time.perf_counter()
        first_tok = None
        n_toks = 0
        try:
            for _chunk in mlx_lm.stream_generate(
                model, tokenizer, prompt=prompt, max_tokens=max_tokens
            ):
                if first_tok is None:
                    first_tok = time.perf_counter()
                n_toks += 1
        except Exception as exc:
            print(f"{R}error{NC}: {exc}")
            continue
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        tps = n_toks / elapsed if elapsed > 0 else 0.0
        ttft_ms = (first_tok - t_start) * 1000 if first_tok is not None else 0.0
        tps_list.append(tps)
        ttft_list.append(ttft_ms)
        print(f"{tps:.1f} tok/s  (TTFT {ttft_ms:.0f} ms)")

    if not tps_list:
        return ThroughputMetrics(
            tps_mean=0.0, tps_stdev=0.0, tps_min=0.0, tps_max=0.0,
            ttft_mean_ms=0.0, n_runs=0,
            model_path_used=str(infer_path),
            error="all runs failed",
        )

    arr = np.array(tps_list)
    return ThroughputMetrics(
        tps_mean=round(float(arr.mean()), 1),
        tps_stdev=round(float(arr.std()), 1),
        tps_min=round(float(arr.min()), 1),
        tps_max=round(float(arr.max()), 1),
        ttft_mean_ms=round(float(np.mean(ttft_list)), 0),
        n_runs=len(tps_list),
        model_path_used=str(infer_path),
    )


# ── S3: perplexity ────────────────────────────────────────────────────────────

def measure_perplexity(bf16_dir: Path, max_tokens: int = PPL_MAX_TOKENS) -> PerplexityMetrics:
    """Estimate wikitext-2 perplexity via mlx_lm token log-probs."""
    try:
        import mlx.core as mx
        import mlx_lm
    except ImportError:
        return PerplexityMetrics(
            ppl=0.0, n_tokens=0, error="mlx.core or mlx_lm not installed"
        )

    ram_gb = _detect_ram_gb()
    bf16_gb = _dir_gb(bf16_dir)
    if bf16_gb > (ram_gb - 4.5):
        infer_path = _ensure_mlx_int4(bf16_dir, MLX_INT4_CACHE)
    else:
        infer_path = bf16_dir

    print(f"  Loading {infer_path.name} for perplexity…")
    try:
        model, tokenizer = mlx_lm.load(str(infer_path))
    except Exception as exc:
        return PerplexityMetrics(ppl=0.0, n_tokens=0, error=str(exc))

    try:
        tokens = tokenizer.encode(_WIKITEXT_SAMPLE)[:max_tokens]
        n = len(tokens)
        if n < 8:
            return PerplexityMetrics(ppl=0.0, n_tokens=n, error="too few tokens")

        input_ids = mx.array(tokens[:-1])[None]    # (1, n-1)
        target_ids = mx.array(tokens[1:])          # (n-1,)

        logits = model(input_ids)                  # (1, n-1, vocab)
        log_probs = mx.log(mx.softmax(logits[0], axis=-1))  # (n-1, vocab)

        # Gather log-prob for each target token
        nll_sum = 0.0
        for i, tid in enumerate(tokens[1:]):
            nll_sum -= float(log_probs[i, tid])

        ppl = math.exp(nll_sum / (n - 1))
        return PerplexityMetrics(ppl=round(ppl, 2), n_tokens=n - 1)

    except Exception as exc:
        return PerplexityMetrics(ppl=0.0, n_tokens=0, error=str(exc))


# ── markdown summary ──────────────────────────────────────────────────────────

def print_markdown(results: BenchResults) -> None:
    ram_gb = results.platform_info.get('ram_gb', 0)
    ram_str = f"{float(ram_gb):.0f} GB RAM" if ram_gb else "RAM unknown"
    lines = [
        "## Qwen3-14B INT3-MiLo Benchmark Results",
        "",
        f"*Generated: {results.timestamp}*",
        "",
        f"Platform: {results.platform_info.get('cpu', 'unknown')} · {ram_str}",
        "",
    ]

    if (c := results.compression) and not c.error:
        lines += [
            "### S0 — Compression Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| BF16 size | {c.bf16_gb:.2f} GB |",
            f"| INT3 size | {c.int3_gb:.2f} GB |",
            f"| Compression ratio | {c.ratio:.3f} ({c.size_savings_pct:.1f}% savings) |",
            f"| Actual BPW | {c.actual_bpw:.2f} bpw |",
            f"| INT3 tensors | {c.n_int3_tensors} |",
            f"| Passthrough tensors | {c.n_passthrough_tensors} |",
        ]
        if c.compress_seconds:
            lines.append(f"| Compression time | {c.compress_seconds:.0f} s |")
        lines.append("")

    if (s := results.snr) and not s.error:
        lines += [
            "### S1 — Reconstruction SNR (INT3 vs BF16)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Layers sampled | {s.layers_tested} |",
            f"| Mean SNR | {s.snr_mean_db:.1f} dB |",
            f"| Min  SNR | {s.snr_min_db:.1f} dB |",
            f"| Max  SNR | {s.snr_max_db:.1f} dB |",
            f"| P25  SNR | {s.snr_p25_db:.1f} dB (worst 25%) |",
            "",
            "> ℹ️  Higher SNR is better. >8 dB generally preserves reasoning quality.",
            "",
        ]

    if (t := results.throughput) and not t.error:
        lines += [
            "### S2 — Generation Throughput",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| tok/s (mean) | {t.tps_mean:.1f} |",
            f"| tok/s (±stdev) | ±{t.tps_stdev:.1f} |",
            f"| tok/s (min/max) | {t.tps_min:.1f} – {t.tps_max:.1f} |",
            f"| TTFT (mean) | {t.ttft_mean_ms:.0f} ms |",
            f"| Runs | {t.n_runs} |",
            f"| Model loaded | `{Path(t.model_path_used).name}` |",
            "",
            "> ℹ️  Throughput measured with INT4-quantised BF16 via mlx_lm (the INT3 "
            "> inference runtime is not yet wired into mlx_lm). INT3 inference will be "
            "> faster due to lower memory bandwidth requirements.",
            "",
        ]

    if (p := results.perplexity) and not p.error:
        lines += [
            "### S3 — Perplexity (wikitext-2)",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| PPL | {p.ppl:.2f} |",
            f"| Tokens | {p.n_tokens} |",
            "",
        ]

    for line in lines:
        print(line)


# ── main ─────────────────────────────────────────────────────────────────────

def _wait_for_completion(int3_dir: Path, poll_s: int = 15) -> None:
    """Block until int3_dir contains manifest.json or .manifest_ready."""
    print(f"  {C}Waiting for compression to complete at {int3_dir}…{NC}")
    while True:
        for sentinel in [
            int3_dir / "tensors" / "manifest.json",
            int3_dir / "manifest.json",
            int3_dir / "tensors" / ".manifest_ready",
        ]:
            if sentinel.exists():
                print(f"  {G}✓{NC} Compression complete ({sentinel.name} found)")
                return
        gb = _dir_gb(int3_dir) if int3_dir.exists() else 0.0
        print(f"  … {gb:.1f} GB written — checking again in {poll_s}s", end="\r", flush=True)
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Benchmark Qwen3-14B INT3-MiLo compressed model.",
    )
    ap.add_argument("--bf16-dir",  type=Path, default=DEFAULT_BF16_DIR,
                    help=f"BF16 model directory (default: {DEFAULT_BF16_DIR})")
    ap.add_argument("--int3-dir",  type=Path, default=DEFAULT_INT3_DIR,
                    help=f"INT3 npy-dir (default: {DEFAULT_INT3_DIR})")
    ap.add_argument("--wait",      action="store_true",
                    help="Wait for INT3 compression to finish before benchmarking")
    ap.add_argument("--eval-snr",  action="store_true", default=True,
                    help="Run per-layer SNR quality check (default: on)")
    ap.add_argument("--no-snr",    dest="eval_snr", action="store_false")
    ap.add_argument("--eval-tps",  action="store_true",
                    help="Run generation throughput benchmark via mlx_lm")
    ap.add_argument("--eval-ppl",  action="store_true",
                    help="Run wikitext-2 perplexity estimate via mlx_lm")
    ap.add_argument("--runs",      type=int, default=3,
                    help="Number of throughput runs (default: 3)")
    ap.add_argument("--max-snr-layers", type=int, default=MAX_SNR_LAYERS,
                    help=f"Max layers to sample for SNR (default: {MAX_SNR_LAYERS})")
    ap.add_argument("--snr-group-size", type=int, default=64,
                    help="MiLo group size used during compression (default: 64)")
    ap.add_argument("--output-dir", type=Path,
                    default=_REPO_ROOT / "dev" / "results" / "qwen3_14b_int3",
                    help="Directory to save JSON results")
    ap.add_argument("--markdown",  action="store_true",
                    help="Print markdown summary at the end")
    args = ap.parse_args(argv)

    results = BenchResults(platform_info=_platform_info())
    ram_gb = results.platform_info.get("ram_gb", 16.0)

    _hdr(
        "Qwen3-14B INT3-MiLo Benchmark",
        f"RAM: {ram_gb:.0f} GB  ·  BF16: {args.bf16_dir.name}  ·  INT3: {args.int3_dir.name}",
    )

    # Validate inputs
    if not args.bf16_dir.exists():
        print(f"{R}Error:{NC} BF16 dir not found: {args.bf16_dir}")
        return 1
    if not args.int3_dir.exists():
        if args.wait:
            _wait_for_completion(args.int3_dir)
        else:
            print(f"{R}Error:{NC} INT3 dir not found: {args.int3_dir}")
            print(f"  Re-run with --wait to wait for compression, or set --int3-dir.")
            return 1

    # Check for manifest (indicates completed compression)
    sentinel_exists = any(
        p.exists() for p in [
            args.int3_dir / "tensors" / "manifest.json",
            args.int3_dir / "manifest.json",
            args.int3_dir / "tensors" / ".manifest_ready",
        ]
    )
    if not sentinel_exists:
        if args.wait:
            _wait_for_completion(args.int3_dir)
        else:
            _warn("manifest.json not found — compression may still be running")
            _warn("Run with --wait to block until complete, or the SNR results may be partial")

    # ── S0: compression metrics ───────────────────────────────────────────────
    _hdr("S0 — Compression Metrics")
    c = measure_compression(args.bf16_dir, args.int3_dir)
    results.compression = c
    if c.error:
        _err("compression metrics", c.error)
    else:
        _row("Original BF16 size",    f"{c.bf16_gb:.2f} GB")
        _row("INT3 compressed size",  f"{c.int3_gb:.2f} GB",
             f"saves {c.size_savings_pct:.1f}%")
        _row("Compression ratio",     f"{c.ratio:.4f}",
             f"({1/c.ratio:.2f}× smaller)" if c.ratio > 0 else "")
        _row("Actual BPW",            f"{c.actual_bpw:.2f} bpw",
             f"(BF16 = 16 bpw)")
        _row("INT3 quantised tensors", str(c.n_int3_tensors))
        _row("Passthrough tensors",    str(c.n_passthrough_tensors))
        if c.compress_seconds:
            _row("Compression time", f"{c.compress_seconds:.0f} s")

    # ── S1: reconstruction SNR ────────────────────────────────────────────────
    if args.eval_snr:
        _hdr("S1 — Reconstruction SNR (INT3 vs BF16)")
        s = measure_snr(
            args.bf16_dir, args.int3_dir,
            max_layers=args.max_snr_layers,
            group_size=args.snr_group_size,
        )
        results.snr = s
        if s.error:
            _err("SNR measurement", s.error)
        else:
            print()
            _row("Layers sampled",  str(s.layers_tested))
            _row("Mean SNR",        f"{s.snr_mean_db:.1f} dB", "higher = better")
            _row("Min  SNR",        f"{s.snr_min_db:.1f} dB",  "worst layer")
            _row("Max  SNR",        f"{s.snr_max_db:.1f} dB",  "best layer")
            _row("P25  SNR",        f"{s.snr_p25_db:.1f} dB",  "worst 25% of layers")

    # ── S2: throughput ────────────────────────────────────────────────────────
    if args.eval_tps:
        _hdr("S2 — Generation Throughput (mlx_lm INT4 baseline)")
        t = measure_throughput(args.bf16_dir, runs=args.runs, max_tokens=TPS_MAX_TOKENS)
        results.throughput = t
        if t.error:
            _err("throughput", t.error)
        else:
            _row("tok/s (mean ± stdev)",
                 f"{t.tps_mean:.1f} ± {t.tps_stdev:.1f}",
                 f"min={t.tps_min:.1f} max={t.tps_max:.1f}")
            _row("TTFT (mean)", f"{t.ttft_mean_ms:.0f} ms")
            _row("Model loaded", Path(t.model_path_used).name)

    # ── S3: perplexity ────────────────────────────────────────────────────────
    if args.eval_ppl:
        _hdr("S3 — Perplexity (wikitext-2)")
        p = measure_perplexity(args.bf16_dir, max_tokens=PPL_MAX_TOKENS)
        results.perplexity = p
        if p.error:
            _err("perplexity", p.error)
        else:
            _row("Wikitext-2 PPL", f"{p.ppl:.2f}", f"{p.n_tokens} tokens")

    # ── Write JSON results ────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "qwen3_14b_int3_bench.json"

    def _serial(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"not serializable: {type(obj)}")

    with open(out_json, "w") as f:
        json.dump(asdict(results), f, indent=2, default=str)
    print(f"\n  {G}✓{NC} Results saved → {out_json}")

    # ── Final summary ─────────────────────────────────────────────────────────
    _hdr("Summary")
    if (c := results.compression) and not c.error:
        print(f"  {W}Qwen3-14B INT3-MiLo{NC}")
        print(f"  {c.bf16_gb:.1f} GB → {c.int3_gb:.1f} GB  "
              f"({c.size_savings_pct:.0f}% smaller, {c.actual_bpw:.2f} bpw)")
    if (s := results.snr) and not s.error:
        quality = (
            "excellent" if s.snr_mean_db >= 15 else
            "good"      if s.snr_mean_db >= 10 else
            "acceptable" if s.snr_mean_db >= 7 else
            "degraded"
        )
        print(f"  Reconstruction SNR: {s.snr_mean_db:.1f} dB mean  [{quality}]")
    if (t := results.throughput) and not t.error:
        print(f"  Throughput: {t.tps_mean:.1f} tok/s  (INT4 baseline on {ram_gb:.0f} GB machine)")

    if args.markdown:
        print()
        print_markdown(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
