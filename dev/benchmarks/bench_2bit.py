#!/usr/bin/env python3
"""
bench_2bit.py — 2-bit quantization comparison benchmark.

Evaluates four near-2-bit weight-compression methods on the same weight matrix:

  int4   4-bit nibble-packed scalar quantization   (squish baseline, ~4.5 bpw)
  vptq   Vector Post-Training Quantization          (NeurIPS 2025, Phase 7)
  aqlm   Additive Quantization of Language Models   (ICML 2024, Phase 9A)
  quip   QuIP# E8-lattice + incoherence processing  (2024, Phase 9B)

Stage 1 — weight-reconstruction (always runs; no GPU/model required):
  bpw            bits per weight after compression
  snr_db         signal-to-noise ratio of reconstructed vs. original [dB]
  compress_ms    wall-clock time for the compress step [ms]
  decompress_ms  wall-clock time for the decompress step [ms]

Stage 2 — model evaluation (requires --model-dir and mlx + mlx_lm):
  perplexity     wikitext-2 test perplexity (exp of mean token NLL)
  tps            tokens / second on a fixed-length generation burst

Results are written to dev/results/quant_2bit_comparison.json and an ASCII
summary table is printed to stdout.

Usage
-----
  python3 dev/benchmarks/bench_2bit.py
  python3 dev/benchmarks/bench_2bit.py --model-dir models/Qwen2.5-1.5B
  python3 dev/benchmarks/bench_2bit.py --output dev/results/quant_2bit_comparison.json
  python3 dev/benchmarks/bench_2bit.py --markdown
  python3 dev/benchmarks/bench_2bit.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ── paths ─────────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = _REPO_ROOT / "dev" / "results" / "quant_2bit_comparison.json"

# ── colour helpers (same convention as bench_wave25_26.py) ────────────────────
G = "\033[32m"; Y = "\033[33m"; C = "\033[36m"; W = "\033[1;37m"
D = "\033[2m";  NC = "\033[0m"; R = "\033[31m"; B = "\033[1m"

RNG = np.random.default_rng(42)

# ── benchmark constants ───────────────────────────────────────────────────────
# Weight matrix dimensions are kept small so the VPTQ k-means++ initialisation
# (O(n_groups × k²) in pure Python) completes in < 5 s.  Production models use
# 4096×4096 weight matrices; scale the configs up for offline profiling.
BENCH_ROWS      = 64    # synthetic weight-matrix height
BENCH_COLS      = 64    # synthetic weight-matrix width  (must be ≥ INT4_GROUP_SIZE)
INT4_GROUP_SIZE = 64    # standard group size for INT4 (1 group per row here)
VPTQ_GROUP_SIZE = 8     # VPTQ weight-vector length (sub-vector size)
# Use k=16 primary codebook (4-bit codes) instead of k=256 so k-means++ init
# stays O(n_groups × k²) = O(512 × 256) ≈ fast, while still demonstrating
# sub-2-bit total bpw.  For a full-quality run use n_codebook_entries=256.
VPTQ_N_PRIMARY  = 16    # primary-codebook entries (4-bit codes)
VPTQ_N_RESIDUAL = 4     # residual-codebook entries (2-bit residual)
VPTQ_ITERS      = 5     # k-means iterations
TPS_GEN_TOKENS  = 128   # tokens generated for TPS measure
PPL_MAX_TOKENS  = 2048  # wikitext-2 tokens consumed for perplexity

# ── wikitext-2 excerpt ────────────────────────────────────────────────────────
# First ~800 chars of the wikitext-2-raw-v1 test split; used as fallback when
# the datasets library is not installed or the machine is offline.
_WIKITEXT_SAMPLE = (
    "= Valkyria Chronicles III = \n"
    "Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , "
    "lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles "
    "III outside Japan , is a tactical role @-@ playing video game developed by Sega and "
    "Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is "
    "the third game in the Valkyria series . Employing the same fusion of tactical and "
    "real @-@ time gameplay as its predecessors , the story runs parallel to the first game "
    "and follows the Nameless , a penal military unit serving the nation of Gallia during "
    "the Second Europan War who perform missions too sensitive for the regular army to carry "
    "out directly . The game began development in 2010 , carrying over a large portion of "
    "the engine and graphics from Valkyria Chronicles II . While it retained the standard "
    "features of the series , it also incorporated additional ones , such as special valkyria "
    "abilities that can be activated a certain number of times during a mission . The game "
    "was well received in Japan , but was not ported to other territories ."
)


# ── display helpers ───────────────────────────────────────────────────────────

def _hdr(title: str) -> None:
    print(f"\n{W}{'─' * 72}{NC}")
    print(f"{C}  {title}{NC}")
    print(f"{W}{'─' * 72}{NC}")


def _row(label: str, val: str, extra: str = "") -> None:
    print(f"  {label:<48} {G}{val:>16}{NC}  {D}{extra}{NC}")


def _skip(label: str, reason: str = "") -> None:
    print(f"  {Y}~ SKIP{NC}  {label:<48} {D}{reason}{NC}")


def _err(label: str, reason: str = "") -> None:
    print(f"  {R}✗ ERROR{NC}  {label:<46} {D}{reason}{NC}")


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    """Stage-1 weight-reconstruction metrics for a single compression method."""

    status:         str             # "ok" | "skip" | "error"
    reason:         str  = ""
    bpw:            float | None = None
    snr_db:         float | None = None
    compress_ms:    float | None = None
    decompress_ms:  float | None = None
    perplexity:     float | None = None
    tps:            float | None = None
    backend:        str  = ""


# ── pure-NumPy asymmetric INT4 (no Rust extension required) ──────────────────

def _int4_quantize_np(
    W: np.ndarray,
    group_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Asymmetric per-group INT4 quantisation (pure NumPy fallback).

    Returns
    -------
    packed      : (rows, padded_cols // 2) uint8  — nibble-packed [lo | hi]
    scales      : (rows, n_groups) float32
    zero_points : (rows, n_groups) float32
    """
    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape

    # Pad columns to a multiple of group_size.
    pad = (-cols) % group_size
    if pad:
        W = np.pad(W, ((0, 0), (0, pad)))
    padded_cols = W.shape[1]
    n_groups = padded_cols // group_size

    Wg = W.reshape(rows, n_groups, group_size)
    w_min = Wg.min(axis=-1)          # (rows, n_groups)
    w_max = Wg.max(axis=-1)
    scales = (w_max - w_min) / 15.0  # map [wmin, wmax] → [0, 15]
    zero_points = w_min

    scales_safe = np.where(scales == 0.0, 1.0, scales)
    q = np.round((Wg - zero_points[..., None]) / scales_safe[..., None])
    q = q.clip(0, 15).astype(np.uint8).reshape(rows, padded_cols)

    # Nibble pack: lo nibble = even-index column, hi nibble = odd-index column.
    lo = q[:, 0::2] & 0x0F
    hi = q[:, 1::2] & 0x0F
    packed = (lo | (hi << 4)).astype(np.uint8)

    return packed, scales.astype(np.float32), zero_points.astype(np.float32)


def _int4_dequantize_np(
    packed: np.ndarray,
    scales: np.ndarray,
    zero_points: np.ndarray,
    group_size: int = 64,
    original_cols: int | None = None,
) -> np.ndarray:
    """Reconstruct float32 weights from nibble-packed INT4 + per-group stats."""
    rows = packed.shape[0]
    padded_cols = packed.shape[1] * 2
    n_groups = scales.shape[1]

    lo = (packed & 0x0F).astype(np.float32)
    hi = ((packed >> 4) & 0x0F).astype(np.float32)
    q = np.empty((rows, padded_cols), dtype=np.float32)
    q[:, 0::2] = lo
    q[:, 1::2] = hi

    q_grouped = q.reshape(rows, n_groups, group_size)
    W_approx = q_grouped * scales[..., None] + zero_points[..., None]
    W_approx = W_approx.reshape(rows, padded_cols)

    if original_cols is not None:
        W_approx = W_approx[:, :original_cols]
    return W_approx


def _int4_bpw(group_size: int, asymmetric: bool = True) -> float:
    """Theoretical bits-per-weight for INT4 nibble packing.

    Accounts for:
      • 4 bits / weight (nibble packed)
      • float32 scale per group     (+32 / group_size bpw)
      • float32 zero_point per group, only when asymmetric (+32 / group_size bpw)
    """
    overhead = (32.0 / group_size) * (2 if asymmetric else 1)
    return 4.0 + overhead


# ── signal-quality helper ─────────────────────────────────────────────────────

def _snr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Signal-to-noise ratio in dB.

    SNR_dB = 10 * log10( E[W²] / E[(W − Ŵ)²] )

    Higher is better.  Representative values:
      INT4 (group=64)      ~33–42 dB
      VPTQ (~2 bpw)        ~28–36 dB
    """
    orig  = original.astype(np.float64).ravel()
    recon = reconstructed.astype(np.float64).ravel()
    n = min(orig.size, recon.size)
    orig, recon = orig[:n], recon[:n]

    sig_power   = float(np.mean(orig ** 2))
    noise_power = float(np.mean((orig - recon) ** 2))

    if noise_power == 0.0:
        return float("inf")
    if sig_power == 0.0:
        return float("-inf")
    return 10.0 * math.log10(sig_power / noise_power)


# ── stage-1 per-method benchmarks ────────────────────────────────────────────

def bench_int4(W: np.ndarray) -> MethodResult:
    """Benchmark INT4 nibble quantisation.

    Prefers the squish_quant Rust extension (symmetric INT4); falls back to the
    pure-NumPy asymmetric implementation when Rust is unavailable.
    """
    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape
    backend = "numpy"

    _sq = None
    try:
        import squish_quant as _sq  # type: ignore[import]
    except ImportError:
        pass

    if _sq is not None:
        backend = "rust"
        try:
            t0 = time.perf_counter()
            packed, scales = _sq.quantize_int4_grouped(
                np.ascontiguousarray(W), INT4_GROUP_SIZE
            )
            compress_ms = (time.perf_counter() - t0) * 1e3

            t0 = time.perf_counter()
            W_approx = _sq.dequantize_int4_grouped(
                np.ascontiguousarray(packed, dtype=np.uint8),
                np.ascontiguousarray(scales, dtype=np.float32),
                INT4_GROUP_SIZE,
            )
            decompress_ms = (time.perf_counter() - t0) * 1e3

            # Rust: symmetric INT4 with float32 scales (no zero_point).
            bpw = _int4_bpw(INT4_GROUP_SIZE, asymmetric=False)
            snr = _snr_db(W, W_approx[:rows, :cols])
            return MethodResult(
                status="ok", bpw=bpw, snr_db=snr,
                compress_ms=compress_ms, decompress_ms=decompress_ms,
                backend=backend,
            )
        except Exception:
            backend = "numpy"  # fall through to NumPy path

    # Pure-NumPy asymmetric fallback.
    t0 = time.perf_counter()
    packed, scales, zp = _int4_quantize_np(W, INT4_GROUP_SIZE)
    compress_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    W_approx = _int4_dequantize_np(packed, scales, zp, INT4_GROUP_SIZE, cols)
    decompress_ms = (time.perf_counter() - t0) * 1e3

    # Asymmetric: scale + zero_point, both float32.
    bpw = _int4_bpw(INT4_GROUP_SIZE, asymmetric=True)
    snr = _snr_db(W, W_approx)
    return MethodResult(
        status="ok", bpw=bpw, snr_db=snr,
        compress_ms=compress_ms, decompress_ms=decompress_ms,
        backend=backend,
    )


def bench_vptq(W: np.ndarray) -> MethodResult:
    """Benchmark VPTQ vector quantisation (primary + residual codebooks)."""
    try:
        from squish.vptq import VPTQConfig, VPTQQuantizer
    except ImportError as exc:
        return MethodResult(status="skip", reason=f"squish.vptq unavailable: {exc}")

    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape

    cfg = VPTQConfig(
        n_codebook_entries=VPTQ_N_PRIMARY,
        group_size=VPTQ_GROUP_SIZE,
        n_residual_entries=VPTQ_N_RESIDUAL,
        n_fit_iters=VPTQ_ITERS,
    )
    quant = VPTQQuantizer(cfg)

    t0 = time.perf_counter()
    layer = quant.compress(W)
    compress_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    W_approx = quant.decompress(layer)
    decompress_ms = (time.perf_counter() - t0) * 1e3

    # BPW = index bits (primary + residual) + float32 col-scales.
    index_bits = layer.compressed_bits        # (log2 primary + log2 residual) × n_groups
    scale_bits = cols * 32                    # float32 per column
    bpw = (index_bits + scale_bits) / (rows * cols)

    snr = _snr_db(W, W_approx)
    return MethodResult(
        status="ok", bpw=bpw, snr_db=snr,
        compress_ms=compress_ms, decompress_ms=decompress_ms,
        backend="vptq-numpy",
    )


def bench_aqlm(W: np.ndarray) -> MethodResult:
    """Benchmark AQLM additive-codebook quantisation (Phase 9A).

    Returns a SKIP result until squish/aqlm.py is implemented.
    """
    try:
        from squish.aqlm import AQLMConfig, AQLMQuantizer  # type: ignore[import]
    except ImportError:
        return MethodResult(
            status="skip",
            reason="squish.aqlm not yet implemented (Phase 9A)",
        )

    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape

    cfg = AQLMConfig()
    quant = AQLMQuantizer(cfg)

    t0 = time.perf_counter()
    layer = quant.compress(W)
    compress_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    W_approx = quant.decompress(layer)
    decompress_ms = (time.perf_counter() - t0) * 1e3

    bpw = layer.compressed_bits / (rows * cols)
    snr = _snr_db(W, W_approx)
    return MethodResult(
        status="ok", bpw=bpw, snr_db=snr,
        compress_ms=compress_ms, decompress_ms=decompress_ms,
        backend="aqlm-numpy",
    )


def bench_quip(W: np.ndarray) -> MethodResult:
    """Benchmark QuIP# E8-lattice + incoherence-processing quantisation (Phase 9B).

    squish/quip_sharp.py uses a ``quantize`` / ``quip_dequantize`` API rather than
    the ``compress`` / ``decompress`` convention of the other methods.
    """
    try:
        from squish.quip_sharp import (  # type: ignore[import]
            QuIPSharpConfig,
            QuIPSharpQuantizer,
            quip_dequantize,
        )
    except ImportError:
        return MethodResult(
            status="skip",
            reason="squish.quip_sharp not available",
        )

    W = np.asarray(W, dtype=np.float32)
    rows, cols = W.shape

    cfg   = QuIPSharpConfig()
    quant = QuIPSharpQuantizer(cfg)

    t0 = time.perf_counter()
    layer = quant.quantize(W)
    compress_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    W_approx = quip_dequantize(layer).astype(np.float32)
    decompress_ms = (time.perf_counter() - t0) * 1e3

    # BPW: 8-bit E8 index + 16-bit residual scale per 8-D chunk.
    # Rotation matrix overhead is excluded (amortised across many layers in a
    # real model; would inflate BPW for the small synthetic weight matrix used
    # in this benchmark).
    n_chunks   = int(layer.e8_indices.size)
    index_bits = n_chunks * 8       # 1.0 bpw
    scale_bits = n_chunks * 16      # 2.0 bpw
    bpw = (index_bits + scale_bits) / (rows * cols)

    snr = _snr_db(W, W_approx)
    return MethodResult(
        status="ok", bpw=bpw, snr_db=snr,
        compress_ms=compress_ms, decompress_ms=decompress_ms,
        backend="quip-numpy",
    )


# Method registry: (key, bench_fn, display_label)
_METHODS: list[tuple[str, Any, str]] = [
    ("int4", bench_int4, "INT4 nibble (baseline)"),
    ("vptq", bench_vptq, "VPTQ (NeurIPS 2025)"),
    ("aqlm", bench_aqlm, "AQLM 2-bit (Phase 9A)"),
    ("quip", bench_quip, "QuIP# 2-bit (Phase 9B)"),
]


# ── stage-2 model evaluation ──────────────────────────────────────────────────

def _load_wikitext(max_tokens: int) -> str:
    """Return a wikitext-2 test-split string.

    Tries the HuggingFace datasets library first; falls back to the bundled
    excerpt when unavailable or the machine is offline.
    """
    try:
        from datasets import load_dataset  # type: ignore[import]

        ds = load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False
        )
        texts: list[str] = []
        target_chars = max_tokens * 6  # rough chars-per-token estimate
        for row in ds:
            text = row["text"].strip()
            if text:
                texts.append(text)
            if sum(len(t) for t in texts) >= target_chars:
                break
        return " ".join(texts)
    except Exception:
        return _WIKITEXT_SAMPLE


def eval_model_perplexity_and_tps(
    model_dir: str,
    max_tokens: int = PPL_MAX_TOKENS,
    tps_tokens: int = TPS_GEN_TOKENS,
) -> dict[str, float | None]:
    """Evaluate FP16 perplexity + TPS for a model via mlx_lm.

    Returns
    -------
    {"perplexity": float | None, "tps": float | None}
    Both are None on any failure.
    """
    result: dict[str, float | None] = {"perplexity": None, "tps": None}

    try:
        import mlx.core as mx  # type: ignore[import]  # noqa: F401
        from mlx_lm import generate, load  # type: ignore[import]
    except ImportError:
        return result

    model, tokenizer = load(model_dir)

    # ── perplexity ────────────────────────────────────────────────────────────
    corpus = _load_wikitext(max_tokens)
    enc = tokenizer(corpus, return_tensors="np", add_special_tokens=True)
    input_ids: np.ndarray = enc["input_ids"][0][:max_tokens]

    window   = 1024
    stride   = 512
    nll_sum  = 0.0
    n_tokens = 0

    for start in range(0, len(input_ids) - 1, stride):
        end      = min(start + window, len(input_ids))
        chunk_in = input_ids[start : end - 1]
        chunk_tgt = input_ids[start + 1 : end]

        import mlx.core as mx  # type: ignore[import]

        x      = mx.array(chunk_in[None, :])
        out    = model(x)
        logits = np.array(out.logits[0])  # (T, vocab)

        for t_idx, tgt_id in enumerate(chunk_tgt):
            logv = logits[t_idx].astype(np.float64)
            log_z = math.log(np.sum(np.exp(logv - logv.max()))) + logv.max()
            nll_sum  += log_z - logv[tgt_id]
            n_tokens += 1

        if end == len(input_ids):
            break

    if n_tokens > 0:
        result["perplexity"] = math.exp(nll_sum / n_tokens)

    # ── throughput ────────────────────────────────────────────────────────────
    try:
        prompt = _WIKITEXT_SAMPLE[:200]
        t0 = time.perf_counter()
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=tps_tokens)
        elapsed = time.perf_counter() - t0
        result["tps"] = tps_tokens / elapsed if elapsed > 0 else None
    except Exception:
        pass

    return result


# ── ASCII table ───────────────────────────────────────────────────────────────

def _print_table(method_results: dict[str, MethodResult]) -> None:
    """Print a colour ASCII comparison table to stdout."""
    col_w = [28, 7, 10, 14, 15, 14, 13]
    hdr_str = (
        f"{'Method':<{col_w[0]}} {'BPW':>{col_w[1]}} {'SNR (dB)':>{col_w[2]}}"
        f" {'Compress ms':>{col_w[3]}} {'Decompress ms':>{col_w[4]}}"
        f" {'Perplexity':>{col_w[5]}} {'TPS':>{col_w[6]}}"
    )
    sep = "─" * sum(col_w + [len(col_w) - 1])

    _hdr("2-bit Quantization Comparison — Results")
    print(f"  {W}{hdr_str}{NC}")
    print(f"  {D}{sep}{NC}")

    for name, _, label in _METHODS:
        r = method_results.get(name)
        if r is None:
            continue
        if r.status == "skip":
            _skip(label, r.reason)
            continue
        if r.status == "error":
            _err(label, r.reason)
            continue

        bpw_s  = f"{r.bpw:.2f}"          if r.bpw           is not None else "—"
        snr_s  = f"{r.snr_db:.1f}"       if r.snr_db        is not None else "—"
        cmp_s  = f"{r.compress_ms:.1f}"  if r.compress_ms   is not None else "—"
        dcmp_s = f"{r.decompress_ms:.2f}"if r.decompress_ms is not None else "—"
        ppl_s  = f"{r.perplexity:.2f}"   if r.perplexity    is not None else "n/a"
        tps_s  = f"{r.tps:.1f}"          if r.tps           is not None else "n/a"

        row_str = (
            f"{label:<{col_w[0]}} {bpw_s:>{col_w[1]}} {snr_s:>{col_w[2]}}"
            f" {cmp_s:>{col_w[3]}} {dcmp_s:>{col_w[4]}}"
            f" {ppl_s:>{col_w[5]}} {tps_s:>{col_w[6]}}"
        )
        print(f"  {G}{row_str}{NC}")

    print(f"  {D}{sep}{NC}")
    print()


def _print_markdown_table(method_results: dict[str, MethodResult]) -> None:
    """Print a Markdown-formatted comparison table to stdout."""
    print("\n## 2-bit Quantization Comparison\n")
    print("| Method | BPW | SNR (dB) | Compress (ms) | Decompress (ms) | Perplexity | TPS |")
    print("|--------|----:|---------:|--------------:|----------------:|-----------:|----:|")

    for name, _, label in _METHODS:
        r = method_results.get(name)
        if r is None or r.status in ("skip", "error"):
            reason = r.reason if r else ""
            print(f"| {label} | — | — | — | — | {reason or '—'} | — |")
            continue

        bpw_s  = f"{r.bpw:.2f}"          if r.bpw           is not None else "—"
        snr_s  = f"{r.snr_db:.1f}"       if r.snr_db        is not None else "—"
        cmp_s  = f"{r.compress_ms:.1f}"  if r.compress_ms   is not None else "—"
        dcmp_s = f"{r.decompress_ms:.2f}"if r.decompress_ms is not None else "—"
        ppl_s  = f"{r.perplexity:.2f}"   if r.perplexity    is not None else "n/a"
        tps_s  = f"{r.tps:.1f}"          if r.tps           is not None else "n/a"
        print(f"| {label} | {bpw_s} | {snr_s} | {cmp_s} | {dcmp_s} | {ppl_s} | {tps_s} |")

    print()


# ── JSON serialisation ────────────────────────────────────────────────────────

def _result_to_dict(r: MethodResult) -> dict[str, Any]:
    """Convert MethodResult → JSON-serialisable dict (inf → None)."""
    d: dict[str, Any] = asdict(r)
    for k, v in d.items():
        if isinstance(v, float) and not math.isfinite(v):
            d[k] = None
    return d


def _dict_to_result(d: dict[str, Any]) -> MethodResult:
    """Reconstruct a MethodResult from a serialised dict."""
    return MethodResult(
        status=d.get("status", "error"),
        reason=d.get("reason", ""),
        bpw=d.get("bpw"),
        snr_db=d.get("snr_db"),
        compress_ms=d.get("compress_ms"),
        decompress_ms=d.get("decompress_ms"),
        perplexity=d.get("perplexity"),
        tps=d.get("tps"),
        backend=d.get("backend", ""),
    )


# ── main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(
    model_dir: str | None = None,
    max_tokens: int = PPL_MAX_TOKENS,
    tps_tokens: int = TPS_GEN_TOKENS,
    dry_run: bool = False,
) -> tuple[dict[str, Any], dict[str, MethodResult]]:
    """Execute all benchmark stages.

    Parameters
    ----------
    model_dir  : path to an mlx_lm model directory for stage-2;
                 None → skip stage-2.
    max_tokens : wikitext-2 tokens for perplexity computation.
    tps_tokens : tokens to generate for TPS measurement.
    dry_run    : skip model loading even if model_dir is provided.

    Returns
    -------
    (results_dict, method_results)
      results_dict   — JSON-serialisable dict (written to disk).
      method_results — dict[method_key → MethodResult] (for table printing).
    """
    # ── stage 1: weight-reconstruction ───────────────────────────────────────
    W = RNG.standard_normal((BENCH_ROWS, BENCH_COLS)).astype(np.float32) * 0.02
    method_results: dict[str, MethodResult] = {}

    _hdr(f"Stage 1 — Weight Reconstruction  ({BENCH_ROWS}×{BENCH_COLS} synthetic weights)")

    for name, bench_fn, label in _METHODS:
        try:
            result = bench_fn(W)
        except Exception as exc:
            result = MethodResult(status="error", reason=str(exc))

        method_results[name] = result

        if result.status == "ok":
            bpw_s  = f"{result.bpw:.2f} bpw"         if result.bpw           is not None else "—"
            snr_s  = f"{result.snr_db:.1f} dB"        if result.snr_db        is not None else "—"
            cmp_s  = f"{result.compress_ms:.1f} ms"   if result.compress_ms   is not None else "—"
            dcmp_s = f"{result.decompress_ms:.2f} ms" if result.decompress_ms is not None else "—"
            _row(
                label, bpw_s,
                f"SNR={snr_s}  compress={cmp_s}  decompress={dcmp_s}"
            )
        elif result.status == "skip":
            _skip(label, result.reason)
        else:
            _err(label, result.reason)

    # ── stage 2: model evaluation ─────────────────────────────────────────────
    model_metrics: dict[str, float | None] = {"perplexity": None, "tps": None}

    if model_dir and not dry_run:
        _hdr(f"Stage 2 — Model Evaluation  ({Path(model_dir).name})")
        model_metrics = eval_model_perplexity_and_tps(
            model_dir, max_tokens=max_tokens, tps_tokens=tps_tokens
        )
        if model_metrics.get("perplexity") is not None:
            _row("FP16 perplexity (wikitext-2)", f"{model_metrics['perplexity']:.3f}")
        else:
            _skip("FP16 perplexity", "evaluation failed or mlx_lm not installed")
        if model_metrics.get("tps") is not None:
            _row("FP16 throughput", f"{model_metrics['tps']:.1f} TPS")
        else:
            _skip("FP16 throughput", "generation failed or mlx_lm not installed")
    elif model_dir and dry_run:
        _hdr("Stage 2 — Model Evaluation  [DRY RUN — skipped]")
    else:
        _hdr("Stage 2 — Model Evaluation  [pass --model-dir to enable]")
        _skip("perplexity", "no model dir provided")
        _skip("tps",        "no model dir provided")

    # ── build result dict ─────────────────────────────────────────────────────
    try:
        import squish as _squish_pkg
        squish_ver = getattr(_squish_pkg, "__version__", "unknown")
    except ImportError:
        squish_ver = "unknown"

    results: dict[str, Any] = {
        "meta": {
            "squish_version": squish_ver,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "weight_shape": [BENCH_ROWS, BENCH_COLS],
            "int4_group_size": INT4_GROUP_SIZE,
            "vptq_config": {
                "n_codebook_entries": VPTQ_N_PRIMARY,
                "group_size": VPTQ_GROUP_SIZE,
                "n_residual_entries": VPTQ_N_RESIDUAL,
                "n_fit_iters": VPTQ_ITERS,
            },
            "model_dir": model_dir,
            "ppl_max_tokens": max_tokens,
            "tps_gen_tokens": tps_tokens,
            "dry_run": dry_run,
        },
        "model_baseline": model_metrics,
        "methods": {
            name: _result_to_dict(r)
            for name, r in method_results.items()
        },
    }

    return results, method_results


# ── CLI entry point ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Squish 2-bit quantization comparison benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--model-dir",
        metavar="PATH",
        default=None,
        help="Path to an mlx_lm-compatible model dir for perplexity + TPS.",
    )
    p.add_argument(
        "--output",
        metavar="PATH",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "--ppl-tokens",
        metavar="N",
        type=int,
        default=PPL_MAX_TOKENS,
        help=f"Wikitext-2 tokens for perplexity (default: {PPL_MAX_TOKENS}).",
    )
    p.add_argument(
        "--tps-tokens",
        metavar="N",
        type=int,
        default=TPS_GEN_TOKENS,
        help=f"Tokens to generate for TPS measurement (default: {TPS_GEN_TOKENS}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Stage-1 (weight reconstruction) only; skip model loading.",
    )
    p.add_argument(
        "--markdown",
        action="store_true",
        help="Also print a Markdown-formatted results table.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    results, method_results = run_benchmark(
        model_dir=args.model_dir,
        max_tokens=args.ppl_tokens,
        tps_tokens=args.tps_tokens,
        dry_run=args.dry_run,
    )

    _print_table(method_results)

    if args.markdown:
        _print_markdown_table(method_results)

    # Write JSON.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(f"{G}✓{NC}  Results written to {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
