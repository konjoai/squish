#!/usr/bin/env python3
"""Squish demo server — exposes the real squish library via a tiny JSON HTTP API.

Stdlib only (http.server + threading) plus numpy + the squish library itself.

Endpoints
---------
GET  /                — serves demo/index.html
GET  /api/health      — returns {"status": "ok", "service": "squish-demo"}
POST /api/compress    — runs ONE QuantizedKVCache(mode=…) end-to-end and
                        returns measured SNR (dB), memory (bytes), wall-clock
                        time (ms), and the actual storage-buffer shape/dtype.
POST /api/recommend   — calls real recommended_kv_mode_3tier(ctx_len).
POST /api/benchmark   — runs INT8 + INT4 + INT2 on the same input and returns
                        a comparative result table with real measured numbers.

All POST bodies are JSON.  All responses are JSON with permissive CORS so the
HTML can be opened either from filesystem (file://) or through this server.

Usage
-----
    python3 demo/server.py
    # → http://127.0.0.1:8001

The server is single-threaded by design — KV-cache benchmarks contend on
NumPy's BLAS pool, so concurrent requests would just slow each other down.
"""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Make squish importable from a fresh checkout ────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                # noqa: E402

from squish.kv.kv_cache import (                                  # noqa: E402
    KV_INT2_AUTO_THRESHOLD,
    KV_INT4_DEFAULT_THRESHOLD,
    QuantizedKVCache,
    estimate_kv_memory,
    recommend_mode_for_budget,
    recommended_kv_mode_3tier,
)
from urllib.parse import parse_qs, urlsplit                       # noqa: E402

INDEX_HTML = ROOT / "demo" / "index.html"
GAME_DIR    = ROOT / "game" / "squish-world"
GAME_INDEX  = GAME_DIR / "index.html"

_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png":  "image/png",
    ".svg":  "image/svg+xml",
    ".ico":  "image/x-icon",
    ".wasm": "application/wasm",
}


# ── Limits — keep one request comfortably under ~1 second on a laptop ──────
ALLOWED_MODES = ("int8", "int4", "int2")
MAX_CTX_LEN   = 4096
MAX_HEAD_DIM  = 256
MAX_N_HEADS   = 32
INF_SNR_CAP   = 999.0       # JSON cannot represent +inf — cap for serialisation

# ── Per-request stats store — updated after every compress / benchmark run ──
# ``GET /api/stats`` returns this; it is None until at least one measurement
# has been made.  The server is single-threaded, so no locking is needed.
_last_stats: "dict | None" = None
# Rolling history for the analytics endpoint: most-recent N results per mode.
_STATS_HISTORY_LEN = 20
_stats_history: "list[dict]" = []


def _record_stats(result: dict) -> None:
    """Update the module-level stats store with a fresh measurement."""
    global _last_stats                                      # noqa: PLW0603
    m = result.get("metrics", {})
    _last_stats = {
        "mode":                result.get("mode"),
        "ctx_len":             result.get("ctx_len"),
        "head_dim":            result.get("head_dim"),
        "n_heads":             result.get("n_heads"),
        "snr_db":              result.get("snr_db"),
        "memory_bytes":        result.get("memory_bytes"),
        "compression_ratio":   result.get("compression_ratio"),
        "elapsed_ms":          result.get("elapsed_ms"),
        "tokens_compressed":   m.get("tokens_compressed", 0),
        "tokens_fp16":         m.get("tokens_fp16", 0),
        "tokens_evicted":      m.get("tokens_evicted", 0),
        "bits_used":           m.get("bits_used", 0),
        "memory_saved_bytes":  m.get("memory_saved_bytes", 0),
        "cache_hit_rate":      m.get("cache_hit_rate", 0.0),
        "ts":                  time.time(),
    }
    _stats_history.append(_last_stats)
    if len(_stats_history) > _STATS_HISTORY_LEN:
        _stats_history.pop(0)


# ── /api/calculate limits — pure arithmetic, no cache run, so cap by sanity ─
ALLOWED_BITS  = (16, 8, 4, 2)
MIN_PARAMS_B  = 0.05      # 50 M params — smaller than this is degenerate
MAX_PARAMS_B  = 1000.0    # 1 T — beyond this the closed-form is meaningless
MAX_CALC_CTX  = 1_000_000
KV_BUFFERS    = 2         # K + V
FP16_BPP      = 2         # bytes per parameter at fp16

# Architecture lookup keyed on parameter count.  Numbers picked to match the
# real (n_layers, n_kv_heads, head_dim) of widely-shipped open-weight models —
# Qwen2.5, Llama 3, Phi-3, Mistral 7B — so the closed-form KV math lines up
# with what the actual model would allocate.  The mapping is a reasonable
# default; callers who know their architecture should pass it explicitly.
_ARCH_TABLE: tuple = (
    # (params_b_max, n_layers, n_kv_heads, head_dim, label)
    (0.6,           24,  2, 64,  "0.5B-class"),
    (1.0,           24,  2, 128, "1B-class"),
    (2.0,           28,  2, 128, "1.5B–2B-class"),
    (4.0,           32,  4, 128, "3B-class"),
    (9.0,           32,  8, 128, "7B–8B-class"),
    (15.0,          40,  8, 128, "13B-class"),
    (35.0,          48,  8, 128, "30B-class"),
    (75.0,          80,  8, 128, "70B-class"),
    (float("inf"),  96,  8, 128, "100B+-class"),
)


def _arch_for_params(params_b: float) -> dict:
    """Pick a representative (n_layers, n_kv_heads, head_dim) for ``params_b``.

    The table is keyed on real Qwen / Llama / Phi architectures.  Returned
    keys: ``n_layers``, ``n_kv_heads``, ``head_dim``, ``label`` (the human
    name of the bracket).
    """
    for cap, n_layers, n_kv_heads, head_dim, label in _ARCH_TABLE:
        if params_b <= cap:
            return {
                "n_layers":   n_layers,
                "n_kv_heads": n_kv_heads,
                "head_dim":   head_dim,
                "label":      label,
            }
    # Unreachable: the last row caps at +inf.
    raise RuntimeError("arch table exhausted — unreachable")


def _bits_to_mode(bits: int) -> str:
    """Map a precision bit-width to the matching squish KV-cache mode label."""
    if bits == 16: return "fp16"
    if bits ==  8: return "int8"
    if bits ==  4: return "int4"
    if bits ==  2: return "int2"
    raise ValueError(f"precision_bits must be one of {ALLOWED_BITS}, got {bits}")


def _calculate(model_params_b: float, context_len: int,
               precision_bits: int) -> dict:
    """Closed-form total memory at a given (model size, context, precision).

    Uses ``squish.kv.kv_cache.estimate_kv_memory`` for the KV side — the
    same code path production ``make_kv_cache`` callers hit — and
    ``params * bits / 8`` for the weight side.  No model load, no cache
    construction, no GPU work.
    """
    if not (MIN_PARAMS_B <= model_params_b <= MAX_PARAMS_B):
        raise ValueError(
            f"model_params_b must be in [{MIN_PARAMS_B}, {MAX_PARAMS_B}], "
            f"got {model_params_b}"
        )
    if not (0 < context_len <= MAX_CALC_CTX):
        raise ValueError(
            f"context_len must be in (0, {MAX_CALC_CTX}], got {context_len}"
        )
    mode = _bits_to_mode(precision_bits)        # raises on bad bits

    arch = _arch_for_params(model_params_b)
    params = int(model_params_b * 1e9)

    # Weights: params × bits / 8.
    weights_bytes      = params * precision_bits // 8
    weights_fp16_bytes = params * FP16_BPP

    # KV cache: defer to the production closed-form, including the FP16
    # baseline that estimate_kv_memory computes internally.
    est = estimate_kv_memory(
        n_layers=arch["n_layers"],
        n_kv_heads=arch["n_kv_heads"],
        head_dim=arch["head_dim"],
        context_tokens=context_len,
        mode=mode,
    )

    fp16_total = weights_fp16_bytes + est.fp16_baseline_bytes
    total      = weights_bytes      + est.total_bytes
    savings    = fp16_total - total
    ratio      = (fp16_total / total) if total > 0 else 1.0

    return {
        "model_params_b":          model_params_b,
        "context_len":             context_len,
        "precision_bits":          precision_bits,
        "mode":                    mode,
        "arch":                    arch,
        "weights_bytes":           weights_bytes,
        "weights_fp16_bytes":      weights_fp16_bytes,
        "kv_cache_bytes":          est.total_bytes,
        "kv_cache_fp16_bytes":     est.fp16_baseline_bytes,
        "kv_compression_ratio":    est.compression_ratio,
        "kv_bytes_per_token":      est.bytes_per_token,
        "total_bytes":             total,
        "fp16_total_bytes":        fp16_total,
        "savings_bytes":           savings,
        "total_compression_ratio": ratio,
    }


# ── W109 GET /api/recommend limits ────────────────────────────────────────
GET_REC_MAX_MODEL_B    = 200
GET_REC_MAX_CTX_LEN    = 1_000_000
GET_REC_MAX_BUDGET_MB  = 1_000_000

_REC_ARCH_TABLE: tuple = (
    # (size_b, n_layers, n_kv_heads, head_dim, label)
    (0.5,  24, 2,  64, "Qwen2.5-0.5B"),
    (1.5,  28, 2, 128, "Qwen2.5-1.5B"),
    (3.0,  36, 2, 128, "Qwen2.5-3B"),
    (7.0,  28, 4, 128, "Qwen2.5-7B"),
    (8.0,  32, 8, 128, "Llama-3.1-8B"),
    (14.0, 48, 8, 128, "Qwen2.5-14B"),
    (32.0, 64, 8, 128, "Qwen2.5-32B"),
    (70.0, 80, 8, 128, "Llama-3.1-70B"),
)

# ── measurement helpers ────────────────────────────────────────────────────


def _snr_db(signal: np.ndarray, recon: np.ndarray) -> float:
    s = signal.astype(np.float64)
    r = recon.astype(np.float64)
    sig_pow = float(np.mean(s * s))
    err_pow = float(np.mean((s - r) ** 2))
    if sig_pow == 0:
        return 0.0
    if err_pow == 0:
        return INF_SNR_CAP
    return float(10.0 * np.log10(sig_pow / err_pow))


def _run_cache(mode: str, ctx_len: int, head_dim: int, n_heads: int,
               seed: int = 7) -> dict:
    """Construct a real ``QuantizedKVCache`` of ``mode``, push ``ctx_len``
    synthetic tokens through ``.update()``, then measure:

      * end-to-end ``.update()`` wall-clock time (ms)
      * RAM held by the layer's compressed buffer (bytes)
      * SNR of ``.get_full_kv()`` reconstruction vs the float16 ground truth

    The cache window is fixed at 2 so almost every token gets quantised — the
    SNR therefore reflects the actual codec, not the FP16 recent window.
    """
    rng = np.random.default_rng(seed)
    keys = [rng.standard_normal((n_heads, head_dim)).astype(np.float16) * 0.3
            for _ in range(ctx_len)]
    vals = [rng.standard_normal((n_heads, head_dim)).astype(np.float16) * 0.3
            for _ in range(ctx_len)]
    cache = QuantizedKVCache(n_layers=1, window=2, mode=mode)

    t0 = time.perf_counter()
    for k, v in zip(keys, vals):
        cache.update(0, k, v)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    layer = cache._layers[0]
    full_k, _ = layer.get_full_kv()
    ground_truth = np.stack(keys, axis=1)        # (n_heads, ctx_len, head_dim)
    snr = _snr_db(ground_truth, full_k)

    fp16_baseline_bytes = ctx_len * n_heads * head_dim * 2 * 2  # K + V, f16
    # Capture CompressionResult before the cache is GC'd.
    cr = cache.metrics()
    return {
        "mode":               mode,
        "ctx_len":            ctx_len,
        "head_dim":           head_dim,
        "n_heads":            n_heads,
        "snr_db":             snr,
        "memory_bytes":       int(layer.memory_bytes),
        "fp16_baseline_bytes": fp16_baseline_bytes,
        "compression_ratio":  fp16_baseline_bytes / max(1, layer.memory_bytes),
        "elapsed_ms":         elapsed_ms,
        "old_q_shape":  list(layer.keys_old_q.shape)
                        if layer.keys_old_q is not None else None,
        "old_q_dtype":  str(layer.keys_old_q.dtype)
                        if layer.keys_old_q is not None else None,
        "metrics": {
            "tokens_compressed":  cr.tokens_compressed,
            "tokens_fp16":        cr.tokens_fp16,
            "tokens_evicted":     cr.tokens_evicted,
            "bits_used":          cr.bits_used,
            "memory_saved_bytes": cr.memory_saved_bytes,
            "cache_hit_rate":     cr.cache_hit_rate,
        },
    }


# ── Param validation ──────────────────────────────────────────────────────


def _int_param(payload: dict, key: str, default: int, lo: int, hi: int) -> int:
    raw = payload.get(key, default)
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key!r} must be an integer, got {raw!r}")
    if not (lo <= v <= hi):
        raise ValueError(f"{key!r} must be in [{lo}, {hi}], got {v}")
    return v


def _arch_for_model_size_b(model_size_b: float) -> dict:
    """Snap ``model_size_b`` to the closest preset in ``_REC_ARCH_TABLE``.

    Tie-breaks toward the *larger* of two equidistant rows — memory
    planning should overestimate, not underestimate.
    """
    best_idx = 0
    best_delta = float("inf")
    for idx, (size_b, *_rest) in enumerate(_REC_ARCH_TABLE):
        delta = abs(size_b - model_size_b)
        if delta < best_delta or (delta == best_delta and size_b > _REC_ARCH_TABLE[best_idx][0]):
            best_delta = delta
            best_idx = idx
    size_b, n_layers, n_kv_heads, head_dim, label = _REC_ARCH_TABLE[best_idx]
    return {
        "preset_size_b": size_b,
        "n_layers":      n_layers,
        "n_kv_heads":    n_kv_heads,
        "head_dim":      head_dim,
        "preset_label":  label,
    }


def _query_int(qs: dict, key: str, default: "int | None", lo: int, hi: int,
               *, required: bool = False) -> "int | None":
    """Validate an integer query parameter (urlsplit().query parsed dict)."""
    raw_list = qs.get(key)
    if not raw_list:
        if required:
            raise ValueError(f"missing required query parameter {key!r}")
        return default
    raw = raw_list[0]
    try:
        v = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key!r} must be an integer, got {raw!r}")
    if not (lo <= v <= hi):
        raise ValueError(f"{key!r} must be in [{lo}, {hi}], got {v}")
    return v


def _query_float(qs: dict, key: str, default: "float | None", lo: float, hi: float,
                 *, required: bool = False) -> "float | None":
    raw_list = qs.get(key)
    if not raw_list:
        if required:
            raise ValueError(f"missing required query parameter {key!r}")
        return default
    raw = raw_list[0]
    try:
        v = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"{key!r} must be a number, got {raw!r}")
    if not (lo <= v <= hi):
        raise ValueError(f"{key!r} must be in [{lo}, {hi}], got {v}")
    return v


def _format_mb(byte_count: int) -> float:
    return round(byte_count / 1e6, 2)


def _build_recommendation(
    model_size_b: float,
    ctx_len: int,
    budget_mb: "float | None",
    *,
    window: int = 128,
) -> dict:
    """Closed-form recommendation. No tokens generated; no model loaded."""
    arch = _arch_for_model_size_b(model_size_b)
    n_layers, n_kv_heads, head_dim = arch["n_layers"], arch["n_kv_heads"], arch["head_dim"]

    by_context = recommended_kv_mode_3tier(ctx_len)

    by_budget: "str | None" = None
    if budget_mb is not None:
        budget_bytes = int(budget_mb * 1e6)
        by_budget = recommend_mode_for_budget(
            n_layers, n_kv_heads, head_dim, ctx_len, budget_bytes, window=window,
        )

    # Resolution: budget wins when both are present and they disagree, since
    # picking by context alone risks an OOM if the budget is binding. The
    # "basis" field tells the caller which constraint actually decided.
    if by_budget is None and budget_mb is not None:
        mode = "none"
        basis = "budget"
    elif budget_mb is None:
        mode  = by_context
        basis = "context"
    else:
        mode  = by_budget
        basis = "budget" if by_budget != by_context else "agreement"

    memory_mb: dict = {}
    for tier in ("fp16", "int8", "int4", "int2"):
        try:
            est = estimate_kv_memory(
                n_layers, n_kv_heads, head_dim, ctx_len, tier, window=window,
            )
        except ValueError:
            continue
        memory_mb[tier] = _format_mb(est.total_bytes + est.recent_window_bytes)

    # Reasoning string — short, factual, no marketing.
    base_phrases = {
        "int8": (
            f"INT8 (~{memory_mb.get('int8', 0):.0f} MB total) — full "
            f"quality, default for context ≤ {KV_INT2_AUTO_THRESHOLD:,}."
        ),
        "int4": (
            f"INT4 (~{memory_mb.get('int4', 0):.0f} MB total, ~22 dB SNR) "
            f"— squish's recommended band "
            f"({KV_INT2_AUTO_THRESHOLD:,}-{KV_INT4_DEFAULT_THRESHOLD:,} tokens)."
        ),
        "int2": (
            f"INT2 with Hadamard rotation (~{memory_mb.get('int2', 0):.0f} "
            f"MB total, 7× smaller than fp16) — keeps long contexts "
            f"inside a typical RAM budget."
        ),
    }
    if mode == "none":
        reason = (
            f"Even INT2 (~{memory_mb.get('int2', 0):.0f} MB) exceeds the "
            f"{budget_mb:.0f} MB budget at {ctx_len:,} tokens — shrink "
            f"context, layers, or pick a smaller model."
        )
    elif basis == "budget" and by_budget != by_context:
        # Budget made the call and it differs from the pure-context pick.
        # Show both to make the trade-off explicit.
        reason = (
            f"{model_size_b}B model · {ctx_len:,} tokens · "
            f"{budget_mb:.0f} MB budget → {base_phrases[mode]} "
            f"(context alone would have picked {by_context}; the budget "
            f"can fit a higher-quality tier — choose accordingly)."
        )
    else:
        reason = f"{model_size_b}B model · {ctx_len:,} tokens → {base_phrases[mode]}"

    return {
        "model_size_b":    model_size_b,
        "ctx_len":         ctx_len,
        "budget_mb":       budget_mb,
        "mode":            mode,
        "reason":          reason,
        "basis":           basis,
        "by_context":      by_context,
        "by_budget":       by_budget,
        "arch":            arch,
        "memory_mb":       memory_mb,
        "thresholds": {
            "int4_above": KV_INT2_AUTO_THRESHOLD,
            "int2_above": KV_INT4_DEFAULT_THRESHOLD,
        },
        "live": True,
    }


def _check_mode_compat(mode: str, head_dim: int) -> None:
    if mode == "int2" and head_dim % 4:
        raise ValueError(
            f"INT2 mode requires head_dim divisible by 4 (got {head_dim})"
        )
    if mode == "int4" and head_dim % 2:
        raise ValueError(
            f"INT4 mode requires head_dim divisible by 2 (got {head_dim})"
        )


# ── HTTP handler ──────────────────────────────────────────────────────────


class Handler(BaseHTTPRequestHandler):
    server_version = "SquishDemo/1.0"

    # ── helpers ──────────────────────────────────────────────────────────
    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age",       "86400")

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type",   "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str) -> None:
        if not path.exists():
            self._send_json(404, {"error": f"{path.name} not found at {path}"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type",   ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",  "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n == 0:
            return {}
        raw = self.rfile.read(n).decode("utf-8")
        if not raw.strip():
            return {}
        return json.loads(raw)

    def log_message(self, fmt: str, *args) -> None:
        ts = time.strftime("%H:%M:%S")
        sys.stdout.write(
            f"  \033[2m[{ts}]\033[0m {self.address_string()}  {fmt % args}\n"
        )
        sys.stdout.flush()

    # ── HTTP verbs ───────────────────────────────────────────────────────
    def do_OPTIONS(self) -> None:                                    # noqa: N802
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:                                        # noqa: N802
        split = urlsplit(self.path)
        path = split.path
        if path in ("/", "/index.html", "/demo", "/demo/"):
            self._send_file(INDEX_HTML, "text/html; charset=utf-8")
            return
        if path == "/api/health":
            self._send_json(200, {
                "status":      "ok",
                "service":     "squish-demo",
                "thresholds":  {
                    "int4_above": KV_INT2_AUTO_THRESHOLD,
                    "int2_above": KV_INT4_DEFAULT_THRESHOLD,
                },
                "limits":      {
                    "max_ctx_len":  MAX_CTX_LEN,
                    "max_head_dim": MAX_HEAD_DIM,
                    "max_n_heads":  MAX_N_HEADS,
                },
            })
            return
        if path == "/api/recommend":
            try:
                qs = parse_qs(split.query, keep_blank_values=False)
                model_size_b = _query_float(
                    qs, "model_size_b", default=None,
                    lo=0.05, hi=GET_REC_MAX_MODEL_B, required=True,
                )
                ctx_len = _query_int(
                    qs, "ctx_len", default=None,
                    lo=1, hi=GET_REC_MAX_CTX_LEN, required=True,
                )
                budget_mb = _query_float(
                    qs, "budget_mb", default=None,
                    lo=1.0, hi=GET_REC_MAX_BUDGET_MB,
                )
            except ValueError as exc:
                self._send_json(400, {"error": str(exc)})
                return
            payload = _build_recommendation(model_size_b, ctx_len, budget_mb)
            self._send_json(200, payload)
            return
        if path == "/api/stats":
            if _last_stats is None:
                self._send_json(503, {
                    "error": "No measurements yet — run /api/compress or /api/benchmark first.",
                })
            else:
                self._send_json(200, {
                    "last":    _last_stats,
                    "history": _stats_history[-_STATS_HISTORY_LEN:],
                    "n_runs":  len(_stats_history),
                    "live":    True,
                })
            return
        self._send_json(404, {"error": f"GET {path} not found"})

    def do_POST(self) -> None:                                       # noqa: N802
        try:
            payload = self._read_json()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON: {exc}"})
            return

        try:
            if self.path == "/api/compress":
                self._handle_compress(payload)
                return
            if self.path == "/api/recommend":
                self._handle_recommend(payload)
                return
            if self.path == "/api/benchmark":
                self._handle_benchmark(payload)
                return
            if self.path in ("/api/calculate", "/calculate"):
                self._handle_calculate(payload)
                return
            self._send_json(404, {"error": f"POST {self.path} not found"})
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:                                       # pragma: no cover
            sys.stderr.write(f"  ERROR: {type(exc).__name__}: {exc}\n")
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    # ── Route handlers ───────────────────────────────────────────────────
    def _handle_compress(self, payload: dict) -> None:
        mode = payload.get("mode", "int8")
        if mode not in ALLOWED_MODES:
            raise ValueError(f"mode must be one of {ALLOWED_MODES}, got {mode!r}")
        ctx_len  = _int_param(payload, "ctx_len",   512, 16, MAX_CTX_LEN)
        head_dim = _int_param(payload, "head_dim",  128,  4, MAX_HEAD_DIM)
        n_heads  = _int_param(payload, "n_heads",     4,  1, MAX_N_HEADS)
        _check_mode_compat(mode, head_dim)
        result = _run_cache(mode, ctx_len, head_dim, n_heads)
        result["live"] = True
        _record_stats(result)
        self._send_json(200, result)

    def _handle_recommend(self, payload: dict) -> None:
        ctx_len = _int_param(payload, "ctx_len", 8192, 0, 1_000_000)
        mode = recommended_kv_mode_3tier(ctx_len)
        reasons = {
            "int8": "Short context — full quality, smallest memory cost.",
            "int4": "Medium context — half the memory at excellent quality.",
            "int2": "Long context — maximum compression for >16K tokens.",
        }
        self._send_json(200, {
            "ctx_len":   ctx_len,
            "mode":      mode,
            "reason":    reasons[mode],
            "thresholds": {
                "int4_above": KV_INT2_AUTO_THRESHOLD,
                "int2_above": KV_INT4_DEFAULT_THRESHOLD,
            },
            "live": True,
        })

    def _handle_calculate(self, payload: dict) -> None:
        # model_params_b: floats are allowed (1.5, 7.0, 70.0, …) so we don't
        # use _int_param here — just bound-check explicitly.
        raw_p = payload.get("model_params_b", payload.get("params_b"))
        if raw_p is None:
            raise ValueError("'model_params_b' is required")
        try:
            params_b = float(raw_p)
        except (TypeError, ValueError):
            raise ValueError(
                f"'model_params_b' must be a number, got {raw_p!r}"
            )

        context_len = _int_param(
            payload, "context_len", 8_192, 1, MAX_CALC_CTX,
        )
        precision_bits = _int_param(
            payload, "precision_bits", 4, 2, 16,
        )
        if precision_bits not in ALLOWED_BITS:
            raise ValueError(
                f"precision_bits must be one of {ALLOWED_BITS}, "
                f"got {precision_bits}"
            )

        result = _calculate(params_b, context_len, precision_bits)
        result["live"] = True
        self._send_json(200, result)

    def _handle_benchmark(self, payload: dict) -> None:
        ctx_len  = _int_param(payload, "ctx_len",  1024, 16, MAX_CTX_LEN)
        head_dim = _int_param(payload, "head_dim",  128,  4, MAX_HEAD_DIM)
        n_heads  = _int_param(payload, "n_heads",    4,  1, MAX_N_HEADS)

        results = []
        for mode in ALLOWED_MODES:
            try:
                _check_mode_compat(mode, head_dim)
            except ValueError as exc:
                results.append({"mode": mode, "skipped": True, "reason": str(exc)})
                continue
            r = _run_cache(mode, ctx_len, head_dim, n_heads)
            _record_stats(r)
            results.append(r)

        # FP16 reference row — not from the cache, but the same byte budget the
        # cache is being measured against, so the table makes sense.
        fp16_bytes = ctx_len * n_heads * head_dim * 2 * 2
        self._send_json(200, {
            "ctx_len":  ctx_len,
            "head_dim": head_dim,
            "n_heads":  n_heads,
            "fp16_baseline_bytes": fp16_bytes,
            "results":  results,
            "live": True,
        })


# ── entry point ────────────────────────────────────────────────────────────


def main(host: str = "127.0.0.1", port: int = 8001) -> int:
    httpd = HTTPServer((host, port), Handler)
    print()
    print("  ┌─────────────────────────────────────────────────────────────┐")
    print("  │  Squish demo server                                         │")
    print(f"  │  running at http://{host}:{port}".ljust(63) + "│")
    print("  │                                                             │")
    print("  │  Open the URL above in a browser, or open demo/index.html   │")
    print("  │  directly — either way it will detect this server and       │")
    print("  │  switch to live mode.                                       │")
    print("  │                                                             │")
    print("  │  Press Ctrl-C to stop.                                      │")
    print("  └─────────────────────────────────────────────────────────────┘")
    print()
    print(f"  importing real squish.kv.kv_cache from: {ROOT}")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  shutdown — bye 👋")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
