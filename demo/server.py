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
    recommended_kv_mode_3tier,
)

INDEX_HTML = ROOT / "demo" / "index.html"

# ── Limits — keep one request comfortably under ~1 second on a laptop ──────
ALLOWED_MODES = ("int8", "int4", "int2")
MAX_CTX_LEN   = 4096
MAX_HEAD_DIM  = 256
MAX_N_HEADS   = 32
INF_SNR_CAP   = 999.0       # JSON cannot represent +inf — cap for serialisation


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
        if self.path in ("/", "/index.html", "/demo", "/demo/"):
            self._send_file(INDEX_HTML, "text/html; charset=utf-8")
            return
        if self.path == "/api/health":
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
        self._send_json(404, {"error": f"GET {self.path} not found"})

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
            results.append(_run_cache(mode, ctx_len, head_dim, n_heads))

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
