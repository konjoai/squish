#!/usr/bin/env python3
"""v4 measured benchmark: Squish daemon + KV cache vs Ollama warm.

Three configurations:

1. ``ollama``         — long-running ``ollama serve`` (warm; one priming request before measurement)
2. ``squish_daemon``  — long-running ``python -m squish.server`` started exactly like
                        ``squish daemon start`` (no UDS — that path is broken on this
                        branch for mlx-native quant; see PRECHECK.md)
3. ``squish_kv``      — same long-running server PLUS ``--disk-prompt-cache`` +
                        ``--kv-cache-mode int8`` (the only flag combo that actually
                        populates the disk cache; PromptKVStore is not wired up)

For each config we measure:

* ``ttft_first_s``    — first request after a long warm-up. Comparable across all configs.
* ``ttft_repeat_s``   — second (or later) request with the **same exact prompt**.
                        For squish_kv this is the cache-hit path. For the others it's a
                        baseline second-request TTFT.
* ``warm_tps``        — sustained tokens/sec on a 200-token decode.
* ``peak_rss_bytes``  — full process-tree peak RSS sampled every 50 ms.
* ``disk_bytes``      — on-disk model size.

Per-run raw data is written to ``results/benchmarks_v4/runs/<timestamp>.json``.

Spec decode (the v4 PR's other headline feature) is *not* benchmarked here —
``--draft-model`` crashes during server init on this branch:

    ImportError: cannot import name 'load_draft_model' from 'squish.speculative'

See ``results/benchmarks_v4/PRECHECK.md`` for the full feature-by-feature
audit of what v4 actually ships vs what its methodology.json claims.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import statistics as stats
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import psutil

# ── Config ────────────────────────────────────────────────────────────────────

REPEAT_PROMPT = (
    "You are a senior engineer reviewing a pull request. The PR adds a new "
    "Redis-backed session cache to the auth service. The diff is roughly 240 "
    "lines and touches login, logout, and a small middleware shim. Give a "
    "two-sentence review covering the most important risk."
)
"""A long-ish, realistic prompt; the cache test re-sends this exact text."""

UNIQUE_PROMPT_PREFIX = "Once upon a time on a tiny island called {city}, "
"""Used to construct unique prompts per first-request run so we measure cold-prefill TTFT."""

UNIQUE_PROMPT_CITIES = [
    "Hökenäs",
    "Lobamba",
    "Mariehamn",
    "Avarua",
    "Yaren",
    "Ouanaminthe",
    "Asunción",
    "Buenos Aires",
]

WARM_TPS_PROMPT = "Write a 200-word description of the Renaissance in two paragraphs."

RUNS = 5
"""Number of measurement runs for each metric. Reported as median + min/max/p95."""

WARMUP_REQUESTS = 2
"""Throwaway requests after server start to fully warm the model + Metal kernels."""

OLLAMA_BIN = "/usr/local/bin/ollama"
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:7b"

SQUISH_BIN = "/Users/wscholl/squish/.venv/bin/squish"
SQUISH_PY = "/Users/wscholl/squish/.venv/bin/python"
SQUISH_HOST = "127.0.0.1"
SQUISH_PORT = 11435
SQUISH_API_KEY = "squish"
SQUISH_MODEL_PATH = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"

KV_CACHE_DIR = "/tmp/squish_kv_v4"
PROMPT_KV_DIR = "/tmp/squish_prompt_kv_v4_1"
DRAFT_MODEL_PATH = "/Users/wscholl/models/Qwen2.5-1.5B-Instruct-int4"

REPO_ROOT = Path(__file__).resolve().parents[2]
# Results subdirectory: v4 for the original 3-config bench, v4_1 for the
# wired-features re-run with --prompt-kv-cache and --draft-model added.
_V4_1 = os.environ.get("SQUISH_BENCH_V4_1", "1") == "1"
OUT_ROOT = REPO_ROOT / "results" / ("benchmarks_v4_1" if _V4_1 else "benchmarks_v4") / "runs"
TS = time.strftime("%Y%m%dT%H%M%S")
OUT_DIR = OUT_ROOT / TS
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR = Path("/tmp/bench_v4_logs")
LOG_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Process / readiness helpers ────────────────────────────────────────────────


def kill_all_serving() -> None:
    patterns = [
        "ollama serve",
        "ollama runner",
        "ollama_llama_server",
        "Ollama.app",
        "Ollama Helper",
        "squish.server",
        "squishd",
    ]
    for p in patterns:
        subprocess.run(["pkill", "-f", p], capture_output=True)
    time.sleep(3)


def wait_ready(url: str, timeout: float = 240) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                r.read()
                return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            time.sleep(0.1)
    return False


class RSSSampler(threading.Thread):
    """Sample peak RSS of a process tree every 50 ms until stop()."""

    def __init__(self, root_pid: int) -> None:
        super().__init__(daemon=True)
        self.root_pid = root_pid
        self._stop = threading.Event()
        self.peak_bytes = 0
        self.samples = 0

    def run(self) -> None:
        try:
            root = psutil.Process(self.root_pid)
        except psutil.NoSuchProcess:
            return
        while not self._stop.is_set():
            try:
                tree = root.memory_info().rss
                for child in root.children(recursive=True):
                    try:
                        tree += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            if tree > self.peak_bytes:
                self.peak_bytes = tree
            self.samples += 1
            time.sleep(0.05)

    def stop(self) -> None:
        self._stop.set()
        self.join(timeout=2)


# ── HTTP streaming clients ────────────────────────────────────────────────────


def stream_ollama(prompt: str, max_tokens: int = 100) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0.0},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t_req = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    eval_count = 0
    eval_duration_ns = 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = d.get("response", "")
            if chunk and t_first is None:
                t_first = time.perf_counter()
            if chunk:
                parts.append(chunk)
            if d.get("done"):
                eval_count = d.get("eval_count") or 0
                eval_duration_ns = d.get("eval_duration") or 0
    t_done = time.perf_counter()
    tps = eval_count / (eval_duration_ns / 1e9) if (eval_count and eval_duration_ns) else None
    return {
        "ttft_s": (t_first - t_req) if t_first else None,
        "total_s": t_done - t_req,
        "completion_tokens": eval_count or len(parts),
        "tokens_per_sec": tps,
        "response": "".join(parts),
    }


def stream_squish(prompt: str, max_tokens: int = 100) -> dict[str, Any]:
    body = json.dumps(
        {
            "model": "squish",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SQUISH_API_KEY}",
        },
    )
    t_req = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    completion_tokens: int | None = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.strip()
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            choices = d.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content") or ""
                if chunk and t_first is None:
                    t_first = time.perf_counter()
                if chunk:
                    parts.append(chunk)
            usage = d.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
    t_done = time.perf_counter()
    n_tokens = completion_tokens or len(parts)
    gen_window = (t_done - t_first) if t_first else None
    tps = (n_tokens / gen_window) if (gen_window and gen_window > 0 and n_tokens) else None
    return {
        "ttft_s": (t_first - t_req) if t_first else None,
        "total_s": t_done - t_req,
        "completion_tokens": n_tokens,
        "tokens_per_sec": tps,
        "response": "".join(parts),
    }


# ── Server launchers ──────────────────────────────────────────────────────────


def start_ollama(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    env = {**os.environ, "OLLAMA_HOST": f"{OLLAMA_HOST}:{OLLAMA_PORT}"}
    proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        env=env,
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def _squish_server_cmd(extra: list[str]) -> list[str]:
    return [
        SQUISH_PY,
        "-m",
        "squish.server",
        "--mlx-model-dir",
        SQUISH_MODEL_PATH,
        "--port",
        str(SQUISH_PORT),
        "--host",
        SQUISH_HOST,
        "--log-level",
        "warning",
        *extra,
    ]


def start_squish_daemon(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    """Same code path as `squish daemon start` (just `python -m squish.server`)."""
    proc = subprocess.Popen(
        _squish_server_cmd([]),
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def start_squish_kv(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    """squish.server with the disk prompt cache + int8 KV mode (the only combo that works)."""
    # Clean cache dir between configs so the first prompt is a true miss
    if os.path.isdir(KV_CACHE_DIR):
        shutil.rmtree(KV_CACHE_DIR)
    os.makedirs(KV_CACHE_DIR, exist_ok=True)
    extra = [
        "--kv-cache-mode",
        "int8",
        "--disk-prompt-cache",
        KV_CACHE_DIR,
        "--disk-prompt-cache-size",
        "32",
    ]
    proc = subprocess.Popen(
        _squish_server_cmd(extra),
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def start_squish_pkv(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    """v4.1 Fix 2: squish.server with the new --prompt-kv-cache flag (fp16 path)."""
    if os.path.isdir(PROMPT_KV_DIR):
        shutil.rmtree(PROMPT_KV_DIR)
    os.makedirs(PROMPT_KV_DIR, exist_ok=True)
    extra = [
        "--prompt-kv-cache",
        PROMPT_KV_DIR,
        "--prompt-kv-cache-max-gb",
        "1.0",
    ]
    proc = subprocess.Popen(
        _squish_server_cmd(extra),
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def start_squish_spec(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    """v4.1 Fix 1: squish.server with --draft-model for speculative decoding."""
    extra = [
        "--draft-model",
        DRAFT_MODEL_PATH,
    ]
    proc = subprocess.Popen(
        _squish_server_cmd(extra),
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def stop_server(proc: subprocess.Popen, sampler: RSSSampler) -> None:
    sampler.stop()
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


# ── Per-config benchmark ──────────────────────────────────────────────────────

CONFIGS = {
    "ollama": {
        "label": "Ollama (warm)",
        "ready_url": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version",
        "start": start_ollama,
        "stream": stream_ollama,
    },
    "squish_daemon": {
        "label": "Squish daemon (warm)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start": start_squish_daemon,
        "stream": stream_squish,
    },
    "squish_kv": {
        "label": "Squish + disk KV cache (legacy)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start": start_squish_kv,
        "stream": stream_squish,
    },
    # v4.1 Fix 2: new fp16 prompt-kv-cache path
    "squish_pkv": {
        "label": "Squish + --prompt-kv-cache (v4.1)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start": start_squish_pkv,
        "stream": stream_squish,
    },
    # v4.1 Fix 1: speculative decoding wired
    "squish_spec": {
        "label": "Squish + spec decode (v4.1)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start": start_squish_spec,
        "stream": stream_squish,
    },
}


def run_config(cfg_id: str) -> dict[str, Any]:
    cfg = CONFIGS[cfg_id]
    log(f"=== {cfg_id} : start server ===")
    kill_all_serving()
    log_path = LOG_DIR / f"{cfg_id}_{TS}.log"
    t_kill = time.perf_counter()
    proc, sampler = cfg["start"](log_path)
    try:
        if not wait_ready(cfg["ready_url"], timeout=240):
            raise RuntimeError(f"{cfg_id} did not become ready")
        t_ready = time.perf_counter()
        log(f"  ready in {t_ready - t_kill:.2f}s; priming")
        # Warm-up: throwaway requests, not measured
        for i in range(WARMUP_REQUESTS):
            cfg["stream"]("Hello.", max_tokens=8)

        # ── Phase A: TTFT on a FRESH prompt each run (cold-prefill TTFT) ──
        ttft_first: list[dict[str, Any]] = []
        for i in range(RUNS):
            prompt = UNIQUE_PROMPT_PREFIX.format(city=UNIQUE_PROMPT_CITIES[i]) + (
                "Write a single short sentence about its weather."
            )
            d = cfg["stream"](prompt, max_tokens=1)
            ttft_first.append({"run": i + 1, "prompt": prompt, **d})
            log(f"  ttft_first run {i + 1}: {d['ttft_s']:.3f}s")

        # ── Phase B: TTFT on the SAME prompt repeated (cache-hit TTFT for squish_kv) ──
        # The very first send populates the cache. Then we send the same prompt
        # RUNS times and record TTFT for the cache-hit reads.
        # First send: populate cache (not counted).
        cfg["stream"](REPEAT_PROMPT, max_tokens=1)
        ttft_repeat: list[dict[str, Any]] = []
        for i in range(RUNS):
            d = cfg["stream"](REPEAT_PROMPT, max_tokens=1)
            ttft_repeat.append({"run": i + 1, **d})
            log(f"  ttft_repeat run {i + 1}: {d['ttft_s']:.3f}s")

        # ── Phase C: sustained throughput (warm tok/s on 200-token decode) ──
        warm_tps: list[dict[str, Any]] = []
        for i in range(RUNS):
            d = cfg["stream"](WARM_TPS_PROMPT, max_tokens=200)
            warm_tps.append({"run": i + 1, **d})
            log(
                f"  warm_tps run {i + 1}: {d['tokens_per_sec']:.1f} tok/s "
                f"({d['completion_tokens']} tokens)"
            )
    finally:
        stop_server(proc, sampler)

    return {
        "label": cfg["label"],
        "peak_rss_bytes": sampler.peak_bytes,
        "rss_samples": sampler.samples,
        "ttft_first": ttft_first,
        "ttft_repeat": ttft_repeat,
        "warm_tps": warm_tps,
    }


# ── Summaries ─────────────────────────────────────────────────────────────────


def stats_of(values: list[float | None]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None, "stddev": None}
    p95 = vals[int(len(vals) * 0.95) - 1] if len(vals) > 1 else vals[0]
    return {
        "n": len(vals),
        "median": stats.median(vals),
        "p95": p95,
        "min": min(vals),
        "max": max(vals),
        "stddev": stats.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def summarize(cfg_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "ttft_first_s": stats_of([r["ttft_s"] for r in cfg_data["ttft_first"]]),
        "ttft_repeat_s": stats_of([r["ttft_s"] for r in cfg_data["ttft_repeat"]]),
        "warm_tps": stats_of([r["tokens_per_sec"] for r in cfg_data["warm_tps"]]),
        "peak_rss_bytes": cfg_data["peak_rss_bytes"],
    }


# ── Disk size helpers ─────────────────────────────────────────────────────────


def disk_size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            fp = Path(root) / f
            try:
                total += fp.stat().st_size
            except OSError:
                pass
    return total


def ollama_model_disk_size(model: str) -> int:
    manifest = Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai" / "library"
    name, _, tag = model.partition(":")
    mpath = manifest / name / (tag or "latest")
    if not mpath.exists():
        return 0
    data = json.loads(mpath.read_text())
    blobs_dir = Path.home() / ".ollama" / "models" / "blobs"
    total = 0
    for layer in data.get("layers", []):
        digest = layer.get("digest", "").replace(":", "-")
        bp = blobs_dir / digest
        total += bp.stat().st_size if bp.exists() else int(layer.get("size", 0) or 0)
    return total


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    log(f"Output dir: {OUT_DIR}")
    ollama_disk = ollama_model_disk_size(OLLAMA_MODEL)
    squish_disk = disk_size_bytes(Path(SQUISH_MODEL_PATH))

    results: dict[str, Any] = {
        "timestamp": TS,
        "host": "Apple M3 MacBook Pro 16 GB",
        "ollama_version": subprocess.run(
            [OLLAMA_BIN, "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "squish_version": subprocess.run(
            [SQUISH_BIN, "--version"], capture_output=True, text=True
        ).stdout.strip(),
        "models": {
            "ollama": {"name": OLLAMA_MODEL, "disk_bytes": ollama_disk},
            "squish": {"path": SQUISH_MODEL_PATH, "disk_bytes": squish_disk},
        },
        "warmup_requests": WARMUP_REQUESTS,
        "runs_per_phase": RUNS,
        "prompts": {
            "ttft_first_template": UNIQUE_PROMPT_PREFIX
            + "Write a single short sentence about its weather.",
            "ttft_repeat": REPEAT_PROMPT,
            "warm_tps": WARM_TPS_PROMPT,
        },
        "configs": {},
        "summary": {},
        "notes": [
            "Squish daemon mode uses `python -m squish.server` directly. The new",
            "`squishd` UDS daemon does not load mlx-native quantized models — see PRECHECK.md.",
            "Phase 3 (spec decode) is not measured: the server crashes on --draft-model.",
            "ttft_first uses a different prompt each run; ttft_repeat uses the same",
            "prompt every run after a single cache-priming send.",
        ],
    }

    for cfg_id in CONFIGS:
        cfg_data = run_config(cfg_id)
        results["configs"][cfg_id] = cfg_data
        results["summary"][cfg_id] = summarize(cfg_data)

    out_json = OUT_DIR / "raw.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    log(f"Wrote {out_json}")
    print_summary(results)


def fmt_s(v: float | None) -> str:
    if v is None:
        return "-"
    if v < 1:
        return f"{v * 1000:.0f} ms"
    return f"{v:.2f} s"


def fmt_tps(v: float | None) -> str:
    return f"{v:.1f} tok/s" if v else "-"


def fmt_bytes(n: float | None) -> str:
    if not n:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def winner_or_tie(
    pairs: list[tuple[str, float | None]], higher_better: bool, tie: float = 0.05
) -> str:
    valid = [(L, V) for L, V in pairs if V is not None and V > 0]
    if not valid:
        return "-"
    best_label, best_value = (max if higher_better else min)(valid, key=lambda p: p[1])
    for L, V in valid:
        if L == best_label:
            continue
        ratio = V / best_value if higher_better else best_value / V
        if ratio < (1.0 - tie):
            return best_label
    return "tie"


def print_summary(r: dict[str, Any]) -> None:
    s = r["summary"]
    print()
    print("# v4.1 measured benchmark — Squish vs Ollama (M3 16 GB)")
    print(f"Ollama: {r['ollama_version']}    Squish: {r['squish_version']}")
    print(f"Squish target: {Path(r['models']['squish']['path']).name} (INT4 MLX)")
    print(f"Ollama target: {r['models']['ollama']['name']} (Q4_K_M GGUF)")
    print()
    cfg_order = ["ollama", "squish_daemon", "squish_kv", "squish_pkv", "squish_spec"]
    short_labels = {
        "ollama":         "Ollama",
        "squish_daemon":  "sq daemon",
        "squish_kv":      "sq +disk-KV",
        "squish_pkv":     "sq +pkv (v4.1)",
        "squish_spec":    "sq +spec (v4.1)",
    }
    header = " | ".join([f"{'Metric':<38}"] + [f"{short_labels[c]:>15}" for c in cfg_order] + [f"{'Winner':>14}"])
    print(header)
    print("-" * len(header))

    def _row(name: str, key: str, higher_better: bool, fmt_fn) -> None:
        vals = [s[c][key]["median"] if isinstance(s[c][key], dict) else s[c][key] for c in cfg_order]
        winner = winner_or_tie(list(zip(cfg_order, vals)), higher_better=higher_better)
        cells = [fmt_fn(v) for v in vals]
        print(" | ".join([f"{name:<38}"] + [f"{c:>15}" for c in cells] + [f"{winner:>14}"]))

    def _row_const(name: str, vals: list[float], higher_better: bool, fmt_fn) -> None:
        winner = winner_or_tie(list(zip(cfg_order, vals)), higher_better=higher_better)
        cells = [fmt_fn(v) for v in vals]
        print(" | ".join([f"{name:<38}"] + [f"{c:>15}" for c in cells] + [f"{winner:>14}"]))

    _row("TTFT (fresh prompt)",          "ttft_first_s",  False, fmt_s)
    _row("TTFT (repeated prompt)",       "ttft_repeat_s", False, fmt_s)
    _row("Warm tok/s (200-tok decode)",  "warm_tps",      True,  fmt_tps)
    peak = [s[c]["peak_rss_bytes"] for c in cfg_order]
    _row_const("Peak RAM (process tree)", peak, False, fmt_bytes)
    disk = [r["models"]["ollama"]["disk_bytes"]] + [r["models"]["squish"]["disk_bytes"]] * 4
    _row_const("Disk size (model)", disk, False, fmt_bytes)

    print()
    for phase_label, phase_key in [
        ("Per-run TTFT (fresh prompt) — 5 runs",   "ttft_first"),
        ("Per-run TTFT (repeated prompt) — 5 runs", "ttft_repeat"),
    ]:
        print(phase_label + ":")
        for c in cfg_order:
            ttfts = [run["ttft_s"] for run in r["configs"][c][phase_key]]
            ttft_str = " ".join(f"{t * 1000:.0f}ms" if t else "    -" for t in ttfts)
            print(f"  {short_labels[c]:<16}  {ttft_str}")
        print()

    print("Per-run warm tok/s — 5 runs:")
    for c in cfg_order:
        tps = [run["tokens_per_sec"] for run in r["configs"][c]["warm_tps"]]
        tps_str = " ".join(f"{t:.1f}" if t else "  -" for t in tps)
        print(f"  {short_labels[c]:<16}  {tps_str}")


if __name__ == "__main__":
    sys.exit(main() or 0)
