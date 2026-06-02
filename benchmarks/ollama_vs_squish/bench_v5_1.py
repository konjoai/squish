#!/usr/bin/env python3
"""v5.1 unified benchmark harness.

Adds four metric dimensions on top of v5's harness:

1. **End-to-end response time** (``e2e_200tok_s``): full wall time for a
   fixed 200-token completion.  This is what users perceive.

2. **Inter-token latency** (``itl_p50_ms`` / ``itl_p95_ms``): per-token
   gap times during steady-state generation, excluding the first token
   (which is TTFT).  Reveals thermal drift and decode-loop variance.

3. **Multiple context lengths**: 75-token (existing back-compat), 500,
   2000, 4000 tokens.  Cold TTFT + warm tok/s + e2e_200 at each size.

4. **INT3 column**: a third squish config using the on-disk
   ``Qwen2.5-7B-Instruct-int3`` model (when present).  Same code path
   as INT4; the only change is the model dir.

The harness preserves the v5 ``bench_v5_longctx.py`` shared-prefix
scenario for the 4000-token case (the agent workload) — that's still the
clearest block-cache win row.

Output: ``results/benchmarks_v5_1/runs/<UTC-timestamp>/raw.json``.
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

# ── Hardware / paths ─────────────────────────────────────────────────────────

OLLAMA_BIN   = "/usr/local/bin/ollama"
OLLAMA_HOST  = "127.0.0.1"
OLLAMA_PORT  = 11434
OLLAMA_MODEL = "qwen2.5:7b"

SQUISH_BIN  = "/Users/wscholl/squish/.venv/bin/squish"
SQUISH_PY   = "/Users/wscholl/squish/.venv/bin/python"
SQUISH_HOST = "127.0.0.1"
SQUISH_PORT = 11435
SQUISH_API_KEY    = "squish"

SQUISH_MODEL_INT4 = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"
SQUISH_MODEL_INT3 = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int3"

PKV_CACHE_DIR     = "/tmp/squish_pkv_v5_1"
BLOCK_CACHE_DIR   = "/tmp/squish_blocks_v5_1"
BLOCK_CACHE_DIR_INT3 = "/tmp/squish_blocks_v5_1_int3"

REPO_ROOT  = Path(__file__).resolve().parents[2]
OUT_ROOT   = REPO_ROOT / "results" / "benchmarks_v5_1" / "runs"
TS         = time.strftime("%Y%m%dT%H%M%S")
OUT_DIR    = OUT_ROOT / TS
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR    = Path("/tmp/bench_v51_logs")
LOG_DIR.mkdir(exist_ok=True)

# ── Prompts at four context lengths ──────────────────────────────────────────

# A 75-token-ish prompt — matches the v4.2 benchmark.
_P75 = (
    "You are a senior engineer reviewing a pull request. "
    "The PR adds a new Redis-backed session cache to the auth service. "
    "The diff is roughly 240 lines and touches login, logout, and a small "
    "middleware shim. Give a two-sentence review covering the most "
    "important risk."
)

# Building blocks for longer prompts — repeat with mild variation.
_REPEAT_CHUNK = (
    "The codebase has the following conventions: every public function "
    "must have a docstring; every test must be deterministic; every database "
    "migration must be reversible. The team's CI runs the full test suite, "
    "a linter, a type-checker, and a mutation-testing pass. PRs are auto-merged "
    "when they pass review-mode and have two approvals. Pay particular attention "
    "to: thread safety, TTL refresh under contention, fallback latency budget "
    "when Redis is slow, and the impact on session token rotation if a key is "
    "evicted mid-request. "
)

def _build_prompt_to_tokens(seed: str, target_tokens: int) -> str:
    """Append repeat-chunks of context to *seed* until the result tokenizes
    to ~target_tokens.  Uses the Qwen2.5 tokenizer."""
    from mlx_lm import load
    _, tok = load(SQUISH_MODEL_INT4)
    base = seed
    while True:
        ids = tok.encode(base)
        if len(ids) >= target_tokens:
            return base
        base += _REPEAT_CHUNK


def _build_variation_suffix(seed_prompt: str, idx: int) -> str:
    """Append a unique short tail to *seed_prompt* — different per idx."""
    tails = [
        " Summarize the most important risk in two sentences.",
        " List the top three test gaps you'd expect to see filled.",
        " Identify one place where thread-safety could be subtly wrong.",
        " Propose one alternative naming for the cache module.",
        " Write a one-line approval comment if the PR meets bar.",
        " Identify one race condition that could occur under load.",
        " Suggest one production observability metric to add.",
        " Describe one regression test that's missing.",
    ]
    return seed_prompt + tails[idx % len(tails)]


# Will be populated by main() once we can load the tokenizer.
PROMPTS: dict[str, str] = {}

WARM_TPS_PROMPT = "Summarize the Renaissance in two paragraphs of 100 words each."

# ── Runs / phases ────────────────────────────────────────────────────────────

RUNS = 5
E2E_MAX_TOKENS = 200


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Process helpers ───────────────────────────────────────────────────────────

def kill_all_serving() -> None:
    patterns = [
        "ollama serve", "ollama runner", "ollama_llama_server",
        "Ollama.app", "Ollama Helper",
        "squish.server", "squishd",
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


# ── Streaming clients (capture per-token timestamps) ──────────────────────────

def stream_ollama(prompt: str, max_tokens: int = 1) -> dict[str, Any]:
    body = json.dumps({
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0.0},
    }).encode()
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        data=body, headers={"Content-Type": "application/json"},
    )
    t_req = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    chunk_timestamps: list[float] = []
    eval_count = 0
    eval_duration_ns = 0
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = d.get("response", "")
            if chunk:
                t = time.perf_counter()
                chunk_timestamps.append(t)
                if t_first is None:
                    t_first = t
                parts.append(chunk)
            if d.get("done"):
                eval_count = d.get("eval_count") or 0
                eval_duration_ns = d.get("eval_duration") or 0
    t_done = time.perf_counter()
    tps = eval_count / (eval_duration_ns / 1e9) if (eval_count and eval_duration_ns) else None
    return {
        "ttft_s":             (t_first - t_req) if t_first else None,
        "total_s":            t_done - t_req,
        "completion_tokens":  eval_count or len(parts),
        "tokens_per_sec":     tps,
        "chunk_timestamps":   chunk_timestamps,
        "t_req":              t_req,
    }


def stream_squish(prompt: str, max_tokens: int = 1) -> dict[str, Any]:
    body = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "max_tokens": max_tokens, "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/chat/completions",
        data=body, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {SQUISH_API_KEY}",
        },
    )
    t_req = time.perf_counter()
    t_first: float | None = None
    parts: list[str] = []
    chunk_timestamps: list[float] = []
    completion_tokens: int | None = None
    with urllib.request.urlopen(req, timeout=600) as resp:
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
                if chunk:
                    t = time.perf_counter()
                    chunk_timestamps.append(t)
                    if t_first is None:
                        t_first = t
                    parts.append(chunk)
            usage = d.get("usage")
            if usage:
                completion_tokens = usage.get("completion_tokens")
    t_done = time.perf_counter()
    n_tokens = completion_tokens or len(parts)
    gen_window = (t_done - t_first) if t_first else None
    tps = (n_tokens / gen_window) if (gen_window and gen_window > 0 and n_tokens) else None
    return {
        "ttft_s":             (t_first - t_req) if t_first else None,
        "total_s":            t_done - t_req,
        "completion_tokens":  n_tokens,
        "tokens_per_sec":     tps,
        "chunk_timestamps":   chunk_timestamps,
        "t_req":              t_req,
    }


def _inter_token_stats(d: dict[str, Any]) -> dict[str, float | None]:
    """Compute inter-token latency percentiles from a stream result.

    Drops the first token's gap (that's the TTFT, measured separately).
    Returns p50_ms, p95_ms, count.
    """
    ts = d.get("chunk_timestamps") or []
    if len(ts) < 3:  # need >= 2 decode-only intervals
        return {"itl_p50_ms": None, "itl_p95_ms": None, "itl_count": 0}
    gaps = [(ts[i + 1] - ts[i]) * 1000.0 for i in range(1, len(ts) - 1)]
    if not gaps:
        return {"itl_p50_ms": None, "itl_p95_ms": None, "itl_count": 0}
    gaps_sorted = sorted(gaps)
    p50 = gaps_sorted[len(gaps_sorted) // 2]
    p95_idx = int(len(gaps_sorted) * 0.95)
    p95 = gaps_sorted[min(p95_idx, len(gaps_sorted) - 1)]
    return {"itl_p50_ms": p50, "itl_p95_ms": p95, "itl_count": len(gaps)}


# ── Server launchers ─────────────────────────────────────────────────────────

def start_ollama(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    env = {**os.environ, "OLLAMA_HOST": f"{OLLAMA_HOST}:{OLLAMA_PORT}"}
    proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env=env,
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def _squish_cmd(model_path: str, extra: list[str]) -> list[str]:
    return [
        SQUISH_PY, "-m", "squish.server",
        "--mlx-model-dir", model_path,
        "--port", str(SQUISH_PORT), "--host", SQUISH_HOST,
        "--log-level", "warning", *extra,
    ]


def start_squish_daemon(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    proc = subprocess.Popen(
        _squish_cmd(SQUISH_MODEL_INT4, []),
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def start_squish_pkv(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    if os.path.isdir(PKV_CACHE_DIR):
        shutil.rmtree(PKV_CACHE_DIR)
    os.makedirs(PKV_CACHE_DIR, exist_ok=True)
    proc = subprocess.Popen(
        _squish_cmd(SQUISH_MODEL_INT4, ["--prompt-kv-cache", PKV_CACHE_DIR]),
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def start_squish_block(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    if os.path.isdir(BLOCK_CACHE_DIR):
        shutil.rmtree(BLOCK_CACHE_DIR)
    os.makedirs(BLOCK_CACHE_DIR, exist_ok=True)
    proc = subprocess.Popen(
        _squish_cmd(SQUISH_MODEL_INT4, [
            "--block-kv-cache", BLOCK_CACHE_DIR,
            "--block-kv-size", "64",
        ]),
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
        env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def start_squish_block_int3(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    if os.path.isdir(BLOCK_CACHE_DIR_INT3):
        shutil.rmtree(BLOCK_CACHE_DIR_INT3)
    os.makedirs(BLOCK_CACHE_DIR_INT3, exist_ok=True)
    proc = subprocess.Popen(
        _squish_cmd(SQUISH_MODEL_INT3, [
            "--block-kv-cache", BLOCK_CACHE_DIR_INT3,
            "--block-kv-size", "64",
        ]),
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
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


# ── Per-config registry ──────────────────────────────────────────────────────

CONFIGS: dict[str, dict[str, Any]] = {
    "ollama": {
        "label":     "Ollama (warm)",
        "ready_url": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version",
        "start":     start_ollama,
        "stream":    stream_ollama,
        "quant":     "Q4_K_M",
    },
    "squish_daemon": {
        "label":     "Squish daemon INT4",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_daemon,
        "stream":    stream_squish,
        "quant":     "INT4",
    },
    "squish_pkv": {
        "label":     "Squish +pkv INT4",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_pkv,
        "stream":    stream_squish,
        "quant":     "INT4",
    },
    "squish_block": {
        "label":     "Squish +block INT4",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_block,
        "stream":    stream_squish,
        "quant":     "INT4",
    },
    "squish_block_int3": {
        "label":     "Squish +block INT3",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_block_int3,
        "stream":    stream_squish,
        "quant":     "INT3",
    },
}


def run_one_request(cfg_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    cfg = CONFIGS[cfg_id]
    return cfg["stream"](prompt, max_tokens=max_tokens)


def run_one_phase(cfg_id: str, prompt: str, label: str) -> dict[str, Any]:
    """For a single prompt, run RUNS measurements at max_tokens=1 (TTFT) and
    RUNS at max_tokens=200 (e2e+itl).  Returns combined stats."""
    ttft_runs: list[dict[str, Any]] = []
    e2e_runs:  list[dict[str, Any]] = []
    for i in range(RUNS):
        d = run_one_request(cfg_id, prompt, max_tokens=1)
        ttft_runs.append({"run": i + 1, **d})
    for i in range(RUNS):
        d = run_one_request(cfg_id, prompt, max_tokens=E2E_MAX_TOKENS)
        itl = _inter_token_stats(d)
        e2e_runs.append({"run": i + 1, **d, **itl})
    return {
        "label":      label,
        "ttft_runs":  ttft_runs,
        "e2e_runs":   e2e_runs,
    }


def run_config(cfg_id: str, prompts: dict[str, str]) -> dict[str, Any]:
    cfg = CONFIGS[cfg_id]
    log(f"=== {cfg_id} : start server ===")
    kill_all_serving()
    log_path = LOG_DIR / f"v51_{cfg_id}_{TS}.log"
    proc, sampler = cfg["start"](log_path)
    try:
        if not wait_ready(cfg["ready_url"], timeout=300):
            raise RuntimeError(f"{cfg_id} did not become ready")
        log("  ready; priming with short request")
        cfg["stream"]("Hello.", max_tokens=4)

        phases: dict[str, dict[str, Any]] = {}
        for pname, pstr in prompts.items():
            log(f"  ── phase {pname} ({len(pstr)} chars) ──")
            phases[pname] = run_one_phase(cfg_id, pstr, pname)
            ts = phases[pname]["ttft_runs"]
            es = phases[pname]["e2e_runs"]
            ttft_med = stats.median([x["ttft_s"] for x in ts if x["ttft_s"]]) if ts else None
            e2e_med  = stats.median([x["total_s"] for x in es if x["total_s"]]) if es else None
            tps_med  = stats.median([x["tokens_per_sec"] for x in es if x["tokens_per_sec"]]) if es else None
            ttft_str = f"{ttft_med * 1000:.0f}ms" if ttft_med else "-"
            e2e_str  = f"{e2e_med:.2f}s"          if e2e_med  else "-"
            tps_str  = f"{tps_med:.1f}t/s"        if tps_med  else "-"
            log(f"    -> ttft={ttft_str}  e2e_200={e2e_str}  tps={tps_str}")
    finally:
        stop_server(proc, sampler)

    return {
        "label":          cfg["label"],
        "quant":          cfg.get("quant", "?"),
        "peak_rss_bytes": sampler.peak_bytes,
        "rss_samples":    sampler.samples,
        "phases":         phases,
    }


# ── Summary ──────────────────────────────────────────────────────────────────

def stats_of(values: "list[float | None]") -> dict[str, "float | None"]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "p95": None, "min": None, "max": None, "stddev": None}
    p95 = vals[int(len(vals) * 0.95) - 1] if len(vals) > 1 else vals[0]
    return {
        "n":       len(vals),
        "median":  stats.median(vals),
        "p95":     p95,
        "min":     min(vals),
        "max":     max(vals),
        "stddev":  stats.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def summarize_phase(phase_data: dict[str, Any]) -> dict[str, Any]:
    ttft_vals = [r["ttft_s"] for r in phase_data["ttft_runs"]]
    e2e_vals  = [r["total_s"] for r in phase_data["e2e_runs"]]
    tps_vals  = [r["tokens_per_sec"] for r in phase_data["e2e_runs"]]
    itl50_vals = [r.get("itl_p50_ms") for r in phase_data["e2e_runs"]]
    itl95_vals = [r.get("itl_p95_ms") for r in phase_data["e2e_runs"]]
    return {
        "ttft_s":           stats_of(ttft_vals),
        "e2e_200tok_s":     stats_of(e2e_vals),
        "warm_tps":         stats_of(tps_vals),
        "itl_p50_ms":       stats_of(itl50_vals),
        "itl_p95_ms":       stats_of(itl95_vals),
    }


def summarize(cfg_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "peak_rss_bytes": cfg_data["peak_rss_bytes"],
        "quant":          cfg_data["quant"],
        "phases":         {
            pname: summarize_phase(pdata)
            for pname, pdata in cfg_data["phases"].items()
        },
    }


# ── Disk size helpers ────────────────────────────────────────────────────────

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


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    log(f"Output dir: {OUT_DIR}")
    # Build prompts at 4 context lengths once.
    log("Constructing prompts at 75, 500, 2000, 4000 tokens …")
    p75   = _P75
    p500  = _build_prompt_to_tokens(_P75, target_tokens=500)
    p2000 = _build_prompt_to_tokens(_P75, target_tokens=2000)
    p4000 = _build_prompt_to_tokens(_P75, target_tokens=4000)
    from mlx_lm import load
    _, tok = load(SQUISH_MODEL_INT4)
    n75   = len(tok.encode(p75))
    n500  = len(tok.encode(p500))
    n2000 = len(tok.encode(p2000))
    n4000 = len(tok.encode(p4000))
    log(f"  built: 75≈{n75}t, 500≈{n500}t, 2000≈{n2000}t, 4000≈{n4000}t")
    PROMPTS.update({
        "p75":   p75,
        "p500":  p500,
        "p2000": p2000,
        "p4000": p4000,
    })

    # Skip INT3 if the model isn't present.
    if not os.path.isdir(SQUISH_MODEL_INT3):
        del CONFIGS["squish_block_int3"]
        log("INT3 column SKIPPED — model not on disk.")
    else:
        log(f"INT3 column ENABLED — {SQUISH_MODEL_INT3}")

    ollama_disk = ollama_model_disk_size(OLLAMA_MODEL)
    int4_disk   = disk_size_bytes(Path(SQUISH_MODEL_INT4))
    int3_disk   = disk_size_bytes(Path(SQUISH_MODEL_INT3)) if os.path.isdir(SQUISH_MODEL_INT3) else 0

    results: dict[str, Any] = {
        "timestamp":      TS,
        "host":           "Apple M3 MacBook Pro 16 GB",
        "ollama_version": subprocess.run(
            [OLLAMA_BIN, "--version"], capture_output=True, text=True,
        ).stdout.strip(),
        "squish_version": subprocess.run(
            [SQUISH_BIN, "--version"], capture_output=True, text=True,
        ).stdout.strip(),
        "models": {
            "ollama":      {"name": OLLAMA_MODEL,      "disk_bytes": ollama_disk},
            "squish_int4": {"path": SQUISH_MODEL_INT4, "disk_bytes": int4_disk},
            "squish_int3": {"path": SQUISH_MODEL_INT3, "disk_bytes": int3_disk},
        },
        "prompt_token_counts": {
            "p75": n75, "p500": n500, "p2000": n2000, "p4000": n4000,
        },
        "runs_per_metric": RUNS,
        "e2e_max_tokens":  E2E_MAX_TOKENS,
        "configs":         {},
        "summary":         {},
        "notes": [
            "Each phase measures the same prompt at the named context length.",
            "TTFT runs use max_tokens=1 (5 runs). E2E runs use max_tokens=200 (5 runs).",
            "Inter-token latency excludes the first chunk (that's TTFT).",
            "INT3 column appears only if Qwen2.5-7B-Instruct-int3 is on disk.",
        ],
    }

    for cfg_id in list(CONFIGS.keys()):
        cfg_data = run_config(cfg_id, PROMPTS)
        results["configs"][cfg_id] = cfg_data
        results["summary"][cfg_id] = summarize(cfg_data)

    out_json = OUT_DIR / "raw.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    log(f"Wrote {out_json}")
    print_summary(results)


def fmt_s(v: "float | None") -> str:
    if v is None:
        return "-"
    if v < 1:
        return f"{v * 1000:.0f} ms"
    return f"{v:.2f} s"


def fmt_tps(v: "float | None") -> str:
    return f"{v:.1f} tok/s" if v else "-"


def fmt_ms(v: "float | None") -> str:
    if v is None:
        return "-"
    return f"{v:.1f} ms"


def fmt_bytes(n: "float | None") -> str:
    if not n:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def print_summary(r: dict[str, Any]) -> None:
    s = r["summary"]
    print()
    print("# v5.1 unified benchmark — Squish vs Ollama (M3 16 GB)")
    print(f"Ollama: {r['ollama_version']}    Squish: {r['squish_version']}")
    print(f"Prompt sizes: 75≈{r['prompt_token_counts']['p75']}t,"
          f" 500≈{r['prompt_token_counts']['p500']}t,"
          f" 2000≈{r['prompt_token_counts']['p2000']}t,"
          f" 4000≈{r['prompt_token_counts']['p4000']}t")
    print()
    cfg_order = list(r["configs"].keys())
    short_labels = {
        "ollama":             "Ollama",
        "squish_daemon":      "sq daemon I4",
        "squish_pkv":         "sq +pkv I4",
        "squish_block":       "sq +block I4",
        "squish_block_int3":  "sq +block I3",
    }

    # Per-phase summary tables
    phase_order = ["p75", "p500", "p2000", "p4000"]
    for phase in phase_order:
        print(f"── prompt size {phase} ({r['prompt_token_counts'][phase]} tokens) ──")
        header = " | ".join([f"{'Metric':<32}"] + [f"{short_labels[c]:>14}" for c in cfg_order])
        print(header)
        print("-" * len(header))
        rows = [
            ("TTFT (cold)",      "ttft_s",       fmt_s),
            ("E2E 200-tok",      "e2e_200tok_s", fmt_s),
            ("Warm tok/s",       "warm_tps",     fmt_tps),
            ("Inter-token p50",  "itl_p50_ms",   fmt_ms),
            ("Inter-token p95",  "itl_p95_ms",   fmt_ms),
        ]
        for name, key, fmt in rows:
            cells = []
            for c in cfg_order:
                ph = s[c]["phases"].get(phase, {})
                v = ph.get(key, {}).get("median") if isinstance(ph.get(key), dict) else None
                cells.append(fmt(v))
            print(" | ".join([f"{name:<32}"] + [f"{v:>14}" for v in cells]))
        print()

    # Peak RSS + disk size — single line each
    print("── per-config peak RSS / disk size ──")
    print(" | ".join([f"{'Metric':<32}"] + [f"{short_labels[c]:>14}" for c in cfg_order]))
    print("-" * 90)
    print(" | ".join([f"{'peak RSS':<32}"]
                    + [f"{fmt_bytes(s[c]['peak_rss_bytes']):>14}" for c in cfg_order]))
    disk_map = {
        "ollama":            r["models"]["ollama"]["disk_bytes"],
        "squish_daemon":     r["models"]["squish_int4"]["disk_bytes"],
        "squish_pkv":        r["models"]["squish_int4"]["disk_bytes"],
        "squish_block":      r["models"]["squish_int4"]["disk_bytes"],
        "squish_block_int3": r["models"]["squish_int3"]["disk_bytes"],
    }
    print(" | ".join([f"{'disk (model only)':<32}"]
                    + [f"{fmt_bytes(disk_map.get(c)):>14}" for c in cfg_order]))


if __name__ == "__main__":
    sys.exit(main() or 0)
