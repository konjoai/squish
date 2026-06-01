#!/usr/bin/env python3
"""Side-by-side benchmark: Ollama vs Squish on a single prompt.

Cold phase: 5 runs each — kill server, restart, single inference, kill.
Warm phase: 3 runs each — server already loaded, repeat inference.

Peak RSS is sampled across the full process tree (server + spawned runner)
because Ollama loads the model in a child `ollama runner` process. /usr/bin/time -l
on the parent alone would understate RSS for Ollama; we use psutil sampling
for a like-for-like comparison.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median
from typing import Any

import psutil

PROMPT = "What is the capital of France? Answer in one sentence."
COLD_RUNS = 5
WARM_RUNS = 3

OLLAMA_BIN = "/usr/local/bin/ollama"
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_MODEL = "qwen2.5:7b"

SQUISH_BIN = "/Users/wscholl/squish/.venv/bin/squish"
SQUISH_HOST = "127.0.0.1"
SQUISH_PORT = 11435
SQUISH_API_KEY = "squish"
SQUISH_MODEL_PATH = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"

RESULTS_DIR = Path("/Users/wscholl/squish/results")
TS = time.strftime("%Y%m%d_%H%M%S")
OUT_JSON = RESULTS_DIR / f"ollama_vs_squish_M3_{TS}.json"
LOG_DIR = Path("/tmp/bench_logs")
LOG_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def kill_all_serving() -> None:
    """Kill all Ollama and Squish processes (and their model runners)."""
    patterns = [
        "ollama serve",
        "ollama runner",
        "ollama_llama_server",
        "Ollama.app",
        "Ollama Helper",
        "squish run",
        "squish.cli run",
        "squish.server",
    ]
    for p in patterns:
        subprocess.run(["pkill", "-f", p], capture_output=True)
    time.sleep(3)


def wait_ready(url: str, timeout: float = 180) -> bool:
    """Poll URL until any HTTP response (even 4xx) indicates the server is listening.

    A 401 from squish's auth-gated endpoints still means the process is up and serving,
    so we treat any HTTPError as success too.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                r.read()
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(0.25)
    return False


class RSSSampler(threading.Thread):
    """Sample peak RSS of a process tree until stop()."""

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
            tree = 0
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


def stream_ollama(prompt: str) -> dict[str, Any]:
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}).encode()
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    parts: list[str] = []
    eval_count = None
    eval_duration_ns = None
    load_duration_ns = None
    prompt_eval_count = None
    prompt_eval_duration_ns = None
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = d.get("response", "")
            if chunk and ttft is None:
                ttft = time.perf_counter() - t0
            if chunk:
                chunks += 1
                parts.append(chunk)
            if d.get("done"):
                eval_count = d.get("eval_count")
                eval_duration_ns = d.get("eval_duration")
                load_duration_ns = d.get("load_duration")
                prompt_eval_count = d.get("prompt_eval_count")
                prompt_eval_duration_ns = d.get("prompt_eval_duration")
    total = time.perf_counter() - t0
    tps = None
    if eval_count and eval_duration_ns:
        tps = eval_count / (eval_duration_ns / 1e9)
    return {
        "ttft_s": ttft,
        "total_s": total,
        "stream_chunks": chunks,
        "completion_tokens": eval_count,
        "completion_duration_s": (eval_duration_ns / 1e9) if eval_duration_ns else None,
        "load_duration_s": (load_duration_ns / 1e9) if load_duration_ns else None,
        "prompt_tokens": prompt_eval_count,
        "prompt_eval_duration_s": (prompt_eval_duration_ns / 1e9) if prompt_eval_duration_ns else None,
        "tokens_per_sec": tps,
        "response": "".join(parts),
    }


def stream_squish(prompt: str) -> dict[str, Any]:
    body = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 100,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(
        f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {SQUISH_API_KEY}"},
    )
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    parts: list[str] = []
    prompt_tokens = None
    completion_tokens = None
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
                if chunk and ttft is None:
                    ttft = time.perf_counter() - t0
                if chunk:
                    chunks += 1
                    parts.append(chunk)
            usage = d.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
    total = time.perf_counter() - t0
    gen_window = (total - ttft) if (ttft is not None and total > ttft) else None
    tokens_for_rate = completion_tokens or chunks
    tps = (tokens_for_rate / gen_window) if (gen_window and gen_window > 0 and tokens_for_rate) else None
    return {
        "ttft_s": ttft,
        "total_s": total,
        "stream_chunks": chunks,
        "completion_tokens": completion_tokens or chunks,
        "prompt_tokens": prompt_tokens,
        "tokens_per_sec": tps,
        "response": "".join(parts),
    }


def start_ollama_server(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
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


def start_squish_server(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    proc = subprocess.Popen(
        [
            SQUISH_BIN, "run", SQUISH_MODEL_PATH,
            "--port", str(SQUISH_PORT),
            "--host", SQUISH_HOST,
            "--api-key", SQUISH_API_KEY,
            "--no-browser",
            "--log-level", "warning",
        ],
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
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


def cold_run(tool: str, idx: int) -> dict[str, Any]:
    log(f"[{tool}] COLD run {idx + 1}/{COLD_RUNS} — killing prior servers")
    kill_all_serving()
    log_path = LOG_DIR / f"{tool}_cold_{idx}.log"
    t_proc_start = time.perf_counter()
    if tool == "ollama":
        proc, sampler = start_ollama_server(log_path)
        ready_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version"
    else:
        proc, sampler = start_squish_server(log_path)
        ready_url = f"http://{SQUISH_HOST}:{SQUISH_PORT}/health"

    try:
        if not wait_ready(ready_url, timeout=180):
            raise RuntimeError(f"{tool} did not become ready")
        t_ready = time.perf_counter()
        log(f"[{tool}] server ready in {t_ready - t_proc_start:.2f}s — sending prompt")
        infer = stream_ollama(PROMPT) if tool == "ollama" else stream_squish(PROMPT)
        t_done = time.perf_counter()
    finally:
        stop_server(proc, sampler)

    return {
        "phase": "cold",
        "run": idx + 1,
        "server_ready_s": t_ready - t_proc_start,
        "wall_total_s": t_done - t_proc_start,
        "peak_rss_bytes": sampler.peak_bytes,
        "rss_samples": sampler.samples,
        "inference": infer,
    }


def warm_run(tool: str, idx: int) -> dict[str, Any]:
    log(f"[{tool}] WARM run {idx + 1}/{WARM_RUNS}")
    infer = stream_ollama(PROMPT) if tool == "ollama" else stream_squish(PROMPT)
    return {
        "phase": "warm",
        "run": idx + 1,
        "inference": infer,
    }


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
    """Sum the layer blob sizes referenced by the model manifest."""
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
        if bp.exists():
            total += bp.stat().st_size
        else:
            total += int(layer.get("size", 0) or 0)
    return total


def run_phase(tool: str) -> dict[str, Any]:
    log(f"=== {tool.upper()} cold phase: {COLD_RUNS} runs ===")
    cold_runs = [cold_run(tool, i) for i in range(COLD_RUNS)]

    log(f"=== {tool.upper()} warm phase: starting fresh server, then {WARM_RUNS} runs ===")
    kill_all_serving()
    warm_log = LOG_DIR / f"{tool}_warm_server.log"
    if tool == "ollama":
        proc, sampler = start_ollama_server(warm_log)
        ready_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version"
    else:
        proc, sampler = start_squish_server(warm_log)
        ready_url = f"http://{SQUISH_HOST}:{SQUISH_PORT}/health"

    warm_runs: list[dict[str, Any]] = []
    try:
        if not wait_ready(ready_url, timeout=180):
            raise RuntimeError(f"{tool} warm-phase server did not start")
        # Prime: load model into memory with one throwaway call
        log(f"[{tool}] priming model (throwaway request, not counted)")
        if tool == "ollama":
            stream_ollama(PROMPT)
        else:
            stream_squish(PROMPT)
        for i in range(WARM_RUNS):
            warm_runs.append(warm_run(tool, i))
    finally:
        stop_server(proc, sampler)

    return {"cold_runs": cold_runs, "warm_runs": warm_runs}


def summarize(runs: list[dict[str, Any]], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        vals: list[float] = []
        for r in runs:
            v = r.get("inference", {}).get(k) if "inference" in r else r.get(k)
            if v is not None:
                vals.append(float(v))
        out[k] = median(vals) if vals else None
    return out


def main() -> None:
    log(f"Output will be written to: {OUT_JSON}")
    RESULTS_DIR.mkdir(exist_ok=True)

    ollama_disk = ollama_model_disk_size(OLLAMA_MODEL)
    squish_disk = disk_size_bytes(Path(SQUISH_MODEL_PATH))

    results: dict[str, Any] = {
        "timestamp": TS,
        "host": "Apple M3 MacBook Pro 16GB",
        "prompt": PROMPT,
        "ollama": {
            "model": OLLAMA_MODEL,
            "model_disk_bytes": ollama_disk,
            "version": subprocess.run([OLLAMA_BIN, "--version"], capture_output=True, text=True).stdout.strip(),
        },
        "squish": {
            "model": "Qwen2.5-7B-Instruct-int4",
            "model_path": SQUISH_MODEL_PATH,
            "model_disk_bytes": squish_disk,
            "version": subprocess.run([SQUISH_BIN, "--version"], capture_output=True, text=True).stdout.strip(),
        },
    }

    for tool in ("ollama", "squish"):
        phase = run_phase(tool)
        # Cold medians
        cold_inference = summarize(
            phase["cold_runs"],
            ["ttft_s", "total_s", "tokens_per_sec", "completion_tokens"],
        )
        cold_server = {
            "wall_total_s_median": median([r["wall_total_s"] for r in phase["cold_runs"]]),
            "server_ready_s_median": median([r["server_ready_s"] for r in phase["cold_runs"]]),
            "peak_rss_bytes_median": median([r["peak_rss_bytes"] for r in phase["cold_runs"]]),
            "peak_rss_bytes_max": max([r["peak_rss_bytes"] for r in phase["cold_runs"]]),
        }
        warm_inference = summarize(
            phase["warm_runs"],
            ["ttft_s", "total_s", "tokens_per_sec", "completion_tokens"],
        )
        results[tool].update({
            "cold_runs_raw": phase["cold_runs"],
            "warm_runs_raw": phase["warm_runs"],
            "cold_median_inference": cold_inference,
            "cold_median_server": cold_server,
            "warm_median_inference": warm_inference,
        })

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    log(f"Wrote {OUT_JSON}")

    # Print summary table
    print_summary(results)


def fmt_bytes(n: float | None) -> str:
    if not n:
        return "—"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def fmt_s(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 1:
        return f"{v * 1000:.0f} ms"
    return f"{v:.2f} s"


def ratio(a: float | None, b: float | None, higher_is_better: bool = False) -> str:
    if a is None or b is None or a == 0 or b == 0:
        return "—"
    if higher_is_better:
        return f"{b / a:.2f}×"
    return f"{a / b:.2f}×"


def print_summary(r: dict[str, Any]) -> None:
    o = r["ollama"]
    s = r["squish"]
    print()
    print("# Ollama vs Squish — M3 MacBook Pro 16GB")
    print()
    print(f"Model: Ollama `{o['model']}` (Q4_K_M GGUF) vs Squish `{s['model']}` (INT4 MLX)")
    print(f"Ollama: {o['version']}")
    print(f"Squish: {s['version']}")
    print()
    print(f"| Metric              | Ollama          | Squish          | Delta   |")
    print(f"|---------------------|-----------------|-----------------|---------|")
    o_cold_ttft = o["cold_median_inference"]["ttft_s"]
    s_cold_ttft = s["cold_median_inference"]["ttft_s"]
    print(f"| Cold TTFT (median)  | {fmt_s(o_cold_ttft):<15} | {fmt_s(s_cold_ttft):<15} | {ratio(o_cold_ttft, s_cold_ttft):<7} |")
    o_cold_rss = o["cold_median_server"]["peak_rss_bytes_median"]
    s_cold_rss = s["cold_median_server"]["peak_rss_bytes_median"]
    print(f"| Cold peak RAM       | {fmt_bytes(o_cold_rss):<15} | {fmt_bytes(s_cold_rss):<15} | {ratio(o_cold_rss, s_cold_rss):<7} |")
    o_warm_tps = o["warm_median_inference"]["tokens_per_sec"]
    s_warm_tps = s["warm_median_inference"]["tokens_per_sec"]
    print(f"| Warm tokens/sec     | {(f'{o_warm_tps:.1f} tok/s' if o_warm_tps else '—'):<15} | {(f'{s_warm_tps:.1f} tok/s' if s_warm_tps else '—'):<15} | {ratio(o_warm_tps, s_warm_tps, higher_is_better=True):<7} |")
    o_wall = o["cold_median_server"]["wall_total_s_median"]
    s_wall = s["cold_median_server"]["wall_total_s_median"]
    print(f"| Cold total wall     | {fmt_s(o_wall):<15} | {fmt_s(s_wall):<15} | {ratio(o_wall, s_wall):<7} |")
    print(f"| Disk size (model)   | {fmt_bytes(o['model_disk_bytes']):<15} | {fmt_bytes(s['model_disk_bytes']):<15} | {ratio(o['model_disk_bytes'], s['model_disk_bytes']):<7} |")
    print()
    print(f"Raw results: {OUT_JSON}")


if __name__ == "__main__":
    main()
