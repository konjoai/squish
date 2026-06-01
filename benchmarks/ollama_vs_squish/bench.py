#!/usr/bin/env python3
"""Side-by-side benchmark: Ollama vs Squish (three modes) on a single prompt.

Configurations measured (cold x5 + warm x3 each, median reported):

  1. ollama                - Ollama 0.18+ (qwen2.5:7b Q4_K_M GGUF)
  2. squish_eager          - squish.server with default eager load
  3. squish_lazy           - squish.server --lazy (bind first, load on first request)
  4. squish_preload_async  - squish.server --preload-async (bind first, bg-thread load)

For each tool we record TWO cold-time metrics so reviewers can pick the
fair comparison for their scenario:

  * cold_wall_s          - kill -> first-token-streamed-back. The user-
                           perspective metric. Includes server spawn, model
                           load, prefill, and first decode token.
  * cold_ttft_steady_s   - server-ready -> first-token. Continuity metric
                           from the v1 RESULTS.md. For eager-squish this is
                           just first-token latency (model is already loaded
                           when the port binds). For lazy/preload-async/
                           ollama this still includes the load cost because
                           "server ready" there only means "port is bound."

Persistent daemon mode is NOT measured - squish does not currently expose
a keep-alive daemon API distinct from the standalone server.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
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
    patterns = [
        "ollama serve", "ollama runner", "ollama_llama_server",
        "Ollama.app", "Ollama Helper",
        "squish run", "squish.cli run", "squish.server",
    ]
    for p in patterns:
        subprocess.run(["pkill", "-f", p], capture_output=True)
    time.sleep(3)


def wait_ready(url: str, timeout: float = 180) -> bool:
    """Return True as soon as the server responds with anything (4xx counts)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                r.read()
                return True
        except urllib.error.HTTPError:
            return True
        except Exception:
            time.sleep(0.1)
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


def stream_ollama(prompt: str, t0_external: float | None = None) -> dict[str, Any]:
    body = json.dumps({"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}).encode()
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t_req = time.perf_counter()
    t_first: float | None = None
    chunks = 0
    parts: list[str] = []
    eval_count = None
    eval_duration_ns = None
    load_duration_ns = None
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
                chunks += 1
                parts.append(chunk)
            if d.get("done"):
                eval_count = d.get("eval_count")
                eval_duration_ns = d.get("eval_duration")
                load_duration_ns = d.get("load_duration")
    t_done = time.perf_counter()
    tps = None
    if eval_count and eval_duration_ns:
        tps = eval_count / (eval_duration_ns / 1e9)
    return {
        "t_first":           t_first,
        "ttft_request_s":    (t_first - t_req) if t_first else None,
        "total_request_s":   t_done - t_req,
        "stream_chunks":     chunks,
        "completion_tokens": eval_count,
        "load_duration_s":   (load_duration_ns / 1e9) if load_duration_ns else None,
        "tokens_per_sec":    tps,
        "response":          "".join(parts),
    }


def stream_squish(prompt: str, t0_external: float | None = None) -> dict[str, Any]:
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
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {SQUISH_API_KEY}"},
    )
    t_req = time.perf_counter()
    t_first: float | None = None
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
                if chunk and t_first is None:
                    t_first = time.perf_counter()
                if chunk:
                    chunks += 1
                    parts.append(chunk)
            usage = d.get("usage")
            if usage:
                prompt_tokens = usage.get("prompt_tokens")
                completion_tokens = usage.get("completion_tokens")
    t_done = time.perf_counter()
    gen_window = (t_done - t_first) if t_first else None
    tokens_for_rate = completion_tokens or chunks
    tps = (tokens_for_rate / gen_window) if (gen_window and gen_window > 0 and tokens_for_rate) else None
    return {
        "t_first":           t_first,
        "ttft_request_s":    (t_first - t_req) if t_first else None,
        "total_request_s":   t_done - t_req,
        "stream_chunks":     chunks,
        "completion_tokens": completion_tokens or chunks,
        "prompt_tokens":     prompt_tokens,
        "tokens_per_sec":    tps,
        "response":          "".join(parts),
    }


# Server launch configurations


def _ollama_cmd() -> list[str]:
    return [OLLAMA_BIN, "serve"]


def _squish_cmd(mode: str) -> list[str]:
    base = [
        SQUISH_BIN, "run", SQUISH_MODEL_PATH,
        "--port", str(SQUISH_PORT),
        "--host", SQUISH_HOST,
        "--api-key", SQUISH_API_KEY,
        "--no-browser",
        "--log-level", "warning",
    ]
    if mode == "eager":
        return base
    if mode == "lazy":
        return base + ["--lazy"]
    if mode == "preload_async":
        return base + ["--preload-async"]
    raise ValueError(f"unknown squish mode: {mode}")


CONFIGS: dict[str, dict[str, Any]] = {
    "ollama": {
        "label":     "Ollama",
        "cmd":       _ollama_cmd(),
        "env":       {"OLLAMA_HOST": f"{OLLAMA_HOST}:{OLLAMA_PORT}"},
        "ready_url": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version",
        "stream_fn": stream_ollama,
    },
    "squish_eager": {
        "label":     "Squish (eager)",
        "cmd":       _squish_cmd("eager"),
        "env":       {},
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "stream_fn": stream_squish,
    },
    "squish_lazy": {
        "label":     "Squish (lazy)",
        "cmd":       _squish_cmd("lazy"),
        "env":       {},
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "stream_fn": stream_squish,
    },
    "squish_preload_async": {
        "label":     "Squish (preload-async)",
        "cmd":       _squish_cmd("preload_async"),
        "env":       {},
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "stream_fn": stream_squish,
    },
}


def start_server(cfg_id: str, log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    cfg = CONFIGS[cfg_id]
    env = {**os.environ, **cfg["env"]}
    proc = subprocess.Popen(
        cfg["cmd"],
        stdout=open(log_path, "wb"),
        stderr=subprocess.STDOUT,
        env=env,
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


def cold_run(cfg_id: str, idx: int) -> dict[str, Any]:
    log(f"[{cfg_id}] COLD run {idx + 1}/{COLD_RUNS} - killing prior servers")
    kill_all_serving()
    log_path = LOG_DIR / f"{cfg_id}_cold_{idx}.log"
    cfg = CONFIGS[cfg_id]
    t_kill_done = time.perf_counter()
    proc, sampler = start_server(cfg_id, log_path)
    try:
        if not wait_ready(cfg["ready_url"], timeout=180):
            raise RuntimeError(f"{cfg_id} did not become ready")
        t_ready = time.perf_counter()
        log(f"[{cfg_id}] server ready in {t_ready - t_kill_done:.2f}s - sending prompt")
        infer = cfg["stream_fn"](PROMPT, t0_external=t_kill_done)
        t_done = time.perf_counter()
    finally:
        stop_server(proc, sampler)

    cold_wall_s = (infer["t_first"] - t_kill_done) if infer.get("t_first") else None
    cold_ttft_steady_s = (infer["t_first"] - t_ready) if infer.get("t_first") else None
    return {
        "phase":              "cold",
        "run":                idx + 1,
        "server_ready_s":     t_ready - t_kill_done,
        "cold_wall_s":        cold_wall_s,
        "cold_ttft_steady_s": cold_ttft_steady_s,
        "wall_total_s":       t_done - t_kill_done,
        "peak_rss_bytes":     sampler.peak_bytes,
        "rss_samples":        sampler.samples,
        "inference":          infer,
    }


def warm_run(cfg_id: str, idx: int) -> dict[str, Any]:
    log(f"[{cfg_id}] WARM run {idx + 1}/{WARM_RUNS}")
    cfg = CONFIGS[cfg_id]
    infer = cfg["stream_fn"](PROMPT)
    return {"phase": "warm", "run": idx + 1, "inference": infer}


def run_config(cfg_id: str) -> dict[str, Any]:
    log(f"=== {cfg_id.upper()} cold phase ({COLD_RUNS} runs) ===")
    cold_runs = [cold_run(cfg_id, i) for i in range(COLD_RUNS)]

    log(f"=== {cfg_id.upper()} warm phase ({WARM_RUNS} runs) ===")
    kill_all_serving()
    warm_log = LOG_DIR / f"{cfg_id}_warm_server.log"
    proc, sampler = start_server(cfg_id, warm_log)
    cfg = CONFIGS[cfg_id]
    warm_runs: list[dict[str, Any]] = []
    try:
        if not wait_ready(cfg["ready_url"], timeout=180):
            raise RuntimeError(f"{cfg_id} warm-phase server did not start")
        log(f"[{cfg_id}] priming model (throwaway request, not counted)")
        cfg["stream_fn"](PROMPT)
        for i in range(WARM_RUNS):
            warm_runs.append(warm_run(cfg_id, i))
    finally:
        stop_server(proc, sampler)

    return {"cold_runs": cold_runs, "warm_runs": warm_runs}


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
        if bp.exists():
            total += bp.stat().st_size
        else:
            total += int(layer.get("size", 0) or 0)
    return total


def summarize_inference(runs: list[dict[str, Any]], keys: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for k in keys:
        vals = [r["inference"].get(k) for r in runs if r.get("inference")]
        vals = [float(v) for v in vals if v is not None]
        out[k] = median(vals) if vals else None
    return out


def summarize_run_field(runs: list[dict[str, Any]], key: str) -> float | None:
    vals = [r.get(key) for r in runs if r.get(key) is not None]
    vals = [float(v) for v in vals]
    return median(vals) if vals else None


def main() -> None:
    log(f"Output -> {OUT_JSON}")
    RESULTS_DIR.mkdir(exist_ok=True)

    ollama_disk = ollama_model_disk_size(OLLAMA_MODEL)
    squish_disk = disk_size_bytes(Path(SQUISH_MODEL_PATH))

    results: dict[str, Any] = {
        "timestamp": TS,
        "host":      "Apple M3 MacBook Pro 16GB",
        "prompt":    PROMPT,
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
        "configs": {},
    }

    for cfg_id in ("ollama", "squish_eager", "squish_lazy", "squish_preload_async"):
        phase = run_config(cfg_id)
        cold = phase["cold_runs"]
        warm = phase["warm_runs"]
        results["configs"][cfg_id] = {
            "label":         CONFIGS[cfg_id]["label"],
            "cold_runs_raw": cold,
            "warm_runs_raw": warm,
            "cold_median": {
                "cold_wall_s":        summarize_run_field(cold, "cold_wall_s"),
                "cold_ttft_steady_s": summarize_run_field(cold, "cold_ttft_steady_s"),
                "server_ready_s":     summarize_run_field(cold, "server_ready_s"),
                "wall_total_s":       summarize_run_field(cold, "wall_total_s"),
                "peak_rss_bytes":     summarize_run_field(cold, "peak_rss_bytes"),
                "tokens_per_sec":     summarize_inference(cold, ["tokens_per_sec"])["tokens_per_sec"],
                "completion_tokens":  summarize_inference(cold, ["completion_tokens"])["completion_tokens"],
            },
            "warm_median": summarize_inference(
                warm,
                ["ttft_request_s", "total_request_s", "tokens_per_sec", "completion_tokens"],
            ),
        }

    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    log(f"Wrote {OUT_JSON}")
    print_summary(results)


def fmt_bytes(n: float | None) -> str:
    if not n:
        return "-"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"


def fmt_s(v: float | None) -> str:
    if v is None:
        return "-"
    if v < 1:
        return f"{v * 1000:.0f} ms"
    return f"{v:.2f} s"


def fmt_tps(v: float | None) -> str:
    return f"{v:.1f} tok/s" if v else "-"


def winner_or_tie(
    label_value_pairs: list[tuple[str, float | None]],
    higher_is_better: bool,
    tie_threshold: float = 0.05,
) -> str:
    pairs = [(L, V) for L, V in label_value_pairs if V is not None and V > 0]
    if not pairs:
        return "-"
    best_label, best_value = (max if higher_is_better else min)(pairs, key=lambda p: p[1])
    for L, V in pairs:
        if L == best_label:
            continue
        ratio = V / best_value if higher_is_better else best_value / V
        if ratio < (1.0 - tie_threshold):
            return best_label
    return "tie"


def print_summary(r: dict[str, Any]) -> None:
    o    = r["configs"]["ollama"]
    s_e  = r["configs"]["squish_eager"]
    s_l  = r["configs"]["squish_lazy"]
    s_pa = r["configs"]["squish_preload_async"]

    print()
    print("# Ollama vs Squish - three serving modes  (M3 MacBook Pro 16GB)")
    print()
    print(f"Ollama: {r['ollama_version']}")
    print(f"Squish: {r['squish_version']}")
    print(
        f"Model:  Ollama `{r['models']['ollama']['name']}` (Q4_K_M GGUF)  "
        f"vs  Squish `{Path(r['models']['squish']['path']).name}` (INT4 MLX)"
    )
    print()
    print(
        "| Metric                              "
        "| Ollama        | Squish (eager) | Squish (lazy) | Squish (preload-async) "
        "| Winner        |"
    )
    print(
        "|-------------------------------------"
        "|---------------|----------------|---------------|------------------------"
        "|---------------|"
    )

    rows = [
        ("Cold wall (kill -> first token)",
         o["cold_median"]["cold_wall_s"], s_e["cold_median"]["cold_wall_s"],
         s_l["cold_median"]["cold_wall_s"], s_pa["cold_median"]["cold_wall_s"],
         False, fmt_s),
        ("Cold TTFT (server-ready -> first)",
         o["cold_median"]["cold_ttft_steady_s"], s_e["cold_median"]["cold_ttft_steady_s"],
         s_l["cold_median"]["cold_ttft_steady_s"], s_pa["cold_median"]["cold_ttft_steady_s"],
         False, fmt_s),
        ("Warm tokens/sec",
         o["warm_median"]["tokens_per_sec"], s_e["warm_median"]["tokens_per_sec"],
         s_l["warm_median"]["tokens_per_sec"], s_pa["warm_median"]["tokens_per_sec"],
         True, fmt_tps),
        ("Peak RAM (full process tree)",
         o["cold_median"]["peak_rss_bytes"], s_e["cold_median"]["peak_rss_bytes"],
         s_l["cold_median"]["peak_rss_bytes"], s_pa["cold_median"]["peak_rss_bytes"],
         False, fmt_bytes),
        ("Disk size (model)",
         r["models"]["ollama"]["disk_bytes"],
         r["models"]["squish"]["disk_bytes"],
         r["models"]["squish"]["disk_bytes"],
         r["models"]["squish"]["disk_bytes"],
         False, fmt_bytes),
    ]

    label_map = {
        "ollama":  "Ollama",
        "eager":   "Squish eager",
        "lazy":    "Squish lazy",
        "preload": "Squish preload",
    }

    for name, v_o, v_e, v_l, v_pa, higher_better, fmt in rows:
        winner = winner_or_tie(
            [("ollama", v_o), ("eager", v_e), ("lazy", v_l), ("preload", v_pa)],
            higher_is_better=higher_better,
        )
        winner_label = label_map.get(winner, winner)
        print(
            f"| {name:<37} "
            f"| {fmt(v_o):<13} | {fmt(v_e):<14} | {fmt(v_l):<13} | {fmt(v_pa):<22} "
            f"| {winner_label:<13} |"
        )

    print()
    print(f"Raw artifact: {OUT_JSON}")


if __name__ == "__main__":
    main()
