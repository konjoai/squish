#!/usr/bin/env python3
"""v5 long-context benchmark: shared-prefix TTFT vs Ollama.

The v4.2 benchmark used 75-token prompts.  v5 introduces a long-context
scenario where the same 1500-token prefix is sent repeatedly with only
the trailing 50 tokens varying — the agent / coding-assistant workload.

Configurations measured (5 runs each, median reported):

  * ``ollama``                   — qwen2.5:7b (no prefix cache; re-prefills the full
                                    1500+50 tokens every request)
  * ``squish_daemon``            — default fp16 KV; same fate as Ollama (re-prefill
                                    everything)
  * ``squish_pkv``               — v4.2 PromptKVStore; the full-prompt SHA-256 key
                                    means it misses on every variation
  * ``squish_block``             — v5 BlockKVCache; the 1500-token prefix is split
                                    into 64-token blocks and reused across variations,
                                    so only the 50-token suffix is prefilled

Scenarios:

  * **cold** — fresh long prompt the cache has never seen (first send).
                The block cache populates here; no win expected.
  * **variation** — same 1500-token prefix + a different 50-token suffix.
                   Block cache should drop TTFT dramatically vs the others.
  * **warm tok/s** — sustained 100-token decode.

Raw per-run JSON: ``results/benchmarks_v5/runs/<UTC-timestamp>/long_ctx.json``.
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

# A ~1500-token prompt: a system-prompt / context dump that an agent would
# typically pin at the top of every turn.  The repeated sentence pattern is
# intentionally varied enough that the tokenizer produces ~1500 tokens.
_LONG_BASE = (
    "You are a senior staff engineer reviewing pull requests in a large "
    "monorepo.  The codebase has the following conventions: every public "
    "function must have a docstring; every test must be deterministic; "
    "every database migration must be reversible.  The team's CI runs the "
    "full test suite, a linter, a type-checker, and a mutation-testing pass. "
    "PRs are auto-merged when they pass review-mode and have two approvals. "
    "You have just been assigned to review a PR.  The PR description says: "
    "'This change introduces a new Redis-backed session cache that wraps the "
    "existing session middleware.  The cache stores session tokens keyed by "
    "user ID with a TTL of 24 hours.  Sessions are evicted by LRU when the "
    "cache exceeds 100,000 entries.  On a cache miss the middleware falls "
    "back to the database lookup as before, then populates the cache.  "
    "Benchmarks: p50 session-lookup latency drops from 18ms to 1.2ms; p99 "
    "from 180ms to 9ms; cache hit rate at steady state is 94 percent.  "
    "Author confirms backwards compatibility — sessions created before the "
    "cache deployment are still readable.  No new dependencies beyond redis-py "
    "which is already pinned in requirements.txt.  Author has added unit tests "
    "covering: cache hit, cache miss, eviction under load, fallback when "
    "Redis is unreachable, and TTL expiry.  Integration tests verify multi-"
    "instance consistency.  Migration plan: deploy behind a feature flag at "
    "10 percent traffic, monitor p99 and error rate for 24h, ramp to 50 "
    "percent then 100 percent over a week.'  The diff modifies these files: "
    "auth/middleware.py adds the cache wrapper; auth/session.py adds the "
    "create/read/invalidate methods with cache integration; auth/cache.py is "
    "a new module exposing a thin Redis client; tests/auth/test_middleware.py "
    "extends the existing test suite; tests/auth/test_cache.py is a new file "
    "covering the cache module in isolation; requirements.txt adds no new "
    "lines (redis-py already pinned); docs/architecture.md updates the auth "
    "flow diagram and the cache-coherency section; .env.example adds the "
    "REDIS_URL config variable.  Pay particular attention to: thread safety, "
    "TTL refresh under contention, fallback latency budget when Redis is "
    "slow, and the impact on session token rotation if a key is evicted "
    "mid-request.  The author's previous PR introduced a subtle bug where "
    "session.invalidate() removed only the local copy and not the cached "
    "one, so be alert to similar coherence issues.  Use the following review "
    "format: start with a one-sentence summary; then list strengths in "
    "bullet form; then list risks in bullet form ordered by severity; then "
    "list test gaps; then either approve, request changes, or comment.  Do "
    "not make assumptions about code you have not seen; ask questions when "
    "behavior is unclear.  When suggesting refactors, propose specific "
    "function signatures or class structures rather than vague advice.  "
    "Pay particular attention to: error handling around Redis timeouts "
    "and connection failures; the interaction between TTL expiry and "
    "session-token rotation; backwards compatibility for sessions created "
    "before the cache was deployed; and any failure modes when the cache "
    "is partially evicted under memory pressure.  Be concrete about which "
    "test cases you would expect to see added or strengthened."
)

# Variations: same base, different "ask".
_VARIATIONS = [
    "  Now: summarize the most important risk in two sentences.",
    "  Now: list the top three test gaps you would expect to see filled.",
    "  Now: identify one place where thread-safety could be subtly wrong.",
    "  Now: write a one-line approval comment if the PR meets bar.",
    "  Now: propose one alternative naming for the cache module.",
]

WARM_TPS_PROMPT = "Summarize the Renaissance in two paragraphs of 100 words each."

RUNS = 5

OLLAMA_BIN   = "/usr/local/bin/ollama"
OLLAMA_HOST  = "127.0.0.1"
OLLAMA_PORT  = 11434
OLLAMA_MODEL = "qwen2.5:7b"

SQUISH_BIN  = "/Users/wscholl/squish/.venv/bin/squish"
SQUISH_PY   = "/Users/wscholl/squish/.venv/bin/python"
SQUISH_HOST = "127.0.0.1"
SQUISH_PORT = 11435
SQUISH_API_KEY    = "squish"
SQUISH_MODEL_PATH = "/Users/wscholl/models/Qwen2.5-7B-Instruct-int4"

BLOCK_CACHE_DIR   = "/tmp/squish_blocks_v5"
PKV_CACHE_DIR     = "/tmp/squish_pkv_v5"

REPO_ROOT  = Path(__file__).resolve().parents[2]
OUT_ROOT   = REPO_ROOT / "results" / "benchmarks_v5" / "runs"
TS         = time.strftime("%Y%m%dT%H%M%S")
OUT_DIR    = OUT_ROOT / TS
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR    = Path("/tmp/bench_v4_logs")
LOG_DIR.mkdir(exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Process / readiness helpers (copied from bench_v4.py) ─────────────────────

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


# ── Streaming clients ────────────────────────────────────────────────────────

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
    }


# ── Server launchers ──────────────────────────────────────────────────────────

def start_ollama(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    env = {**os.environ, "OLLAMA_HOST": f"{OLLAMA_HOST}:{OLLAMA_PORT}"}
    proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"],
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env=env,
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


def _squish_cmd(extra: list[str]) -> list[str]:
    return [
        SQUISH_PY, "-m", "squish.server",
        "--mlx-model-dir", SQUISH_MODEL_PATH,
        "--port", str(SQUISH_PORT), "--host", SQUISH_HOST,
        "--log-level", "warning", *extra,
    ]


def start_squish_daemon(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    proc = subprocess.Popen(
        _squish_cmd([]),
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
        _squish_cmd(["--prompt-kv-cache", PKV_CACHE_DIR]),
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
        _squish_cmd([
            "--block-kv-cache", BLOCK_CACHE_DIR,
            "--block-kv-size",  "64",
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


# ── Per-config benchmark ──────────────────────────────────────────────────────

CONFIGS = {
    "ollama": {
        "label":     "Ollama (warm)",
        "ready_url": f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version",
        "start":     start_ollama,
        "stream":    stream_ollama,
    },
    "squish_daemon": {
        "label":     "Squish daemon (no cache)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_daemon,
        "stream":    stream_squish,
    },
    "squish_pkv": {
        "label":     "Squish + --prompt-kv-cache (v4.2)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_pkv,
        "stream":    stream_squish,
    },
    "squish_block": {
        "label":     "Squish + --block-kv-cache (v5)",
        "ready_url": f"http://{SQUISH_HOST}:{SQUISH_PORT}/health",
        "start":     start_squish_block,
        "stream":    stream_squish,
    },
}


def _make_variation_prompt(idx: int) -> str:
    return _LONG_BASE + _VARIATIONS[idx % len(_VARIATIONS)]


def run_config(cfg_id: str) -> dict[str, Any]:
    cfg = CONFIGS[cfg_id]
    log(f"=== {cfg_id} : start server ===")
    kill_all_serving()
    log_path = LOG_DIR / f"v5_{cfg_id}_{TS}.log"
    proc, sampler = cfg["start"](log_path)
    try:
        if not wait_ready(cfg["ready_url"], timeout=240):
            raise RuntimeError(f"{cfg_id} did not become ready")
        log("  ready; priming with short request")
        # Warm up the model with a short prompt so the long-context measurement
        # isn't conflated with first-request startup costs.
        cfg["stream"]("Hello.", max_tokens=4)

        # ── Phase A: COLD long-prompt TTFT (first send of the long base) ──
        # For cached configs (pkv, block), this primes the cache.
        cold_runs: list[dict[str, Any]] = []
        for i in range(RUNS):
            # Distinct prompt each cold run (different last sentence) so PKV
            # gets no hits in this phase.  Block cache also miss-then-store on
            # the first one; subsequent cold runs MAY benefit because the
            # 1500-token prefix is shared across variations.
            prompt = _make_variation_prompt(i)
            d = cfg["stream"](prompt, max_tokens=1)
            cold_runs.append({"run": i + 1, **d})
            log(f"  cold run {i + 1}: ttft={int(d['ttft_s'] * 1000) if d['ttft_s'] else '-'}ms")

        # ── Phase B: VARIATION TTFT (same 1500-token prefix, different
        #            50-token suffix each run).  This is the agent workload. ──
        # First, ensure the prefix is primed exactly once.
        cfg["stream"](_make_variation_prompt(0), max_tokens=1)
        variation_runs: list[dict[str, Any]] = []
        for i in range(RUNS):
            prompt = _make_variation_prompt(i)
            d = cfg["stream"](prompt, max_tokens=1)
            variation_runs.append({"run": i + 1, **d})
            log(f"  variation run {i + 1}: ttft={int(d['ttft_s'] * 1000) if d['ttft_s'] else '-'}ms")

        # ── Phase C: warm tok/s ──
        warm_runs: list[dict[str, Any]] = []
        for i in range(RUNS):
            d = cfg["stream"](WARM_TPS_PROMPT, max_tokens=100)
            warm_runs.append({"run": i + 1, **d})
            _tps_v = d.get("tokens_per_sec")
            _tps_str = f"{_tps_v:.1f}" if _tps_v else "    -"
            log(f"  warm_tps run {i + 1}: {_tps_str} tok/s "
                f"({d.get('completion_tokens', '-')} tokens)")
    finally:
        stop_server(proc, sampler)

    return {
        "label":           cfg["label"],
        "peak_rss_bytes":  sampler.peak_bytes,
        "rss_samples":     sampler.samples,
        "cold_runs":       cold_runs,
        "variation_runs":  variation_runs,
        "warm_runs":       warm_runs,
    }


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


def summarize(cfg_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "cold_ttft_s":     stats_of([r["ttft_s"] for r in cfg_data["cold_runs"]]),
        "variation_ttft_s":stats_of([r["ttft_s"] for r in cfg_data["variation_runs"]]),
        "warm_tps":        stats_of([r["tokens_per_sec"] for r in cfg_data["warm_runs"]]),
        "peak_rss_bytes":  cfg_data["peak_rss_bytes"],
    }


def main() -> None:
    log(f"Output dir: {OUT_DIR}")
    # Count prompt tokens via a quick mlx_lm load (warns if base prompt is too short)
    try:
        from mlx_lm import load
        _, tok = load(SQUISH_MODEL_PATH)
        n_base = len(tok.encode(_LONG_BASE + _VARIATIONS[0]))
        log(f"long-context prompt: {n_base} tokens")
    except Exception as exc:
        n_base = None
        log(f"could not count tokens ({exc})")

    results: dict[str, Any] = {
        "timestamp": TS,
        "host":      "Apple M3 MacBook Pro 16 GB",
        "ollama_version": subprocess.run(
            [OLLAMA_BIN, "--version"], capture_output=True, text=True,
        ).stdout.strip(),
        "squish_version": subprocess.run(
            [SQUISH_BIN, "--version"], capture_output=True, text=True,
        ).stdout.strip(),
        "n_long_prompt_tokens": n_base,
        "runs_per_phase":  RUNS,
        "prompts": {
            "long_base":   _LONG_BASE,
            "variations":  _VARIATIONS,
            "warm_tps":    WARM_TPS_PROMPT,
        },
        "configs":  {},
        "summary":  {},
        "notes": [
            "cold_ttft_s measures the first send of each variation prompt "
            "(no shared prompt with prior runs, but cache configs may benefit "
            "from the shared 1500-token prefix even on a 'cold' run).",
            "variation_ttft_s measures repeated sends of variation prompts "
            "after one priming send.  This is the agent workload — same long "
            "prefix, different short tail every turn.",
            "warm_tps is unrelated to the long context; measures sustained "
            "decode on a short prompt.",
        ],
    }

    for cfg_id in CONFIGS:
        cfg_data = run_config(cfg_id)
        results["configs"][cfg_id] = cfg_data
        results["summary"][cfg_id] = summarize(cfg_data)

    out_json = OUT_DIR / "long_ctx.json"
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
    print("# v5 long-context benchmark — Squish vs Ollama (M3 16 GB)")
    print(f"Ollama: {r['ollama_version']}    Squish: {r['squish_version']}")
    print(f"Long-base prompt: ~{r.get('n_long_prompt_tokens', '?')} tokens "
          f"({len(_VARIATIONS)} suffix variations)")
    print()
    cfg_order = ["ollama", "squish_daemon", "squish_pkv", "squish_block"]
    short_labels = {
        "ollama":         "Ollama",
        "squish_daemon":  "sq daemon",
        "squish_pkv":     "sq +pkv (v4.2)",
        "squish_block":   "sq +block (v5)",
    }
    header = " | ".join([f"{'Metric':<38}"] + [f"{short_labels[c]:>15}" for c in cfg_order])
    print(header)
    print("-" * len(header))

    def _row(name: str, key: str, fmt_fn) -> None:
        vals = [s[c][key]["median"] if isinstance(s[c][key], dict) else s[c][key] for c in cfg_order]
        cells = [fmt_fn(v) for v in vals]
        print(" | ".join([f"{name:<38}"] + [f"{c:>15}" for c in cells]))

    _row("cold long-prompt TTFT (1st of each variation)",       "cold_ttft_s",      fmt_s)
    _row("variation TTFT (after priming, shared prefix)",       "variation_ttft_s", fmt_s)
    _row("warm tok/s (short-prompt 100-tok decode)",            "warm_tps",         fmt_tps)
    print(" | ".join([f"{'peak RSS (process tree)':<38}"]
                     + [f"{fmt_bytes(s[c]['peak_rss_bytes']):>15}" for c in cfg_order]))

    print()
    print("Per-run variation TTFT (ms):")
    for c in cfg_order:
        vals = [run["ttft_s"] for run in r["configs"][c]["variation_runs"]]
        v_str = " ".join(f"{t * 1000:.0f}ms" if t else "    -" for t in vals)
        print(f"  {short_labels[c]:<16}  {v_str}")


if __name__ == "__main__":
    sys.exit(main() or 0)
