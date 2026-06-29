"""System definitions and streaming clients — Squish INT4/INT3 and Ollama.

Encodes the fairness rules as configuration so they cannot be forgotten per run:

* **Like-for-like head-to-head** — ``squish_int4`` vs ``ollama_q4km`` (both
  ~4-bit). ``squish_int3`` is a separate *capability* system, never the
  head-to-head number.
* **Don't credit Squish for Ollama unloading** — Ollama runs with
  ``keep_alive=-1`` (model stays resident) and ``num_ctx`` sized to at least
  prompt+generation for every length. Exact flags are recorded per run.
* **Separate prefill from decode** — every stream captures TTFT and per-token
  timestamps so decode tok/s is measured independently of prefill.
* **Determinism** — temperature 0, fixed seed, fixed 200-token generation.
* **Speculative / prompt-lookup state** — recorded; a system variant runs it OFF
  for the isolation pass.

This module imports no MLX or psutil at top level, so it loads anywhere; the
launchers shell out to the already-installed ``squish``/``ollama`` binaries.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import cache_probe
from .memory import RSSSampler

# ── endpoints / paths (override via env on the bench host) ────────────────────

OLLAMA_BIN = os.environ.get("BENCH_OLLAMA_BIN", "/usr/local/bin/ollama")
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = int(os.environ.get("BENCH_OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.environ.get("BENCH_OLLAMA_MODEL", "qwen2.5:7b")

SQUISH_PY = os.environ.get("BENCH_SQUISH_PY", os.path.expanduser("~/squish/.venv/bin/python"))
SQUISH_HOST = "127.0.0.1"
SQUISH_PORT = int(os.environ.get("BENCH_SQUISH_PORT", "11435"))
SQUISH_API_KEY = os.environ.get("BENCH_SQUISH_API_KEY", "squish")
SQUISH_MODEL_INT4 = os.environ.get(
    "BENCH_SQUISH_INT4", os.path.expanduser("~/models/Qwen2.5-7B-Instruct-int4")
)
SQUISH_MODEL_INT3 = os.environ.get(
    "BENCH_SQUISH_INT3", os.path.expanduser("~/models/Qwen2.5-7B-Instruct-int3")
)

GEN_TOKENS = 200  # fixed generation length for e2e
GEN_HEADROOM = 256  # num_ctx = prompt + this, rounded up
FIXED_SEED = 1234


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── stream result ─────────────────────────────────────────────────────────────


@dataclass
class StreamResult:
    ttft_s: float | None
    total_s: float
    completion_tokens: int
    decode_tps: float | None
    chunk_timestamps: list[float]
    prompt_tokens: int | None = None
    done_chunk: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] | None = None
    failed: bool = False
    error: str = ""


def num_ctx_for(prompt_tokens: int) -> int:
    """num_ctx sized to prompt+generation, rounded to a 1024 multiple."""
    need = prompt_tokens + GEN_TOKENS + GEN_HEADROOM
    return ((need + 1023) // 1024) * 1024


# ── shared streaming loop ─────────────────────────────────────────────────────


@dataclass
class _Acc:
    """Per-token timestamp accumulator shared by both stream clients."""

    t_req: float
    t_first: float | None = None
    parts: list[str] = field(default_factory=list)
    stamps: list[float] = field(default_factory=list)

    def add(self, chunk: str) -> None:
        if not chunk:
            return
        t = time.perf_counter()
        self.stamps.append(t)
        if self.t_first is None:
            self.t_first = t
        self.parts.append(chunk)


def _run_stream(req, parse_line) -> tuple[_Acc, dict[str, Any], str | None]:
    """Drive an HTTP stream, delegating each raw line to ``parse_line``.

    ``parse_line(raw, acc, meta)`` returns True to stop. Returns
    ``(acc, meta, error)`` where error is None on success.
    """
    acc = _Acc(time.perf_counter())
    meta: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                if parse_line(raw, acc, meta):
                    break
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return acc, meta, f"{type(exc).__name__}: {exc}"
    return acc, meta, None


# ── Ollama ────────────────────────────────────────────────────────────────────


def _parse_ollama_line(raw: bytes, acc: _Acc, meta: dict[str, Any]) -> bool:
    if not raw.strip():
        return False
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        return False
    acc.add(d.get("response", ""))
    if d.get("done"):
        meta["done"] = d
    return False


def stream_ollama(prompt: str, max_tokens: int, num_ctx: int) -> StreamResult:
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
            "keep_alive": -1,  # model stays resident — no unload penalty credited
            "options": {
                "num_predict": max_tokens,
                "temperature": 0.0,
                "seed": FIXED_SEED,
                "num_ctx": num_ctx,
            },
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    acc, meta, error = _run_stream(req, _parse_ollama_line)
    if error is not None:
        return StreamResult(
            None, time.perf_counter() - acc.t_req, 0, None, acc.stamps, failed=True, error=error
        )
    t_done = time.perf_counter()
    done = meta.get("done", {})
    eval_count = done.get("eval_count") or len(acc.parts)
    eval_ns = done.get("eval_duration") or 0
    tps = eval_count / (eval_ns / 1e9) if (eval_count and eval_ns) else None
    return StreamResult(
        ttft_s=(acc.t_first - acc.t_req) if acc.t_first else None,
        total_s=t_done - acc.t_req,
        completion_tokens=eval_count,
        decode_tps=tps,
        chunk_timestamps=acc.stamps,
        prompt_tokens=done.get("prompt_eval_count"),
        done_chunk=done,
    )


def start_ollama(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
    env = {**os.environ, "OLLAMA_HOST": f"{OLLAMA_HOST}:{OLLAMA_PORT}"}
    proc = subprocess.Popen(
        [OLLAMA_BIN, "serve"], stdout=open(log_path, "wb"), stderr=subprocess.STDOUT, env=env
    )
    sampler = RSSSampler(proc.pid)
    sampler.start()
    return proc, sampler


# ── Squish ────────────────────────────────────────────────────────────────────


def _parse_squish_line(raw: bytes, acc: _Acc, meta: dict[str, Any]) -> bool:
    line = raw.strip()
    if not line.startswith(b"data:"):
        return False
    payload = line[5:].strip()
    if payload == b"[DONE]":
        return True
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        return False
    choices = d.get("choices") or []
    if choices:
        acc.add((choices[0].get("delta") or {}).get("content") or "")
    if d.get("usage"):
        meta["usage"] = d["usage"]
    return False


def stream_squish(prompt: str, max_tokens: int, num_ctx: int) -> StreamResult:
    body = json.dumps(
        {
            "model": "squish",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": FIXED_SEED,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {SQUISH_API_KEY}"},
    )
    acc, meta, error = _run_stream(req, _parse_squish_line)
    if error is not None:
        return StreamResult(
            None, time.perf_counter() - acc.t_req, 0, None, acc.stamps, failed=True, error=error
        )
    t_done = time.perf_counter()
    usage = meta.get("usage")
    comp = (usage or {}).get("completion_tokens") or len(acc.parts)
    window = (t_done - acc.t_first) if acc.t_first else None
    tps = (comp / window) if (window and window > 0 and comp) else None
    return StreamResult(
        ttft_s=(acc.t_first - acc.t_req) if acc.t_first else None,
        total_s=t_done - acc.t_req,
        completion_tokens=comp,
        decode_tps=tps,
        chunk_timestamps=acc.stamps,
        prompt_tokens=(usage or {}).get("prompt_tokens"),
        usage=usage,
    )


def squish_metrics() -> dict[str, float]:
    """Snapshot Squish's /metrics counters+gauges (for cache-hit measurement)."""
    try:
        url = f"http://{SQUISH_HOST}:{SQUISH_PORT}/v1/metrics"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {SQUISH_API_KEY}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return cache_probe.parse_prometheus(resp.read().decode())
    except (urllib.error.URLError, OSError) as exc:
        _log(f"  [squish] /metrics unavailable: {exc}")
        return {}


def _squish_cmd(model_path: str, prompt_lookup: bool, cache_dirs: dict[str, str]) -> list[str]:
    cmd = [
        SQUISH_PY,
        "-m",
        "squish.server",
        "--mlx-model-dir",
        model_path,
        "--port",
        str(SQUISH_PORT),
        "--host",
        SQUISH_HOST,
        "--log-level",
        "warning",
        "--block-kv-cache",
        cache_dirs["block"],
        "--block-kv-size",
        "64",
        "--prompt-kv-cache",
        cache_dirs["pkv"],
    ]
    if not prompt_lookup:
        # Isolation pass: disable speculative prompt-lookup so its contribution
        # to the headline is explicit, not hidden.
        cmd += ["--no-prompt-lookup"]
    return cmd


def _make_squish_start(model_path: str, prompt_lookup: bool, tag: str) -> Callable[[Path], tuple]:
    def _start(log_path: Path) -> tuple[subprocess.Popen, RSSSampler]:
        import shutil

        dirs = {"block": f"/tmp/sqmx_block_{tag}", "pkv": f"/tmp/sqmx_pkv_{tag}"}
        for d in dirs.values():
            if os.path.isdir(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        proc = subprocess.Popen(
            _squish_cmd(model_path, prompt_lookup, dirs),
            stdout=open(log_path, "wb"),
            stderr=subprocess.STDOUT,
            env={**os.environ, "SQUISH_API_KEY": SQUISH_API_KEY},
        )
        sampler = RSSSampler(proc.pid)
        sampler.start()
        return proc, sampler

    return _start


# ── system registry ───────────────────────────────────────────────────────────


@dataclass
class System:
    name: str
    label: str
    quant: str
    role: str  # "head_to_head" | "capability" | "cross_check" | "isolation"
    ready_url: str
    start: Callable[[Path], tuple]
    stream: Callable[..., StreamResult]
    prompt_lookup: bool
    metrics: Callable[[], dict[str, float]] | None
    version_cmd: list[str]

    def read_version(self) -> str:
        try:
            out = subprocess.run(self.version_cmd, capture_output=True, text=True, timeout=20)
            return out.stdout.strip().replace("\n", " | ")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"<version unavailable: {exc}>"


def build_systems() -> dict[str, System]:
    """The standard system set. INT3 entries are dropped if the model is absent."""
    sq_ready = f"http://{SQUISH_HOST}:{SQUISH_PORT}/health"
    ol_ready = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/version"
    systems = {
        "ollama_q4km": System(
            "ollama_q4km",
            "Ollama Q4_K_M",
            "Q4_K_M",
            "head_to_head",
            ol_ready,
            start_ollama,
            stream_ollama,
            prompt_lookup=False,
            metrics=None,
            version_cmd=[OLLAMA_BIN, "--version"],
        ),
        "squish_int4": System(
            "squish_int4",
            "Squish INT4 (block+pkv)",
            "INT4",
            "head_to_head",
            sq_ready,
            _make_squish_start(SQUISH_MODEL_INT4, True, "int4"),
            stream_squish,
            prompt_lookup=True,
            metrics=squish_metrics,
            version_cmd=[SQUISH_PY, "-m", "squish", "--version"],
        ),
        "squish_int3": System(
            "squish_int3",
            "Squish INT3 (capability)",
            "INT3",
            "capability",
            sq_ready,
            _make_squish_start(SQUISH_MODEL_INT3, True, "int3"),
            stream_squish,
            prompt_lookup=True,
            metrics=squish_metrics,
            version_cmd=[SQUISH_PY, "-m", "squish", "--version"],
        ),
        "squish_int4_nospec": System(
            "squish_int4_nospec",
            "Squish INT4 (prompt-lookup OFF)",
            "INT4",
            "isolation",
            sq_ready,
            _make_squish_start(SQUISH_MODEL_INT4, False, "int4nospec"),
            stream_squish,
            prompt_lookup=False,
            metrics=squish_metrics,
            version_cmd=[SQUISH_PY, "-m", "squish", "--version"],
        ),
    }
    if not os.path.isdir(SQUISH_MODEL_INT3):
        systems.pop("squish_int3", None)
        _log("INT3 system SKIPPED — model dir not present.")
    return systems


def wait_ready(url: str, timeout: float = 300.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                r.read()
                return True
        except urllib.error.HTTPError:
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.2)
    return False


def stop_server(proc: subprocess.Popen, sampler: RSSSampler) -> None:
    sampler.stop()
    if proc.poll() is None:
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def kill_all_serving() -> None:
    for p in (
        "ollama serve",
        "ollama runner",
        "ollama_llama_server",
        "Ollama.app",
        "Ollama Helper",
        "squish.server",
        "squishd",
    ):
        subprocess.run(["pkill", "-f", p], capture_output=True)
    time.sleep(3)
