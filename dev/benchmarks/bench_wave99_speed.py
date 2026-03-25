#!/usr/bin/env python3
"""dev/benchmarks/bench_wave99_speed.py

Wave 99 — Baseline + Post-Fix Speed Benchmark
==============================================
Measures TTFT, decode tok/s, and total latency against a live Squish server.
Also runs a 25-sample lm_eval ARC-Easy quality probe to catch coherence regressions.

Usage:
  # With server already running:
  python3 dev/benchmarks/bench_wave99_speed.py --host 127.0.0.1 --port 11435

  # Save result to JSON:
  python3 dev/benchmarks/bench_wave99_speed.py --out dev/results/bench_wave99_baseline.json

  # Compare two saved results:
  python3 dev/benchmarks/bench_wave99_speed.py --compare baseline.json postfix.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
API  = "http://{host}:{port}/v1"


# ── ANSI helpers ──────────────────────────────────────────────────────────────
G  = "\033[32m"
R  = "\033[31m"
Y  = "\033[33m"
B  = "\033[34m"
W  = "\033[1m"
DIM= "\033[2m"
RE = "\033[0m"


def _ok(msg):  print(f"  {G}✓{RE}  {msg}")
def _fail(msg):print(f"  {R}✗{RE}  {msg}")
def _info(msg):print(f"  {B}·{RE}  {msg}")
def _section(title): print(f"\n{W}── {title} {'─' * max(1, 60-len(title))}{RE}")


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TokenResult:
    prompt_id:  str
    ttft_ms:    float          # time to first token (ms)
    total_ms:   float          # wall time for full response (ms)
    tokens:     int            # total tokens generated
    tps:        float          # decode tok/s
    coherent:   bool = True    # rough coherence flag (non-empty, non-gibberish)
    error:      str  = ""


@dataclass
class BenchResult:
    timestamp:    str
    model:        str
    host:         str
    port:         int
    n_requests:   int
    ttft_p50:     float
    ttft_p95:     float
    tps_p50:      float
    tps_p95:      float
    total_ms_p50: float
    errors:       int
    lmeval_arc_acc:   float = -1.0   # ARC-Easy accuracy; -1 if not run
    tokens:       list[TokenResult] = field(default_factory=list)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _wait_ready(base: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(f"{base}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as r:
                data = json.loads(r.read())
                if data.get("status") in ("ok", "no_model"):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _get_model(base: str) -> str:
    try:
        req = urllib.request.Request(f"{base}/models")
        with urllib.request.urlopen(req, timeout=5) as r:
            data = json.loads(r.read())
            models = data.get("data", [])
            if models:
                return models[0].get("id", "unknown")
    except Exception:
        pass
    return "unknown"


def _chat_sse(base: str, prompt: str, max_tokens: int = 200,
              temperature: float = 0.6) -> TokenResult:
    """Send one chat request; stream SSE, measure TTFT and tok/s."""
    payload = json.dumps({
        "model": "squish",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data    = payload,
        headers = {"Content-Type": "application/json", "Authorization": "Bearer squish"},
        method  = "POST",
    )
    t_start = time.perf_counter()
    ttft_ms = -1.0
    tokens  = 0
    text    = ""
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_str = line[5:].strip()
                if payload_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload_str)
                except json.JSONDecodeError:
                    continue
                delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                fragment = delta.get("content") or ""
                if fragment:
                    if ttft_ms < 0:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    tokens += 1
                    text += fragment
        total_ms = (time.perf_counter() - t_start) * 1000.0
        tps = tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0
        # Rough coherence: response is non-empty and doesn't look like pure repetition
        coherent = (
            len(text.strip()) > 10
            and not _is_repetitive(text)
        )
        return TokenResult(
            prompt_id = prompt[:40],
            ttft_ms   = max(ttft_ms, 0.0),
            total_ms  = total_ms,
            tokens    = tokens,
            tps       = tps,
            coherent  = coherent,
        )
    except Exception as exc:
        total_ms = (time.perf_counter() - t_start) * 1000.0
        return TokenResult(
            prompt_id = prompt[:40],
            ttft_ms   = 0.0,
            total_ms  = total_ms,
            tokens    = 0,
            tps       = 0.0,
            coherent  = False,
            error     = str(exc),
        )


def _is_repetitive(text: str) -> bool:
    """Detect obviously incoherent repetitive output."""
    if len(text) < 60:
        return False
    words = text.split()
    if len(words) < 20:
        return False
    # Check if more than 40% of words are a repeated token
    from collections import Counter
    c = Counter(words)
    most_common_count = c.most_common(1)[0][1]
    return most_common_count / len(words) > 0.40


# ── Prompt suite ──────────────────────────────────────────────────────────────

PROMPTS = [
    # TTFT-heavy: short prompt, short answer
    ("ttft_greeting",     "Say hello in exactly two words.",                60),
    ("ttft_math_small",   "What is 17 × 23?",                               60),
    # Decode-heavy: longer responses
    ("decode_explain",    "Explain how a transformer attention mechanism works in 3 sentences.", 150),
    ("decode_code",       "Write a Python function that reverses a linked list.",               200),
    ("decode_story",      "Write a two-sentence story about a robot learning to cook.",         100),
    # Mixed
    ("mixed_reasoning",   "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", 120),
    ("mixed_haiku",       "Write a haiku about machine learning.",            60),
    ("mixed_summarise",   "Summarise the French Revolution in two sentences.", 120),
]


# ── lm_eval quick probe ───────────────────────────────────────────────────────

def _run_lmeval_probe(host: str, port: int, n_samples: int = 25) -> float:
    """Run a 25-sample ARC-Easy probe via lm_eval's local-completions backend.

    Returns accuracy (0.0–1.0) or -1.0 on failure.
    """
    try:
        import lm_eval
        from lm_eval import evaluator
        from lm_eval.models.openai_completions import LocalChatCompletion
    except ImportError:
        _info("lm_eval not installed — skipping quality probe")
        return -1.0

    _info(f"Running lm_eval ARC-Easy probe ({n_samples} samples) …")
    try:
        results = evaluator.simple_evaluate(
            model       = "local-chat-completions",
            model_args  = f"model=squish,base_url=http://{host}:{port}/v1,api_key=squish,num_concurrent=1",
            tasks       = ["arc_easy"],
            limit       = n_samples,
            num_fewshot = 0,
            log_samples = False,
            verbosity   = "ERROR",
        )
        acc = results["results"]["arc_easy"].get("acc,none", -1.0)
        return float(acc)
    except Exception as exc:
        _info(f"lm_eval probe failed: {exc}")
        return -1.0


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(host: str, port: int, warmup: int = 2,
                  lmeval: bool = True, n_lmeval: int = 25) -> BenchResult:
    base = f"http://{host}:{port}/v1"

    _section("Squish Speed Benchmark — Wave 99")
    _info(f"Target: {base}")

    if not _wait_ready(base, timeout=10):
        _fail("Server not reachable — is squish running?")
        sys.exit(1)
    _ok("Server reachable")

    model = _get_model(base)
    _info(f"Model: {model}")

    # Warmup runs
    _section(f"Warmup ({warmup} requests, results discarded)")
    for i in range(warmup):
        r = _chat_sse(base, "Say hi.", max_tokens=20, temperature=0.0)
        _info(f"  warmup {i+1}: ttft={r.ttft_ms:.0f}ms  tps={r.tps:.1f}")

    # Benchmark runs
    _section(f"Benchmark ({len(PROMPTS)} prompts × 2 runs = {len(PROMPTS)*2} requests)")
    raw: list[TokenResult] = []
    for run in range(2):   # 2 passes for stability
        for pid, prompt, max_tok in PROMPTS:
            r = _chat_sse(base, prompt, max_tokens=max_tok, temperature=0.6)
            if r.error:
                _fail(f"{pid}: {r.error[:80]}")
            else:
                mark = G + "✓" + RE if r.coherent else R + "✗" + RE
                print(f"  {mark}  {pid:<28}  ttft={r.ttft_ms:6.0f}ms  "
                      f"tps={r.tps:5.1f}  tokens={r.tokens:3d}  "
                      f"{'INCOHERENT' if not r.coherent else ''}")
            raw.append(r)

    # Stats
    good = [r for r in raw if not r.error]
    errors = len(raw) - len(good)
    ttfts = sorted(r.ttft_ms for r in good)
    tpss  = sorted(r.tps    for r in good)
    totals= sorted(r.total_ms for r in good)

    def _pct(vals, p):
        if not vals: return 0.0
        idx = max(0, int(len(vals) * p / 100) - 1)
        return vals[idx]

    ttft_p50 = _pct(ttfts, 50)
    ttft_p95 = _pct(ttfts, 95)
    tps_p50  = _pct(tpss,  50)
    tps_p95  = _pct(tpss,  95)
    total_p50= _pct(totals, 50)

    _section("Results")
    print(f"  TTFT  p50={ttft_p50:.0f}ms   p95={ttft_p95:.0f}ms")
    print(f"  tok/s p50={tps_p50:.1f}   p95={tps_p95:.1f}")
    print(f"  total p50={total_p50:.0f}ms")
    incoherent = sum(1 for r in good if not r.coherent)
    if incoherent:
        _fail(f"{incoherent} incoherent response(s)")
    else:
        _ok("All responses coherent")
    if errors:
        _fail(f"{errors} errors")

    # lm_eval probe
    arc_acc = -1.0
    if lmeval:
        arc_acc = _run_lmeval_probe(host, port, n_samples=n_lmeval)
        if arc_acc >= 0:
            label = G + "PASS" + RE if arc_acc >= 0.60 else R + "FAIL" + RE
            print(f"  ARC-Easy acc ({n_lmeval} samples): {arc_acc:.3f}  [{label}]")

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    return BenchResult(
        timestamp    = ts,
        model        = model,
        host         = host,
        port         = port,
        n_requests   = len(raw),
        ttft_p50     = ttft_p50,
        ttft_p95     = ttft_p95,
        tps_p50      = tps_p50,
        tps_p95      = tps_p95,
        total_ms_p50 = total_p50,
        errors       = errors,
        lmeval_arc_acc = arc_acc,
        tokens       = raw,
    )


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(before_path: str, after_path: str) -> None:
    with open(before_path) as f: b = json.load(f)
    with open(after_path)  as f: a = json.load(f)

    def _delta(key: str, lower_is_better: bool = True) -> str:
        bv, av = b[key], a[key]
        if bv == 0: return "n/a"
        delta_pct = (av - bv) / bv * 100
        if lower_is_better:
            color = G if delta_pct <= -5 else (R if delta_pct >= 5 else W)
        else:
            color = G if delta_pct >= 5 else (R if delta_pct <= -5 else W)
        sign = "+" if delta_pct >= 0 else ""
        return f"{color}{sign}{delta_pct:.1f}%{RE}"

    _section("Before → After Comparison")
    print(f"  {'Metric':<22} {'Before':>10} {'After':>10}  {'Delta':>10}")
    print(f"  {'─'*22} {'─'*10} {'─'*10}  {'─'*10}")
    print(f"  {'TTFT p50 (ms)':<22} {b['ttft_p50']:>10.0f} {a['ttft_p50']:>10.0f}  {_delta('ttft_p50', True):>10}")
    print(f"  {'TTFT p95 (ms)':<22} {b['ttft_p95']:>10.0f} {a['ttft_p95']:>10.0f}  {_delta('ttft_p95', True):>10}")
    print(f"  {'tok/s p50':<22} {b['tps_p50']:>10.1f} {a['tps_p50']:>10.1f}  {_delta('tps_p50', False):>10}")
    print(f"  {'tok/s p95':<22} {b['tps_p95']:>10.1f} {a['tps_p95']:>10.1f}  {_delta('tps_p95', False):>10}")
    print(f"  {'Total p50 (ms)':<22} {b['total_ms_p50']:>10.0f} {a['total_ms_p50']:>10.0f}  {_delta('total_ms_p50', True):>10}")
    if b.get('lmeval_arc_acc', -1) >= 0 and a.get('lmeval_arc_acc', -1) >= 0:
        print(f"  {'ARC-Easy acc':<22} {b['lmeval_arc_acc']:>10.3f} {a['lmeval_arc_acc']:>10.3f}  {_delta('lmeval_arc_acc', False):>10}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Wave 99 speed benchmark")
    ap.add_argument("--host",     default="127.0.0.1")
    ap.add_argument("--port",     type=int, default=11435)
    ap.add_argument("--warmup",   type=int, default=2, help="Warmup requests (discarded)")
    ap.add_argument("--no-lmeval", action="store_true",   help="Skip lm_eval quality probe")
    ap.add_argument("--lmeval-n", type=int, default=25,   help="lm_eval ARC-Easy sample count")
    ap.add_argument("--out",      default="", help="Save JSON to this path")
    ap.add_argument("--compare",  nargs=2, metavar=("BEFORE", "AFTER"),
                    help="Compare two saved result JSONs")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1])
        return

    result = run_benchmark(
        host    = args.host,
        port    = args.port,
        warmup  = args.warmup,
        lmeval  = not args.no_lmeval,
        n_lmeval= args.lmeval_n,
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Serialise: convert TokenResult list to dicts
        d = asdict(result)
        out_path.write_text(json.dumps(d, indent=2))
        print(f"\n  Saved → {out_path}")
    else:
        print(f"\n  (Use --out FILE to save for comparison)")


if __name__ == "__main__":
    main()
