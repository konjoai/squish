#!/usr/bin/env python3
"""
bench_qwen25_7b_blazing.py — Qwen2.5-7B INT2 blazing-mode benchmark

Runs two server passes against the local Qwen2.5-7B-Instruct-int2 model:

  Pass A:  squish serve ... --blazing --int2          (no agent)
  Pass B:  squish serve ... --blazing --agent --int2  (agent preset)

Then generates a side-by-side comparison across four benchmark suites:
  1. Throughput  — TTFT, tokens/sec, latency across 8 diverse prompts
  2. Tool Calling — 6 structured function-dispatch tests
  3. Reasoning   — 5 multi-hop / logic / code tests
  4. Agentic     — 4 planning / decomposition tests

Usage
-----
    python3 dev/benchmarks/bench_qwen25_7b_blazing.py
    python3 dev/benchmarks/bench_qwen25_7b_blazing.py --port 11435
    python3 dev/benchmarks/bench_qwen25_7b_blazing.py --verbose
    python3 dev/benchmarks/bench_qwen25_7b_blazing.py --save dev/results/

Requirements
------------
    squish serve must not already be running on the target port.
    Model path: models/Qwen2.5-7B-Instruct-int2 (in repo root)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import urllib.request
import urllib.error

# ── Repo root ─────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

MODEL_NAME  = "Qwen2.5-7B-Instruct-int2"
MODEL_PATH  = str(REPO_ROOT / "models" / MODEL_NAME)
API_KEY     = "squish"
TIMEOUT_S   = 90
WARMUP_SECS = 120  # time to wait for the server to become ready (7B INT2 loads slowly)

# ── ANSI colour helpers ───────────────────────────────────────────────────────
G  = "\033[32m";  R = "\033[31m";  Y = "\033[33m"
C  = "\033[36m";  W = "\033[1;37m"; D = "\033[2m"; NC = "\033[0m"; B = "\033[1m"
PASS_  = f"{G}✓ PASS{NC}"
FAIL_  = f"{R}✗ FAIL{NC}"
SKIP_  = f"{Y}~ SKIP{NC}"


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class PerfResult:
    prompt:   str
    ttft_ms:  float
    tps:      float
    n_tokens: int
    elapsed:  float


@dataclass
class TestResult:
    name:    str
    passed:  bool
    elapsed: float
    tps:     float
    notes:   list[str] = field(default_factory=list)


@dataclass
class PassSummary:
    label:        str
    perf:         list[PerfResult]   = field(default_factory=list)
    tests:        list[TestResult]   = field(default_factory=list)
    load_secs:    float = 0.0


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    port: int = 11435,
    model: str = "squish",
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[dict, float]:
    """POST /v1/chat/completions; returns (response_json, elapsed_s)."""
    payload: dict[str, Any] = {
        "model":       model,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.loads(resp.read())
    return data, time.perf_counter() - t0


def _text(resp: dict) -> str:
    return resp["choices"][0]["message"].get("content") or ""


def _tool_calls(resp: dict) -> list[dict]:
    return resp["choices"][0]["message"].get("tool_calls") or []


def _tps(resp: dict, elapsed: float) -> float:
    usage = resp.get("usage", {})
    total = usage.get("completion_tokens", 0)
    return round(total / elapsed, 1) if elapsed > 0 else 0.0


def _wait_ready(port: int, max_wait: float = WARMUP_SECS) -> bool:
    """Poll /v1/models until the server is up, or return False on timeout."""
    deadline = time.perf_counter() + max_wait
    url = f"http://127.0.0.1:{port}/v1/models"
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


# ── Server lifecycle ──────────────────────────────────────────────────────────

def _start_server(port: int, blazing: bool, agent: bool) -> subprocess.Popen:
    """Start squish serve as a background process and wait until ready."""
    cmd = [
        sys.executable, "-m", "squish", "serve",
        MODEL_PATH,
        "--port",     str(port),
        "--int2",
        "--no-browser",
        "--log-level", "warning",
    ]
    if blazing:
        cmd.append("--blazing")
    if agent:
        cmd.append("--agent")

    label = "blazing" + ("+agent" if agent else "")
    print(f"\n{C}  Starting server [{label}] on port {port}…{NC}")
    print(f"  {D}cmd: {' '.join(cmd)}{NC}")

    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    t_start = time.perf_counter()
    ready = _wait_ready(port, max_wait=WARMUP_SECS)
    load_secs = time.perf_counter() - t_start

    if not ready:
        proc.terminate()
        raise RuntimeError(
            f"Server [{label}] did not become ready within {WARMUP_SECS}s"
        )
    print(f"  {G}Server ready in {load_secs:.1f}s{NC}")
    return proc


def _stop_server(proc: subprocess.Popen) -> None:
    """Gracefully terminate the server process."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    print(f"  {D}Server stopped.{NC}")


# ════════════════════════════════════════════════════════════════════
#  SUITE 1 — THROUGHPUT
# ════════════════════════════════════════════════════════════════════

PERF_PROMPTS = [
    ("greeting",      "Hello! What can you help me with?",                         64),
    ("math",          "What is 17 squared? Reply with just the number.",            16),
    ("haiku",         "Write a short haiku about the ocean.",                       48),
    ("python_basics", "Explain what a Python list comprehension is in two sentences.", 80),
    ("short_code",    "Write a Python function that returns the nth fibonacci number.", 120),
    ("reasoning",     "A store sold 240 items. 60% were electronics. How many non-electronics items? Show working.", 120),
    ("summarise",     "Summarise the key differences between supervised and unsupervised machine learning in 3 bullets.", 150),
    ("long_gen",      "Write a short Python class called Cache with get, set, and delete methods. Include docstrings.", 300),
]


def suite_throughput(port: int, verbose: bool) -> list[PerfResult]:
    print(f"\n{C}{'═'*62}{NC}")
    print(f"{W}  SUITE 1 — Throughput (TTFT + tokens/sec){NC}")
    print(f"{C}{'═'*62}{NC}")

    results: list[PerfResult] = []
    for name, prompt, max_tok in PERF_PROMPTS:
        try:
            resp, elapsed = _chat(
                [{"role": "user", "content": prompt}],
                port=port,
                max_tokens=max_tok,
            )
            usage = resp.get("usage", {})
            n_tok = usage.get("completion_tokens", 0)
            tps   = round(n_tok / elapsed, 1) if elapsed > 0 else 0.0
            # TTFT is not directly exposed via non-streaming; use elapsed/n_tok
            # as a proxy (first-token contribution ~= elapsed - (n_tok-1)/tps)
            ttft_ms = round((elapsed - max(0, (n_tok - 1) / tps if tps > 0 else 0)) * 1000, 1)
            pr = PerfResult(name, ttft_ms, tps, n_tok, elapsed)
            results.append(pr)
            print(f"  {G}✓{NC}  {name:<18}  {tps:5.1f} tok/s  "
                  f"{n_tok:3d} tok  {elapsed:.2f}s")
            if verbose:
                print(f"     {D}→ {_text(resp)[:120].strip()}{NC}")
        except Exception as exc:
            print(f"  {R}✗{NC}  {name:<18}  ERROR: {exc}")
            results.append(PerfResult(name, 0, 0, 0, 0))
    return results


# ════════════════════════════════════════════════════════════════════
#  SUITE 2 — TOOL CALLING
# ════════════════════════════════════════════════════════════════════

_TOOLS_WEATHER = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather data for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]

_TOOLS_MULTI = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python snippet and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout": {"type": "integer", "default": 10},
                },
                "required": ["code"],
            },
        },
    },
]


def _record(tests: list[TestResult], name: str, passed: bool,
            elapsed: float, tps: float, notes: list[str] = ()) -> None:
    r = TestResult(name, passed, elapsed, tps, list(notes))
    tests.append(r)
    status = PASS_ if passed else FAIL_
    print(f"  {status}  {W}{name}{NC}  {D}({elapsed:.2f}s  {tps} tok/s){NC}")
    for n in notes:
        print(f"         {D}{n}{NC}")


def suite_tool_calling(port: int, verbose: bool) -> list[TestResult]:
    print(f"\n{C}{'═'*62}{NC}")
    print(f"{W}  SUITE 2 — Tool Calling{NC}")
    print(f"{C}{'═'*62}{NC}")

    tests: list[TestResult] = []

    # 2.1 — single dispatch
    try:
        resp, el = _chat(
            [{"role": "user", "content": "What's the weather in Tokyo?"}],
            tools=_TOOLS_WEATHER, port=port,
        )
        calls = _tool_calls(resp)
        tps   = _tps(resp, el)
        args  = {}
        if calls:
            try: args = json.loads(calls[0]["function"]["arguments"])
            except Exception: pass
        passed = (len(calls) == 1 and calls[0]["function"]["name"] == "get_weather")
        _record(tests, "2.1 Single tool dispatch", passed, el, tps,
                [f"tool={calls[0]['function']['name'] if calls else 'none'}",
                 f"args={args}"])
    except Exception as e:
        _record(tests, "2.1 Single tool dispatch", False, 0, 0, [str(e)])

    # 2.2 — correct argument types
    try:
        resp, el = _chat(
            [{"role": "user", "content": "Get the weather in Paris in fahrenheit."}],
            tools=_TOOLS_WEATHER, port=port,
        )
        calls = _tool_calls(resp)
        tps   = _tps(resp, el)
        args  = {}
        if calls:
            try: args = json.loads(calls[0]["function"]["arguments"])
            except Exception: pass
        passed = (len(calls) == 1
                  and "paris" in args.get("city", "").lower()
                  and args.get("unit") == "fahrenheit")
        _record(tests, "2.2 Argument types", passed, el, tps,
                [f"city={args.get('city')}", f"unit={args.get('unit')}"])
    except Exception as e:
        _record(tests, "2.2 Argument types", False, 0, 0, [str(e)])

    # 2.3 — select correct tool from 3
    try:
        resp, el = _chat(
            [{"role": "user", "content": "Search the web for the latest MLX release."}],
            tools=_TOOLS_MULTI, port=port,
        )
        calls = _tool_calls(resp)
        tps   = _tps(resp, el)
        args  = {}
        if calls:
            try: args = json.loads(calls[0]["function"]["arguments"])
            except Exception: pass
        passed = (len(calls) >= 1 and calls[0]["function"]["name"] == "search_web")
        _record(tests, "2.3 Tool selection (3 tools)", passed, el, tps,
                [f"chose={calls[0]['function']['name'] if calls else 'none'}",
                 f"query={args.get('query','')[:60]}"])
    except Exception as e:
        _record(tests, "2.3 Tool selection (3 tools)", False, 0, 0, [str(e)])

    # 2.4 — no tool when not needed
    try:
        resp, el = _chat(
            [{"role": "user", "content": "What is 6 times 7?"}],
            tools=_TOOLS_MULTI, port=port,
        )
        calls = _tool_calls(resp)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = (len(calls) == 0 and "42" in text)
        _record(tests, "2.4 No tool (answer = 42)", passed, el, tps,
                [f"tool_calls={len(calls)}", f"answer={text[:60].strip()}"])
    except Exception as e:
        _record(tests, "2.4 No tool (answer = 42)", False, 0, 0, [str(e)])

    # 2.5 — tool result synthesis
    try:
        msgs = [
            {"role": "user",
             "content": "What's the weather in Berlin? Use the tool then summarize."},
            {"role": "assistant", "content": None,
             "tool_calls": [{
                 "id": "call_x1",
                 "type": "function",
                 "function": {
                     "name":      "get_weather",
                     "arguments": '{"city":"Berlin","unit":"celsius"}',
                 },
             }]},
            {"role": "tool",
             "tool_call_id": "call_x1",
             "content": '{"city":"Berlin","temp":9,"condition":"overcast","humidity":80}'},
        ]
        resp, el = _chat(msgs, port=port, max_tokens=150)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = ("9" in text or "overcast" in text.lower()) and len(text) > 20
        _record(tests, "2.5 Tool result synthesis", passed, el, tps,
                [f"response={text[:100].strip()}"])
    except Exception as e:
        _record(tests, "2.5 Tool result synthesis", False, 0, 0, [str(e)])

    # 2.6 — multi-step: two tool calls in one response
    try:
        resp, el = _chat(
            [{"role": "user",
              "content": "Search the web for Python 3.14 release date AND read the file /tmp/notes.txt."}],
            tools=_TOOLS_MULTI, port=port, max_tokens=300,
        )
        calls = _tool_calls(resp)
        tps   = _tps(resp, el)
        names = [c["function"]["name"] for c in calls] if calls else []
        passed = (len(calls) >= 2
                  and "search_web" in names and "read_file" in names)
        _record(tests, "2.6 Parallel tool dispatch (×2)", passed, el, tps,
                [f"n_calls={len(calls)}", f"names={names[:4]}"])
    except Exception as e:
        _record(tests, "2.6 Parallel tool dispatch (×2)", False, 0, 0, [str(e)])

    return tests


# ════════════════════════════════════════════════════════════════════
#  SUITE 3 — REASONING
# ════════════════════════════════════════════════════════════════════

def suite_reasoning(port: int, verbose: bool) -> list[TestResult]:
    print(f"\n{C}{'═'*62}{NC}")
    print(f"{W}  SUITE 3 — Reasoning{NC}")
    print(f"{C}{'═'*62}{NC}")

    tests: list[TestResult] = []

    # 3.1 — syllogism
    try:
        resp, el = _chat([
            {"role": "system", "content": "Answer concisely."},
            {"role": "user",
             "content": ("If all Glips are Snorps, and all Snorps are Flurbs, "
                         "are all Glips Flurbs? Answer yes or no first.")},
        ], port=port, max_tokens=80)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = text.lower().strip().startswith("yes") or text.lower()[:20].startswith("yes")
        _record(tests, "3.1 Syllogistic logic", passed, el, tps,
                [f"answer={text[:80].strip()}"])
    except Exception as e:
        _record(tests, "3.1 Syllogistic logic", False, 0, 0, [str(e)])

    # 3.2 — multi-step arithmetic
    try:
        resp, el = _chat([
            {"role": "user",
             "content": ("A train at 90 km/h departs 08:30, arrives 11:00. "
                         "Distance in km? Show work then state the final number.")},
        ], port=port, max_tokens=200)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = "225" in text
        _record(tests, "3.2 Multi-step arithmetic (225 km)", passed, el, tps,
                [f"contains_225={'225' in text}", f"snippet={text[:100].strip()}"])
    except Exception as e:
        _record(tests, "3.2 Multi-step arithmetic", False, 0, 0, [str(e)])

    # 3.3 — code trace
    try:
        resp, el = _chat([
            {"role": "user",
             "content": textwrap.dedent("""\
                What does g(4) return?

                def g(n):
                    if n <= 1: return n
                    return g(n-1) + g(n-2)

                Reply with just the number.""")},
        ], port=port, max_tokens=20)
        text  = _text(resp).strip()
        tps   = _tps(resp, el)
        passed = "3" in text[:10]
        _record(tests, "3.3 Code trace g(4)=3", passed, el, tps,
                [f"answer={text[:30]}"])
    except Exception as e:
        _record(tests, "3.3 Code trace", False, 0, 0, [str(e)])

    # 3.4 — counterfactual / negation
    try:
        resp, el = _chat([
            {"role": "user",
             "content": ("There are 5 red balls and 3 blue balls. "
                         "If I remove 2 red balls, how many balls remain? "
                         "Just state the number.")},
        ], port=port, max_tokens=20)
        text  = _text(resp).strip()
        tps   = _tps(resp, el)
        passed = "6" in text[:10]
        _record(tests, "3.4 Counterfactual (remove 2 → 6 remain)", passed, el, tps,
                [f"answer={text[:30]}"])
    except Exception as e:
        _record(tests, "3.4 Counterfactual", False, 0, 0, [str(e)])

    # 3.5 — self-consistency / verification
    try:
        resp, el = _chat([
            {"role": "user",
             "content": ("Is this Python snippet correct? "
                         "def double(x): return x * 3\n"
                         "Answer: does it correctly double a number? "
                         "Say yes or no with a brief reason.")},
        ], port=port, max_tokens=80)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = "no" in text.lower()[:20] or "wrong" in text.lower() or "incorrect" in text.lower()
        _record(tests, "3.5 Code correctness check (expects NO)", passed, el, tps,
                [f"answer={text[:100].strip()}"])
    except Exception as e:
        _record(tests, "3.5 Code verification", False, 0, 0, [str(e)])

    return tests


# ════════════════════════════════════════════════════════════════════
#  SUITE 4 — AGENTIC PLANNING
# ════════════════════════════════════════════════════════════════════

def suite_agentic(port: int, verbose: bool) -> list[TestResult]:
    print(f"\n{C}{'═'*62}{NC}")
    print(f"{W}  SUITE 4 — Agentic Planning{NC}")
    print(f"{C}{'═'*62}{NC}")

    tests: list[TestResult] = []

    # 4.1 — task decomposition
    try:
        resp, el = _chat([
            {"role": "system",
             "content": "You are an autonomous AI agent. Break tasks into numbered steps."},
            {"role": "user",
             "content": "Plan how to build a simple REST API with Python and FastAPI. "
                        "Give exactly 5 sequential numbered steps."},
        ], port=port, max_tokens=300)
        text  = _text(resp)
        tps   = _tps(resp, el)
        # Count numbered steps present
        import re
        steps = re.findall(r"(?m)^\s*\d+[.)]\s+\S", text)
        passed = len(steps) >= 4
        _record(tests, "4.1 Task decomposition (≥4 steps)", passed, el, tps,
                [f"steps_found={len(steps)}", f"snippet={text[:120].strip()}"])
    except Exception as e:
        _record(tests, "4.1 Task decomposition", False, 0, 0, [str(e)])

    # 4.2 — sequential decision making with context
    try:
        msgs = [
            {"role": "system",
             "content": "You are an agent. At each step say what action you will take and why."},
            {"role": "user",
             "content": "I want to rename all .txt files in /data/ to .bak. "
                        "What is the FIRST action you take and what tool/command?"},
        ]
        resp, el = _chat(msgs, port=port, max_tokens=150)
        text  = _text(resp)
        tps   = _tps(resp, el)
        # Expects mention of listing/ls or find as a first step
        keywords = ["list", "ls", "find", "glob", "os.listdir", "scandir", "directory"]
        passed = any(kw in text.lower() for kw in keywords)
        _record(tests, "4.2 First-step planning (list files)", passed, el, tps,
                [f"keywords={[k for k in keywords if k in text.lower()][:3]}",
                 f"snippet={text[:100].strip()}"])
    except Exception as e:
        _record(tests, "4.2 First-step planning", False, 0, 0, [str(e)])

    # 4.3 — goal-state recognition
    try:
        resp, el = _chat([
            {"role": "user",
             "content": ("You are monitoring a pipeline. "
                         "Step 1: fetch data — done. "
                         "Step 2: clean data — done. "
                         "Step 3: train model — still running. "
                         "Step 4: evaluate — not started. "
                         "Is the pipeline complete? Say yes or no. "
                         "What is the current blocking step?")},
        ], port=port, max_tokens=80)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = ("no" in text.lower()[:30] or "not" in text.lower()[:30]) and (
            "train" in text.lower() or "step 3" in text.lower()
        )
        _record(tests, "4.3 Pipeline state recognition", passed, el, tps,
                [f"answer={text[:120].strip()}"])
    except Exception as e:
        _record(tests, "4.3 Pipeline state recognition", False, 0, 0, [str(e)])

    # 4.4 — self-correction from negative feedback
    try:
        msgs = [
            {"role": "user",
             "content": "Write a Python function that returns True if a number is even."},
            {"role": "assistant",
             "content": "def is_even(n): return n % 2 == 1"},
            {"role": "user",
             "content": "That's wrong — it returns True for odd numbers. Fix it."},
        ]
        resp, el = _chat(msgs, port=port, max_tokens=100)
        text  = _text(resp)
        tps   = _tps(resp, el)
        passed = "== 0" in text or "% 2 == 0" in text
        _record(tests, "4.4 Self-correction (even check)", passed, el, tps,
                [f"fixed={'== 0' in text}", f"snippet={text[:100].strip()}"])
    except Exception as e:
        _record(tests, "4.4 Self-correction", False, 0, 0, [str(e)])

    return tests


# ════════════════════════════════════════════════════════════════════
#  RUN ONE PASS
# ════════════════════════════════════════════════════════════════════

def run_pass(
    label: str,
    port: int,
    blazing: bool,
    agent: bool,
    verbose: bool,
) -> PassSummary:
    print(f"\n\n{B}{C}{'╔' + '═'*60 + '╗'}{NC}")
    print(f"{B}{C}║  PASS: {label:<52}║{NC}")
    print(f"{B}{C}{'╚' + '═'*60 + '╝'}{NC}")

    summary = PassSummary(label=label)

    t_start = time.perf_counter()
    proc = _start_server(port=port, blazing=blazing, agent=agent)
    summary.load_secs = time.perf_counter() - t_start

    try:
        # Warm-up chat (not recorded)
        try:
            _chat([{"role": "user", "content": "Hello"}], port=port, max_tokens=8)
        except Exception:
            pass

        summary.perf  = suite_throughput(port, verbose)
        summary.tests = (
            suite_tool_calling(port, verbose)
            + suite_reasoning(port, verbose)
            + suite_agentic(port, verbose)
        )
    finally:
        _stop_server(proc)

    return summary


# ════════════════════════════════════════════════════════════════════
#  COMPARISON REPORT
# ════════════════════════════════════════════════════════════════════

def _avg(vals: list[float]) -> float:
    vals = [v for v in vals if v > 0]
    return round(sum(vals) / len(vals), 1) if vals else 0.0


def print_comparison(pa: PassSummary, pb: PassSummary) -> None:
    """Print a side-by-side comparison table of the two passes."""

    def _delta(a: float, b: float, higher_is_better: bool = True) -> str:
        if a == 0 or b == 0:
            return ""
        d = b - a
        pct = abs(d) / a * 100
        if d == 0:
            return ""
        arrow = "▲" if (d > 0) == higher_is_better else "▼"
        colour = G if (d > 0) == higher_is_better else R
        return f"  {colour}{arrow}{pct:.1f}%{NC}"

    sep = f"{D}{'─'*72}{NC}"
    print(f"\n\n{B}{W}{'═'*72}{NC}")
    print(f"{B}{W}  COMPARISON: {pa.label}  vs  {pb.label}{NC}")
    print(f"{B}{W}{'═'*72}{NC}")

    # ── Throughput ────────────────────────────────────────────────────────────
    print(f"\n{C}  Throughput{NC}")
    print(sep)
    print(f"  {'Prompt':<20}  {'A tok/s':>8}  {'B tok/s':>8}  {'Δ':>12}")
    print(sep)
    paired: dict[str, tuple[PerfResult | None, PerfResult | None]] = {}
    for p in pa.perf:
        paired[p.prompt] = (p, None)
    for p in pb.perf:
        paired[p.prompt] = (paired.get(p.prompt, (None, None))[0], p)
    for name, (a, b) in paired.items():
        a_tps = a.tps if a else 0
        b_tps = b.tps if b else 0
        dstr  = _delta(a_tps, b_tps)
        print(f"  {name:<20}  {a_tps:>8.1f}  {b_tps:>8.1f}{dstr}")
    print(sep)
    avg_a = _avg([p.tps for p in pa.perf])
    avg_b = _avg([p.tps for p in pb.perf])
    print(f"  {'AVERAGE':<20}  {avg_a:>8.1f}  {avg_b:>8.1f}{_delta(avg_a, avg_b)}")

    # ── Test pass rate ────────────────────────────────────────────────────────
    print(f"\n{C}  Test Results (all suites){NC}")
    print(sep)
    print(f"  {'Test':<42}  {'A':>4}  {'B':>4}")
    print(sep)
    by_name: dict[str, tuple[bool | None, bool | None]] = {}
    for t in pa.tests:
        by_name[t.name] = (t.passed, None)
    for t in pb.tests:
        by_name[t.name] = (by_name.get(t.name, (None, None))[0], t.passed)
    for name, (a_pass, b_pass) in by_name.items():
        a_str = f"{G}✓{NC}" if a_pass else (f"{R}✗{NC}" if a_pass is not None else " ")
        b_str = f"{G}✓{NC}" if b_pass else (f"{R}✗{NC}" if b_pass is not None else " ")
        print(f"  {name:<42}  {a_str:>4}  {b_str:>4}")

    a_passed = sum(1 for t in pa.tests if t.passed)
    b_passed = sum(1 for t in pb.tests if t.passed)
    a_total  = len(pa.tests)
    b_total  = len(pb.tests)
    print(sep)
    print(
        f"  {'SCORE':<42}  "
        f"{G if a_passed == a_total else Y}{a_passed}/{a_total}{NC}  "
        f"{G if b_passed == b_total else Y}{b_passed}/{b_total}{NC}"
    )

    # ── Overall verdict ───────────────────────────────────────────────────────
    print(f"\n{B}{W}  VERDICT{NC}")
    print(sep)
    winner_tps   = "B" if avg_b > avg_a else ("A" if avg_a > avg_b else "TIE")
    winner_tests = "B" if b_passed > a_passed else ("A" if a_passed > b_passed else "TIE")
    tps_diff     = abs(avg_b - avg_a)
    tps_pct      = round(tps_diff / avg_a * 100, 1) if avg_a > 0 else 0
    test_diff    = b_passed - a_passed

    print(f"  Throughput winner : {C}{winner_tps}{NC}  "
          f"({avg_a:.1f} vs {avg_b:.1f} tok/s, Δ{tps_pct:.1f}%)")
    print(f"  Test-pass winner  : {C}{winner_tests}{NC}  "
          f"({a_passed}/{a_total} vs {b_passed}/{b_total}, Δ{test_diff:+d} tests)")

    agent_label = pb.label
    if winner_tps == "B" or winner_tests == "B":
        verdict = f"{G}YES — {agent_label} improves results{NC}"
    elif winner_tps == "A" or winner_tests == "A":
        verdict = f"{Y}NO  — {pa.label} (without agent) is faster/more accurate{NC}"
    else:
        verdict = f"{D}NEUTRAL — both configurations performed equally{NC}"
    print(f"\n  --agent helps?  →  {verdict}")
    print(f"{B}{W}{'═'*72}{NC}\n")


# ════════════════════════════════════════════════════════════════════
#  SAVE RESULTS
# ════════════════════════════════════════════════════════════════════

def _save(pa: PassSummary, pb: PassSummary, out_dir: str) -> None:
    """Write JSON results to out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _summary_dict(p: PassSummary) -> dict:
        return {
            "label":     p.label,
            "load_secs": p.load_secs,
            "avg_tps":   _avg([r.tps for r in p.perf]),
            "perf": [{
                "prompt":   r.prompt,
                "tps":      r.tps,
                "n_tokens": r.n_tokens,
                "elapsed":  r.elapsed,
            } for r in p.perf],
            "tests": [{
                "name":    t.name,
                "passed":  t.passed,
                "elapsed": t.elapsed,
                "tps":     t.tps,
                "notes":   t.notes,
            } for t in p.tests],
            "pass_rate": f"{sum(t.passed for t in p.tests)}/{len(p.tests)}",
        }

    data = {
        "model":     MODEL_NAME,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pass_a":    _summary_dict(pa),
        "pass_b":    _summary_dict(pb),
    }
    fp = out / "bench_qwen25_7b_blazing.json"
    fp.write_text(json.dumps(data, indent=2))
    print(f"  Results saved → {fp}")


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port",    type=int,   default=11435)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--save",    metavar="DIR", default="",
                        help="Directory to save JSON results")
    args = parser.parse_args()

    print(f"\n{B}{C}Squish Qwen2.5-7B INT2 Blazing Benchmark{NC}")
    print(f"{D}Model: {MODEL_PATH}{NC}")
    print(f"{D}Port:  {args.port}{NC}")
    print(f"{D}Time:  {time.strftime('%Y-%m-%d %H:%M:%S')}{NC}")

    if not Path(MODEL_PATH).exists():
        print(f"\n{R}ERROR: Model not found at {MODEL_PATH}{NC}")
        sys.exit(1)

    # ── Pass A: --blazing (no --agent) ────────────────────────────────────────
    pass_a = run_pass(
        label   = "blazing (no agent)",
        port    = args.port,
        blazing = True,
        agent   = False,
        verbose = args.verbose,
    )

    # Brief pause between passes so the port is freed
    time.sleep(3)

    # ── Pass B: --blazing --agent ─────────────────────────────────────────────
    pass_b = run_pass(
        label   = "blazing + agent",
        port    = args.port,
        blazing = True,
        agent   = True,
        verbose = args.verbose,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(pass_a, pass_b)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_dir = args.save or str(REPO_ROOT / "dev" / "results")
    _save(pass_a, pass_b, save_dir)


if __name__ == "__main__":
    main()
