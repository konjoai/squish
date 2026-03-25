#!/usr/bin/env python3
"""
Internal runner: executes all 4 benchmark suites against a running squish
server and prints results + JSON summary.

Usage (called from bench_qwen25_7b_blazing.py):
    python3 dev/benchmarks/_run_bench_pass.py \
        --port 11435 \
        --label "blazing (no agent)" \
        --out /tmp/pass_a.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from typing import Any

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--port",  type=int, default=11435)
ap.add_argument("--label", default="?")
ap.add_argument("--out",   default="")
ap.add_argument("--verbose", action="store_true")
args = ap.parse_args()

PORT  = args.port
API   = "squish"
TOUT  = 90

# ── ANSI ──────────────────────────────────────────────────────────────────────
G  = "\033[32m"; R = "\033[31m"; C = "\033[36m"
W  = "\033[1;37m"; D = "\033[2m"; NC = "\033[0m"
OK = f"{G}PASS{NC}"; FL = f"{R}FAIL{NC}"

# ── HTTP helpers ──────────────────────────────────────────────────────────────

def chat(msgs: list[dict], tools: list[dict] | None = None,
         max_tokens: int = 200, temperature: float = 0.0) -> tuple[dict, float]:
    payload: dict[str, Any] = {
        "model":       "squish",
        "messages":    msgs,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "stream":      False,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {API}",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TOUT) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - t0


def txt(r: dict) -> str:
    return r["choices"][0]["message"].get("content") or ""


def tcs(r: dict) -> list[dict]:
    return r["choices"][0]["message"].get("tool_calls") or []


def ntps(r: dict, el: float) -> float:
    n = r.get("usage", {}).get("completion_tokens", 0)
    return round(n / el, 1) if el > 0 else 0.0


# ── Data ──────────────────────────────────────────────────────────────────────
results: list[dict] = []
perf_rows: list[dict] = []


def record(name: str, passed: bool, el: float, tp: float, note: str = "") -> None:
    results.append({"name": name, "passed": passed, "el": round(el, 2), "tps": tp})
    status = OK if passed else FL
    marker = f"{G}✓{NC}" if passed else f"{R}✗{NC}"
    print(f"  {marker}  {name:<45}  {tp:5.1f} tok/s  {D}{note[:60]}{NC}")


# ══════════════════════════════════════════════════════════════════
#  SUITE 1 — THROUGHPUT
# ══════════════════════════════════════════════════════════════════

PERF_PROMPTS = [
    ("greeting",      "Hello! What can you help me with?",                          64),
    ("math",          "What is 17 squared? Reply with just the number.",             16),
    ("haiku",         "Write a short haiku about the ocean.",                        48),
    ("py_basics",     "Explain what a list comprehension is in 2 sentences.",        80),
    ("short_code",    "Write a Python function returning the nth fibonacci number.", 120),
    ("reasoning",     "Store sold 240 items, 60% electronics. How many non-electronics? Show working.", 120),
    ("summarise",     "Summarise supervised vs unsupervised ML in 3 bullets.",       150),
    ("long_gen",      "Write a Python Cache class with get, set, delete + docstrings.", 300),
]

print(f"\n{C}{'='*64}{NC}")
print(f"{W}  SUITE 1 — Throughput (TTFT proxy + tokens/sec){NC}")
print(f"{C}{'='*64}{NC}")

tps_vals: list[float] = []
for pname, prompt, max_tok in PERF_PROMPTS:
    try:
        resp, el = chat([{"role": "user", "content": prompt}], max_tokens=max_tok)
        usage = resp.get("usage", {})
        n_tok = usage.get("completion_tokens", 0)
        tp_   = round(n_tok / el, 1) if el > 0 else 0.0
        tps_vals.append(tp_)
        perf_rows.append({"prompt": pname, "tps": tp_, "n_tokens": n_tok, "elapsed": round(el, 2)})
        print(f"  {G}✓{NC}  {pname:<18}  {tp_:5.1f} tok/s  {n_tok:3d} tok  {el:.2f}s")
        if args.verbose:
            print(f"     {D}→ {txt(resp)[:120].strip()}{NC}")
    except Exception as exc:
        print(f"  {R}✗{NC}  {pname:<18}  ERROR: {exc}")
        perf_rows.append({"prompt": pname, "tps": 0.0, "n_tokens": 0, "elapsed": 0.0})

avg_tps = round(sum(tps_vals) / len(tps_vals), 1) if tps_vals else 0.0
print(f"\n  AVG throughput: {C}{avg_tps} tok/s{NC}")


# ══════════════════════════════════════════════════════════════════
#  SUITE 2 — TOOL CALLING
# ══════════════════════════════════════════════════════════════════

TW = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
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

TM = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
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
            "description": "Execute Python and return stdout.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    },
]

print(f"\n{C}{'='*64}{NC}")
print(f"{W}  SUITE 2 — Tool Calling{NC}")
print(f"{C}{'='*64}{NC}")

# 2.1 single dispatch
try:
    r, el = chat([{"role": "user", "content": "What's the weather in Tokyo?"}], tools=TW)
    cs = tcs(r); tp = ntps(r, el)
    arg: dict = {}
    if cs:
        try: arg = json.loads(cs[0]["function"]["arguments"])
        except Exception: pass
    record("2.1 Single tool dispatch",
           len(cs) == 1 and cs[0]["function"]["name"] == "get_weather",
           el, tp, f"tool={cs[0]['function']['name'] if cs else 'none'}")
except Exception as e:
    record("2.1 Single tool dispatch", False, 0, 0, str(e))

# 2.2 arg types
try:
    r, el = chat([{"role": "user", "content": "Get the weather in Paris in fahrenheit."}], tools=TW)
    cs = tcs(r); tp = ntps(r, el); arg = {}
    if cs:
        try: arg = json.loads(cs[0]["function"]["arguments"])
        except Exception: pass
    record("2.2 Argument types",
           len(cs) == 1 and "paris" in arg.get("city", "").lower() and arg.get("unit") == "fahrenheit",
           el, tp, f"city={arg.get('city')} unit={arg.get('unit')}")
except Exception as e:
    record("2.2 Argument types", False, 0, 0, str(e))

# 2.3 tool selection
try:
    r, el = chat([{"role": "user", "content": "Search the web for the latest MLX release."}], tools=TM)
    cs = tcs(r); tp = ntps(r, el); arg = {}
    if cs:
        try: arg = json.loads(cs[0]["function"]["arguments"])
        except Exception: pass
    record("2.3 Tool selection (3 tools)",
           len(cs) >= 1 and cs[0]["function"]["name"] == "search_web",
           el, tp, f"chose={cs[0]['function']['name'] if cs else 'none'}")
except Exception as e:
    record("2.3 Tool selection", False, 0, 0, str(e))

# 2.4 no tool
try:
    r, el = chat([{"role": "user", "content": "What is 6 times 7?"}], tools=TM)
    cs = tcs(r); t_ = txt(r); tp = ntps(r, el)
    record("2.4 No tool when not needed (answer=42)",
           len(cs) == 0 and "42" in t_,
           el, tp, f"calls={len(cs)} ans={t_[:40].strip()}")
except Exception as e:
    record("2.4 No tool", False, 0, 0, str(e))

# 2.5 tool result synthesis
try:
    msgs = [
        {"role": "user", "content": "What's the weather in Berlin? Use the tool then summarize."},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "t1", "type": "function",
                          "function": {"name": "get_weather",
                                       "arguments": '{"city":"Berlin","unit":"celsius"}'}}]},
        {"role": "tool", "tool_call_id": "t1",
         "content": '{"city":"Berlin","temp":9,"condition":"overcast","humidity":80}'},
    ]
    r, el = chat(msgs, max_tokens=150); t_ = txt(r); tp = ntps(r, el)
    record("2.5 Tool result synthesis",
           ("9" in t_ or "overcast" in t_.lower()) and len(t_) > 20,
           el, tp, t_[:60].strip())
except Exception as e:
    record("2.5 Tool result synthesis", False, 0, 0, str(e))

# 2.6 parallel tool calls
try:
    r, el = chat(
        [{"role": "user",
          "content": "Search the web for Python 3.14 release AND read the file /tmp/notes.txt."}],
        tools=TM, max_tokens=300,
    )
    cs = tcs(r); tp = ntps(r, el)
    names = [c["function"]["name"] for c in cs] if cs else []
    record("2.6 Parallel tool dispatch (x2)",
           len(cs) >= 2 and "search_web" in names and "read_file" in names,
           el, tp, f"n={len(cs)} names={names[:3]}")
except Exception as e:
    record("2.6 Parallel tool dispatch", False, 0, 0, str(e))


# ══════════════════════════════════════════════════════════════════
#  SUITE 3 — REASONING
# ══════════════════════════════════════════════════════════════════

print(f"\n{C}{'='*64}{NC}")
print(f"{W}  SUITE 3 — Reasoning{NC}")
print(f"{C}{'='*64}{NC}")

# 3.1 syllogism
try:
    r, el = chat([
        {"role": "system", "content": "Answer concisely."},
        {"role": "user",
         "content": "If all Glips are Snorps, and all Snorps are Flurbs, are all Glips Flurbs? Say yes or no first."},
    ], max_tokens=80)
    t_ = txt(r); tp = ntps(r, el)
    record("3.1 Syllogistic logic",
           t_.lower().strip().startswith("yes"),
           el, tp, t_[:60].strip())
except Exception as e:
    record("3.1 Syllogistic logic", False, 0, 0, str(e))

# 3.2 multi-step arithmetic
try:
    r, el = chat([
        {"role": "user",
         "content": "A train at 90 km/h departs 08:30, arrives 11:00. Distance in km? Show work then state final number."},
    ], max_tokens=200)
    t_ = txt(r); tp = ntps(r, el)
    record("3.2 Multi-step arithmetic (225 km)",
           "225" in t_,
           el, tp, f"225_found={'225' in t_} snippet={t_[:80].strip()}")
except Exception as e:
    record("3.2 Multi-step arithmetic", False, 0, 0, str(e))

# 3.3 code trace
try:
    r, el = chat([
        {"role": "user",
         "content": "What does g(4) return?\ndef g(n):\n    if n<=1: return n\n    return g(n-1)+g(n-2)\nReply with just the number."},
    ], max_tokens=20)
    t_ = txt(r).strip(); tp = ntps(r, el)
    record("3.3 Code trace g(4)=3",
           "3" in t_[:10],
           el, tp, f"ans={t_[:20]}")
except Exception as e:
    record("3.3 Code trace", False, 0, 0, str(e))

# 3.4 counterfactual
try:
    r, el = chat([
        {"role": "user",
         "content": "5 red balls + 3 blue balls. Remove 2 red balls. How many remain? Just the number."},
    ], max_tokens=20)
    t_ = txt(r).strip(); tp = ntps(r, el)
    record("3.4 Counterfactual (6 remain)",
           "6" in t_[:10],
           el, tp, f"ans={t_[:20]}")
except Exception as e:
    record("3.4 Counterfactual", False, 0, 0, str(e))

# 3.5 code correctness
try:
    r, el = chat([
        {"role": "user",
         "content": "Is this correct?  def double(x): return x*3\nDoes it correctly double a number? Say yes or no with a brief reason."},
    ], max_tokens=80)
    t_ = txt(r); tp = ntps(r, el)
    record("3.5 Code correctness (expects NO)",
           "no" in t_.lower()[:20] or "incorrect" in t_.lower() or "wrong" in t_.lower(),
           el, tp, t_[:80].strip())
except Exception as e:
    record("3.5 Code check", False, 0, 0, str(e))


# ══════════════════════════════════════════════════════════════════
#  SUITE 4 — AGENTIC PLANNING
# ══════════════════════════════════════════════════════════════════

print(f"\n{C}{'='*64}{NC}")
print(f"{W}  SUITE 4 — Agentic Planning{NC}")
print(f"{C}{'='*64}{NC}")

# 4.1 task decomposition
try:
    r, el = chat([
        {"role": "system", "content": "You are an autonomous agent. Break tasks into numbered steps."},
        {"role": "user",
         "content": "Plan building a REST API with Python + FastAPI. Give exactly 5 sequential numbered steps."},
    ], max_tokens=300)
    t_ = txt(r); tp = ntps(r, el)
    steps = re.findall(r"(?m)^\s*\d+[.)]\s+\S", t_)
    record("4.1 Task decomposition (>=4 steps)",
           len(steps) >= 4,
           el, tp, f"steps={len(steps)}")
except Exception as e:
    record("4.1 Task decomposition", False, 0, 0, str(e))

# 4.2 first-step planning
try:
    r, el = chat([
        {"role": "system", "content": "You are an agent."},
        {"role": "user",
         "content": "I want to rename all .txt files in /data/ to .bak. What is the FIRST action you take?"},
    ], max_tokens=150)
    t_ = txt(r); tp = ntps(r, el)
    kws = ["list", "ls", "find", "glob", "os.listdir", "scandir", "directory"]
    record("4.2 First-step planning (list files)",
           any(k in t_.lower() for k in kws),
           el, tp, t_[:80].strip())
except Exception as e:
    record("4.2 First-step planning", False, 0, 0, str(e))

# 4.3 pipeline state
try:
    r, el = chat([
        {"role": "user",
         "content": ("Pipeline: step1 fetch=done, step2 clean=done, step3 train=running, step4 eval=not started. "
                     "Is it complete? What is blocking?")},
    ], max_tokens=80)
    t_ = txt(r); tp = ntps(r, el)
    record("4.3 Pipeline state recognition",
           ("no" in t_.lower()[:30]) and ("train" in t_.lower() or "step 3" in t_.lower()),
           el, tp, t_[:100].strip())
except Exception as e:
    record("4.3 Pipeline state", False, 0, 0, str(e))

# 4.4 self-correction
try:
    msgs = [
        {"role": "user", "content": "Write a Python function that returns True if a number is even."},
        {"role": "assistant", "content": "def is_even(n): return n % 2 == 1"},
        {"role": "user", "content": "That is wrong — it returns True for odd numbers. Fix it."},
    ]
    r, el = chat(msgs, max_tokens=100)
    t_ = txt(r); tp = ntps(r, el)
    record("4.4 Self-correction (fix even check)",
           "== 0" in t_ or "% 2 == 0" in t_,
           el, tp, t_[:80].strip())
except Exception as e:
    record("4.4 Self-correction", False, 0, 0, str(e))


# ══════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════

n_passed = sum(1 for x in results if x["passed"])
n_total  = len(results)
all_tps  = [x["tps"] for x in results if x["tps"] > 0]
avg_q    = round(sum(all_tps) / len(all_tps), 1) if all_tps else 0.0

print(f"\n{W}{'='*64}{NC}")
print(f"{W}  SUMMARY: {args.label}{NC}")
print(f"{W}{'='*64}{NC}")
print(f"  Tests passed : {G if n_passed == n_total else ''}{n_passed}/{n_total}{NC}")
print(f"  Avg tok/s    : {C}{avg_q}{NC}")
failed = [x["name"] for x in results if not x["passed"]]
if failed:
    print(f"  {R}Failed:{NC}")
    for f in failed:
        print(f"    • {f}")

# ── Write JSON ────────────────────────────────────────────────────
out_data = {
    "label":      args.label,
    "passed":     n_passed,
    "total":      n_total,
    "pass_rate":  f"{n_passed}/{n_total}",
    "avg_tps":    avg_q,
    "perf":       perf_rows,
    "tests":      results,
}

if args.out:
    import pathlib
    pathlib.Path(args.out).write_text(json.dumps(out_data, indent=2))
    print(f"\n  Results → {args.out}")
else:
    print("\n" + json.dumps(out_data, indent=2))
