#!/usr/bin/env python3
"""Battle-test suite for the performance optimizations landed in the last ~4 days.

Each case A/Bs an optimization against its pre-optimization baseline, checking
BOTH correctness (byte-identical output where the contract is lossless) AND the
performance delta, then issues a PASS/FAIL/WARN verdict. Results are written to
``benchmarks/results/<stamp>_battle/battle_test.json``.

Coverage (git-confirmed, since 2026-06-17):
  #155 prompt-lookup default + adaptive guard + async cooldown pipelining + prefix-KV reuse
  #153 per-request KV cache for batched decode (O(n²)→O(n))
  #152 fp16 npy-dir loader passthrough (no fp32 upcast)
  #154 PyramidKV per-layer SnapKV budget (opt-in)
  #59  single-token detokenization memoization (hot path)

Run:
  ~/squish/.venv/bin/python benchmarks/perf/battle_test.py [--quick] [--model DIR]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from battle_harness import (  # noqa: E402
    Case, Stat, hardware_info, measure, print_table, verdict, write_results,
)

_MODEL = os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct-int4")
_RESULTS = Path(__file__).resolve().parents[1] / "results"

_PROMPTS = {
    "copy": ("CONTEXT: The Falcon-9 first stage landed on the droneship at 08:42 UTC. "
             "The payload was a 4.2-tonne communications satellite. " * 3
             + "\nTASK: Quote the droneship sentence, then the payload mass.\nANSWER:"),
    "chat": "What are three benefits of regular exercise? Answer concisely.\n",
    "repeat": "Repeat exactly: red green blue red green blue red green blue red",
}


def _greedy_ref(model, ids, n, eos):
    """Plain single-token greedy — the pre-speculation baseline."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    c = make_prompt_cache(model)
    if len(ids) > 1:
        model(mx.array(ids[:-1], mx.uint32)[None], cache=c)
        mx.eval([x.state for x in c])
    cur, out = ids[-1], []
    for _ in range(n):
        lg = model(mx.array([[cur]], mx.uint32), cache=c)
        mx.eval(lg)
        cur = int(mx.argmax(lg[0, -1]).item())
        out.append(cur)
        if cur in eos:
            break
    return out


def _tps(stat: Stat) -> float:
    return 1.0 / stat.p50  # samples are per-token seconds → tokens/sec at p50


# ── Cases ─────────────────────────────────────────────────────────────────────

def case_prompt_lookup(model, tok, eos, reps) -> Case:
    """#155 — n-gram speculative decode vs plain greedy on copy-heavy text."""
    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate
    ids = tok.encode(_PROMPTS["copy"])
    n = 96
    base = measure(lambda: _greedy_ref(model, ids, n, eos),
                   warmup=2, reps=reps, per_call_div=len)
    opt = measure(lambda: [t for t, _ in prompt_lookup_generate(
        model, ids, n, eos_ids=eos, reuse_prefix=False)], warmup=2, reps=reps, per_call_div=len)
    # lossless check on the near-tie-free repetitive prompt
    rid = tok.encode(_PROMPTS["repeat"])
    ref = _greedy_ref(model, rid, 32, eos)
    got = [t for t, _ in prompt_lookup_generate(model, rid, 32, eos_ids=eos, reuse_prefix=False)]
    sp = _tps(opt) / _tps(base)
    return Case("prompt-lookup speculative", "#155", "decode tok/s", _tps(base), _tps(opt),
                sp, got == ref, verdict(lossless=got == ref, speedup=sp, min_speedup=1.1),
                detail=f"copy-heavy decode; greedy-identity on repeat prompt={got == ref}")


def case_adaptive_guard(model, tok, eos, reps) -> Case:
    """#155 — guard must not regress open-chat vs plain greedy (>5% p95 = fail)."""
    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate
    ids = tok.encode(_PROMPTS["chat"])
    n = 96
    base = measure(lambda: _greedy_ref(model, ids, n, eos), warmup=2, reps=reps, per_call_div=len)
    opt = measure(lambda: [t for t, _ in prompt_lookup_generate(
        model, ids, n, eos_ids=eos, reuse_prefix=False)], warmup=2, reps=reps, per_call_div=len)
    # p95 regression gate: optimized p95 per-token latency must not exceed greedy by >5%
    p95_ok = opt.p95 <= base.p95 * 1.05
    sp = _tps(opt) / _tps(base)
    v = "PASS" if p95_ok else verdict(lossless=None, speedup=sp, min_speedup=1.0, regress_guard=True)
    return Case("adaptive accept guard", "#155", "chat tok/s", _tps(base), _tps(opt), sp,
                None, v, detail=f"p95 base={base.p95*1e3:.1f}ms opt={opt.p95*1e3:.1f}ms "
                f"(≤+5% required) → {'ok' if p95_ok else 'REGRESS'}")


def case_async_pipelining(model, tok, eos, reps) -> Case:
    """#155 — async_eval cooldown pipelining vs sync cooldown (lossless, no regression)."""
    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate
    ids = tok.encode(_PROMPTS["chat"])
    n = 96

    def run(pipe):
        return [t for t, _ in prompt_lookup_generate(
            model, ids, n, eos_ids=eos, reuse_prefix=False, pipeline_cooldown=pipe)]

    base = measure(lambda: run(False), warmup=2, reps=reps, per_call_div=len)
    opt = measure(lambda: run(True), warmup=2, reps=reps, per_call_div=len)
    lossless = run(True) == run(False)
    sp = _tps(opt) / _tps(base)
    return Case("async cooldown pipelining", "#155", "chat tok/s", _tps(base), _tps(opt), sp,
                lossless, verdict(lossless=lossless, speedup=sp, min_speedup=1.0, regress_guard=True),
                detail="sync vs mx.async_eval cooldown block")


def case_prefix_reuse(model, tok, eos, reps) -> Case:
    """#155 — in-memory prompt-prefix KV reuse: cold vs warm prefill (TTFT).

    Warm measures only the trim + new-suffix prefill (the prefix KV was paid for
    by the *prior* request), so the prime is built OUTSIDE the timed region.
    """
    import mlx.core as mx
    from squish.kv.prompt_prefix_cache import PromptPrefixCache, prefill_with_reuse
    from squish.speculative.prompt_lookup_batched import prompt_lookup_generate
    p1 = tok.encode("Background: " + "the river flows past the old stone bridge. " * 16)
    g1 = [t for t, _ in prompt_lookup_generate(model, p1, 16, eos_ids=eos, reuse_prefix=False)]
    p2 = p1 + g1 + tok.encode(" Then answer briefly:")
    prefix_ids = list(p1) + g1
    n = max(4, reps)

    def _timed(fn):
        for _ in range(2):
            fn()
        s = []
        for _ in range(n):
            t0 = time.perf_counter(); c = fn(); mx.eval([x.state for x in c])
            s.append(time.perf_counter() - t0)
        return Stat.of(s)

    cold = _timed(lambda: prefill_with_reuse(model, p2, None))

    def warm_once():
        pc = PromptPrefixCache(min_prefix=128)
        prime = prefill_with_reuse(model, prefix_ids + [0], None)  # built untimed below
        mx.eval([x.state for x in prime])
        pc.store(prefix_ids, prime)
        t0 = time.perf_counter()
        c = prefill_with_reuse(model, p2, pc)
        mx.eval([x.state for x in c])
        return time.perf_counter() - t0

    for _ in range(2):
        warm_once()
    warm = Stat.of([warm_once() for _ in range(n)])
    # lossless: warm-reuse vs cold full-generation output identical
    wf = [t for t, _ in prompt_lookup_generate(model, p2, 16, eos_ids=eos, reuse_prefix=True)]
    cf = [t for t, _ in prompt_lookup_generate(model, p2, 16, eos_ids=eos, reuse_prefix=False)]
    sp = cold.p50 / warm.p50  # prefill ms: lower is better → cold/warm
    return Case("prompt-prefix KV reuse", "#155", "prefill ms", cold.p50 * 1e3, warm.p50 * 1e3,
                sp, wf == cf, verdict(lossless=wf == cf, speedup=sp, min_speedup=2.0),
                detail=f"{len(p2)}-tok extending prompt; warm=suffix-only prefill")


def case_detok_memo(tok, reps) -> Case:
    """#59 — single-token detokenization memoization on the hot path."""
    from squish.serving.token_decode_cache import TokenDecodeCache
    cache = TokenDecodeCache(tok)
    stream = [tok.encode("the quick brown fox jumps over the lazy dog. ")[i % 9]
              for i in range(4000)]  # realistic repetitive id stream

    def memoized():
        return [cache.decode(t) for t in stream]

    def raw():
        return [tok.decode([t]) for t in stream]

    cache.clear()
    base = measure(raw, warmup=1, reps=max(3, reps // 2), per_call_div=len)
    cache.clear()
    opt = measure(memoized, warmup=1, reps=max(3, reps // 2), per_call_div=len)
    lossless = memoized() == raw()
    sp = base.p50 / opt.p50
    return Case("detok memoization", "#59", "decodes/sec", 1 / base.p50, 1 / opt.p50, sp,
                lossless, verdict(lossless=lossless, speedup=sp, min_speedup=1.3),
                detail="TokenDecodeCache vs raw tokenizer.decode on repeated ids")


def case_fp16_passthrough(tmp: Path) -> Case:
    """#152 — npy-dir passthrough returns float16 (half the bytes, bf16 bit-identical)."""
    import mlx.core as mx
    from squish.quant.compressed_loader import _dequantize_npy_dir
    tdir = tmp / "tensors"; tdir.mkdir(parents=True, exist_ok=True)
    sk = "w0"
    orig = (np.random.default_rng(0).standard_normal((128, 64)) * 0.1).astype(np.float16)
    np.save(tdir / f"{sk}__pt.npy", orig)
    np.save(tdir / f"{sk}__shape.npy", np.array(orig.shape))
    out = _dequantize_npy_dir(tdir, sk)
    is_f16 = np.dtype(out.dtype) == np.float16
    # f16→bf16 must equal f16→f32→bf16 (the property that makes returning f16 safe)
    bf_direct = np.array(mx.array(out).astype(mx.bfloat16).astype(mx.float32))
    bf_via32 = np.array(mx.array(out.astype(np.float32)).astype(mx.bfloat16).astype(mx.float32))
    identical = bool(np.array_equal(bf_direct, bf_via32))
    ok = is_f16 and identical
    return Case("fp16 npy-dir passthrough", "#152", "bytes/elem", 4.0, 2.0 if is_f16 else 4.0,
                2.0 if is_f16 else 1.0, identical, "PASS" if ok else "FAIL",
                detail=f"dtype={out.dtype} (want float16); bf16 bit-identical={identical}")


def case_pyramidkv(_) -> Case:
    """#154 — PyramidKV per-layer budgets: mean preserved, lower>upper, floored."""
    from squish.kv.kv_cache import _compute_layer_budgets
    n, budget, floor = 28, 2048, 32
    uni = _compute_layer_budgets(n, budget, "uniform", 0.5, floor)
    pyr = _compute_layer_budgets(n, budget, "pyramid", 0.5, floor)
    mean_ok = abs(sum(pyr) / n - budget) <= budget * 0.05
    taper_ok = pyr[0] > pyr[-1] and all(pyr[i] >= pyr[i + 1] - 1 for i in range(n - 1))
    floor_ok = min(pyr) >= floor and uni == [budget] * n
    ok = mean_ok and taper_ok and floor_ok
    spread = pyr[0] / max(1, pyr[-1])
    return Case("PyramidKV layer budget", "#154", "L0/Ln ratio", 1.0, spread, spread, ok,
                "PASS" if ok else "FAIL",
                detail=f"mean≈budget={mean_ok} taper={taper_ok} floor={floor_ok}; "
                f"L0={pyr[0]} Ln={pyr[-1]}")


def case_batched_decode(model, tok, reps) -> Case:
    """#153 — per-request KV-cached batched decode: throughput SCALING + identity.

    #153 fixed the batched path from an O(n²) padded re-forward (no KV cache) to
    per-request KV caching that is bit-identical to single-stream. Its win shows
    as throughput scaling under concurrency, so we measure aggregate tok/s for a
    coalesced batch vs sequential single-stream. Output identity is reported but
    not gated here (GPU near-ties); the bit-identity contract is gated
    deterministically on CPU by tests/serving/test_scheduler_kv_cache.py.
    """
    import threading

    from squish.serving.scheduler import BatchScheduler
    prompts = [_PROMPTS["chat"], _PROMPTS["copy"][:200], "List three colors:\n",
               "Name two planets:\n", "Define gravity:\n", "Count to five:\n",
               "Spell cat:\n", "Say hello:\n"]
    n = 64
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    t0 = time.perf_counter()
    seq_ids = [_greedy_ref(model, tok.encode(p), n, eos) for p in prompts]
    seq_tps = sum(len(o) for o in seq_ids) / (time.perf_counter() - t0)
    seq_text = [tok.decode([t for t in o if t not in eos]) for o in seq_ids]

    sched = BatchScheduler(model, tok, max_batch_size=8, batch_window_ms=80).start()
    try:
        res: dict[int, str] = {}

        def _run(i, p):
            res[i] = "".join(tt for tt, _ in sched.submit_sync(p, max_tokens=n, temperature=0.0))

        t0 = time.perf_counter()
        threads = [threading.Thread(target=_run, args=(i, p)) for i, p in enumerate(prompts)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        batch_text = [res[i] for i in range(len(prompts))]
        batch_toks = sum(len(tok.encode(t)) for t in batch_text)
        batch_tps = batch_toks / (time.perf_counter() - t0)
    finally:
        sched.stop()
    match = sum(b.startswith(s[:40]) or s.startswith(b[:40])
                for b, s in zip(batch_text, seq_text))
    sp = batch_tps / seq_tps
    return Case("batched decode (scheduler)", "#153", "aggregate tok/s", seq_tps, batch_tps,
                sp, None, verdict(lossless=None, speedup=sp, min_speedup=1.0, regress_guard=True),
                detail=f"batch=8 vs sequential; matches single-stream {match}/{len(prompts)} "
                f"(GPU near-ties; exact identity gated on CPU)")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_suite(quick: bool, model_path: str) -> "list[Case]":
    import tempfile
    from mlx_lm import load
    reps = 3 if quick else 7
    model, tok = load(model_path)
    eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
    cases: list[Case] = []
    runners = [
        ("prompt-lookup", lambda: case_prompt_lookup(model, tok, eos, reps)),
        ("guard", lambda: case_adaptive_guard(model, tok, eos, reps)),
        ("async-pipe", lambda: case_async_pipelining(model, tok, eos, reps)),
        ("prefix-reuse", lambda: case_prefix_reuse(model, tok, eos, reps)),
        ("detok-memo", lambda: case_detok_memo(tok, reps)),
        ("pyramidkv", lambda: case_pyramidkv(None)),
        ("batched", lambda: case_batched_decode(model, tok, reps)),
    ]
    with tempfile.TemporaryDirectory() as td:
        runners.insert(5, ("fp16-passthrough", lambda td=td: case_fp16_passthrough(Path(td))))
        for label, fn in runners:
            try:
                print(f"  running: {label} …", flush=True)
                cases.append(fn())
            except (RuntimeError, ValueError, OSError, ImportError, AttributeError,
                    TypeError, KeyError, IndexError, AssertionError) as exc:
                # A broken case is surfaced as a FAIL row, not silently swallowed.
                cases.append(Case(label, "?", "—", 0, 0, 0, None, "FAIL", detail=f"errored: {exc!r}"))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="Perf battle-test for 4-day optimizations.")
    ap.add_argument("--quick", action="store_true", help="reduced reps (smoke test).")
    ap.add_argument("--model", default=_MODEL)
    args = ap.parse_args()
    if not os.path.isdir(args.model):
        print(f"model not found: {args.model}")
        return 2
    hw = hardware_info()
    print(f"Battle-test  model={os.path.basename(args.model)}  quick={args.quick}")
    print(f"hardware: {hw}")
    cases = run_suite(args.quick, args.model)
    print_table(cases)
    # deterministic timestamp from wall clock is fine post-run (not inside any cached path)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    tag = "battle_quick" if args.quick else "battle"
    path = write_results(cases, hw, tag, _RESULTS, stamp)
    failed = sum(c.verdict == "FAIL" for c in cases)
    print(f"\nresults: {path}")
    print(f"PASS={sum(c.verdict=='PASS' for c in cases)}  "
          f"WARN={sum(c.verdict=='WARN' for c in cases)}  FAIL={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
