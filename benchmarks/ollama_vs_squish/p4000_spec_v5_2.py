#!/usr/bin/env python3
"""v5.2 long-context (p4000) spec economics check.

Phase-4 short-context data showed net tok/s < 1.0× for the 1.5B draft. The open
question is whether a long, attention-bound prompt (~4000 tokens) shifts the
cost structure enough for spec to win net throughput. This builds a ~4000-token
prompt and times mlx_lm canonical greedy vs SpeculativeGenerator K=2/K=4 at
temp=0, reporting acceptance, net tok/s, net×, and byte-identity.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

MODELS = Path.home() / "models"
TARGET = MODELS / "Qwen2.5-7B-Instruct-int4"
DRAFT  = MODELS / "Qwen2.5-1.5B-Instruct-int4"
N = 200
CTX_TOKENS = 4000

_SEED = (
    "You are a senior engineer reviewing a large pull request that adds a "
    "Redis-backed session cache to the auth service."
)
_CHUNK = (
    "The codebase conventions require every public function to have a docstring, "
    "every test to be deterministic, and every migration to be reversible. CI "
    "runs the full suite, a linter, a type-checker, and a mutation pass. Pay "
    "attention to thread safety, TTL refresh under contention, fallback latency "
    "when Redis is slow, and token rotation if a key is evicted mid-request. "
)


def chat(tok, u):
    return tok.apply_chat_template([{"role": "user", "content": u}],
                                   tokenize=False, add_generation_prompt=True)


def build_ctx(tok):
    base = _SEED + " "
    while len(tok.encode(base)) < CTX_TOKENS:
        base += _CHUNK
    return base + " Summarize the single most important risk in two sentences."


def main():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from squish.speculative import SpeculativeGenerator

    target, ttok = load(str(TARGET))
    draft, dtok = load(str(DRAFT))
    raw = build_ctx(ttok)
    ntok = len(ttok.encode(raw))
    prompt = chat(ttok, raw)
    print(f"context tokens ≈ {ntok}\n")

    t0 = time.time()
    gt = "".join(r.text for r in stream_generate(
        target, ttok, prompt, max_tokens=N, sampler=make_sampler(temp=0.0)))
    gt_dt = time.time() - t0
    gt_rate = N / gt_dt
    print(f"baseline mlx greedy: {gt_rate:.2f} tok/s ({gt_dt:.2f}s)\n")
    print(f"{'config':<14}{'acc':>7}{'tok/s':>9}{'net×':>8}{'identical':>11}")

    for k in (2, 4):
        spec = SpeculativeGenerator(target, ttok, draft_model=draft,
                                    draft_tokenizer=dtok, k=k, ngram_max_n=0)
        t0 = time.time()
        toks = []
        for txt, fin in spec.stream(prompt, max_tokens=N, temperature=0.0,
                                    top_p=1.0, seed=42):
            toks.append(txt)
            if fin:
                break
        dt = time.time() - t0
        out = "".join(toks)
        rate = len(toks) / dt
        ident = (out == gt) or gt.startswith(out) or out.startswith(gt)
        print(f"{'K=' + str(k):<14}{spec.acceptance_rate:>7.3f}{rate:>9.2f}"
              f"{rate / gt_rate:>8.2f}{str(ident):>11}")


if __name__ == "__main__":
    main()
