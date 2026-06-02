#!/usr/bin/env python3
"""v5.2 Phase 4 — draft-pairing + K sweep, acceptance, net tok/s, correctness.

Only Qwen2.5-1.5B-Instruct-int4 shares the 7B tokenizer family in ~/models/
(no 0.5B, no 3B), so the brief's 3-row matrix collapses to one draft. We sweep
K and the n-gram pre-fill instead, and test whether K=1 (verify batch shape
[1,1] == sequential decode) yields bit-identical greedy output — isolating the
batched-verify numerical effect on int4.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path

MODELS = Path.home() / "models"
TARGET = MODELS / "Qwen2.5-7B-Instruct-int4"
DRAFT  = MODELS / "Qwen2.5-1.5B-Instruct-int4"
PROMPT = "Write a detailed technical explanation of how a CPU cache works, including L1, L2, and L3 levels."
N = 200


def chat(tok, u):
    return tok.apply_chat_template([{"role": "user", "content": u}],
                                   tokenize=False, add_generation_prompt=True)


def main():
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from squish.speculative import SpeculativeGenerator

    target, ttok = load(str(TARGET))
    draft, dtok = load(str(DRAFT))
    prompt = chat(ttok, PROMPT)

    t0 = time.time()
    gt = "".join(r.text for r in stream_generate(
        target, ttok, prompt, max_tokens=N, sampler=make_sampler(temp=0.0)))
    gt_dt = time.time() - t0
    gt_rate = N / gt_dt
    print(f"baseline mlx greedy: {gt_rate:.2f} tok/s ({gt_dt:.2f}s)\n")
    print(f"{'config':<22}{'acc':>7}{'tok/s':>9}{'net×':>8}{'identical':>11}")

    rows = []
    for ng, k in [(0, 1), (0, 2), (0, 4), (0, 6), (8, 4)]:
        spec = SpeculativeGenerator(target, ttok, draft_model=draft,
                                    draft_tokenizer=dtok, k=k, ngram_max_n=ng)
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
        label = f"ngram={ng} K={k}"
        print(f"{label:<22}{spec.acceptance_rate:>7.3f}{rate:>9.2f}"
              f"{rate/gt_rate:>8.2f}{str(ident):>11}")
        rows.append((label, spec.acceptance_rate, rate, rate / gt_rate, ident))

    best = max(rows, key=lambda r: r[2])
    print(f"\nbest net tok/s: {best[0]}  ({best[2]:.2f} tok/s, "
          f"{best[3]:.2f}×, acc={best[1]:.3f}, identical={best[4]})")


if __name__ == "__main__":
    main()
