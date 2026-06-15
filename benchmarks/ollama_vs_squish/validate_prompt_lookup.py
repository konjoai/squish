#!/usr/bin/env python3
"""Greedy-identity + speedup validation for batched prompt-lookup decoding.

Proves the new decoder is token-for-token identical to plain greedy (the
correctness bar the prior temp-0 speculative attempt failed) and measures how
many forward passes it saves on repetitive output.

Runs on CPU by default (set MLX_GPU=1 to use the GPU) so it doesn't contend
with other GPU work.
"""
import os
import sys
import time

import mlx.core as mx

if not os.environ.get("MLX_GPU"):
    mx.set_default_device(mx.cpu)

from mlx_lm import load  # noqa: E402
from mlx_lm.models import cache as _cache  # noqa: E402

sys.path.insert(0, "/Users/wscholl/squish")
from squish.speculative.prompt_lookup_batched import prompt_lookup_generate  # noqa: E402

MODEL = os.environ.get("PL_MODEL", "/Users/wscholl/models/Qwen2.5-1.5B-Instruct-int4")
MAX_TOKENS = 120


def greedy_reference(model, prompt_ids, max_tokens, eos_ids):
    """Plain one-token-at-a-time greedy — the ground truth."""
    c = _cache.make_prompt_cache(model)
    if len(prompt_ids) > 1:
        model(mx.array(prompt_ids[:-1], mx.uint32)[None], cache=c)
    cur = prompt_ids[-1]
    out, forwards = [], 0
    for _ in range(max_tokens):
        logits = model(mx.array([cur], mx.uint32)[None], cache=c)
        forwards += 1
        t = int(mx.argmax(logits[0, -1]).item())
        out.append(t)
        cur = t
        if t in eos_ids:
            break
    return out, forwards


def main():
    print(f"device: {'gpu' if os.environ.get('MLX_GPU') else 'cpu'}  model: {MODEL}")
    model, tok = load(MODEL)
    eos_ids = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])

    # A prompt that induces repetitive output (so n-gram drafts get accepted).
    prompt = (
        "Repeat this list three times exactly:\n"
        "- apple\n- banana\n- cherry\n- date\n- elderberry\n\nList:"
    )
    pids = tok.encode(prompt)
    print(f"prompt tokens: {len(pids)}")

    t0 = time.perf_counter()
    ref, ref_fwd = greedy_reference(model, pids, MAX_TOKENS, eos_ids)
    t_ref = time.perf_counter() - t0

    print(f"\nreference greedy : {len(ref)} tokens, {ref_fwd} forwards, {t_ref:.2f}s")

    all_ok = True
    for nd in (0, 1, 4):
        t0 = time.perf_counter()
        pl, steps = [], 0
        for t, _n in prompt_lookup_generate(
            model, pids, MAX_TOKENS, ngram_min=2, ngram_max=3, num_draft=nd, eos_ids=eos_ids
        ):
            pl.append(t)
        t_pl = time.perf_counter() - t0
        identical = ref == pl
        all_ok = all_ok and identical
        print(f"  num_draft={nd}: {'✅ identical' if identical else '❌ DIVERGES'}  "
              f"({len(pl)} tok, {t_pl:.2f}s, {ref_fwd/max(t_ref,1e-9)*t_pl:.0f}~speedup-na)")
        if not identical:
            for i, (a, b) in enumerate(zip(ref, pl)):
                if a != b:
                    print(f"     first diverge @{i}: ref={a} pl={b}  "
                          f"ctx={tok.decode(ref[max(0,i-2):i])!r}")
                    break
    print(f"\nVERDICT: {'✅ GREEDY-IDENTICAL across all num_draft' if all_ok else '❌ BUG REMAINS'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
