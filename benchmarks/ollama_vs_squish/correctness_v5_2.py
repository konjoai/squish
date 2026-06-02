#!/usr/bin/env python3
"""v5.2 spec-decode correctness gate.

Loads the 7B-int4 target and the 1.5B-int4 draft once, then decodes the same
prompt two ways at temp=0 / seed=42:

  * reference  — SpeculativeGenerator with no draft (ngram_max_n=0) → plain
                 greedy autoregressive decode (the non-spec path).
  * spec       — SpeculativeGenerator with the 1.5B draft, K=4 → draft/verify
                 with rejection sampling.

The two token streams MUST be byte-identical. Also reports the measured
acceptance rate of the spec run. Read-only; no server involved.

Usage:
    python correctness_v5_2.py [max_tokens]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

MODELS = Path.home() / "models"
TARGET = MODELS / "Qwen2.5-7B-Instruct-int4"
DRAFT  = MODELS / "Qwen2.5-1.5B-Instruct-int4"

PROMPT = "Write a detailed technical explanation of how a CPU cache works, including L1, L2, and L3 levels."


def _chat(tok, user: str) -> str:
    msgs = [{"role": "user", "content": user}]
    if hasattr(tok, "apply_chat_template"):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    return user


def main() -> None:
    max_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    from mlx_lm import load
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from squish.speculative import SpeculativeGenerator

    print(f"loading target  {TARGET.name} …", flush=True)
    target, ttok = load(str(TARGET))
    print(f"loading draft   {DRAFT.name} …", flush=True)
    draft, dtok = load(str(DRAFT))

    # Scope guard: fail loud on tokenizer mismatch.
    tv = getattr(ttok, "vocab_size", None) or len(ttok.encode("hello"))
    dv = getattr(dtok, "vocab_size", None) or len(dtok.encode("hello"))
    teos = getattr(ttok, "eos_token_id", None)
    deos = getattr(dtok, "eos_token_id", None)
    print(f"target vocab={tv} eos={teos}   draft vocab={dv} eos={deos}")
    if teos != deos:
        raise SystemExit(f"TOKENIZER MISMATCH: target eos {teos} != draft eos {deos} — aborting")

    prompt = _chat(ttok, PROMPT)

    print("\n── reference (mlx_lm canonical greedy, temp=0) ──", flush=True)
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    t0 = time.time()
    ref_toks: list[str] = []
    for resp in stream_generate(target, ttok, prompt, max_tokens=max_tokens,
                                sampler=sampler):
        ref_toks.append(resp.text)
    ref_dt = time.time() - t0
    ref_out = "".join(ref_toks)
    print(f"  {len(ref_toks)} tokens in {ref_dt:.2f}s  ({len(ref_toks)/ref_dt:.2f} tok/s)")

    print("\n── spec decode (1.5B draft, K=4) ──", flush=True)
    spec_gen = SpeculativeGenerator(target, ttok, draft_model=draft,
                                    draft_tokenizer=dtok, k=4)
    t0 = time.time()
    spec_toks: list[str] = []
    for txt, fin in spec_gen.stream(prompt, max_tokens=max_tokens,
                                    temperature=0.0, top_p=1.0, seed=42):
        spec_toks.append(txt)
        if fin is not None:
            break
    spec_dt = time.time() - t0
    spec_out = "".join(spec_toks)
    acc = spec_gen.acceptance_rate
    print(f"  {len(spec_toks)} tokens in {spec_dt:.2f}s  ({len(spec_toks)/spec_dt:.2f} tok/s)")
    print(f"  acceptance rate = {acc:.3f}  "
          f"(accepted {spec_gen.accepted_total} / proposed {spec_gen.proposed_total}, "
          f"{spec_gen.steps} verify cycles)")

    print("\n── correctness ──")
    identical = ref_out == spec_out
    print(f"  reference chars = {len(ref_out)}   spec chars = {len(spec_out)}")
    print(f"  IDENTICAL: {identical}")
    if not identical:
        # show first divergence
        n = min(len(ref_out), len(spec_out))
        i = next((j for j in range(n) if ref_out[j] != spec_out[j]), n)
        print(f"  first divergence at char {i}:")
        print(f"    ref : …{ref_out[max(0,i-40):i+40]!r}")
        print(f"    spec: …{spec_out[max(0,i-40):i+40]!r}")

    speedup = (len(spec_toks)/spec_dt) / (len(ref_toks)/ref_dt) if ref_dt and spec_dt else 0.0
    print(f"\n  decode speedup (spec/ref tok-rate) = {speedup:.2f}×")
    print("\nRESULT:", "PASS" if identical else "FAIL — outputs differ")
    sys.exit(0 if identical else 1)


if __name__ == "__main__":
    main()
