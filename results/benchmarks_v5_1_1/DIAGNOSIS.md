# v5.1.1 — Diagnosis: why `squish_daemon` shows 38 s TTFT at 4000 tokens

**Host:** Apple M3 MacBook Pro, 16 GB unified memory, macOS 25.5.0
**Date:** 2026-06-02
**Baseline under investigation:** `results/benchmarks_v5_1/runs/20260602T082624/raw.json`

## Hypothesis

`squish_daemon`'s 38.37 s cold TTFT at the 4000-token prompt is an artifact
of running the daemon **with no KV-cache flags** — every send is a full cold
prefill of the entire prompt — not a performance regression in the inference
path.

## Result: **CONFIRMED.**

### 1. How `bench_v5_1.py` constructs each squish config

All squish configs are launched through the same helper:

```python
def _squish_cmd(model_path, extra):
    return [SQUISH_PY, "-m", "squish.server",
            "--mlx-model-dir", model_path,
            "--port", ..., "--host", ..., "--log-level", "warning", *extra]
```

The per-config `extra` flag list is the only difference:

| Config              | `extra` flags passed                                        |
|---------------------|-------------------------------------------------------------|
| `squish_daemon`     | `[]` — **none**                                             |
| `squish_pkv`        | `--prompt-kv-cache <DIR>`                                   |
| `squish_block`      | `--block-kv-cache <DIR> --block-kv-size 64`                |
| `squish_block_int3` | `--block-kv-cache <DIR> --block-kv-size 64` (+ INT3 model) |

Source: `start_squish_daemon` passes `_squish_cmd(SQUISH_MODEL_INT4, [])`;
`start_squish_block` passes `["--block-kv-cache", BLOCK_CACHE_DIR,
"--block-kv-size", "64"]`; `start_squish_pkv` passes `["--prompt-kv-cache",
PKV_CACHE_DIR]`.

### 2. `squish_daemon` passes no caching flag — verified

`start_squish_daemon` forwards an **empty** `extra` list. No
`--prompt-kv-cache`, no `--block-kv-cache`, no `--disk-prompt-cache`. With no
cache enabled, the server has nothing to look up: every request walks the full
prefill path over all 4053 prompt tokens.

The bench sends the **same** prompt 5× per phase and reports the median TTFT.
On a cached config, runs 2–5 are warm hits and the median reflects a warm hit.
On `squish_daemon` there is no cache, so **all five runs are cold** — the
median of 5 cold prefills of a 4053-token prompt is a genuine ~38 s. This is
the architectural baseline (raw, uncached daemon), not a regression.

### 3. What `squish_block` does that `squish_daemon` doesn't

`squish_block` passes `--block-kv-cache`, which enables the v5 paged
block-prefix cache. On runs 2–5 the prompt's 64-token blocks are already
cached, so the server restores the matched-prefix KV state and prefills only
the (empty/short) suffix. p4000 block-cache TTFT is 1.11 s vs daemon's 38.4 s
— a 34× reduction — confirming the gap is entirely "cache vs no-cache," not a
slow code path.

### 4. Can `--block-kv-cache` and `--prompt-kv-cache` coexist?

**Yes.** They are independent flags at every layer:

- `squish/server.py` argparse: two separate `ap.add_argument` calls
  (`--prompt-kv-cache` at ~4271, `--block-kv-cache` at ~4286). No
  `add_mutually_exclusive_group`. The `--block-kv-cache` help text reads
  "Works alongside --prompt-kv-cache."
- `squish/cli.py:cmd_run` (~1429–1436): two independent `if` blocks forward
  each flag to the spawned server — passing both is supported.
- `squish/server.py` main() wiring (~4765, ~4787): `_prompt_kv_store` and
  `_block_kv_cache` are set independently; enabling one does not disable the
  other.

### 5. Precedence when both are enabled (important caveat for the combined row)

In the request path, the **block** lookup runs first (`server.py` ~2148), then
the **PKV** lookup (~2291). They are merged at ~2383:

```python
if _bkv_first_token_text is not None and _pkv_first_token_text is None:
    ...   # adopt the block-cache result
```

So the block result is adopted **only when PKV did not produce a first token.**
Consequences:

- **Exact-match repeat (the bench's same-prompt-5× pattern):** PKV hits on its
  whole-prompt hash and produces the first token via its cached-logit fast
  path → PKV takes precedence. The combined config's warm TTFT ≈ the PKV row
  (single-digit-ms hits). Block cache sits behind it as a generalization net.
- **Shifting-prefix repeat (PKV misses, block hits):** the PKV miss branch
  (~2343) runs a **full manual prefill** of the whole prompt to capture its
  logit, which sets `_pkv_first_token_text` and therefore *suppresses* the
  block-cache result at the merge. The block hit is effectively wasted on this
  path. This is an **efficiency** caveat (redundant prefill on cold/miss
  sends), not a correctness bug — output tokens are unchanged.

This means the "recommended" combined config is legitimately the best
single-config choice (it wins exact-match repeats via PKV **and** keeps block
cache available for prefix reuse across distinct prompts), but its headline
win on the bench's repeated-prompt measurement comes from the PKV path. The
combined row should be read as "PKV speed on exact repeats + block-cache
coverage on prefix reuse," and we document the redundant-prefill caveat
honestly rather than fixing core this session (scope guard).

## §6 Post-run correction — measured precedence (block runs first, not PKV)

§5 above predicted, from a static read, that PKV would take precedence on
exact-match repeats and the combined config would inherit PKV's single-digit-ms
TTFT. **The measured run + a trace check disproved this.** Trace
(`combined_path_trace.log`), same prompt sent 3× with both caches on:

```
run1: block-kv-cache MISS  suffix_prefilled=121      # cold, full prefill
      prompt-kv-cache MISS  prefilled 121 tokens      # cold, full prefill (redundant)
      both STORED
run2: block-kv-cache HIT  matched_tokens=64/121  suffix_prefilled=57   ← runs FIRST
      prompt-kv-cache HIT-fast  logit=cached → defer-restore
run3: (identical to run2)   server-side ttft ≈ 0.43 s
```

What actually happens in the request path (`server.py`): the **block** lookup
executes *before* the PKV lookup. On a warm run it matches only the whole
64-token blocks (the prompt's trailing partial block is never cached) and
**re-prefills the partial suffix** (`suffix_prefilled=57`) plus restores the
matched blocks' KV — both on the critical path. By the time the PKV `HIT-fast`
short-circuit is reached, that block work is already spent; the merge gate then
adopts the PKV token but cannot refund the block path's time.

**Consequence for the headline:** the combined `squish_recommended` config's
TTFT tracks **block-cache** (279 ms / 707 ms / 1.28 s / 1.87 s at
75/500/2000/4000), **not** PKV's single-digit ms (9 ms at p2000). The PKV
fast-hit benefit is masked by the block-first execution order. The combined
config is therefore a *generalist* (it populates both caches and wins
long-context e2e vs Ollama) but it does **not** beat PKV-only on pure
exact-match repeats — PKV-only remains the TTFT champion for that workload.

This is an **execution-ordering** efficiency issue, not a correctness bug
(output tokens are unchanged) and not a flag-coexistence failure (both flags
work). Making the combined config inherit PKV's TTFT would require reordering
the lookups or skipping the block suffix-prefill when PKV will hit — a **core
change, out of scope** for this benchmark-configuration session. Tracked as a
v5.2 follow-up. The v5.1.1 `RESULTS.md` documents the measured behaviour
honestly rather than the predicted best-case.

## Conclusion

The 38 s number is explained entirely by the missing cache flags on
`squish_daemon`. The combined "recommended" config is the right *default* to
publish (it is how a user who doesn't know their workload should deploy squish,
and it wins long-context e2e against Ollama), but its TTFT is block-cache-class,
not PKV-class — documented honestly in the headline and ablation tables.
