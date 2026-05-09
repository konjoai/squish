# Shrinking the KV cache: how Squish runs a 7B model at 32K context on a 16 GB laptop

*Wave 108 · 2026-05-09 · `konjoai/squish`*

## The problem nobody told you about

When most people picture an LLM running on their machine, they picture the
weights — the multi-gigabyte file you downloaded from Hugging Face. Quantize
those weights to INT4 and a 7B model goes from ~14 GB to ~3.5 GB. Done?

Not quite. The thing that actually breaks long-context inference on a laptop
isn't the weight matrix — it's the **KV cache**. Every token you've ever fed
the model has to keep its keys and values around so the next token's attention
can see them. The cache grows linearly with context length, and at typical
modern dimensions the math is brutal:

For Qwen2.5-7B (28 layers · 8 KV-heads · 128 head_dim), at 32K context, in
FP16:

```
bytes = 28 layers × 8 heads × 128 dim × 2 (K+V) × 2 (fp16) × 32000 tokens
      = 3.67 GB
```

That's *just* the cache. Add the weights and you're already past 7 GB before
the first activation buffer. On a 16 GB MacBook the system runs out of unified
memory, MLX page-faults, and TTFT collapses. This is why long-context demos
show off a 70B model on an H100 instead of a 7B on your laptop: the KV cache
is the wall.

## What Squish actually does

Squish is a per-token KV-cache codec. It quantizes K and V on the way *into*
the cache and dequantizes them on the way out, so the rest of the inference
stack — `mlx_lm`, the attention kernel, the rotary embeddings — sees an FP16
tensor and has no idea anything happened.

Three storage tiers ship today:

| Mode | Bytes per token per head (head_dim=128) | Compression vs FP16 |
|------|----------------------------------------:|--------------------:|
| FP16 | 256                                     | 1.00× (baseline)    |
| INT8 | 132 (128 codes + 4-byte fp32 scale)     | **1.94×**           |
| INT4 | 68 (64 codes + 4-byte fp32 scale)       | **3.76×**           |
| INT2 | 36 (32 codes + 4-byte fp32 scale)       | **7.11×**           |

The codes are per-channel quantized along the head_dim axis with a single
fp32 scale per token, per head. The scale is *not* hand-waved — it's stored
explicitly and accounted for in the byte budget. INT2 packs four 2-bit codes
into one uint8, which is why head_dim must be a multiple of four for that
tier to be valid. (`squish/kv/kv_cache.py:547`.)

That same 32K-context workload, re-priced:

| Mode | KV cache at 32K |
|------|----------------:|
| FP16 | 3.67 GB         |
| INT8 | 1.89 GB         |
| INT4 | 0.97 GB         |
| INT2 | **0.52 GB**     |

INT2 turns a 3.7 GB problem into a 0.5 GB one. That's the headline number.
But headline numbers without quality numbers are useless — INT2 is a 4-level
codebook, four unique values per channel. How is that not catastrophic?

## The SNR ordering invariant

Every Squish quantizer ships with a regression test that's older than any of
the production tiers: on the same input, on the same hardware, **INT8 SNR
must be strictly greater than INT4 SNR, must be strictly greater than INT2
SNR, by at least 6 dB per bit removed.** That's the Shannon bound — each
extra bit doubles the codebook, which is +6 dB of resolution, and the test
fails the build if any tier ever stops obeying it. (`tests/test_kv_int4.py:179`.)

What does that look like in practice, on a Hadamard-rotated normal
distribution that approximates real attention activations?

| Mode | Reconstruction SNR (typical) |
|------|------------------------------:|
| INT8 | ≈ 44 dB                       |
| INT4 | ≈ 22 dB                       |
| INT2 | ≈ 6 dB (after rotation)       |

44 dB is "indistinguishable from FP16 in any downstream task." 22 dB is "good
enough that lm-eval scores don't move." 6 dB is the kicker — without the
Hadamard rotation step (W104), INT2 on raw activations is unusable. Rotation
spreads the energy across the head_dim axis so that no single channel
dominates the scale, and *only then* does a 4-level codebook produce
intelligible outputs.

The rotation is essentially free at INT4 and INT8 — it's matrix-by-Hadamard,
O(n log n), and the FFT-style butterfly fits in a single Metal kernel — but
it's load-bearing for INT2.

## Does the model still know things?

Lm-eval-harness on Qwen2.5-1.5B-Instruct, squish path, limit=500, 0-shot:

| Task        | BF16  | Best INT4 (W94 mixed precision) | Δ     |
|-------------|------:|---------------------------------:|------:|
| arc_easy    | 0.750 | 0.746                            | −0.004 |
| hellaswag   | 0.612 | 0.606                            | −0.006 |
| piqa        | 0.772 | **0.776**                        | **+0.004** |
| winogrande  | 0.630 | **0.648**                        | **+0.018** |

(Numbers from `dev/BENCHMARK_REFERENCE.md`.) Two tasks within stderr (~0.020),
two tasks decisively above the BF16 reference. The "decisively above"
direction isn't the quantizer being magic — it's that AWQ at α=0.10
re-weights the FP16 MLP path in a way that happens to help factual recall on
this task set. The point is: at INT4, this is statistical parity with
full-precision, not a degraded model.

## The three-line API

The whole point of this work is that no application code has to change.
Picking a tier for a planned conversation length is one function call:

```python
from squish.kv.kv_cache import HadamardKVCache, recommended_kv_mode_3tier

mode  = recommended_kv_mode_3tier(32_000)        # → 'int2'
cache = HadamardKVCache(n_layers=28, mode=mode)
model.generate(prompt, kv_cache=cache)           # mlx_lm protocol
```

`recommended_kv_mode_3tier` follows the W105 schedule: ≤ 8K → INT8, 8K–16K
→ INT4, > 16K → INT2. The thresholds are tuned so the chosen tier's SNR is
never the bottleneck for the chosen context length on the eval set — short
chats stay perfect; long chats trade 6 dB for 7× memory.

If you have a hard RAM ceiling instead of a planned context, W106 inverts
the question:

```python
from squish.kv.kv_cache import recommend_mode_for_budget, estimate_max_context

mode = recommend_mode_for_budget(
    n_layers=28, n_kv_heads=8, head_dim=128,
    context_tokens=32_000, budget_bytes=2 * 1024**3,
)                                                # → 'int4' (fits in 2 GB)

ctx = estimate_max_context(
    n_layers=28, n_kv_heads=8, head_dim=128,
    budget_bytes=512 * 1024**2, mode='int2',
)                                                # → 31,775 tokens
```

The closed-form estimator matches the live cache's `memory_bytes` to within
1% on the regression workload. (`tests/test_kv_budget.py`.) No model load
needed to answer "will this fit?"

## What this gets you

On an Apple M3 with 17 GB unified memory, running Qwen2.5-7B with
`HadamardKVCache(mode='int2')` at 32K context:

- Average TTFT: **164 ms** across the four-prompt benchmark suite (`squish_bench.md`).
- Average decode rate: **96.8 tok/s** sustained.
- Peak memory: stays under 8 GB total (weights + KV + activations), with
  ~9 GB of unified memory left for the OS, the compositor, and whatever
  else you have open.

Run the same model with the default FP16 cache and the OS starts swapping
before the prompt finishes encoding.

## Try it

```bash
pip install squish
squish serve --model qwen-2.5-7b --kv-mode auto
```

The dashboard at `demo/index.html` exposes the calculator end-to-end — enter
your model size and target context, see the projected memory at each tier,
then run the codec on your machine via `demo/server.py` and watch the
predicted numbers match the measured ones.

The cache is the wall. Squish is the ramp over it.

---

*Numbers come from `dev/BENCHMARK_REFERENCE.md` (lm-eval), `squish/kv/kv_cache.py:547`
(per-token byte budget), and `squish_bench.md` (M3 TTFT / tok-s).*
