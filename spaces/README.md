---
title: Squish KV-Cache Quantization
emoji: 🥒
colorFrom: green
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: bsl-1.1
short_description: Try INT8/INT4/INT2 KV-cache quantization in your browser.
tags:
  - quantization
  - kv-cache
  - llm
  - inference
  - apple-silicon
---

# squish · KV-cache quantization demo

A zero-install browser demo of the three KV-cache storage tiers shipped
in [squish](https://github.com/squishai/squish): **INT8**, **INT4**, and
**INT2** — plus the optional Hadamard rotation that makes the low-bit
modes viable on real activations.

## Two tabs

**Tensor Inspector** — pick a synthetic activation distribution
(Gaussian / heavy-tailed / outlier-spiked), choose whether to apply the
QuaRot-style randomised Hadamard rotation, and see SNR, bytes-per-token,
and compression ratio at every tier. The "outlier-spiked, no rotation"
example is the dramatic INT2 bin-collapse failure that motivated the W104
codec design — flip rotation on and watch SNR jump 8-10 dB.

**Memory Budgeter** — pick a real squish-community model preset (Qwen2.5
0.5B / 1.5B / 3B / 7B, Llama-3.1-8B), set a context length and a RAM
budget, and see closed-form KV-cache memory at every tier with a "fits /
over by N MB" verdict per row. The numbers are bit-identical to what
`squish.kv.kv_cache.estimate_kv_memory` returns inside the inference
server — same code path, no fudge factors.

## How to read the SNR numbers

| Tier | Typical SNR (rotated) | Compression vs fp16 (head_dim=128) |
|------|------------------------|-------------------------------------|
| INT8 | ~44 dB                 | 1.94×                               |
| INT4 | ~22 dB                 | 3.76×                               |
| INT2 | ~6 dB                  | 7.11×                               |

Each extra bit of code precision is worth ~6 dB of SNR (the Shannon
quantisation bound). The Hadamard rotation buys back roughly the gap
between heavy-tailed and Gaussian inputs — typically +5 dB at INT4 and
+8-10 dB at INT2. Without rotation, naive INT2 falls below 0 dB on
outlier-heavy activations and the cache is effectively destroyed.

## Source

The Gradio app is `app.py` in this Space; the pure logic lives in
`_logic.py` and is unit-tested in the squish repository's
`tests/test_spaces_demo.py`. Both files mirror the source at
[github.com/squishai/squish/tree/main/spaces](https://github.com/squishai/squish/tree/main/spaces).

```python
# This is the production code path — same module, same numbers:
from squish.kv.kv_cache import make_kv_cache, recommended_kv_mode_3tier

mode  = recommended_kv_mode_3tier(planned_context_tokens=32_000)  # → "int2"
cache = make_kv_cache(n_layers=28, planned_context=32_000)        # ready for mlx_lm
```

## License

BUSL-1.1 — same as squish itself. See the
[upstream LICENSE](https://github.com/squishai/squish/blob/main/LICENSE).
