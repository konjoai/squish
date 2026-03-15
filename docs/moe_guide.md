# MoE Models Guide

Squish has first-class support for Mixture-of-Experts (MoE) models. These models
have a large total parameter count but only activate a small subset (the "active
parameters") during each forward pass, making them much faster than their total size
suggests.

---

## MoE Models in the Catalog

| Model ID | Total params | Active params | INT4 size | Fits 16 GB M3? |
|---|---|---|---|---|
| `qwen3:30b-a3b` | 30B | ~3B | ~5.0 GB | Yes (11 GB agent headroom) |

To see all MoE models:

```bash
squish catalog --tag moe
```

The `squish catalog` output shows a `[MoE: X total / Y active]` badge for these
models so you can immediately identify them.

---

## Why MoE Models Are Efficient

A standard 30B dense model requires ~80 GB of VRAM and is not feasible on consumer
hardware. `qwen3:30b-a3b` uses a sparse MoE architecture where 30B parameters are
distributed across expert networks, but each token only routes through ~3B worth of
parameters. At INT4 compression, the total weight size is ~5 GB — comparable to a
dense 3B model but with 30B total capacity.

---

## Running a MoE Model

### Basic serving

```bash
squish serve qwen3:30b-a3b
```

### Agent mode (recommended for long-context tasks)

The `--agent` preset automatically enables `--moe-lookahead` for MoE models:

```bash
squish serve qwen3:30b-a3b --agent
```

This activates:
- `--agent-kv`: asymmetric INT2 KV cache (6× footprint reduction)
- `--moe-lookahead`: expert prefetching via EMA-delta hidden state prediction
- `--chunk-prefill`: bounded time-to-first-token
- `batch-size=1`: single-slot serving for agent loops

### Manual MoE lookahead

```bash
squish serve qwen3:30b-a3b --moe-lookahead --moe-lookahead-steps 3
```

---

## Expert Lookahead Router (Phase 14)

The `--moe-lookahead` flag activates the `MoELookaheadRouter` from
`squish/moe/moe_lookahead.py`.

### How it works

1. After each decode step, the router computes an Exponential Moving Average (EMA)
   of the frame-to-frame delta of the mean hidden state.
2. The EMA delta is used to project the hidden state `k` steps into the future.
3. The sparse MoE router is applied to each projected state to predict which
   experts will be needed.
4. The union of all predicted expert indices forms the **prefetch set**.
5. The next actual decode step evaluates whether any actual expert was in the
   prefetch set (**hit rate**).

### Benchmarking lookahead

```bash
python dev/benchmarks/bench_moe_lookahead.py --n-experts 64 --top-k 2
```

Sample output on synthetic traces resembling DeepSeek-Coder-V2-Lite:

```
  Regime         Hit rate    Latency µs/tok
  ──────────── ──────────  ────────────────
  flat             100.0%          580.00
  random            91.0%          572.00
  drifting          92.5%          570.00
```

### Lookahead configuration

| Flag | Default | Description |
|---|---|---|
| `--moe-lookahead` | `False` | Enable MoE expert lookahead |
| `--moe-lookahead-steps` | `3` | Lookahead horizon steps |

---

## DeepSeek-Coder-V2-Lite Setup on 16 GB M3

DeepSeek-Coder-V2-Lite (16B total / 2.4B active) is not yet in the default
squish catalog. To use it as a custom model:

1. **Download the MLX-converted weights:**

```bash
huggingface-cli download mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx \
  --local-dir ~/models/deepseek-coder-v2-lite
```

2. **Start squish with agent mode:**

```bash
squish serve ~/models/deepseek-coder-v2-lite --agent --moe-lookahead --port 11434
```

3. **Verify with a code generation request:**

```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"squish","messages":[{"role":"user","content":"Write a Python quicksort."}],"max_tokens":256}'
```

### Expected performance on M3 Pro 16 GB

| Config | TPS | Peak RAM |
|---|---|---|
| Default FP16 (baseline) | ~15 | ~9 GB |
| INT4 + agent-kv | ~42 | ~6 GB |
| INT4 + agent-kv + moe-lookahead | ~46 | ~6 GB |

*(Simulated estimates — actual numbers depend on context length and model.)*

---

## Identifying MoE Models Programmatically

```python
from squish.catalog import list_catalog

moe_models = [e for e in list_catalog() if e.moe]
for m in moe_models:
    active = f"{m.active_params_b:.1f}B" if m.active_params_b else "unknown"
    print(f"{m.id:30s}  {m.params:>5} total  {active:>8} active")
```
