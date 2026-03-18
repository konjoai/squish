#!/usr/bin/env python3
"""
Direct benchmark of the Qwen3-8B INT4 model — no server, no HTTP.
Measures: load time, TTFT, tokens/sec across several prompts.
"""
import sys, time, statistics
sys.path.insert(0, '/Users/wscholl/squish')

MODEL_DIR  = '/Users/wscholl/squish/models/Qwen3-8B-bf16'
COMP_DIR   = '/Users/wscholl/squish/models/Qwen3-8B-bf16-compressed'
MAX_TOKENS = 100   # cap so bench doesn't run forever
# /no_think suffix disables Qwen3 chain-of-thought reasoning (faster, direct answers)
PROMPTS = [
    "What is 7 multiplied by 8? /no_think",
    "Name the capital of France. /no_think",
    "Write a haiku about autumn leaves. /no_think",
    "What is the boiling point of water in Celsius? /no_think",
    "Explain what a neural network is in one sentence. /no_think",
]

print("=" * 60)
print("  Squish Qwen3-8B INT4 Direct Benchmark")
print("=" * 60)

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[1/2] Loading model...")
from squish.quant.compressed_loader import load_compressed_model
import mlx_lm

t_load_start = time.perf_counter()
model, tokenizer, stats = load_compressed_model(
    MODEL_DIR, COMP_DIR, verbose=True, return_stats=True
)
load_time = time.perf_counter() - t_load_start
print(f"\n  Loader : {stats['loader']}")
print(f"  Load   : {load_time:.1f}s")

# ── Benchmark ─────────────────────────────────────────────────────────────────
print(f"\n[2/2] Benchmarking {len(PROMPTS)} prompts (max {MAX_TOKENS} tokens each)...")
print("-" * 60)

all_ttft   = []
all_tps    = []
all_tokens = []

for i, user_text in enumerate(PROMPTS, 1):
    # Apply Qwen3 chat template
    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"User: {user_text}\nAssistant:"

    prompt_tokens = len(tokenizer.encode(prompt)) if hasattr(tokenizer, 'encode') else 0

    # Stream and time
    t_start   = time.perf_counter()
    ttft      = None
    n_tokens  = 0
    output    = []

    for item in mlx_lm.stream_generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
        tok_text = item.text if hasattr(item, 'text') else str(item)
        if ttft is None:
            ttft = time.perf_counter() - t_start
        n_tokens += 1
        output.append(tok_text)

    total_time = time.perf_counter() - t_start
    tps = n_tokens / total_time if total_time > 0 else 0
    full_output = "".join(output)

    # Strip <think>...</think> block for display
    import re
    display = re.sub(r'<think>.*?</think>', '', full_output, flags=re.DOTALL).strip()
    if not display:
        display = full_output  # fallback if no think block

    all_ttft.append(ttft or 0)
    all_tps.append(tps)
    all_tokens.append(n_tokens)

    print(f"\n[{i}/{len(PROMPTS)}] {user_text[:50]!r}")
    print(f"  Prompt tokens : {prompt_tokens}")
    print(f"  Output tokens : {n_tokens}")
    print(f"  TTFT          : {ttft:.3f}s")
    print(f"  Total time    : {total_time:.2f}s")
    print(f"  Throughput    : {tps:.1f} tok/s")
    print(f"  Response      : {display[:150]!r}")

print("\n" + "=" * 60)
print("  Summary")
print("=" * 60)
print(f"  Model         : Qwen3-8B  INT4 (group-16, --no-awq)")
print(f"  Loader        : {stats['loader']}")
print(f"  Load time     : {load_time:.1f}s")
print(f"  Prompts       : {len(PROMPTS)}")
print(f"  Avg tokens    : {statistics.mean(all_tokens):.0f}")
print(f"  Avg TTFT      : {statistics.mean(all_ttft):.3f}s  (min {min(all_ttft):.3f}s)")
print(f"  Avg throughput: {statistics.mean(all_tps):.1f} tok/s  (max {max(all_tps):.1f})")
print(f"  P50 tps       : {statistics.median(all_tps):.1f} tok/s")
print("=" * 60)
