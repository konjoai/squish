#!/usr/bin/env python3
"""Debug script to isolate the streaming hang."""
import sys, time, traceback
sys.path.insert(0, '/Users/wscholl/squish')

print("=== Step 1: import ===")
try:
    from squish.quant.compressed_loader import load_compressed_model
    import mlx_lm
    print("imports OK")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 2: load model ===")
try:
    t0 = time.perf_counter()
    model, tok, stats = load_compressed_model(
        '/Users/wscholl/squish/models/Qwen3-8B-bf16',
        '/Users/wscholl/squish/models/Qwen3-8B-bf16-compressed',
        verbose=True, return_stats=True
    )
    print(f"Loaded in {time.perf_counter()-t0:.1f}s  loader={stats['loader']}")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("\n=== Step 3: chat template ===")
msgs = [{'role': 'user', 'content': 'What is 2+2? Answer briefly.'}]
try:
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    print(f"Prompt ({len(prompt)} chars): {prompt[:120]!r}")
except Exception as e:
    prompt = 'What is 2+2? Answer briefly.'
    print(f"chat_template fallback ({e}): {prompt!r}")

print("\n=== Step 4: stream_generate (first 8 tokens) ===")
try:
    t1 = time.perf_counter()
    for i, item in enumerate(mlx_lm.stream_generate(model, tok, prompt, max_tokens=8)):
        txt = item.text if hasattr(item, 'text') else str(item)
        print(f"  [{i}] {txt!r}  ({time.perf_counter()-t1:.3f}s)", flush=True)
    print(f"Done in {time.perf_counter()-t1:.1f}s")
except Exception as e:
    traceback.print_exc()
    sys.exit(1)

print("\n=== All steps OK ===")
