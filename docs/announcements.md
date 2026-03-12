# Squish — Community Announcement Drafts

Pre-written posts for HN, r/LocalLLaMA, and Twitter/X.
Customise dates and benchmark numbers from a real `bench_eoe.py` run before posting.

---

## Hacker News — Show HN

**Title:** Show HN: Squish – sub-second local LLM loading on Apple Silicon (54× faster cold start)

**Body:**

Squish is an open-source local LLM server for Apple Silicon that solves a specific
problem: `.safetensors` was designed for training checkpoints, not runtime loading.
Every boot pays the same dtype-conversion tax — 8–28 seconds and 2.4 GB of RAM for
data that hasn't changed since last time.

The fix is a one-time format conversion that stores weights in a BF16 Metal-native
safetensors file. `mx.load()` memory-maps it directly into unified memory — no CPU-side
allocation, no dtype conversion.

Results on Apple Silicon M-series (Qwen2.5-1.5B-Instruct):
- Cold load: 0.53 s vs 28.81 s (54× faster)
- RAM during load: 160 MB vs 2,400 MB (15× less)
- lm-eval accuracy: within ±1.5 pp on ARC/HellaSwag/WinoGrande/PIQA

The server is OpenAI-compatible (/v1/*) and Ollama-compatible (/api/*) so you switch
with one environment variable. It also ships 100+ composable inference optimisation
modules (speculative decoding, KV compression, attention variants, quantisation) as
toggleable flags.

GitHub: https://github.com/wesleyscholl/squish
Pre-squished weights: https://huggingface.co/squish-community
Paper: https://arxiv.org/abs/XXXX.XXXXX [when available]

---

## r/LocalLLaMA

**Title:** Squish: 54× faster model loading on Apple Silicon — sub-second cold start, 15× less RAM

**Body:**

Hey r/LocalLLaMA,

I've been working on a local LLM inference tool for Apple Silicon that addresses something
that's been bugging me: why does every model load take 8–25+ seconds even if I loaded
the same model yesterday?

The answer is that `.safetensors` stores weights in a format designed for training
checkpoints. Every boot, the loader re-allocates ~2.4 GB in CPU heap and converts dtypes
even though nothing changed. Ollama and LM Studio have the same problem — they use the
same base loading path.

**Squish fixes it with a one-time conversion** to a BF16 MLX-native format. After that:

| | Cold mlx_lm | Squish (cached) |
|--|--|--|
| Load time | 28.81 s | **0.53 s** |
| RAM (load phase) | 2,400 MB | **160 MB** |
| Delete original .safetensors? | No | **Yes** |

**Accuracy:** lm-eval benchmarks on INT8-compressed Qwen2.5-1.5B show max ±1.5 pp
delta vs bf16 baseline across ARC-Easy, HellaSwag, WinoGrande, PIQA.

**APIs:** Drop-in for OpenAI (`export OPENAI_BASE_URL=http://localhost:11435/v1`) and
Ollama (`export OLLAMA_HOST=http://localhost:11435`). Web chat UI at `/chat`.

**100+ inference modules:** KV cache compression, EAGLE-3/MEDUSA speculative decoding,
SageAttention2, quantisation (FP8/INT3/NF4), continuous batching, multi-tenant serving —
all as CLI flags on `squish serve`.

**Quick start:**

```bash
pip install squish
squish catalog          # browse 29 models
squish pull qwen3:8b    # download + compress once
squish run qwen3:8b     # serve on :11435
```

GitHub: https://github.com/wesleyscholl/squish
Pre-squished HF weights: https://huggingface.co/squish-community

Would love feedback on
1. The benchmark numbers (run `bench_eoe.py` yourself!)
2. Whether the module flag API makes sense
3. Models you'd like to see pre-squished

---

## Twitter / X

**Thread (1/5):**

Squish: 54× faster local LLM cold start on Apple Silicon.

Qwen2.5-1.5B: 0.53 s vs 28.81 s cold.
RAM during load: 160 MB vs 2,400 MB.
Delete the original .safetensors. You don't need it anymore.

github.com/wesleyscholl/squish

**Thread (2/5):**

The problem: .safetensors was designed for training checkpoints.
Every boot re-allocates ~2.4 GB in CPU heap and re-converts dtypes.
The data hasn't changed.

Squish converts once to BF16 MLX-native format.
mx.load() mmap-maps directly into unified memory.
Zero CPU allocation. Zero dtype conversion.

**Thread (3/5):**

Accuracy on INT8 compression (lm-eval-harness, 200 samples):

ARC-Easy:  73.5% vs 74.5% (−1.0 pp) ✅
HellaSwag: 62.0% vs 63.5% (−1.5 pp) ✅
WinoGrande: 67.0% vs 65.5% (+1.5 pp) ✅
PIQA: 76.5% vs 77.5% (−1.0 pp) ✅

All within measurement noise at n=200.

**Thread (4/5):**

Drop-in for OpenAI and Ollama clients — zero code changes:

export OPENAI_BASE_URL=http://localhost:11435/v1
# or
export OLLAMA_HOST=http://localhost:11435

Also ships 100+ composable inference modules as CLI flags:
EAGLE-3, MEDUSA, SageAttn2, FlashMLA, FP8, INT3, continuous batching, multi-tenant WFQ...

**Thread (5/5):**

Quick start:

pip install squish
squish pull qwen3:8b   # one-time 5-min compression
squish run qwen3:8b    # sub-second every boot

Pre-squished weights (no conversion needed):
hf.co/squish-community

Paper: arxiv.org/abs/XXXX.XXXXX [when submitted]

---

## Discord / Slack template

> **🚀 Squish v7.0.1 is out**
>
> Sub-second local LLM loading on Apple Silicon. 54× faster cold start, 15× less RAM,
> statistically identical accuracy. Drop-in for OpenAI and Ollama.
>
> Install: `pip install squish`
> Docs: https://github.com/wesleyscholl/squish
> Pre-squished models: https://huggingface.co/squish-community
