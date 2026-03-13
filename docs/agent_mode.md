# Squish Agent Mode

**`squish serve --agent` — Agentic Runtime for Apple Silicon**

Agent mode enables a single-flag agentic configuration optimized for
16 GB M-series systems running multi-turn tool-calling agents (OpenClaw,
Continue.dev, LangChain, custom agent loops).

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Apple Silicon | M1 | M3 / M3 Pro |
| Unified RAM | 8 GB | 16 GB |
| Storage | 20 GB free | 40 GB free (multiple models) |
| macOS | 13.0 Ventura | 14.0+ Sonoma |

---

## Quick Start

```bash
# Pull a recommended agent model
squish pull qwen3:8b

# Start in agent mode
squish serve qwen3:8b --agent

# The server is now ready at http://localhost:11435/v1
```

---

## What `--agent` Enables

The `--agent` flag is a preset that automatically activates:

| Flag | Module | Benefit |
|------|--------|---------|
| `--agent-kv` | `agent_kv.py` | INT2 history KV, 6× footprint reduction |
| `--grammar` | `grammar_engine.py` | 100% valid JSON tool calls |
| `--chunked-prefill` | `chunked_prefill.py` | Bounded TTFT for long system prompts |
| `--radix-cache` | `radix_cache.py` | Prefix deduplication across turns |
| `--paged-kv` | `paged_attention.py` | Zero KV fragmentation |
| `--prompt-lookup` | `prompt_lookup.py` | N-gram speculation (doc-heavy agents) |
| `--power-monitor` | `power_monitor.py` | Battery-aware mode switching |
| `--metal-fusion` | `metal_fusion.py` | Fused RoPE/QKV/SwiGLU kernels |
| `--fault-tolerance` | `fault_tolerance.py` | Last-resort OOM safety net |

**Automatic behaviors (no additional flags needed):**
- `max_batch_size = 1` (agent workloads are single-user)
- `context_length` auto-sized from available UMA: `min(32768, floor(free_gb × 2048))`
- Per-turn memory log: `Turn N | KV: X.X GB | Free UMA: Y.Y GB`

---

## Recommended Models (16 GB M3)

| Model | Size (INT4) | Use case |
|-------|------------|---------|
| `qwen3:8b` | ~4.8 GB | Best general-purpose coding + tool-calling |
| `qwen3:14b` | ~8.2 GB | Best reasoning at 16 GB |
| `llama3.1:8b` | ~4.8 GB | Broadly compatible, safe default |
| `deepseek-r1:7b` | ~4.1 GB | Best chain-of-thought reasoning |

```bash
squish pull qwen3:8b
squish serve qwen3:8b --agent
```

---

## AgentKV: 6× KV Cache Compression

Standard FP16 KV cache at 32K context for a 14B model requires ~12.5 GB — more
than the entire 16 GB budget. Agent mode uses a three-tier KV layout:

```
KV layout for 32K-token context on Qwen2.5-14B:
┌──────────────────┬──────────────────────────┬────────────────┐
│  Attention Sink  │   Historical Middle       │  Local Window  │
│  tokens 0–3      │   tokens 4–(N-128)        │  tokens N-128–N│
│  FP16 (hot)      │   INT2 group-quantized    │  FP16 (rolling)│
│  ~0.001 GB       │   ~2.1 GB (vs 12.5 FP16)  │  ~0.25 GB      │
└──────────────────┴──────────────────────────┴────────────────┘
Total: ~2.35 GB vs 12.5 GB FP16  →  5.3× compression
```

**Why this works:**
- **Attention sinks** (StreamingLLM, 2023): first few tokens receive disproportionate
  attention weight and must stay in high precision to preserve coherence
- **Local window**: most recent context dominates next-token prediction — kept FP16
- **INT2 history**: coarse value of distant KV pairs still guides attention; 2-bit is
  sufficient for memory-bounded workloads (validated by PQCache, CommVQ papers)

---

## macOS Memory Governor

On macOS, agent mode starts a background `MemoryGovernor` thread that reads
`vm_stat` every 500 ms and triggers memory-recovery actions before the OS reaches
page-swap:

| Threshold | Action |
|-----------|--------|
| Free UMA < 1.5 GB (CAUTION) | Disable non-essential KV cache tiers |
| Free UMA < 0.8 GB (CRITICAL) | Force AgentKV INT2 across all layers |
| Free UMA < 0.4 GB (EMERGENCY) | Flush context cache + reduce batch size |

This runs automatically — no additional flags needed on macOS.

---

## OpenClaw Integration

```python
# OpenClaw / OpenDevin configuration
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11435/v1",
    api_key="squish",
)

# Tool-calling agent loop
response = client.chat.completions.create(
    model="squish",
    messages=[
        {"role": "system", "content": "You are a coding agent with access to code execution tools."},
        {"role": "user",   "content": "Refactor this Python module to use async/await."},
    ],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python code and return the output",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"}
                    },
                    "required": ["code"],
                },
            },
        }
    ],
    tool_choice="auto",
    stream=True,
)
```

---

## Continue.dev Configuration

Add to `.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Squish Agent (local)",
      "provider": "openai",
      "model": "squish",
      "apiBase": "http://localhost:11435/v1",
      "apiKey": "squish",
      "contextLength": 32768,
      "completionOptions": {
        "temperature": 0.1,
        "maxTokens": 4096
      }
    }
  ]
}
```

Start squish in agent mode first:

```bash
squish serve qwen3:8b --agent
```

---

## LangChain Integration

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import Tool

llm = ChatOpenAI(
    base_url="http://localhost:11435/v1",
    api_key="squish",
    model="squish",
    streaming=True,
    temperature=0,
)

# Define tools
tools = [
    Tool(
        name="python_repl",
        func=lambda code: exec(code) or "executed",
        description="Execute Python code",
    ),
]

agent = create_tool_calling_agent(llm, tools, prompt=...)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
executor.invoke({"input": "Analyze this dataset and produce a summary."})
```

---

## Memory Budget Guide

| Model | Weights | 8K ctx KV | 16K ctx KV | 32K ctx KV | 16 GB viable? |
|-------|---------|-----------|------------|------------|------------:|
| Qwen3-8B INT4 | 4.8 GB | 0.4 GB | 0.7 GB | 1.2 GB | ✓ Yes |
| Qwen3-14B INT4 | 8.2 GB | 0.6 GB | 1.1 GB | 2.1 GB | ✓ Yes |
| Qwen3-32B INT4 | 17.4 GB | — | — | — | ✗ No |

*AgentKV INT2 history estimates. macOS kernel uses ~3.5 GB.*

---

## Flags Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--agent` | off | Enable all agent-mode optimizations |
| `--agent-kv` | off | AgentKV INT2 history tier only |
| `--grammar` | off | XGrammar JSON tool-call enforcement |
| `--radix-cache` | off | Token-prefix KV deduplication |
| `--paged-kv` | off | Paged KV blocks (no fragmentation) |
| `--chunked-prefill` | off | Process long prompts in chunks |
| `--metal-fusion` | off | Fused Metal kernels for RoPE/SwiGLU |

---

## Troubleshooting

**First turn is slow**: The Metal JIT warms up on the first request — this is expected.
Subsequent turns at the same context length will be fast.

**Model keeps reloading**: Ensure you're using the same `--agent` flag every invocation.
The model stays resident after `squish serve` starts.

**Grammar enforcement failing**: Ensure your tool schema uses `"type": "object"` with
`"required"` fields. XGrammar requires a valid JSON Schema.

**Memory CRITICAL warnings**: Run `squish doctor` to check free UMA. Consider a smaller
model or reduce `--max-tokens` to free KV headroom.

**Context limit exceeded**: The auto-sized `context_length = min(32768, floor(free_gb × 2048))`
may cap at 8192 on low-memory systems. Override with `--max-kv-size 16384`.

---

*Phase 13D — Agent Preset | Squish v10.1*
