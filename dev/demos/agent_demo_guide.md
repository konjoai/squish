# Agent Demo Recording Guide

Step-by-step guide for recording the squish agent demo.

## Hardware Requirements

- M3 MacBook Pro (or M2/M4 equivalent)
- At least 16 GB unified memory
- Terminal with large font (recommended: 20pt Menlo or JetBrains Mono)
- iTerm2, Ghostty, or WezTerm for true-color ANSI rendering

## Pre-flight Setup

```bash
# 1. Ensure squish is installed and model is available
squish pull squish-community/Qwen2.5-1.5B-Instruct-int4

# 2. Open a tmux session with a side-by-side split
tmux new-session -s demo
# Split vertically: server logs on right, client on left
tmux split-window -h -p 40

# 3. In the RIGHT pane, start the squish server with --agent flag
squish serve squish-community/Qwen2.5-1.5B-Instruct-int4 \
    --agent \
    --port 11435 \
    --prefix-kv-store 200
```

## Demo Flow (5 Steps, ~90 seconds total)

### Step 1: Start the Server (0-15s)

In the right pane, run:
```bash
squish serve squish-community/Qwen2.5-1.5B-Instruct-int4 --agent --port 11435
```

Watch for the startup banner — it should show:
- Load time: ~0.33-0.53 s
- Peak RAM: ~402 MB
- KV cache: quantized enabled

### Step 2: OpenClaw Tool Call (15-35s)

In the left pane, send a tool-calling request:
```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "squish",
    "messages": [{"role": "user", "content": "What is the weather in London?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
          "type": "object",
          "properties": {"location": {"type": "string"}},
          "required": ["location"]
        }
      }
    }],
    "tool_choice": "auto"
  }' | python3 -m json.tool
```

In the server logs (right pane) you should see the tool call parsed and returned.

### Step 3: Show KV Reuse in Logs (35-50s)

Send a follow-up request that reuses the prefix:
```bash
curl -s http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "squish",
    "messages": [
      {"role": "user", "content": "What is the weather in London?"},
      {"role": "assistant", "content": "I will check the weather for you."},
      {"role": "user", "content": "Also check Paris."}
    ]
  }' | python3 -m json.tool
```

Point out in the right-pane logs: `dispatch → prefix-cache HIT` or `RadixAttention` prefix reuse.

### Step 4: Run the 20-Turn Scripted Session (50-75s)

Use the record_agent_demo.py script for a reproducible scripted session:
```bash
python3 dev/demos/record_agent_demo.py
```

This sends 20 predefined tool-call turns and prints TTFT and token counts per turn.
The logs in the right pane will show KV reuse accumulating across turns.

### Step 5: Show MoE Lookahead Stats (75-90s)

Check the /v1/metrics endpoint to display throughput stats:
```bash
curl -s http://localhost:11435/v1/metrics | python3 -m json.tool
```

Point out:
- `tokens_gen`: total tokens generated
- `avg_ttft_ms`: average time-to-first-token
- `tps`: tokens per second

## Recording Instructions

Use asciinema for a reproducible, playable recording:
```bash
# Start recording
asciinema rec squish-agent-demo.cast --cols 220 --rows 50

# ... run the demo steps above ...

# Stop recording (Ctrl+D)
```

Convert to GIF for embedding in README / HN post:
```bash
# Install agg (asciinema GIF generator)
cargo install agg  # or: brew install agg

# Generate GIF at 2x speed, 220 cols
agg squish-agent-demo.cast squish-agent-demo.gif \
    --cols 220 --rows 50 \
    --speed 2.0 \
    --font-size 14
```

## Target: 90 Seconds

| Segment | Duration | Content |
|---------|----------|---------|
| Server startup | 0-15s | squish serve, show fast cold start |
| Tool call | 15-35s | OpenClaw weather tool, show response |
| KV reuse | 35-50s | Follow-up turns, show prefix cache hit |
| 20-turn session | 50-75s | Scripted session with TTFT logging |
| Metrics | 75-90s | /v1/metrics, highlight throughput |

## Tmux Layout

```
+----------------------------------+------------------+
|  Left pane (60%): curl commands  | Right pane (40%) |
|  python record_agent_demo.py     | squish serve     |
|  curl /v1/metrics                | server logs      |
+----------------------------------+------------------+
```

Set this up with:
```bash
tmux new-session -s squish-demo \; \
    split-window -h -p 40 \; \
    select-pane -t 0
```
