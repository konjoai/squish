# OpenClaw Integration

[OpenClaw](https://github.com/open-claw/openclaw) is an open-source AI agent framework for macOS that provides tool-use, memory, and multi-step reasoning on top of a local OpenAI-compatible endpoint.

Squish exposes a fully OpenAI-compatible API at `http://localhost:11435/v1`, so OpenClaw can use any Squish-hosted model with zero code changes.

---

## Quick Start

### 1. Start Squish

```bash
squish serve qwen2.5:7b
```

The server starts on `http://localhost:11435` by default. It exposes:
- `POST /v1/chat/completions`
- `GET /v1/models`
- `POST /v1/completions`

### 2. Configure OpenClaw

In your OpenClaw config file (usually `~/.openclaw/config.json` or the project-level `openclaw.json`), point the agent at the Squish server:

```json
{
  "agent": {
    "model": "openai/qwen2.5:7b",
    "openaiBaseUrl": "http://localhost:11435/v1",
    "apiKey": "squish"
  }
}
```

> **Note:** Squish does not require an API key by default. Set `"apiKey"` to any non-empty string — the value is ignored unless you started Squish with `--api-key`.

### 3. Run OpenClaw

```bash
openclaw run "Summarise the README in this repo"
```

OpenClaw will route all model calls through Squish.

---

## Recommended Model

**Qwen2.5-7B INT3** is the recommended model for OpenClaw tool-use workflows:

| Model | RAM | TTFT | Tool use |
|-------|-----|------|----------|
| `qwen2.5:7b` INT4 | ~5.5 GB | < 600 ms | ✓ Reliable |
| `qwen2.5:7b` INT4 | ~2.5 GB | < 300 ms | ✓ Good |
| `qwen2.5:1.5b` INT4 | ~1.0 GB | < 200 ms | ~ Limited |

Anything below 4B parameters has unreliable structured JSON / tool-call output. For multi-step agent loops requiring consistent tool-use, use `qwen2.5:7b` or larger.

---

## Using a Custom Port

```bash
squish run qwen2.5:7b
```

```json
{
  "agent": {
    "openaiBaseUrl": "http://localhost:11435/v1"
  }
}
```

---

## Using a Private API Key

```bash
squish serve qwen2.5:7b --api-key mysecretkey
```

```json
{
  "agent": {
    "openaiBaseUrl": "http://localhost:11435/v1",
    "apiKey": "mysecretkey"
  }
}
```

---

## Streaming

Squish supports server-sent events (SSE) streaming (`"stream": true`). OpenClaw enables streaming by default and will receive tokens incrementally.

---

## Troubleshooting

**"Connection refused"** — Squish is not running. Start it with `squish serve <model>`.

**"model not found"** — The model ID in your OpenClaw config does not match the model Squish loaded. Use `curl http://localhost:11435/v1/models` to see the available model IDs, then update your config.

**Slow first response** — On the first run after compression, Squish builds an optimised weight cache. Subsequent starts load in 3–5 seconds.

**Tool calls return malformed JSON** — This is a model-capability issue, not a Squish or OpenClaw bug. Switch to `qwen2.5:7b` or a larger model. See [Model Capability Reality Checks](../ARCHITECTURE.md).
