# Using Squish as a Local Coding Agent in VS Code

Run a fully offline LLM coding agent directly inside VS Code — no API keys, no data leaving your machine.

---

## Overview

Squish exposes an OpenAI-compatible API on `http://localhost:11435/v1`. Any VS Code extension that supports a custom LLM endpoint works as-is. Three paths are covered here, from simplest to most powerful:

| Path | What it gives you | Setup time |
|------|-------------------|------------|
| **Squish VS Code Extension** | Built-in sidebar chat, start/stop server, streaming | 2 min |
| **Continue.dev** | Inline autocomplete + agent chat, slash commands, codebase context | 5 min |
| **GitHub Copilot Chat (custom endpoint)** | Copilot UX with local inference | 5 min |

---

## Option 1 — Squish VS Code Extension (built-in)

The Squish repository ships a first-party VS Code extension at `extensions/vscode/squish-vscode/`.

### Install from source

```bash
# 1. Build the extension package
cd /path/to/squish/extensions/vscode/squish-vscode
npm install
npm run compile
npx vsce package          # produces squish-X.Y.Z.vsix

# 2. Install in VS Code
code --install-extension squish-*.vsix
```

Or open VS Code → `Ctrl+Shift+P` → **Extensions: Install from VSIX…** and select the `.vsix`.

### First run

1. Start your model server:
   ```bash
   squish run qwen3:8b      # auto-enables --agent on Apple Silicon
   ```
2. Click the **flask icon** (🧪) in the VS Code activity bar.
3. Type a message and press **Enter**.

### What's included

| Feature | Detail |
|---------|--------|
| Sidebar chat | Full conversation history per session |
| Streaming | Tokens appear in real time as they generate |
| Model selector | `Squish: Select Model` quick-pick from active server |
| Server lifecycle | `Squish: Start Server` / `Stop Server` commands |
| Status bar | Live server status + current model name |
| Auto-start | Optionally start server when VS Code opens |

### Settings

Open `settings.json` (or **Settings → Extensions → Squish**):

```json
{
  "squish.host":         "127.0.0.1",
  "squish.port":         11435,
  "squish.apiKey":       "squish",
  "squish.model":        "qwen3:8b",
  "squish.autoStart":    false,
  "squish.maxTokens":    2048,
  "squish.temperature":  0.7,
  "squish.systemPrompt": "You are a senior software engineer. Answer concisely."
}
```

### Architecture note

The extension runs entirely in the VS Code extension host — no webview-to-server direct calls. The host process fetches from `http://squish.host:squish.port`, relays streamed tokens to the webview via `postMessage`, and enforces a strict CSP (`default-src 'none'`). No network traffic leaves `localhost`.

---

## Option 2 — Continue.dev (inline autocomplete + agent)

[Continue.dev](https://continue.dev) is the most capable VS Code agent integration for local models.
It provides:
- Tab autocomplete (fills in code as you type)
- `@codebase` context (indexes your repo for semantic search)
- Slash commands (`/edit`, `/review`, `/fix`)
- Chat panel with file references

### Install Continue

1. Install the Continue extension from the VS Code Marketplace:
   ```
   ext install Continue.continue
   ```
   Or: `Ctrl+Shift+P` → **Extensions: Install Extensions** → search "Continue"

2. Start a squish server:
   ```bash
   squish run qwen3:8b
   ```

### Configure squish as the backend

Open `~/.continue/config.json` (Continue creates this on first run) and paste:

```json
{
  "models": [
    {
      "title": "Squish — qwen3:8b (local)",
      "provider": "openai",
      "model": "qwen3:8b",
      "apiBase": "http://localhost:11435/v1",
      "apiKey": "squish",
      "contextLength": 32768,
      "completionOptions": {
        "maxTokens": 2048,
        "temperature": 0.2
      }
    }
  ],
  "tabAutocompleteModel": {
    "title": "Squish autocomplete",
    "provider": "openai",
    "model": "qwen3:4b",
    "apiBase": "http://localhost:11435/v1",
    "apiKey": "squish"
  },
  "embeddingsProvider": {
    "provider": "openai",
    "model": "qwen3:0.6b",
    "apiBase": "http://localhost:11435/v1",
    "apiKey": "squish"
  }
}
```

> **Model selection guide:**
> - Chat/agent: `qwen3:8b` (best quality at 16 GB RAM)
> - Autocomplete: `qwen3:4b` (faster, lower latency)
> - Embeddings: `qwen3:0.6b` or `smollm2:135m` (sub-100ms per chunk)

### Enable codebase indexing

After configuring, open the Continue sidebar (default: `Ctrl+L`), then:
1. Click the settings icon → **Index codebase**
2. Continue will embed your repo files for `@codebase` queries

### Recommended Continue workflow

```
@codebase What does the auth middleware do?
@file src/api/routes.ts Refactor this handler to extract validation
/edit Make this function async and add error handling
/review Does this implementation handle the edge case where...
```

---

## Option 3 — GitHub Copilot Chat with local Squish backend

If you have GitHub Copilot installed, you can point it at Squish via VS Code's `languageModels` setting. This is the most experimental option but gives you the full Copilot UX with local inference.

### Setup

Add to `settings.json`:

```json
{
  "github.copilot.advanced": {
    "debug.overrideEngine": "squish-local"
  }
}
```

For Copilot Chat's agent mode, squish's `--agent` preset wires:
- **AgentKV INT2** asymmetric KV compression (50% memory reduction on long contexts)
- **Grammar-enforced tool calling** via XGrammar (prevents malformed JSON tool responses)
- **Semantic response cache** (40–60% TTFT reduction on repeated queries in the same session)
- **RadixTree prefix cache** (reuses shared system prompt + tool definitions across turns)

Start with the agent preset explicitly:
```bash
# Agent preset is auto-enabled on Apple Silicon when you run squish run
# To set it manually:
squish run qwen3:8b --agent
```

> Then configure VS Code to use the local endpoint as shown above.

---

## SquishBar — macOS Menu Bar App

The macOS menu bar app (`apps/macos/SquishBar`) handles the server lifecycle so you don't need a terminal open.

### Build

```bash
cd apps/macos/SquishBar
swift build -c release
.build/release/SquishBar
```

Or open `Package.swift` in Xcode and Run (⌘R).

The menu bar icon turns **green** when the server is running and shows live tok/s in the title. Click to:
- Start/stop the server
- Select the active model  
- Open the web chat UI
- Copy the API URL to clipboard (for pasting into Continue, VS Code, or any OpenAI client)

Set `squish.autoStart = true` to have it start the server automatically when your Mac wakes.

---

## Recommended Model by Use Case

| Use case | Recommended model | RAM needed | Typical speed |
|----------|-------------------|------------|---------------|
| Inline autocomplete | `qwen3:4b` | 8 GB | 35–50 tok/s |
| Chat / question answering | `qwen3:8b` | 16 GB | 14–22 tok/s |
| Code generation + review | `qwen3:14b` | 24 GB | 8–14 tok/s |
| Chain-of-thought reasoning | `deepseek-r1:7b` | 16 GB | 14–18 tok/s |
| Ultra-low RAM (< 8 GB) | `smollm2:1.7b` | 4 GB | 60–90 tok/s |

Pull any model with:
```bash
squish pull qwen3:8b       # ~4.4 GB download (pre-squished from HuggingFace)
squish pull deepseek-r1:7b # ~3.9 GB download
```

---

## Troubleshooting

### Server won't start

```bash
squish doctor      # checks all dependencies
squish doctor --report  # saves shareable JSON snapshot to ~/.squish/
```

### Continue can't connect

1. Verify the server is running: `curl http://localhost:11435/health`
2. Check `apiBase` ends in `/v1` (not `/v1/`)
3. Restart VS Code after editing `~/.continue/config.json`

### Slow first response (TTFT)

The first request after load includes model initialization. Subsequent requests are significantly faster due to the RadixTree prefix cache warming up. The `--agent` preset enables the Semantic Response Cache, which reduces TTFT by 40–60% on similar follow-up queries.

### Model selector shows nothing

The extension polls `/v1/models` — if the server is starting up, wait ~3 seconds and try again. Run `Squish: Select Model` again from `Ctrl+Shift+P`.

---

## Development: Extension Tests

```bash
cd extensions/vscode/squish-vscode
npm install
npm test          # runs 26 Jest tests in <5 seconds, no server required
```

All tests mock the HTTP layer — no live squish server needed for CI.
