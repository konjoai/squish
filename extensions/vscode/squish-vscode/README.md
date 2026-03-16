# Squish — Local AI Chat for VS Code

Chat with a locally running [Squish](https://github.com/squishai/squish) model directly from VS Code.
Tokens stream in real-time via the OpenAI-compatible API — no internet required, no data leaves your machine.

## Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3)
- [Squish](https://github.com/squishai/squish) installed and a model loaded:
  ```bash
  pip install squish
  squish run qwen3:8b   # starts the server on port 11435
  ```

## Features

| Feature | Description |
|---------|-------------|
| **Sidebar chat panel** | Click the flask icon in the activity bar to open the Squish chat view |
| **Real-time streaming** | Tokens appear as they are generated — no waiting for the full response |
| **Conversation history** | Multi-turn context is maintained per session; use **Clear History** to reset |
| **Model selector** | `Squish: Select Model` command picks from models available on the server |
| **Server lifecycle** | `Squish: Start Server` / `Squish: Stop Server` commands manage the daemon |
| **Status bar** | Shows server status and current model in the VS Code status bar |

## Quick Start

1. Start your Squish server (or let Auto Start do it):
   ```bash
   squish run qwen3:8b
   ```
2. Open VS Code → click the **flask icon** in the activity bar.
3. Type a message and press **Enter** (or click **Send**).

## Commands

| Command | Description |
|---------|-------------|
| `Squish: Open Chat Panel` | Show the chat sidebar |
| `Squish: Start Server` | Start `squish run` in the background |
| `Squish: Stop Server` | Kill the background server process |
| `Squish: Select Model` | Quick-pick from `/v1/models` |
| `Squish: Clear Chat History` | Reset conversation context |

## Settings

Configure via **Settings → Extensions → Squish** or `settings.json`:

| Setting | Default | Description |
|---------|---------|-------------|
| `squish.host` | `127.0.0.1` | Squish server hostname |
| `squish.port` | `11435` | Squish server port |
| `squish.apiKey` | `squish` | Bearer token for the API |
| `squish.model` | `7b` | Model to start when using Start Server |
| `squish.autoStart` | `false` | Auto-start server when VS Code opens |
| `squish.maxTokens` | `1024` | Maximum tokens per response |
| `squish.temperature` | `0.7` | Sampling temperature |
| `squish.systemPrompt` | *(empty)* | Custom system prompt injected at session start |

## Architecture

The extension runs entirely in the extension host process — no webview-to-server network calls.
The extension host fetches from `http://squish.host:squish.port` and relays streamed tokens to the
webview via `postMessage`, keeping the CSP policy strict (`default-src 'none'`).

## Development

```bash
cd extensions/vscode/squish-vscode
npm install
npm run compile   # tsc → out/
npm test          # jest unit tests (26 tests)
# Press F5 in VS Code to launch the Extension Development Host
```

## License

MIT — see [LICENSE](../../../LICENSE) in the squish repository root.
