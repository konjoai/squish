# SquishBar — macOS Menu Bar App

SquishBar is a lightweight macOS menu-bar application that lets you manage a
local Squish inference server from your status bar, without needing a terminal.

## Features

- **Live tok/s readout** in the menu bar (updates every 5 s)
- **Start / Stop** the Squish server with one click
- **Model picker** — switch between any model in `/v1/models` without restarting manually
- **Pull Model…** — download and compress a catalog model from the menu
- **Compression progress bar** — shows pull progress inline while a model downloads
- **Global hotkey** (default `⌘⌥S`) — open the Chat UI from any app (requires Accessibility permission)
- **Persistent settings** — host, port, API key, preferred model, and hotkey stored in `UserDefaults`
- **Fully offline** — no cloud services; talks only to `localhost:11435` by default

## Requirements

- macOS 13 Ventura or later (Apple Silicon or Intel)
- Xcode 15+ / Swift 5.9+ for building from source
- A working `squish` installation (`pip install squish`)

## Build from Source

```bash
cd apps/macos/SquishBar

# Debug build (fast, no optimizations):
swift build

# Release .app bundle:
make

# Ad-hoc–signed .app:
make release

# Distributable DMG:
make dmg

# Open immediately:
make run
```

The build outputs:
| File | Description |
|------|-------------|
| `SquishBar.app/` | macOS application bundle |
| `SquishBar.dmg` | Disk image for distribution (`make dmg`) |

## Usage

1. Run `make && make run` (or double-click `SquishBar.app`).
2. The SquishBar icon appears in your menu bar. Click it to open the panel.
3. Click **Start Server** to launch `squish run <model>` in the background.
4. Once the model loads, tok/s appears live next to the icon.
5. Use **Switch Model** to pick a different model — the server restarts automatically.
6. Use **Pull Model…** to download a new catalog model; a progress bar tracks the pull.

## Configuration

All settings persist across restarts via `UserDefaults`:

| Key (`squish.*`) | Default | Description |
|------------------|---------|-------------|
| `host` | `127.0.0.1` | Server hostname |
| `port` | `11435` | Server port |
| `apiKey` | `squish` | Bearer token for `/health` and `/v1/models` |
| `model` | `qwen3:8b` | Preferred model passed to `squish run` |
| `hotkey` | `⌘⌥S` | Global hotkey to open Chat UI |

Open **Settings…** in the menu to change any of these.

## Global Hotkey

The default hotkey `⌘⌥S` opens `http://localhost:11435/chat` in your browser
from any application.

On first launch, macOS prompts for **Accessibility** permission. Grant it in
**System Settings → Privacy & Security → Accessibility** to enable the hotkey.
If permission is not granted, the hotkey is silently disabled and everything
else continues to work normally.

## API Endpoints Used

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Polls server status, tok/s, model name, uptime |
| `GET /v1/models` | Fetches available model list for the picker |
| `GET /chat` | Opened in browser when Chat UI button is clicked |

## Distribution

To build a DMG for sharing:

```bash
make dmg
# → SquishBar.dmg (ad-hoc signed, drag-to-Applications layout)
```

Recipients can drag `SquishBar.app` to `/Applications` and right-click →
Open to bypass Gatekeeper on the first launch (standard for ad-hoc–signed apps).

## Troubleshooting

**Icon doesn't appear:**
Run `make run` from the `apps/macos/SquishBar/` directory and check the terminal
for Swift build errors.

**Server shows "offline" even when squish is running:**
Verify the port matches (`squish run` defaults to `11435`). Change it in
Settings… if you use a custom port.

**"squish: command not found" on start:**
Ensure `squish` is on your `$PATH`; alternatively place it at `~/.local/bin/squish`.
SquishBar checks both locations.

**Global hotkey not working:**
Go to **System Settings → Privacy & Security → Accessibility** and enable
SquishBar. Relaunch the app after granting permission.
