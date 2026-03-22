# Phone Bridge

Control your Mac and GitHub Copilot from your phone via Telegram.

---

## How it works

```
Your phone
   │  Telegram message
   ▼
Telegram servers
   │  Bot API (HTTPS polling)
   ▼
phone_bridge/bot.py  ← running as a daemon on your Mac
   │  subprocess
   ▼
shell / gh-copilot-cli / git
```

The bot only accepts messages from user IDs you explicitly whitelist in `.env`.
All other senders receive no response — the bot does not reveal its existence.

---

## Prerequisites

| Tool | How to get it |
|------|---------------|
| Python 3.11+ | `brew install python` |
| `gh` CLI + Copilot extension | `brew install gh && gh extension install github/gh-copilot` |
| Telegram account | [telegram.org](https://telegram.org) |

---

## Setup

### 1 — Create a Telegram bot (one minute)

1. Open Telegram, search **@BotFather**, send `/newbot`.
2. Follow the prompts — choose any name and username.
3. Copy the token (format: `7123456789:AAFxyz...`).

### 2 — Find your Telegram user ID

Message **@userinfobot** on Telegram.  It replies instantly with your numeric ID (e.g. `123456789`).

### 3 — Configure the bridge

```bash
cd squish/tools/phone_bridge
cp .env.example .env
# Edit .env with your token, user ID, and preferred working directory
```

`.env` contents:
```
TELEGRAM_BOT_TOKEN=7123456789:AAFxyz...
ALLOWED_USER_IDS=123456789
WORKING_DIR=/Users/you/squish
```

> Add multiple user IDs separated by commas: `ALLOWED_USER_IDS=111111111,222222222`

### 4 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5 — Run the bot

```bash
python bot.py
```

To keep it running in the background (macOS):
```bash
nohup python bot.py >> ~/phone_bridge.log 2>&1 &
```

---

## Commands

| Command | What it does |
|---------|-------------|
| `/run <cmd>` | Execute a shell command on your Mac |
| `/ask <question>` | Ask GitHub Copilot CLI (`gh copilot suggest`) |
| `/file <path>` | Read a file and send its contents |
| `/ls [path]` | List a directory |
| `/cd <path>` | Change the session working directory |
| `/pwd` | Show current session working directory |
| `/git` | `git status` + last 10 commits |
| `/commit <msg>` | `git add -A && git commit -m && git push` |
| `/help` | Command reference |

**Plain text messages** (without a `/` prefix) are executed directly as shell commands.

---

## Examples

```
/run python -m pytest tests/ -x
/ask how do I zero-pad a number in bash
/file squish/server.py
/ls squish/kv
/cd squish
/git
/commit feat(kv): add radix cache eviction
```

---

## Security

- **User ID allowlist** — only Telegram accounts whose numeric ID appears in `ALLOWED_USER_IDS` can interact with the bot.  Telegram user IDs are permanent and cannot be spoofed.
- **Destructive command filter** — patterns like `rm -rf`, `dd`, `mkfs`, `sudo`, and writes to block devices are blocked before execution.
- **No web exposure** — the bot uses long-polling (outbound HTTPS to Telegram's servers).  No inbound port is opened on your Mac.
- **`.env` is gitignored** — your token and user ID never enter version control.

---

## Alternative messaging apps

| App | Difficulty | Notes |
|-----|-----------|-------|
| **Telegram** | Easy | Free bot API, no approval, used here |
| **WhatsApp** | Hard | Requires Meta Business API + phone number + approval, or use unofficial `whatsapp-web.js` (Node.js, may break with updates) |
| **Signal** | Medium | Use [`signal-cli`](https://github.com/AsamK/signal-cli) (Java) as a gateway; more complex setup but fully open-source |
| **Discord** | Easy | Similar to Telegram; use `discord.py`; replace the Telegram handler layer with Discord's |
| **Slack** | Medium | Needs a Slack workspace; use `slack_bolt`; good for team use |

The bot architecture (`bot.py`) is messaging-agnostic below the handler layer.  You can swap Telegram for any of the above by replacing the `Application` / `Update` / handler wiring while keeping all the helper functions (`is_dangerous`, `run_shell`, etc.) unchanged.

---

## Running tests

```bash
cd squish/tools/phone_bridge
pip install pytest
pytest test_bot.py -v
```

---

## Auto-start on login (macOS LaunchAgent)

Create `~/Library/LaunchAgents/com.squish.phone-bridge.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>com.squish.phone-bridge</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/python3</string>
    <string>/Users/you/squish/tools/phone_bridge/bot.py</string>
  </array>
  <key>WorkingDirectory</key>  <string>/Users/you/squish/tools/phone_bridge</string>
  <key>RunAtLoad</key>         <true/>
  <key>KeepAlive</key>         <true/>
  <key>StandardOutPath</key>   <string>/tmp/phone_bridge.log</string>
  <key>StandardErrorPath</key> <string>/tmp/phone_bridge.err</string>
</dict>
</plist>
```

Then load it:
```bash
launchctl load ~/Library/LaunchAgents/com.squish.phone-bridge.plist
```
