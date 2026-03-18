# Squish CLI Revamp Plan
> Created: 2026-03-17

**Goal:** `squish run qwen3:8b` does everything. No required flags. Ever.

---

## The Problem

Current pain points that drive users away:

| Pain | Example |
|------|---------|
| Required flags | `convert-model` needs `--source-path` and `--output-path` — both required, no defaults |
| No auto-pipeline | Download ≠ compress. User must run 2–3 commands to get started |
| Confusing command names | `it`, `pull-head`, `convert-model` aren't discoverable |
| Flag explosion | `squish run` has 23 flags — help output is overwhelming |
| No TUI feedback | Long operations (compress, pull) show no progress |
| Duplicate commands | `run` and `serve` are identical (~95 lines duplicated) |
| No stored config | WhatsApp/Signal tokens required every run |
| Broken help text | `convert-model` suggests `--mlx-model-dir` flag that doesn't exist |

---

## The Ideal UX

```bash
# Install
pip install squish

# That's it — one command, everything happens automatically
squish run qwen3:8b
# ✓ Detects 16 GB RAM → qwen3:8b is a good fit
# ⬇  Downloading Qwen3-8B  [████████████] 15.2 GB  4.2 MB/s
# 🔧  Compressing to INT4   [████████████] 36 layers  done in 4m
# 🚀  Server ready at http://127.0.0.1:11435

# First time with no args — meets you where you are
squish
# Welcome to Squish! Detected 16 GB RAM.
# Recommended: qwen3:8b (fits in ~10 GB compressed)
# Run `squish run qwen3:8b` to get started.

# Chat instantly
squish chat

# See what's available
squish catalog        # rich table with tag badges
squish models         # local models with disk usage

# Compress an existing model (no paths needed)
squish compress       # interactive picker of local models
squish compress qwen3:8b

# Fine-tune
squish train qwen3:8b --dataset my_data.jsonl

# Messaging (prompts for tokens once, saves them)
squish run qwen3:8b --whatsapp
```

---

## Implementation Phases

### Phase 1 — Foundation  `squish/ui.py`
1. Add `rich` dependency to `pyproject.toml` and `requirements.txt`
2. Create `squish/ui.py` — shared TUI helpers:
   - `console` — global Rich Console instance
   - `banner()` — welcome screen with ASCII art + version
   - `spinner(msg)` — context manager wrapping Rich spinner
   - `progress(desc, total)` — download/compress progress bar
   - `model_picker(models)` — interactive list picker (arrows + enter)
   - `confirm(msg)` — y/n prompt with default
   - `success(msg)` / `warn(msg)` / `error(msg)` + `hint(msg)` — consistent styled output

### Phase 2 — `squish run` as the one-liner
3. **Auto-pipeline in `squish run`:**
   - If model not found locally → auto-pull with progress bar (no separate `squish pull` needed)
   - If model not compressed → auto-compress inline with spinner
   - If model ready → start server
   - `model` arg defaults to RAM-matched recommendation (already implemented — wire it up to the auto-pull)
4. **Merge `run`/`serve` duplication** — single `_cmd_run_or_serve(args)` function; `serve` registers the same function. Keeps both names working.
5. **Two-tier `--help`:**
   - Default help shows only: `[model]`, `--port`, `--agent`, `--whatsapp`, `--signal`, `--host`
   - `squish run --advanced` shows remaining 18 flags

### Phase 3 — Command renames + simplification
6. **`it` → `compress`** — keep `it` as hidden deprecated alias for 2 releases
   - No required args: `squish compress` → shows model picker of local BF16 models
   - `squish compress qwen3:8b` → resolves from catalog, auto-derives output as `<name>-squished`
   - `--output` optional override
7. **`convert-model` → `quantize`** — keep `convert-model` as deprecated alias
   - No required args: `squish quantize` → interactive picker
   - Auto-derives `--output-path` as `<source>-q<ffn_bits>bit`
   - Fix broken help text (`--mlx-model-dir` → correct path)
8. **`pull` auto-compresses** — after download, automatically runs `squish compress` unless `--no-compress` is passed
9. **`squish models` TUI** — rich table:
   ```
   Model              BF16     Compressed   Status      Last used
   qwen3:8b           15 GB    10.1 GB      ✅ ready    2 hours ago
   qwen2.5:1.5b        3 GB     2.0 GB      ✅ ready    yesterday
   llama3.1:8b        16 GB      —          ↓ download  —
   ```
10. **`squish catalog` TUI** — searchable rich table with badges:
    - 🔥 fast  🧠 reasoning  📦 small  🔬 moe  ⚡ prebuilt available

### Phase 4 — First-run onboarding
11. **`squish` with no args:**
    - If no local models and no server running → welcome banner + RAM recommendation + "Run `squish run <recommendation>` to get started"
    - If server running → show status (model, uptime, tok/s)
    - If local models but no server → show model list + "Run `squish chat` to start"
12. **`squish setup` polish:**
    - Rich prompts replacing plain input()
    - Spinner during pull + compress
    - Ends: "🎉 You're ready! Run: `squish chat qwen3:8b`"
13. **`squish doctor` polish:**
    - Coloured ✅ / ⚠️ / ❌ per check
    - Compact single-line per check (not wall of text)
    - `--fix` flag: auto-installs missing deps where possible

### Phase 5 — Config system
14. **`~/.squish/config.json`** — first-class config file
15. **`squish config` subcommand:**
    ```bash
    squish config show          # print current config
    squish config set key value # set a value
    squish config get key       # get a value
    ```
    Keys: `default_model`, `port`, `host`, `whatsapp.*`, `signal.*`
16. **WhatsApp/Signal first-use prompt:**
    - `squish run qwen3:8b --whatsapp` → if tokens missing, walks through 4-question setup, saves to `~/.squish/config.json`
    - Subsequent runs read from config automatically

### Phase 6 — Test + commit
17. Update tests in `tests/test_cli*.py` for renamed commands + new default behaviours
18. Update `README.md` quickstart section
19. `git commit -m "feat(cli): TUI revamp — rich progress, auto-pipeline, simplified commands"`

---

## Command Mapping (old → new)

| Old | New | Notes |
|-----|-----|-------|
| `squish it <model>` | `squish compress [model]` | `it` kept as hidden alias |
| `squish convert-model --source-path X --output-path Y` | `squish quantize [model]` | paths auto-derived; `convert-model` kept as alias |
| `squish pull <model>` | unchanged | now also auto-compresses after download |
| `squish serve` | unchanged | implementation merged with `run` |
| `squish pull-head` | unchanged | niche enough to keep as-is |
| `squish train-adapter` | `squish train` | `train-adapter` kept as alias |
| `squish merge-model` | `squish merge` | `merge-model` kept as alias |

---

## Files to Change

| File | Change |
|------|--------|
| `squish/ui.py` | **NEW** — rich helpers module |
| `squish/cli.py` | All phased changes above |
| `squish/catalog.py` | No changes needed |
| `pyproject.toml` | Add `rich>=13.0` dependency |
| `requirements.txt` | Add `rich>=13.0` |
| `tests/test_cli*.py` | Update for renamed commands |
| `README.md` | Update quickstart |
| `extensions/vscode/squish-vscode/src/serverManager.ts` | No changes needed |

---

## Out of Scope
- No browser/web UI changes
- No WhatsApp/Signal backend changes (only credential UX)
- ngrok / VPN / tunnel deferred (decision pending)
- No gRPC/REST catalog server changes
