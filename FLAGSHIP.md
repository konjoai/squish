# Squish Flagship Roadmap — 根性

> Make every surface truly Konjo: beautiful, passionate, stripped to the essence,
> an experience of discovery. One local engine — four ways in — each a flagship.

This is the comprehensive feature list to rival Claude Desktop, OpenAI Codex,
Cursor, and Open Interpreter — built on a **local-only** engine. Tiers are
ordered by value-per-effort. ✅ shipped · 🟡 in progress · ⬜ planned.

## Surfaces
- **Web cockpit** — `squish/static/index.html` served at `/chat` (vanilla JS, no build)
- **VS Code** — `extensions/vscode/squish-vscode` ("Squish Agent")
- **macOS** — `apps/macos/SquishBar` (menu-bar + windows)
- **Backend** — `squish/server.py`, the shared agentic engine

---

## Tier 1 — Agent UX foundation (clean, live, in-context) ✅ SHIPPED

The agent should be a joy to *watch* and *trust*.

- ✅ **Clean agent stream (server-side).** The agent loop streams genuine
  reasoning text only — `ToolCallStreamFilter` suppresses `<tool_call>` syntax at
  the source, so all clients get clean text for free. *(squish/server.py,
  serving/tool_calling.py + unit tests)*
- ✅ **Beautiful tool cards (web + macOS).** Friendly tool name + icon, the key
  argument inline, live status (spinner → ✓/✗), elapsed time, and the actual
  **output** rendered readably — no raw JSON. Verified live in-browser. VS Code
  already streams tool calls/results into the chat.
- ✅ **VS Code: live editing of open files.** `apply_edit`/`write_file` via
  `WorkspaceEdit` — edits land in the open document, show in the gutter, are
  undoable, and reveal the change. *(+ tests)*
- ✅ **VS Code: auto workspace context.** Active file, selection, diagnostics, and
  open tabs injected into every agent turn (Cursor/Copilot parity). *(+ test)*
- ✅ **VS Code: agent reasoning + tool visibility in chat.** Tool calls + results
  stream into the chat panel.
- ✅ **Robustness.** String-encoded arguments coerced to dicts; a failing tool
  becomes a tool *error* the agent recovers from, never a dead stream. *(+ tests)*

## Tier 2 — Multimodal & files 🟡 IN PROGRESS

The agent should see what you see.

- ✅ **Document text extraction.** `squish_read_document` extracts text from PDF,
  DOCX (stdlib-only, no new dep), CSV, Markdown, JSON, code, plain text — so the
  agent can analyse any referenced file across every UI. *(+ tests, verified live)*
- 🟡 **File upload → context.** Web UI reads text attachments client-side today.
  *Remaining:* a binary-upload endpoint + macOS file picker/drag-drop.
- ⬜ **OpenAI-style content blocks in the chat API.** Accept
  `content: [{type:"text"}, {type:"image_url"}, …]` without breaking string
  content. *(squish/server.py)*
- ⬜ **Image / screenshot analysis (VLM).** Wire `mlx-vlm` + a vision model
  (gemma-3-4b-it) so the agent can actually analyze images. Reuses the existing
  multimodal radix KV cache. *(stretch — depends on mlx-vlm)*

## Tier 3 — Agent-driven web browser

The agent should be able to *use* the web, not just fetch it.

- ⬜ **macOS: built-in browser the agent drives.** A Browser tab/window
  (WKWebView + JS bridge) the agent can navigate, read, click, and fill via
  client-side tools (requires a client-side tool-execution channel).
- ⬜ **Web cockpit: agent browse panel.** Server-side headless/fetch-driven
  browsing surfaced live in the UI (navigate, read, extract), building on the
  existing `squish_fetch_url` / `squish_web_search` tools.

## Tier 4 — Konjo polish & robustness

건조 — strip to the essence. ቆንጆ — execute with beauty.

- ⬜ **Konjo aesthetic pass** across all UIs: motion, typography, restraint,
  discovery. Consistent design language.
- 🟡 **Comprehensive e2e tests**: `tests/e2e/test_agent_e2e.py` exercises chat +
  agent across short/medium/long/complex prompts plus robustness (empty input,
  50k-char pastes, tool-error recovery, clean-stream). Gated behind `SQUISH_E2E=1`
  so it runs against a live server without breaking CI. *Remaining:* per-UI
  webview/Swift harnesses.
- ⬜ **Slash commands & prompt library**, conversation export, model-aware
  defaults, keyboard-first navigation.

---

*Status is updated as tiers land. The repo is a gift to the next contributor —
human or AI. Mahiberawi Nuro.*
