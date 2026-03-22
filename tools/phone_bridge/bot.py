#!/usr/bin/env python3
"""
phone_bridge/bot.py — Control your Mac and GitHub Copilot from Telegram.

Setup (one-time):
  1. Open Telegram → search @BotFather → /newbot → copy the token.
  2. Message @userinfobot on Telegram to get your numeric user ID.
  3. cp .env.example .env  and fill in the two required values.
  4. pip install -r requirements.txt
  5. python bot.py

Available commands from your phone:
  /run <cmd>      — run a shell command on your Mac
  /ask <question> — ask GitHub Copilot CLI (requires: gh extension install github/gh-copilot)
  /file <path>    — read a file and send its contents
  /ls [path]      — list a directory
  /cd <path>      — change working directory for this session
  /pwd            — show current working directory
  /git            — git status + recent log
  /commit <msg>   — git add -A && commit && push
  /help           — show this list

Plain text messages (no slash) are executed as shell commands.

Security model:
  - ALLOWED_USER_IDS is an allowlist of Telegram numeric user IDs.
  - All other users receive no response (not even an error).
  - Destructive shell patterns are blocked before execution.
  - Shell commands run with shell=True, scoped to the session cwd;
    this is intentional and safe because only your whitelisted user IDs
    can reach this code path.
"""

from __future__ import annotations

import html
import logging
import os
import re
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── environment ───────────────────────────────────────────────────────────────

load_dotenv()

_BOT_TOKEN: str = os.environ["TELEGRAM_BOT_TOKEN"]
_ALLOWED_USER_IDS: frozenset[int] = frozenset(
    int(uid.strip())
    for uid in os.environ["ALLOWED_USER_IDS"].split(",")
    if uid.strip()
)
_DEFAULT_CWD: Path = Path(
    os.environ.get("WORKING_DIR", str(Path.home()))
).expanduser().resolve()

# ── module-level state ────────────────────────────────────────────────────────

# Per Telegram user-ID session cwd; cleared when the bot restarts.
_session_cwd: dict[int, Path] = {}

# ── constants ─────────────────────────────────────────────────────────────────

MAX_MESSAGE_CHARS: int = 3800  # Telegram hard limit is 4096; leave headroom
COMMAND_TIMEOUT_S: int = 30
COPILOT_TIMEOUT_S: int = 60

# Block commands that could irrecoverably destroy data or escalate privileges.
_DANGEROUS_RE = re.compile(
    r"""
      \brm\s+-[^\s]*r          # rm -r* (recursive remove)
    | \bdd\b                   # dd (raw disk write)
    | \bmkfs\b                 # mkfs (format)
    | >\s*/dev/sd              # redirect to raw block device
    | >\s*/dev/disk            # redirect to raw disk (macOS)
    | :>\s*/                   # truncate root paths
    | \bsudo\b                 # sudo escalation
    | \bchmod\s+[0-7][0-7][2367]\b   # world-writable (others write bit set)
    | \bshred\b                # shred (secure delete)
    | \bwipe\b                 # wipe
    """,
    re.IGNORECASE | re.VERBOSE,
)

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────


def is_authorized(update: Update) -> bool:
    """Return True only if the sender is in the allowlist."""
    user = update.effective_user
    return user is not None and user.id in _ALLOWED_USER_IDS


def get_cwd(user_id: int) -> Path:
    """Return the session working directory for *user_id*."""
    return _session_cwd.get(user_id, _DEFAULT_CWD)


def truncate(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Trim *text* to *limit* characters, preserving head and tail context."""
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return text[:half] + f"\n…[{omitted} chars omitted]…\n" + text[-half:]


def is_dangerous(command: str) -> bool:
    """Return True if *command* matches a known destructive pattern."""
    return bool(_DANGEROUS_RE.search(command))


def run_shell(command: str, cwd: Path, timeout: int = COMMAND_TIMEOUT_S) -> tuple[str, int]:
    """
    Execute *command* in a shell at *cwd*.

    Returns (combined stdout+stderr, returncode).
    Raises subprocess.TimeoutExpired if the command exceeds *timeout* seconds.
    """
    result = subprocess.run(
        command,
        shell=True,  # noqa: S602 — intentional; callers enforce the auth guard
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = (result.stdout + result.stderr).strip()
    return combined or "(no output)", result.returncode


def format_pre(text: str) -> str:
    """Wrap *text* in an HTML <pre> block safe for Telegram."""
    return f"<pre>{html.escape(truncate(text))}</pre>"


async def reply_pre(update: Update, text: str) -> None:
    """Send *text* as a monospace block reply."""
    await update.message.reply_text(format_pre(text), parse_mode=ParseMode.HTML)


async def require_auth(update: Update) -> bool:
    """
    Silently drop the update if the sender is not authorised.

    Returns True when the caller may proceed, False when it must stop.
    """
    if not is_authorized(update):
        logger.warning(
            "Rejected message from user_id=%s username=%s",
            update.effective_user and update.effective_user.id,
            update.effective_user and update.effective_user.username,
        )
        # No reply — do not reveal the bot exists to strangers.
        return False
    return True


# ── command handlers ──────────────────────────────────────────────────────────


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send the command reference."""
    if not await require_auth(update):
        return
    lines = [
        "<b>Phone Bridge — commands</b>",
        "",
        "/run &lt;cmd&gt;       — shell command",
        "/ask &lt;question&gt;  — GitHub Copilot CLI",
        "/file &lt;path&gt;     — read file",
        "/ls [path]       — list directory",
        "/cd &lt;path&gt;      — change session cwd",
        "/pwd             — show session cwd",
        "/git             — status + recent log",
        "/commit &lt;msg&gt;   — git add -A, commit, push",
        "/help            — this message",
        "",
        "Plain text → runs as a shell command.",
    ]
    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.HTML)


async def cmd_run(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute an arbitrary shell command."""
    if not await require_auth(update):
        return
    command = " ".join(context.args or [])
    if not command:
        await update.message.reply_text("Usage: /run &lt;command&gt;", parse_mode=ParseMode.HTML)
        return
    if is_dangerous(command):
        await reply_pre(update, f"BLOCKED — destructive pattern detected.\n$ {command}")
        return
    cwd = get_cwd(update.effective_user.id)
    try:
        output, rc = run_shell(command, cwd)
    except subprocess.TimeoutExpired:
        output, rc = f"(timed out after {COMMAND_TIMEOUT_S} s)", 124
    await reply_pre(update, f"$ {command}\n[exit {rc}]\n\n{output}")


async def cmd_ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Ask GitHub Copilot CLI a question."""
    if not await require_auth(update):
        return
    question = " ".join(context.args or [])
    if not question:
        await update.message.reply_text(
            "Usage: /ask &lt;question&gt;\n"
            "Requires: <code>gh extension install github/gh-copilot</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    cwd = get_cwd(update.effective_user.id)
    # Try `gh copilot suggest` first; fall back to `gh copilot explain`.
    for gh_cmd in (
        f'gh copilot suggest -t shell "{question}"',
        f'gh copilot explain "{question}"',
    ):
        try:
            output, rc = run_shell(gh_cmd, cwd, timeout=COPILOT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            await reply_pre(update, f"(gh copilot timed out after {COPILOT_TIMEOUT_S} s)")
            return
        if rc == 0:
            break
    await reply_pre(update, output)


async def cmd_file(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Read a file and return its contents."""
    if not await require_auth(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /file &lt;path&gt;", parse_mode=ParseMode.HTML)
        return
    path = Path(context.args[0]).expanduser()
    if not path.is_absolute():
        path = get_cwd(update.effective_user.id) / path
    path = path.resolve()
    try:
        content = path.read_text(errors="replace")
    except (OSError, PermissionError) as exc:
        await update.message.reply_text(f"Error: {html.escape(str(exc))}", parse_mode=ParseMode.HTML)
        return
    await reply_pre(update, f"{path}\n\n{content}")


async def cmd_ls(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """List a directory."""
    if not await require_auth(update):
        return
    if context.args:
        target = Path(context.args[0]).expanduser()
        if not target.is_absolute():
            target = get_cwd(update.effective_user.id) / target
    else:
        target = get_cwd(update.effective_user.id)
    target = target.resolve()
    try:
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        lines = [f"{'d' if e.is_dir() else 'f'}  {e.name}" for e in entries]
        output = "\n".join(lines) or "(empty directory)"
    except (OSError, PermissionError) as exc:
        output = str(exc)
    await reply_pre(update, f"{target}/\n\n{output}")


async def cmd_cd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Change the session working directory."""
    if not await require_auth(update):
        return
    if not context.args:
        await update.message.reply_text("Usage: /cd &lt;path&gt;", parse_mode=ParseMode.HTML)
        return
    path = Path(context.args[0]).expanduser()
    if not path.is_absolute():
        path = get_cwd(update.effective_user.id) / path
    path = path.resolve()
    if not path.is_dir():
        await update.message.reply_text(
            f"Not a directory: <code>{html.escape(str(path))}</code>",
            parse_mode=ParseMode.HTML,
        )
        return
    _session_cwd[update.effective_user.id] = path
    await update.message.reply_text(
        f"cwd → <code>{html.escape(str(path))}</code>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_pwd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the session working directory."""
    if not await require_auth(update):
        return
    cwd = get_cwd(update.effective_user.id)
    await update.message.reply_text(
        f"<code>{html.escape(str(cwd))}</code>",
        parse_mode=ParseMode.HTML,
    )


async def cmd_git(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show git status and recent log."""
    if not await require_auth(update):
        return
    cwd = get_cwd(update.effective_user.id)
    status, _ = run_shell("git status --short --branch", cwd)
    log, _ = run_shell("git log --oneline -10", cwd)
    await reply_pre(update, f"STATUS\n{status}\n\nRECENT LOG\n{log}")


async def cmd_commit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Stage all changes, commit, and push."""
    if not await require_auth(update):
        return
    message = " ".join(context.args or [])
    if not message:
        await update.message.reply_text("Usage: /commit &lt;message&gt;", parse_mode=ParseMode.HTML)
        return
    cwd = get_cwd(update.effective_user.id)
    add_out, _ = run_shell("git add -A", cwd)
    # Use -- to avoid interpreting message as a git flag.
    commit_out, commit_rc = run_shell(f"git commit -m -- {message!r}", cwd)
    if commit_rc != 0:
        await reply_pre(update, f"git commit failed\n\n{add_out}\n{commit_out}")
        return
    push_out, push_rc = run_shell("git push", cwd)
    status = "pushed" if push_rc == 0 else f"push failed (exit {push_rc})"
    await reply_pre(update, f"{commit_out}\n\n{push_out}\n\n[{status}]")


async def handle_plain_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Treat plain (non-command) messages as shell commands."""
    if not await require_auth(update):
        return
    command = (update.message.text or "").strip()
    if not command:
        return
    if is_dangerous(command):
        await reply_pre(update, f"BLOCKED — destructive pattern detected.\n$ {command}")
        return
    cwd = get_cwd(update.effective_user.id)
    try:
        output, rc = run_shell(command, cwd)
    except subprocess.TimeoutExpired:
        output, rc = f"(timed out after {COMMAND_TIMEOUT_S} s)", 124
    await reply_pre(update, f"$ {command}\n[exit {rc}]\n\n{output}")


# ── entry point ───────────────────────────────────────────────────────────────


def build_app() -> Application:
    """Construct and return the bot Application (does not start polling)."""
    app = Application.builder().token(_BOT_TOKEN).build()
    app.add_handler(CommandHandler(["start", "help"], cmd_help))
    app.add_handler(CommandHandler("run", cmd_run))
    app.add_handler(CommandHandler("ask", cmd_ask))
    app.add_handler(CommandHandler("file", cmd_file))
    app.add_handler(CommandHandler("ls", cmd_ls))
    app.add_handler(CommandHandler("cd", cmd_cd))
    app.add_handler(CommandHandler("pwd", cmd_pwd))
    app.add_handler(CommandHandler("git", cmd_git))
    app.add_handler(CommandHandler("commit", cmd_commit))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_text))
    return app


def main() -> None:
    logger.info("Starting Phone Bridge.")
    logger.info("Allowed user IDs: %s", sorted(_ALLOWED_USER_IDS))
    logger.info("Default working dir: %s", _DEFAULT_CWD)
    build_app().run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
