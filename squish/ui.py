"""
squish/ui.py

Shared TUI helpers for the Squish CLI.

Uses `rich` for progress bars, spinners, styled output, and interactive
prompts.  Falls back to plain ASCII output when rich is not installed so
the CLI remains usable in a minimal install environment.

Public API
──────────
  console                   — global Rich Console instance
  banner()                  — welcome screen with mascot logo + version
  logo_image()              — render the squish mascot (PNG via Pillow, else ASCII art)
  spinner(msg)              — context manager wrapping a Rich spinner
  progress(desc, total)     — download/compress progress bar context manager
  model_picker(models)      — interactive list picker (arrow keys + enter)
  confirm(msg, default)     — y/n prompt with default answer
  success(msg)              — styled ✅ message
  warn(msg)                 — styled ⚠  message
  error(msg)                — styled ✗  message
  hint(msg)                 — styled dim hint line
  status_badge(online)      — ● ONLINE / ● OFFLINE Rich string
  quant_badge(quant)        — coloured INT4/INT3/INT2/INT8 badge string
  startup_panel(...)        — styled server startup info panel
  server_status_panel(...)  — styled squish ps panel
  chat_header(...)          — styled squish chat session header
  panel(lines, title)       — generic violet-bordered info panel
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Generator, Sequence

# ── Rich availability check ──────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )
    from rich.prompt import Confirm
    from rich.rule import Rule
    from rich.table import Table
    from rich.theme import Theme
    from rich import box as _rich_box
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


# ── Squish brand colours ─────────────────────────────────────────────────────

# 256-colour indexed values — standardised across all terminals, downscale
# gracefully to 16-colour ANSI on older hosts.  Using 256-colour (not 24-bit
# RGB) is the CONSISTENT choice: the same index number always maps to the same
# physical colour on every terminal that supports 256 colours (≈ 99% of modern
# terminals).  24-bit RGB can be remapped or mis-rendered by tmux, SSH
# forwarding, or custom colour profiles, producing the coloured chaos shown in
# the screenshot.
_SQUISH_THEME = Theme({
    "squish.purple":  "color(93)",    # #8700ff deep violet
    "squish.violet":  "color(135)",   # #af5fff medium purple
    "squish.lilac":   "color(183)",   # #d7afff plum / lavender
    "squish.pink":    "color(205)",   # #ff5faf hot pink
    "squish.teal":    "color(45)",    # #00d7ff turquoise
    "squish.green":   "color(78)",    # #5fd787 sea green
    "squish.white":   "default",      # honours the terminal's own default text
    "squish.dim":     "color(241)",   # #626262 dim grey
    "squish.warn":    "color(220)",   # #ffd700 gold
    "squish.error":   "color(203)",   # #ff5f5f soft red
})

if _RICH_AVAILABLE:
    # Force 256-colour mode.  This is more portable than 24-bit truecolor:
    # terminals that lie about truecolor support, or multiplex through tmux
    # without tc passthrough, will still render 256-colour indices correctly.
    console = Console(theme=_SQUISH_THEME, highlight=False, color_system="256")
else:  # pragma: no cover
    # Minimal fallback object that supports .print() / .rule()
    class _FallbackConsole:  # type: ignore[no-redef]
        def print(self, *args, **kwargs) -> None:
            text = " ".join(str(a) for a in args)
            # Strip Rich markup like [bold] tags
            import re
            text = re.sub(r'\[/?[^\]]*\]', '', text)
            print(text)

        def rule(self, title: str = "", **kwargs) -> None:
            width = 60
            if title:
                pad = (width - len(title) - 2) // 2
                print(f"{'─' * pad} {title} {'─' * pad}")
            else:
                print("─" * width)

    console = _FallbackConsole()  # type: ignore[assignment]


# ── Mascot logo ──────────────────────────────────────────────────────────────
#
# Hand-crafted Unicode block-art of the Squish kawaii blob mascot.
# Matches assets/squish-logo-1.png: rounded purple body, big eyes, blush,
# little arms and legs, sparkles.  Uses half-block (▄) and braille-dots
# for the pupils + highlights so it renders cleanly in any monospace font.
#
#  Colour key (brand palette):
#    V = violet #8B5CF6  P = purple #7C3AED  L = lilac #A78BFA
#    PK = pink #EC4899   W = white #F8FAFC   DIM = slate #64748B
#    G = green #34D399   T = teal #22D3EE
#
_MASCOT_LINES: list[tuple[str, ...]] = [
    # Each tuple: (escape_code_or_reset, text_fragment, ...)
    # Pre-rendered as a list of Rich markup strings for easy console.print.
    ("[squish.dim]",         "          ✦       ✦           "),
    ("[squish.pink]",        "      ✦                   ✦   "),
    ("[squish.violet]",      "      ╭───────────────────╮   "),
    ("[squish.violet]",      "    ╭─╯                   ╰─╮ "),
    ("[squish.violet]",      "   ╭╯  "),
    ("[squish.white]",       "◉ "),
    ("[squish.violet]",      "               "),
    ("[squish.white]",       "◉ "),
    ("[squish.violet]",      " ╰╮"),
    ("[squish.violet]",      "   │  "),
    ("[squish.white]",       "● "),
    ("[squish.violet]",      "  "),
    ("[squish.pink]",        "◡◡◡"),
    ("[squish.violet]",      "  "),
    ("[squish.white]",       "● "),
    ("[squish.violet]",      "  │ "),
    ("[squish.violet]",      "  ╭╯  "),
    ("[squish.lilac]",       "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"),
    ("[squish.violet]",      "  ╰╮"),
    ("[squish.violet]",      " ╭─╯  "),
    ("[squish.lilac]",       "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"),
    ("[squish.violet]",      "  ╰─╮"),
    ("[squish.violet]",      " │    "),
    ("[squish.lilac]",       "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"),
    ("[squish.violet]",      "    │"),
    ("[squish.violet]",      " ╰─╮  "),
    ("[squish.lilac]",       "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓"),
    ("[squish.violet]",      "  ╭─╯"),
    ("[squish.violet]",      "   ╰─────────────────────╯  "),
    ("[squish.violet]",      "      ╰──╯    ╰────╯    ╰──╯ "),
    ("[squish.dim]",         "  ✦             ✦        ✦   "),
]

# Compact single-colour fallback for terminals without Rich
_ASCII_MASCOT = r'''
       .-"""""""-.
      /  O     O  \
     |  (  ~~~  )  |
     |   \     /   |
      \   '---'   /
       '-._____.-'
      /  |     |  \
     '   '     '   ' '''

# Legacy text logo kept for --no-color / pipe contexts
_ASCII_LOGO = r"""
  ____  ___  _   _ ___ ____  _   _
 / ___||_ _|| | | |_ _/ ___|| | | |
 \___ \ | | | | | || |\___ \| |_| |
  ___) || | | |_| || | ___) |  _  |
 |____/|___| \___/|___|____/|_| |_|"""


def logo_image() -> None:
    """Render the Squish mascot to the terminal.

    Uses Rich markup with 256-colour indexed theme values so the mascot
    renders consistently on every modern terminal.  Falls back to plain ASCII
    when Rich is unavailable or output is not a TTY.

    NOTE: Pillow / PNG half-block rendering has been intentionally removed.
    The source PNG contains the full wordmark + background, which produces
    unreadable coloured noise when rasterised to terminal half-blocks.
    """
    # ── Rich markup mascot (primary — works on all 256-colour terminals) ──────
    if _RICH_AVAILABLE:
        _mascot_rows = [
            "  [squish.dim]✦[/]          [squish.pink]✦[/]     [squish.dim]✦[/]",
            "    [squish.pink]✦[/]                      [squish.pink]✦[/]",
            "    [squish.violet]╭───────────────────────╮[/]",
            "  [squish.violet]╭─╯[/]                       [squish.violet]╰─╮[/]",
            "  [squish.violet]│[/]  [squish.white]◉[/][squish.violet]        [/][squish.pink]▄▄▄[/][squish.violet]        [/][squish.white]◉[/]  [squish.violet]│[/]",
            "  [squish.violet]│[/]  [squish.white]●[/][squish.violet]   [/][squish.pink]▗▄▄▄▄▄▄▄▄▄▗[/][squish.violet]   [/][squish.white]●[/]  [squish.violet]│[/]",
            "  [squish.violet]╰╮[/][squish.lilac]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[/][squish.violet]╭╯[/]",
            "  [squish.violet]╭╯[/][squish.lilac]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[/][squish.violet]╰╮[/]",
            "  [squish.violet]│[/][squish.lilac]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[/][squish.violet]│[/]",
            "  [squish.violet]╰─╮[/][squish.lilac]▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓[/][squish.violet]╭─╯[/]",
            "    [squish.violet]╰─────────────────────╯[/]",
            "       [squish.violet]╰──╯[/]   [squish.violet]╰────╯[/]   [squish.violet]╰──╯[/]",
            "  [squish.dim]✦[/]               [squish.dim]✦[/]       [squish.pink]✦[/]",
        ]
        for _row in _mascot_rows:
            console.print(_row)
        return

    # ── Plain ASCII fallback (no-color / piped output) ────────────────────────
    print(_ASCII_MASCOT)  # pragma: no cover


def banner() -> None:
    """Print the Squish welcome banner: mascot + gradient wordmark + version rule."""
    try:
        from squish import __version__ as _ver
    except Exception:
        _ver = "9.0.0"

    if _RICH_AVAILABLE:
        logo_image()
        console.print()
        console.print(
            "  [bold squish.violet]squish[/]  [squish.dim]— private local inference[/]"
        )
        console.rule(f"[squish.dim]v{_ver}[/squish.dim]")
        console.print()
    else:  # pragma: no cover
        print(_ASCII_MASCOT)
        print(f"  squish v{_ver}")
        print()


# ── Spinner context manager ───────────────────────────────────────────────────

@contextlib.contextmanager
def spinner(msg: str) -> Generator[None, None, None]:
    """
    Context manager that shows a spinner while work is in progress.

    Usage::

        with spinner("Compressing model"):
            do_heavy_work()
    """
    if _RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(style="squish.violet"),
            TextColumn("[squish.white]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as prog:
            prog.add_task(msg)
            yield
    else:  # pragma: no cover
        print(f"  {msg}…", end=" ", flush=True)
        yield
        print("done.")


# ── Download / compress progress bar ─────────────────────────────────────────

@contextlib.contextmanager
def progress(desc: str, total: int | None = None) -> Generator["_ProgressHandle", None, None]:
    """
    Context manager yielding a progress handle with an ``update(n)`` method.

    ``total`` is in bytes for downloads or steps for compress.  Passing
    ``None`` shows an indeterminate bar.

    Usage::

        with progress("Downloading qwen3:8b", total=file_size) as bar:
            for chunk in stream:
                bar.update(len(chunk))
    """
    if _RICH_AVAILABLE:
        columns = [
            SpinnerColumn(style="squish.violet"),
            TextColumn("[squish.white]{task.description}"),
            BarColumn(bar_width=40, style="squish.purple", complete_style="squish.violet"),
            TaskProgressColumn(),
        ]
        if total is not None:
            columns += [DownloadColumn(), TransferSpeedColumn(), TimeRemainingColumn()]
        else:
            columns.append(TimeElapsedColumn())

        with Progress(*columns, console=console) as prog:
            task_id = prog.add_task(desc, total=total)
            yield _ProgressHandle(lambda n: prog.advance(task_id, n))
    else:  # pragma: no cover
        print(f"  {desc}…", flush=True)
        yield _ProgressHandle(lambda n: None)


class _ProgressHandle:
    """Thin wrapper returned by the ``progress`` context manager."""

    def __init__(self, advance_fn):
        self._advance = advance_fn

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by ``n`` units."""
        self._advance(n)


# ── Interactive model picker ──────────────────────────────────────────────────

def model_picker(models: Sequence[str], prompt: str = "Select a model") -> str | None:
    """
    Display an interactive list and return the selected model name.

    Falls back to a plain numbered list prompt when rich is not available or
    stdin is not a TTY.

    Returns ``None`` if the user cancels (Ctrl-C / empty input on fallback).
    """
    if not models:
        return None

    if _RICH_AVAILABLE and sys.stdin.isatty():
        try:
            # Use questionary if available for arrow-key navigation
            import questionary  # type: ignore[import]
            return questionary.select(prompt, choices=list(models)).ask()
        except ImportError:
            pass

    # Numbered fallback
    print(f"\n  {prompt}:")
    for i, m in enumerate(models, start=1):
        print(f"    {i}. {m}")
    print()
    raw = input("  Enter number (or press Enter to cancel): ").strip()
    if not raw:
        return None
    try:
        idx = int(raw) - 1
        if 0 <= idx < len(models):
            return models[idx]
    except ValueError:
        pass
    return None


# ── y/n confirm ───────────────────────────────────────────────────────────────

def confirm(msg: str, default: bool = True) -> bool:
    """
    Prompt the user for a yes/no answer.

    Returns ``True`` for yes, ``False`` for no.  On non-interactive stdin
    returns the ``default``.
    """
    if not sys.stdin.isatty():
        return default

    if _RICH_AVAILABLE:
        return Confirm.ask(f"[squish.white]{msg}[/squish.white]", default=default)

    # Plain fallback
    hint = " [Y/n]" if default else " [y/N]"
    raw = input(f"  {msg}{hint}: ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


# ── Styled output helpers ─────────────────────────────────────────────────────

def success(msg: str) -> None:
    """Print a success message with a green check mark."""
    if _RICH_AVAILABLE:
        console.print(f"  [squish.green]✓[/squish.green]  [squish.white]{msg}[/squish.white]")
    else:  # pragma: no cover
        print(f"  ✓  {msg}")


def warn(msg: str) -> None:
    """Print a warning message with a yellow caution symbol."""
    if _RICH_AVAILABLE:
        console.print(f"  [squish.warn]⚠ [/squish.warn]  [squish.white]{msg}[/squish.white]")
    else:  # pragma: no cover
        print(f"  ⚠  {msg}", file=sys.stderr)


def error(msg: str) -> None:
    """Print an error message with a red cross."""
    if _RICH_AVAILABLE:
        console.print(f"  [squish.error]✗[/squish.error]  [squish.white]{msg}[/squish.white]")
    else:  # pragma: no cover
        print(f"  ✗  {msg}", file=sys.stderr)


def hint(msg: str) -> None:
    """Print a dim hint / suggestion line."""
    if _RICH_AVAILABLE:
        console.print(f"  [squish.dim]{msg}[/squish.dim]")
    else:  # pragma: no cover
        print(f"  {msg}")


def header(title: str, subtitle: str | None = None) -> None:
    """Print a branded section panel header.

    Uses a Rich Panel with rounded corners and violet border when Rich is
    available; falls back to a plain Unicode box otherwise.

    Parameters
    ----------
    title:    Primary heading text.
    subtitle: Optional secondary line shown below the title.
    """
    if _RICH_AVAILABLE:
        if subtitle:
            body = (
                f"[squish.white bold]{title}[/]\n"
                f"[squish.dim]{subtitle}[/]"
            )
        else:
            body = f"[squish.white bold]{title}[/]"
        console.print(
            Panel(body, box=_rich_box.ROUNDED, border_style="squish.violet", padding=(0, 2))
        )
    else:  # pragma: no cover
        lines = [title] + ([subtitle] if subtitle else [])
        width = max(len(ln) for ln in lines) + 4
        print(f"╭{'─' * width}╮")
        for ln in lines:
            print(f"│  {ln}{' ' * (width - len(ln) - 2)}│")
        print(f"╰{'─' * width}╯")


# ── Rich table helpers ────────────────────────────────────────────────────────

def make_table(columns: Sequence[str], title: str | None = None) -> "Table | None":
    """
    Create and return a Rich Table with squish brand styling.

    Returns ``None`` when rich is not available (caller should fall back to
    plain print).
    """
    if not _RICH_AVAILABLE:  # pragma: no cover
        return None
    tbl = Table(
        title=title,
        box=_rich_box.ROUNDED,
        header_style="color(135) bold",
        border_style="squish.dim",
        show_lines=False,
        title_style="bold",
    )
    for col in columns:
        tbl.add_column(col, style="squish.white")
    return tbl


# ── Status / quant badges ─────────────────────────────────────────────────────

def status_badge(online: bool) -> str:
    """Return a Rich-markup coloured ● ONLINE / ● OFFLINE badge string."""
    if online:
        return "[bold squish.green]● ONLINE[/]"
    return "[bold squish.error]● OFFLINE[/]"


def quant_badge(quant: str) -> str:
    """Return a Rich-markup coloured quantisation tier badge.

    INT4  → green   (production)
    INT8  → teal    (high quality)
    INT3  → amber + ⚠  (experimental)
    INT2  → red   + ⚠⚠ (research only)
    """
    q = quant.upper().strip()
    if "INT4" in q or "4BIT" in q:
        return "[bold squish.green]INT4[/]"
    if "INT8" in q or "8BIT" in q:
        return "[bold squish.teal]INT8[/]"
    if "INT3" in q or "3BIT" in q:
        return "[bold squish.warn]INT3 ⚠[/]"
    if "INT2" in q or "2BIT" in q:
        return "[bold squish.error]INT2 ⚠⚠[/]"
    return f"[squish.lilac]{quant}[/]"


# ── Startup panel ─────────────────────────────────────────────────────────────

def startup_panel(
    model: str,
    endpoint: str,
    web_ui: str,
    mode: str,
    api_key: str,
) -> None:
    """Print the server startup info as a styled Rich Panel.

    Replaces the plain ``_box([...])`` call in ``cmd_run()``.
    Falls back to a simple bordered box when Rich is unavailable.
    """
    if _RICH_AVAILABLE:
        from rich.panel import Panel
        from rich.table import Table as _T
        from rich import box as _b

        tbl = _T(box=None, show_header=False, padding=(0, 1))
        tbl.add_column(style="squish.dim",   no_wrap=True)
        tbl.add_column(style="squish.white", no_wrap=True)

        tbl.add_row("Model",    f"[bold squish.lilac]{model}[/]")
        tbl.add_row("Mode",     mode)
        tbl.add_row("Endpoint", f"[squish.violet]{endpoint}[/]")
        tbl.add_row("Web UI",   f"[squish.teal]{web_ui}[/]")
        tbl.add_row("API key",  f"[squish.dim]{api_key}[/]")
        tbl.add_row("",         "")
        tbl.add_row("OpenAI",   f"[squish.dim]OPENAI_BASE_URL={endpoint}[/]")
        _port_only = endpoint.rsplit(":", 1)[-1].split("/")[0]
        tbl.add_row("Ollama",   f"[squish.dim]OLLAMA_HOST=http://127.0.0.1:{_port_only}[/]")
        tbl.add_row("",         "")
        tbl.add_row("",         "[squish.dim]Press Ctrl+C to stop[/]")

        console.print()
        console.print(Panel(
            tbl,
            title="[bold squish.violet]Squish[/]  [squish.dim]Local Inference Server[/]",
            border_style="squish.purple",
            padding=(0, 1),
        ))
        console.print()
    else:  # pragma: no cover
        print()
        print(f"  ┌──────────────────────────────────────┐")
        print(f"  │  Squish — Local Inference Server      │")
        print(f"  │  Model    : {model:<26}│")
        print(f"  │  Endpoint : {endpoint:<26}│")
        print(f"  │  Web UI   : {web_ui:<26}│")
        print(f"  │  API key  : {api_key:<26}│")
        print(f"  │  Press Ctrl+C to stop                 │")
        print(f"  └──────────────────────────────────────┘")
        print()


# ── Server status panel ───────────────────────────────────────────────────────

def server_status_panel(
    models: list[dict],
    host: str,
    port: int,
) -> None:
    """Render ``squish ps`` output as a Rich Panel with quant badges.

    ``models`` is the list from the ``/api/ps`` response.
    Renders an offline panel when ``models`` is empty.
    """
    base = f"http://{host}:{port}"
    if _RICH_AVAILABLE:
        from rich.panel import Panel
        from rich.table import Table as _T

        if not models:
            console.print()
            console.print(Panel(
                f"  {status_badge(False)}  No server running at [squish.dim]{base}[/]\n\n"
                "  Start with: [squish.violet]squish run <model>[/]",
                title="[bold squish.violet]squish ps[/]",
                border_style="squish.purple",
                padding=(0, 1),
            ))
            console.print()
            return

        tbl = _T(
            box=_rich_box.ROUNDED,
            header_style="color(135) bold",
            border_style="squish.dim",
            show_lines=False,
        )
        tbl.add_column("Model",   style="bold")
        tbl.add_column("Quant",   style="squish.white",  no_wrap=True)
        tbl.add_column("Params",  style="squish.lilac",  justify="right")
        tbl.add_column("RAM",     style="squish.lilac",  justify="right")
        tbl.add_column("Context", style="squish.dim",    justify="right")

        for m in models:
            name       = m.get("name", "unknown")
            size_bytes = m.get("size", 0)
            size_str   = (
                f"{size_bytes / 1e9:.1f} GB" if size_bytes >= 1e9
                else f"{size_bytes / 1e6:.0f} MB" if size_bytes > 0
                else "—"
            )
            details  = m.get("details", {})
            param_sz = details.get("parameter_size", "—")
            quant    = details.get("quantization_level", "")
            ctx      = details.get("context_length", 0)
            ctx_str  = f"{ctx:,}" if ctx else "—"
            tbl.add_row(
                name,
                quant_badge(quant) if quant else "[squish.dim]—[/]",
                param_sz,
                size_str,
                ctx_str,
            )

        console.print()
        console.print(Panel(
            tbl,
            title=f"[bold squish.violet]squish ps[/]  {status_badge(True)}",
            border_style="squish.purple",
            padding=(0, 1),
        ))
        console.print()
    else:  # pragma: no cover
        print()
        if not models:
            print(f"  ● OFFLINE  No server running at {base}")
            print("  Start with: squish run <model>")
        else:
            for m in models:
                name  = m.get("name", "unknown")
                size  = m.get("size", 0)
                size_str = f"{size/1e9:.1f} GB" if size >= 1e9 else f"{size/1e6:.0f} MB"
                print(f"  ● {name}  {size_str}")
        print()


# ── Chat session header ───────────────────────────────────────────────────────

def chat_header(model_name: str, host: str, port: int) -> None:
    """Print the styled header for an interactive ``squish chat`` session."""
    if _RICH_AVAILABLE:
        console.print()
        console.rule(
            f"[bold squish.violet]Squish Chat[/]  "
            f"[squish.dim]{model_name}[/]  "
            f"{status_badge(True)}",
            style="squish.purple",
        )
        console.print(
            "  [squish.dim]/quit[/]  exit  "
            "[squish.dim]│[/]  [squish.dim]/clear[/]  reset  "
            "[squish.dim]│[/]  [squish.dim]/system[/]  change system prompt",
        )
        console.rule(style="squish.dim")
        console.print()
    else:  # pragma: no cover
        print()
        print(f"  Squish Chat — {model_name}  ● ONLINE")
        print("  /quit  /clear  /system")
        print("  " + "─" * 52)
        print()


# ── Generic info panel ────────────────────────────────────────────────────────

def panel(lines: Sequence[str], title: str = "") -> None:
    """Print a violet-bordered Rich Panel with plain-text lines.

    Replaces ``_box([...])`` for medium-priority informational output.
    Falls back to the simple box when Rich is unavailable.
    """
    if _RICH_AVAILABLE:
        from rich.panel import Panel
        body = "\n".join(
            f"  [squish.dim]{ln}[/]" if not ln.strip() else f"  [squish.white]{ln}[/]"
            for ln in lines
        )
        _title = f"[bold squish.violet]{title}[/]" if title else ""
        console.print()
        console.print(Panel(body, title=_title, border_style="squish.purple", padding=(0, 1)))
        console.print()
    else:  # pragma: no cover
        w = max((len(ln) for ln in lines), default=30) + 4
        print(f"\n  ┌{'─'*w}┐")
        if title:
            print(f"  │  {title:<{w-2}}│")
            print(f"  │{'─'*w}│")
        for ln in lines:
            print(f"  │  {ln:<{w-2}}│")
        print(f"  └{'─'*w}┘\n")
