"""
squish/ui.py

Shared TUI helpers for the Squish CLI.

Uses `rich` for progress bars, spinners, styled output, and interactive
prompts.  Falls back to plain ASCII output when rich is not installed so
the CLI remains usable in a minimal install environment.

Public API
──────────
  console                   — global Rich Console instance
  banner()                  — welcome screen with ASCII art + version
  spinner(msg)              — context manager wrapping a Rich spinner
  progress(desc, total)     — download/compress progress bar context manager
  model_picker(models)      — interactive list picker (arrow keys + enter)
  confirm(msg, default)     — y/n prompt with default answer
  success(msg)              — styled ✅ message
  warn(msg)                 — styled ⚠  message
  error(msg)                — styled ✗  message
  hint(msg)                 — styled dim hint line
"""
from __future__ import annotations

import contextlib
import sys
from typing import Generator, Sequence

# ── Rich availability check ──────────────────────────────────────────────────

try:
    from rich.console import Console
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
    from rich.table import Table
    from rich.theme import Theme
    from rich import box as _rich_box
    _RICH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _RICH_AVAILABLE = False


# ── Squish brand colours ─────────────────────────────────────────────────────

_SQUISH_THEME = Theme({
    "squish.purple":  "rgb(124,58,237)",
    "squish.violet":  "rgb(139,92,246)",
    "squish.lilac":   "rgb(167,139,250)",
    "squish.pink":    "rgb(236,72,153)",
    "squish.teal":    "rgb(34,211,238)",
    "squish.green":   "rgb(52,211,153)",
    "squish.white":   "rgb(248,250,252)",
    "squish.dim":     "rgb(100,116,139)",
    "squish.warn":    "rgb(251,191,36)",
    "squish.error":   "rgb(248,113,113)",
})

if _RICH_AVAILABLE:
    console = Console(theme=_SQUISH_THEME, highlight=False)
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


# ── ASCII art banner ──────────────────────────────────────────────────────────

_ASCII_LOGO = r"""
  ____  ___  _   _ ___ ____  _   _
 / ___||_ _|| | | |_ _/ ___|| | | |
 \___ \ | | | | | || |\___ \| |_| |
  ___) || | | |_| || | ___) |  _  |
 |____/|___| \___/|___|____/|_| |_|"""


def banner() -> None:
    """Print the Squish welcome banner with version."""
    try:
        from squish import __version__ as _ver
    except Exception:
        _ver = "9.0.0"

    if _RICH_AVAILABLE:
        console.print(f"[squish.violet]{_ASCII_LOGO}[/squish.violet]")
        console.rule(f"[squish.dim]v{_ver}[/squish.dim]")
        console.print()
    else:  # pragma: no cover
        print(_ASCII_LOGO)
        print(f"  Squish v{_ver}")
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
        box=_rich_box.SIMPLE,
        header_style="rgb(139,92,246) bold",
        border_style="rgb(100,116,139)",
        show_lines=False,
    )
    for col in columns:
        tbl.add_column(col, style="rgb(248,250,252)")
    return tbl
