"""
squish/_term.py

Shared ANSI true-colour terminal utilities used by cli.py and server.py.

Colour strategy
---------------
By default Squish emits 24-bit RGB escape codes (\\033[38;2;R;G;Bm), which
bypass the host terminal's colour theme palette remapping.  This means brand
colours render predictably regardless of the active theme.

When the terminal has a *light* background (e.g. macOS Terminal.app in light
mode) the dark-optimised palette may have poor contrast.  Squish detects this
via the ``COLORFGBG`` env var (set by most POSIX terminals) and automatically
switches to a deeper, high-contrast variant.  Set ``SQUISH_DARK_BG=0`` or
``SQUISH_DARK_BG=1`` to force a specific mode.

Public API
----------
    has_truecolor(fd=1)           → bool      (True when fd is a true-colour TTY)
    detect_dark_background()      → bool      (True when terminal background is dark)
    C                             → _Palette  (theme-aware brand colour constants)
    gradient(text, stops)         → str       (left-to-right RGB gradient text)
    LOGO_GRAD                     → list      (purple → pink → teal stop list)
"""
from __future__ import annotations

import os
import sys

__all__ = ["has_truecolor", "detect_dark_background", "C", "gradient", "LOGO_GRAD"]


def has_truecolor(fd: int = 1) -> bool:
    """Return True when file-descriptor *fd* (1=stdout, 2=stderr) is a
    true-colour TTY and NO_COLOR is not set."""
    try:
        is_tty = os.isatty(fd)
    except Exception:
        return False
    return (
        is_tty
        and "NO_COLOR" not in os.environ
        and (
            os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit")
            or os.environ.get("TERM_PROGRAM", "") in (
                "iTerm.app", "WezTerm", "Ghostty", "Hyper", "vscode", "warp",
                "Apple_Terminal",
            )
            or "kitty" in os.environ.get("TERM", "")
            or "direct" in os.environ.get("TERM", "")
            or bool(os.environ.get("FORCE_COLOR", ""))
        )
    )


def detect_dark_background() -> bool:
    """Return True when the terminal likely has a dark background.

    Priority order:

    1. ``SQUISH_DARK_BG`` env var — explicit override.
       ``1`` / ``true`` / ``yes`` → dark;  ``0`` / ``false`` / ``no`` → light.
    2. ``COLORFGBG`` env var — set by xterm, macOS Terminal.app, most Linux
       terminals.  Format is ``"fg;bg"`` where the background index ``< 7``
       indicates a dark palette colour (0 = black, 6 = dark cyan).
    3. Dark fallback — developer terminals are overwhelmingly dark-themed.

    To explicitly enable the light-background palette in an unsupported
    terminal, set::

        export SQUISH_DARK_BG=0
    """
    override = os.environ.get("SQUISH_DARK_BG", "").strip().lower()
    if override in ("1", "true", "yes"):
        return True
    if override in ("0", "false", "no"):
        return False
    cfg = os.environ.get("COLORFGBG", "").strip()
    if cfg:
        try:
            bg_idx = int(cfg.rsplit(";", 1)[-1])
            # Palette indices 0–6 are dark colours; 7–15 are light/bright.
            return bg_idx < 7
        except (ValueError, IndexError):
            pass
    return True  # dark fallback


# ── Module-level flags ────────────────────────────────────────────────────────

try:
    _stdout_fd = sys.stdout.fileno() if hasattr(sys.stdout, "fileno") else 1
except Exception:
    _stdout_fd = 1
_TC = has_truecolor(_stdout_fd)
_IS_DARK_BG: bool = detect_dark_background()


def _k(s: str) -> str:
    """Return *s* only when stdout is a true-colour TTY."""
    return s if _TC else ""


class _Palette:
    """Dark-background palette — Squish brand 24-bit colours."""
    DP  = _k("\033[38;2;88;28;135m")    # deep purple  #581C87
    P   = _k("\033[38;2;124;58;237m")   # purple       #7C3AED
    V   = _k("\033[38;2;139;92;246m")   # violet       #8B5CF6
    L   = _k("\033[38;2;167;139;250m")  # lilac        #A78BFA
    MG  = _k("\033[38;2;192;132;252m")  # med-purple   #C084FC
    PK  = _k("\033[38;2;236;72;153m")   # pink         #EC4899
    LPK = _k("\033[38;2;249;168;212m")  # light pink   #F9A8D4
    T   = _k("\033[38;2;34;211;238m")   # teal         #22D3EE
    LT  = _k("\033[38;2;165;243;252m")  # light teal   #A5F3FC
    G   = _k("\033[38;2;52;211;153m")   # mint green   #34D399
    W   = _k("\033[38;2;248;250;252m")  # near-white   #F8FAFC
    SIL = _k("\033[38;2;180;185;210m")  # silver       #B4B9D2
    DIM = _k("\033[38;2;100;116;139m")  # dim slate    #64748B
    B   = _k("\033[1m")                 # bold
    R   = _k("\033[0m")                 # reset all


class _PaletteLight:
    """Light-background palette — deeper, high-contrast brand colours.

    Used automatically when ``detect_dark_background()`` returns ``False``
    (detected via ``COLORFGBG`` or ``SQUISH_DARK_BG=0``).
    Each colour is a darker/more-saturated variant of the corresponding
    dark-palette entry to ensure legibility on white/light backgrounds.
    """
    DP  = _k("\033[38;2;67;20;105m")    # deeper purple  #431469
    P   = _k("\033[38;2;88;28;135m")    # dark purple    #581C87
    V   = _k("\033[38;2;109;40;217m")   # dark violet    #6D28D9
    L   = _k("\033[38;2;124;58;237m")   # purple         #7C3AED
    MG  = _k("\033[38;2;139;92;246m")   # violet         #8B5CF6
    PK  = _k("\033[38;2;157;23;77m")    # deep pink      #9D174D
    LPK = _k("\033[38;2;219;39;119m")   # pink           #DB2777
    T   = _k("\033[38;2;6;182;212m")    # teal           #06B6D4
    LT  = _k("\033[38;2;8;145;178m")    # dark teal      #0891B2
    G   = _k("\033[38;2;16;185;129m")   # green          #10B981
    W   = _k("\033[38;2;15;23;42m")     # near-black     #0F172A
    SIL = _k("\033[38;2;71;85;105m")    # slate          #475569
    DIM = _k("\033[38;2;51;65;85m")     # dim slate      #334155
    B   = _k("\033[1m")                 # bold
    R   = _k("\033[0m")                 # reset all


# Select palette at import time based on detected background brightness.
C: _Palette = _Palette() if _IS_DARK_BG else _PaletteLight()  # type: ignore[assignment]

# Purple → pink → teal gradient used for the big logo and accent lines.
LOGO_GRAD: list[tuple[int, int, int]] = [
    ( 88,  28, 135),   # deep purple
    (124,  58, 237),   # purple
    (139,  92, 246),   # violet
    (192, 100, 220),   # lavender-pink
    (236,  72, 153),   # pink
    ( 34, 211, 238),   # teal
]


def gradient(
    text: str,
    stops: list[tuple[int, int, int]],
    *,
    force_color: bool | None = None,
) -> str:
    """Interpolate a left-to-right RGB gradient across *text*.

    Only emits escape codes when stdout is a true-colour TTY (or when
    *force_color* is explicitly ``True``); otherwise returns *text* unchanged
    so plain-terminal output stays readable.

    Parameters
    ----------
    force_color : bool | None
        Override the module-level TTY detection.  ``True`` forces ANSI output;
        ``False`` forces plain text; ``None`` (default) uses the autodetected
        ``_TC`` flag.
    """
    use_color = _TC if force_color is None else force_color
    if not use_color or not text:
        return text
    n = len(text)
    k = len(stops) - 1
    out: list[str] = []
    for i, ch in enumerate(text):
        t = i / max(n - 1, 1)
        seg = min(int(t * k), k - 1)
        frac = t * k - seg
        r1, g1, b1 = stops[seg]
        r2, g2, b2 = stops[seg + 1]
        r = int(r1 + (r2 - r1) * frac)
        g = int(g1 + (g2 - g1) * frac)
        b = int(b1 + (b2 - b1) * frac)
        out.append(f"\033[38;2;{r};{g};{b}m{ch}")
    out.append("\033[0m")
    return "".join(out)
