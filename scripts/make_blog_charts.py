#!/usr/bin/env python3
"""Generate the three launch-article charts from verified BENCHMARKS.md values.

All numbers below are transcribed by hand from BENCHMARKS.md (the only
permitted source) and cross-checked against the article. No value is
estimated. Run from the repo root:

    python scripts/make_blog_charts.py

Outputs SVG (for the page) and PNG (for social cards) into
docs/assets/blog/.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render, no display required
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ── Brand + light/dark-friendly palette ──────────────────────────────
SQUISH = "#7B3FF7"  # Squish brand purple (matches docs accent)
SQUISH_LIGHT = "#B888FF"  # lighter purple for the "holds" cohort
OLLAMA = "#64748B"  # neutral slate for the comparison tool
COLLAPSE = "#E0526A"  # muted red signalling the INT2 collapse
INK = "#6B7280"  # mid-gray text/axes legible on light AND dark
GRID = "#9CA3AF"  # grid lines, drawn at low alpha

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets" / "blog"
OUT.mkdir(parents=True, exist_ok=True)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 13,
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "DejaVu Sans", "Arial"],
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "svg.fonttype": "none",  # keep text as text in the SVG
            "figure.dpi": 120,
        }
    )


def _despine(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(length=0)


def _save(fig: plt.Figure, name: str) -> None:
    for ext in ("svg", "png"):
        fig.savefig(
            OUT / f"{name}.{ext}",
            format=ext,
            bbox_inches="tight",
            pad_inches=0.18,
            transparent=True,
        )
    plt.close(fig)
    print(f"wrote {name}.svg + {name}.png")


# ── chart-speed ──────────────────────────────────────────────────────
# Source: BENCHMARKS.md §1b "E2E @ 4000-token prompt" row.
# Thermally-controlled set: Ollama 0.30.7 37.5 s vs Squish 3.8 s = 9.8x.
def chart_speed() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.0, 5.25))  # 1200x630 @120dpi -> social-ready
    tools = ["Ollama 0.30.7", "Squish"]
    secs = [37.5, 3.8]
    colors = [OLLAMA, SQUISH]
    bars = ax.bar(tools, secs, color=colors, width=0.55, zorder=3)
    ax.set_ylabel("Full-response time (seconds)")
    ax.set_title(
        "Full response at a 4000-token prompt",
        fontsize=15,
        fontweight="700",
        pad=14,
    )
    ax.set_ylim(0, 42)
    ax.yaxis.grid(True, color=GRID, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    for bar, val in zip(bars, secs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.7,
            f"{val:g} s",
            ha="center",
            va="bottom",
            fontweight="700",
            fontsize=14,
        )
    # 9.8x callout: arrow lands on the bar's right portion, clear of the
    # centred "3.8 s" value label (which sits over the bar's centre).
    ax.annotate(
        "9.8× faster",
        xy=(1.18, 3.8),
        xytext=(1.18, 17),
        ha="center",
        fontsize=14,
        fontweight="800",
        color=SQUISH,
        arrowprops=dict(arrowstyle="-|>", color=SQUISH, lw=2),
    )
    _save(fig, "chart-speed")


# ── chart-accuracy ───────────────────────────────────────────────────
# Source: BENCHMARKS.md §1b note (INT3 0.551 vs INT4 0.541, arc_easy
# acc_norm, Qwen2.5-7B) and §3 (naive INT2 ~0.29 ~= random). No FP16 bar
# (no FP16 value exists in BENCHMARKS.md). Random baseline drawn at 0.25.
def chart_accuracy() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.4, 5.25))
    labels = ["INT4", "INT3", "INT2"]
    acc = [0.541, 0.551, 0.29]
    colors = [SQUISH, SQUISH_LIGHT, COLLAPSE]
    bars = ax.bar(labels, acc, color=colors, width=0.42, zorder=3)
    ax.set_ylabel("arc_easy acc_norm")
    ax.set_title(
        "Qwen2.5-7B accuracy by precision: INT3 holds, INT2 collapses",
        fontsize=14.5,
        fontweight="700",
        pad=14,
    )
    ax.set_ylim(0, 0.62)
    ax.yaxis.grid(True, color=GRID, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    for bar, val in zip(bars, acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="700",
            fontsize=14,
        )
    # random baseline marker (4-way arc_easy chance = 0.25)
    ax.axhline(0.25, color=INK, linestyle="--", linewidth=1.4, alpha=0.8, zorder=2)
    ax.text(
        1.5,
        0.265,
        "random (0.25)",
        ha="center",
        va="bottom",
        fontsize=11.5,
        style="italic",
        color=INK,
    )
    _save(fig, "chart-accuracy")


# ── chart-footprint ──────────────────────────────────────────────────
# Source: BENCHMARKS.md §1b (Peak RAM 5.14 vs 3.5 GB) and §0 / §2
# (7B disk: Ollama INT4 4.36 GB vs Squish INT3 3.56 GB).
def chart_footprint() -> None:
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 5.25))
    groups = ["Peak RAM\n(inference)", "Disk size\n(7B model)"]
    ollama = [5.14, 4.36]
    squish = [3.50, 3.56]
    x = range(len(groups))
    w = 0.36
    b1 = ax.bar([i - w / 2 for i in x], ollama, w, label="Ollama", color=OLLAMA, zorder=3)
    b2 = ax.bar([i + w / 2 for i in x], squish, w, label="Squish", color=SQUISH, zorder=3)
    ax.set_ylabel("Gigabytes (GB)")
    ax.set_title(
        "Memory and disk footprint",
        fontsize=15,
        fontweight="700",
        pad=14,
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 6.0)
    ax.yaxis.grid(True, color=GRID, alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    _despine(ax)
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.08,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontweight="700",
                fontsize=12.5,
            )
    leg = ax.legend(frameon=False, loc="upper right")
    for txt in leg.get_texts():
        txt.set_color(INK)
    _save(fig, "chart-footprint")


def main() -> None:
    # Silence the "Inter not found" noise if the font is unavailable;
    # DejaVu Sans is the bundled fallback and renders identically small.
    fm.findfont("Inter", fallback_to_default=True)
    chart_speed()
    chart_accuracy()
    chart_footprint()
    print(f"\nAll charts written to {OUT}")


if __name__ == "__main__":
    main()
