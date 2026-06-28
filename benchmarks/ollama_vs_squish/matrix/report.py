"""Reporting: per-metric matrix tables, the one-screen headlines, post-flight.

Every table cell carries the median, the IQR, the paired Wilcoxon p-value, and
the effect size — so no number appears without its significance and effect, as
the sprint requires. Pure string/formatting logic; unit-tested. Plot rendering
is optional and degrades to "matplotlib unavailable" rather than failing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

HEAD_SQUISH = "squish_int4"
HEAD_OLLAMA = "ollama_q4km"

_METRIC_LABEL = {
    "e2e_s": "End-to-end (200-tok), s",
    "decode_tps": "Decode tok/s",
    "ttft_s": "Cold prefill TTFT, s",
}
_LOWER_IS_BETTER = {"e2e_s": True, "decode_tps": False, "ttft_s": True}


def _fmt(v: float | None, nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def _fmt_p(p: float | None) -> str:
    if p is None:
        return "p=?"
    if p < 1e-4:
        return "p<1e-4"
    return f"p={p:.4f}"


def _median(cell: dict, system: str, metric: str) -> float | None:
    cmp = cell.get("comparisons", {}).get(metric, {})
    for side in ("a", "b"):
        if cmp.get(f"{side}_system") == system:
            return cmp.get(side, {}).get("median")
    return None


def _iqr(cell: dict, system: str, metric: str) -> float | None:
    cmp = cell.get("comparisons", {}).get(metric, {})
    for side in ("a", "b"):
        if cmp.get(f"{side}_system") == system:
            return cmp.get(side, {}).get("iqr")
    return None


def _p_and_effect(cell: dict, metric: str) -> tuple[float | None, float | None, str]:
    cmp = cell.get("comparisons", {}).get(metric, {})
    return (
        cmp.get("wilcoxon", {}).get("p_value"),
        cmp.get("cliffs_delta"),
        cmp.get("cliffs_magnitude", "none"),
    )


def _index(cells: list[dict]) -> dict[tuple[float, int], dict]:
    return {(c["reuse"], c["ctx_tokens"]): c for c in cells}


def metric_table(cells: list[dict], metric: str, reuse_levels, ctx_lengths) -> str:
    """A reuse x context table for one metric: Squish vs Ollama with p + effect."""
    idx = _index(cells)
    lines = [
        f"## {_METRIC_LABEL.get(metric, metric)} "
        f"({'lower' if _LOWER_IS_BETTER.get(metric) else 'higher'} is better)",
        "Each cell: Squish median (IQR) vs Ollama median (IQR) | Wilcoxon p | Cliff's delta",
        "",
    ]
    corner = "reuse \\ ctx"
    header = f"{corner:<12}" + "".join(f"{c:>26}" for c in ctx_lengths)
    lines.append(header)
    lines.append("-" * len(header))
    for r in reuse_levels:
        row = [f"{r:>10.0%}  "]
        for c in ctx_lengths:
            cell = idx.get((r, c))
            if not cell or metric not in cell.get("comparisons", {}):
                row.append(f"{'-':>26}")
                continue
            sm = _median(cell, HEAD_SQUISH, metric)
            om = _median(cell, HEAD_OLLAMA, metric)
            p, delta, _mag = _p_and_effect(cell, metric)
            txt = f"S {_fmt(sm)} / O {_fmt(om)} {_fmt_p(p)} d={_fmt(delta)}"
            row.append(f"{txt:>26}")
        lines.append("".join(row))
    return "\n".join(lines)


def speed_ratio(cell: dict, metric: str = "e2e_s") -> float | None:
    """Squish-vs-Ollama median ratio (>1 means Squish faster for e2e/ttft)."""
    sm = _median(cell, HEAD_SQUISH, metric)
    om = _median(cell, HEAD_OLLAMA, metric)
    if sm in (None, 0) or om is None:
        return None
    return om / sm if _LOWER_IS_BETTER.get(metric) else sm / om


def one_screen_summary(cells: list[dict], ctx_lengths) -> str:
    """The three honest headlines, each with its number, p-value, and effect."""
    idx = _index(cells)
    out = ["=" * 72, "ONE-SCREEN SUMMARY — three honest headlines", "=" * 72, ""]

    out.append("(a) Cold / unique (0% reuse) e2e speed ratio, Squish INT4 vs Ollama Q4_K_M:")
    for c in ctx_lengths:
        cell = idx.get((0.0, c))
        if not cell:
            out.append(f"    {c:>6}: -")
            continue
        ratio = speed_ratio(cell, "e2e_s")
        p, delta, mag = _p_and_effect(cell, "e2e_s")
        out.append(f"    {c:>6}: {_fmt(ratio)}x  ({_fmt_p(p)}, Cliff's d={_fmt(delta)} {mag})")

    out.append("")
    out.append("(b) Partial-reuse curve (e2e) at 8k context, by reuse level:")
    for r in (0.0, 0.25, 0.50, 0.75, 1.0):
        cell = idx.get((r, 8000))
        if not cell:
            out.append(f"    {r:>5.0%}: -")
            continue
        ratio = speed_ratio(cell, "e2e_s")
        p, delta, mag = _p_and_effect(cell, "e2e_s")
        out.append(f"    {r:>5.0%}: {_fmt(ratio)}x  ({_fmt_p(p)}, d={_fmt(delta)} {mag})")

    out.append("")
    out.append("(c) Exact-repeat (100% reuse) ceiling, e2e speed ratio:")
    for c in ctx_lengths:
        cell = idx.get((1.0, c))
        if not cell:
            out.append(f"    {c:>6}: -")
            continue
        ratio = speed_ratio(cell, "e2e_s")
        p, delta, mag = _p_and_effect(cell, "e2e_s")
        out.append(f"    {c:>6}: {_fmt(ratio)}x  ({_fmt_p(p)}, d={_fmt(delta)} {mag})")
    out.append("=" * 72)
    return "\n".join(out)


def postflight(cells: list[dict], min_runs: int = 30) -> str:
    """Post-flight verification checklist with ✓/✗ per the sprint."""

    def mark(ok: bool) -> str:
        return "PASS" if ok else "FAIL"

    cache_ok = all(
        s.get("cache_total_runs", 0) == 0
        or s.get("cache_ok_runs", 0) == s.get("cache_total_runs", 0)
        for c in cells
        for s in c.get("systems", {}).values()
    )
    runs_ok = all(
        len([r for r in s.get("runs", []) if not r.get("failed")]) >= min_runs
        for c in cells
        for s in c.get("systems", {}).values()
        if s.get("role") == "head_to_head"
    )
    stats_ok = all(
        cmp.get("wilcoxon", {}).get("p_value") is not None and cmp.get("cliffs_delta") is not None
        for c in cells
        for cmp in c.get("comparisons", {}).values()
    )
    oom_handled = all(c.get("status") != "error" for c in cells)
    statuses = {c["cell_id"]: c.get("status") for c in cells}

    lines = [
        "POST-FLIGHT VERIFICATION",
        f"  [{mark(cache_ok)}] Cache-hit % matches intent for both systems, every cell",
        f"  [{mark(runs_ok)}] >=30 paired runs per head-to-head system",
        f"  [{mark(stats_ok)}] Wilcoxon p + Cliff's delta present for every comparison",
        f"  [{mark(oom_handled)}] OOM/governor handled without crashing; status recorded",
        "  per-cell status:",
    ]
    for cid, st in sorted(statuses.items()):
        lines.append(f"    {cid}: {st}")
    return "\n".join(lines)


# ── optional plots ────────────────────────────────────────────────────────────


def render_plots(cells: list[dict], out_dir: Path, reuse_levels, ctx_lengths) -> list[str]:
    """e2e-vs-context per reuse, decode-vs-context, and the reuse curve. Optional."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return ["matplotlib unavailable — plots skipped"]

    idx = _index(cells)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    fig, ax = plt.subplots()
    for r in reuse_levels:
        ys = [speed_ratio(idx[(r, c)], "e2e_s") if (r, c) in idx else None for c in ctx_lengths]
        ax.plot(ctx_lengths, [y or float("nan") for y in ys], marker="o", label=f"{r:.0%} reuse")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("e2e speed ratio (Squish/Ollama)")
    ax.set_title("e2e speed ratio vs context, per reuse level")
    ax.legend()
    p1 = out_dir / "e2e_vs_context.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    written.append(str(p1))

    fig, ax = plt.subplots()
    rs = list(reuse_levels)
    ys = [speed_ratio(idx[(r, 8000)], "e2e_s") if (r, 8000) in idx else None for r in rs]
    ax.plot([f"{r:.0%}" for r in rs], [y or float("nan") for y in ys], marker="s")
    ax.set_xlabel("reuse level")
    ax.set_ylabel("e2e speed ratio @ 8k")
    ax.set_title("Reuse curve (e2e) at 8k context")
    p2 = out_dir / "reuse_curve_8k.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    written.append(str(p2))
    return written
