#!/usr/bin/env python3
"""Generate a self-contained shields-style coverage badge SVG from a Cobertura
coverage.xml — no third-party service, stdlib only.

Usage:
    python3 scripts/make_coverage_badge.py coverage.xml coverage.svg

The line-rate is read from the root ``<coverage line-rate="...">`` attribute
that ``pytest-cov --cov-report=xml`` emits. The output SVG is committed to the
orphan ``badges`` branch by CI and referenced from the README, so the badge is
free forever and depends on nothing but GitHub itself.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET


def _color(pct: int) -> str:
    """Shields-conventional colour band for a coverage percentage."""
    if pct >= 90:
        return "#4c1"      # brightgreen
    if pct >= 80:
        return "#97ca00"   # green
    if pct >= 70:
        return "#a4a61d"   # yellowgreen
    if pct >= 60:
        return "#dfb317"   # yellow
    if pct >= 50:
        return "#fe7d37"   # orange
    return "#e05d44"       # red


def _svg(pct: int) -> str:
    label, msg = "coverage", f"{pct}%"
    lw = 61                       # fixed width of the "coverage" label area
    mw = 7 * len(msg) + 10        # message area scales with digit count
    w = lw + mw
    lx = lw * 5                   # label text centre (×10 for the scale(.1) trick)
    mx = (lw + mw / 2) * 10       # message text centre
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="20" '
        f'role="img" aria-label="{label}: {msg}">'
        f'<title>{label}: {msg}</title>'
        '<linearGradient id="s" x2="0" y2="100%">'
        '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<clipPath id="r"><rect width="{w}" height="20" rx="3" fill="#fff"/></clipPath>'
        '<g clip-path="url(#r)">'
        f'<rect width="{lw}" height="20" fill="#555"/>'
        f'<rect x="{lw}" width="{mw}" height="20" fill="{_color(pct)}"/>'
        f'<rect width="{w}" height="20" fill="url(#s)"/></g>'
        '<g fill="#fff" text-anchor="middle" '
        'font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">'
        f'<text x="{lx}" y="150" fill="#010101" fill-opacity=".3" '
        f'transform="scale(.1)" textLength="{(lw - 10) * 10}">{label}</text>'
        f'<text x="{lx}" y="140" transform="scale(.1)" '
        f'textLength="{(lw - 10) * 10}">{label}</text>'
        f'<text x="{mx:.0f}" y="150" fill="#010101" fill-opacity=".3" '
        f'transform="scale(.1)" textLength="{(mw - 10) * 10}">{msg}</text>'
        f'<text x="{mx:.0f}" y="140" transform="scale(.1)" '
        f'textLength="{(mw - 10) * 10}">{msg}</text></g></svg>'
    )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    xml_path, out_path = argv[1], argv[2]
    rate = float(ET.parse(xml_path).getroot().get("line-rate", "0"))
    pct = round(rate * 100)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(_svg(pct))
    print(f"coverage {pct}% -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
