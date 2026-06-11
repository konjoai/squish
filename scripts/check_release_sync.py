#!/usr/bin/env python3
"""check_release_sync.py — Verify all release artifacts point at the same version.

Usage:
  python3 scripts/check_release_sync.py
  python3 scripts/check_release_sync.py --expected-version 9.33.5

Exits 0 if all checks pass, 1 if any fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _check_pyproject() -> str:
    content = _read(REPO_ROOT / "pyproject.toml")
    m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not m:
        raise ValueError("version not found in pyproject.toml")
    return m.group(1)


def _check_init() -> str:
    content = _read(REPO_ROOT / "squish" / "__init__.py")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if not m:
        raise ValueError("__version__ not found in squish/__init__.py")
    return m.group(1)


def _check_pypi() -> str:
    url = "https://pypi.org/pypi/squish-ai/json"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data: dict = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OSError(f"PyPI request failed: {exc}") from exc
    return str(data["info"]["version"])


def _check_formula_url() -> str:
    content = _read(REPO_ROOT / "Formula" / "squish.rb")
    m = re.search(r"squish[_-]ai[_-]([0-9][0-9.]+)\.tar\.gz", content)
    if not m:
        raise ValueError("version not found in Formula/squish.rb url")
    return m.group(1)


def _check_plist() -> str:
    plist = REPO_ROOT / "apps" / "macos" / "SquishBar" / "Resources" / "Info.plist"
    if not plist.exists():
        raise FileNotFoundError(f"Info.plist not found at {plist}")
    content = _read(plist)
    m = re.search(
        r"<key>CFBundleShortVersionString</key>\s*<string>([^<]+)</string>",
        content,
    )
    if not m:
        raise ValueError("CFBundleShortVersionString not found in Info.plist")
    return m.group(1)


def _check_bottle_sha(expected_version: str) -> str:
    content = _read(REPO_ROOT / "Formula" / "squish.rb")
    m = re.search(
        r"bottle do.*?sha256 cellar:[^,]+,\s*arm64_tahoe:\s*\"([a-f0-9]{64})\"",
        content,
        re.DOTALL,
    )
    if not m:
        raise ValueError("bottle sha256 not found in Formula/squish.rb")
    formula_sha = m.group(1)

    url_m = re.search(r'root_url\s+"([^"]+)"', content)
    if not url_m:
        raise ValueError("root_url not found in bottle block")

    rebuild_m = re.search(r"rebuild\s+(\d+)", content)
    rebuild = rebuild_m.group(1) if rebuild_m else None
    suffix = f".{rebuild}" if rebuild else ""
    bottle_file = f"squish-{expected_version}.arm64_tahoe.bottle{suffix}.tar.gz"
    bottle_url = f"{url_m.group(1)}/{bottle_file}"

    print(f"  (fetching bottle from {bottle_url} …)")
    try:
        with urllib.request.urlopen(bottle_url, timeout=60) as resp:
            actual_sha = hashlib.sha256(resp.read()).hexdigest()
    except urllib.error.URLError as exc:
        raise OSError(f"could not fetch bottle: {exc}") from exc

    if actual_sha != formula_sha:
        raise ValueError(
            f"sha256 mismatch:\n    formula: {formula_sha}\n    actual:  {actual_sha}"
        )
    return formula_sha


def _run_check(
    label: str,
    fn,
    expected: Optional[str],
    results: list[tuple[str, bool, str]],
) -> Optional[str]:
    try:
        value = fn()
        ok = expected is None or value == expected
        results.append((label, ok, value))
        return value
    except (ValueError, OSError, FileNotFoundError, KeyError) as exc:
        results.append((label, False, f"ERROR: {exc}"))
        return None


def run_checks(expected_version: Optional[str]) -> bool:
    results: list[tuple[str, bool, str]] = []

    pyproject_ver = _run_check(
        "pyproject.toml version", _check_pyproject, expected_version, results
    )

    if expected_version is None and pyproject_ver:
        expected_version = pyproject_ver
        print(f"Auto-detected expected version: {expected_version}\n")

    _run_check("squish/__init__.py __version__", _check_init, expected_version, results)
    _run_check("PyPI latest version", _check_pypi, expected_version, results)
    _run_check("Formula/squish.rb url version", _check_formula_url, expected_version, results)
    _run_check(
        "apps/macos/SquishBar Info.plist CFBundleShortVersionString",
        _check_plist,
        expected_version,
        results,
    )

    if expected_version:
        _run_check(
            "Formula/squish.rb bottle sha256",
            lambda: _check_bottle_sha(expected_version),  # type: ignore[arg-type]
            None,
            results,
        )
    else:
        results.append(
            ("Formula/squish.rb bottle sha256", False, "skipped — no expected version")
        )

    width = max(len(r[0]) for r in results) + 2
    all_pass = all(r[1] for r in results)

    for label, ok, value in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {label:<{width}}  {value}")

    print()
    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks FAILED.")

    return all_pass


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="check_release_sync",
        description="Verify all release artifacts point at the same version.",
    )
    p.add_argument(
        "--expected-version",
        metavar="VERSION",
        help="Version to check against (e.g. 9.33.5). Auto-detected from pyproject.toml if omitted.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    passed = run_checks(args.expected_version)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
