#!/usr/bin/env python3
"""dev/scripts/import_scan.py — Cross-file import dependency analyser.

Builds an import dependency graph for the squish package using AST parsing
(no execution required) and produces two actionable reports:

  Report A — orphan modules
      Files with zero inbound imports from other squish modules.
      Candidates for removal or consolidation.

  Report B — dead feature flags
      Module-level variables assigned ``None`` in server.py that are never
      assigned a non-None value in any live code path
      (i.e. the only assignments are ``_foo = None``).

Usage
-----
  python3 dev/scripts/import_scan.py [--squish-root PATH] [--report {A,B,AB}]

Output is printed to stdout as a human-readable summary.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Report A — orphan modules
# ---------------------------------------------------------------------------

def _collect_py_files(pkg_root: Path) -> list[Path]:
    return sorted(pkg_root.rglob("*.py"))


def _parse_imports(file: Path) -> list[str]:
    """Return a list of module paths imported by *file* (relative to squish package)."""
    try:
        tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return imports


def _build_inbound_graph(files: list[Path], pkg_root: Path) -> dict[str, int]:
    """Return a dict: module_path → inbound_import_count."""
    # Map each file to its dotted module name
    file_to_module: dict[Path, str] = {}
    for f in files:
        rel = f.relative_to(pkg_root)
        parts = list(rel.with_suffix("").parts)
        file_to_module[f] = ".".join(parts)

    module_set = set(file_to_module.values())
    inbound: dict[str, int] = {m: 0 for m in module_set}

    for f in files:
        for imp in _parse_imports(f):
            # Normalise: relative to pkg root
            for mod in module_set:
                if imp == mod or imp.startswith(mod + ".") or mod.endswith("." + imp):
                    if file_to_module.get(f) != mod:  # don't count self-import
                        inbound[mod] = inbound.get(mod, 0) + 1
    return inbound


def report_orphan_modules(pkg_root: Path) -> None:
    """Print Report A: modules with zero inbound imports."""
    print("=" * 70)
    print("Report A — Orphan modules (zero inbound imports)")
    print("=" * 70)
    files = _collect_py_files(pkg_root)
    inbound = _build_inbound_graph(files, pkg_root)

    orphans = sorted(
        (mod for mod, count in inbound.items() if count == 0),
    )

    # Filter out __init__, __main__, and obvious entry points
    skip_patterns = ("__init__", "__main__", "cli", "server", "main")
    candidates = [m for m in orphans if not any(p in m for p in skip_patterns)]

    if not candidates:
        print("No orphan modules found.")
    else:
        print(f"Found {len(candidates)} module(s) with zero inbound imports:\n")
        for mod in candidates:
            print(f"  {mod}")

    print()


# ---------------------------------------------------------------------------
# Report B — dead None-only feature flags in server.py
# ---------------------------------------------------------------------------

def _find_none_globals(server_py: Path) -> dict[str, list[int]]:
    """Return a dict: var_name → list of line numbers where assigned.

    Tracks module-level names that are assigned ``None`` at least once.
    """
    try:
        tree = ast.parse(server_py.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return {}

    assignments: dict[str, list[int]] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    name = target.id
                    if name.startswith("_"):
                        assignments.setdefault(name, []).append(node.lineno)

    return assignments


def _find_none_assignments(server_py: Path) -> set[str]:
    """Return set of names that are ONLY ever assigned None at module level."""
    try:
        tree = ast.parse(server_py.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return set()

    none_assigned: set[str] = set()
    non_none_assigned: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                name = target.id
                if isinstance(node.value, ast.Constant) and node.value.value is None:
                    none_assigned.add(name)
                else:
                    non_none_assigned.add(name)

    return none_assigned - non_none_assigned


def report_dead_flags(server_py: Path) -> None:
    """Print Report B: module-level globals only ever assigned None."""
    print("=" * 70)
    print("Report B — Dead feature flags (module-level = None only)")
    print(f"  Source: {server_py}")
    print("=" * 70)

    if not server_py.exists():
        print(f"  ERROR: {server_py} not found.")
        print()
        return

    dead = sorted(_find_none_assignments(server_py))
    if not dead:
        print("No dead-None globals detected.")
    else:
        print(f"Found {len(dead)} module-level name(s) only ever assigned None:\n")
        for name in dead:
            print(f"  {name}")
        print()
        print("These may be dead feature flags or uninitialised globals.")
        print("Review each before removal — some may be set inside functions.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--squish-root",
        default="",
        metavar="PATH",
        help="Path to the squish package root (default: auto-detect)",
    )
    ap.add_argument(
        "--report",
        choices=["A", "B", "AB"],
        default="AB",
        help="Which report(s) to generate (default: AB)",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    # Locate squish package root
    if args.squish_root:
        pkg_root = Path(args.squish_root).expanduser().resolve()
    else:
        # Assume script lives in dev/scripts/ → ../../squish/squish
        script_dir = Path(__file__).resolve().parent
        pkg_root = script_dir.parent.parent / "squish" / "squish"
        if not pkg_root.is_dir():
            pkg_root = script_dir.parent.parent / "squish"

    print(f"Squish package root: {pkg_root}\n")

    if args.report in ("A", "AB"):
        report_orphan_modules(pkg_root)

    if args.report in ("B", "AB"):
        server_py = pkg_root / "server.py"
        report_dead_flags(server_py)


if __name__ == "__main__":
    main()
