#!/usr/bin/env python3
"""Module count gate for squish.

This script enforces the current approved Python module inventory for:
- squish/ (full package)
- squish/squash/ (enterprise/security subpackage)
"""

from pathlib import Path

EXPECTED_SQUISH_PY = 136
EXPECTED_SQUASH_PY = 40


def _count_py_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    squish_root = repo_root / "squish"
    squash_root = squish_root / "squash"

    squish_count = _count_py_files(squish_root)
    squash_count = _count_py_files(squash_root)

    ok = True

    if squish_count != EXPECTED_SQUISH_PY:
        print(
            f"FAIL: squish/ Python module count is {squish_count}; "
            f"expected {EXPECTED_SQUISH_PY}."
        )
        ok = False
    else:
        print(f"OK: squish/ Python module count is {squish_count}.")

    if squash_count != EXPECTED_SQUASH_PY:
        print(
            f"FAIL: squish/squash/ Python module count is {squash_count}; "
            f"expected {EXPECTED_SQUASH_PY}."
        )
        ok = False
    else:
        print(f"OK: squish/squash/ Python module count is {squash_count}.")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
