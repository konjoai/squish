"""squish/bench/__main__.py — runnable benchmark entry point.

Allows: python -m squish.bench [options]

This thin wrapper delegates straight to cmd_bench via build_parser() so all
existing flags (``--tps-min``, ``--ttft-max``, ``--port``, ...) are available
with no duplication.

CI usage
--------
  python -m squish.bench --tps-min 30
  python -m squish.bench --tps-min 30 --ttft-max 500

Exit codes
----------
  0  All prompts completed; optional gate thresholds met (or not set).
  1  User/input error (no server, bad argument).
  2  Performance gate failure (--tps-min or --ttft-max threshold breached).
"""
from __future__ import annotations

import sys


def main() -> None:
    from squish.cli import build_parser
    parser = build_parser()
    # Re-parse with "bench" pre-inserted so `python -m squish.bench --port 11435`
    # is equivalent to `squish bench --port 11435`.
    if sys.argv[1:] and sys.argv[1] != "bench":
        sys.argv.insert(1, "bench")
    elif not sys.argv[1:]:
        sys.argv.append("bench")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
