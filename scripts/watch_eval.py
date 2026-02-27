#!/usr/bin/env python3
"""
watch_eval.py — Wait for all 5 eval checkpoints, then run update_paper_v15.py.

Usage:
    python3 scripts/watch_eval.py [--interval 60] [--dry-run]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR  = REPO_ROOT / "eval_output_7b_full"
TASKS     = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "piqa"]


def checkpoints_present():
    return [t for t in TASKS if (EVAL_DIR / f"eval_compressed_{t}.json").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=60,
                    help="Poll interval in seconds (default: 60)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Pass --dry-run to update_paper_v15.py (no file writes)")
    args = ap.parse_args()

    print(f"Watching {EVAL_DIR} for all 5 task checkpoints...")
    print(f"Tasks: {', '.join(TASKS)}")
    print(f"Poll interval: {args.interval}s\n")

    while True:
        done   = checkpoints_present()
        remain = [t for t in TASKS if t not in done]

        print(f"[{time.strftime('%H:%M:%S')}] "
              f"Done: {', '.join(done) or '—'}  |  "
              f"Waiting: {', '.join(remain) or '—'}")

        if not remain:
            print("\nAll 5 checkpoints present — running update_paper_v15.py …\n")
            cmd = [sys.executable, str(REPO_ROOT / "evals" / "update_paper_v15.py")]
            if args.dry_run:
                cmd.append("--dry-run")
            result = subprocess.run(cmd, cwd=str(REPO_ROOT))
            sys.exit(result.returncode)

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
