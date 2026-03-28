"""Scan for dead local assignments (assigned once, never read) in server.py.

Strategy: within each function body, find local variables that are:
  1. Assigned exactly once (e.g. `x = <expr>`)
  2. Never read in that function (no reference other than the LHS assignment)

Uses regex + scope heuristics (not full AST) for speed.
False-positive prone — results need manual verification.
"""
import re
from pathlib import Path

SERVER = Path("squish/server.py")
src = SERVER.read_text()

# --- Simple approach: look for exception aliases in except clauses
# `except Foo as e:` where `e` is never referenced in the handler body.
# This is already covered by dead_except_alias_scan.py.

# --- Different angle: find lines like `_varname = something` at function
# indent level (4 spaces) where _varname only appears once in the function.

# Extract functions from server.py
func_pattern = re.compile(
    r"^(def |async def )(\w+)\(", re.MULTILINE
)

func_starts = [(m.start(), m.group(2)) for m in func_pattern.finditer(src)]

lines = src.splitlines()
line_offsets = [0]
for line in lines:
    line_offsets.append(line_offsets[-1] + len(line) + 1)

def byte_to_lineno(offset):
    import bisect
    return bisect.bisect_right(line_offsets, offset)

candidates = []

for i, (start, fname) in enumerate(func_starts):
    end = func_starts[i + 1][0] if i + 1 < len(func_starts) else len(src)
    body = src[start:end]
    body_lines = body.splitlines()

    # Find local assignments at 4-space (top-level function body) or 8-space
    assign_pattern = re.compile(r"^\s{4,16}(_\w+)\s*=\s*(?!None$)", re.MULTILINE)
    assignments = {}
    for m in assign_pattern.finditer(body):
        varname = m.group(1)
        if varname not in assignments:
            assignments[varname] = []
        assignments[varname].append(m.start())

    for varname, positions in assignments.items():
        if len(positions) != 1:
            continue  # assigned multiple times — skip
        # Count all occurrences of the varname in the body
        all_refs = [m.start() for m in re.finditer(r"\b" + re.escape(varname) + r"\b", body)]
        if len(all_refs) <= 1:
            # only the assignment itself — dead
            lno = byte_to_lineno(start) + body[:positions[0]].count("\n")
            candidates.append((lno, fname, varname))

print(f"Dead-local candidates: {len(candidates)}")
for lno, fname, varname in sorted(candidates)[:40]:
    print(f"  L{lno:5d}  {fname:40s}  {varname}")
