"""Scan for module-level string/tuple/dict constants that are never referenced.

Strategy:
  1. Find ALL_CAPS or _SCREAMING pattern module-level assignments (not functions/classes).
  2. Check if they appear outside the assignment line (as reads).
  3. Report any with zero reads.
"""
import re
from pathlib import Path

SERVER = Path("squish/server.py")
src = SERVER.read_text()
lines = src.splitlines()

# Find module-level constant assignments: ALL_CAPS_NAME = ...
# Must be at column 0, must not be inside a function/class.
# Pattern: _UPPER or UPPER at start of line
const_pattern = re.compile(r"^(_?[A-Z][A-Z0-9_]{2,})\s*[:=]", re.MULTILINE)

consts = {}
for m in const_pattern.finditer(src):
    name = m.group(1)
    lno = src[:m.start()].count("\n") + 1
    if name not in consts:
        consts[name] = lno

dead = []
for name, lno in sorted(consts.items(), key=lambda x: x[1]):
    # Count all references in entire file (any position)
    all_refs = [m.start() for m in re.finditer(r"\b" + re.escape(name) + r"\b", src)]
    # Remove the definition itself
    read_refs = len(all_refs) - 1  # subtract 1 for the assignment itself
    if read_refs == 0:
        dead.append((lno, name))

print(f"Total module-level constants scanned: {len(consts)}")
print(f"Dead (zero reads): {len(dead)}")
for lno, name in dead:
    print(f"  L{lno:5d}  {name}")
