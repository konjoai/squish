"""Scan for argparse add_argument entries whose dest/attr is never read via args.*

Strategy:
  1. Parse all add_argument calls → collect dest name.
  2. For each dest name, search for `args.<dest>` or `getattr(args, "<dest>"` reads.
  3. Report any with zero reads — but we already know all 89 are live.

This is a wider scan: look also for `setattr(args, "X", ...)` calls where X
is set but `args.X` or `getattr(args, "X"` is never read afterward.
"""

import re
from pathlib import Path

SERVER = Path("squish/server.py")
src = SERVER.read_text()

# Find all setattr(args, "name", ...) calls
setattr_pattern = re.compile(r'setattr\(args,\s*["\'](\w+)["\']')
setattr_names = {}
for m in setattr_pattern.finditer(src):
    name = m.group(1)
    lno = src[:m.start()].count("\n") + 1
    setattr_names[name] = lno

# Find all reads: args.name or getattr(args, "name")
read_direct = re.compile(r"\bargs\.(\w+)\b")
read_getattr = re.compile(r'getattr\(args,\s*["\'](\w+)["\']')

all_reads = set()
for m in read_direct.finditer(src):
    all_reads.add(m.group(1))
for m in read_getattr.finditer(src):
    all_reads.add(m.group(1))

dead = []
for name, lno in sorted(setattr_names.items(), key=lambda x: x[1]):
    if name not in all_reads:
        dead.append((lno, name))

print(f"setattr(args, ...) calls: {len(setattr_names)}")
print(f"Dead (never read via args.X or getattr): {len(dead)}")
for lno, name in dead:
    print(f"  L{lno:5d}  {name}")
