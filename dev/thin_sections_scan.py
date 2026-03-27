"""Scan for Wave/Phase section headers with zero or minimal code content."""
import re
from pathlib import Path

path = Path("squish/server.py")
lines = path.read_text().splitlines()

# Find all '# ──' style headers
headers = []
for i, line in enumerate(lines):
    if re.match(r"#\s*──", line.strip()):
        headers.append((i + 1, line.strip()))

print(f"Total section headers: {len(headers)}")
print()

for idx, (lno, hdr) in enumerate(headers[:-1]):
    next_lno = headers[idx + 1][0]
    # Lines between this header and next (non-blank, non-comment)
    between = [
        l for l in lines[lno : next_lno - 1]
        if l.strip() and not l.strip().startswith("#")
    ]
    if len(between) == 0:
        print(f"  EMPTY  L{lno}: {hdr[:90]}")
    elif len(between) <= 2:
        print(f"  THIN({len(between)}) L{lno}: {hdr[:90]}")
        for bl in between:
            print(f"             {bl[:80]}")
