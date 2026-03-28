"""Find consecutive single-line try/except feature-init blocks in main().

Specifically: blocks of the form
    try:
        from squish.X import Y as _Z
        _Z_instance = Y(args)
        _info("feature", "...")
    except Exception as _e:
        _warn(f"[feature] Skipped: {_e}")

These are the Wave 37/50/51/54 feature loaders. Look for any with the
import commented out or import that fails unconditionally.
"""
import re
from pathlib import Path

SERVER = Path("squish/server.py")
lines = SERVER.read_text().splitlines()

# Find all lines with both `try:` and  following `except Exception` pattern
# Report the line range and feature name for manual review

try_blocks = []
in_try = False
try_start = 0
try_lines = []

i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.strip()
    
    if stripped == "try:":
        in_try = True
        try_start = i + 1
        try_lines = [line]
    elif in_try:
        try_lines.append(line)
        if stripped.startswith("except Exception"):
            # grab the except body
            if i + 1 < len(lines) and "_warn(" in lines[i + 1]:
                warn_line = lines[i + 1].strip()
                # Extract feature tag from _warn(f"[tag]
                m = re.search(r'\[([^\]]+)\]', warn_line)
                tag = m.group(1) if m else "?"
                # Check if the try body imports from squish
                imports = [l.strip() for l in try_lines if l.strip().startswith("from squish")]
                if imports:
                    try_blocks.append({
                        "start": try_start,
                        "end": i + 2,
                        "tag": tag,
                        "import": imports[0],
                        "try_lines": try_lines,
                    })
            in_try = False
            try_lines = []
    i += 1

print(f"Feature-init try/except blocks with squish imports: {len(try_blocks)}")
print()
for b in try_blocks:
    body = [l.strip() for l in b["try_lines"] if l.strip() and l.strip() != "try:"]
    print(f"  L{b['start']:5d}  [{b['tag']:30s}]  {len(body)} body lines")
    for bl in body[:2]:
        print(f"             {bl[:80]}")
