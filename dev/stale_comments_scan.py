"""Find multi-line standalone comment blocks (3+ consecutive comment lines)
in server.py that reference variables or functions that no longer exist
in the file. These are stale docs for removed features.

A comment block is "stale" if it contains multiple references to
_var_names or function names that have zero code references in the file.
"""
import pathlib
import re

src = pathlib.Path("squish/server.py").read_text()
lines = src.splitlines()

# Build a set of all identifiers that exist in code (non-comment) lines
code_lines = [l for l in lines if not l.strip().startswith('#') and l.strip()]
code_src = '\n'.join(code_lines)

# Find runs of 3+ consecutive comment-only lines
chunks: list[tuple[int, int, list[str]]] = []
i = 0
while i < len(lines):
    if lines[i].strip().startswith('#') and lines[i].strip():
        j = i
        block = []
        while j < len(lines) and (lines[j].strip().startswith('#') or not lines[j].strip()):
            if lines[j].strip().startswith('#'):
                block.append(lines[j])
            j += 1
        if len(block) >= 3:
            chunks.append((i + 1, j, block))
        i = j
    else:
        i += 1

print(f"Comment blocks (3+ lines): {len(chunks)}")

# For each block, find _var_names mentioned and check if they exist in code
stale = []
for start, end, block in chunks:
    block_text = '\n'.join(block)
    mentioned_vars = re.findall(r'\b(_\w+)\b', block_text)
    if not mentioned_vars:
        continue
    dead_vars = [v for v in set(mentioned_vars) if not re.search(r'\b' + re.escape(v) + r'\b', code_src)]
    if len(dead_vars) >= 2:
        stale.append((start, end, block, dead_vars))

print(f"Stale comment blocks (2+ dead var refs): {len(stale)}")
for start, end, block, dead_vars in stale[:10]:
    print(f"\n  L{start}-{end}: dead vars: {dead_vars[:5]}")
    for line in block[:5]:
        print(f"    {line.rstrip()}")
    if len(block) > 5:
        print(f"    ... ({len(block) - 5} more lines)")
