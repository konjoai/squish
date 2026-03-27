"""Scan server.py for `try: from squish.X import ...` blocks where
the imported names are never referenced outside that try/except block.

These are dead optional-import blocks left over from removed features.
"""
import pathlib
import re

src = pathlib.Path("squish/server.py").read_text()
lines = src.splitlines()

# Find all try/except blocks that contain `from squish. import` or `import squish.`
# Pattern: find `    try:` followed within ~10 lines by a `from ... import ...`
# then find what names are imported and check if they're used

dead_blocks: list[tuple[int, str, list[str]]] = []

i = 0
while i < len(lines):
    line = lines[i].rstrip()
    # Look for indented `try:` lines
    m_try = re.match(r'^(\s+)try:\s*$', line)
    if m_try:
        indent = m_try.group(1)
        block_start = i
        # Scan forward to collect the try block (until we hit except/finally at same indent)
        j = i + 1
        imported_names: list[str] = []
        except_line = -1
        while j < len(lines) and j < i + 30:
            tl = lines[j].rstrip()
            # Check for `except` or `finally` at the same indent level
            if re.match(re.escape(indent) + r'except\b', tl) or re.match(re.escape(indent) + r'finally\b', tl):
                except_line = j
                break
            # Look for import statements within the try block
            m_imp = re.match(r'\s+from\s+[\w.]+\s+import\s+(.*)', tl)
            if m_imp:
                for name in [n.strip().split(' as ')[-1].strip() for n in m_imp.group(1).split(',')]:
                    if name:
                        imported_names.append(name)
            m_imp2 = re.match(r'\s+import\s+([\w.]+)(?:\s+as\s+(\w+))?', tl)
            if m_imp2:
                alias = m_imp2.group(2) or m_imp2.group(1).split('.')[-1]
                imported_names.append(alias)
            j += 1

        if imported_names and except_line > 0:
            # Check if any imported name is used outside this try block
            # (from block_start to except_line is the try body)
            try_body_lines = set(range(block_start, except_line + 1))
            dead_names = []
            for name in imported_names:
                # Find all references to this name in the entire file
                refs = [
                    src[:m.start()].count('\n')
                    for m in re.finditer(r'\b' + re.escape(name) + r'\b', src)
                ]
                # Count refs outside the try block
                outside_refs = [r for r in refs if r not in try_body_lines]
                if len(outside_refs) == 0:
                    dead_names.append(name)
            
            if dead_names and len(dead_names) == len(imported_names):
                dead_blocks.append((block_start + 1, indent, dead_names))
    i += 1

print(f"Fully dead try-import blocks: {len(dead_blocks)}")
for lineno, indent, names in dead_blocks[:20]:
    print(f"\n  L{lineno} ({repr(indent)}):")
    for n in names:
        print(f"    imported: {n}")
