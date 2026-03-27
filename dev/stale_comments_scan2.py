"""Revised stale comment scanner — broader rules:
1. Blocks with 1+ dead var refs (not 2+)
2. Also catch blocks of all-commented-code (lines like # _var = ...)
"""
import pathlib
import re

src = pathlib.Path("squish/server.py").read_text()
lines = src.splitlines()

code_lines = [l for l in lines if not l.strip().startswith('#') and l.strip()]
code_src = '\n'.join(code_lines)

# Find all identifiers that appear in code (not comments)
known_ids = set(re.findall(r'\b(_\w+)\b', code_src))

# Find consecutive commented-code lines (# followed by Python-like code)
commented_code_re = re.compile(
    r'^\s*#\s+'                  # comment prefix
    r'(?:'
    r'    |\s*(?:if|for|while|try|except|from|import|def|class|with|return)\b'
    r'|\s*_\w+\s*='
    r'|\s*_\w+\s*\('
    r')'
)

# Find blocks of 3+ consecutive commented-CODE lines (not just narrative)
i = 0
commented_code_blocks: list[tuple[int, int, list[str]]] = []
while i < len(lines):
    if commented_code_re.match(lines[i]):
        j = i
        block = []
        while j < len(lines) and (commented_code_re.match(lines[j]) or not lines[j].strip()):
            if lines[j].strip():
                block.append(lines[j])
            j += 1
        if len(block) >= 3:
            commented_code_blocks.append((i + 1, j, block))
        i = j
    else:
        i += 1

print(f"Commented-code blocks (3+ lines): {len(commented_code_blocks)}")
for start, end, block in commented_code_blocks[:15]:
    print(f"\n  L{start}-{end} ({len(block)} lines):")
    for bl in block[:4]:
        print(f"    {bl.rstrip()}")
    if len(block) > 4:
        print(f"    ... ({len(block)-4} more)")
