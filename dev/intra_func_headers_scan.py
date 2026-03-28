"""Find # ── section headers that are INSIDE functions (not at module level).

These are intra-function navigation comments. Report any that are inside
functions other than main() — they may be candidates for removal if they
don't add documentation value.
"""
import re
from pathlib import Path

SERVER = Path("squish/server.py")
src = SERVER.read_text()
lines = src.splitlines()

# Track current function
func_pattern = re.compile(r"^(async )?def (\w+)\(")
current_func = "<module>"
func_depth = 0  # indentation depth of current function

results = []

for i, line in enumerate(lines):
    stripped = line.strip()
    
    # Detect function definitions at module level (no leading whitespace before def)
    m = func_pattern.match(line)
    if m:
        current_func = m.group(2)
    
    # Is this line a # ── section header?
    if re.match(r"\s+#\s*──", line):  # indented header (inside function)
        indent = len(line) - len(line.lstrip())
        if current_func != "<module>":
            results.append((i + 1, current_func, indent, stripped[:80]))

# Count by function
from collections import Counter
by_func = Counter(r[1] for r in results)

print(f"Intra-function # ── headers: {len(results)}")
print()
print("By function (top 15):")
for fname, count in by_func.most_common(15):
    print(f"  {fname:50s}  {count}")
print()
print("Functions other than main():")
for fname, count in by_func.most_common():
    if fname != "main":
        print(f"  {fname:50s}  {count}")
        for r in results:
            if r[1] == fname:
                print(f"    L{r[0]:5d}  {r[3][:75]}")
