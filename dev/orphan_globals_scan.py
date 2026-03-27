"""Scan server.py for orphaned `global _foo` declarations in main() —
declared but never assigned (neither directly nor via globals()[...])."""
import pathlib
import re

src = pathlib.Path("squish/server.py").read_text()
lines = src.splitlines()

# Collect all assignments in the file (including globals()[...])
all_assignments: set[str] = set()
for line in lines:
    m = re.match(r'\s*(_{1,2}\w+)\s*=(?!=)', line)
    if m:
        all_assignments.add(m.group(1))
for m in re.finditer(r"globals\(\)\[[\"'](\w+)[\"']\]\s*=", src):
    all_assignments.add(m.group(1))

# Collect `global _foo` declarations inside main() — with line numbers
in_main = False
main_global_decls: dict[str, list[int]] = {}
for i, line in enumerate(lines, 1):
    if re.match(r'^def main\(', line):
        in_main = True
    if not in_main:
        continue
    m = re.match(r'\s+global\s+(.*)', line)
    if m:
        for var in [v.strip() for v in m.group(1).split(',')]:
            if var:
                main_global_decls.setdefault(var, []).append(i)

# Orphaned = declared in main() with global, but never assigned anywhere
orphaned = {
    var: lns
    for var, lns in main_global_decls.items()
    if var not in all_assignments
}

print(f"Total `global` declarations in main(): {len(main_global_decls)}")
print(f"Orphaned (no assignment anywhere):     {len(orphaned)}")
print()
for var, lns in sorted(orphaned.items(), key=lambda x: x[1][0]):
    print(f"  L{lns[0]:>5}: global {var}")
