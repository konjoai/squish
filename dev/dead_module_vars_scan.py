"""Scan server.py module-level `_var = None` (or similar) declarations
that have zero read/write references anywhere else in the file or in
other squish/ Python files.

These are module-level dead globals — set once to None but never used.
"""
import pathlib
import re

src_path = pathlib.Path("squish/server.py")
src = src_path.read_text()
lines = src.splitlines()

# Collect module-level (no leading indent) variable assignments like:
#   _foo                = None  # ...
#   _foo_bar            = None  # ...
# also including non-None literal inits at module level
module_var_re = re.compile(r'^(_\w+)\s*=\s*(?:None|False|True|\[\]|\{\}|0|""|\'\')')

module_vars: dict[str, int] = {}  # var -> line number
for i, line in enumerate(lines, 1):
    m = module_var_re.match(line)
    if m:
        module_vars[m.group(1)] = i

# For each module-level var, count ALL references in server.py
# excluding the assignment line itself
dead = []
for var, lineno in sorted(module_vars.items(), key=lambda x: x[1]):
    # All occurrences
    all_refs = list(re.finditer(re.escape(var), src))
    total = len(all_refs)
    
    if total == 0:
        continue  # shouldn't happen
    
    # Count references that are NOT inside `global var` stmts in main()
    # and NOT the module-level assignment line itself
    live_refs = 0
    for m in all_refs:
        line_idx = src[:m.start()].count('\n')
        line_text = lines[line_idx]
        # Skip: the module-level assignment line itself
        if line_idx + 1 == lineno:
            continue
        # Skip: `global _var, ...` or `global ..., _var` lines
        if re.match(r'\s+global\b', line_text) and var in line_text:
            continue
        # Skip: standalone comment lines
        if line_text.strip().startswith('#'):
            continue
        live_refs += 1
    
    if live_refs == 0:
        dead.append((lineno, var))

print(f"Module-level vars with no live references: {len(dead)}")
print()
for lineno, var in dead:
    defline = lines[lineno - 1].rstrip()
    print(f"  L{lineno:>5}: {defline}")
