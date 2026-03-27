#!/usr/bin/env python3
"""Wave 122 scoping: find module-level constants (non-None, non-None-initialized)
that are assigned once and never read back in server.py."""
import ast, re

with open('squish/server.py') as f:
    src = f.read()
    lines = src.splitlines()

tree = ast.parse(src)

# Find module-level simple assignments: _VAR = <literal or simple value>
# (not _VAR = None — those were Wave 120 dead globals)
module_assigns = {}  # name -> lineno
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.startswith('_'):
                val = node.value
                # Skip None assignments (Wave 120 already handled)
                if isinstance(val, ast.Constant) and val.value is None:
                    continue
                module_assigns[target.id] = node.lineno
    elif isinstance(node, ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.target.id.startswith('_'):
            module_assigns[node.target.id] = node.lineno

print(f"Module-level _var assignments (non-None): {len(module_assigns)}")

# Check which are referenced (as Name loads, attribute targets, etc.)
referenced = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
        referenced.add(node.id)
    # Also: f-strings, format strings — covered by ast.Name in the AST

# Find potentially dead constants (assigned but never loaded)
dead_consts = {k: v for k, v in module_assigns.items() if k not in referenced}

print(f"Potentially dead constants: {len(dead_consts)}")
for name, lineno in sorted(dead_consts.items(), key=lambda x: x[1]):
    print(f"  L{lineno}: {name} = {lines[lineno-1].split('=', 1)[1].strip()[:60]}")
