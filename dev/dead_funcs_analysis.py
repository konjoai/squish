#!/usr/bin/env python3
"""Wave 122 scoping: find module-level functions in server.py that may be dead."""
import ast

with open('squish/server.py') as f:
    src = f.read()

tree = ast.parse(src)

# Collect all call names throughout the file
calls = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)

# Find module-level function defs
module_funcs = []
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        module_funcs.append((node.name, node.lineno, len(ast.dump(node))))

print(f"Module-level functions: {len(module_funcs)}")
print()
dead = []
for name, lineno, size in module_funcs:
    if name not in calls:
        dead.append((name, lineno, size))

print(f"Potentially dead (never called by name): {len(dead)}")
for name, lineno, size in dead:
    est_lines = size // 80  # rough estimate
    print(f"  L{lineno}: {name}  (~{est_lines} lines)")

print()
print("Called functions (sanity check):")
for name, lineno, size in module_funcs:
    if name in calls:
        print(f"  L{lineno}: {name}")
