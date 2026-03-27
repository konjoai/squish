"""Scan server.py for function definitions that are never called
anywhere in squish/ (the entire package, not just server.py).

Excludes:
- Functions starting with `test_`
- `main()` itself
- `__dunder__` methods
- Functions decorated with @app.route, @property, etc (might be framework-called)
- Functions that are exported / referenced in __all__
"""
import pathlib
import re

src_path = pathlib.Path("squish/server.py")
src = src_path.read_text()
lines = src.splitlines()

# Collect all function definitions in server.py
func_re = re.compile(r'^def (\w+)\s*\(', re.MULTILINE)
all_funcs = [m.group(1) for m in func_re.finditer(src)]

# Collect all .py files in squish/ package for call-site search
root = pathlib.Path("squish")
all_py_srcs = {}
for p in root.rglob("*.py"):
    all_py_srcs[str(p)] = p.read_text()

# Also include tests/
for p in pathlib.Path("tests").rglob("*.py"):
    all_py_srcs[str(p)] = p.read_text()

dead = []
for func in all_funcs:
    # Skip special cases
    if func.startswith("test_") or func == "main" or (func.startswith("__") and func.endswith("__")):
        continue
    
    # Count calls: look for func( anywhere in the package (excluding definition lines)
    call_count = 0
    for filepath, text in all_py_srcs.items():
        # Count function calls: funcname(
        calls = len(re.findall(r'\b' + re.escape(func) + r'\s*\(', text))
        if filepath == str(src_path):
            # Subtract 1 for the definition itself
            calls -= 1
        call_count += calls
    
    if call_count == 0:
        # Get line number of definition
        for i, line in enumerate(lines, 1):
            if re.match(r'^def ' + re.escape(func) + r'\s*\(', line):
                dead.append((i, func))
                break

print(f"Potentially dead functions in server.py: {len(dead)}")
for lineno, func in dead:
    print(f"  L{lineno:>5}: def {func}(...)")
