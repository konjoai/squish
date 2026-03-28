"""Scan for # ── Wave/Phase comment headers in main() that are obsolete
because the code block under them only references features removed in prior waves.

Strategy: look at the # ── Wave N headers inside main() and check if the
global variables they set are dead (zero references outside main() itself).

Also: look for _info("feature", ...) log lines where the corresponding
feature global was removed (replaced with None or absent).
"""
import re
from pathlib import Path

SERVER = Path("squish/server.py")
src = SERVER.read_text()
lines = src.splitlines()

# Find the main() function body
main_start = None
for i, line in enumerate(lines):
    if re.match(r"^def main\(", line):
        main_start = i
        break

if main_start is None:
    print("Could not find main()")
    raise SystemExit(1)

main_body = "\n".join(lines[main_start:])
outside_main = src[:sum(len(l)+1 for l in lines[:main_start])]

# Find all module-level globals assigned in main() (via 'global X' + 'X = ...')
global_stmts = re.findall(r"^\s+global\s+([\w, ]+)", main_body, re.MULTILINE)
main_globals = set()
for stmt in global_stmts:
    for name in re.split(r",\s*", stmt):
        main_globals.add(name.strip())

print(f"Globals declared in main(): {len(main_globals)}")

# For each global, check if it's referenced outside main()
dead_globals_in_main = []
for name in sorted(main_globals):
    # Count refs in outside_main (excluding global declarations)
    refs = [m.start() for m in re.finditer(r"\b" + re.escape(name) + r"\b", outside_main)]
    # Subtract module-level assignments
    assign_refs = len([m for m in re.finditer(r"^" + re.escape(name) + r"\s*[:=]", outside_main, re.MULTILINE)])
    read_refs = len(refs) - assign_refs
    if read_refs == 0:
        dead_globals_in_main.append(name)

print(f"Globals set in main() but never read outside main(): {len(dead_globals_in_main)}")
for name in dead_globals_in_main:
    # Find where it's set in main
    m = re.search(r"global " + re.escape(name) + r"\b", main_body)
    if m:
        lno_in_main = main_body[:m.start()].count("\n")
        print(f"  L~{main_start + lno_in_main + 1:5d}  {name}")
    else:
        print(f"  ??  {name}")
