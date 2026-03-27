#!/usr/bin/env python3
"""
Wave 120 — Dead Global Purge
Removes 188 module-level `_var = None` globals and their `global _var`
declarations from server.py.

False positives excluded (live via globals() dict assignment):
  _lazy_expert, _structured_sparsity

Run from repo root:
    python3 dev/wave120_purge.py [--dry-run]
"""
import re
import sys
import shutil
from pathlib import Path

SERVER = Path("squish/server.py")
FALSE_POSITIVES = {"_lazy_expert", "_structured_sparsity"}

def load():
    return SERVER.read_text()

def find_dead_globals(text):
    init_none = set(re.findall(r'^(_\w+)\s*=\s*None\b', text, re.MULTILINE))
    assigned_non_none = set()
    for var in init_none:
        # Regular indented assignment
        if re.search(r'^\s+' + re.escape(var) + r'\s*=[^=\n]', text, re.MULTILINE):
            assigned_non_none.add(var)
        # globals() dict assignment
        if re.search(r'globals\(\)\["' + re.escape(var) + r'"\]\s*=', text):
            assigned_non_none.add(var)
    dead = (init_none - assigned_non_none) - FALSE_POSITIVES
    return sorted(dead)

def remove_lines(text, dead):
    lines = text.split('\n')
    keep = []
    removed_module = 0
    removed_global_decl = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Module-level: `_varname     = None  # ... comment`
        m = re.match(r'^(_\w+)\s*=\s*None\b', line)
        if m and m.group(1) in dead:
            removed_module += 1
            continue

        # main() global declaration: `    global _varname`
        # Could be `    global _varname` (solo) or part of combined line
        # We handle solo-line only since that's the pattern we see
        m2 = re.match(r'^(\s+)global\s+(_\w+)\s*$', line)
        if m2 and m2.group(2) in dead:
            removed_global_decl += 1
            continue

        keep.append(line)

    result = '\n'.join(keep)
    return result, removed_module, removed_global_decl

def main():
    dry_run = '--dry-run' in sys.argv
    text = load()
    dead = find_dead_globals(text)

    print(f"Dead globals found: {len(dead)}")
    print(f"False positives excluded: {sorted(FALSE_POSITIVES)}")
    print()

    # Check for complex cases (> 2 refs) — need manual review
    complex_cases = []
    for var in dead:
        count = len(re.findall(r'\b' + re.escape(var) + r'\b', text))
        if count > 2:
            complex_cases.append((var, count))
    if complex_cases:
        print("Complex cases (>2 refs — manual hot-path review needed):")
        for var, count in sorted(complex_cases, key=lambda x: -x[1]):
            refs = [(m.start(), text[max(0,m.start()-40):m.start()+60].replace('\n','↵'))
                    for m in re.finditer(r'\b' + re.escape(var) + r'\b', text)]
            print(f"  {var} ({count} refs)")
            for pos, ctx in refs:
                lineno = text[:pos].count('\n') + 1
                print(f"    L{lineno}: ...{ctx}...")
        print()

    new_text, rm_mod, rm_decl = remove_lines(text, dead)

    old_lines = text.count('\n')
    new_lines = new_text.count('\n')
    delta = old_lines - new_lines

    print(f"Lines removed: {delta} ({rm_mod} module-level decls + {rm_decl} global stmts)")
    print(f"  {old_lines} → {new_lines} lines")

    if dry_run:
        print("\nDry-run mode — no files written.")
        return

    # Backup
    backup = SERVER.with_suffix('.py.wave119')
    shutil.copy2(SERVER, backup)
    print(f"\nBackup: {backup}")

    SERVER.write_text(new_text)
    print(f"Written: {SERVER}")
    print("\nNext: python3 dev/wave120_purge.py --verify")

if __name__ == "__main__":
    main()
