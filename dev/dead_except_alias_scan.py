"""Scan server.py for variables that are assigned inside a function or
block but then never read before the scope exits or is re-assigned.

Focus specifically on `_e` (exception aliases in bare `except` blocks),
`_ignored`, and similar naming patterns that suggest intentionally-swallowed
variables. Only report if TRULY never referenced.

Also scan for: `pass`-only except clauses that are not `except Exception as _e`
and for redundant `# type: ignore` comments on lines containing dead names.
"""
import pathlib
import re

src = pathlib.Path("squish/server.py").read_text()
lines = src.splitlines()

# Strategy: look for `except ... as _e:` followed by NO use of `_e`
# in the same except block.
except_alias_dead: list[tuple[int, str]] = []

i = 0
while i < len(lines):
    line = lines[i]
    # Except clause with alias
    m = re.match(r'^(\s+)except\s+.*\s+as\s+(\w+)\s*:', line)
    if m:
        indent = m.group(1)
        alias = m.group(2)
        except_start = i
        
        # Collect the except body
        j = i + 1
        body_lines = []
        while j < len(lines):
            bl = lines[j]
            # If we hit something at <= the except's indent that isn't continuation
            if bl.strip() and not bl.startswith(indent + ' ') and not bl.startswith(indent + '\t'):
                break
            body_lines.append(bl)
            j += 1
        
        # Check if alias is used in the except body
        body_text = '\n'.join(body_lines)
        uses = len(re.findall(r'\b' + re.escape(alias) + r'\b', body_text))
        
        if uses == 0 and alias.startswith('_'):
            except_alias_dead.append((i + 1, alias))
    i += 1

print(f"Except aliases never used in body: {len(except_alias_dead)}")
for lineno, alias in except_alias_dead[:30]:
    print(f"  L{lineno:>5}: except ... as {alias}")

print()

# Also count total `except ... as _e:` patterns to get the ratio
total_excepts_with_alias = len(re.findall(r'\bexcept\b.*\bas\s+_\w+\s*:', src))
unused_count = len(except_alias_dead)
used_count = total_excepts_with_alias - unused_count
print(f"Total except aliases: {total_excepts_with_alias}")
print(f"  Used: {used_count}")
print(f"  Unused: {unused_count}")
