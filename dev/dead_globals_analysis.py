#!/usr/bin/env python3
"""Wave 120 scoping: find globals initialized to None that are never assigned non-None."""
import re, sys

with open('squish/server.py') as f:
    text = f.read()

init_none = set(re.findall(r'^(_\w+)\s*=\s*None\b', text, re.MULTILINE))
assigned_non_none = set()
for var in init_none:
    pattern = r'^\s+' + re.escape(var) + r'\s*=[^=\n]'
    if re.search(pattern, text, re.MULTILINE):
        assigned_non_none.add(var)

truly_dead = sorted(init_none - assigned_non_none)
live = sorted(assigned_non_none)
print('Total None-init globals:', len(init_none))
print('Live (assigned non-None):', len(live))
print('Truly dead:', len(truly_dead))
print('Live (first 20):', live[:20])
print()
print('DEAD globals:')
for v in truly_dead:
    count = len(re.findall(r'\b' + re.escape(v) + r'\b', text))
    print(f'  {v} ({count} refs)')
