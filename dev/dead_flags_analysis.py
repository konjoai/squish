#!/usr/bin/env python3
"""Wave 121 scoping: find argparse flags not consumed anywhere after parsing."""
import re

with open('squish/server.py') as f:
    text = f.read()

# Extract all flag names from add_argument calls
# e.g. add_argument("--chunk-prefill", ...) → "chunk_prefill"
flag_re = re.compile(r'add_argument\(\s*["\']--(%s)["\']' % r'[a-z0-9][a-z0-9-]*')
flags = {}
for m in flag_re.finditer(text):
    raw = m.group(1)       # dash-separated
    attr = raw.replace('-', '_')   # Python attr name
    flags[raw] = attr

print(f"Total registered flags: {len(flags)}")

# Check which are consumed: either as getattr(args, "attr") or args.attr
consumed = {}
for raw, attr in flags.items():
    if (f'getattr(args, "{attr}"' in text
            or f'getattr(args, \'{attr}\'' in text
            or f'args.{attr}' in text):
        consumed[raw] = attr

dead_flags = {k: v for k, v in flags.items() if k not in consumed}
print(f"Consumed flags: {len(consumed)}")
print(f"Potentially unconsumed: {len(dead_flags)}")
print()
if dead_flags:
    print("Dead flags (registered but never read):")
    for raw, attr in sorted(dead_flags.items()):
        # Find the add_argument line
        m = re.search(r'[^\n]+add_argument[^\n]*--' + re.escape(raw) + r'[^\n]*', text)
        if m:
            line_no = text[:m.start()].count('\n') + 1
            print(f"  L{line_no}: --{raw}")
        else:
            print(f"  --{raw}")
