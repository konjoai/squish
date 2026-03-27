"""Scan server.py for argparse flags registered via add_argument() but
whose corresponding args.NAME is never read in the codebase.

A flag `--my-flag` maps to `args.my_flag`. If `args.my_flag` (and
`getattr(args, "my_flag", ...)`) never appear, the flag is dead weight.
"""
import pathlib
import re

src_path = pathlib.Path("squish/server.py")
src = src_path.read_text()

# Find all add_argument() calls — extract the flag name
# Handles: add_argument("--my-flag", ...) and add_argument("--my-flag", "--short", ...)
# Find all add_argument() calls — extract the flag name AND optional dest=
flag_re = re.compile(
    r'add_argument\(\s*"(--[\w-]+)"[^)]*?\)',
    re.MULTILINE | re.DOTALL,
)

flags: list[tuple[int, str, str]] = []  # (line, raw_flag, attr_name)
for m in flag_re.finditer(src):
    call_text = m.group(0)
    raw = m.group(1)  # e.g. "--my-flag"
    # Check for explicit dest="..."
    dest_m = re.search(r'\bdest\s*=\s*["\'](\w+)["\']', call_text)
    if dest_m:
        attr = dest_m.group(1)
    else:
        attr = raw.lstrip("-").replace("-", "_")
    lineno = src[:m.start()].count('\n') + 1
    flags.append((lineno, raw, attr))

print(f"Total registered flags: {len(flags)}")
print()

dead = []
for lineno, raw, attr in flags:
    # Count reads: args.attr or getattr(args, "attr"
    reads = (
        len(re.findall(r'\bargs\.' + re.escape(attr) + r'\b', src))
        + len(re.findall(r'getattr\s*\(\s*args\s*,\s*["\']' + re.escape(attr) + r'["\']', src))
    )
    if reads == 0:
        dead.append((lineno, raw, attr))

print(f"Flags with zero reads of args.NAME: {len(dead)}")
print()
for lineno, raw, attr in dead:
    print(f"  L{lineno:>5}: {raw:<35} (args.{attr})")
