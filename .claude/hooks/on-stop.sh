#!/bin/bash
cat << 'EOF'

--- SQUISH HANDOFF ---
SHIPPED      [what was completed]
TESTS        [passing / failing / count]
PUSHED       [commit hash or "not pushed — reason"]
NEXT SESSION [the exact next task — not "continue the work"]
DISCOVERIES  [papers, repos, techniques found this session]
HEALTH       [Green / Yellow / Red — one line]
EOF
