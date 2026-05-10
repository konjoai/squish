---
name: konjo-ship
description: Konjo sprint completion checklist for squish.
user-invocable: true
---
# Konjo Ship — squish

## Sprint Completion Checklist
```
[ ] All tests pass — `python -m pytest` green
[ ] `ruff check` and `ruff format --check` clean
[ ] Quantization accuracy gates verified if applicable
[ ] CHANGELOG.md updated
[ ] PLAN.md updated (wave state)
[ ] MODULES.md updated if new modules added
[ ] git add && git commit -m "type(scope): description" && git push
```

## Session Handoff Template
```
SHIPPED      [what was completed]
TESTS        [passing / failing / count]
PUSHED       [commit hash]
NEXT SESSION [exact next task / wave]
DISCOVERIES  [papers, repos, techniques found]
HEALTH       [Green / Yellow / Red]
```
