# Squish Launch Checklist

This document provides a five-phase pre-launch checklist for new releases of Squish.
Work through each phase in order. A release is **not ready to ship** until every item
in Phases 1–3 is checked. Phases 4 and 5 are post-ship.

---

## Phase 1 — Code & Quality Gate

Verify these pass locally before any tag or release branch is cut.

- [ ] `pytest --timeout=120` → 0 failures, 0 errors
- [ ] `python3 -c "import squish; print(squish.__version__)"` matches `pyproject.toml` version
- [ ] `pyproject.toml` version, `CHANGELOG.md` `[Unreleased]` heading, and `squish/__init__.py`
      `__version__` are all identical
- [ ] `CHANGELOG.md` has an entry dated today under the new version heading
- [ ] No `# TODO` or `# FIXME` comments in `squish/` (non-experimental) Python files
- [ ] `squish --help` exits 0 and prints usage
- [ ] `squish serve --help` exits 0 and lists all required flags
- [ ] Module count check: `find squish/ -name "*.py" | grep -v experimental | wc -l` ≤ 100
- [ ] `time python3 -c "import squish"` < 2 seconds on M3

---

## Phase 2 — Memory & Latency Contracts

Run on the canonical hardware (M3 16 GB) before tagging.

| Model | Metric | Contract | Measured |
|-------|--------|----------|---------|
| `qwen2.5:1.5b` INT4 | peak Metal RSS | < 1.5 GB | |
| `qwen2.5:1.5b` INT3 | peak Metal RSS | < 1.0 GB | |
| `qwen3:8b` INT4 | peak Metal RSS | < 6.0 GB | |
| `qwen2.5:1.5b` | TTFT (p95) | < 300 ms | |
| `qwen3:8b` | TTFT (p95) | < 600 ms | |
| Server startup RSS | `squish serve --dry-run` before model load | < 200 MB | |

- [ ] All contracts met (or written exception filed in this checklist)
- [ ] Benchmark result saved to `benchmarks/results/` with timestamp + hardware metadata
- [ ] `scripts/run_baseline.sh` reproduces published README numbers within 10%

---

## Phase 3 — Integration Smoke Tests

- [ ] `squish compress <model>` → produces `<model>-int4` directory, no errors
- [ ] `squish run <model>` (BF16 base, no compressed dir) → auto-compresses and starts server
- [ ] `squish serve <model>` → server starts, `/health` returns `{"status":"ok","version":"<ver>"}`
- [ ] `curl -s http://localhost:3333/v1/models | jq .` → lists the loaded model
- [ ] Chat completions round-trip: `curl -s http://localhost:3333/v1/chat/completions -d '{"model":"...","messages":[{"role":"user","content":"ping"}]}'` returns a non-empty response
- [ ] Streaming: same request with `"stream":true` returns SSE chunks
- [ ] `squish doctor` exits 0 with no errors
- [ ] `squish setup` wizard runs without error in a fresh `~/.squish/` directory
- [ ] VS Code extension: `npm test` in `extensions/vscode/` → all Jest tests pass
- [ ] SquishBar macOS app: builds cleanly with `xcodebuild -scheme SquishBar`
- [ ] OpenClaw integration: start Squish, run `openclaw run "hello"` → response received

---

## Phase 4 — Release Packaging (ship day)

- [ ] All Phase 1–3 items checked
- [ ] `git tag -a v<version> -m "Release <version>"` + `git push origin v<version>`
- [ ] GitHub Release created with notes from `CHANGELOG.md`
- [ ] `pypi` package published: `python3 -m build && twine upload dist/*`
- [ ] Homebrew formula (`Formula/squish.rb`) URL and SHA updated to new release tarball
- [ ] Docker image built and pushed: `docker build -t squish:<version> .`
- [ ] `README.md` version badge updated
- [ ] HF model card updated if new benchmark numbers differ from card

---

## Phase 5 — Community (24–48h post-ship)

- [ ] Hacker News post (Tuesday–Thursday 9–10 AM PT; use template in `dev/community_posts.md`)
- [ ] Reddit r/LocalLLaMA post (same timing; use template)
- [ ] Twitter/X thread (4 tweets; use template)
- [ ] LinkedIn announcement (use template)
- [ ] Respond to all GitHub issues opened within 48 h of launch
- [ ] Monitor Homebrew install telemetry and PyPI download stats (day 1, day 7)
- [ ] Archive this checklist with the release date as `docs/planning/LAUNCH_CHECKLIST_v<version>.md`

---

## Exception Process

If a contract cannot be met before launch, file a written exception here:

| Item | Exception reason | Owner | Resolution target |
|------|-----------------|-------|------------------|
| | | | |

An exception requires sign-off from the project lead. Ship without sign-off is a hard stop.
