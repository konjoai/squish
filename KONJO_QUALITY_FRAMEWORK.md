# Konjo Code Quality Framework
## Three Walls Against AI Slop — Language-Agnostic, Gate-Enforced

**Version:** May 2026 · **Scope:** All KonjoAI repositories and any project using Claude Code
**Reference implementation:** lopi (Rust) · **Applicable languages:** Rust, Python, Mojo, TypeScript

---

## The Problem

Research makes the failure mode measurable:

- AI-assisted code generates **1.7× more logical and correctness bugs** than traditional development (CodeRabbit 2026)
- AI agents change tests so broken code passes instead of fixing the actual code (Baltes, Cheong, Treude 2026)
- AI agent commits degraded the Maintainability Index in **56.1% of commits** and increased Cyclomatic Complexity in **42.7%** (MSR 2026)
- AI agent code review suggestions produce **significantly larger increases in code complexity and size** than human reviewer suggestions (arXiv 2603.15911)
- **Homogeneous AI review pipelines echo rather than cancel errors** — an agent that wrote the code reviewing its own output shares the same training distribution and exhibits correlated failures (arXiv 2603.25773)
- Pass-rate benchmarks **systematically undermeasure extension robustness** — agent-generated code deteriorates under repeated editing as each turn extends the anti-patterns of prior turns (SlopCodeBench, arXiv 2603.24755)

**The conclusion:** a single-model self-review loop cannot catch its own slop. The only solutions are (1) executable specifications as external ground truth, (2) deterministic tooling that cannot be reasoned past, and (3) adversarial review from a distinct session. All three are enforced here.

---

## The Three Walls

```
Wall 1: Pre-Commit Hook     ← local, fast (< 60s), blocks the commit
Wall 2: CI Gate             ← GitHub Actions, blocks the PR merge
Wall 3: Konjo Review Agent  ← Claude Opus in a separate session, blocks the merge
```

Every commit must pass Wall 1. Every PR must pass Wall 2. Every merge to main must pass Wall 3.
No bypass flags. No `--no-verify`. No `skip-review` comments.

---

## Quality Gate Reference Table

All thresholds are enforced by CI. "Hard Block" = PR cannot merge.

| Gate | Hard Block | Target | Tool (Rust) | Tool (Python) |
|------|-----------|--------|-------------|---------------|
| Line coverage | ≥ 80% | ≥ 95% | cargo-llvm-cov | pytest-cov |
| Mutation survival (changed files) | ≤ 10% | 0% | cargo-mutants | mutmut |
| Cognitive complexity per function | ≤ 15 | ≤ 10 | clippy | radon cc |
| Lint violations | 0 | 0 | clippy -D warnings | ruff check |
| Format violations | 0 | 0 | rustfmt | ruff format |
| Dead code | 0 | 0 | rustc -W dead_code | vulture |
| Undocumented public APIs | 0 | 0 | rustdoc -D missing_docs | interrogate |
| p50 latency regression | ≤ 5% | 0% | criterion | pytest-benchmark |
| Function body length | ≤ 50 lines | ≤ 30 lines | loc count | radon |
| File length | ≤ 500 lines | ≤ 300 lines | loc count | loc count |
| DRY violations (>10L, >85% similar) | 0 | 0 | dry_check.py | dry_check.py |
| unwrap()/expect() in non-test Rust | 0 | 0 | clippy::unwrap_used | — |
| Silent error swallowing | 0 | 0 | — | grep / ast |
| Known CVEs in dependencies | 0 | 0 | cargo-audit | safety / pip-audit |
| License violations | 0 | 0 | cargo-deny | — |

---

## Wall 1: Pre-Commit Hook

**File:** `.konjo/hooks/pre-commit`
**Install:** `bash .konjo/scripts/install-hooks.sh`
**For Rust repos:** also add `cargo-husky` as a dev-dependency (auto-installs hook on `cargo build`)
**Runtime:** < 60 seconds

**What it checks:**

1. **Rust** (if `Cargo.toml` present and `.rs` files staged):
   - `cargo check` — compilation gate
   - `cargo clippy -- -D warnings -D clippy::unwrap_used` — zero violations
   - `unwrap()`/`expect()` scan on non-test files — hard block
   - Dead code warning count — warns (hard block in CI)

2. **Python** (if `pyproject.toml` or `requirements.txt` present and `.py` files staged):
   - `ruff check` — zero lint violations
   - `ruff format --check` — zero format violations
   - Silent `except:` / `except Exception:` scan — hard block
   - `mypy` — warns at pre-commit, hard block in CI

3. **Universal** (all repos, all staged files):
   - File size check: warn if any source file > 500 lines
   - DRY check on staged files only (fast path)
   - TODO/FIXME/HACK scan — hard block

4. **Wall 3 preview** (if `ANTHROPIC_API_KEY` is set):
   - Runs `konjo_review.py --soft-fail` on the staged diff
   - Advisory only at pre-commit; hard block in CI

---

## Wall 2: CI Gate (GitHub Actions)

**File:** `.github/workflows/konjo-gate.yml`
**Triggers:** all pull requests to main, push to main

**Five parallel gates:**

### G1 — Static Analysis
`cargo fmt --check` · `clippy -D warnings -D pedantic -D unwrap_used` · `cargo audit` · `cargo deny` · dead code zero-tolerance

### G2 — Tests + Coverage
`cargo llvm-cov nextest --fail-under-lines 80` · coverage gate at exactly 80% (will increase to 95% per ratchet schedule)

### G3 — Mutation Testing (PRs only)
`cargo-mutants` scoped to changed files · survival rate ≤ 10% · surfaces tests that would pass with bugs inserted

### G4 — Complexity + Size + DRY + Docs
`clippy::cognitive_complexity` — any function > 15 fails · file size gate (500 line limit) · `dry_check.py` (0 violations) · `rustdoc -D missing_docs` (0 undocumented public APIs)

### G5 — Adversarial Review (PRs only)
`konjo_review.py` using `claude-opus-4-6` · independent session from the builder · asks ten mandatory questions · BLOCKER verdict blocks merge

**Final gate:** all five must pass for merge to be allowed.

---

## Wall 3: Konjo Adversarial Review

**Model:** `claude-opus-4-6` (the critic must not match the builder model)
**File:** `.konjo/scripts/konjo_review.py`
**Research basis:** Homogeneous review pipelines echo errors. Opus reviewing Sonnet output introduces model diversity that reduces correlated failures.

**The Ten Mandatory Questions:**

| # | Question | BLOCKER if |
|---|----------|-----------|
| Q1 | Correctness — logical errors, off-by-ones, race conditions | any found |
| Q2 | Coverage blind spots — untested inputs, silent failures | critical paths uncovered |
| Q3 | Dead code — unreachable, unused, commented-out | any found |
| Q4 | Documentation — public APIs documented, math explained | any public API undocumented |
| Q5 | Error handling — no swallowed errors, no bare unwrap | any found |
| Q6 | DRY — no duplicate blocks >10L at >85% similarity | any found |
| Q7 | Complexity — no function >50L, no file >500L, complexity ≤ 15 | any exceeded |
| Q8 | Security — no injection surface, no sensitive data logged | any found |
| Q9 | Performance — no O(n²) regression, no blocking async | in hot paths |
| Q10 | Konjo Standard — seaworthy under load for 30 days? | ship would sink |

**Output:** structured JSON with `APPROVED` / `WARNING` / `BLOCKER` verdict, posted as a PR comment.

---

## File and Directory Structure

```
repo/
├── .konjo/
│   ├── hooks/
│   │   └── pre-commit          ← Wall 1: the hook script
│   ├── scripts/
│   │   ├── konjo_review.py     ← Wall 3: adversarial review (Claude Opus)
│   │   ├── dry_check.py        ← DRY detector (cross-language, stdlib-only)
│   │   └── install-hooks.sh    ← Bootstrap installer
│   └── deny.toml               ← cargo-deny config (Rust repos)
├── .cargo-husky/
│   └── hooks/
│       └── pre-commit          ← cargo-husky delegate → .konjo/hooks/pre-commit
├── .github/
│   └── workflows/
│       └── konjo-gate.yml      ← Wall 2: CI gate
├── .claude/
│   └── skills/
│       ├── konjo-quality/      ← Quality framework agent reference
│       └── konjo-retrofit/     ← Retrofit protocol for existing repos
└── CLAUDE.md                   ← Project-specific quality rules
```

---

## Installing the Framework in a New Repo

```bash
# 1. Copy the .konjo/ directory from lopi
cp -r /path/to/lopi/.konjo /path/to/target-repo/
cp /path/to/lopi/.github/workflows/konjo-gate.yml /path/to/target-repo/.github/workflows/
cp -r /path/to/lopi/.claude/skills/konjo-quality /path/to/target-repo/.claude/skills/
cp -r /path/to/lopi/.claude/skills/konjo-retrofit /path/to/target-repo/.claude/skills/

# 2. Install hooks
cd /path/to/target-repo
bash .konjo/scripts/install-hooks.sh

# 3. For Rust repos: add cargo-husky
# Add to Cargo.toml [dev-dependencies]:
#   cargo-husky = { version = "1", default-features = false, features = ["user-hooks"] }

# 4. Add ANTHROPIC_API_KEY to GitHub Actions secrets

# 5. Test the hook
git commit --allow-empty -m "test: verify konjo pre-commit hook"
```

---

## Retrofit Protocol (Existing Repos)

Use the `/konjo-retrofit` skill or `.claude/skills/konjo-retrofit/SKILL.md` for the full protocol.

**TL;DR:** measure first, set gates at current baseline minus 2%, install, then ratchet up 5% per sprint until all gates reach their hard limits. Never install a gate that fails on day 1.

---

## Performance Metrics Tracked in CI

Beyond code quality, these metrics are tracked per-run and gated:

### Test Performance
- Total test suite wall time (regression > 10% = WARNING, > 30% = BLOCK)
- Slowest 3 tests reported per run (if any test takes > 5s, investigate)

### Runtime Performance
- Criterion benchmark p50/p95 per benchmark (regression > 5% p50 = BLOCK)
- Peak RSS during benchmark run (regression > 10% = BLOCK)
- Heap allocations per operation (regression > 15% = WARNING)

### API Performance (where applicable)
- Response time p50/p95/p99 for HTTP endpoints
- Throughput (requests/sec) under fixed load
- Error rate under load

All metrics stored in `benchmarks/results/<timestamp>_<git-sha>/` — never overwritten.

---

## Why Each Gate Exists

| Gate | Without it |
|------|-----------|
| Coverage | Tests that don't cover the code tell you nothing about whether the code works |
| Mutation testing | 100% line coverage but 50% mutation survival = tests that can't catch bugs |
| Adversarial review | Self-review catches code against itself, not against intent |
| Dead code | Dead code becomes zombie code: it compiles, occupies cognitive space, and confuses future contributors |
| DRY enforcement | Duplicate code means a bug fixed in one place silently persists in all copies |
| Complexity gate | Functions with cognitive complexity > 15 cannot be reasoned about, tested completely, or maintained |
| Documentation | Undocumented APIs are attack surfaces — you can't audit what you can't read |

---

## Cost Model (Wall 3 API calls)

Wall 3 runs once per PR, uses `claude-opus-4-6`.

Typical diff: 500–2000 tokens input + 3000-4000 tokens output.
System prompt: ~1800 tokens (cached after first call in CI run).

**Estimated cost per PR:** ~$0.02–0.05 (with prompt caching reducing repeated costs).
**At 50 PRs/month:** ~$1–3/month total.

This is the cheapest part of running the framework. Do not disable it to save money.

---

*건조. 根性. Make it Konjo — build, ship, repeat.*
