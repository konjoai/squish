---
name: konjo-retrofit
description: Retrofit the Konjo Quality Framework onto an existing repo that predates it. Use when asked to add konjo quality gates, improve code quality, audit an existing codebase, or run a quality sprint on any repo. Provides the step-by-step migration plan and triage protocol.
user-invocable: true
---

# Konjo Retrofit — Existing Repo Quality Migration

## The Problem With Retrofitting Blind

Installing hard quality gates on an existing codebase without measuring first causes one of two outcomes:
1. **Gates fail on day 1** — blocks all work, team disables the gates in frustration
2. **Gates are set too loose** — they pass everything, provide no value

The Retrofit Protocol solves this by measuring before gating, then ratcheting up incrementally.

---

## Step 1 — Baseline Audit (measure everything, fix nothing yet)

Run these and save the output. Do not fix violations yet — just establish the baseline.

```bash
# Coverage baseline
cargo llvm-cov nextest --workspace --json > coverage_baseline.json    # Rust
python -m pytest --cov --cov-report=json > coverage_baseline.json     # Python

# Lint baseline
cargo clippy --workspace --all-targets --message-format json > clippy_baseline.json  # Rust
ruff check --output-format json > ruff_baseline.json                                  # Python

# Dead code
RUSTFLAGS="-W dead_code" cargo check --workspace 2>&1 | grep "dead_code" > dead_code.txt  # Rust
vulture . --min-confidence 60 > vulture_baseline.txt                                        # Python

# Complexity
radon cc . -n C > complexity_baseline.txt        # Python (grade C = cyclomatic > 10)
# For Rust: cargo clippy -W clippy::cognitive_complexity 2>&1 | grep "cognitive_complexity"

# DRY
python3 .konjo/scripts/dry_check.py --json > dry_baseline.json

# File sizes
find . -name "*.rs" -o -name "*.py" | grep -v target | \
  xargs wc -l | sort -n | tail -20 > large_files.txt

# Mutation testing (sample — don't run full corpus on first audit)
cargo mutants --lib --jobs 2 2>&1 | tail -50 > mutation_sample.txt   # Rust
mutmut run --paths-to-mutate src/ 2>&1 | tail -50 > mutation_sample.txt  # Python
```

---

## Step 2 — Triage

Classify all violations into three categories:

| Priority | Category | Definition | Handle |
|----------|----------|------------|--------|
| P0 | **CRITICAL** | Security issues, data corruption bugs, race conditions | Fix immediately, before framework install |
| P1 | **DEBT** | Coverage < 60%, unwrap() in production paths, undocumented public APIs | Fix in first 2 sprints |
| P2 | **STYLE** | Length violations, moderate duplication, complexity > 20 | Fix incrementally, 1-2 per sprint |

**Never retrofit CRITICAL and DEBT in the same commit.** Fix P0 first, measure, commit. Then P1.

---

## Step 3 — Install Framework (at current baseline, not ideal baseline)

Install with **warn-only mode** for the first sprint:

```bash
# Copy framework files
bash .konjo/scripts/install-hooks.sh

# Install CI with warn-only gates
# Set coverage gate at: current_coverage - 2% (don't let it regress, but don't block yet)
# Set dead code gate to: warn, not block
# Set Wall 3 review to: --soft-fail (report only)
```

**Rationale:** A gate that fails on the day it's installed will be disabled. Set it just above current, then ratchet.

---

## Step 4 — The Coverage Ratchet

Coverage gates increment on a fixed schedule. Never move the gate backward.

| Sprint | Coverage Gate | Action |
|--------|-------------|--------|
| Sprint 0 (install) | current - 2% | No regressions allowed |
| Sprint 1 | current | Hold the line |
| Sprint 2 | current + 5% | Write missing tests |
| Sprint 3 | current + 10% | Continue |
| Sprint N | 80% | Hard floor (production gate) |
| Long-term | 95% | Target |

Add to PLAN.md: the current coverage gate value. Never let anyone lower it.

---

## Step 5 — Complexity and Length Ratchet

For each oversized function or file:

**Protocol (one function at a time):**
1. Write a **characterization test** first — captures current behavior before touching anything
2. Make the smallest refactor that improves the metric
3. Run full test suite — all tests must pass
4. Run coverage — must not decrease
5. Commit: `refactor(scope): extract X into Y [complexity: 23→11 lines: 80→35]`

**Never refactor + change behavior in the same commit.**

---

## Step 6 — DRY Cleanup

Run `dry_check.py --json` and sort violations by similarity descending.
Address highest-similarity violations first (easiest wins).

For each violation:
1. Identify the canonical location (the version with the best name/tests)
2. Write a test for the canonical version if it doesn't have one
3. Replace all duplicates with calls to the canonical version
4. Confirm tests pass

---

## Step 7 — Activate Wall 3 (adversarial review)

After Walls 1 and 2 are stable (at least 2 consecutive clean CI runs):

```bash
# Week 1: soft-fail mode (report only, does not block merge)
python3 .konjo/scripts/konjo_review.py --soft-fail

# Week 2: full blocking mode
python3 .konjo/scripts/konjo_review.py  # exits 1 on BLOCKER
```

Add `ANTHROPIC_API_KEY` to GitHub Actions secrets before enabling.

---

## Repo-Type Checklists

### Pure Rust (lopi)
- [ ] `cargo audit` clean
- [ ] `cargo deny check` configured and clean
- [ ] `cargo-husky` dev-dep added to Cargo.toml
- [ ] `cargo llvm-cov` gives ≥ 80% coverage
- [ ] `clippy -D unwrap_used` passes
- [ ] `RUSTDOCFLAGS="-D missing_docs" cargo doc` passes
- [ ] `.konjo/deny.toml` committed
- [ ] `.github/workflows/konjo-gate.yml` active

### Pure Python (squish, kyro, miru, kairu, squash)
- [ ] `ruff check` clean; `ruff format --check` clean
- [ ] `mypy --strict` clean (or each `# type: ignore` justified)
- [ ] `vulture` < 5% dead code
- [ ] `pytest --cov --cov-fail-under=80` passes
- [ ] `radon cc -n C` zero grade-C functions
- [ ] `interrogate --fail-under 80` on public APIs
- [ ] `pre-commit` framework installed with `.pre-commit-config.yaml`
- [ ] `.github/workflows/konjo-gate.yml` adapted for Python

### Rust + Python hybrid (kohaku, toki, drex, vectro)
- [ ] Both Rust and Python checklists above
- [ ] Python-only mode is always the correctness baseline
- [ ] Rust and Python implementations agree to documented tolerance
- [ ] PyO3 bindings tested from both sides

---

## Documenting the Retrofit Sprint

After each retrofit sprint, add to CHANGELOG.md:

```markdown
## [vX.Y.Z] — YYYY-MM-DD — Quality Retrofit Sprint N

### Improved
- Coverage: X% → Y%
- Dead code: N warnings → 0
- Functions refactored: list them
- DRY violations resolved: list them

### Quality Gates Now Active
- Coverage gate: Z%
- Wall 1 (pre-commit): active
- Wall 2 (CI): active
- Wall 3 (adversarial review): soft-fail / blocking
```

---

## The Shipbuilder's Checklist (Final Verification)

Before declaring retrofit complete, answer these:

- [ ] The test suite runs in < 5 minutes (if not, profile and optimize)
- [ ] A PR that deletes a function without updating its callers would fail CI
- [ ] A PR that introduces an unwrap() would fail CI
- [ ] A PR that drops coverage below 80% would fail CI
- [ ] A PR is reviewed by the Konjo Critic before merge
- [ ] Running the code for 30 days without restart would not cause memory growth
- [ ] The code can be read and extended by someone who wasn't there when it was written

If any of these are false, the retrofit is not complete.
