---
name: konjo-quality
description: Konjo Code Quality Framework — all gate definitions, thresholds, tools, and enforcement points. Auto-load when writing tests, reviewing code quality, refactoring, or when quality gate failures are mentioned. Applies the Three-Wall framework to prevent AI slop.
user-invocable: true
---

# Konjo Quality Framework — Agent Reference

## Why This Exists

AI-assisted code produces **1.7× more logical and correctness bugs** than traditional development (CodeRabbit 2026). AI agents change tests so broken code passes instead of fixing the code. AI self-review is architecturally circular — it checks code against itself, not against intent. This framework provides external ground truth via three independent walls that cannot be reasoned past.

The Konjo Critic (Wall 3) uses `claude-opus-4-6` in a separate session. The builder has blind spots from the construction process; the critic comes in cold with a different capability profile to reduce correlated failures.

---

## The Three Walls

| Wall | When | What | Blocks |
|------|------|------|--------|
| **Wall 1** | Pre-commit hook | Format, lint, unwrap scan, DRY (staged only), TODO scan | The commit |
| **Wall 2** | CI / GitHub Actions | Coverage, mutation, complexity, size, docs, audit, review | The merge |
| **Wall 3** | CI (PRs only) | Claude Opus adversarial review against 10 mandatory questions | The merge |

---

## Hard Quality Thresholds (all enforced by CI)

| Metric | Hard Block | Target | Tool |
|--------|-----------|--------|------|
| Line coverage | ≥ 80% | ≥ 95% | cargo-llvm-cov / pytest-cov |
| Mutation survival (changed files) | ≤ 10% | 0% | cargo-mutants / mutmut |
| Cognitive complexity per function | ≤ 15 | ≤ 10 | clippy / radon |
| Lint violations | 0 | 0 | clippy / ruff |
| Dead code warnings | 0 | 0 | rustc / vulture |
| Undocumented public APIs | 0 | 0 | rustdoc / interrogate |
| p50 latency regression | ≤ 5% | 0% | criterion / pytest-benchmark |
| Function body length | ≤ 50 lines | ≤ 30 lines | wc / radon |
| File length | ≤ 500 lines | ≤ 300 lines | wc |
| DRY violations (>10L, >85% similar) | 0 | 0 | dry_check.py |
| unwrap()/expect() in non-test Rust | 0 | 0 | clippy::unwrap_used |
| Silent error swallowing | 0 | 0 | grep / ast check |
| Known CVEs in dependencies | 0 | 0 | cargo-audit / safety |

---

## Before Writing Any Code

1. **State the purpose in one sentence.** If you can't state it clearly, don't write it.
2. **Search the codebase first.** `rg "similar_function"` before writing a new one. If it exists: extend it.
3. **Write the test first** (TDD preferred, mandatory for anything non-trivial).
4. Check: could this be a method on an existing type? A trait implementation? A generic over an existing abstraction?

---

## Line Budgets (hard enforcement)

- **Function body**: 50 lines maximum. 30 target. Stop at 40 and extract.
- **File**: 500 lines maximum. 300 target. Create a new module at 400.
- **Cognitive complexity per function**: ≤ 15 (clippy::cognitive_complexity).
- If approaching a limit: **split now**, not as a TODO.

---

## Zero-Tolerance Rules

These cause a CI BLOCKER. No exceptions. No bypasses.

**All languages:**
- Dead code (functions, variables, imports never used)
- Commented-out code
- TODO/FIXME/HACK in production code
- Undocumented public APIs
- Silent error swallowing (bare `except:` or catch-all with no log)
- Duplicate code blocks (>10 lines, >85% similar) — abstract into shared function
- Tests that test implementation rather than behavior

**Rust:**
- `unwrap()`, `expect()`, `panic!()`, `todo!()`, `unimplemented!()` outside test code
- `dbg!()`, `print!()`, `println!()` in non-test code
- `unsafe` without a comment explaining the invariant that makes it safe

**Python:**
- `except:` or `except Exception:` without explicit `log.warning` and re-raise
- Implicit `None` returns from functions with documented return types
- Mutable default arguments

---

## DRY Mandate

Before writing any function:
1. `rg "similar_function_name"` — does this already exist?
2. Could this be a method on an existing type?
3. Could this be a generic function over a trait/protocol?
4. Would an iterator adapter replace this 20-line loop?
5. Is there a standard library function that does this?

**If any answer is yes: use the existing abstraction or create a shared one.**
Never duplicate. Never copy-paste. Extract.

---

## Test Requirements

Every code file must have a corresponding test file.
Every public function requires at minimum:
- One test for the happy path
- One test for each error variant
- One test for edge cases (empty, zero, max, overflow, concurrent access)

**Tests must test BEHAVIOR, not IMPLEMENTATION.**
- Don't test internal field values — test observable outputs
- Don't mock what you are testing
- Don't write a test that passes even if you delete the function it's testing (mutation testing will catch this)

---

## The Ten Review Questions (Wall 3 will ask all of these)

Ask yourself these before pushing. The Critic will ask them again independently:

1. **Q1 Correctness** — Does this code actually do what it claims? Any logical errors, off-by-ones, race conditions?
2. **Q2 Coverage Blind Spots** — What inputs would cause silent failure the tests won't catch? Are error paths tested?
3. **Q3 Dead Code** — Any unreachable code, unused variable, commented-out block? Zero tolerance.
4. **Q4 Documentation** — Every public API documented? Does it match the implementation? Math in comments?
5. **Q5 Error Handling** — Any errors swallowed? Any bare except/unwrap? Fallbacks that mask real failures?
6. **Q6 DRY** — Any block of logic appearing >once at >85% similarity over >10 lines?
7. **Q7 Complexity** — Any function >50 lines, >15 cognitive complexity, any file >500 lines?
8. **Q8 Security** — Prompt injection? Logging sensitive data? Missing validation? Missing rate limits?
9. **Q9 Performance** — Any O(n²) where O(n log n) is obvious? Blocking I/O on async? Unnecessary allocation?
10. **Q10 Konjo Standard** — Is this seaworthy under 10,000 concurrent requests for 30 days? What would you cut?

---

## Running the Gates Locally

```bash
# Wall 1 equivalent (run before every commit):
cargo fmt --all
cargo clippy --workspace -- -D warnings -D clippy::unwrap_used -D clippy::pedantic
cargo nextest run --workspace --lib
cargo audit

# Coverage check:
cargo llvm-cov nextest --workspace --fail-under-lines 80

# DRY check:
python3 .konjo/scripts/dry_check.py --staged-only

# Wall 3 preview (requires ANTHROPIC_API_KEY):
git diff HEAD~1 | python3 .konjo/scripts/konjo_review.py
```

---

## Language-Specific Addenda

### Rust
- `cargo clippy -- -D warnings -D clippy::pedantic -D clippy::unwrap_used` must be clean
- `RUSTDOCFLAGS="-D missing_docs" cargo doc --workspace --no-deps` must be clean
- `cargo audit` zero advisories; `cargo deny check` zero license/ban violations
- Accumulate in FP32 for quantized matmuls; document any deviation with a benchmark

### Python
- `ruff check` zero violations; `ruff format --check` clean
- `mypy --strict` clean (or explicit `# type: ignore` with justification comment)
- `vulture` zero dead code; `radon cc -n C` zero functions above grade C (cyclomatic complexity > 10)
- `interrogate --fail-under 100` on public API surfaces
- `bandit -ll` zero high-severity issues

### Mixed Rust + Python
- Python-only mode is always the correctness baseline for SIMD/FFI code
- Rust and Python implementations of the same algorithm must agree to within documented tolerance
- PyO3 bindings: test both the Rust side and the Python side independently

---

## Install the Framework in a New Repo

```bash
# Copy .konjo/ from lopi
cp -r /path/to/lopi/.konjo /path/to/target-repo/
cp /path/to/lopi/.github/workflows/konjo-gate.yml /path/to/target-repo/.github/workflows/

# Install hooks
bash .konjo/scripts/install-hooks.sh

# Add ANTHROPIC_API_KEY to GitHub Actions secrets (Settings → Secrets)
```
