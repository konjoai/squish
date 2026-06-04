#!/usr/bin/env bash
# Run with: bash scripts/test_cli.sh
# Requires: squish installed and on PATH (or python3 -m squish.cli available)

set -euo pipefail

# Detect how to invoke squish: prefer the installed binary, fall back to the
# module invocation when running from a source tree.
if command -v squish &>/dev/null; then
    SQUISH=squish
elif python3 -m squish.cli --version &>/dev/null 2>&1; then
    SQUISH="python3 -m squish.cli"
else
    echo "ERROR: squish not found. Install it or run from the source tree." >&2
    exit 1
fi

PASS=0
FAIL=0
SKIP=0

pass() { echo "PASS  $1"; PASS=$((PASS + 1)); }
fail() { echo "FAIL  $1 — $2"; FAIL=$((FAIL + 1)); }
skip() { echo "SKIP  $1 — $2"; SKIP=$((SKIP + 1)); }

run_cmd() {
    # run_cmd <timeout_secs> <args...>
    # Captures combined stdout+stderr into $CMD_OUTPUT; exit code into $CMD_EXIT.
    local timeout_secs="$1"; shift
    # shellcheck disable=SC2086
    CMD_OUTPUT=$(timeout "$timeout_secs" $SQUISH "$@" 2>&1) && CMD_EXIT=0 || CMD_EXIT=$?
}

echo "=== squish CLI integration tests ==="
echo

# ── squish --version ──────────────────────────────────────────────────────────
run_cmd 10 --version
if echo "$CMD_OUTPUT" | grep -qE '[0-9]+\.[0-9]+\.[0-9]+'; then
    pass "squish --version (version string found: $(echo "$CMD_OUTPUT" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1))"
else
    fail "squish --version" "no semver string in output: $CMD_OUTPUT"
fi

# ── squish --help ─────────────────────────────────────────────────────────────
run_cmd 10 --help
MISSING=""
for kw in run catalog doctor; do
    echo "$CMD_OUTPUT" | grep -q "$kw" || MISSING="$MISSING $kw"
done
if [ -z "$MISSING" ]; then
    pass "squish --help (keywords found: run catalog doctor)"
else
    fail "squish --help" "missing keywords:$MISSING"
fi

# ── squish catalog ────────────────────────────────────────────────────────────
run_cmd 15 catalog
if [ "$CMD_EXIT" -ne 0 ]; then
    fail "squish catalog" "exited $CMD_EXIT"
elif ! echo "$CMD_OUTPUT" | grep -q "qwen3:8b"; then
    fail "squish catalog" "output does not contain 'qwen3:8b'"
elif ! echo "$CMD_OUTPUT" | grep -q "⚡ yes"; then
    fail "squish catalog" "output does not contain '⚡ yes' (no prebuilt marker)"
else
    pass "squish catalog"
fi

# ── squish search qwen3 ───────────────────────────────────────────────────────
run_cmd 15 search qwen3
MISSING_SEARCH=""
for model in "qwen3:0.6b" "qwen3:8b"; do
    echo "$CMD_OUTPUT" | grep -q "$model" || MISSING_SEARCH="$MISSING_SEARCH $model"
done
if [ "$CMD_EXIT" -ne 0 ]; then
    fail "squish search qwen3" "exited $CMD_EXIT"
elif [ -n "$MISSING_SEARCH" ]; then
    fail "squish search qwen3" "missing models:$MISSING_SEARCH"
else
    pass "squish search qwen3"
fi

# ── squish search fakething123 ────────────────────────────────────────────────
run_cmd 15 search fakething123
if [ "$CMD_EXIT" -ne 0 ]; then
    pass "squish search fakething123 (exited non-zero as expected)"
elif echo "$CMD_OUTPUT" | grep -qiE "no (catalog entries|results|matches)"; then
    pass "squish search fakething123 (no-results message found)"
else
    fail "squish search fakething123" "expected non-zero exit or no-results message; got exit $CMD_EXIT"
fi

# ── squish models ─────────────────────────────────────────────────────────────
run_cmd 15 models
if [ "$CMD_EXIT" -eq 0 ]; then
    pass "squish models (exited 0)"
else
    fail "squish models" "exited $CMD_EXIT"
fi

# ── squish run fakefamily:99b ─────────────────────────────────────────────────
run_cmd 15 run fakefamily:99b
if [ "$CMD_EXIT" -eq 0 ]; then
    fail "squish run fakefamily:99b" "expected non-zero exit but got 0"
elif echo "$CMD_OUTPUT" | grep -qiE "unknown model|not found"; then
    pass "squish run fakefamily:99b (error message and non-zero exit)"
else
    fail "squish run fakefamily:99b" "expected 'Unknown model' or 'not found' in output (exit $CMD_EXIT); got: $(echo "$CMD_OUTPUT" | head -3)"
fi

# ── squish doctor ─────────────────────────────────────────────────────────────
# Only FAIL if the command itself crashes; individual checks may fail in CI.
run_cmd 30 doctor
if [ "$CMD_EXIT" -ne 0 ]; then
    fail "squish doctor" "command crashed with exit $CMD_EXIT"
elif ! echo "$CMD_OUTPUT" | grep -q "macOS / Apple Silicon"; then
    fail "squish doctor" "output does not contain 'macOS / Apple Silicon' check label"
else
    pass "squish doctor (command ran without crashing)"
fi

# ── squish compat ─────────────────────────────────────────────────────────────
run_cmd 15 compat
if [ "$CMD_EXIT" -ne 0 ]; then
    fail "squish compat" "exited $CMD_EXIT"
elif ! echo "$CMD_OUTPUT" | grep -q "OPENAI_BASE_URL"; then
    fail "squish compat" "output does not contain 'OPENAI_BASE_URL'"
else
    pass "squish compat"
fi

# ── squish ps ────────────────────────────────────────────────────────────────
run_cmd 15 ps
if [ "$CMD_EXIT" -eq 0 ]; then
    pass "squish ps (exited 0)"
else
    fail "squish ps" "exited $CMD_EXIT"
fi

# ── squish config ────────────────────────────────────────────────────────────
run_cmd 15 config
if [ "$CMD_EXIT" -eq 0 ]; then
    pass "squish config (exited 0)"
else
    fail "squish config" "exited $CMD_EXIT"
fi

# ── Commands requiring a running server (skipped) ─────────────────────────────
skip "squish chat" "requires a running server"
skip "squish run <real-model>" "requires a running server and downloaded model"

# ── Summary ───────────────────────────────────────────────────────────────────
echo
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="
echo

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
