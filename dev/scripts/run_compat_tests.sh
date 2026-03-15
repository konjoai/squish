#!/usr/bin/env bash
# run_compat_tests.sh — start squish serve, wait for health, run OpenAI compat tests.
#
# Usage:
#   ./dev/scripts/run_compat_tests.sh [--model MODEL] [--port PORT]
#
# Defaults: model=qwen2.5:1.5b  port=8765
#
# Exit code: pytest's exit code (0 = all pass, 1 = failures, 5 = no tests).
set -euo pipefail

SERVE_MODEL="${SERVE_MODEL:-qwen2.5:1.5b}"
SERVE_PORT="${SERVE_PORT:-8765}"
WAIT_TIMEOUT=30
HEALTH_URL="http://localhost:${SERVE_PORT}/health"

# ── parse CLI ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) SERVE_MODEL="$2"; shift 2 ;;
        --port)  SERVE_PORT="$2";  shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# ── start server ─────────────────────────────────────────────────────────────
echo "Starting squish serve --agent --model ${SERVE_MODEL} --port ${SERVE_PORT} …"
python -m squish.server --agent --model "${SERVE_MODEL}" --port "${SERVE_PORT}" &
SERVER_PID=$!

cleanup() {
    echo "Stopping server (PID ${SERVER_PID}) …"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "Server stopped."
}
trap cleanup EXIT

# ── wait for /health ──────────────────────────────────────────────────────────
echo "Waiting up to ${WAIT_TIMEOUT}s for ${HEALTH_URL} …"
elapsed=0
while true; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "${HEALTH_URL}" 2>/dev/null || echo "000")
    if [[ "${code}" == "200" ]]; then
        echo "Server is up (HTTP ${code})."
        break
    fi
    if [[ ${elapsed} -ge ${WAIT_TIMEOUT} ]]; then
        echo "ERROR: server did not respond within ${WAIT_TIMEOUT}s (last code: ${code})." >&2
        exit 1
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

# ── run integration tests ─────────────────────────────────────────────────────
export SQUISH_TEST_BASE_URL="http://localhost:${SERVE_PORT}"
echo "Running tests/integration/test_openai_compat.py …"
EXIT_CODE=0
python -m pytest tests/integration/test_openai_compat.py -v || EXIT_CODE=$?

# ── summary ──────────────────────────────────────────────────────────────────
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "PASS: all OpenAI compat tests passed."
else
    echo "FAIL: OpenAI compat tests exited with code ${EXIT_CODE}."
fi
exit "${EXIT_CODE}"
