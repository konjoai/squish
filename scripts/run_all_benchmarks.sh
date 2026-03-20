#!/usr/bin/env bash
# run_all_benchmarks.sh — Benchmark all locally-available squish models
# Runs smallest → largest, saves per-model markdown + a master summary.
# Usage: bash scripts/run_all_benchmarks.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_DIR/results/benchmarks/$TIMESTAMP"
SUMMARY_FILE="$RESULTS_DIR/BENCHMARK_SUMMARY.md"
PORT=11435
HOST="127.0.0.1"
MAX_TOKENS=256
SERVER_TIMEOUT=360   # max seconds to wait for server ready
LOG_LEVEL="warning"

mkdir -p "$RESULTS_DIR"

# ── Models ordered smallest → largest (by parameter count / disk size) ───────
# Qwen3-14B-bf16 is omitted — only has config/tokenizer files, no weights.
# Qwen3-8B-bf16 is attempted last — 15 GB BF16 may hit 15.5 GB Metal budget.
# Squished Qwen2.5-1.5B variants appear after the base 1.5B model for comparison.
MODELS=(
    "Qwen3-0.6B-bf16"
    "Llama-3.2-1B-Instruct-bf16"
    "gemma-3-1b-it-bf16"
    "Qwen2.5-1.5B-Instruct-bf16"
    "Qwen2.5-1.5B-Instruct-squished-int4-awq"
    "Qwen2.5-1.5B-Instruct-squished-int4-mse"
    "Qwen2.5-1.5B-Instruct-squished-mixed"
    "Qwen2.5-1.5B-Instruct-squished-mixed-v2"
    "Qwen2.5-1.5B-Instruct-squished-mixed-v3"
    "Qwen2.5-1.5B-Instruct-squished-fp16attn-noawq"
    "Qwen2.5-1.5B-Instruct-squished-fp16embed"
    "Qwen2.5-1.5B-Instruct-squished-fp16mlp"
    "Qwen2.5-1.5B-Instruct-squished-g8-mixed"
    "Qwen2.5-1.5B-Instruct-squished-lossless"
    "Qwen2.5-1.5B-Instruct-bf16-compressed"
    "Llama-3.2-3B-Instruct-bf16"
    "Qwen3-4B-bf16"
    "gemma-3-4b-it-bf16"
    "Mistral-7B-Instruct-v0.3-bf16"
    "Qwen2.5-7B-Instruct-bf16"
    "Qwen3-8B-bf16-compressed"
    "Qwen3-8B-bf16"
)

# ── Helpers ──────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_server() {
    local server_pid="$1"
    local deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
    log "  Waiting for server on $HOST:$PORT ..."
    while true; do
        if curl -sf "http://$HOST:$PORT/v1/models" \
               -H "Authorization: Bearer squish" > /dev/null 2>&1; then
            return 0
        fi
        # Fail fast if the server process died
        if ! kill -0 "$server_pid" 2>/dev/null; then
            log "  ERROR: server process exited unexpectedly (PID $server_pid)"
            return 1
        fi
        if [[ $(date +%s) -ge $deadline ]]; then
            log "  ERROR: server did not start within ${SERVER_TIMEOUT}s"
            return 1
        fi
        sleep 3
    done
}

kill_server() {
    local pid="$1"
    if [[ "$pid" -gt 0 ]] && kill -0 "$pid" 2>/dev/null; then
        log "  Stopping server (PID $pid) ..."
        kill "$pid" 2>/dev/null || true
        # Wait up to 20 s for clean exit
        local i=0
        while kill -0 "$pid" 2>/dev/null && [[ $i -lt 20 ]]; do
            sleep 1; i=$((i+1))
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
    # Also free the port if any stale process holds it
    local port_pid
    port_pid=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    if [[ -n "$port_pid" ]]; then
        kill -9 $port_pid 2>/dev/null || true
    fi
    sleep 2
}

# ── Summary header ────────────────────────────────────────────────────────────

cat > "$SUMMARY_FILE" <<HEADER
# Squish — Full Model Benchmark Results

Generated: $(date '+%Y-%m-%d %H:%M:%S') by \`scripts/run_all_benchmarks.sh\`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × ${MAX_TOKENS} max tokens · streams measured via OpenAI-compat API

| Model | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|-------------:|----------:|--------|
HEADER

PASSED=0
FAILED=0

# ── Per-model benchmark loop ─────────────────────────────────────────────────

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

    echo ""
    log "════════════════════════════════════════════════════"
    log "Model: $MODEL_NAME"

    if [[ ! -d "$MODEL_PATH" ]]; then
        log "  SKIP — directory not found: $MODEL_PATH"
        echo "| \`$MODEL_NAME\` | n/a | n/a | SKIP (not found) |" >> "$SUMMARY_FILE"
        continue
    fi

    # Skip squished-only models (no config.json means they can't be served standalone)
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        log "  SKIP — no config.json in $MODEL_PATH (squished-only, needs base model)"
        echo "| \`$MODEL_NAME\` | n/a | n/a | SKIP (no config.json — squished-only) |" >> "$SUMMARY_FILE"
        continue
    fi

    SAFE_NAME="${MODEL_NAME//\//_}"
    MODEL_BENCH_FILE="$RESULTS_DIR/${SAFE_NAME}.md"
    MODEL_LOG="$RESULTS_DIR/${SAFE_NAME}.server.log"

    # Kill any stale server (port-only cleanup, no PID)
    kill_server 0 2>/dev/null || true

    # Start server in background
    log "  Starting server for $MODEL_NAME ..."
    python3 "$REPO_DIR/squish/cli.py" run "$MODEL_PATH" \
        --port $PORT \
        --log-level $LOG_LEVEL \
        > "$MODEL_LOG" 2>&1 &
    SERVER_PID=$!
    log "  Server PID: $SERVER_PID"

    # Wait for ready
    if ! wait_for_server "$SERVER_PID"; then
        log "  FAILED: server startup timed out"
        kill_server "$SERVER_PID"
        echo "| \`$MODEL_NAME\` | n/a | n/a | FAIL (startup timeout) |" >> "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi
    log "  Server ready."

    # Run benchmark
    log "  Running bench (max_tokens=$MAX_TOKENS) ..."
    BENCH_EXIT=0
    BENCH_OUTPUT=""
    BENCH_OUTPUT=$(python3 "$REPO_DIR/squish/cli.py" bench \
        --port $PORT \
        --max-tokens $MAX_TOKENS \
        --markdown \
        --save "$MODEL_BENCH_FILE" \
        2>&1) || BENCH_EXIT=$?

    if [[ $BENCH_EXIT -ne 0 ]]; then
        log "  FAILED: bench command returned exit $BENCH_EXIT"
        echo "$BENCH_OUTPUT"
        kill_server "$SERVER_PID"
        echo "| \`$MODEL_NAME\` | n/a | n/a | FAIL (bench error) |" >> "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        continue
    fi

    log "  Bench complete. Results → $MODEL_BENCH_FILE"
    echo "$BENCH_OUTPUT"

    # Extract avg TTFT and avg Tok/s from the Average row:
    # Format:  | **Average** | **116** | — | **46.4** |
    if [[ -f "$MODEL_BENCH_FILE" ]]; then
        AVG_ROW=$(grep "Average" "$MODEL_BENCH_FILE" | head -1 || true)
        if [[ -n "$AVG_ROW" ]]; then
            # Extract bold numbers from: | **Average** | **116** | — | **46.4** |
            # Use macOS-compatible extended grep (-E) instead of Perl (-P)
            AVG_TTFT=$(echo "$AVG_ROW" | grep -oE '\*\*[0-9]+\*\*' | head -1 | tr -d '*' || echo "?")
            AVG_TPS=$(echo "$AVG_ROW"  | grep -oE '\*\*[0-9]+\.?[0-9]*\*\*' | tail -1 | tr -d '*' || echo "?")
            echo "| \`$MODEL_NAME\` | $AVG_TTFT | $AVG_TPS | OK |" >> "$SUMMARY_FILE"
        else
            echo "| \`$MODEL_NAME\` | (see file) | (see file) | OK |" >> "$SUMMARY_FILE"
        fi
    fi

    kill_server "$SERVER_PID"
    PASSED=$((PASSED + 1))
    log "  Done with $MODEL_NAME."
done

# ── Footer ────────────────────────────────────────────────────────────────────

cat >> "$SUMMARY_FILE" <<FOOTER

---
**Run completed**: $(date '+%Y-%m-%d %H:%M:%S')  
**Passed**: $PASSED / $((PASSED + FAILED)) models  
**Results dir**: \`$RESULTS_DIR\`

Individual markdown tables saved as \`<model_name>.md\` in the results directory.
FOOTER

echo ""
log "════════════════════════════════════════════════════"
log "BENCHMARK COMPLETE: $PASSED passed, $FAILED failed"
log "Summary: $SUMMARY_FILE"
log "Results dir: $RESULTS_DIR"
cat "$SUMMARY_FILE"
