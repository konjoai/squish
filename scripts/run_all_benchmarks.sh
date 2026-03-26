#!/usr/bin/env bash
# run_all_benchmarks.sh — Benchmark all locally-available squish models
#
# Runs two tiers for models that support them:
#   • squish  — default (all optimizations on) — squish performance numbers
#   • stock   — --stock (no squish optimizations, comparable to Ollama/mlx_lm)
#
# Squished variants are supported via the fixed _resolve_model auto-detection
# (pass squished dir as the model path; base model config is auto-found).
#
# Usage:
#   bash scripts/run_all_benchmarks.sh              # squish tier only (default)
#   BENCH_STOCK=1 bash scripts/run_all_benchmarks.sh  # also run stock comparison tier

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$REPO_DIR/models"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$REPO_DIR/results/benchmarks/$TIMESTAMP"
SUMMARY_FILE="$RESULTS_DIR/BENCHMARK_SUMMARY.md"
PORT=11435
HOST="127.0.0.1"
MAX_TOKENS=256
SERVER_TIMEOUT=900
LOG_LEVEL="warning"
BENCH_STOCK="${BENCH_STOCK:-0}"
BENCH_MAXOPT="${BENCH_MAXOPT:-0}"
# Hard ceiling for squish INT4 dir size (GB).  On M3 16 GB, loading a squish
# INT4 dir larger than this fills Metal memory and causes an OOM crash.  The
# squish format stores BF16 passthrough tensors alongside INT4 weights, so a
# large model's INT4 dir can be 14 GB even though only ~5 GB is quantized.
# Measured safe ceiling: gemma-3-4b-it-int4 (8.7 GB) passes; anything ≥ 12 GB
# (Qwen3-4B-int4, Qwen2.5-7B-Instruct-int4, Qwen3-8B-int4) crashes the host.
MAX_MODEL_DISK_GB=9

mkdir -p "$RESULTS_DIR"

# ── Models ordered smallest → largest ────────────────────────────────────────
# - Pass BF16 dirs for models that need auto-compress; cli.py resolves to INT4.
# - Native MLX INT3 dirs (config.json with quantization={bits:3}) are safe to
#   pass directly — _model_is_already_quantized() skips auto-compress for them.
# - Qwen3-14B-bf16 omitted: config/tokenizer only, no weights.
# - Mistral-7B-Instruct-v0.3-bf16 omitted: weights never downloaded.
#
# EXCLUDED (OOM on M3 16 GB — squish INT4 dir ≥ 12 GB loads fully into Metal):
#   Qwen3-4B-bf16           → Qwen3-4B-int4           = 14 GB → crash
#   Qwen2.5-7B-Instruct-bf16→ Qwen2.5-7B-Instruct-int4= 14 GB → crash
#   Qwen3-8B-bf16           → Qwen3-8B-int4           = 14 GB → crash
MODELS=(
    # ── ≤ 1.5B BF16: squish INT4 dirs ≤ 5.4 GB, safe on M3 16 GB ──────────
    "Qwen3-0.6B-bf16"
    "Llama-3.2-1B-Instruct-bf16"
    "gemma-3-1b-it-bf16"
    "Qwen2.5-1.5B-Instruct-bf16"
    # ── 1B native MLX INT3: 606 MB, safe ─────────────────────────────────────
    "Llama-3.2-1B-Instruct-int3"
    # ── 3B BF16: int4 dir = 6 GB, safe ──────────────────────────────────────
    "Llama-3.2-3B-Instruct-bf16"
    # ── 4B via safe squish INT4 dir (8.7 GB ≤ ceiling) ───────────────────────
    "gemma-3-4b-it-bf16"
    # ── 8B native MLX INT3: 3.8 GB, no BF16 passthrough, safe ───────────────
    "Qwen3-8B-int3"
)

# ── Helpers ───────────────────────────────────────────────────────────────────

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
        local i=0
        while kill -0 "$pid" 2>/dev/null && [[ $i -lt 20 ]]; do
            sleep 1; i=$((i+1))
        done
        kill -9 "$pid" 2>/dev/null || true
    fi
    local port_pid
    port_pid=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    if [[ -n "$port_pid" ]]; then
        kill -9 $port_pid 2>/dev/null || true
    fi
    sleep 2
}

# Extract avg TTFT (ms) and avg Tok/s from a bench markdown file.
# Average row format: | **Average** | **4093** | — | **61.4** |
extract_avg() {
    local f="$1"
    local AVG_ROW
    AVG_ROW=$(grep "Average" "$f" 2>/dev/null | head -1 || true)
    if [[ -n "$AVG_ROW" ]]; then
        AVG_TTFT=$(echo "$AVG_ROW" | grep -oE '\*\*[0-9]+\*\*' | head -1 | tr -d '*' || echo "?")
        AVG_TPS=$(echo "$AVG_ROW"  | grep -oE '\*\*[0-9]+\.?[0-9]*\*\*' | tail -1 | tr -d '*' || echo "?")
    else
        AVG_TTFT="?"
        AVG_TPS="?"
    fi
}

# Run one bench tier for the current model.
# $1 = tier name ("baseline" or "optimized")
# $2 = extra server flags (e.g. "--all-optimizations")
bench_tier() {
    local TIER="$1"
    local EXTRA_FLAGS="$2"
    local SAFE_NAME="${MODEL_NAME//\//_}"
    local MODEL_BENCH_FILE="$RESULTS_DIR/${SAFE_NAME}_${TIER}.md"
    local MODEL_LOG="$RESULTS_DIR/${SAFE_NAME}_${TIER}.server.log"

    kill_server 0

    log "  [$TIER] Starting server for $MODEL_NAME ..."
    # shellcheck disable=SC2086
    python3 "$REPO_DIR/squish/cli.py" run "$MODEL_PATH" \
        --port $PORT \
        --log-level $LOG_LEVEL \
        $EXTRA_FLAGS \
        > "$MODEL_LOG" 2>&1 &
    SERVER_PID=$!
    log "  [$TIER] Server PID: $SERVER_PID"

    if ! wait_for_server "$SERVER_PID"; then
        log "  [$TIER] FAILED: server startup timed out or crashed"
        kill_server "$SERVER_PID"
        echo "| \`$MODEL_NAME\` | $TIER | n/a | n/a | FAIL (startup) |" >> "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        return 1
    fi
    log "  [$TIER] Server ready."

    log "  [$TIER] Running bench (max_tokens=$MAX_TOKENS) ..."
    BENCH_EXIT=0
    BENCH_OUTPUT=$(python3 "$REPO_DIR/squish/cli.py" bench \
        --port $PORT \
        --max-tokens $MAX_TOKENS \
        --markdown \
        --save "$MODEL_BENCH_FILE" \
        2>&1) || BENCH_EXIT=$?

    echo "$BENCH_OUTPUT"

    if [[ $BENCH_EXIT -ne 0 ]]; then
        log "  [$TIER] FAILED: bench returned exit $BENCH_EXIT"
        kill_server "$SERVER_PID"
        echo "| \`$MODEL_NAME\` | $TIER | n/a | n/a | FAIL (bench error) |" >> "$SUMMARY_FILE"
        FAILED=$((FAILED + 1))
        return 1
    fi

    log "  [$TIER] Bench complete → $MODEL_BENCH_FILE"
    extract_avg "$MODEL_BENCH_FILE"

    # If all prompts errored (OOM / server crash mid-run), avg will be "?"
    if [[ "$AVG_TTFT" == "?" && "$AVG_TPS" == "?" ]]; then
        log "  [$TIER] FAILED: no valid results (server likely OOM or crashed mid-run)"
        echo "| \`$MODEL_NAME\` | $TIER | n/a | n/a | FAIL (OOM/crash) |" >> "$SUMMARY_FILE"
        kill_server "$SERVER_PID"
        FAILED=$((FAILED + 1))
        return 1
    fi

    echo "| \`$MODEL_NAME\` | $TIER | $AVG_TTFT | $AVG_TPS | OK |" >> "$SUMMARY_FILE"

    kill_server "$SERVER_PID"
    PASSED=$((PASSED + 1))
    return 0
}

# ── Summary header ────────────────────────────────────────────────────────────

cat > "$SUMMARY_FILE" <<HEADER
# Squish — Full Model Benchmark Results

Generated: $(date '+%Y-%m-%d %H:%M:%S') by \`scripts/run_all_benchmarks.sh\`

Platform: Apple M3 · 17 GB Unified RAM · MLX Metal backend  
Benchmark: 4 prompts × ${MAX_TOKENS} max tokens · streams measured via OpenAI-compat API  
Tiers: squish=default (auto-profile/blazing) · maxopt=--all-optimizations · stock=--stock (Ollama-comparable)

| Model | Tier | Avg TTFT (ms) | Avg Tok/s | Status |
|-------|------|-------------:|----------:|--------|
HEADER

PASSED=0
FAILED=0

# ── Per-model benchmark loop ──────────────────────────────────────────────────

for MODEL_NAME in "${MODELS[@]}"; do
    MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

    echo ""
    log "════════════════════════════════════════════════════"
    log "Model: $MODEL_NAME"

    if [[ ! -d "$MODEL_PATH" ]]; then
        log "  SKIP — directory not found: $MODEL_PATH"
        echo "| \`$MODEL_NAME\` | — | n/a | n/a | SKIP (not found) |" >> "$SUMMARY_FILE"
        continue
    fi

    # OOM guard: if the model's auto-resolved INT4 dir exceeds the ceiling, skip.
    # Strips -bf16 / -fp16 suffix to reconstruct the squish INT4 dir name, then
    # measures its size.  Native MLX INT3 dirs have no -int4 sibling, so they
    # pass through untouched (they load only quantized weights, no BF16 copies).
    _oom_base="${MODEL_NAME%-bf16}"; _oom_base="${_oom_base%-fp16}"
    _oom_int4_dir="$MODELS_DIR/${_oom_base}-int4"
    if [[ -d "$_oom_int4_dir" ]]; then
        _oom_gb=$(du -sk "$_oom_int4_dir" 2>/dev/null | awk '{printf "%.0f", $1/1024/1024}')
        if [[ "${_oom_gb:-0}" -gt "${MAX_MODEL_DISK_GB}" ]]; then
            log "  SKIP (OOM guard) — ${MODEL_NAME} resolves to ${_oom_base}-int4 (${_oom_gb} GB > ${MAX_MODEL_DISK_GB} GB ceiling)"
            echo "| \`$MODEL_NAME\` | — | n/a | n/a | SKIP (OOM: ${_oom_gb}GB > ${MAX_MODEL_DISK_GB}GB) |" >> "$SUMMARY_FILE"
            continue
        fi
    fi

    # Run squish tier (default optimizations — auto-profile / blazing mode)
    bench_tier "squish" "" || true

    # Run maxopt tier when BENCH_MAXOPT=1 (--all-optimizations: reproduces README-claimed numbers)
    if [[ "$BENCH_MAXOPT" == "1" ]]; then
        bench_tier "maxopt" "--all-optimizations" || true
    fi

    # Run stock tier when BENCH_STOCK=1 (plain mlx_lm, no squish optimizations)
    if [[ "$BENCH_STOCK" == "1" ]]; then
        bench_tier "stock" "--stock" || true
    fi

    log "  Done with $MODEL_NAME."
done

# ── Footer ────────────────────────────────────────────────────────────────────

cat >> "$SUMMARY_FILE" <<FOOTER

---
**Run completed**: $(date '+%Y-%m-%d %H:%M:%S')  
**Passed**: $PASSED / $((PASSED + FAILED))  
**Results dir**: \`$RESULTS_DIR\`

Individual markdown tables saved as \`<model>_<tier>.md\` in the results directory.
FOOTER

echo ""
log "════════════════════════════════════════════════════"
log "BENCHMARK COMPLETE: $PASSED passed, $FAILED failed"
log "Summary: $SUMMARY_FILE"
log "Results dir: $RESULTS_DIR"
cat "$SUMMARY_FILE"


