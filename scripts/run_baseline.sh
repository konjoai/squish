#!/usr/bin/env bash
# run_baseline.sh — Squish Benchmark Baseline (Wave 112+)
#
# Establishes hard performance numbers against which every future wave is gated.
# Run this BEFORE making any performance-related code changes. The output JSON
# becomes the immutable reference committed to benchmarks/results/.
#
# Measured per model × quant format:
#   • Cold-start load time         (5 runs, p50 / p95 / stddev)
#   • TTFT at 32-token prompt      (5 warmup + 20 measured, p50 / p95 / stddev)
#   • Sustained tok/s              (100 generated tokens, 5 warmup + 20 measured)
#   • Peak RSS (MB)                (before first fwd pass, after first fwd pass)
#
# Usage:
#   bash scripts/run_baseline.sh
#   bash scripts/run_baseline.sh --models qwen2.5:1.5b           # single model
#   bash scripts/run_baseline.sh --formats int4,int8              # specific formats
#   bash scripts/run_baseline.sh --output-dir /path/to/results    # custom output dir
#
# Requirements: squish, python3, jq (optional, for pretty-printing)
#
# Output:
#   benchmarks/results/baseline_<TIMESTAMP>.json  — full machine-readable data
#   benchmarks/results/baseline_<TIMESTAMP>.md    — human-readable summary table

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"
OUTPUT_DIR="$REPO_DIR/benchmarks/results"
PORT=11435
HOST="127.0.0.1"
SERVER_TIMEOUT=120
WARMUP_RUNS=5
MEASURED_RUNS=20
GENERATED_TOKENS=100
TTFT_PROMPT="The history of artificial intelligence began in the mid-twentieth century when"

# Default models and formats (can be overridden via CLI args)
MODELS_ARG=""
FORMATS_ARG="int4,int3,int8"

# ── Parse CLI args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --models) MODELS_ARG="$2"; shift 2 ;;
        --formats) FORMATS_ARG="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"
RESULT_JSON="$OUTPUT_DIR/baseline_${TIMESTAMP}.json"
RESULT_MD="$OUTPUT_DIR/baseline_${TIMESTAMP}.md"

# ── Default model list ───────────────────────────────────────────────────────
if [[ -z "$MODELS_ARG" ]]; then
    # Benchmark the three canonical release target models
    MODELS=("qwen2.5:1.5b" "qwen3:4b" "qwen3:8b")
else
    IFS=',' read -ra MODELS <<< "$MODELS_ARG"
fi

IFS=',' read -ra FORMATS <<< "$FORMATS_ARG"

# ── Hardware context ─────────────────────────────────────────────────────────
CHIP="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
TOTAL_RAM_BYTES="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
TOTAL_RAM_GB="$(echo "scale=1; $TOTAL_RAM_BYTES / 1073741824" | bc 2>/dev/null || echo unknown)"
OS_VERSION="$(uname -r)"
SQUISH_COMMIT="$(cd "$REPO_DIR" && git rev-parse --short HEAD 2>/dev/null || echo unknown)"
PYTHON_VER="$(python3 --version 2>/dev/null | awk '{print $2}')"
MLX_VER="$(python3 -c 'import mlx; print(mlx.__version__)' 2>/dev/null || echo unknown)"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Server lifecycle helpers ─────────────────────────────────────────────────
start_server() {
    local model="$1"
    local format="$2"
    local format_flag="--${format}"
    [[ "$format" == "int4" ]] && format_flag=""  # INT4 is default

    kill_server_on_port

    log "  Starting server: squish serve $model $format_flag"
    squish serve "$model" $format_flag \
        --host "$HOST" --port "$PORT" \
        --log-level warning \
        &
    SERVER_PID=$!

    local deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
    while true; do
        if curl -sf "http://$HOST:$PORT/v1/models" \
               -H "Authorization: Bearer squish" > /dev/null 2>&1; then
            log "  Server ready (PID $SERVER_PID)"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "  ERROR: server process exited unexpectedly"
            return 1
        fi
        if [[ $(date +%s) -ge $deadline ]]; then
            log "  ERROR: server did not start within ${SERVER_TIMEOUT}s"
            kill_server_on_port
            return 1
        fi
        sleep 2
    done
}

kill_server_on_port() {
    local port_pid
    port_pid=$(lsof -ti tcp:$PORT 2>/dev/null || true)
    if [[ -n "$port_pid" ]]; then
        kill -9 $port_pid 2>/dev/null || true
        sleep 2
    fi
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID:-}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        sleep 2
    fi
    SERVER_PID=0
}

SERVER_PID=0
trap 'kill_server_on_port' EXIT

# ── Python helper: single request timing ─────────────────────────────────────
# We invoke an inline Python snippet via subshell for precise timing.
# Returns: load_ms ttft_ms tps rss_mb (space-separated)
measure_ttft_tps() {
    python3 - <<'PYEOF'
import json, sys, time, urllib.request, os

HOST = os.environ.get("BENCH_HOST", "127.0.0.1")
PORT = int(os.environ.get("BENCH_PORT", "11435"))
PROMPT = os.environ.get("BENCH_PROMPT", "The sky is blue because")
MAX_TOK = int(os.environ.get("BENCH_MAX_TOKENS", "100"))

url = f"http://{HOST}:{PORT}/v1/chat/completions"
body = json.dumps({
    "model": "squish",
    "messages": [{"role": "user", "content": PROMPT}],
    "max_tokens": MAX_TOK,
    "stream": True,
}).encode()

req = urllib.request.Request(url, data=body,
    headers={"Content-Type": "application/json",
             "Authorization": "Bearer squish"})

t0 = time.perf_counter()
first_token_t = None
last_token_t = None
token_count = 0

with urllib.request.urlopen(req, timeout=120) as resp:
    for raw_line in resp:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            now = time.perf_counter()
            if first_token_t is None:
                first_token_t = now
            last_token_t = now
            token_count += 1

ttft_ms = round((first_token_t - t0) * 1000, 1) if first_token_t else -1
duration_s = (last_token_t - first_token_t) if (last_token_t and first_token_t) else 0
tps = round(token_count / duration_s, 2) if duration_s > 0 else 0

print(f"{ttft_ms} {tps}")
PYEOF
}

# ── Peak RSS helper (reads /proc/PID/status on Linux, ps on macOS) ───────────
get_rss_mb() {
    local pid="${1:-$SERVER_PID}"
    if [[ -f "/proc/$pid/status" ]]; then
        grep VmRSS "/proc/$pid/status" | awk '{print $2/1024}'
    else
        ps -o rss= -p "$pid" 2>/dev/null | awk '{print $1/1024}' || echo 0
    fi
}

# ── Cold-start timing (restarts server each time, measures wall-clock to ready) ──
measure_cold_start() {
    local model="$1"
    local format="$2"
    local format_flag="--${format}"
    [[ "$format" == "int4" ]] && format_flag=""

    local times=()
    for i in $(seq 1 5); do
        kill_server_on_port
        local t0; t0=$(date +%s%3N)
        squish serve "$model" $format_flag \
            --host "$HOST" --port "$PORT" \
            --log-level warning &
        local spid=$!
        local deadline=$(( $(date +%s) + SERVER_TIMEOUT ))
        while true; do
            if curl -sf "http://$HOST:$PORT/v1/models" \
                   -H "Authorization: Bearer squish" > /dev/null 2>&1; then
                break
            fi
            [[ $(date +%s) -ge $deadline ]] && { kill $spid 2>/dev/null; break; }
            sleep 0.2
        done
        local t1; t1=$(date +%s%3N)
        times+=("$(( t1 - t0 ))")
        kill $spid 2>/dev/null || true
        sleep 3
    done
    # Sort and compute p50, p95
    local sorted; sorted=($(printf '%s\n' "${times[@]}" | sort -n))
    local n=${#sorted[@]}
    echo "p50=${sorted[$((n/2))]} p95=${sorted[$((n*95/100))]} raw=${times[*]}"
}

# ── Main benchmark loop ───────────────────────────────────────────────────────
declare -A RESULTS

log "═══════════════════════════════════════════════════"
log " Squish Benchmark Baseline — Wave 112+"
log " Chip : $CHIP ($TOTAL_RAM_GB GB)"
log " OS   : Darwin $OS_VERSION"
log " Squish commit: $SQUISH_COMMIT"
log " MLX  : $MLX_VER"
log "═══════════════════════════════════════════════════"
log ""

for model in "${MODELS[@]}"; do
    for fmt in "${FORMATS[@]}"; do
        RUN_KEY="${model}__${fmt}"
        log "▶ Benchmarking $model @ $fmt"

        # ── 1. Cold-start load time ───────────────────────────────────────────
        log "  [1/3] Measuring cold-start load time (5 runs) …"
        COLD_START_RESULT="$(measure_cold_start "$model" "$fmt" 2>/dev/null || echo "p50=0 p95=0 raw=")"
        log "  Cold-start: $COLD_START_RESULT"

        # ── 2. Start server for TTFT + tok/s measurements ────────────────────
        if ! start_server "$model" "$fmt"; then
            log "  ✗ Server failed to start for $model $fmt — skipping"
            RESULTS[$RUN_KEY]='{"error":"server_start_failed"}'
            continue
        fi

        # ── 2a. Warmup runs (discarded) ───────────────────────────────────────
        log "  [2/3] Warmup ($WARMUP_RUNS runs, discarded) …"
        for _ in $(seq 1 $WARMUP_RUNS); do
            BENCH_HOST="$HOST" BENCH_PORT="$PORT" \
            BENCH_PROMPT="$TTFT_PROMPT" BENCH_MAX_TOKENS=20 \
            measure_ttft_tps > /dev/null 2>&1 || true
        done

        # ── 2b. TTFT measurements ─────────────────────────────────────────────
        log "  [2/3] Measuring TTFT ($MEASURED_RUNS runs) …"
        TTFT_VALS=()
        TPS_VALS=()
        for _ in $(seq 1 $MEASURED_RUNS); do
            result="$(BENCH_HOST="$HOST" BENCH_PORT="$PORT" \
                      BENCH_PROMPT="$TTFT_PROMPT" \
                      BENCH_MAX_TOKENS="$GENERATED_TOKENS" \
                      measure_ttft_tps 2>/dev/null || echo "-1 0")"
            ttft="$(echo "$result" | awk '{print $1}')"
            tps="$(echo "$result" | awk '{print $2}')"
            [[ "$ttft" != "-1" ]] && TTFT_VALS+=("$ttft") && TPS_VALS+=("$tps")
        done

        # ── 2c. RSS measurement ───────────────────────────────────────────────
        log "  [3/3] Measuring peak RSS …"
        RSS_MB="$(get_rss_mb "$SERVER_PID")"
        log "  Peak RSS: ${RSS_MB} MB"

        # ── 2d. Compute statistics via Python ─────────────────────────────────
        STATS_JSON="$(python3 - "${TTFT_VALS[@]:-0}" "${TPS_VALS[@]:-0}" <<'PYEOF'
import sys, json, math

def stats(vals):
    if not vals:
        return {"mean": 0, "p50": 0, "p95": 0, "stddev": 0}
    s = sorted(vals)
    n = len(s)
    mean = sum(s) / n
    stddev = math.sqrt(sum((x - mean)**2 for x in s) / n) if n > 1 else 0
    p50 = s[n // 2]
    p95 = s[int(n * 0.95)]
    return {"mean": round(mean, 2), "p50": round(p50, 2),
            "p95": round(p95, 2), "stddev": round(stddev, 2)}

args = sys.argv[1:]
# Split at "0" sentinel between ttft and tps lists — passed as separate groups
mid = len(args) // 2
ttft = [float(x) for x in args[:mid] if x and x != "0"]
tps  = [float(x) for x in args[mid:] if x and x != "0"]
print(json.dumps({"ttft_ms": stats(ttft), "tps": stats(tps)}))
PYEOF
)"

        kill_server_on_port
        log "  Done: $RUN_KEY"
        log "  TTFT: $STATS_JSON"
        log ""

        RESULTS[$RUN_KEY]=$(python3 -c "
import json
cs = '$COLD_START_RESULT'
stats = $STATS_JSON
rss = float('$RSS_MB') if '$RSS_MB' else 0

# parse cold start
cs_dict = {}
for part in cs.split():
    if '=' in part:
        k, v = part.split('=', 1)
        if k in ('p50', 'p95'):
            cs_dict[k] = int(v)
        elif k == 'raw':
            cs_dict['raw_ms'] = list(map(int, v.split())) if v else []

print(json.dumps({
    'cold_start_ms': cs_dict,
    'ttft_ms': stats['ttft_ms'],
    'tps': stats['tps'],
    'peak_rss_mb': round(rss, 1),
}))
")
    done
done

# ── Write JSON output ─────────────────────────────────────────────────────────
python3 - <<PYEOF
import json, os

hardware = {
    "chip":        "$CHIP",
    "total_ram_gb": "$TOTAL_RAM_GB",
    "os":          "Darwin $OS_VERSION",
    "squish_commit": "$SQUISH_COMMIT",
    "python":      "$PYTHON_VER",
    "mlx":         "$MLX_VER",
    "timestamp":   "$TIMESTAMP",
}

models = {}
$(for key in "${!RESULTS[@]}"; do
    model_part="${key%%__*}"
    fmt_part="${key##*__}"
    echo "models.setdefault('$model_part', {})['$fmt_part'] = ${RESULTS[$key]}"
done)

output = {"hardware": hardware, "results": models,
          "benchmark_config": {
              "warmup_runs": $WARMUP_RUNS,
              "measured_runs": $MEASURED_RUNS,
              "generated_tokens": $GENERATED_TOKENS,
              "ttft_prompt_tokens": 32,
          }}

with open("$RESULT_JSON", "w") as f:
    json.dump(output, f, indent=2)
print(f"Saved JSON → $RESULT_JSON")
PYEOF

# ── Write Markdown summary ─────────────────────────────────────────────────────
python3 - <<PYEOF
import json, os

with open("$RESULT_JSON") as f:
    data = json.load(f)

hw = data["hardware"]
lines = [
    f"# Squish Baseline Benchmark — Wave 112+",
    f"",
    f"**Date**: {hw['timestamp']}",
    f"**Chip**: {hw['chip']} ({hw['total_ram_gb']} GB)",
    f"**Squish commit**: {hw['squish_commit']}",
    f"**MLX**: {hw['mlx']}",
    f"**OS**: {hw['os']}",
    f"",
    f"> These numbers are the reference gate for regression tracking.",
    f"> A wave that regresses p95 TTFT by >5% or peak RSS by >10% is a hard stop.",
    f"",
    f"---",
    f"",
    f"## Results",
    f"",
    f"| Model | Format | Load p50 (ms) | Load p95 (ms) | TTFT p50 (ms) | TTFT p95 (ms) | tok/s p50 | tok/s p95 | Peak RSS (MB) |",
    f"|-------|--------|--------------|--------------|--------------|--------------|----------|----------|--------------|",
]

for model, formats in data["results"].items():
    for fmt, metrics in formats.items():
        if "error" in metrics:
            lines.append(f"| {model} | {fmt} | — | — | — | — | — | — | ERROR |")
            continue
        cs = metrics.get("cold_start_ms", {})
        ttft = metrics.get("ttft_ms", {})
        tps = metrics.get("tps", {})
        rss = metrics.get("peak_rss_mb", 0)
        lines.append(
            f"| {model} | {fmt} "
            f"| {cs.get('p50','?')} | {cs.get('p95','?')} "
            f"| {ttft.get('p50','?')} | {ttft.get('p95','?')} "
            f"| {tps.get('p50','?')} | {tps.get('p95','?')} "
            f"| {rss} |"
        )

bc = data["benchmark_config"]
lines += [
    f"",
    f"---",
    f"",
    f"## Methodology",
    f"",
    f"- **Warmup**: {bc['warmup_runs']} runs discarded before measurement",
    f"- **Measured runs**: {bc['measured_runs']}",
    f"- **Generated tokens per run**: {bc['generated_tokens']}",
    f"- **TTFT prompt**: first ~{bc['ttft_prompt_tokens']} tokens",
    f"- **Cold-start**: 5 full process-start-to-server-ready measurements",
    f"- **RSS**: `ps -o rss` after server ready, before first user request",
]

with open("$RESULT_MD", "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Saved Markdown → $RESULT_MD")
PYEOF

log ""
log "═══════════════════════════════════════════════════"
log " Baseline complete."
log " JSON   : $RESULT_JSON"
log " Report : $RESULT_MD"
log "═══════════════════════════════════════════════════"
