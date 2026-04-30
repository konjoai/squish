#!/usr/bin/env bash
# dev/benchmarks/run_w103_ship_gate.sh — W103.4d ship-gate orchestrator.
#
# Single command, fail-fast, unattended:
#   1. compress Qwen2.5-7B at SQINT2 (CPU-bound NumPy: ~30–60 min on M3 16 GB)
#   2. arc_easy@200 canary (~30 min) — bail if score < 50% (clearly broken)
#   3. full 5-eval @limit=500 overnight run (~10–14 h)
#
# All output streams to results/w103_ship_gate_<ts>/{compress,canary,full}.log
# and the final BENCHMARK_TABLE.md. Exit code is 0 only on full success.
#
# Usage:
#   bash dev/benchmarks/run_w103_ship_gate.sh
#   bash dev/benchmarks/run_w103_ship_gate.sh --dry-run            # plan only
#   bash dev/benchmarks/run_w103_ship_gate.sh --skip-compress      # eval-only (cache must exist)
#   bash dev/benchmarks/run_w103_ship_gate.sh --canary-only        # stop after arc_easy@200
#   bash dev/benchmarks/run_w103_ship_gate.sh --canary-threshold 60  # custom canary cutoff (default 50)
#
# Environment:
#   MODELS_ROOT  base dir for model checkpoints (default: $HOME/models)
#   SHIP_GATE    full-suite arc_easy threshold (default: 65 — the W103 ship target)
#
# Notes:
#   - The base BF16 source must be at $MODELS_ROOT/Qwen2.5-7B-Instruct-bf16/.
#     Pre-download once before running this script.
#   - Re-runs are idempotent: existing SQINT2 npy-dir + eval cache are reused.
#     Pass --force-compress to redo compression from scratch.
#   - On x86 / non-Apple-Silicon, --dry-run is the only valid mode (lm_eval
#     requires Metal). The dry-run plan is printed and verified clean.

set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
G="\033[32m"; Y="\033[33m"; C="\033[36m"; W="\033[1;37m"; R="\033[31m"; D="\033[2m"; NC="\033[0m"

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
MODELS_ROOT="${MODELS_ROOT:-$HOME/models}"
TS="$(date +%Y%m%dT%H%M%S)"
RESULTS_DIR="$REPO_ROOT/results/w103_ship_gate_$TS"

# ── config ───────────────────────────────────────────────────────────────────
MODEL_BF16_DIR="$MODELS_ROOT/Qwen2.5-7B-Instruct-bf16"
MODEL_SQINT2_DIR="$MODELS_ROOT/Qwen2.5-7B-Instruct-sqint2"
MODEL_REGISTRY_NAME="Qwen2.5-7B-sqint2"
CANARY_LIMIT=200
FULL_LIMIT=500
SHIP_GATE="${SHIP_GATE:-65}"
CANARY_THRESHOLD=50

# ── argparse ─────────────────────────────────────────────────────────────────
DRY_RUN=0
SKIP_COMPRESS=0
CANARY_ONLY=0
FORCE_COMPRESS=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)            DRY_RUN=1 ;;
        --skip-compress)      SKIP_COMPRESS=1 ;;
        --canary-only)        CANARY_ONLY=1 ;;
        --force-compress)     FORCE_COMPRESS=1 ;;
        --canary-threshold)   CANARY_THRESHOLD="${2:?--canary-threshold requires a number}"; shift ;;
        --models-root)        MODELS_ROOT="${2:?--models-root requires a path}"
                              MODEL_BF16_DIR="$MODELS_ROOT/Qwen2.5-7B-Instruct-bf16"
                              MODEL_SQINT2_DIR="$MODELS_ROOT/Qwen2.5-7B-Instruct-sqint2"
                              shift ;;
        -h|--help)
            sed -n '2,32p' "$0"
            exit 0 ;;
        *)
            echo -e "${R}Unknown argument: $1${NC}" >&2
            exit 2 ;;
    esac
    shift
done

# ── helpers ──────────────────────────────────────────────────────────────────
hdr() {  echo -e "\n${W}${1}${NC}"; }
step() { echo -e "  ${C}→${NC}  $1"; }
ok()   { echo -e "  ${G}✓${NC}  $1"; }
warn() { echo -e "  ${Y}⚠${NC}  $1"; }
err()  { echo -e "  ${R}✗${NC}  $1" >&2; }

run_or_dry() {
    local label="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        echo -e "  ${D}\$ $*${NC}"
        ok "DRY-RUN: would run $label"
        return 0
    fi
    echo -e "  ${D}\$ $*${NC}"
    "$@"
}

# ── platform sanity check ───────────────────────────────────────────────────
PLATFORM_ARCH="$(uname -m)"
PLATFORM_OS="$(uname -s)"
hdr "═══ W103.4d SHIP GATE — Qwen2.5-7B SQINT2 ═══"
echo "  Started:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Repo:           $REPO_ROOT"
echo "  Models root:    $MODELS_ROOT"
echo "  BF16 source:    $MODEL_BF16_DIR"
echo "  SQINT2 output:  $MODEL_SQINT2_DIR"
echo "  Results dir:    $RESULTS_DIR"
echo "  Platform:       $PLATFORM_OS / $PLATFORM_ARCH"
echo "  Python:         $($PYTHON --version 2>&1)"
echo "  Canary:         arc_easy @limit=$CANARY_LIMIT  threshold ≥ ${CANARY_THRESHOLD}%"
echo "  Ship gate:      arc_easy @limit=$FULL_LIMIT   threshold ≥ ${SHIP_GATE}%"
echo "  Mode:           DRY_RUN=$DRY_RUN  SKIP_COMPRESS=$SKIP_COMPRESS  CANARY_ONLY=$CANARY_ONLY  FORCE=$FORCE_COMPRESS"

if [[ "$DRY_RUN" != "1" && "$PLATFORM_OS" != "Darwin" ]]; then
    err "lm_eval requires Apple Silicon. Run with --dry-run on $PLATFORM_OS or move to an M-series Mac."
    exit 3
fi
if [[ "$DRY_RUN" != "1" && "$PLATFORM_ARCH" != "arm64" ]]; then
    err "lm_eval requires arm64. Detected $PLATFORM_ARCH; run with --dry-run."
    exit 3
fi

# ── prep results dir ─────────────────────────────────────────────────────────
# Always create — tee paths reference it from both real and dry-run paths.
mkdir -p "$RESULTS_DIR"
if [[ "$DRY_RUN" != "1" ]]; then
    {
        echo "# W103.4d ship-gate run — started $(date)"
        echo "Repo:        $REPO_ROOT"
        echo "Models root: $MODELS_ROOT"
        echo "BF16 source: $MODEL_BF16_DIR"
        echo "SQINT2 dir:  $MODEL_SQINT2_DIR"
        echo "Canary:      arc_easy @limit=$CANARY_LIMIT threshold=${CANARY_THRESHOLD}%"
        echo "Ship gate:   arc_easy @limit=$FULL_LIMIT threshold=${SHIP_GATE}%"
        echo "Platform:    $PLATFORM_OS / $PLATFORM_ARCH"
        echo "Python:      $($PYTHON --version 2>&1)"
    } > "$RESULTS_DIR/run_meta.txt"
fi

cd "$REPO_ROOT"

# ── Stage 1: compress ────────────────────────────────────────────────────────
hdr "═══ Stage 1 — SQINT2 compress ═══"
if [[ "$SKIP_COMPRESS" == "1" ]]; then
    if [[ ! -f "$MODEL_SQINT2_DIR/manifest.json" ]]; then
        err "--skip-compress set but no manifest.json at $MODEL_SQINT2_DIR/"
        exit 4
    fi
    ok "Skipping compress; reusing existing SQINT2 npy-dir at $MODEL_SQINT2_DIR/"
else
    if [[ ! -d "$MODEL_BF16_DIR" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            warn "BF16 source not found: $MODEL_BF16_DIR (continuing dry-run)"
        else
            err "BF16 source not found: $MODEL_BF16_DIR"
            echo "    Pre-download with:"
            echo "      huggingface-cli download Qwen/Qwen2.5-7B-Instruct \\"
            echo "        --local-dir $MODEL_BF16_DIR"
            exit 5
        fi
    fi

    if [[ "$FORCE_COMPRESS" == "1" && -d "$MODEL_SQINT2_DIR" ]]; then
        warn "Removing existing $MODEL_SQINT2_DIR/ (--force-compress)"
        run_or_dry "rm SQINT2 dir" rm -rf "$MODEL_SQINT2_DIR"
    fi

    if [[ -f "$MODEL_SQINT2_DIR/manifest.json" ]]; then
        SIZE_GB=$(du -sk "$MODEL_SQINT2_DIR" 2>/dev/null | awk '{print $1/1024/1024}')
        ok "SQINT2 npy-dir exists ($(printf '%.2f' "${SIZE_GB:-0}") GB) — reusing (use --force-compress to redo)"
    else
        step "Compressing Qwen2.5-7B-Instruct → SQINT2 (expected ~30–60 min)"
        run_or_dry "squish compress --format sqint2" \
            "$PYTHON" -m squish.cli compress \
                --format sqint2 \
                --input  "$MODEL_BF16_DIR" \
                --output "$MODEL_SQINT2_DIR" \
            2>&1 | tee "$RESULTS_DIR/compress.log"

        if [[ "$DRY_RUN" != "1" && ! -f "$MODEL_SQINT2_DIR/manifest.json" ]]; then
            err "compress finished but no manifest.json at $MODEL_SQINT2_DIR/"
            exit 6
        fi
        ok "SQINT2 compress done (log: $RESULTS_DIR/compress.log)"
    fi
fi

# ── Stage 2: arc_easy@200 canary ─────────────────────────────────────────────
hdr "═══ Stage 2 — arc_easy @limit=$CANARY_LIMIT canary ═══"
step "Goal: bail fast if SQINT2 inference is broken (< ${CANARY_THRESHOLD}% on arc_easy)."

CANARY_OUT="$RESULTS_DIR/canary"
run_or_dry "canary lm_eval" \
    "$PYTHON" "$REPO_ROOT/dev/benchmarks/run_overnight_bench.py" \
        --models      Qwen2.5-7B \
        --bits        sqint2 \
        --eval-only \
        --limit       "$CANARY_LIMIT" \
        --models-root "$MODELS_ROOT" \
        --results-dir "$CANARY_OUT" \
    2>&1 | tee "$RESULTS_DIR/canary.log"

# Canary score extract (skip in dry-run).
if [[ "$DRY_RUN" != "1" ]]; then
    CANARY_SCORE="$($PYTHON - <<EOF
import json, glob, sys
files = sorted(glob.glob("$CANARY_OUT/lmeval_*.json"))
if not files:
    print("MISSING"); sys.exit(0)
for fp in reversed(files):
    try:
        d = json.loads(open(fp).read())
    except Exception:
        continue
    s = d.get("scores", {})
    if isinstance(s, dict) and "arc_easy" in s and s["arc_easy"] is not None:
        print(f"{float(s['arc_easy']):.2f}"); sys.exit(0)
print("MISSING")
EOF
    )"
    if [[ "$CANARY_SCORE" == "MISSING" ]]; then
        err "No arc_easy score parsed from canary output. See $RESULTS_DIR/canary.log"
        exit 7
    fi
    AWK_CMP="$(awk -v a="$CANARY_SCORE" -v b="$CANARY_THRESHOLD" 'BEGIN { print (a >= b) ? "PASS" : "FAIL" }')"
    if [[ "$AWK_CMP" == "FAIL" ]]; then
        err "Canary FAILED: arc_easy = ${CANARY_SCORE}% < ${CANARY_THRESHOLD}% threshold"
        echo "    Investigate before launching the overnight run." >&2
        exit 8
    fi
    ok "Canary PASSED: arc_easy = ${CANARY_SCORE}% (≥ ${CANARY_THRESHOLD}%)"
fi

if [[ "$CANARY_ONLY" == "1" ]]; then
    hdr "═══ DONE — canary only ═══"
    ok "Canary stage complete. Pass --canary-only to stop here. Run without it for the full overnight pass."
    exit 0
fi

# ── Stage 3: full overnight ──────────────────────────────────────────────────
hdr "═══ Stage 3 — full lm_eval @limit=$FULL_LIMIT (overnight, ~10–14 h) ═══"
step "Five tasks: arc_easy, arc_challenge, hellaswag, winogrande, piqa, openbookqa."
warn "Do not interrupt. Logs stream to $RESULTS_DIR/full.log; results to $RESULTS_DIR/full/."

FULL_OUT="$RESULTS_DIR/full"
run_or_dry "full lm_eval" \
    "$PYTHON" "$REPO_ROOT/dev/benchmarks/run_overnight_bench.py" \
        --models      Qwen2.5-7B \
        --bits        sqint2 \
        --eval-only \
        --limit       "$FULL_LIMIT" \
        --models-root "$MODELS_ROOT" \
        --results-dir "$FULL_OUT" \
    2>&1 | tee "$RESULTS_DIR/full.log"

# ── ship-gate evaluation ─────────────────────────────────────────────────────
if [[ "$DRY_RUN" != "1" ]]; then
    FULL_SCORE="$($PYTHON - <<EOF
import json, glob, sys
files = sorted(glob.glob("$FULL_OUT/lmeval_*.json"))
if not files:
    print("MISSING"); sys.exit(0)
for fp in reversed(files):
    try:
        d = json.loads(open(fp).read())
    except Exception:
        continue
    s = d.get("scores", {})
    if isinstance(s, dict) and "arc_easy" in s and s["arc_easy"] is not None:
        print(f"{float(s['arc_easy']):.2f}"); sys.exit(0)
print("MISSING")
EOF
    )"
    if [[ "$FULL_SCORE" == "MISSING" ]]; then
        err "No arc_easy score parsed from full output. See $RESULTS_DIR/full.log"
        exit 9
    fi
    GATE_CMP="$(awk -v a="$FULL_SCORE" -v b="$SHIP_GATE" 'BEGIN { print (a >= b) ? "PASS" : "FAIL" }')"
    hdr "═══ SHIP GATE RESULT ═══"
    if [[ "$GATE_CMP" == "PASS" ]]; then
        ok "PASS  arc_easy = ${FULL_SCORE}% ≥ ${SHIP_GATE}% (W103 ship target)"
        echo "  Results: $RESULTS_DIR"
        echo "  Next:    review BENCHMARK_TABLE.md, share results, tag the release."
        exit 0
    else
        err "MISS  arc_easy = ${FULL_SCORE}% < ${SHIP_GATE}% (W103 ship target)"
        echo "  Results: $RESULTS_DIR"
        echo "  Next:    tune residual_rank / sparse_frac / refine_iters and re-run."
        exit 10
    fi
else
    ok "DRY-RUN complete. Pass without --dry-run on Apple Silicon to execute."
    echo "  Plan saved (would have streamed to): $RESULTS_DIR"
fi
