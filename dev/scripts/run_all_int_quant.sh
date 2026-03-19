#!/usr/bin/env bash
# run_all_int_quant.sh — Run the INT4/INT3/INT2 quantization benchmark across
# all 10 target models, in order from smallest to largest.
#
# Prerequisites
# ─────────────
#   • squish installed: pip install -e .
#   • mlx + mlx_lm installed (Apple Silicon): pip install mlx mlx-lm
#   • lm-eval installed (optional, for T3): pip install lm-eval
#   • Models downloaded to $MODELS_DIR (default: ~/models/)
#
# Usage
# ─────
#   # Compression only (fastest, no GPU compute required):
#   ./dev/scripts/run_all_int_quant.sh --bits 4
#
#   # Full benchmark — throughput + perplexity, INT4 only:
#   ./dev/scripts/run_all_int_quant.sh --bits 4 --eval-tps --eval-ppl
#
#   # All bit levels, all tests:
#   ./dev/scripts/run_all_int_quant.sh --bits all --eval-tps --eval-ppl --eval-acc
#
#   # Single model only:
#   ./dev/scripts/run_all_int_quant.sh --models "Qwen2.5-1.5B" --bits 4
#
#   # Keep compressed files for inspection:
#   ./dev/scripts/run_all_int_quant.sh --bits 4 --keep-compressed
#
# Environment variables
# ─────────────────────
#   MODELS_DIR   — directory where models are stored (default: ~/models)
#   RUNS         — number of throughput runs per model (default: 3)
#   OUTPUT_DIR   — results directory (default: dev/results/int_quant)
#   SKIP_PULL    — set to 1 to skip squish pull download step
#
# Batching order (smallest → largest to validate pipeline early)
# ─────────────────────────────────────────────────────────────────
#   Batch A: Qwen2.5-1.5B (3.1 GB) · Llama-3.2-3B (6.4 GB) · Gemma-3-4B (8.6 GB)
#   Batch B: Qwen2.5-7B (14.0 GB) · Mistral-7B-v0.3 (14.5 GB) · DeepSeek-R1-Distill-7B (15.2 GB)
#   Batch C: Llama-3.1-8B (16.0 GB) · Qwen3-8B (16.4 GB)
#   Batch D: Qwen2.5-14B (29.6 GB) · Phi-4 (29.4 GB)

set -euo pipefail

# ── colours ───────────────────────────────────────────────────────────────────
C='\033[36m'; G='\033[32m'; Y='\033[33m'; R='\033[31m'; NC='\033[0m'; B='\033[1m'

# ── repo root ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# ── defaults ──────────────────────────────────────────────────────────────────
MODELS_DIR="${MODELS_DIR:-${HOME}/models}"
RUNS="${RUNS:-3}"
OUTPUT_DIR="${OUTPUT_DIR:-dev/results/int_quant}"
SKIP_PULL="${SKIP_PULL:-0}"

BITS="4"
EVAL_TPS=""
EVAL_PPL=""
EVAL_ACC=""
KEEP_COMPRESSED=""
MARKDOWN_FLAG=""
TARGET_MODELS=()   # empty = all models

# ── argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bits)         BITS="$2";          shift 2 ;;
        --runs)         RUNS="$2";          shift 2 ;;
        --models-dir)   MODELS_DIR="$2";    shift 2 ;;
        --output-dir)   OUTPUT_DIR="$2";    shift 2 ;;
        --eval-tps)     EVAL_TPS="--eval-tps";  shift ;;
        --eval-ppl)     EVAL_PPL="--eval-ppl";  shift ;;
        --eval-acc)     EVAL_ACC="--eval-acc";  shift ;;
        --keep-compressed) KEEP_COMPRESSED="--keep-compressed"; shift ;;
        --markdown)     MARKDOWN_FLAG="--markdown"; shift ;;
        --skip-pull)    SKIP_PULL=1;        shift ;;
        --models)       IFS=',' read -ra TARGET_MODELS <<< "$2"; shift 2 ;;
        --dry-run)      DRY_RUN=1;          shift ;;
        -h|--help)
            head -60 "$0" | grep "^#" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo -e "${R}Unknown argument: $1${NC}" >&2
            exit 1
            ;;
    esac
done

DRY_RUN="${DRY_RUN:-0}"

# ── model catalog ────────────────────────────────────────────────────────────
# Format: "display_name|hf_repo|expected_bf16_gb|squish_pull_id"
# squish_pull_id = argument passed to `squish pull` (catalog shorthand or HF repo)
declare -A MODEL_DIRS=(
  ["Qwen2.5-1.5B"]="${MODELS_DIR}/Qwen2.5-1.5B-Instruct"
  ["Qwen2.5-7B"]="${MODELS_DIR}/Qwen2.5-7B-Instruct"
  ["Qwen2.5-14B"]="${MODELS_DIR}/Qwen2.5-14B-Instruct"
  ["Qwen3-8B"]="${MODELS_DIR}/Qwen3-8B"
  ["Llama-3.2-3B"]="${MODELS_DIR}/Llama-3.2-3B-Instruct"
  ["Llama-3.1-8B"]="${MODELS_DIR}/Llama-3.1-8B-Instruct"
  ["Mistral-7B-v0.3"]="${MODELS_DIR}/Mistral-7B-Instruct-v0.3"
  ["Phi-4"]="${MODELS_DIR}/Phi-4"
  ["Gemma-3-4B"]="${MODELS_DIR}/gemma-3-4b-it"
  ["DeepSeek-R1-Distill-7B"]="${MODELS_DIR}/DeepSeek-R1-Distill-Qwen-7B"
)

declare -A HF_REPOS=(
  ["Qwen2.5-1.5B"]="Qwen/Qwen2.5-1.5B-Instruct"
  ["Qwen2.5-7B"]="Qwen/Qwen2.5-7B-Instruct"
  ["Qwen2.5-14B"]="Qwen/Qwen2.5-14B-Instruct"
  ["Qwen3-8B"]="Qwen/Qwen3-8B"
  ["Llama-3.2-3B"]="meta-llama/Llama-3.2-3B-Instruct"
  ["Llama-3.1-8B"]="meta-llama/Meta-Llama-3.1-8B-Instruct"
  ["Mistral-7B-v0.3"]="mistralai/Mistral-7B-Instruct-v0.3"
  ["Phi-4"]="microsoft/phi-4"
  ["Gemma-3-4B"]="google/gemma-3-4b-it"
  ["DeepSeek-R1-Distill-7B"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# Ordered list for batched execution (smallest → largest)
ALL_MODELS=(
  "Qwen2.5-1.5B"
  "Llama-3.2-3B"
  "Gemma-3-4B"
  "Qwen2.5-7B"
  "Mistral-7B-v0.3"
  "DeepSeek-R1-Distill-7B"
  "Llama-3.1-8B"
  "Qwen3-8B"
  "Qwen2.5-14B"
  "Phi-4"
)

# Filter to target models if specified
if [[ ${#TARGET_MODELS[@]} -gt 0 ]]; then
    RUN_MODELS=("${TARGET_MODELS[@]}")
else
    RUN_MODELS=("${ALL_MODELS[@]}")
fi

# ── banner ────────────────────────────────────────────────────────────────────
echo -e "\n${B}${C}  Squish INT Quantization Benchmark — All Models${NC}"
echo -e "${C}  ──────────────────────────────────────────────────────────${NC}"
echo -e "  Models dir:  ${MODELS_DIR}"
echo -e "  Output dir:  ${OUTPUT_DIR}"
echo -e "  Bits:        ${BITS}"
echo -e "  Runs:        ${RUNS}"
echo -e "  Tests:       ${EVAL_TPS:-[none]} ${EVAL_PPL} ${EVAL_ACC}"
echo -e "  Models:      ${RUN_MODELS[*]}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo -e "  ${Y}DRY RUN — no benchmark will be executed${NC}"
fi
echo ""

mkdir -p "${OUTPUT_DIR}"

# ── helpers ───────────────────────────────────────────────────────────────────
run_bench() {
    local model_name="$1"
    local model_dir="${MODEL_DIRS[$model_name]}"

    echo -e "\n${B}${C}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${B}${C}║  ${model_name}${NC}"
    echo -e "${B}${C}╚══════════════════════════════════════════════════════╝${NC}"

    # ── download if missing ───────────────────────────────────────────────────
    if [[ ! -d "${model_dir}" ]]; then
        if [[ "${SKIP_PULL}" == "1" ]]; then
            echo -e "  ${Y}⚠ model dir not found and --skip-pull set: ${model_dir}${NC}"
            echo -e "  ${Y}  Skipping ${model_name}${NC}"
            return 0
        fi
        echo -e "  ${C}→ Downloading ${model_name} from HuggingFace…${NC}"
        hf_repo="${HF_REPOS[$model_name]}"
        if [[ "${DRY_RUN}" == "1" ]]; then
            echo -e "  ${C}[dry-run] huggingface-cli download ${hf_repo} --local-dir ${model_dir}${NC}"
        else
            huggingface-cli download "${hf_repo}" \
                --local-dir "${model_dir}" \
                --local-dir-use-symlinks False \
                --ignore-patterns "*.gguf" "runs/*" "*.bin" \
                || {
                    echo -e "  ${R}✗ Download failed for ${model_name}. Skipping.${NC}"
                    return 0
                }
        fi
    else
        echo -e "  ${G}✓ Model found: ${model_dir}${NC}"
    fi

    # ── run bench ─────────────────────────────────────────────────────────────
    local cmd=(
        python3 dev/benchmarks/bench_int_quant.py
        --model-dir  "${model_dir}"
        --bits       "${BITS}"
        --runs       "${RUNS}"
        --output-dir "${OUTPUT_DIR}"
        ${EVAL_TPS}
        ${EVAL_PPL}
        ${EVAL_ACC}
        ${KEEP_COMPRESSED}
        ${MARKDOWN_FLAG}
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo -e "  ${C}[dry-run] ${cmd[*]}${NC}"
        return 0
    fi

    echo -e "  ${C}→ Running benchmark…${NC}"
    "${cmd[@]}" || {
        echo -e "  ${R}✗ Benchmark failed for ${model_name} (exit $?). Continuing.${NC}"
    }
}

# ── main loop ─────────────────────────────────────────────────────────────────
PASS=0
FAIL=0
SKIP=0
START_TIME=$(date +%s)

for model_name in "${RUN_MODELS[@]}"; do
    if [[ -z "${MODEL_DIRS[$model_name]+_}" ]]; then
        echo -e "  ${Y}⚠ Unknown model: ${model_name} — skipping${NC}"
        (( SKIP++ ))
        continue
    fi
    run_bench "${model_name}" && (( PASS++ )) || (( FAIL++ ))
done

# ── aggregate results ─────────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

echo -e "\n${B}${C}  Aggregating results…${NC}"
if [[ "${DRY_RUN}" != "1" ]]; then
    python3 dev/benchmarks/aggregate_int_quant.py \
        --results-dir "${OUTPUT_DIR}" \
        --output      "docs/benchmark_int_quant.md" \
        --json-output "${OUTPUT_DIR}/combined.json" \
        || echo -e "  ${Y}⚠ aggregation failed${NC}"
fi

# ── summary ───────────────────────────────────────────────────────────────────
echo ""
echo -e "${B}${C}  ─────────────────────────────────────────────────────${NC}"
echo -e "${B}${C}  Benchmark run complete${NC}"
echo -e "  Total time:  ${ELAPSED_MIN}m ${ELAPSED}s"
echo -e "  Passed:      ${G}${PASS}${NC}"
echo -e "  Failed:      ${R}${FAIL}${NC}"
echo -e "  Skipped:     ${Y}${SKIP}${NC}"
echo -e "  Results dir: ${OUTPUT_DIR}"
if [[ -f "docs/benchmark_int_quant.md" ]]; then
    echo -e "  Markdown:    docs/benchmark_int_quant.md"
fi
echo ""
