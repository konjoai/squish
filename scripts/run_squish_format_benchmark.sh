#!/usr/bin/env bash
# run_squish_format_benchmark.sh — Wave 70 SQUIZD format benchmark orchestrator.
#
# Runs the 21-model × 4-variant benchmark suite and writes a summary to
# docs/BENCHMARK_SQUIZD_FORMAT.md.
#
# Usage:
#   ./scripts/run_squish_format_benchmark.sh [--output-dir DIR] [--dry-run]
#
# Options:
#   --output-dir DIR   Write results to DIR instead of docs/
#   --dry-run          Print what would run without executing Python.
#   --models  LIST     Comma-separated model list override.
#   --variants LIST    Comma-separated variant list override (astc,int4,int4-sparse,full).
#
# Requirements:
#   • Python 3.10+ with squish installed (pip install -e .)
#   • numpy

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/docs"
DRY_RUN=false
MODELS_OVERRIDE=""
VARIANTS_OVERRIDE=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --models)
            MODELS_OVERRIDE="$2"
            shift 2
            ;;
        --variants)
            VARIANTS_OVERRIDE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Environment checks
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found in PATH." >&2
    exit 1
fi

PYTHON_MIN="3.10"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ! python3 -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
    echo "ERROR: Python ${PYTHON_MIN}+ required, found ${PYTHON_VERSION}." >&2
    exit 1
fi

# Verify squish is importable.
if ! python3 -c "import squish" 2>/dev/null; then
    echo "ERROR: squish package not importable.  Run: pip install -e ${REPO_ROOT}" >&2
    exit 1
fi

# Verify numpy is importable.
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "ERROR: numpy not found.  Run: pip install numpy" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/BENCHMARK_SQUIZD_FORMAT.md"

# ---------------------------------------------------------------------------
# Build Python invocation
# ---------------------------------------------------------------------------
PYTHON_ARGS=()
if [[ -n "${MODELS_OVERRIDE}" ]]; then
    PYTHON_ARGS+=("--models" "${MODELS_OVERRIDE}")
fi
if [[ -n "${VARIANTS_OVERRIDE}" ]]; then
    PYTHON_ARGS+=("--variants" "${VARIANTS_OVERRIDE}")
fi
PYTHON_ARGS+=("--output" "${OUTPUT_FILE}")

PYTHON_CMD=(
    python3 -m squish.bench.squish_bench
    run-benchmark
    "${PYTHON_ARGS[@]}"
)

# ---------------------------------------------------------------------------
# Execute or preview
# ---------------------------------------------------------------------------
if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[dry-run] Would run:"
    echo "  ${PYTHON_CMD[*]}"
    echo "[dry-run] Output: ${OUTPUT_FILE}"
    exit 0
fi

echo "============================================================"
echo "  Squish SQUIZD Format Benchmark — Wave 70"
echo "  Repository: ${REPO_ROOT}"
echo "  Output:     ${OUTPUT_FILE}"
echo "  Python:     $(python3 --version)"
echo "============================================================"
echo ""

cd "${REPO_ROOT}"

# Time the full suite.
SECONDS=0

echo ">>> Running benchmark suite..."
"${PYTHON_CMD[@]}"

ELAPSED=${SECONDS}
ELAPSED_MINS=$(( ELAPSED / 60 ))
ELAPSED_SECS=$(( ELAPSED % 60 ))

echo ""
echo "============================================================"
echo "  Done in ${ELAPSED_MINS}m ${ELAPSED_SECS}s"
echo "  Results: ${OUTPUT_FILE}"
echo "============================================================"
