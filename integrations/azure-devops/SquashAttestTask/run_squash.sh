#!/usr/bin/env bash
# run_squash.sh — Squash AI Model SBOM Attestation, bash runner
#
# Fallback entry-point for Linux/macOS ADO agents that invoke the task via
# bash instead of PowerShell Core.  For most modern ADO agent pools (Ubuntu
# 22.04+, macOS Sonoma+) the PowerShell3 handler in run_squash.ps1 is
# preferred because it runs identically on all platforms.  This script exists
# for minimal-footprint agents where PSCore is absent.
#
# ADO injects task inputs as INPUT_<UPPERCASED_INPUT_NAME> environment vars.
# Boolean inputs arrive as the strings "true" / "false".

set -euo pipefail

# ---------------------------------------------------------------------------
# Input binding
# ---------------------------------------------------------------------------

MODEL_PATH="${INPUT_MODELPATH:?modelPath input is required}"
POLICIES="${INPUT_POLICIES:-enterprise-strict}"
SIGN="${INPUT_SIGN:-false}"
FAIL_ON_VIOLATION="${INPUT_FAILONVIOLATION:-true}"
OUTPUT_DIR="${INPUT_OUTPUTDIR:-}"
SQUISH_VERSION="${INPUT_SQUISHVERSION:-}"
PYTHON_EXE="${INPUT_PYTHONEXECUTABLE:-python3}"
PUBLISH_ARTIFACTS="${INPUT_PUBLISHARTIFACTS:-true}"

echo "##[section]Squash — AI Model SBOM Attestation"
echo "  model path : ${MODEL_PATH}"
echo "  policies   : ${POLICIES}"
echo "  sign       : ${SIGN}"
echo "  fail gate  : ${FAIL_ON_VIOLATION}"

# ---------------------------------------------------------------------------
# Install squish[squash]
# ---------------------------------------------------------------------------

if [ -n "${SQUISH_VERSION}" ]; then
    PKG="squish[squash]==${SQUISH_VERSION}"
else
    PKG="squish[squash]"
fi

echo ""
echo "##[group]Installing ${PKG}"
"${PYTHON_EXE}" -m pip install --quiet "${PKG}"
echo "##[endgroup]"

# ---------------------------------------------------------------------------
# Build squash CLI argument list
# ---------------------------------------------------------------------------

RESULT_PATH="/tmp/squash-result-$$.json"

CLI_ARGS=("attest" "--model-path" "${MODEL_PATH}" "--json-result" "${RESULT_PATH}")

IFS=',' read -r -a POLICY_LIST <<< "${POLICIES}"
for POLICY in "${POLICY_LIST[@]}"; do
    TRIMMED="${POLICY// /}"
    [ -n "${TRIMMED}" ] && CLI_ARGS+=("--policy" "${TRIMMED}")
done

[ "${SIGN}" = "true" ] && CLI_ARGS+=("--sign")
[ -n "${OUTPUT_DIR}" ] && CLI_ARGS+=("--output-dir" "${OUTPUT_DIR}")

# ---------------------------------------------------------------------------
# Run squash attest
# ---------------------------------------------------------------------------

echo ""
echo "##[group]squash ${CLI_ARGS[*]}"
RUN_RC=0
squash "${CLI_ARGS[@]}" || RUN_RC=$?
echo "##[endgroup]"

# ---------------------------------------------------------------------------
# Parse result JSON and set pipeline output variables
# ---------------------------------------------------------------------------

PASSED="false"
SCAN_STATUS="skipped"

if [ -f "${RESULT_PATH}" ]; then
    PASSED=$(python3 -c "
import json, sys
try:
    d = json.load(open('${RESULT_PATH}'))
    print('true' if d.get('passed') else 'false')
except Exception:
    print('false')
")
    SCAN_STATUS=$(python3 -c "
import json
try:
    d = json.load(open('${RESULT_PATH}'))
    print(d.get('scan_status', 'skipped'))
except Exception:
    print('skipped')
")

    # Set ADO pipeline output variables
    echo "##vso[task.setvariable variable=SQUASH_PASSED;isOutput=true]${PASSED}"
    echo "##vso[task.setvariable variable=SQUASH_SCAN_STATUS;isOutput=true]${SCAN_STATUS}"

    python3 - <<'PYEOF'
import json, os, sys

result_path = os.environ.get('_SQUASH_RESULT', '')
try:
    d = json.load(open(result_path))
except Exception:
    sys.exit(0)

artifacts = d.get('artifacts', {})
var_map = {
    'SQUASH_CYCLONEDX_PATH':     artifacts.get('cyclonedx', ''),
    'SQUASH_SPDX_JSON_PATH':     artifacts.get('spdx_json', ''),
    'SQUASH_MASTER_RECORD_PATH': artifacts.get('master_record', ''),
}
for var, val in var_map.items():
    if val:
        print(f"##vso[task.setvariable variable={var};isOutput=true]{val}")

print()
passed     = d.get('passed', False)
scan_status = d.get('scan_status', 'skipped')
print(f"Squash result: passed={str(passed).lower()}  scan={scan_status}")
for policy_name, pr in d.get('policy_results', {}).items():
    status = 'PASS' if pr.get('passed') else 'FAIL'
    errors   = pr.get('error_count',   0)
    warnings = pr.get('warning_count', 0)
    print(f"  [{status}] {policy_name} : {errors} error(s), {warnings} warning(s)")
PYEOF

    # Publish artifacts to pipeline run
    if [ "${PUBLISH_ARTIFACTS}" = "true" ]; then
        python3 - <<PUBEOF
import json, os
try:
    d = json.load(open('${RESULT_PATH}'))
    for path in d.get('artifacts', {}).values():
        if path and os.path.isfile(path):
            print(f"##vso[artifact.upload artifactname=squash-attestation;]{path}")
except Exception:
    pass
PUBEOF
    fi

else
    echo "##vso[task.logissue type=warning]squash did not produce a result file at '${RESULT_PATH}'."
    PASSED="false"
fi

# ---------------------------------------------------------------------------
# Final task status
# ---------------------------------------------------------------------------

if [ "${FAIL_ON_VIOLATION}" = "true" ] && [ "${PASSED}" != "true" ]; then
    echo "##vso[task.complete result=Failed;]Squash attestation FAILED for '${MODEL_PATH}'. Scan: ${SCAN_STATUS}."
    exit 1
fi

echo "##vso[task.complete result=Succeeded;]Squash attestation passed."
exit 0
