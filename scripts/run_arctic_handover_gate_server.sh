#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/autodl-tmp/external/handx-venv/bin/python}"
AUDIT_SCRIPT="${AUDIT_SCRIPT:-/root/autodl-tmp/arctic_scripts/audit_arctic_handover_gate.py}"
ARCTIC_ROOT="${ARCTIC_ROOT:-/root/autodl-tmp/arctic_data/data}"
BODY_MODELS="${BODY_MODELS:-/root/autodl-tmp/arctic_runtime/body_models}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/arctic_handover_gate_v1_20260921}"
SHUTDOWN_DELAY_SECONDS="${SHUTDOWN_DELAY_SECONDS:-30}"

mkdir -p "${OUTPUT_DIR}"
rm -f \
  "${OUTPUT_DIR}/SUCCESS" \
  "${OUTPUT_DIR}/FAILED" \
  "${OUTPUT_DIR}/exit_code" \
  "${OUTPUT_DIR}/shutdown.log"

date -Is > "${OUTPUT_DIR}/started_at"
"${PYTHON_BIN}" "${AUDIT_SCRIPT}" \
  --arctic-root "${ARCTIC_ROOT}" \
  --body-models "${BODY_MODELS}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda \
  --frame-chunk 16 \
  > "${OUTPUT_DIR}/run.log" 2>&1
exit_code=$?

echo "${exit_code}" > "${OUTPUT_DIR}/exit_code"
date -Is > "${OUTPUT_DIR}/finished_at"
if [ "${exit_code}" -eq 0 ]; then
  touch "${OUTPUT_DIR}/SUCCESS"
else
  touch "${OUTPUT_DIR}/FAILED"
fi

sync
sleep "${SHUTDOWN_DELAY_SECONDS}"
shutdown -h now > "${OUTPUT_DIR}/shutdown.log" 2>&1
