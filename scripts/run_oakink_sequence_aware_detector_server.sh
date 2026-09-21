#!/usr/bin/env bash
set -u

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python3}"
SCRIPT="${SCRIPT:-/root/autodl-tmp/oakink_scripts/evaluate_oakink_sequence_aware_handover_detector.py}"
ONLINE_PREDICTIONS="${ONLINE_PREDICTIONS:-/root/autodl-tmp/oakink_online_handover_detector_v1_20260921/online_predictions.npz}"
CANDIDATES="${CANDIDATES:-/root/autodl-tmp/oakink_handover_manifest_v1_20260921/role_switch_candidates.jsonl.gz}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/oakink_sequence_aware_detector_v1_20260921}"
THRESHOLD="${THRESHOLD:-0.6525985410038043}"

mkdir -p "${OUTPUT_DIR}"
rm -f \
  "${OUTPUT_DIR}/SUCCESS" \
  "${OUTPUT_DIR}/FAILED" \
  "${OUTPUT_DIR}/exit_code"

date -Is > "${OUTPUT_DIR}/started_at"
"${PYTHON_BIN}" "${SCRIPT}" \
  --online-predictions "${ONLINE_PREDICTIONS}" \
  --candidates "${CANDIDATES}" \
  --threshold "${THRESHOLD}" \
  --output-dir "${OUTPUT_DIR}" \
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
if [ "${AUTO_SHUTDOWN:-0}" = "1" ]; then
  sleep "${SHUTDOWN_DELAY_SECONDS:-30}"
  shutdown -h now > "${OUTPUT_DIR}/shutdown.log" 2>&1 || true
fi
