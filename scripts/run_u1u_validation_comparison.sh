#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the processed_data directory.}"
: "${BASELINE_CKPT:?Set BASELINE_CKPT to the frozen U0 checkpoint.}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen validation/test JSON.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for the comparison artifacts.}"

U1U_CKPT="${U1U_CKPT:-${OUTPUT_ROOT}/U1U_smoke_300_fp32_candidate_20260911/weights/model-final-300.pt}"
COMPARISON_NAME="${COMPARISON_NAME:-U1U_validation_comparison_20260911}"
COMPARISON_ROOT="${OUTPUT_ROOT}/${COMPARISON_NAME}"

if [[ -e "${COMPARISON_ROOT}" ]]; then
  echo "Refusing to overwrite existing comparison directory: ${COMPARISON_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${U1U_CKPT}" ]]; then
  echo "Missing U1U checkpoint: ${U1U_CKPT}" >&2
  exit 2
fi

export DATA_ROOT BASELINE_CKPT SPLIT_MANIFEST OUTPUT_ROOT="${COMPARISON_ROOT}"
export DISABLE_AMP=1
export COMPUTE_HAND_CONTACT_METRICS_V3=1
export SKIP_EVAL_MESH_EXPORT=1
export EVAL_SPLIT=validation
export GUIDANCE=off

TRAIN_STEPS=300 SAVE_EVERY=300 SMOKE_TEST=0 DISABLE_AMP=1 \
  EXP_NAME=U0_FT_300 \
  bash scripts/train_u0_ft.sh

U0_FT_CKPT="${COMPARISON_ROOT}/U0_FT_300/weights/model-final-300.pt"
if [[ ! -f "${U0_FT_CKPT}" ]]; then
  echo "Missing U0-FT checkpoint: ${U0_FT_CKPT}" >&2
  exit 2
fi

EVAL_CKPT="${BASELINE_CKPT}" ROLE=U0 GUIDANCE=off \
  bash scripts/evaluate_u0_u1.sh

EVAL_CKPT="${U0_FT_CKPT}" ROLE=U0_FT GUIDANCE=off \
  bash scripts/evaluate_u0_u1.sh

EVAL_CKPT="${U1U_CKPT}" ROLE=U1U GUIDANCE=off \
  bash scripts/evaluate_u0_u1.sh

echo "Validation comparison complete: ${COMPARISON_ROOT}"
