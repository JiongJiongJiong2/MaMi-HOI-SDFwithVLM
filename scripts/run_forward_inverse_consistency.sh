#!/usr/bin/env bash
set -euo pipefail

: "${TRAIN_NPZ:?Set TRAIN_NPZ}"
: "${VAL_NPZ:?Set VAL_NPZ}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"

PYBIN="${PYBIN:-/root/miniconda3/envs/mami_hoi/bin/python}"
EPOCHS="${EPOCHS:-40}"
SEEDS="${SEEDS:-11 12 13}"

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd):$(pwd)/utils${PYTHONPATH:+:$PYTHONPATH}"
for seed in ${SEEDS}; do
  mkdir -p "${OUTPUT_ROOT}/seed_${seed}"
  PYTHONUNBUFFERED=1 "${PYBIN}" -u scripts/train_forward_inverse_consistency.py \
    --train_npz "${TRAIN_NPZ}" \
    --val_npz "${VAL_NPZ}" \
    --output_dir "${OUTPUT_ROOT}/seed_${seed}" \
    --epochs "${EPOCHS}" \
    --seed "${seed}" \
    --device cuda
done
