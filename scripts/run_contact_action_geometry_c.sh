#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-/root/autodl-tmp/contact_action_20260914}"
SNAPSHOT="${SNAPSHOT:-$ROOT/repo/source-snapshot-hoidyn-20260914}"
PYBIN="${PYBIN:-/root/miniconda3/envs/mami_hoi/bin/python}"
DATA="${DATA:-$ROOT/large_event_20260914}"
PROCESSED="${PROCESSED:-/root/autodl-tmp/mamihoi/data/processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/local_geometry_c_20260915/$MODE}"

cd "$SNAPSHOT"
unset OMP_NUM_THREADS
export PYTHONPATH="$SNAPSHOT/utils:$SNAPSHOT"

if [[ "$MODE" == "smoke" ]]; then
  EPOCHS=2
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-0}"
  MAX_TRAIN=1024
  MAX_VAL=512
  EVAL_BATCH=128
  BATCH_SIZE=128
elif [[ "$MODE" == "formal" ]]; then
  EPOCHS=30
  read -r -a SEEDS <<< "${SEEDS_OVERRIDE:-11 12 13}"
  MAX_TRAIN=0
  MAX_VAL=0
  EVAL_BATCH=512
  BATCH_SIZE=512
else
  echo "usage: $0 [smoke|formal]" >&2
  exit 2
fi

for seed in "${SEEDS[@]}"; do
  output_dir="$OUTPUT_ROOT/seed_${seed}/C"
  mkdir -p "$output_dir"
  echo "running seed=$seed arm=C"
  "$PYBIN" scripts/train_contact_action_world_model.py \
    --train_npz "$DATA/train.npz" \
    --val_npz "$DATA/val.npz" \
    --output_dir "$output_dir" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate 0.001 \
    --weight_decay 0.0001 \
    --hidden_size 128 \
    --seed "$seed" \
    --train_mode mixed \
    --teacher_forcing_ratio 0.5 \
    --event_pos_weight 4.0 \
    --contact_calibration_weight 0.2 \
    --residual_scale 0.0 \
    --max_eval_horizon 8 \
    --geometry_mode normal \
    --geometry_data_root "$PROCESSED" \
    --geometry_patch_grid 5 \
    --geometry_radius_normalized 0.08 \
    --geometry_output_size 64 \
    --contrastive_weight 0.1 \
    --contrastive_temperature 0.1 \
    --max_train_samples "$MAX_TRAIN" \
    --max_val_samples "$MAX_VAL" \
    --eval_batch_size "$EVAL_BATCH" \
    --device cuda \
    > "$output_dir/train.log" 2>&1
done
