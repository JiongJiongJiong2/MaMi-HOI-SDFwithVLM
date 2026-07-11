#!/usr/bin/env bash
set -euo pipefail

# Example:
# DATA_ROOT=/root/autodl-tmp/processed_data \
# BASELINE_CKPT=/root/autodl-tmp/baseline/weights/model-9.pt \
# bash scripts/train_dynamic_sdf.sh

: "${DATA_ROOT:?Set DATA_ROOT to the processed_data directory.}"
: "${BASELINE_CKPT:?Set BASELINE_CKPT to a MaMi-HOI baseline checkpoint.}"

PROJECT="${PROJECT:-./dynamic_sdf_experiments}"
EXP_NAME="${EXP_NAME:-E1_dynamic_sdf}"
TRAIN_STEPS="${TRAIN_STEPS:-20000}"
SAVE_EVERY="${SAVE_EVERY:-20000}"
LOSS_W_SDF="${LOSS_W_SDF:-1.0}"
SEED="${SEED:-1}"

python ./train/trainer_control_GAPA_chois.py \
  --window=120 \
  --batch_size=32 \
  --data_root_folder="${DATA_ROOT}" \
  --project="${PROJECT}" \
  --exp_name="${EXP_NAME}" \
  --seed="${SEED}" \
  --wandb_pj_name="chois_dynamic_sdf" \
  --entity="" \
  --finetune_model="${BASELINE_CKPT}" \
  --train_num_steps="${TRAIN_STEPS}" \
  --save_and_sample_every="${SAVE_EVERY}" \
  --input_first_human_pose \
  --use_random_frame_bps \
  --add_language_condition \
  --use_object_keypoints \
  --loss_w_feet=1 \
  --loss_w_fk=0.5 \
  --loss_w_obj_pts=1 \
  --use_dynamic_sdf \
  --loss_w_sdf="${LOSS_W_SDF}"
