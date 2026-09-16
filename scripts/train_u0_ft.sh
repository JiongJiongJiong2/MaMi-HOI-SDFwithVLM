#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the processed_data directory.}"
: "${BASELINE_CKPT:?Set BASELINE_CKPT to a MaMi-HOI baseline checkpoint.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for U0-FT artifacts.}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen validation/test JSON.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/t2m_eval${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

EXP_NAME="${EXP_NAME:-U0_FT_300}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"
SAVE_EVERY="${SAVE_EVERY:-300}"
SEED="${SEED:-1}"
SMOKE_TEST="${SMOKE_TEST:-0}"
DISABLE_AMP="${DISABLE_AMP:-1}"
export SMPLH_PATH="${SMPLH_PATH:-${DATA_ROOT}/smpl_all_models/smplh_amass}"

run_dir="${OUTPUT_ROOT}/${EXP_NAME}"
mkdir -p "${run_dir}"

args=(
  --window=120
  --batch_size=32
  --data_root_folder="${DATA_ROOT}"
  --project="${OUTPUT_ROOT}"
  --exp_name="${EXP_NAME}"
  --seed="${SEED}"
  --experiment_split_manifest="${SPLIT_MANIFEST}"
  --eval_split=validation
  --wandb_pj_name="chois_u0_ft"
  --entity=""
  --finetune_model="${BASELINE_CKPT}"
  --train_num_steps="${TRAIN_STEPS}"
  --save_and_sample_every="${SAVE_EVERY}"
  --input_first_human_pose
  --use_random_frame_bps
  --add_language_condition
  --use_object_keypoints
  --loss_w_feet=1
  --loss_w_fk=0.5
  --loss_w_obj_pts=1
)

if [[ "${SMOKE_TEST}" == "1" ]]; then
  args+=(--smoke_test)
fi
if [[ "${DISABLE_AMP}" == "1" ]]; then
  args+=(--disable_amp)
fi

python ./train/trainer_control_GAPA_chois.py "${args[@]}" \
  2>&1 | tee "${run_dir}/console.log"
