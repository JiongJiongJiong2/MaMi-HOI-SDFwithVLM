#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the processed_data directory.}"
: "${BASELINE_CKPT:?Set BASELINE_CKPT to a MaMi-HOI baseline checkpoint.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for HOI-Dyn artifacts.}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen validation/test JSON.}"
: "${HOI_DYN_CONFIG:?Set HOI_DYN_CONFIG to the HOI-Dyn model YAML.}"
: "${HOI_DYN_CKPT:?Set HOI_DYN_CKPT to the frozen HOI-Dyn checkpoint.}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/t2m_eval:${REPO_ROOT}/utils${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

EXP_NAME="${EXP_NAME:-HOI_DYN_300}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"
SAVE_EVERY="${SAVE_EVERY:-300}"
SEED="${SEED:-1}"
ENABLE_HOI_DYN="${ENABLE_HOI_DYN:-1}"
DISABLE_AMP="${DISABLE_AMP:-1}"
LOSS_W_HOI_DYN="${LOSS_W_HOI_DYN:-1.0}"
HOI_DYN_MAX_STEP="${HOI_DYN_MAX_STEP:-1}"
HOI_DYN_LOSS_TYPE="${HOI_DYN_LOSS_TYPE:-pc}"
HOI_DYN_DYN_LOSS_TYPE="${HOI_DYN_DYN_LOSS_TYPE:-res}"
HOI_DYN_CONTACT_SOURCE="${HOI_DYN_CONTACT_SOURCE:-pred}"
HOI_DYN_ROT_WEIGHT="${HOI_DYN_ROT_WEIGHT:-0.05}"
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
  --wandb_pj_name="chois_hoi_dyn"
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

if [[ "${ENABLE_HOI_DYN}" == "1" ]]; then
  args+=(
    --use_hoi_dyn
    --hoi_dyn_config="${HOI_DYN_CONFIG}"
    --hoi_dyn_ckpt="${HOI_DYN_CKPT}"
    --loss_w_hoi_dyn="${LOSS_W_HOI_DYN}"
    --hoi_dyn_max_step="${HOI_DYN_MAX_STEP}"
    --hoi_dyn_loss_type="${HOI_DYN_LOSS_TYPE}"
    --hoi_dyn_dyn_loss_type="${HOI_DYN_DYN_LOSS_TYPE}"
    --hoi_dyn_contact_source="${HOI_DYN_CONTACT_SOURCE}"
    --hoi_dyn_rot_weight="${HOI_DYN_ROT_WEIGHT}"
  )
fi

if [[ "${DISABLE_AMP}" == "1" ]]; then
  args+=(--disable_amp)
fi

python ./train/trainer_control_GAPA_chois.py "${args[@]}" \
  2>&1 | tee "${run_dir}/console.log"
