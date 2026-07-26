#!/usr/bin/env bash
set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the processed_data directory.}"
: "${EVAL_CKPT:?Set EVAL_CKPT to the frozen U0 or trained U1 checkpoint.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for evaluation artifacts.}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the frozen validation/test JSON.}"

ROLE="${ROLE:-U0}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
GUIDANCE="${GUIDANCE:-off}"
SEED="${SEED:-1}"
ENABLE_DYNAMIC_SDF_DIAGNOSTICS="${ENABLE_DYNAMIC_SDF_DIAGNOSTICS:-0}"
export SMPLH_PATH="${SMPLH_PATH:-${DATA_ROOT}/smpl_all_models/smplh_amass}"

if [[ "${GUIDANCE}" != "off" && "${GUIDANCE}" != "on" ]]; then
  echo "GUIDANCE must be 'off' or 'on'." >&2
  exit 2
fi
if [[ "${EVAL_SPLIT}" != "validation" && "${EVAL_SPLIT}" != "test" ]]; then
  echo "EVAL_SPLIT must be 'validation' or 'test'." >&2
  exit 2
fi

run_name="${ROLE}_${EVAL_SPLIT}_guidance-${GUIDANCE}"
run_dir="${OUTPUT_ROOT}/${run_name}"
mkdir -p "${run_dir}"

args=(
  --window=120
  --batch_size=32
  --data_root_folder="${DATA_ROOT}"
  --pretrained_model="${EVAL_CKPT}"
  --project="${OUTPUT_ROOT}/runtime"
  --exp_name="${run_name}"
  --save_res_folder="${run_dir}"
  --seed="${SEED}"
  --experiment_split_manifest="${SPLIT_MANIFEST}"
  --eval_split="${EVAL_SPLIT}"
  --input_first_human_pose
  --use_random_frame_bps
  --add_language_condition
  --use_object_keypoints
  --add_semantic_contact_labels
  --loss_w_feet=1
  --loss_w_fk=0.5
  --loss_w_obj_pts=1
  --test_sample_res
  --compute_metrics
)

if [[ "${GUIDANCE}" == "on" ]]; then
  args+=(--use_guidance_in_denoising)
fi
if [[ "${ENABLE_DYNAMIC_SDF_DIAGNOSTICS}" == "1" ]]; then
  args+=(--use_dynamic_sdf --dynamic_sdf_diagnostics)
fi

python ./train/trainer_control_GAPA_chois.py "${args[@]}" \
  2>&1 | tee "${run_dir}/console.log"
