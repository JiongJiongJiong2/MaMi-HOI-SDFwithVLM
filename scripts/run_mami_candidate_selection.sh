#!/usr/bin/env bash
set -euo pipefail

: "${REPO_ROOT:?Set REPO_ROOT to the MaMi source snapshot}"
: "${PYBIN:?Set PYBIN to the mami_hoi Python executable}"
: "${DATA_ROOT:?Set DATA_ROOT to processed_data}"
: "${BASELINE_CHECKPOINT:?Set BASELINE_CHECKPOINT to the frozen MaMi checkpoint}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to the full validation manifest}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT for candidate-selection artifacts}"
: "${WORLD_MODEL_CHECKPOINT:?Set WORLD_MODEL_CHECKPOINT to the v8 checkpoint}"

SEEDS="${SEEDS:-1 2 3 4}"
SCREEN_COUNT="${SCREEN_COUNT:-10}"
SCREEN_MANIFEST="${SCREEN_MANIFEST:-${OUTPUT_ROOT}/screen_manifest.json}"
SELECTION_JSON="${OUTPUT_ROOT}/candidate_selection.json"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/utils${PYTHONPATH:+:${PYTHONPATH}}"
unset OMP_NUM_THREADS
export SMPLH_PATH="${SMPLH_PATH:-${DATA_ROOT}/smpl_all_models/smplh_amass}"
mkdir -p "${OUTPUT_ROOT}"

"${PYBIN}" "${REPO_ROOT}/scripts/make_contact_selection_manifest.py" \
  --input "${SPLIT_MANIFEST}" \
  --output "${SCREEN_MANIFEST}" \
  --count "${SCREEN_COUNT}" \
  --seed 1

candidate_args=()
for seed in ${SEEDS}; do
  candidate_dir="${OUTPUT_ROOT}/candidate_seed_${seed}"
  mkdir -p "${candidate_dir}"
  "${PYBIN}" "${REPO_ROOT}/train/trainer_control_GAPA_chois.py" \
    --window=120 \
    --batch_size=32 \
    --data_root_folder="${DATA_ROOT}" \
    --pretrained_model="${BASELINE_CHECKPOINT}" \
    --project="${candidate_dir}/runtime" \
    --exp_name="candidate_seed_${seed}" \
    --save_res_folder="${candidate_dir}" \
    --seed="${seed}" \
    --experiment_split_manifest="${SCREEN_MANIFEST}" \
    --eval_split=validation \
    --input_first_human_pose \
    --use_random_frame_bps \
    --add_language_condition \
    --use_object_keypoints \
    --add_semantic_contact_labels \
    --loss_w_feet=1 \
    --loss_w_fk=0.5 \
    --loss_w_obj_pts=1 \
    --test_sample_res \
    --compute_metrics \
    --compute_hand_contact_metrics_v3 \
    --save_contact_protocol_fields \
    --skip_eval_mesh_export \
    --disable_amp
  candidate_args+=(--candidate_dir "${candidate_dir}")
done

"${PYBIN}" "${REPO_ROOT}/scripts/select_mami_candidates_with_world_model.py" \
  "${candidate_args[@]}" \
  --world_model_checkpoint="${WORLD_MODEL_CHECKPOINT}" \
  --data_root_folder="${DATA_ROOT}" \
  --output="${SELECTION_JSON}" \
  --device cuda \
  --max_sequences="${SCREEN_COUNT}"

echo "selection result: ${SELECTION_JSON}"
