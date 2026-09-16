#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/contact_action_20260914}"
SNAPSHOT="${SNAPSHOT:-$ROOT/repo/source-snapshot-hoidyn-20260914}"
PYBIN="${PYBIN:-/root/miniconda3/envs/mami_hoi/bin/python}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/mamihoi/data/processed_data}"
CHECKPOINT="${CHECKPOINT:-$ROOT/large_event_20260914/model_v8_gpu_fast/best.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/analytic_contact_baseline_v1_20260916}"

declare -A CANDIDATE_ROOTS=(
  ["10_30"]="${ROOT}/mami_l1_heldout_20_20260915"
  ["31_50"]="${ROOT}/mami_l1_heldout_31_50_20260915"
  ["51_70"]="${ROOT}/mami_l1_heldout_51_70_20260915"
)

cd "$SNAPSHOT"
unset OMP_NUM_THREADS
export PYTHONPATH="$SNAPSHOT/utils:$SNAPSHOT"

for cohort in 10_30 31_50 51_70; do
  candidate_root="${CANDIDATE_ROOTS[$cohort]}"
  output_dir="$OUTPUT_ROOT/$cohort"
  mkdir -p "$output_dir"
  echo "running analytic cohort=$cohort"
  "$PYBIN" scripts/run_contact_action_chunk_correction.py \
    --candidate_root "$candidate_root" \
    --world_model_checkpoint "$CHECKPOINT" \
    --data_root_folder "$DATA_ROOT" \
    --output "$output_dir/result.json" \
    --candidate_seed 1 \
    --history 4 \
    --horizon 8 \
    --stride 4 \
    --alpha 0.002 \
    --num_random 8 \
    --proxy_contact_threshold_norm 0.02 \
    --contact_threshold_m 0.05 \
    --penetration_weight 20.0 \
    --scorer_mode geom_only \
    --rollout_mode analytic \
    --event_weight 0.0 \
    --device cuda \
    > "$output_dir/run.log" 2>&1
done

echo "analytic contact baseline complete"
