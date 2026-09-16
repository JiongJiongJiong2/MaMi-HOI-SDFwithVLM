#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${ROOT:-/root/autodl-tmp/contact_action_20260914}"
SNAPSHOT="${SNAPSHOT:-$ROOT/repo/source-snapshot-hoidyn-20260914}"
PYBIN="${PYBIN:-/root/miniconda3/envs/mami_hoi/bin/python}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:?CANDIDATE_ROOT is required}"
CHECKPOINT="${CHECKPOINT:?CHECKPOINT is required}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp/mamihoi/data/processed_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/learned_residual_scorer_ab_20260916/$MODE}"

cd "$SNAPSHOT"
unset OMP_NUM_THREADS
export PYTHONPATH="$SNAPSHOT/utils:$SNAPSHOT"

run_arm() {
  local name="$1"
  local scorer_mode="$2"
  local rollout_mode="$3"
  local event_weight="$4"
  mkdir -p "$OUTPUT_ROOT/$name"
  echo "running scorer arm=$name"
  "$PYBIN" scripts/run_contact_action_chunk_correction.py \
    --candidate_root "$CANDIDATE_ROOT" \
    --world_model_checkpoint "$CHECKPOINT" \
    --data_root_folder "$DATA_ROOT" \
    --geometry_data_root "$DATA_ROOT" \
    --output "$OUTPUT_ROOT/$name/result.json" \
    --candidate_seed 1 \
    --history 4 \
    --horizon 8 \
    --stride 4 \
    --alpha 0.002 \
    --num_random 8 \
    --proxy_contact_threshold_norm 0.02 \
    --contact_threshold_m 0.05 \
    --penetration_weight 20.0 \
    --scorer_mode "$scorer_mode" \
    --rollout_mode "$rollout_mode" \
    --event_weight "$event_weight" \
    --device cuda \
    > "$OUTPUT_ROOT/$name/run.log" 2>&1
}

if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  echo "usage: $0 [smoke|formal]" >&2
  exit 2
fi

run_arm geom_only geom_only analytic 0.0
run_arm learned_state learned_state learned 0.0
run_arm hybrid_event hybrid_event learned 0.25
