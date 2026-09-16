#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/contact_action_20260914}"
SNAPSHOT="${SNAPSHOT:-$ROOT/repo/source-snapshot-hoidyn-20260914}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-$ROOT/mami_selection_v0_10_20260915}"
TRAINING_ROOT="${TRAINING_ROOT:-$ROOT/learned_residual_20260916/formal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/learned_residual_scorer_ab_20260916/formal}"
SEEDS=(${SEEDS_OVERRIDE:-21 22 23})

cd "$SNAPSHOT"

for seed in "${SEEDS[@]}"; do
  metrics="$TRAINING_ROOT/seed_${seed}/B_learned_residual/metrics.json"
  while [[ ! -f "$metrics" ]]; do
    echo "waiting for $metrics"
    sleep 60
  done
done

for seed in "${SEEDS[@]}"; do
  checkpoint="$TRAINING_ROOT/seed_${seed}/B_learned_residual/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  fi
  echo "running scorer A/B for seed=$seed"
  CANDIDATE_ROOT="$CANDIDATE_ROOT" \
  CHECKPOINT="$checkpoint" \
  OUTPUT_ROOT="$OUTPUT_ROOT/seed_${seed}" \
  bash scripts/run_contact_action_scorer_ab.sh formal
done

echo "WM Stage 1 scorer A/B complete"
