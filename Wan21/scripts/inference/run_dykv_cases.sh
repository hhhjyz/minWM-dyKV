#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

CASES="${CASES:-baseline,retrieval_no_compression,fixed_novelty,yaw_intrinsics,packed_chunks,packed_chunks_latent,predecessor_chunks,predecessor_chunks_latent,predecessor_query_backfill,retr8_compression_r050,retr12_compression_r050,retr16_compression_r033}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/dykv_cases}"
MODEL_PREFIX="${MODEL_PREFIX:-minwm_dykv}"
SEED="${SEED:-0}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-20}"
DRY_RUN="${DRY_RUN:-0}"
LIST_CASES="${LIST_CASES:-0}"
MBENCH_ROOT="${MBENCH_ROOT:-}"

if [ "$LIST_CASES" = "1" ]; then
  python Wan21/dykv_cases.py
  exit 0
fi

IFS=',' read -r -a CASE_LIST <<< "$CASES"
if [ "${#CASE_LIST[@]}" -eq 0 ]; then
  echo "CASES cannot be empty" >&2
  exit 2
fi
for case_name in "${CASE_LIST[@]}"; do
  python Wan21/dykv_cases.py --validate "$case_name"
done

DATA_PATH="${DATA_PATH:-Wan21/prompts/demos.txt}"
TRAJECTORY_PATH="${TRAJECTORY_PATH:-}"

if [ -n "$MBENCH_ROOT" ]; then
  WORK_DIR="${WORK_DIR:-$OUTPUT_ROOT/_mbench_input}"
  SUBSETS="${SUBSETS:-}"
  CONDITIONS="${CONDITIONS:-}"
  LIMIT="${LIMIT:-}"
  ASSIGNMENTS="${ASSIGNMENTS:-}"
  LINK_MODE="${LINK_MODE:-symlink}"
  PREPARE_ARGS=(
    --dataset-root "$MBENCH_ROOT"
    --work-dir "$WORK_DIR"
    --num-output-frames "$NUM_OUTPUT_FRAMES"
  )
  [ -n "$ASSIGNMENTS" ] && PREPARE_ARGS+=(--assignments "$ASSIGNMENTS")
  [ -n "$SUBSETS" ] && PREPARE_ARGS+=(--subsets "$SUBSETS")
  [ -n "$CONDITIONS" ] && PREPARE_ARGS+=(--conditions "$CONDITIONS")
  [ -n "$LIMIT" ] && PREPARE_ARGS+=(--limit "$LIMIT")
  if [ "$DRY_RUN" = "1" ]; then
    printf 'DRY_RUN: python Wan21/scripts/evaluation/mbench_adapter.py prepare'
    printf ' %q' "${PREPARE_ARGS[@]}"
    printf '\n'
  else
    python Wan21/scripts/evaluation/mbench_adapter.py prepare "${PREPARE_ARGS[@]}"
  fi
  DATA_PATH="$WORK_DIR/prompts.txt"
  TRAJECTORY_PATH="$WORK_DIR/trajectories.txt"
fi

for case_name in "${CASE_LIST[@]}"; do
  case_output="$OUTPUT_ROOT/$case_name"
  dykv_enabled=1
  [ "$case_name" = "baseline" ] && dykv_enabled=0

  echo "=== Running case: $case_name ==="
  (
    export DATA_PATH TRAJECTORY_PATH
    export OUTPUT_FOLDER="$case_output"
    export NUM_OUTPUT_FRAMES SEED DRY_RUN
    export DYKV="$dykv_enabled"
    export DYKV_CASE="$case_name"
    bash Wan21/scripts/inference/run_infer_causal_camera.sh
  )

  if [ -n "$MBENCH_ROOT" ]; then
    model_id="${MODEL_PREFIX}_${case_name}_seed${SEED}"
    PACKAGE_ARGS=(
      --dataset-root "$MBENCH_ROOT"
      --cases "$WORK_DIR/cases.jsonl"
      --generation-manifest "$case_output/generation_manifest.jsonl"
      --model-id "$model_id"
      --link-mode "$LINK_MODE"
    )
    if [ "$DRY_RUN" = "1" ]; then
      printf 'DRY_RUN: python Wan21/scripts/evaluation/mbench_adapter.py package'
      printf ' %q' "${PACKAGE_ARGS[@]}"
      printf '\n'
    else
      python Wan21/scripts/evaluation/mbench_adapter.py package "${PACKAGE_ARGS[@]}"
      echo "MBench package ready: $MBENCH_ROOT/models/$model_id"
    fi
  fi
done

echo "All requested cases completed under: $OUTPUT_ROOT"
