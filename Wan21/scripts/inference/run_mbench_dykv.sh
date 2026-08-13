#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

: "${MBENCH_ROOT:?Set MBENCH_ROOT to an MBench-A dataset directory}"
WORK_DIR="${WORK_DIR:-output/mbench_adapter}"
OUTPUT_FOLDER="${OUTPUT_FOLDER:-output/mbench_dykv}"
MODEL_ID="${MODEL_ID:-minwm_dykv}"
NUM_OUTPUT_FRAMES="${NUM_OUTPUT_FRAMES:-100}"
DYKV_MEMORY_FRAMES="${DYKV_MEMORY_FRAMES:-8}"
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

python Wan21/scripts/evaluation/mbench_adapter.py prepare "${PREPARE_ARGS[@]}"

DATA_PATH="$WORK_DIR/prompts.txt" \
TRAJECTORY_PATH="$WORK_DIR/trajectories.txt" \
OUTPUT_FOLDER="$OUTPUT_FOLDER" \
NUM_OUTPUT_FRAMES="$NUM_OUTPUT_FRAMES" \
DYKV=1 \
DYKV_MEMORY_FRAMES="$DYKV_MEMORY_FRAMES" \
bash Wan21/scripts/inference/run_infer_causal_camera.sh

python Wan21/scripts/evaluation/mbench_adapter.py package \
  --dataset-root "$MBENCH_ROOT" \
  --cases "$WORK_DIR/cases.jsonl" \
  --generation-manifest "$OUTPUT_FOLDER/generation_manifest.jsonl" \
  --model-id "$MODEL_ID" \
  --link-mode "$LINK_MODE"

echo "MBench package ready: $MBENCH_ROOT/models/$MODEL_ID"
