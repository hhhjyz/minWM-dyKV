#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.."; pwd)"
cd "$PROJECT_ROOT"

export NCCL_DEBUG=WARN
export PYTHONPATH="$PWD/HY15:$PWD/Wan21:$PWD/shared:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0

CONFIG_PATH="Wan21/configs/causal_forcing_dmd_camera.yaml"
CHECKPOINT_PATH="./ckpts/Wan21/Action2V/dmd/model.pt"
DATA_PATH="output/attn_capture/prompts.txt"
TRAJECTORY_PATH="output/attn_capture/trajectories.txt"
NUM_OUTPUT_FRAMES=40
SEED=0
MASTER_PORT=29625
CAPTURE_LAYERS="0,5,10,15,20,25,29"

for CASE in retrieval_no_compression_honest motion_novelty_backfill_honest; do
  OUTPUT_FOLDER="output/attn_capture/$CASE"
  echo "=== Generating with attention capture: $CASE ==="
  conda run --no-capture-output -n minwm-fa torchrun \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    Wan21/wan_inference.py \
    --config_path "$CONFIG_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_path "$DATA_PATH" \
    --num_output_frames $NUM_OUTPUT_FRAMES \
    --sp_size 1 \
    --seed $SEED \
    --dykv \
    --dykv-case "$CASE" \
    --trajectory_path "$TRAJECTORY_PATH" \
    --capture-attention "attn.json:${CAPTURE_LAYERS}"
  echo "=== Done: $CASE ==="
done

echo "Both cases completed. Attention captures saved under output/attn_capture/"
