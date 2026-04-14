#!/usr/bin/env bash
set -euo pipefail

# GRPO-style post-training: group rollouts + ImageReward + DashScope VQA (prob) + DDPO loss.
# Requires DASHSCOPE_API_KEY unless --skip_vqa. ImageReward needs GPU memory in addition to SDXL.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="${SDXL_BASE:-/public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0}"
JSONL="${GRPO_JSONL:-$HOME/data/grpo/train.jsonl}"
OUT="${SDXL_GRPO_OUTPUT:-$HOME/data/train/sdxl_grpo/run-$(date +%Y%m%d-%H%M)}"
NUM_PROCS="${GRPO_NUM_PROCESSES:-}"
INFER_PROCS="${GRPO_NUM_INFERENCE_PROCESSES:-}"

mkdir -p "$OUT"

# Export DASHSCOPE_API_KEY for VQA; use --skip_vqa in "$@" for smoke runs without API.
ACCELERATE_ARGS=(--mixed_precision=bf16)
TRAIN_ARGS=()

if [[ -n "$NUM_PROCS" ]]; then
  ACCELERATE_ARGS+=(--num_processes="$NUM_PROCS" --multi_gpu)
fi

if [[ -n "$INFER_PROCS" ]]; then
  TRAIN_ARGS+=(--split_infer_train --num_inference_processes="$INFER_PROCS")
fi

exec uv run accelerate launch "${ACCELERATE_ARGS[@]}" -m vehicle_design_train.train_sdxl_grpo \
  --pretrained_model_name_or_path="$BASE" \
  --grpo_jsonl="$JSONL" \
  --output_dir="$OUT" \
  --resolution=1024 \
  --group_size=4 \
  --sample_num_steps=20 \
  --train_timestep_fraction=0.25 \
  --sample_guidance_scale=7.0 \
  --weight_ir=0.5 \
  --weight_vqa=0.5 \
  --vqa_model="${VQA_MODEL:-qwen3.5-35b-a3b}" \
  --gradient_checkpointing \
  --save_steps=50 \
  --log_image_steps=10 \
  "${TRAIN_ARGS[@]}" \
  "$@"
