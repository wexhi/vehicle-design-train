#!/usr/bin/env bash
set -euo pipefail

# 单卡启动。双卡各约 16GB（如 2×A16）请用 launch_train_sdxl_2gpu.sh（DDP，显存不合并，见 README）。

BASE="${SDXL_BASE:-/public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0}"
JSONL="${VEHICLE_ANNOTATION_JSONL:-$HOME/data/data/annotation_state/default/processed.jsonl}"
OUT="${SDXL_OUTPUT_DIR:-$HOME/data/train/sdxl/runs/run-$(date +%Y%m%d-%H%M)}"

mkdir -p "$OUT"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec uv run accelerate launch --mixed_precision=bf16 -m vehicle_design_train.train_sdxl_lora \
  --pretrained_model_name_or_path="$BASE" \
  --annotation_jsonl="$JSONL" \
  --output_dir="$OUT" \
  --report_to=tensorboard \
  --logging_dir=logs \
  --resolution=1024 \
  --center_crop \
  --train_batch_size=8 \
  --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 \
  --rank=16 \
  --checkpointing_steps=100 \
  --num_train_epochs=3 \
  --validation_prompt_file="$ROOT/config/validation_prompts.json" \
  --num_validation_images=2 \
  --validation_steps=100 \
  --seed=42 \
  --caption_field="caption_en_short" \
  "$@"
