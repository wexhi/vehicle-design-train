#!/usr/bin/env bash
set -euo pipefail

# 双卡 DDP：每张卡约 16GB（如 2×A16）时使用。
# 注意：显存不能「合并」成 32GB，每张卡仍须单独装下 UNet+VAE+激活；因此务必开 gradient checkpointing，
# 若仍 OOM 可将 --resolution 改为 768 或安装 xformers 后加 --enable_xformers_memory_efficient_attention。

BASE="${SDXL_BASE:-/public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0}"
JSONL="${VEHICLE_ANNOTATION_JSONL:-$HOME/data/data/annotation_state/default/processed.jsonl}"
OUT="${SDXL_OUTPUT_DIR:-$HOME/data/train/sdxl/runs/run-$(date +%Y%m%d-%H%M)}"
NUM_GPUS="${NUM_GPUS:-2}"

mkdir -p "$OUT"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

exec uv run accelerate launch \
  --num_processes="$NUM_GPUS" \
  --multi_gpu \
  --mixed_precision=bf16 \
  -m vehicle_design_train.train_sdxl_lora \
  --pretrained_model_name_or_path="$BASE" \
  --annotation_jsonl="$JSONL" \
  --output_dir="$OUT" \
  --report_to=tensorboard \
  --logging_dir=logs \
  --resolution=1024 \
  --center_crop \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --learning_rate=1e-4 \
  --rank=16 \
  --gradient_checkpointing \
  --checkpointing_steps=500 \
  --num_train_epochs=3 \
  --validation_prompt_file="$ROOT/config/validation_prompts.json" \
  --num_validation_images=2 \
  --validation_steps=500 \
  --seed=42 \
  --caption_field="caption_en_short" \
  "$@"
