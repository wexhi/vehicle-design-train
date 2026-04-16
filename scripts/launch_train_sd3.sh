#!/usr/bin/env bash
set -euo pipefail

# 单卡启动 SD3.5 Medium 文本条件 LoRA 微调（与 launch_train_sdxl.sh 对齐：annotation_jsonl + TensorBoard）。
# SD3 显存占用高于 SDXL，默认较小 batch + gradient checkpointing；可按显存调高 --train_batch_size。

BASE="${SD3_BASE:-/public/huggingface-models/stabilityai/stable-diffusion-3.5-medium}"
JSONL="${VEHICLE_ANNOTATION_JSONL:-$HOME/data/data/annotation_state/default/processed.jsonl}"
OUT="${SD3_OUTPUT_DIR:-$HOME/data/train/sd3/runs/run-$(date +%Y%m%d-%H%M)}"

mkdir -p "$OUT"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exec uv run accelerate launch --mixed_precision=bf16 -m vehicle_design_train.train_sd3_lora \
  --pretrained_model_name_or_path="$BASE" \
  --annotation_jsonl="$JSONL" \
  --output_dir="$OUT" \
  --report_to=tensorboard \
  --logging_dir=logs \
  --resolution=1024 \
  --center_crop \
  --train_batch_size=2 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --learning_rate=1e-4 \
  --rank=16 \
  --checkpointing_steps=50 \
  --num_train_epochs=1 \
  --validation_prompt_file="$ROOT/config/validation_prompts.json" \
  --num_validation_images=2 \
  --validation_steps=50 \
  --seed=42 \
  --caption_field="caption_en_short" \
  "$@"
