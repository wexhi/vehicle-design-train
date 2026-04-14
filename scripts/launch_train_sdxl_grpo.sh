#!/usr/bin/env bash
set -euo pipefail

# GRPO-style post-training: group rollouts + ImageReward + DashScope VQA (prob) + DDPO loss.
# Requires DASHSCOPE_API_KEY unless --skip_vqa. ImageReward needs GPU memory in addition to SDXL.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Export variables from repo `.env` into the environment (so Accelerate child processes see DASHSCOPE_API_KEY).
if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

BASE="${SDXL_BASE:-/public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0}"
JSONL="${GRPO_JSONL:-$HOME/vehicle-design-train/data/train.jsonl}"
OUT="${SDXL_GRPO_OUTPUT:-$HOME/data/train/sdxl_grpo/run-$(date +%Y%m%d-%H%M)}"
# 默认挂载监督微调 checkpoint 下的 LoRA；可用 SDXL_GRPO_LORA_PATH 覆盖，或命令行 `--lora_path` 再覆盖。
LORA_PATH="${SDXL_GRPO_LORA_PATH:-$HOME/data/train/sdxl/runs/run-20260414-2131/checkpoint-300}"
# Multi-GPU: GRPO_NUM_PROCESSES = Accelerate 进程数；≥2 时加 --multi_gpu。
NUM_PROCS="${GRPO_NUM_PROCESSES:-1}"
# 仅当显式设置且多进程时才启用 split（勿默认 1，否则单卡也会带上 --split_infer_train 并报错）。
INFER_PROCS="${GRPO_NUM_INFERENCE_PROCESSES:-}"
# 非 split：GRPO_GPU_IDS=0,1,2,3 一次性 export 给所有进程。
GRPO_GPU_IDS="${GRPO_GPU_IDS:-}"
# split_infer_train：按 rank 分配物理卡（勿与 GRPO_GPU_IDS 同时 export 总表）。列表长度须与进程数一致：
#   len(GRPO_ROLLOUT_GPU_IDS) == GRPO_NUM_INFERENCE_PROCESSES
#   len(GRPO_TRAIN_GPU_IDS)    == GRPO_NUM_PROCESSES - GRPO_NUM_INFERENCE_PROCESSES
# 当前实现要求 推理进程数 == 训练进程数（例如 2+2）：例 rollout 用 2,3、train 用 0,1。
GRPO_TRAIN_GPU_IDS="${GRPO_TRAIN_GPU_IDS:-}"
GRPO_ROLLOUT_GPU_IDS="${GRPO_ROLLOUT_GPU_IDS:-}"

if [[ -n "$GRPO_TRAIN_GPU_IDS" && -n "$GRPO_ROLLOUT_GPU_IDS" ]]; then
  export GRPO_TRAIN_GPU_IDS GRPO_ROLLOUT_GPU_IDS
elif [[ -n "$GRPO_GPU_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GRPO_GPU_IDS"
fi

mkdir -p "$OUT"

# Export DASHSCOPE_API_KEY for VQA; use --skip_vqa in "$@" for smoke runs without API.
ACCELERATE_ARGS=(--mixed_precision=bf16)
TRAIN_ARGS=()

if [[ "$NUM_PROCS" =~ ^[0-9]+$ ]] && [[ "$NUM_PROCS" -ge 2 ]]; then
  ACCELERATE_ARGS+=(--num_processes="$NUM_PROCS" --multi_gpu)
fi

if [[ -n "$INFER_PROCS" ]] && [[ "$NUM_PROCS" =~ ^[0-9]+$ ]] && [[ "$NUM_PROCS" -ge 2 ]]; then
  TRAIN_ARGS+=(--split_infer_train --num_inference_processes="$INFER_PROCS")
fi

export HF_ENDPOINT=https://hf-mirror.com

exec uv run accelerate launch "${ACCELERATE_ARGS[@]}" -m vehicle_design_train.train_sdxl_grpo \
  --pretrained_model_name_or_path="$BASE" \
  --grpo_jsonl="$JSONL" \
  --lora_path="$LORA_PATH" \
  --output_dir="$OUT" \
  --resolution=1024 \
  --rollout_batch_size=4 \
  --train_batch_size=4 \
  --group_size=4 \
  --sample_num_steps=20 \
  --train_timestep_fraction=0.25 \
  --sample_guidance_scale=7.0 \
  --weight_ir=0.5 \
  --weight_vqa=0.5 \
  --vqa_model="${VQA_MODEL:-qwen3.5-122b-a10b}" \
  --vqa_max_workers=32 \
  --gradient_checkpointing \
  --save_steps=50 \
  --log_image_steps=2 \
  "${TRAIN_ARGS[@]}" \
  "$@"
