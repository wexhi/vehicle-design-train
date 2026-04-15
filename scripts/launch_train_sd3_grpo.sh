#!/usr/bin/env bash
set -euo pipefail

# Flow-GRPO-style post-training for SD3.x / 3.5 (official ODE→SDE step + log-prob, see sd3_sde_with_logprob.py).
# Optional Flow-GRPO-Fast: add e.g. --flow_grpo_fast --fast_sde_window_size=2 (see sd3_rollout.py / flow_grpo sd3_pipeline_with_logprob_fast).
# See vehicle_design_train/train_sd3_grpo.py and arXiv:2505.05470.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
fi

BASE="${SD3_BASE:-/public/huggingface-models/stabilityai/stable-diffusion-3.5-medium}"
JSONL="${GRPO_JSONL:-$HOME/vehicle-design-train/data/train.jsonl}"
OUT="${SD3_GRPO_OUTPUT:-$HOME/data/train/sd3_grpo/run-$(date +%Y%m%d-%H%M)}"
LORA_PATH="${SD3_GRPO_LORA_PATH:-}"
NUM_PROCS="${GRPO_NUM_PROCESSES:-1}"
INFER_PROCS="${GRPO_NUM_INFERENCE_PROCESSES:-}"
GRPO_GPU_IDS="${GRPO_GPU_IDS:-}"
GRPO_TRAIN_GPU_IDS="${GRPO_TRAIN_GPU_IDS:-}"
GRPO_ROLLOUT_GPU_IDS="${GRPO_ROLLOUT_GPU_IDS:-}"

if [[ -n "$GRPO_TRAIN_GPU_IDS" && -n "$GRPO_ROLLOUT_GPU_IDS" ]]; then
  export GRPO_TRAIN_GPU_IDS GRPO_ROLLOUT_GPU_IDS
elif [[ -n "$GRPO_GPU_IDS" ]]; then
  export CUDA_VISIBLE_DEVICES="$GRPO_GPU_IDS"
fi

mkdir -p "$OUT"

ACCELERATE_ARGS=(--mixed_precision=bf16)
TRAIN_ARGS=()

if [[ "$NUM_PROCS" =~ ^[0-9]+$ ]] && [[ "$NUM_PROCS" -ge 2 ]]; then
  ACCELERATE_ARGS+=(--num_processes="$NUM_PROCS" --multi_gpu)
fi

if [[ -n "$INFER_PROCS" ]] && [[ "$NUM_PROCS" =~ ^[0-9]+$ ]] && [[ "$NUM_PROCS" -ge 2 ]]; then
  TRAIN_ARGS+=(--split_infer_train --num_inference_processes="$INFER_PROCS")
fi

LORA_ARGS=()
if [[ -n "$LORA_PATH" ]]; then
  LORA_ARGS+=(--lora_path="$LORA_PATH")
fi

exec uv run accelerate launch "${ACCELERATE_ARGS[@]}" -m vehicle_design_train.train_sd3_grpo \
  --pretrained_model_name_or_path="$BASE" \
  --grpo_jsonl="$JSONL" \
  --output_dir="$OUT" \
  "${LORA_ARGS[@]}" \
  --train_timestep_sample_mode=group_shared_stratified \
  --resolution=1024 \
  --rollout_batch_size=1 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --group_size=2 \
  --sample_num_steps=10 \
  --train_timestep_fraction=0.25 \
  --sample_guidance_scale=4.5 \
  --noise_level=0.7 \
  --max_sequence_length=256 \
  --weight_ir=1 \
  --weight_vqa=0 \
  --gradient_checkpointing \
  --save_steps=50 \
  --logging_steps=1 \
  "${TRAIN_ARGS[@]}" \
  "$@"
