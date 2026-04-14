#!/usr/bin/env bash
# 在「单独目录 + 单独 venv」里启动 vLLM，供 GRPO 训练侧 --vqa_backend vllm_openai 调用。
# 勿与 vehicle-design-train 训练用 .venv 混装 vLLM（避免与 diffusers/torch 版本冲突）。
#
# 一次性准备示例（目录可改）：
#   mkdir -p "$HOME/vllm-serve" && cd "$HOME/vllm-serve"
#   uv venv
#   source .venv/bin/activate
#   uv pip install vllm --torch-backend auto --extra-index-url https://wheels.vllm.ai/nightly
#
# 环境变量（均可选）：
#   VLLM_ROOT          含 .venv 的目录，默认 $HOME/vllm-serve
#   VLLM_VENV          显式指定 venv 路径，默认 $VLLM_ROOT/.venv
#   VLLM_MODEL         HuggingFace 模型 id，默认 Qwen/Qwen3.5-9B
#   VLLM_HOST          监听地址，默认 0.0.0.0
#   VLLM_PORT          端口，默认 8000（训练侧 OPENAI_BASE_URL 应对应 http://127.0.0.1:8000/v1）
#   VLLM_TENSOR_PARALLEL_SIZE  默认 1
#   VLLM_MAX_MODEL_LEN       默认 32768（VQA 短答可降显存；要满上下文可设 262144 等）
#   VLLM_REASONING_PARSER    默认 qwen3（与 Qwen3.5 模型卡一致）
#   VLLM_CUDA_VISIBLE_DEVICES  只给 vLLM 看的物理 GPU，例如 0（训练用 GRPO_GPU_IDS=1 占另一张卡）
#   CUDA_VISIBLE_DEVICES       未设 VLLM_CUDA_VISIBLE_DEVICES 时也可用其指定 vLLM 所用 GPU
#   HF_ENDPOINT              默认 https://hf-mirror.com
#
# 额外参数会原样传给 vllm，例如：
#   ./scripts/launch_vllm_qwen35_grpo_vqa.sh --gpu-memory-utilization 0.85

set -euo pipefail

VLLM_ROOT="${VLLM_ROOT:-$HOME/venvs/vllm-serve}"
VENV_PATH="${VLLM_VENV:-$VLLM_ROOT/.venv}"

if [[ ! -f "$VENV_PATH/bin/activate" ]]; then
  echo "错误：未找到 venv：$VENV_PATH" >&2
  echo "请先在该目录创建环境并安装 vLLM，例如：" >&2
  echo "  mkdir -p \"$VLLM_ROOT\" && cd \"$VLLM_ROOT\" && uv venv && source .venv/bin/activate" >&2
  echo "  uv pip install vllm --torch-backend auto --extra-index-url https://wheels.vllm.ai/nightly" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"

if ! command -v vllm &>/dev/null; then
  echo "错误：在 $VENV_PATH 中未找到 vllm 命令，请在该 venv 内安装 vLLM。" >&2
  exit 1
fi

MODEL="${VLLM_MODEL:-/public/huggingface-models/Qwen/Qwen3.5-9B}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"
TP="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
REASONING="${VLLM_REASONING_PARSER:-qwen3}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -n "${VLLM_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES"
fi

echo "[launch_vllm] venv=$VENV_PATH model=$MODEL host=$HOST port=$PORT tp=$TP max_model_len=$MAX_LEN CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
echo "[launch_vllm] OpenAI base for GRPO: http://127.0.0.1:${PORT}/v1  (本机)  训练加: --vqa_backend vllm_openai --vqa_model $MODEL" >&2

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --reasoning-parser "$REASONING" \
  "$@"
