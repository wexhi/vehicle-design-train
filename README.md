# vehicle-design-train

基于本地 **Stable Diffusion XL Base 1.0** 与车辆标注 `processed.jsonl` 的 **UNet LoRA** 微调（Diffusers 官方 `train_text_to_image_lora_sdxl` 逻辑，已内嵌并支持 `--annotation_jsonl`）。

## 环境

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)；默认 PyPI 镜像在 [`pyproject.toml`](pyproject.toml) 中配置为阿里云，`torch` / `torchvision` 从 PyTorch CUDA 12.4 轮子索引安装。

```bash
cd /path/to/vehicle-design-train
uv sync
```

## 数据

- 标注：`~/data/data/annotation_state/default/processed.jsonl`（或任意路径）
- 默认过滤：`error` 为空、`car_count == 1`、`person_count == 0`；caption 由 `--caption_field` 指定（如 `positive_prompt_en` 或 `caption_en_short`）
- 可选：`--require_complete_vehicle`、`--image_path_prefix_old` / `--image_path_prefix_new` 重写图片绝对路径前缀

## Caption 长度（CLIP 77 token）

```bash
uv run vdt-token-stats \
  --annotation-jsonl "$HOME/data/data/annotation_state/default/processed.jsonl" \
  --pretrained-model /public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0 \
  --max-samples 2000
```

## 训练（导出目录建议）

将 `--output_dir` 指到 `~/data/train/sdxl/runs/<run_name>/`。

```bash
mkdir -p "$HOME/data/train/sdxl/runs/run1"

uv run accelerate launch --mixed_precision=bf16 -m vehicle_design_train.train_sdxl_lora \
  --pretrained_model_name_or_path=/public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0 \
  --annotation_jsonl="$HOME/data/data/annotation_state/default/processed.jsonl" \
  --output_dir="$HOME/data/train/sdxl/runs/run1" \
  --report_to=tensorboard --logging_dir=logs \
  --resolution=1024 --center_crop \
  --train_batch_size=1 --gradient_accumulation_steps=4 \
  --learning_rate=1e-4 --rank=16 \
  --checkpointing_steps=500 \
  --num_train_epochs=3 \
  --validation_prompt_file=./config/validation_prompts.json \
  --num_validation_images=2 \
  --validation_steps=500 \
  --seed=42
```

首次使用请配置 Accelerate（交互或默认）：

```bash
uv run accelerate config
```

### 双卡（例如 2×NVIDIA A16，各约 16GB）

**重要**：两块 16GB 是 **两张卡各 16GB**，不是一张 32GB。多卡 **DDP** 会在每张卡上各放一份模型，显存 **不能相加**；双卡主要提高吞吐，并让 **全局 batch ≈ `train_batch_size × GPU 数 × gradient_accumulation_steps`**。

- 推荐直接用仓库脚本（已带 **`--gradient_checkpointing`**，双卡时把梯度累积略减为 2，全局等效 batch 仍为约 4；可按需改回 4）：

```bash
chmod +x scripts/launch_train_sdxl_2gpu.sh
./scripts/launch_train_sdxl_2gpu.sh
```

- 或手动（指定 2 进程 + 多卡）：

```bash
CUDA_VISIBLE_DEVICES=0,1 uv run accelerate launch --num_processes=2 --multi_gpu --mixed_precision=bf16 \
  -m vehicle_design_train.train_sdxl_lora \
  ... # 其余参数同单卡，并务必加上 --gradient_checkpointing
```

若仍显存不足：把 **`--resolution 768`**，或安装 **xformers** 后增加 **`--enable_xformers_memory_efficient_attention`**（需环境与包可用）。

`accelerate config` 里选择 **multi-GPU**、进程数 **2** 后，也可继续用单卡启动命令，由配置文件决定进程数（与命令行 `--num_processes` 二选一、勿混用冲突即可）。

## 验证频率

- **`--validation_steps N`（`N > 0`）**：每 **N 个全局 optimizer step** 做一次出图验证（TensorBoard / W&B 的横轴为 **global_step**）。启动脚本默认 `500`。
- **`--validation_steps 0`**（默认）：按 **`--validation_epochs`** 每个 epoch 结束验证（旧行为）。

## 多组验证 Prompt（奔驰 / 宝马 / 奥迪 / 大众等）

- **`--validation_prompt_file`**：指向 JSON，可为字符串列表，或 `{"prompts": [...]}`；每条支持 `{"id": "tensorboard子标签", "text": "英文 prompt"}`（`prompt` / `caption` 字段也可作为正文）。
- 默认示例：[`config/validation_prompts.json`](config/validation_prompts.json)（含四家品牌 + 多视角/细节；其中 **`vw_hatch_rear34_detail`** 对应「大众深灰掀背、后 45°、贯穿尾灯、黑轮毂」类描述）。
- **`--num_validation_images`**：对每个 prompt 各生成几张图；多 prompt 时显存与时间线性增长，启动脚本默认 **`2`**。
- 未指定文件时仍可用单行 **`--validation_prompt`**。
- TensorBoard **IMAGES** 里按标签 **`validation/<id>`** 分开展示（如 `validation/mercedes_suv_front`）。
- 验证时终端会显示 **两层进度**：外层按 prompt、内层按每张图；单次 `pipeline` 推理仍会显示 Diffusers 自带的 **去噪步数** 进度条。

## TensorBoard

训练脚本默认 **`--report_to=tensorboard`**，项目已声明依赖 **`tensorboard`**，用于记录 `train_loss`、`lr` 等；验证出图时也会写入 TensorBoard 图像（见脚本中 `log_validation`）。

日志写在 **`$OUTPUT_DIR/logs/`** 下（由 `--logging_dir` 控制，默认 `logs`）。TensorBoard 会递归读取子目录中的事件文件：

```bash
uv run tensorboard --logdir "$HOME/data/train/sdxl/runs/run1/logs" --port 6006
```

（必须提供 **`--logdir`** 指向某次 run 的 `logs` 目录；仅写 `--port` 会找不到数据。）

若启动 TensorBoard 时报错 **`No module named 'pkg_resources'`**：本项目已在依赖里固定 **`setuptools<81`**（TensorBoard 仍依赖 `pkg_resources`；setuptools 82+ 在部分环境下不可用）。在项目根执行 **`uv sync`** 后再运行上述命令。

改用 Weights & Biases 时传 **`--report_to=wandb`**（需自行安装并登录 `wandb`）。

## 推理网格（基座 + LoRA）

```bash
mkdir -p "$HOME/data/train/sdxl/samples/grid1"

uv run vdt-eval-grid \
  --pretrained-model /public/huggingface-models/stabilityai/stable-diffusion-xl-base-1.0 \
  --lora-path "$HOME/data/train/sdxl/runs/run1" \
  --prompts-json ./config/eval_prompts.json \
  --out-dir "$HOME/data/train/sdxl/samples/grid1"
```

## GRPO / DDPO 风格后训练（实验性）

在监督 LoRA 之后，可用 **`train_sdxl_grpo`** 做「每组 **G** 张 rollout → **ImageReward** + **百炼多模态 VQA（logprob → VQAScore 语义）** → 组内相对优势 → **DDIM 高斯 log-prob 的 DDPO 裁剪损失**」更新 UNet LoRA（见仓库内实现与 [Diffusers DDPO 说明](https://huggingface.co/docs/diffusers/training/ddpo)）。

- **数据**：v3 JSONL，默认字段 **`prompt_en`**；**`judge_requirements.judge_questions`**（`question_en` / `expected_answer`）。实现中会**额外**增加一道英文整段题：*`Does this image reflect the following description: "{prompt_en}"? Please answer yes or no.`*（模板可用 `--global_question_template_en` 覆盖）。
- **环境变量**：**`DASHSCOPE_API_KEY`**（百炼）；可选 **`VQA_MODEL`**（默认 `qwen3.5-35b-a3b`）。无密钥时用 **`--skip_vqa`** 仅跑 ImageReward（或两者都关：`--skip_vqa --skip_imagereward` 仅测管线）。
- **显存**：约等于「SDXL 推理 × **G** + ImageReward + 反传」；可调小 **`--resolution`**、**`--group_size`**、**`--sample_num_steps`**、**`--train_timestep_fraction`**，并建议 **`--gradient_checkpointing`**。
- **多卡分组（实验性）**：支持把进程拆成「前半组 rollout/reward，后半组 DDP 训练」，用于类似 LLM GRPO 的推理卡/训练卡分离。当前实现要求两组进程数相等；若样本数不能整除组数，会在每个 epoch 丢弃少量尾样本以保持 DDP 步数一致。
- **启动示例**：[scripts/launch_train_sdxl_grpo.sh](scripts/launch_train_sdxl_grpo.sh)（可按需改 `SDXL_BASE`、`GRPO_JSONL`、`SDXL_GRPO_OUTPUT`）。

```bash
chmod +x scripts/launch_train_sdxl_grpo.sh
export DASHSCOPE_API_KEY="your-key"
./scripts/launch_train_sdxl_grpo.sh \
  --grpo_jsonl /path/to/train.jsonl \
  --output_dir "$HOME/data/train/sdxl_grpo/run1"
```

四卡示例（2 卡推理 + 2 卡训练）：

```bash
export DASHSCOPE_API_KEY="your-key"
export GRPO_NUM_PROCESSES=4
export GRPO_NUM_INFERENCE_PROCESSES=2
CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/launch_train_sdxl_grpo.sh \
  --grpo_jsonl /path/to/train.jsonl \
  --output_dir "$HOME/data/train/sdxl_grpo/run_split_2x2"
```

- **可选 LoRA 起点**：**`--lora_path`** 指向此前保存的 **`unet_lora.safetensors`**（本脚本保存格式）或 Diffusers 兼容目录；**`--merge_lora_into_unet`** 会先融合再挂新 LoRA（显存更高，慎用）。

## 可选 CLIP 打分

需为每张图提供 `manifest`（文件名与 prompt 对齐）时使用 `--manifest-json`。

```bash
uv run vdt-eval-metrics --images-dir "$HOME/data/train/sdxl/samples/grid1" --manifest-json path/to/manifest.json
```

## 许可证

内嵌训练脚本版权归 Hugging Face（Apache 2.0）；请参阅文件头注释。
