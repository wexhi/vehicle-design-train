# vehicle-design-train

面向车辆生成任务的 **Stable Diffusion 3.5 Medium（SD3.5 Medium）** 训练仓库，包含两条主线：

1. **监督式 LoRA 微调（SFT）**：`vehicle_design_train.train_sd3_lora`
2. **Flow-GRPO 后训练**：`vehicle_design_train.train_sd3_grpo`

> ⚠️ **SDXL 路线已废弃（deprecated）**：仓库中与 SDXL 相关的脚本/实现仅为历史保留，不再维护，也不建议继续使用。

---

## 1) 环境准备

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

```bash
cd /workspace/vehicle-design-train
uv sync
```

首次使用 Accelerate：

```bash
uv run accelerate config
```

---

## 2) 数据准备

### 2.1 SD3.5 LoRA 微调数据（annotation jsonl）

- 典型文件：`~/data/data/annotation_state/default/processed.jsonl`
- 常用字段：
  - 图像路径
  - caption（默认示例脚本使用 `caption_en_short`）
- 训练脚本支持常见过滤与路径重写参数（例如完整车体过滤、路径前缀替换等）。

### 2.2 Flow-GRPO 数据（v3 jsonl）

- 默认示例：`data/train.jsonl`
- 必需字段：
  - `prompt_en`（可通过 `--prompt_field` 修改）
  - `judge_requirements.judge_questions`（用于 VQA/规则打分）

---

## 3) SD3.5 Medium 监督式 LoRA 微调（重点）

主脚本：`vehicle_design_train/train_sd3_lora.py`

推荐直接使用仓库启动脚本（已包含一组保守显存参数）：

```bash
chmod +x scripts/launch_train_sd3.sh
./scripts/launch_train_sd3.sh
```

### 3.1 默认配置说明（`scripts/launch_train_sd3.sh`）

- 基座模型：`stable-diffusion-3.5-medium`
- 分辨率：`1024`
- `train_batch_size=2`
- `gradient_accumulation_steps=8`
- `gradient_checkpointing` 开启
- LoRA rank：`16`
- 混合精度：`bf16`
- 日志：TensorBoard（`$OUT/logs`）

可用环境变量覆盖：

- `SD3_BASE`：SD3.5 Medium 模型路径
- `VEHICLE_ANNOTATION_JSONL`：训练标注 jsonl
- `SD3_OUTPUT_DIR`：输出目录

也可继续在命令后附加参数覆写默认值，例如：

```bash
./scripts/launch_train_sd3.sh --num_train_epochs=3 --validation_steps=100
```

### 3.2 训练产物

训练目录中通常包含：

- LoRA 权重与 checkpoint
- TensorBoard 日志（标量与验证图）

查看日志：

```bash
uv run tensorboard --logdir "$HOME/data/train/sd3/runs/<run_name>/logs" --port 6006
```

---

## 4) SD3.5 Medium + Flow-GRPO 后训练（重点）

主脚本：`vehicle_design_train/train_sd3_grpo.py`

该实现是 **SD3 Flow-Match 路线的 GRPO 风格后训练**，核心流程：

1. 对每条 prompt 采样一组 `group_size=G` 张图（rollout）
2. 使用奖励模型计算每张图的 reward（可组合）
3. 组内计算 advantage（相对优势）
4. 使用 Flow-GRPO clipped objective 更新 LoRA 参数

推荐从脚本启动：

```bash
chmod +x scripts/launch_train_sd3_grpo.sh
./scripts/launch_train_sd3_grpo.sh
```

### 4.1 默认配置说明（`scripts/launch_train_sd3_grpo.sh`）

脚本默认采用一套偏“可跑通 + 可观察”的设置：

- `--flow_grpo_fast`
- `--fast_sde_window_size=2`
- `--group_size=8`
- `--rollout_batch_size=8`
- `--train_batch_size=8`
- `--sample_num_steps=10`
- `--train_timestep_fraction=0.25`
- `--noise_level=0.7`
- `--max_sequence_length=256`
- `--gradient_checkpointing`

并默认开启 rollout 可视化日志（JSON/TensorBoard，可选存图）。

### 4.2 奖励与打分

当前训练入口支持组合奖励（按权重求和），包括：

- ImageReward
- VQA 概率打分（DashScope 或 vLLM OpenAI 兼容后端）
- GenEval 远程打分（可选）

默认脚本参数中：

- `--weight_ir=1`
- `--weight_vqa=0`

即默认主要依赖 ImageReward。你可以在启动命令中开启并调高 VQA 权重。

### 4.3 多卡与 split 推理/训练

当 `GRPO_NUM_PROCESSES>=2` 时，脚本会用 accelerate 多进程。

可选 split 模式（部分进程 rollout，部分进程 train）：

- `GRPO_NUM_INFERENCE_PROCESSES`
- `GRPO_TRAIN_GPU_IDS`
- `GRPO_ROLLOUT_GPU_IDS`

不 split 时可直接用：

- `GRPO_GPU_IDS`

### 4.4 常用环境变量

- `GRPO_JSONL`：GRPO 训练 jsonl
- `SD3_GRPO_OUTPUT`：输出目录
- `SD3_GRPO_LORA_PATH`：从已有 LoRA 初始化
- `GRPO_NUM_PROCESSES`：总进程数
- `GRPO_NUM_INFERENCE_PROCESSES`：split 模式下推理进程数
- `SD3_LOG_IMAGE_STEPS`：rollout 日志频率
- `SD3_MAX_LOGGED_IMAGES`：每次记录的最大图数
- `SD3_SAVE_ROLLOUT_SAMPLE_IMAGES=0/1`：是否保存 rollout 图

### 4.5 一个实用启动示例

```bash
export GRPO_JSONL=/path/to/train.jsonl
export SD3_GRPO_OUTPUT=$HOME/data/train/sd3_grpo/run1
export GRPO_NUM_PROCESSES=1

./scripts/launch_train_sd3_grpo.sh \
  --weight_vqa=1 \
  --vqa_backend=vllm_openai \
  --vqa_model=Qwen/Qwen3.5-9B
```

---

## 5) 目录速览（SD3 主链路）

- `vehicle_design_train/train_sd3_lora.py`：SD3.5 LoRA SFT
- `vehicle_design_train/train_sd3_grpo.py`：SD3 Flow-GRPO 后训练
- `vehicle_design_train/grpo/sd3_rollout.py`：SD3 rollout 逻辑
- `vehicle_design_train/grpo/sd3_flow_grpo_loss.py`：Flow-GRPO loss
- `scripts/launch_train_sd3.sh`：SFT 启动脚本
- `scripts/launch_train_sd3_grpo.sh`：GRPO 启动脚本

---

## 6) 许可证

内嵌训练脚本包含基于 Hugging Face Diffusers（Apache 2.0）改造的实现；请遵循各依赖与基座模型许可证。
