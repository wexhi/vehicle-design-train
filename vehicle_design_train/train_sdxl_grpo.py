#!/usr/bin/env python
"""SDXL LoRA post-training with GRPO-style group advantages + DDPO loss (DDIM log-probs)."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import shutil
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from PIL import Image, ImageDraw, ImageFont
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm

from vehicle_design_train.grpo.group_advantage import compute_group_advantages
from vehicle_design_train.grpo.sdxl_ddpo_loss import sdxl_ddpo_calculate_loss
from vehicle_design_train.grpo.sdxl_rollout import ensure_ddim_scheduler, sdxl_ddim_rollout_parallel
from vehicle_design_train.grpo_dataset import list_grpo_jsonl
from vehicle_design_train.rewards.composite import combine_rewards
from vehicle_design_train.rewards.imagereward_scorer import ImageRewardScorer
from vehicle_design_train.rewards.geneval_remote_scorer import GenevalRemoteScorer
from vehicle_design_train.rewards.pickscore_scorer import FlowGrpoPickScoreReward
from vehicle_design_train.rewards.vqa_prob_scorer import (
    DEFAULT_GLOBAL_TEMPLATE_EN,
    DashScopeVqaProbScorer,
)
from vehicle_design_train.rewards.vqa_vllm_scorer import (
    VllmOpenAiStructuredVqaScorer,
    VllmOpenAiVqaProbScorer,
)

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)


def _load_dotenv_files() -> None:
    """Populate os.environ from `.env` (DashScope reads DASHSCOPE_API_KEY from the environment only)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    repo_root = Path(__file__).resolve().parent.parent
    env_repo = repo_root / ".env"
    if env_repo.is_file():
        load_dotenv(env_repo)
    load_dotenv()


def _parse_comma_int_ids(s: str | None) -> list[int] | None:
    if not s or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def _apply_split_infer_gpu_visibility(args: argparse.Namespace) -> None:
    """
    For --split_infer_train: assign one physical GPU per rank via CUDA_VISIBLE_DEVICES
    before Accelerator() / CUDA init. Rollout ranks use rollout_gpu_ids[i], train ranks use train_gpu_ids[j].

    Requires len(rollout_gpu_ids) == num_inference_processes and len(train_gpu_ids) == num_train_processes
    (today num_infer must equal num_train). Set via --rollout_gpu_ids / --train_gpu_ids or env
    GRPO_ROLLOUT_GPU_IDS / GRPO_TRAIN_GPU_IDS.
    """
    if not args.split_infer_train:
        return
    roll_src = (args.rollout_gpu_ids or os.environ.get("GRPO_ROLLOUT_GPU_IDS") or "").strip()
    train_src = (args.train_gpu_ids or os.environ.get("GRPO_TRAIN_GPU_IDS") or "").strip()
    if not roll_src or not train_src:
        return

    rollout_ids = _parse_comma_int_ids(roll_src)
    train_ids = _parse_comma_int_ids(train_src)
    if not rollout_ids or not train_ids:
        return

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        return

    num_infer = args.num_inference_processes if args.num_inference_processes is not None else world // 2
    num_train = world - num_infer
    if num_infer <= 0 or num_train <= 0:
        raise ValueError(f"Invalid WORLD_SIZE={world} vs num_inference_processes={args.num_inference_processes}")
    if len(rollout_ids) != num_infer:
        raise ValueError(
            f"rollout_gpu_ids: expected {num_infer} entries (num_inference_processes), got {len(rollout_ids)}: {rollout_ids}"
        )
    if len(train_ids) != num_train:
        raise ValueError(
            f"train_gpu_ids: expected {num_train} entries (WORLD_SIZE - num_infer), got {len(train_ids)}: {train_ids}"
        )

    rank = int(os.environ.get("RANK", "0"))
    if rank < num_infer:
        phys = rollout_ids[rank]
    else:
        phys = train_ids[rank - num_infer]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(phys)
    logger.info("split_infer_train: rank %s -> CUDA_VISIBLE_DEVICES=%s", rank, phys)

_MAX_TENSOR_NDIM = 6
_DTYPE_TO_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
}
_CODE_TO_DTYPE = {v: k for k, v in _DTYPE_TO_CODE.items()}
_ROLLOUT_TENSOR_KEYS = [
    "latents_b",
    "next_latents_b",
    "logp_b",
    "time_b",
    "advantages",
    "prompt_embeds",
    "negative_prompt_embeds",
    "pooled_prompt_embeds",
    "negative_pooled_prompt_embeds",
    "add_time_ids",
    "reward_metrics",
    "rollout_r_total",
    "rollout_r_ir",
    "rollout_r_vqa",
]
_ROLLOUT_METRIC_NAMES = [
    "reward/mean",
    "reward/std",
    "reward/min",
    "reward/max",
    "reward/zero_std",
    "reward/r_ir_mean",
    "reward/r_ir_std",
    "reward/r_ir_min",
    "reward/r_ir_max",
    "reward/r_vqa_mean",
    "reward/r_vqa_std",
    "reward/r_vqa_min",
    "reward/r_vqa_max",
    "adv/mean",
    "adv/std",
    "adv/min",
    "adv/max",
    "adv/abs_mean",
    "sample/group_size",
    "sample/sample_num_steps",
    "sample/num_train_steps",
    "timing/rollout_sec",
]
_TRAIN_METRIC_NAMES = [
    "train/loss",
    "train/approx_kl",
    "train/clipfrac",
    "train/updates_per_sample",
    "train/lr",
    "timing/train_sec",
]


@dataclass
class SplitRoleConfig:
    enabled: bool
    role: str
    pair_idx: int
    pair_rank: int | None
    num_inference_processes: int
    num_train_processes: int
    train_group: dist.ProcessGroup | None
    is_train_main: bool

    @property
    def is_infer_rank(self) -> bool:
        return self.enabled and self.role == "infer"

    @property
    def is_train_rank(self) -> bool:
        return self.enabled and self.role == "train"

    @property
    def num_pairs(self) -> int:
        return self.num_inference_processes if self.enabled else 0


def parse_args():
    p = argparse.ArgumentParser(description="SDXL GRPO-style RL fine-tuning (DDPO + group advantages)")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument(
        "--grpo_jsonl",
        type=str,
        required=True,
        help="v3 JSONL: prompt_en, judge_requirements; optional geneval=... for --vqa_backend=geneval_remote",
    )
    p.add_argument("--prompt_field", type=str, default="prompt_en")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None, help="Optional existing LoRA weights directory or .safetensors")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--merge_lora_into_unet", action="store_true", help="Fuse LoRA into UNet then attach new trainable LoRA")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--group_size", type=int, default=4, help="G rollouts per prompt (GRPO group)")
    p.add_argument(
        "--rollout_batch_size",
        type=int,
        default=1,
        help="Parallel images per UNet forward during rollout (1 = sequential). Capped by group_size.",
    )
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_samples_per_epoch", type=int, default=None)
    p.add_argument("--sample_num_steps", type=int, default=20)
    p.add_argument("--sample_eta", type=float, default=1.0)
    p.add_argument("--sample_guidance_scale", type=float, default=7.0)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--train_learning_rate", type=float, default=1e-5)
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Micro-batch size for DDPO loss / optimizer.step (independent from rollout_batch_size)",
    )
    p.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Accumulate this many micro-batch backward passes before each optimizer.step (standard mode uses Accelerate; split DDP uses no_sync + scaled loss).",
    )
    p.add_argument("--train_cfg", action="store_true", default=True)
    p.add_argument("--no_train_cfg", action="store_false", dest="train_cfg")
    p.add_argument("--train_clip_range", type=float, default=1e-4)
    p.add_argument("--train_adv_clip_max", type=float, default=5.0)
    p.add_argument("--train_timestep_fraction", type=float, default=0.25, help="Fraction of denoise steps used in policy update")
    p.add_argument(
        "--train_timestep_sample_mode",
        type=str,
        default="group_shared_stratified",
        choices=["independent", "group_shared_uniform", "group_shared_stratified"],
        help="How to pick DDPO timestep indices: independent=per-rollout randperm (legacy); "
        "group_shared_*=same K indices for the whole GRPO group (lower variance). "
        "Stratified splits trajectory index ~high/mid/low noise then samples within each bucket.",
    )
    p.add_argument(
        "--train_timestep_unbiased_scale",
        action="store_true",
        help="If set with --train_timestep_sample_mode=group_shared_uniform and K<T, scale DDPO loss/approx_kl by T/K "
        "(Monte Carlo correction toward full-trajectory sum; not valid for stratified).",
    )
    p.add_argument("--train_num_inner_epochs", type=int, default=1)
    p.add_argument("--weight_ir", type=float, default=0.5)
    p.add_argument(
        "--weight_vqa",
        type=float,
        default=0.5,
        help="Weight for second reward: LLM VQA log-probs, or PickScore if --vqa_backend=pickscore (Flow-GRPO). "
        "Use 0 to skip loading/calling any VQA backend (no DashScope, vLLM, PickScore, or GenEval).",
    )
    p.add_argument("--vqa_global_weight", type=float, default=1.0)
    p.add_argument("--vqa_judge_weight", type=float, default=1.0)
    p.add_argument(
        "--vqa_backend",
        type=str,
        default="dashscope",
        choices=["dashscope", "vllm_openai", "vllm_openai_structured", "pickscore", "geneval_remote"],
        help="vllm_openai=logprob; vllm_openai_structured=<Thought>+<Answer>yes|no</Answer>; pickscore; geneval.",
    )
    p.add_argument(
        "--vqa_model",
        type=str,
        default="qwen3.5-35b-a3b",
        help="DashScope model id or HF id for vLLM (e.g. Qwen/Qwen3.5-9B).",
    )
    p.add_argument(
        "--vqa_openai_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL; default env OPENAI_BASE_URL or http://127.0.0.1:8000/v1",
    )
    p.add_argument(
        "--vqa_openai_api_key",
        type=str,
        default=None,
        help="API key for vLLM server; default env OPENAI_API_KEY or EMPTY",
    )
    p.add_argument(
        "--vqa_enable_thinking",
        action="store_true",
        help="vllm_openai (logprob): enable thinking (default off).",
    )
    p.add_argument("--vqa_structured_max_tokens", type=int, default=4096)
    p.add_argument("--vqa_structured_temperature", type=float, default=0.0)
    p.add_argument(
        "--vqa_structured_disable_thinking",
        action="store_true",
        help="vllm_openai_structured: disable enable_thinking on vLLM.",
    )
    p.add_argument("--vqa_structured_timeout_sec", type=float, default=300.0)
    p.add_argument(
        "--vqa_max_workers",
        type=int,
        default=8,
        help="Concurrent VQA HTTP calls per rollout group (ignored for pickscore).",
    )
    p.add_argument(
        "--pickscore_processor_id",
        type=str,
        default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    )
    p.add_argument("--pickscore_model_id", type=str, default="yuvalkirstain/PickScore_v1")
    p.add_argument(
        "--pickscore_max_batch_size",
        type=int,
        default=None,
        help="PickScore GPU micro-batch; None = whole group at once.",
    )
    p.add_argument("--geneval_server_url", type=str, default="http://127.0.0.1:18085")
    p.add_argument("--geneval_only_strict", action="store_true")
    p.add_argument(
        "--geneval_reward_field",
        type=str,
        default="score",
        choices=["score", "accuracy", "strict_accuracy"],
    )
    p.add_argument("--geneval_timeout_sec", type=float, default=120.0)
    p.add_argument("--geneval_max_batch_size", type=int, default=64)
    p.add_argument("--geneval_max_retries", type=int, default=3)
    p.add_argument("--global_question_template_en", type=str, default=None)
    p.add_argument("--skip_vqa", action="store_true")
    p.add_argument("--skip_imagereward", action="store_true")
    p.add_argument(
        "--imagereward_max_batch_size",
        type=int,
        default=None,
        help="ImageReward GPU micro-batch (same prompt). None = one batch for the whole group; set e.g. 2 if OOM.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--split_infer_train", action="store_true", help="Use half the processes for rollout/reward and half for DDP training")
    p.add_argument("--num_inference_processes", type=int, default=None, help="Number of rollout/reward processes when --split_infer_train is enabled; must equal the number of training processes")
    p.add_argument(
        "--rollout_gpu_ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids for rollout ranks (len == num_inference_processes). Env: GRPO_ROLLOUT_GPU_IDS.",
    )
    p.add_argument(
        "--train_gpu_ids",
        type=str,
        default=None,
        help="Comma-separated physical GPU ids for training ranks (len == num training procs). Env: GRPO_TRAIN_GPU_IDS.",
    )
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument(
        "--log_image_steps",
        type=int,
        default=0,
        help="If > 0, every N global optimizer steps log per-image r_total/r_ir/r_vqa/advantage: "
        "JSON under output_dir/rollout_samples/step_XXXXXXXX/ and TensorBoard (rollout/sample_table + rollout_log/img_XX/*). "
        "TB root: output_dir/logs_grpo. For TB image grids, add --save_rollout_sample_images.",
    )
    p.add_argument(
        "--save_rollout_sample_images",
        action="store_true",
        help="Requires --log_image_steps>0. Saves PNGs under rollout_samples/ and TB rollout/images + rollout/img_XX_with_metrics (cap: --max_logged_images).",
    )
    p.add_argument(
        "--max_logged_images",
        type=int,
        default=4,
        help="Max images per logged step on disk and in TensorBoard when --save_rollout_sample_images is set (typically ≤ group_size).",
    )
    p.add_argument(
        "--checkpointing_steps",
        type=int,
        default=0,
        help="If > 0, save checkpoint-{step}/ (unet_lora.safetensors + training_state.pt with optimizer + resume cursor) every N global optimizer steps. 0 = disabled.",
    )
    p.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help="Max number of checkpoint-* dirs to keep (oldest removed when saving). None = unlimited.",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help='Resume LoRA + optimizer + global_step + epoch/sample offset from this dir, or "latest" under --output_dir.',
    )
    return p.parse_args()


@dataclass
class ResumeState:
    """Training cursor + optimizer state loaded from checkpoint (optimizer dict applied after prepare)."""

    global_step: int
    epoch: int
    sample_idx_in_epoch: int
    optimizer_state: dict


def load_pipeline(args, device: str, torch_dtype: torch.dtype, lora_path_override: str | None = None):
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,
        use_safetensors=True,
        variant="fp16" if torch_dtype == torch.float16 else None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    if args.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()

    target_modules = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
    ]
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    lp = lora_path_override if lora_path_override is not None else args.lora_path
    if lp:
        if lp.endswith(".safetensors"):
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            set_peft_model_state_dict(pipe.unet, load_file(lp))
        elif os.path.isdir(lp):
            pipe.load_lora_weights(lp)
        else:
            d, fn = os.path.dirname(lp) or ".", os.path.basename(lp)
            pipe.load_lora_weights(d, weight_name=fn)

    if args.merge_lora_into_unet:
        pipe.unet.merge_and_unload()
        pipe.unet = get_peft_model(pipe.unet, lora_config)

    return pipe


def stack_rollouts(results: list) -> dict:
    """Stack list of SdxlRolloutResult (same prompt) -> tensors (G, ...)."""
    latents = torch.stack([r.latents_traj for r in results], dim=0)
    logp = torch.stack([r.log_probs for r in results], dim=0)
    ts = results[0].timesteps
    timesteps = ts.unsqueeze(0).expand(len(results), -1)
    return {
        "latents": latents[:, :-1],
        "next_latents": latents[:, 1:],
        "log_probs": logp,
        "timesteps": timesteps,
        "prompt_embeds": results[0].prompt_embeds,
        "negative_prompt_embeds": results[0].negative_prompt_embeds,
        "pooled_prompt_embeds": results[0].pooled_prompt_embeds,
        "negative_pooled_prompt_embeds": results[0].negative_pooled_prompt_embeds,
        "add_time_ids": results[0].add_time_ids,
        "images": [img for r in results for img in r.images_pil],
    }


def tensor_stats_values(x: torch.Tensor) -> tuple[float, float, float, float]:
    x = x.float()
    return (
        x.mean().item(),
        x.std(unbiased=False).item(),
        x.min().item(),
        x.max().item(),
    )


def make_rollout_metrics_tensor(
    r_total: torch.Tensor,
    r_ir_t: torch.Tensor,
    r_vqa_t: torch.Tensor,
    advantages: torch.Tensor,
    args,
    rollout_sec: float,
    num_train_steps: int,
) -> torch.Tensor:
    reward_mean, reward_std, reward_min, reward_max = tensor_stats_values(r_total)
    ir_mean, ir_std, ir_min, ir_max = tensor_stats_values(r_ir_t)
    vqa_mean, vqa_std, vqa_min, vqa_max = tensor_stats_values(r_vqa_t)
    adv_mean, adv_std, adv_min, adv_max = tensor_stats_values(advantages)
    return torch.tensor(
        [
            reward_mean,
            reward_std,
            reward_min,
            reward_max,
            float(reward_std <= 1e-8),
            ir_mean,
            ir_std,
            ir_min,
            ir_max,
            vqa_mean,
            vqa_std,
            vqa_min,
            vqa_max,
            adv_mean,
            adv_std,
            adv_min,
            adv_max,
            advantages.abs().mean().item(),
            float(args.group_size),
            float(args.sample_num_steps),
            float(num_train_steps),
            float(rollout_sec),
        ],
        device=r_total.device,
        dtype=torch.float32,
    )


def make_train_metrics_tensor(
    train_metrics: torch.Tensor,
    updates: int,
    optimizer,
    train_sec: float,
) -> torch.Tensor:
    lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    return torch.tensor(
        [
            train_metrics[0].item(),
            train_metrics[1].item(),
            train_metrics[2].item(),
            float(updates),
            lr,
            float(train_sec),
        ],
        device=train_metrics.device,
        dtype=torch.float32,
    )


def log_tensorboard_scalars(tb_writer: SummaryWriter | None, names: list[str], values: torch.Tensor, global_step: int) -> None:
    if tb_writer is None:
        return
    vals = values.detach().cpu().tolist()
    for name, value in zip(names, vals):
        tb_writer.add_scalar(name, value, global_step)


def _log_grpo_step_console(
    global_step: int,
    rollout_metrics: torch.Tensor | None,
    train_metrics: torch.Tensor | None,
) -> None:
    """Print a one-line summary to the terminal (same ranks / cadence as TensorBoard)."""
    parts: list[str] = [f"step={global_step}"]
    if train_metrics is not None:
        tm = train_metrics.detach().float().cpu()
        parts.append(f"loss={tm[0].item():.6f}")
        parts.append(f"approx_kl={tm[1].item():.6f}")
        parts.append(f"clipfrac={tm[2].item():.4f}")
        parts.append(f"lr={tm[4].item():.2e}")
    if rollout_metrics is not None:
        rm = rollout_metrics.detach().float().cpu()
        parts.append(f"reward_mean={rm[0].item():.4f}")
        parts.append(f"reward_std={rm[1].item():.4f}")
        parts.append(f"r_ir_mean={rm[5].item():.4f}")
        parts.append(f"r_vqa_mean={rm[9].item():.4f}")
        parts.append(f"adv_mean={rm[13].item():.4f}")
        parts.append(f"rollout_sec={rm[21].item():.2f}")
    logger.info("[GRPO] %s", " | ".join(parts))


def maybe_log_step_scalars(
    tb_writer: SummaryWriter | None,
    global_step: int,
    logging_steps: int,
    rollout_metrics: torch.Tensor | None = None,
    train_metrics: torch.Tensor | None = None,
) -> None:
    if global_step % max(logging_steps, 1) != 0:
        return
    if tb_writer is None:
        return
    if rollout_metrics is not None:
        log_tensorboard_scalars(tb_writer, _ROLLOUT_METRIC_NAMES, rollout_metrics, global_step)
    if train_metrics is not None:
        log_tensorboard_scalars(tb_writer, _TRAIN_METRIC_NAMES, train_metrics, global_step)
    _log_grpo_step_console(global_step, rollout_metrics, train_metrics)


def log_run_config(tb_writer: SummaryWriter | None, args, accelerator: Accelerator, split_cfg: SplitRoleConfig) -> None:
    if tb_writer is None:
        return
    config_scalars = {
        "config/group_size": float(args.group_size),
        "config/sample_num_steps": float(args.sample_num_steps),
        "config/sample_eta": float(args.sample_eta),
        "config/sample_guidance_scale": float(args.sample_guidance_scale),
        "config/rollout_batch_size": float(args.rollout_batch_size),
        "config/train_batch_size": float(args.train_batch_size),
        "config/gradient_accumulation_steps": float(max(1, int(args.gradient_accumulation_steps))),
        "config/train_timestep_fraction": float(args.train_timestep_fraction),
        "config/train_timestep_unbiased_scale": 1.0 if args.train_timestep_unbiased_scale else 0.0,
        "config/train_num_inner_epochs": float(args.train_num_inner_epochs),
        "config/train_learning_rate": float(args.train_learning_rate),
        "config/train_clip_range": float(args.train_clip_range),
        "config/train_adv_clip_max": float(args.train_adv_clip_max),
        "config/log_image_steps": float(args.log_image_steps),
        "config/save_rollout_sample_images": 1.0 if args.save_rollout_sample_images else 0.0,
        "config/max_logged_images": float(args.max_logged_images),
        "config/weight_ir": float(args.weight_ir),
        "config/weight_vqa": float(args.weight_vqa),
        "config/vqa_max_workers": float(args.vqa_max_workers),
        "config/pickscore_backend": 1.0 if args.vqa_backend == "pickscore" else 0.0,
        "config/geneval_remote_backend": 1.0 if args.vqa_backend == "geneval_remote" else 0.0,
        "config/vllm_structured_backend": 1.0 if args.vqa_backend == "vllm_openai_structured" else 0.0,
        "system/world_size": float(accelerator.num_processes),
        "system/split_infer_train": 1.0 if split_cfg.enabled else 0.0,
        "system/num_inference_processes": float(split_cfg.num_inference_processes if split_cfg.enabled else 0),
        "system/num_train_processes": float(split_cfg.num_train_processes if split_cfg.enabled else accelerator.num_processes),
    }
    for name, value in config_scalars.items():
        tb_writer.add_scalar(name, value, 0)
    tb_writer.add_text("config/vqa_backend", args.vqa_backend, 0)
    tb_writer.add_text("config/vqa_model", args.vqa_model, 0)
    if args.vqa_backend == "pickscore":
        tb_writer.add_text("config/pickscore_processor_id", args.pickscore_processor_id, 0)
        tb_writer.add_text("config/pickscore_model_id", args.pickscore_model_id, 0)
    if args.vqa_backend == "geneval_remote":
        tb_writer.add_text("config/geneval_server_url", args.geneval_server_url, 0)
        tb_writer.add_text("config/geneval_reward_field", args.geneval_reward_field, 0)
    tb_writer.add_text(
        "config/train_gpu_ids",
        (args.train_gpu_ids or os.environ.get("GRPO_TRAIN_GPU_IDS") or "").strip(),
        0,
    )
    tb_writer.add_text(
        "config/rollout_gpu_ids",
        (args.rollout_gpu_ids or os.environ.get("GRPO_ROLLOUT_GPU_IDS") or "").strip(),
        0,
    )
    tb_writer.add_text("config/train_timestep_sample_mode", args.train_timestep_sample_mode, 0)
    tb_writer.add_scalar("config/checkpointing_steps", float(args.checkpointing_steps), 0)
    if args.checkpoints_total_limit is not None:
        tb_writer.add_scalar("config/checkpoints_total_limit", float(args.checkpoints_total_limit), 0)
    tb_writer.add_text("config/resume_from_checkpoint", args.resume_from_checkpoint or "", 0)


def setup_split_roles(args, accelerator: Accelerator) -> SplitRoleConfig:
    if not args.split_infer_train:
        return SplitRoleConfig(
            enabled=False,
            role="all",
            pair_idx=0,
            pair_rank=None,
            num_inference_processes=0,
            num_train_processes=0,
            train_group=None,
            is_train_main=accelerator.is_main_process,
        )

    if accelerator.num_processes < 2 or not dist.is_initialized():
        raise RuntimeError("--split_infer_train requires accelerate multi-process launch")

    num_infer = args.num_inference_processes or (accelerator.num_processes // 2)
    num_train = accelerator.num_processes - num_infer
    if num_infer <= 0 or num_train <= 0:
        raise ValueError("split inference/train requires both groups to be non-empty")
    if num_infer != num_train:
        raise ValueError("current split mode requires num_inference_processes == num_training_processes")

    train_ranks = list(range(num_infer, accelerator.num_processes))
    train_group = dist.new_group(ranks=train_ranks)
    rank = accelerator.process_index
    if rank < num_infer:
        return SplitRoleConfig(
            enabled=True,
            role="infer",
            pair_idx=rank,
            pair_rank=rank + num_infer,
            num_inference_processes=num_infer,
            num_train_processes=num_train,
            train_group=train_group,
            is_train_main=False,
        )
    return SplitRoleConfig(
        enabled=True,
        role="train",
        pair_idx=rank - num_infer,
        pair_rank=rank - num_infer,
        num_inference_processes=num_infer,
        num_train_processes=num_train,
        train_group=train_group,
        is_train_main=rank == num_infer,
    )


def optimizer_steps_per_sample(args, num_train_steps: int) -> int:
    """DDPO inner loop: count of optimizer.step() per sample (after gradient accumulation)."""
    flat_n = int(args.group_size) * int(num_train_steps)
    micro = int(args.train_num_inner_epochs) * math.ceil(flat_n / max(1, int(args.train_batch_size)))
    acc = max(1, int(args.gradient_accumulation_steps))
    return math.ceil(micro / acc)


def print_training_startup_config(
    args,
    accelerator: Accelerator,
    split_cfg: SplitRoleConfig,
    num_train_steps: int,
    num_samples: int,
    torch_dtype: torch.dtype,
    device: torch.device,
    resume_state: ResumeState | None = None,
    resume_ckpt_dir: str | None = None,
) -> None:
    if not accelerator.is_main_process:
        return
    steps_per_sample = optimizer_steps_per_sample(args, num_train_steps)
    nproc = max(1, int(accelerator.num_processes))
    std_rank0_samples = (num_samples + nproc - 1) // nproc if nproc else num_samples
    if split_cfg.enabled:
        opt_epoch_line = (
            f"  ~opt_steps/pair-rank/epoch:  len(pair_samples)×{steps_per_sample} "
            f"(每 epoch 因 shard 而异; infer/train 各一条进度条)"
        )
    else:
        opt_epoch_line = f"  ~opt_steps/rank0/epoch:    {std_rank0_samples * steps_per_sample}"
    eta_warn = ""
    if float(args.sample_eta) <= 0.0:
        eta_warn = (
            "  WARNING: sample_eta<=0 时 DDIM 退化为确定性步，当前 log_prob 目标对 UNet 无梯度；"
            "请使用 sample_eta>0（默认 1.0）。"
        )
    ts_mode_warn = ""
    if args.train_timestep_unbiased_scale and args.train_timestep_sample_mode != "group_shared_uniform":
        ts_mode_warn = (
            "  WARNING: train_timestep_unbiased_scale 仅对 group_shared_uniform + K<T 生效；"
            f"当前 mode={args.train_timestep_sample_mode}，将不乘 T/K。"
        )
    lines = [
        "=" * 72,
        "GRPO / DDPO 训练配置（启动摘要）",
        "=" * 72,
        f"  output_dir:              {args.output_dir}",
        f"  pretrained_model:        {args.pretrained_model_name_or_path}",
        f"  lora_path:               {args.lora_path or '(none)'}",
        f"  grpo_jsonl:              {args.grpo_jsonl}",
        f"  jsonl_rows (this run):   {num_samples}",
        f"  device / dtype:          {device} / {torch_dtype}",
        f"  accelerate_processes:   {nproc}",
        f"  split_infer_train:       {split_cfg.enabled}"
        + (
            f" (infer={split_cfg.num_inference_processes}, train={split_cfg.num_train_processes})"
            if split_cfg.enabled
            else ""
        ),
        f"  samples/rank (std, max): ~{std_rank0_samples}  (rank0 切片; 多卡时各 rank 相近)",
        "-" * 72,
        f"  resolution:              {args.resolution}",
        f"  group_size (G):          {args.group_size}",
        f"  rollout_batch_size:      {args.rollout_batch_size}",
        f"  sample_num_steps:        {args.sample_num_steps}  eta={args.sample_eta}  guidance={args.sample_guidance_scale}",
        *([eta_warn] if eta_warn else []),
        f"  train_timestep_fraction: {args.train_timestep_fraction}  -> num_train_steps={num_train_steps}",
        f"  train_timestep_sample_mode: {args.train_timestep_sample_mode}",
        f"  train_timestep_unbiased_scale: {args.train_timestep_unbiased_scale}  (T/K loss scale only if uniform+K<T; may need LR retune)",
        *([ts_mode_warn] if ts_mode_warn else []),
        f"  train_batch_size:        {args.train_batch_size}",
        f"  gradient_accumulation:   {max(1, int(args.gradient_accumulation_steps))}",
        f"  train_num_inner_epochs:  {args.train_num_inner_epochs}",
        f"  optimizer_steps/sample:  {steps_per_sample}  (= ceil(inner_epochs × ceil(G×Ttrain / train_batch_size) / grad_acc))",
        opt_epoch_line,
        f"  train_learning_rate:     {args.train_learning_rate}",
        f"  gradient_checkpointing:  {bool(args.gradient_checkpointing)}",
        f"  train_cfg:               {args.train_cfg}",
        f"  lora_rank:               {args.lora_rank}  merge_lora_into_unet={args.merge_lora_into_unet}",
        f"  max_samples_per_epoch:   {args.max_samples_per_epoch}",
        f"  imagereward_max_batch:   {args.imagereward_max_batch_size}",
        "-" * 72,
        f"  weight_ir / weight_vqa:  {args.weight_ir} / {args.weight_vqa}",
        f"  skip_imagereward:        {args.skip_imagereward}",
        f"  skip_vqa:                {args.skip_vqa}",
        f"  vqa_backend:             {args.vqa_backend}  model={args.vqa_model}"
        + (
            f"  pickscore={args.pickscore_processor_id} + {args.pickscore_model_id}"
            if args.vqa_backend == "pickscore"
            else f"  geneval={args.geneval_server_url} field={args.geneval_reward_field}"
            if args.vqa_backend == "geneval_remote"
            else (
                f"  structured_VQA max_tok={args.vqa_structured_max_tokens} thinking={not args.vqa_structured_disable_thinking}"
                if args.vqa_backend == "vllm_openai_structured"
                else ""
            )
        ),
        "-" * 72,
        f"  save_steps:              {args.save_steps}",
        f"  checkpointing_steps:     {args.checkpointing_steps}  (0=关闭; 每 N 个全局 optimizer step 存 checkpoint-{{step}}/)",
        f"  checkpoints_total_limit: {args.checkpoints_total_limit}",
        f"  resume_from_checkpoint:  {args.resume_from_checkpoint or '(none)'}",
        *(
            [
                f"  → resume 已解析:         {resume_ckpt_dir}  "
                f"(global_step={resume_state.global_step} epoch={resume_state.epoch} "
                f"sample_idx_in_epoch={resume_state.sample_idx_in_epoch})"
            ]
            if resume_state is not None and resume_ckpt_dir
            else []
        ),
        f"  logging_steps:           {args.logging_steps}",
        f"  log_image_steps:         {args.log_image_steps}  save_rollout_sample_images={args.save_rollout_sample_images}  max_logged_images={args.max_logged_images}",
        f"  num_epochs:              {args.num_epochs}",
        f"  seed:                    {args.seed}",
        f"  negative_prompt:         {(args.negative_prompt or '(empty)')[:120]}{'…' if len(args.negative_prompt or '') > 120 else ''}",
        f"  CUDA_VISIBLE_DEVICES:    {os.environ.get('CUDA_VISIBLE_DEVICES', '') or '(unset)'}",
        "=" * 72,
    ]
    for line in lines:
        logger.info(line)


def epoch_samples_for_rank(
    samples: list,
    epoch: int,
    args,
    num_shards: int,
    shard_idx: int,
    drop_last: bool,
) -> tuple[list, int]:
    ordered = list(samples)
    random.Random(args.seed + epoch).shuffle(ordered)
    dropped = 0
    if drop_last:
        dropped = len(ordered) % num_shards
        if dropped:
            ordered = ordered[:-dropped]
    return ordered[shard_idx::num_shards], dropped


def get_unet_module(unet):
    return unet.module if isinstance(unet, DDP) else unet


def _ddp_no_sync_if(unet, active: bool):
    """Skip gradient all-reduce on DDP backward when still accumulating."""
    if active and isinstance(unet, DDP):
        return unet.no_sync()
    return nullcontext()


def _ddpo_subsample_loss_scale(args, num_train_steps: int) -> float:
    """Monte Carlo scale T/K for uniform subsample (group_shared_uniform only); split-safe via args."""
    T = int(args.sample_num_steps)
    K = int(num_train_steps)
    if (
        args.train_timestep_unbiased_scale
        and args.train_timestep_sample_mode == "group_shared_uniform"
        and K < T
    ):
        return T / float(K)
    return 1.0


def _trainable_grad_l2_norm(unet: torch.nn.Module) -> tuple[float, int]:
    """Sum of squared grads over requires_grad parameters that received a grad this step."""
    unet_inner = get_unet_module(unet)
    total_sq = 0.0
    n = 0
    for p in unet_inner.parameters():
        if p.requires_grad and p.grad is not None:
            total_sq += float(p.grad.detach().float().pow(2).sum().item())
            n += 1
    return (total_sq**0.5, n)


def trainable_params(unet) -> list[torch.nn.Parameter]:
    return [p for p in get_unet_module(unet).parameters() if p.requires_grad]


def send_tensor(tensor: torch.Tensor, dst: int) -> None:
    tensor = tensor.detach().contiguous()
    if tensor.dtype not in _DTYPE_TO_CODE:
        raise TypeError(f"Unsupported dtype for distributed payload: {tensor.dtype}")
    if tensor.ndim > _MAX_TENSOR_NDIM:
        raise ValueError(f"Tensor ndim {tensor.ndim} exceeds max {_MAX_TENSOR_NDIM}")
    meta = torch.full((_MAX_TENSOR_NDIM + 2,), -1, device=tensor.device, dtype=torch.int64)
    meta[0] = _DTYPE_TO_CODE[tensor.dtype]
    meta[1] = tensor.ndim
    if tensor.ndim:
        meta[2 : 2 + tensor.ndim] = torch.tensor(list(tensor.shape), device=tensor.device, dtype=torch.int64)
    dist.send(meta, dst)
    dist.send(tensor, dst)


def recv_tensor(src: int, device: torch.device) -> torch.Tensor:
    meta = torch.empty((_MAX_TENSOR_NDIM + 2,), device=device, dtype=torch.int64)
    dist.recv(meta, src)
    dtype = _CODE_TO_DTYPE[int(meta[0].item())]
    ndim = int(meta[1].item())
    shape = tuple(int(v.item()) for v in meta[2 : 2 + ndim])
    out = torch.empty(shape, device=device, dtype=dtype)
    dist.recv(out, src)
    return out


def send_rollout_payload(payload: dict[str, torch.Tensor], dst: int) -> None:
    for key in _ROLLOUT_TENSOR_KEYS:
        send_tensor(payload[key], dst)


def recv_rollout_payload(src: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: recv_tensor(src, device) for key in _ROLLOUT_TENSOR_KEYS}


def sync_trainable_params(unet, peer_rank: int, send: bool) -> None:
    for param in trainable_params(unet):
        if send:
            dist.send(param.data.detach().contiguous(), peer_rank)
        else:
            recv_buf = torch.empty_like(param.data)
            dist.recv(recv_buf, peer_rank)
            param.data.copy_(recv_buf)


def _prompt_subseed(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)


def _stratified_timestep_indices(T: int, K: int, gen_cpu: torch.Generator, device: torch.device) -> torch.Tensor:
    """
    Sample K distinct indices from [0, T) with quotas across three trajectory segments.
    Index 0 is the first denoising transition (typically higher noise); T-1 is the last.
    """
    if K >= T:
        row = torch.randperm(T, generator=gen_cpu, device="cpu")[:K]
        return row.to(device=device, dtype=torch.long)

    t1 = max(1, T // 3)
    t2 = max(t1 + 1, (2 * T) // 3)
    if t2 >= T:
        t2 = T - 1
    segs = [
        list(range(0, t1)),
        list(range(t1, t2)),
        list(range(t2, T)),
    ]
    segs = [s for s in segs if len(s) > 0]
    if not segs:
        row = torch.randperm(T, generator=gen_cpu, device="cpu")[:K]
        return row.to(device=device, dtype=torch.long)

    nseg = len(segs)
    base = K // nseg
    rem = K - base * nseg
    quotas = [base + (1 if i < rem else 0) for i in range(nseg)]
    picked: list[int] = []
    for seg, q in zip(segs, quotas):
        if q <= 0:
            continue
        take = min(q, len(seg))
        perm = torch.randperm(len(seg), generator=gen_cpu, device="cpu")[:take]
        st = torch.tensor(seg, dtype=torch.long)
        picked.extend(st[perm].tolist())

    if len(picked) < K:
        used = set(picked)
        pool = [i for i in range(T) if i not in used]
        need = K - len(picked)
        if need > 0 and pool:
            pt = torch.tensor(pool, dtype=torch.long)
            perm = torch.randperm(len(pool), generator=gen_cpu, device="cpu")[:need]
            picked.extend(pt[perm].tolist())

    idx_cpu = torch.tensor(picked[:K], dtype=torch.long)
    shuf = torch.randperm(len(idx_cpu), generator=gen_cpu)
    return idx_cpu[shuf].to(device=device, dtype=torch.long)


def sample_ddpo_train_timestep_indices(
    *,
    T: int,
    K: int,
    G: int,
    mode: str,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Return indices [G, K] into rollout steps 0..T-1 (latents slice)."""
    K = min(max(K, 1), T)
    gen_cpu = torch.Generator(device="cpu")
    gen_cpu.manual_seed(int(seed) & 0x7FFF_FFFF_FFFF_FFFF)

    if K >= T:
        row = torch.randperm(T, generator=gen_cpu, device="cpu")[:K].to(device=device, dtype=torch.long)
        return row.unsqueeze(0).expand(G, -1).contiguous()

    if mode == "independent":
        out = torch.empty((G, K), device=device, dtype=torch.long)
        for i in range(G):
            gen_cpu.manual_seed((int(seed) + i + 1) & 0x7FFF_FFFF_FFFF_FFFF)
            out[i] = torch.randperm(T, generator=gen_cpu, device="cpu")[:K].to(device=device, dtype=torch.long)
        return out

    if mode == "group_shared_uniform":
        row = torch.randperm(T, generator=gen_cpu, device="cpu")[:K].to(device=device, dtype=torch.long)
        return row.unsqueeze(0).expand(G, -1).contiguous()

    if mode == "group_shared_stratified":
        row = _stratified_timestep_indices(T, K, gen_cpu, device)
        return row.unsqueeze(0).expand(G, -1).contiguous()

    raise ValueError(f"Unknown train_timestep_sample_mode: {mode}")


def build_rollout_training_batch(
    pipe,
    sample,
    args,
    device: torch.device,
    global_step: int,
    num_train_steps: int,
    ir_scorer,
    vqa_scorer,
) -> dict[str, torch.Tensor]:
    rollout_start = time.perf_counter()
    prompt = sample.prompt_en
    judges = sample.judge_questions

    results = []
    pipe.unet.eval()
    rb = max(1, min(int(args.rollout_batch_size), int(args.group_size)))
    g_cursor = 0
    while g_cursor < args.group_size:
        chunk = min(rb, args.group_size - g_cursor)
        gens = [
            torch.Generator(device=device).manual_seed(args.seed + global_step * 1000 + g_cursor + j)
            for j in range(chunk)
        ]
        with torch.inference_mode():
            chunk_results = sdxl_ddim_rollout_parallel(
                pipe,
                prompt=prompt,
                negative_prompt=args.negative_prompt or None,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.sample_num_steps,
                guidance_scale=args.sample_guidance_scale,
                eta=args.sample_eta,
                num_parallel=chunk,
                generators=gens,
                output_type="pil",
            )
        results.extend(chunk_results)
        g_cursor += chunk

    imgs = [r.images_pil[0] for r in results]
    prompts_g = [prompt] * args.group_size

    if args.skip_imagereward:
        r_ir_t = torch.zeros(args.group_size, device=device, dtype=torch.float32)
    else:
        r_ir_t = ir_scorer.score(imgs, prompts_g).to(device)

    if vqa_scorer is None:
        r_vqa_t = torch.zeros(args.group_size, device=device, dtype=torch.float32)
    else:
        vqa_vals, _vqa_details = vqa_scorer.score_rollout_group(
            imgs, prompt, judges, geneval_metadata=sample.geneval
        )
        r_vqa_t = torch.tensor(vqa_vals, device=device, dtype=torch.float32)

    r_total = combine_rewards(
        r_ir_t,
        r_vqa_t,
        args.weight_ir,
        args.weight_vqa,
    )
    advantages = compute_group_advantages(r_total.cpu()).to(device)

    stacked = stack_rollouts(results)
    G, T, _c, _h, _w = stacked["latents"].shape
    K = min(max(num_train_steps, 1), T)
    idx_seed = int(args.seed) + int(global_step) * 1_000_003 + _prompt_subseed(prompt)
    idx = sample_ddpo_train_timestep_indices(
        T=T,
        K=K,
        G=G,
        mode=args.train_timestep_sample_mode,
        device=device,
        seed=idx_seed,
    )
    rollout_sec = time.perf_counter() - rollout_start

    return {
        "latents_b": torch.stack([stacked["latents"][i, idx[i]] for i in range(G)], dim=0),
        "next_latents_b": torch.stack([stacked["next_latents"][i, idx[i]] for i in range(G)], dim=0),
        "logp_b": torch.stack([stacked["log_probs"][i, idx[i]] for i in range(G)], dim=0),
        "time_b": torch.stack([stacked["timesteps"][i, idx[i]] for i in range(G)], dim=0),
        "advantages": advantages,
        "prompt_embeds": stacked["prompt_embeds"],
        "negative_prompt_embeds": stacked["negative_prompt_embeds"],
        "pooled_prompt_embeds": stacked["pooled_prompt_embeds"],
        "negative_pooled_prompt_embeds": stacked["negative_pooled_prompt_embeds"],
        "add_time_ids": stacked["add_time_ids"],
        "reward_metrics": make_rollout_metrics_tensor(
            r_total=r_total,
            r_ir_t=r_ir_t,
            r_vqa_t=r_vqa_t,
            advantages=advantages,
            args=args,
            rollout_sec=rollout_sec,
            num_train_steps=num_train_steps,
        ),
        "rollout_r_total": r_total.detach().to(dtype=torch.float32),
        "rollout_r_ir": r_ir_t.detach().to(dtype=torch.float32),
        "rollout_r_vqa": r_vqa_t.detach().to(dtype=torch.float32),
        "images_pil": imgs,
        "prompt_text": prompt,
    }


def should_log_images(global_step: int, log_image_steps: int) -> bool:
    return log_image_steps > 0 and global_step > 0 and global_step % log_image_steps == 0


def _annotate_rollout_image_pil(
    image: Image.Image,
    *,
    idx: int,
    r_total: float,
    r_ir: float,
    r_vqa: float,
    advantage: float,
) -> Image.Image:
    """Draw metrics under the image so TensorBoard IMAGES tab shows values next to each picture."""
    rgb = image.convert("RGB")
    w, h = rgb.size
    line1 = (
        f"#{idx}  r_total={r_total:.4f}  r_ir={r_ir:.4f}  r_vqa={r_vqa:.4f}  advantage={advantage:.4f}"
    )
    font = None
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        try:
            font = ImageFont.truetype(path, max(13, min(20, w // 80)))
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    probe = Image.new("RGB", (w, 24))
    dr = ImageDraw.Draw(probe)
    bbox = dr.textbbox((0, 0), line1, font=font)
    text_h = bbox[3] - bbox[1] + 2
    bar_h = max(40, text_h + 14)
    out = Image.new("RGB", (w, h + bar_h), (28, 28, 30))
    out.paste(rgb, (0, 0))
    draw = ImageDraw.Draw(out)
    draw.text((8, h + 6), line1, fill=(255, 245, 220), font=font)
    return out


def sanitize_filename(text: str, limit: int = 80) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    safe = "".join(keep).strip("_")
    return (safe[:limit] or "prompt").rstrip("_")


def log_rollout_group_reward_advantage(
    tb_writer: SummaryWriter | None,
    output_dir: Path,
    global_step: int,
    prompt: str | None,
    r_total: torch.Tensor,
    r_ir: torch.Tensor,
    r_vqa: torch.Tensor,
    advantages: torch.Tensor,
    *,
    write_disk: bool,
) -> None:
    """Log full rollout group: per-image r_total, r_ir, r_vqa, advantage (JSON + TB text/scalars). No image tensors."""
    n = min(
        int(r_total.shape[0]),
        int(r_ir.shape[0]),
        int(r_vqa.shape[0]),
        int(advantages.shape[0]),
    )
    if n <= 0:
        return

    rt = r_total.detach().float().cpu()[:n]
    ri = r_ir.detach().float().cpu()[:n]
    rv = r_vqa.detach().float().cpu()[:n]
    adv = advantages.detach().float().cpu()[:n]

    rollout_dir = output_dir / "rollout_samples" / f"step_{global_step:08d}"
    resolved = (prompt or "").strip()
    if not resolved:
        ppt = rollout_dir / "prompt.txt"
        if ppt.is_file():
            resolved = ppt.read_text(encoding="utf-8").strip()
    if not resolved:
        resolved = "(unknown prompt)"

    rows: list[dict[str, float | int]] = [
        {
            "index": i,
            "r_total": float(rt[i].item()),
            "r_ir": float(ri[i].item()),
            "r_vqa": float(rv[i].item()),
            "advantage": float(adv[i].item()),
        }
        for i in range(n)
    ]

    if write_disk:
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "prompt.txt").write_text(resolved, encoding="utf-8")
        (rollout_dir / "per_image_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if tb_writer is not None:
        table_lines = [
            "### Prompt",
            "```",
            resolved,
            "```",
            "",
            "| # | r_total | r_ir | r_vqa | advantage |",
            "|---|--------:|-----:|------:|----------:|",
        ]
        for row in rows:
            table_lines.append(
                f"| {row['index']} | {row['r_total']:.4f} | {row['r_ir']:.4f} | {row['r_vqa']:.4f} | {row['advantage']:.4f} |"
            )
        tb_writer.add_text("rollout/sample_table", "\n".join(table_lines), global_step)

        for row in rows:
            i = int(row["index"])
            tag = f"rollout_log/img_{i:02d}"
            tb_writer.add_scalar(f"{tag}/r_total", row["r_total"], global_step)
            tb_writer.add_scalar(f"{tag}/r_ir", row["r_ir"], global_step)
            tb_writer.add_scalar(f"{tag}/r_vqa", row["r_vqa"], global_step)
            tb_writer.add_scalar(f"{tag}/advantage", row["advantage"], global_step)


def log_rollout_sample_image_files(
    tb_writer: SummaryWriter | None,
    output_dir: Path,
    global_step: int,
    images: list[Image.Image],
    prompt: str,
    max_logged_images: int,
    r_total: torch.Tensor,
    r_ir: torch.Tensor,
    r_vqa: torch.Tensor,
    advantages: torch.Tensor,
) -> None:
    """Optional PNGs on disk and TensorBoard image panels (subset capped by max_logged_images)."""
    if not images or max_logged_images <= 0:
        return

    g = len(images)
    n = min(g, max_logged_images, int(r_total.shape[0]), int(advantages.shape[0]))
    if n <= 0:
        return

    chosen = images[:n]
    rt = r_total.detach().float().cpu()[:n]
    ri = r_ir.detach().float().cpu()[:n]
    rv = r_vqa.detach().float().cpu()[:n]
    adv = advantages.detach().float().cpu()[:n]

    prompt_tag = sanitize_filename(prompt)
    rollout_dir = output_dir / "rollout_samples" / f"step_{global_step:08d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    tensors: list[torch.Tensor] = []
    for idx, image in enumerate(chosen):
        save_path = rollout_dir / f"{idx:02d}_{prompt_tag}.png"
        image.save(save_path)
        if tb_writer is not None:
            tensors.append(pil_to_tensor(image.convert("RGB")).float() / 255.0)

    if tb_writer is not None:
        for idx, image in enumerate(chosen):
            ann = _annotate_rollout_image_pil(
                image,
                idx=idx,
                r_total=float(rt[idx].item()),
                r_ir=float(ri[idx].item()),
                r_vqa=float(rv[idx].item()),
                advantage=float(adv[idx].item()),
            )
            t_ann = pil_to_tensor(ann.convert("RGB")).float() / 255.0
            tb_writer.add_image(f"rollout/img_{idx:02d}_with_metrics", t_ann, global_step)

        if tensors:
            tb_writer.add_images(
                "rollout/images",
                torch.stack(tensors, dim=0),
                global_step,
            )


def run_ddpo_update(
    pipe,
    optimizer,
    batch: dict[str, torch.Tensor],
    args,
    torch_dtype: torch.dtype,
    device: torch.device,
    backward_fn,
    grad_trace: dict | None = None,
    accelerator: Accelerator | None = None,
) -> tuple[int, torch.Tensor]:
    train_start = time.perf_counter()
    latents_b = batch["latents_b"]
    next_latents_b = batch["next_latents_b"]
    logp_b = batch["logp_b"]
    time_b = batch["time_b"]
    advantages = batch["advantages"]

    G, num_train_steps, c, h, w = latents_b.shape
    # Split infer/train: train ranks never run rollout, so scheduler is unset; align with infer side.
    ensure_ddim_scheduler(pipe)
    pipe.scheduler.set_timesteps(args.sample_num_steps, device=device)
    pipe.unet.train()
    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=torch_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    flat_n = int(G) * int(num_train_steps)
    tb = max(1, int(args.train_batch_size))
    micro_per_inner = math.ceil(flat_n / tb)
    total_micro_batches = int(args.train_num_inner_epochs) * micro_per_inner
    acc_steps = max(1, int(args.gradient_accumulation_steps))
    ddpo_scale = _ddpo_subsample_loss_scale(args, num_train_steps)

    metrics = torch.zeros(3, device=device, dtype=torch.float32)
    num_updates = 0
    micro_batch_count = 0
    micro_idx = 0

    if accelerator is None:
        optimizer.zero_grad(set_to_none=True)
    else:
        # Accelerate 的 grad-acc 计数器默认跨整个训练递增；我们按「每个 GRPO 样本」跑完所有 micro-batch，
        # 必须每样本重置，否则窗口跨样本且 remainder 步可能永不触发 sync → optimizer 被跳过、global_step 与 pbar 卡住。
        accelerator.step = 0

    for _inner in range(args.train_num_inner_epochs):
        flat_lat = latents_b.reshape(-1, c, h, w)
        flat_next = next_latents_b.reshape(-1, c, h, w)
        flat_logp = logp_b.reshape(-1)
        flat_time = time_b.reshape(-1)
        flat_adv = advantages.unsqueeze(1).expand(-1, num_train_steps).reshape(-1)

        n = flat_lat.shape[0]
        for start in range(0, n, args.train_batch_size):
            micro_idx += 1
            micro_batch_count += 1
            end = min(start + args.train_batch_size, n)
            accum_cm = accelerator.accumulate(pipe.unet) if accelerator is not None else nullcontext()
            with accum_cm:
                loss, approx_kl, clipfrac = sdxl_ddpo_calculate_loss(
                    pipe,
                    flat_lat[start:end],
                    flat_time[start:end],
                    flat_next[start:end],
                    flat_logp[start:end],
                    flat_adv[start:end],
                    batch["prompt_embeds"],
                    batch["negative_prompt_embeds"],
                    batch["pooled_prompt_embeds"],
                    batch["negative_pooled_prompt_embeds"],
                    batch["add_time_ids"],
                    args.sample_guidance_scale,
                    args.train_cfg,
                    args.train_clip_range,
                    args.train_adv_clip_max,
                    args.sample_eta,
                    autocast_cm,
                )
                loss = loss * ddpo_scale
                approx_kl = approx_kl * ddpo_scale
                if accelerator is None:
                    loss_scaled = loss / acc_steps
                    should_sync_step = (micro_idx % acc_steps == 0) or (micro_idx == total_micro_batches)
                    with _ddp_no_sync_if(pipe.unet, acc_steps > 1 and not should_sync_step):
                        loss_scaled.backward()
                    if grad_trace is not None and not grad_trace.get("logged_grad"):
                        gnorm, gcnt = _trainable_grad_l2_norm(pipe.unet)
                        grad_trace["logged_grad"] = True
                        if gcnt == 0:
                            logger.warning(
                                "首次 backward 后没有任何可训练参数的 grad（LoRA 可能未接入计算图，或 loss 与 UNet 已断开）。"
                                "loss=%.6e 若长期如此请检查 sample_eta>0、advantage 是否全 0。",
                                float(loss.detach().item()),
                            )
                        else:
                            logger.info(
                                "首次 optimizer 步: 含 grad 的 trainable 参数数=%d, grad L2 范数=%.6e, loss=%.6e",
                                gcnt,
                                gnorm,
                                float(loss.detach().item()),
                            )
                    if should_sync_step:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        num_updates += 1
                else:
                    backward_fn(loss)
                    if grad_trace is not None and not grad_trace.get("logged_grad"):
                        gnorm, gcnt = _trainable_grad_l2_norm(pipe.unet)
                        grad_trace["logged_grad"] = True
                        if gcnt == 0:
                            logger.warning(
                                "首次 backward 后没有任何可训练参数的 grad（LoRA 可能未接入计算图，或 loss 与 UNet 已断开）。"
                                "loss=%.6e 若长期如此请检查 sample_eta>0、advantage 是否全 0。",
                                float(loss.detach().item()),
                            )
                        else:
                            logger.info(
                                "首次 optimizer 步: 含 grad 的 trainable 参数数=%d, grad L2 范数=%.6e, loss=%.6e",
                                gcnt,
                                gnorm,
                                float(loss.detach().item()),
                            )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            if accelerator is not None and accelerator.sync_gradients:
                num_updates += 1

            metrics += torch.stack([loss.detach(), approx_kl.detach(), clipfrac.detach()]).to(torch.float32)

    if accelerator is not None and micro_idx > 0:
        rem = micro_idx % acc_steps
        if rem != 0:
            accelerator.sync_gradients = True
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            num_updates += 1

    train_metrics = make_train_metrics_tensor(
        train_metrics=metrics / max(micro_batch_count, 1),
        updates=num_updates,
        optimizer=optimizer,
        train_sec=time.perf_counter() - train_start,
    )
    return num_updates, train_metrics


def save_lora(path: Path, pipe, accelerator: Accelerator):
    path.mkdir(parents=True, exist_ok=True)
    raw = pipe.unet
    if isinstance(raw, DDP):
        unet = get_unet_module(raw)
    else:
        unet = get_unet_module(accelerator.unwrap_model(raw))
    state_dict = get_peft_model_state_dict(unet)
    from safetensors.torch import save_file

    save_file(state_dict, path / "unet_lora.safetensors")
    logger.info("Saved LoRA to %s", path)


def _parse_checkpoint_dir_step(name: str) -> int | None:
    m = re.match(r"^checkpoint-(\d+)$", name)
    return int(m.group(1)) if m else None


def resolve_resume_checkpoint_dir(output_dir: Path, resume_arg: str | None) -> Path | None:
    if not resume_arg:
        return None
    out = Path(output_dir)
    if resume_arg.strip().lower() == "latest":
        cdirs = []
        for p in out.glob("checkpoint-*"):
            if p.is_dir():
                st = _parse_checkpoint_dir_step(p.name)
                if st is not None:
                    cdirs.append((st, p))
        if not cdirs:
            raise FileNotFoundError(f'No checkpoint-* under "{out}" for resume_from_checkpoint=latest')
        return max(cdirs, key=lambda x: x[0])[1]
    p = Path(resume_arg)
    if not p.is_dir():
        raise FileNotFoundError(f"resume_from_checkpoint is not a directory: {p}")
    return p.resolve()


def load_resume_state_from_dir(ckpt_dir: Path) -> ResumeState:
    state_path = ckpt_dir / "training_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing {state_path} (not a full training checkpoint)")
    blob = torch.load(state_path, map_location="cpu", weights_only=False)
    if int(blob.get("version", 0)) < 1:
        raise ValueError(f"Unknown checkpoint version in {state_path}")
    return ResumeState(
        global_step=int(blob["global_step"]),
        epoch=int(blob["epoch"]),
        sample_idx_in_epoch=int(blob["sample_idx_in_epoch"]),
        optimizer_state=blob["optimizer"],
    )


def prune_old_checkpoints(output_dir: Path, limit: int | None) -> None:
    if limit is None or limit < 1:
        return
    cdirs = []
    for p in Path(output_dir).glob("checkpoint-*"):
        if p.is_dir():
            st = _parse_checkpoint_dir_step(p.name)
            if st is not None:
                cdirs.append((st, p))
    cdirs.sort(key=lambda x: x[0])
    while len(cdirs) > limit:
        _, old = cdirs.pop(0)
        shutil.rmtree(old, ignore_errors=True)
        logger.info("Removed old checkpoint %s (checkpoints_total_limit=%d)", old, limit)


def save_training_checkpoint(
    ckpt_dir: Path,
    pipe,
    optimizer,
    accelerator: Accelerator,
    *,
    global_step: int,
    epoch: int,
    next_sample_idx_in_epoch: int,
) -> None:
    """unet_lora.safetensors + training_state.pt (optimizer + resume cursor)."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_lora(ckpt_dir, pipe, accelerator)
    torch.save(
        {
            "version": 1,
            "global_step": int(global_step),
            "epoch": int(epoch),
            "sample_idx_in_epoch": int(next_sample_idx_in_epoch),
            "optimizer": optimizer.state_dict(),
        },
        ckpt_dir / "training_state.pt",
    )
    logger.info("Saved training checkpoint %s (global_step=%d)", ckpt_dir, global_step)


def run_standard_training(
    args,
    accelerator: Accelerator,
    pipe,
    samples: list,
    tb_writer,
    ir_scorer,
    vqa_scorer,
    num_train_steps: int,
    torch_dtype: torch.dtype,
    resume: ResumeState | None = None,
):
    device = accelerator.device
    out_dir = Path(args.output_dir)
    if accelerator.num_processes > 1:
        samples = samples[accelerator.process_index :: accelerator.num_processes]

    trainable = trainable_params(pipe.unet)
    optimizer = torch.optim.AdamW(trainable, lr=args.train_learning_rate)
    pipe.unet, optimizer = accelerator.prepare(pipe.unet, optimizer)
    if resume is not None:
        optimizer.load_state_dict(resume.optimizer_state)
        logger.info(
            "Resumed optimizer state (global_step=%d epoch=%d sample_idx=%d)",
            resume.global_step,
            resume.epoch,
            resume.sample_idx_in_epoch,
        )

    steps_per_sample = optimizer_steps_per_sample(args, num_train_steps)
    grad_trace: dict = {}
    global_step = resume.global_step if resume else 0
    start_epoch = resume.epoch if resume else 0
    start_sample_idx = resume.sample_idx_in_epoch if resume else 0
    for epoch in range(start_epoch, args.num_epochs):
        epoch_samples, _dropped = epoch_samples_for_rank(
            samples,
            epoch=epoch,
            args=args,
            num_shards=1,
            shard_idx=0,
            drop_last=False,
        )
        skip_head = start_sample_idx if epoch == start_epoch else 0
        epoch_tail = epoch_samples[skip_head:]
        epoch_opt_total = len(epoch_tail) * steps_per_sample
        pbar = tqdm(
            total=epoch_opt_total,
            unit="opt_step",
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch}",
        )
        samples_seen = skip_head
        for rel_i, sample in enumerate(epoch_tail):
            abs_sample_idx = skip_head + rel_i
            batch = build_rollout_training_batch(
                pipe=pipe,
                sample=sample,
                args=args,
                device=device,
                global_step=global_step,
                num_train_steps=num_train_steps,
                ir_scorer=ir_scorer,
                vqa_scorer=vqa_scorer,
            )
            reward_metrics = batch.pop("reward_metrics")
            images_pil = batch.pop("images_pil")
            prompt_text = batch.pop("prompt_text")
            roll_r_total = batch.pop("rollout_r_total")
            roll_r_ir = batch.pop("rollout_r_ir")
            roll_r_vqa = batch.pop("rollout_r_vqa")

            updates, train_metrics = run_ddpo_update(
                pipe=pipe,
                optimizer=optimizer,
                batch=batch,
                args=args,
                torch_dtype=torch_dtype,
                device=device,
                backward_fn=accelerator.backward,
                grad_trace=grad_trace,
                accelerator=accelerator,
            )
            global_step += updates
            samples_seen += 1
            pbar.update(updates)
            loss_v = float(train_metrics[0].detach().item())
            rmean_v = float(reward_metrics[0].detach().item())
            pbar.set_postfix(
                gstep=global_step,
                samples=samples_seen,
                loss=f"{loss_v:.4f}",
                rwd=f"{rmean_v:.4f}",
                refresh=True,
            )

            if updates > 0:
                maybe_log_step_scalars(
                    tb_writer=tb_writer,
                    global_step=global_step,
                    logging_steps=args.logging_steps,
                    rollout_metrics=reward_metrics,
                    train_metrics=train_metrics,
                )
            if updates > 0 and should_log_images(global_step, args.log_image_steps):
                if accelerator.is_main_process:
                    log_rollout_group_reward_advantage(
                        tb_writer=tb_writer,
                        output_dir=out_dir,
                        global_step=global_step,
                        prompt=prompt_text,
                        r_total=roll_r_total,
                        r_ir=roll_r_ir,
                        r_vqa=roll_r_vqa,
                        advantages=batch["advantages"],
                        write_disk=True,
                    )
                if accelerator.is_main_process and args.save_rollout_sample_images:
                    log_rollout_sample_image_files(
                        tb_writer=tb_writer,
                        output_dir=out_dir,
                        global_step=global_step,
                        images=images_pil,
                        prompt=prompt_text,
                        max_logged_images=args.max_logged_images,
                        r_total=roll_r_total,
                        r_ir=roll_r_ir,
                        r_vqa=roll_r_vqa,
                        advantages=batch["advantages"],
                    )

            if (
                updates > 0
                and accelerator.is_main_process
                and global_step > 0
                and global_step % args.save_steps == 0
            ):
                save_lora(Path(args.output_dir) / f"lora_step_{global_step}", pipe, accelerator)

            ck_every = int(args.checkpointing_steps)
            if (
                ck_every > 0
                and updates > 0
                and accelerator.is_main_process
                and global_step > 0
                and global_step % ck_every == 0
            ):
                prune_old_checkpoints(Path(args.output_dir), args.checkpoints_total_limit)
                ckpt_path = Path(args.output_dir) / f"checkpoint-{global_step}"
                ne, ns = (
                    (epoch + 1, 0)
                    if abs_sample_idx + 1 >= len(epoch_samples)
                    else (epoch, abs_sample_idx + 1)
                )
                save_training_checkpoint(
                    ckpt_path,
                    pipe,
                    optimizer,
                    accelerator,
                    global_step=global_step,
                    epoch=ne,
                    next_sample_idx_in_epoch=ns,
                )
        pbar.close()


def run_split_training(
    args,
    accelerator: Accelerator,
    pipe,
    samples: list,
    tb_writer,
    ir_scorer,
    vqa_scorer,
    num_train_steps: int,
    torch_dtype: torch.dtype,
    split_cfg: SplitRoleConfig,
    resume: ResumeState | None = None,
):
    device = accelerator.device
    out_dir = Path(args.output_dir)
    updates_per_sample = optimizer_steps_per_sample(args, num_train_steps)

    optimizer = None
    if split_cfg.is_train_rank:
        ddp_dev = [torch.cuda.current_device()] if device.type == "cuda" else None
        pipe.unet = DDP(
            pipe.unet,
            device_ids=ddp_dev,
            process_group=split_cfg.train_group,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.AdamW(trainable_params(pipe.unet), lr=args.train_learning_rate)
        if resume is not None:
            optimizer.load_state_dict(resume.optimizer_state)
            if split_cfg.is_train_main:
                logger.info(
                    "Resumed split optimizer (global_step=%d epoch=%d sample_idx=%d)",
                    resume.global_step,
                    resume.epoch,
                    resume.sample_idx_in_epoch,
                )

    grad_trace: dict = {}
    global_step = resume.global_step if resume else 0
    start_epoch = resume.epoch if resume else 0
    start_sample_idx = resume.sample_idx_in_epoch if resume else 0
    for epoch in range(start_epoch, args.num_epochs):
        pair_samples, dropped = epoch_samples_for_rank(
            samples,
            epoch=epoch,
            args=args,
            num_shards=split_cfg.num_pairs,
            shard_idx=split_cfg.pair_idx,
            drop_last=True,
        )
        if split_cfg.is_train_main and dropped:
            logger.warning("Dropping %d samples this epoch so split infer/train pairs stay step-aligned", dropped)
        if not pair_samples:
            raise RuntimeError("No samples left for split infer/train mode; increase dataset size or lower process count")

        skip_head = start_sample_idx if epoch == start_epoch else 0
        pair_tail = pair_samples[skip_head:]
        epoch_opt_total = len(pair_tail) * updates_per_sample
        show_infer_pbar = split_cfg.is_infer_rank and accelerator.is_local_main_process and split_cfg.pair_idx == 0
        show_train_pbar = split_cfg.is_train_rank and split_cfg.is_train_main
        infer_pbar = tqdm(
            total=epoch_opt_total,
            unit="opt_step",
            disable=not show_infer_pbar,
            desc=f"epoch {epoch} infer",
        )
        train_pbar = tqdm(
            total=epoch_opt_total,
            unit="opt_step",
            disable=not show_train_pbar,
            desc=f"epoch {epoch} train",
        )

        if split_cfg.is_infer_rank:
            for sample in pair_tail:
                payload = build_rollout_training_batch(
                    pipe=pipe,
                    sample=sample,
                    args=args,
                    device=device,
                    global_step=global_step,
                    num_train_steps=num_train_steps,
                    ir_scorer=ir_scorer,
                    vqa_scorer=vqa_scorer,
                )
                if should_log_images(global_step + updates_per_sample, args.log_image_steps):
                    _log_step = global_step + updates_per_sample
                    log_rollout_group_reward_advantage(
                        tb_writer=None,
                        output_dir=out_dir,
                        global_step=_log_step,
                        prompt=payload["prompt_text"],
                        r_total=payload["rollout_r_total"],
                        r_ir=payload["rollout_r_ir"],
                        r_vqa=payload["rollout_r_vqa"],
                        advantages=payload["advantages"],
                        write_disk=True,
                    )
                    if args.save_rollout_sample_images:
                        log_rollout_sample_image_files(
                            tb_writer=None,
                            output_dir=out_dir,
                            global_step=_log_step,
                            images=payload["images_pil"],
                            prompt=payload["prompt_text"],
                            max_logged_images=args.max_logged_images,
                            r_total=payload["rollout_r_total"],
                            r_ir=payload["rollout_r_ir"],
                            r_vqa=payload["rollout_r_vqa"],
                            advantages=payload["advantages"],
                        )
                payload.pop("images_pil")
                payload.pop("prompt_text")
                send_rollout_payload(payload, split_cfg.pair_rank)
                sync_trainable_params(pipe.unet, split_cfg.pair_rank, send=False)
                global_step += updates_per_sample
                infer_pbar.update(updates_per_sample)
                infer_pbar.set_postfix(gstep=global_step, refresh=True)
            infer_pbar.close()
            train_pbar.close()
            continue

        for _step in range(skip_head, len(pair_samples)):
            payload = recv_rollout_payload(split_cfg.pair_rank, device)
            roll_r_total = payload.pop("rollout_r_total")
            roll_r_ir = payload.pop("rollout_r_ir")
            roll_r_vqa = payload.pop("rollout_r_vqa")
            reward_metrics = payload.pop("reward_metrics")
            updates, train_metrics = run_ddpo_update(
                pipe=pipe,
                optimizer=optimizer,
                batch=payload,
                args=args,
                torch_dtype=torch_dtype,
                device=device,
                backward_fn=lambda loss: loss.backward(),
                grad_trace=grad_trace,
            )
            global_step += updates
            train_pbar.update(updates)

            reward_metrics = reward_metrics.to(device=device, dtype=torch.float32)
            train_metrics = train_metrics.to(device=device, dtype=torch.float32)
            dist.all_reduce(reward_metrics, group=split_cfg.train_group)
            dist.all_reduce(train_metrics, group=split_cfg.train_group)
            reward_metrics /= split_cfg.num_train_processes
            train_metrics /= split_cfg.num_train_processes

            loss_v = float(train_metrics[0].item())
            rmean_v = float(reward_metrics[0].item())
            train_pbar.set_postfix(
                gstep=global_step,
                loss=f"{loss_v:.4f}",
                rwd=f"{rmean_v:.4f}",
                refresh=True,
            )

            if updates > 0:
                maybe_log_step_scalars(
                    tb_writer=tb_writer,
                    global_step=global_step,
                    logging_steps=args.logging_steps,
                    rollout_metrics=reward_metrics,
                    train_metrics=train_metrics,
                )
            if updates > 0 and should_log_images(global_step, args.log_image_steps):
                if split_cfg.is_train_main:
                    log_rollout_group_reward_advantage(
                        tb_writer=tb_writer,
                        output_dir=out_dir,
                        global_step=global_step,
                        prompt=None,
                        r_total=roll_r_total,
                        r_ir=roll_r_ir,
                        r_vqa=roll_r_vqa,
                        advantages=payload["advantages"],
                        write_disk=False,
                    )
                if split_cfg.is_train_main and args.save_rollout_sample_images:
                    pair_step_dir = out_dir / "rollout_samples" / f"step_{global_step:08d}"
                    tb_images_pil: list[Image.Image] = []
                    if pair_step_dir.exists():
                        for image_path in sorted(pair_step_dir.glob("*.png"))[: args.max_logged_images]:
                            with Image.open(image_path) as im:
                                tb_images_pil.append(im.convert("RGB"))
                    prompt_tb = ""
                    ppt = pair_step_dir / "prompt.txt"
                    if ppt.is_file():
                        prompt_tb = ppt.read_text(encoding="utf-8")
                    if tb_images_pil:
                        log_rollout_sample_image_files(
                            tb_writer=tb_writer,
                            output_dir=out_dir,
                            global_step=global_step,
                            images=tb_images_pil,
                            prompt=prompt_tb or "(missing prompt.txt)",
                            max_logged_images=args.max_logged_images,
                            r_total=roll_r_total,
                            r_ir=roll_r_ir,
                            r_vqa=roll_r_vqa,
                            advantages=payload["advantages"],
                        )

            sync_trainable_params(pipe.unet, split_cfg.pair_rank, send=True)

            if (
                updates > 0
                and split_cfg.is_train_main
                and global_step > 0
                and global_step % args.save_steps == 0
            ):
                save_lora(out_dir / f"lora_step_{global_step}", pipe, accelerator)

            ck_every = int(args.checkpointing_steps)
            if (
                ck_every > 0
                and updates > 0
                and split_cfg.is_train_main
                and global_step > 0
                and global_step % ck_every == 0
            ):
                prune_old_checkpoints(out_dir, args.checkpoints_total_limit)
                ckpt_path = out_dir / f"checkpoint-{global_step}"
                ne, ns = (
                    (epoch + 1, 0)
                    if _step + 1 >= len(pair_samples)
                    else (epoch, _step + 1)
                )
                save_training_checkpoint(
                    ckpt_path,
                    pipe,
                    optimizer,
                    accelerator,
                    global_step=global_step,
                    epoch=ne,
                    next_sample_idx_in_epoch=ns,
                )
        train_pbar.close()
        infer_pbar.close()

    if split_cfg.is_train_main:
        save_lora(out_dir / "lora_final", pipe, accelerator)


def main():
    _load_dotenv_files()
    args = parse_args()
    _apply_split_infer_gpu_visibility(args)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=max(1, int(args.gradient_accumulation_steps)),
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    device = accelerator.device
    if args.mixed_precision == "fp16":
        torch_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = setup_split_roles(args, accelerator)

    resume_ckpt_dir: Path | None = None
    resume_state: ResumeState | None = None
    if args.resume_from_checkpoint:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(out_dir, args.resume_from_checkpoint)
        resume_state = load_resume_state_from_dir(resume_ckpt_dir)
        lora_file = resume_ckpt_dir / "unet_lora.safetensors"
        if not lora_file.is_file():
            raise FileNotFoundError(f"Resume directory missing LoRA weights: {lora_file}")

    tb_dir = out_dir / "logs_grpo"
    should_write_tb = split_cfg.is_train_main if split_cfg.enabled else accelerator.is_main_process
    tb_writer = SummaryWriter(log_dir=str(tb_dir)) if should_write_tb else None
    log_run_config(tb_writer, args, accelerator, split_cfg)

    lora_ckpt = str(resume_ckpt_dir / "unet_lora.safetensors") if resume_ckpt_dir else None
    pipe = load_pipeline(args, str(device), torch_dtype, lora_path_override=lora_ckpt)
    accelerator.wait_for_everyone()
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)
    if split_cfg.is_train_rank:
        pipe.vae.to("cpu")
        pipe.text_encoder.to("cpu")
        if pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    need_rewards = (not split_cfg.enabled) or split_cfg.is_infer_rank
    ir_scorer = (
        None
        if (args.skip_imagereward or not need_rewards)
        else ImageRewardScorer(
            device=device,
            dtype=torch_dtype,
            max_batch_size=args.imagereward_max_batch_size,
        )
    )
    vqa_scorer = None
    if not args.skip_vqa and need_rewards and args.weight_vqa != 0.0:
        if args.vqa_backend == "pickscore":
            vqa_scorer = FlowGrpoPickScoreReward(
                device=device,
                dtype=torch.float32,
                processor_id=args.pickscore_processor_id,
                model_id=args.pickscore_model_id,
                max_batch_size=args.pickscore_max_batch_size,
            )
        elif args.vqa_backend == "geneval_remote":
            vqa_scorer = GenevalRemoteScorer(
                base_url=args.geneval_server_url,
                only_strict=bool(args.geneval_only_strict),
                reward_field=args.geneval_reward_field,
                timeout_sec=args.geneval_timeout_sec,
                max_batch_size=args.geneval_max_batch_size,
                max_retries=args.geneval_max_retries,
            )
        elif args.vqa_backend == "vllm_openai":
            tmpl = args.global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
            base_url = args.vqa_openai_base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
            oa_key = (
                args.vqa_openai_api_key
                if args.vqa_openai_api_key is not None
                else os.environ.get("OPENAI_API_KEY", "EMPTY")
            )
            vqa_scorer = VllmOpenAiVqaProbScorer(
                model=args.vqa_model,
                base_url=base_url,
                api_key=oa_key,
                global_question_template_en=tmpl,
                global_weight=args.vqa_global_weight,
                judge_weight=args.vqa_judge_weight,
                max_workers=args.vqa_max_workers,
                enable_thinking=args.vqa_enable_thinking,
            )
        elif args.vqa_backend == "vllm_openai_structured":
            tmpl = args.global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
            base_url = args.vqa_openai_base_url or os.environ.get("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1"
            oa_key = (
                args.vqa_openai_api_key
                if args.vqa_openai_api_key is not None
                else os.environ.get("OPENAI_API_KEY", "EMPTY")
            )
            vqa_scorer = VllmOpenAiStructuredVqaScorer(
                model=args.vqa_model,
                base_url=base_url,
                api_key=oa_key,
                global_question_template_en=tmpl,
                global_weight=args.vqa_global_weight,
                judge_weight=args.vqa_judge_weight,
                max_workers=args.vqa_max_workers,
                max_tokens=args.vqa_structured_max_tokens,
                temperature=args.vqa_structured_temperature,
                enable_thinking=not args.vqa_structured_disable_thinking,
                timeout_sec=args.vqa_structured_timeout_sec,
            )
        else:
            tmpl = args.global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
            api_key = os.environ.get("DASHSCOPE_API_KEY")
            vqa_scorer = DashScopeVqaProbScorer(
                model=args.vqa_model,
                api_key=api_key,
                global_question_template_en=tmpl,
                global_weight=args.vqa_global_weight,
                judge_weight=args.vqa_judge_weight,
                max_workers=args.vqa_max_workers,
            )

    samples = list_grpo_jsonl(args.grpo_jsonl, prompt_field=args.prompt_field, max_samples=args.max_samples_per_epoch)
    if not samples:
        raise RuntimeError("No samples loaded from JSONL")
    num_train_steps = max(1, int(args.sample_num_steps * args.train_timestep_fraction))
    print_training_startup_config(
        args=args,
        accelerator=accelerator,
        split_cfg=split_cfg,
        num_train_steps=num_train_steps,
        num_samples=len(samples),
        torch_dtype=torch_dtype,
        device=device,
        resume_state=resume_state,
        resume_ckpt_dir=str(resume_ckpt_dir) if resume_ckpt_dir else None,
    )
    if split_cfg.enabled:
        run_split_training(
            args=args,
            accelerator=accelerator,
            pipe=pipe,
            samples=samples,
            tb_writer=tb_writer,
            ir_scorer=ir_scorer,
            vqa_scorer=vqa_scorer,
            num_train_steps=num_train_steps,
            torch_dtype=torch_dtype,
            split_cfg=split_cfg,
            resume=resume_state,
        )
    else:
        run_standard_training(
            args=args,
            accelerator=accelerator,
            pipe=pipe,
            samples=samples,
            tb_writer=tb_writer,
            ir_scorer=ir_scorer,
            vqa_scorer=vqa_scorer,
            num_train_steps=num_train_steps,
            torch_dtype=torch_dtype,
            resume=resume_state,
        )
        if accelerator.is_main_process:
            save_lora(out_dir / "lora_final", pipe, accelerator)

    if split_cfg.enabled:
        accelerator.wait_for_everyone()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
