#!/usr/bin/env python
"""SDXL LoRA post-training with GRPO-style group advantages + DDPO loss (DDIM log-probs)."""

from __future__ import annotations

import argparse
import logging
import math
import os
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
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor
from tqdm.auto import tqdm

from vehicle_design_train.grpo.group_advantage import compute_group_advantages
from vehicle_design_train.grpo.sdxl_ddpo_loss import sdxl_ddpo_calculate_loss
from vehicle_design_train.grpo.sdxl_rollout import sdxl_ddim_rollout
from vehicle_design_train.grpo_dataset import list_grpo_jsonl
from vehicle_design_train.rewards.composite import combine_rewards
from vehicle_design_train.rewards.imagereward_scorer import ImageRewardScorer
from vehicle_design_train.rewards.vqa_prob_scorer import (
    DEFAULT_GLOBAL_TEMPLATE_EN,
    DashScopeVqaProbScorer,
)

logger = get_logger(__name__)
logging.basicConfig(level=logging.INFO)

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
    p.add_argument("--grpo_jsonl", type=str, required=True, help="v3 JSONL with prompt_en and judge_questions")
    p.add_argument("--prompt_field", type=str, default="prompt_en")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None, help="Optional existing LoRA weights directory or .safetensors")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--merge_lora_into_unet", action="store_true", help="Fuse LoRA into UNet then attach new trainable LoRA")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--group_size", type=int, default=4, help="G rollouts per prompt (GRPO group)")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--max_samples_per_epoch", type=int, default=None)
    p.add_argument("--sample_num_steps", type=int, default=20)
    p.add_argument("--sample_eta", type=float, default=1.0)
    p.add_argument("--sample_guidance_scale", type=float, default=7.0)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--train_learning_rate", type=float, default=1e-5)
    p.add_argument("--train_batch_size", type=int, default=1, help="Must divide group_size for simple batching")
    p.add_argument("--train_cfg", action="store_true", default=True)
    p.add_argument("--no_train_cfg", action="store_false", dest="train_cfg")
    p.add_argument("--train_clip_range", type=float, default=1e-4)
    p.add_argument("--train_adv_clip_max", type=float, default=5.0)
    p.add_argument("--train_timestep_fraction", type=float, default=0.25, help="Fraction of denoise steps used in policy update")
    p.add_argument("--train_num_inner_epochs", type=int, default=1)
    p.add_argument("--weight_ir", type=float, default=0.5)
    p.add_argument("--weight_vqa", type=float, default=0.5)
    p.add_argument("--vqa_global_weight", type=float, default=1.0)
    p.add_argument("--vqa_judge_weight", type=float, default=1.0)
    p.add_argument("--vqa_model", type=str, default="qwen3.5-35b-a3b")
    p.add_argument("--global_question_template_en", type=str, default=None)
    p.add_argument("--skip_vqa", action="store_true")
    p.add_argument("--skip_imagereward", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--split_infer_train", action="store_true", help="Use half the processes for rollout/reward and half for DDP training")
    p.add_argument("--num_inference_processes", type=int, default=None, help="Number of rollout/reward processes when --split_infer_train is enabled; must equal the number of training processes")
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
    p.add_argument("--log_image_steps", type=int, default=0, help="If > 0, log rollout images to TensorBoard/local disk every N optimizer steps")
    p.add_argument("--max_logged_images", type=int, default=4, help="Max rollout images to log/save per image logging step")
    return p.parse_args()


def load_pipeline(args, device: str, torch_dtype: torch.dtype):
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

    if args.lora_path:
        lp = args.lora_path
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


def maybe_log_step_scalars(
    tb_writer: SummaryWriter | None,
    global_step: int,
    logging_steps: int,
    rollout_metrics: torch.Tensor | None = None,
    train_metrics: torch.Tensor | None = None,
) -> None:
    if tb_writer is None:
        return
    if global_step % max(logging_steps, 1) != 0:
        return
    if rollout_metrics is not None:
        log_tensorboard_scalars(tb_writer, _ROLLOUT_METRIC_NAMES, rollout_metrics, global_step)
    if train_metrics is not None:
        log_tensorboard_scalars(tb_writer, _TRAIN_METRIC_NAMES, train_metrics, global_step)


def log_run_config(tb_writer: SummaryWriter | None, args, accelerator: Accelerator, split_cfg: SplitRoleConfig) -> None:
    if tb_writer is None:
        return
    config_scalars = {
        "config/group_size": float(args.group_size),
        "config/sample_num_steps": float(args.sample_num_steps),
        "config/sample_eta": float(args.sample_eta),
        "config/sample_guidance_scale": float(args.sample_guidance_scale),
        "config/train_batch_size": float(args.train_batch_size),
        "config/train_timestep_fraction": float(args.train_timestep_fraction),
        "config/train_num_inner_epochs": float(args.train_num_inner_epochs),
        "config/train_learning_rate": float(args.train_learning_rate),
        "config/train_clip_range": float(args.train_clip_range),
        "config/train_adv_clip_max": float(args.train_adv_clip_max),
        "config/log_image_steps": float(args.log_image_steps),
        "config/max_logged_images": float(args.max_logged_images),
        "config/weight_ir": float(args.weight_ir),
        "config/weight_vqa": float(args.weight_vqa),
        "system/world_size": float(accelerator.num_processes),
        "system/split_infer_train": 1.0 if split_cfg.enabled else 0.0,
        "system/num_inference_processes": float(split_cfg.num_inference_processes if split_cfg.enabled else 0),
        "system/num_train_processes": float(split_cfg.num_train_processes if split_cfg.enabled else accelerator.num_processes),
    }
    for name, value in config_scalars.items():
        tb_writer.add_scalar(name, value, 0)


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
    for g in range(args.group_size):
        gen = torch.Generator(device=device).manual_seed(args.seed + global_step * 1000 + g)
        with torch.inference_mode():
            r = sdxl_ddim_rollout(
                pipe,
                prompt=prompt,
                negative_prompt=args.negative_prompt or None,
                height=args.resolution,
                width=args.resolution,
                num_inference_steps=args.sample_num_steps,
                guidance_scale=args.sample_guidance_scale,
                eta=args.sample_eta,
                generator=gen,
                output_type="pil",
            )
        results.append(r)

    imgs = [r.images_pil[0] for r in results]
    prompts_g = [prompt] * args.group_size

    if args.skip_imagereward:
        r_ir_t = torch.zeros(args.group_size, device=device, dtype=torch.float32)
    else:
        r_ir_t = ir_scorer.score(imgs, prompts_g).to(device)

    if vqa_scorer is None:
        r_vqa_t = torch.zeros(args.group_size, device=device, dtype=torch.float32)
    else:
        vqa_vals = []
        for im in imgs:
            rv, _det = vqa_scorer.score_sample(im, prompt, judges)
            vqa_vals.append(rv)
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
    perm_steps = torch.stack([torch.randperm(T, device=device) for _ in range(G)])
    idx = perm_steps[:, :num_train_steps]
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
        "images_pil": imgs,
        "prompt_text": prompt,
    }


def should_log_images(global_step: int, log_image_steps: int) -> bool:
    return log_image_steps > 0 and global_step > 0 and global_step % log_image_steps == 0


def sanitize_filename(text: str, limit: int = 80) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    safe = "".join(keep).strip("_")
    return (safe[:limit] or "prompt").rstrip("_")


def log_rollout_images(
    tb_writer: SummaryWriter | None,
    output_dir: Path,
    global_step: int,
    images: list[Image.Image],
    prompt: str,
    max_logged_images: int,
) -> None:
    if not images or max_logged_images <= 0:
        return

    chosen = images[:max_logged_images]
    prompt_tag = sanitize_filename(prompt)
    rollout_dir = output_dir / "rollout_samples" / f"step_{global_step:08d}"
    rollout_dir.mkdir(parents=True, exist_ok=True)

    tensors = []
    for idx, image in enumerate(chosen):
        save_path = rollout_dir / f"{idx:02d}_{prompt_tag}.png"
        image.save(save_path)
        if tb_writer is not None:
            tensors.append(pil_to_tensor(image.convert("RGB")).float() / 255.0)

    prompt_path = rollout_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    if tb_writer is not None and tensors:
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
) -> tuple[int, torch.Tensor]:
    train_start = time.perf_counter()
    latents_b = batch["latents_b"]
    next_latents_b = batch["next_latents_b"]
    logp_b = batch["logp_b"]
    time_b = batch["time_b"]
    advantages = batch["advantages"]

    G, num_train_steps, c, h, w = latents_b.shape
    pipe.unet.train()
    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=torch_dtype)
        if device.type == "cuda"
        else nullcontext()
    )

    metrics = torch.zeros(3, device=device, dtype=torch.float32)
    num_updates = 0
    for _inner in range(args.train_num_inner_epochs):
        flat_lat = latents_b.reshape(-1, c, h, w)
        flat_next = next_latents_b.reshape(-1, c, h, w)
        flat_logp = logp_b.reshape(-1)
        flat_time = time_b.reshape(-1)
        flat_adv = advantages.unsqueeze(1).expand(-1, num_train_steps).reshape(-1)

        n = flat_lat.shape[0]
        for start in range(0, n, args.train_batch_size):
            end = min(start + args.train_batch_size, n)
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
            optimizer.zero_grad()
            backward_fn(loss)
            optimizer.step()
            metrics += torch.stack([loss.detach(), approx_kl.detach(), clipfrac.detach()]).to(torch.float32)
            num_updates += 1

    train_metrics = make_train_metrics_tensor(
        train_metrics=metrics / max(num_updates, 1),
        updates=num_updates,
        optimizer=optimizer,
        train_sec=time.perf_counter() - train_start,
    )
    return num_updates, train_metrics


def save_lora(path: Path, pipe, accelerator: Accelerator):
    path.mkdir(parents=True, exist_ok=True)
    unet = get_unet_module(accelerator.unwrap_model(pipe.unet))
    state_dict = get_peft_model_state_dict(unet)
    from safetensors.torch import save_file

    save_file(state_dict, path / "unet_lora.safetensors")
    logger.info("Saved LoRA to %s", path)


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
):
    device = accelerator.device
    out_dir = Path(args.output_dir)
    if accelerator.num_processes > 1:
        samples = samples[accelerator.process_index :: accelerator.num_processes]

    trainable = trainable_params(pipe.unet)
    optimizer = torch.optim.AdamW(trainable, lr=args.train_learning_rate)
    pipe.unet, optimizer = accelerator.prepare(pipe.unet, optimizer)

    global_step = 0
    for epoch in range(args.num_epochs):
        epoch_samples, _dropped = epoch_samples_for_rank(
            samples,
            epoch=epoch,
            args=args,
            num_shards=1,
            shard_idx=0,
            drop_last=False,
        )
        pbar = tqdm(epoch_samples, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch}")
        for sample in pbar:
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

            updates, train_metrics = run_ddpo_update(
                pipe=pipe,
                optimizer=optimizer,
                batch=batch,
                args=args,
                torch_dtype=torch_dtype,
                device=device,
                backward_fn=accelerator.backward,
            )
            global_step += updates

            maybe_log_step_scalars(
                tb_writer=tb_writer,
                global_step=global_step,
                logging_steps=args.logging_steps,
                rollout_metrics=reward_metrics,
                train_metrics=train_metrics,
            )
            if should_log_images(global_step, args.log_image_steps):
                log_rollout_images(
                    tb_writer=tb_writer,
                    output_dir=out_dir,
                    global_step=global_step,
                    images=images_pil,
                    prompt=prompt_text,
                    max_logged_images=args.max_logged_images,
                )

            if accelerator.is_main_process and global_step > 0 and global_step % args.save_steps == 0:
                save_lora(Path(args.output_dir) / f"lora_step_{global_step}", pipe, accelerator)


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
):
    device = accelerator.device
    out_dir = Path(args.output_dir)
    updates_per_sample = args.train_num_inner_epochs * math.ceil(
        (args.group_size * num_train_steps) / args.train_batch_size
    )

    optimizer = None
    if split_cfg.is_train_rank:
        pipe.unet = DDP(
            pipe.unet,
            device_ids=[accelerator.local_process_index] if device.type == "cuda" else None,
            process_group=split_cfg.train_group,
            broadcast_buffers=False,
        )
        optimizer = torch.optim.AdamW(trainable_params(pipe.unet), lr=args.train_learning_rate)

    global_step = 0
    for epoch in range(args.num_epochs):
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

        show_progress = split_cfg.is_infer_rank and accelerator.is_local_main_process and split_cfg.pair_idx == 0
        pbar = tqdm(pair_samples, disable=not show_progress, desc=f"epoch {epoch}")

        if split_cfg.is_infer_rank:
            for sample in pbar:
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
                    log_rollout_images(
                        tb_writer=None,
                        output_dir=out_dir,
                        global_step=global_step + updates_per_sample,
                        images=payload["images_pil"],
                        prompt=payload["prompt_text"],
                        max_logged_images=args.max_logged_images,
                    )
                payload.pop("images_pil")
                payload.pop("prompt_text")
                send_rollout_payload(payload, split_cfg.pair_rank)
                sync_trainable_params(pipe.unet, split_cfg.pair_rank, send=False)
                global_step += updates_per_sample
            continue

        for _step in range(len(pair_samples)):
            payload = recv_rollout_payload(split_cfg.pair_rank, device)
            reward_metrics = payload.pop("reward_metrics")
            updates, train_metrics = run_ddpo_update(
                pipe=pipe,
                optimizer=optimizer,
                batch=payload,
                args=args,
                torch_dtype=torch_dtype,
                device=device,
                backward_fn=lambda loss: loss.backward(),
            )
            global_step += updates

            reward_metrics = reward_metrics.to(device=device, dtype=torch.float32)
            train_metrics = train_metrics.to(device=device, dtype=torch.float32)
            dist.all_reduce(reward_metrics, group=split_cfg.train_group)
            dist.all_reduce(train_metrics, group=split_cfg.train_group)
            reward_metrics /= split_cfg.num_train_processes
            train_metrics /= split_cfg.num_train_processes

            maybe_log_step_scalars(
                tb_writer=tb_writer,
                global_step=global_step,
                logging_steps=args.logging_steps,
                rollout_metrics=reward_metrics,
                train_metrics=train_metrics,
            )
            if should_log_images(global_step, args.log_image_steps):
                pair_step_dir = out_dir / "rollout_samples" / f"step_{global_step:08d}"
                if pair_step_dir.exists():
                    tb_images = []
                    for image_path in sorted(pair_step_dir.glob("*.png"))[: args.max_logged_images]:
                        with Image.open(image_path) as im:
                            tb_images.append(pil_to_tensor(im.convert("RGB")).float() / 255.0)
                    if tb_writer is not None and tb_images:
                        tb_writer.add_images("rollout/images", torch.stack(tb_images, dim=0), global_step)

            sync_trainable_params(pipe.unet, split_cfg.pair_rank, send=True)

            if split_cfg.is_train_main and global_step > 0 and global_step % args.save_steps == 0:
                save_lora(out_dir / f"lora_step_{global_step}", pipe, accelerator)

    if split_cfg.is_train_main:
        save_lora(out_dir / "lora_final", pipe, accelerator)


def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
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

    tb_dir = out_dir / "logs_grpo"
    should_write_tb = split_cfg.is_train_main if split_cfg.enabled else accelerator.is_main_process
    tb_writer = SummaryWriter(log_dir=str(tb_dir)) if should_write_tb else None
    log_run_config(tb_writer, args, accelerator, split_cfg)

    pipe = load_pipeline(args, str(device), torch_dtype)
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
    ir_scorer = None if (args.skip_imagereward or not need_rewards) else ImageRewardScorer(device=device, dtype=torch_dtype)
    vqa_scorer = None
    if not args.skip_vqa and need_rewards:
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        tmpl = args.global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
        vqa_scorer = DashScopeVqaProbScorer(
            model=args.vqa_model,
            api_key=api_key,
            global_question_template_en=tmpl,
            global_weight=args.vqa_global_weight,
            judge_weight=args.vqa_judge_weight,
        )

    samples = list_grpo_jsonl(args.grpo_jsonl, prompt_field=args.prompt_field, max_samples=args.max_samples_per_epoch)
    if not samples:
        raise RuntimeError("No samples loaded from JSONL")
    num_train_steps = max(1, int(args.sample_num_steps * args.train_timestep_fraction))
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
        )
        if accelerator.is_main_process:
            save_lora(out_dir / "lora_final", pipe, accelerator)

    if split_cfg.enabled:
        accelerator.wait_for_everyone()
    if tb_writer is not None:
        tb_writer.close()


if __name__ == "__main__":
    main()
