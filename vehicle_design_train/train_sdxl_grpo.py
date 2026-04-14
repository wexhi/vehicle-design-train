#!/usr/bin/env python
"""SDXL LoRA post-training with GRPO-style group advantages + DDPO loss (DDIM log-probs)."""

from __future__ import annotations

import argparse
import logging
import os
import random
from contextlib import nullcontext
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, StableDiffusionXLPipeline
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torch.utils.tensorboard import SummaryWriter
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
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=50)
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

    tb_dir = out_dir / "logs_grpo"
    tb_writer = SummaryWriter(log_dir=str(tb_dir)) if accelerator.is_main_process else None

    pipe = load_pipeline(args, str(device), torch_dtype)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    if pipe.text_encoder_2 is not None:
        pipe.text_encoder_2.requires_grad_(False)

    ir_scorer = None if args.skip_imagereward else ImageRewardScorer(device=device, dtype=torch_dtype)
    vqa_scorer = None
    if not args.skip_vqa:
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
    random.shuffle(samples)
    if accelerator.num_processes > 1:
        samples = samples[accelerator.process_index :: accelerator.num_processes]

    trainable = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.train_learning_rate)
    pipe.unet, optimizer = accelerator.prepare(pipe.unet, optimizer)
    num_train_steps = max(1, int(args.sample_num_steps * args.train_timestep_fraction))

    global_step = 0
    for epoch in range(args.num_epochs):
        random.shuffle(samples)
        pbar = tqdm(samples, disable=not accelerator.is_local_main_process, desc=f"epoch {epoch}")
        for sample in pbar:
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

            if tb_writer is not None:
                tb_writer.add_scalar("reward/mean", r_total.mean().item(), global_step)
                tb_writer.add_scalar("reward/r_ir_mean", r_ir_t.mean().item(), global_step)
                tb_writer.add_scalar("reward/r_vqa_mean", r_vqa_t.mean().item(), global_step)

            stacked = stack_rollouts(results)
            G, T, c, h, w = stacked["latents"].shape
            perm_steps = torch.stack(
                [torch.randperm(T, device=device) for _ in range(G)]
            )
            idx = perm_steps[:, :num_train_steps]

            latents_b = torch.stack(
                [stacked["latents"][i, idx[i]] for i in range(G)],
                dim=0,
            )
            next_latents_b = torch.stack(
                [stacked["next_latents"][i, idx[i]] for i in range(G)],
                dim=0,
            )
            logp_b = torch.stack(
                [stacked["log_probs"][i, idx[i]] for i in range(G)],
                dim=0,
            )
            time_b = torch.stack(
                [stacked["timesteps"][i, idx[i]] for i in range(G)],
                dim=0,
            )

            pipe.unet.train()
            autocast_cm = (
                torch.autocast(device_type="cuda", dtype=torch_dtype)
                if device.type == "cuda"
                else nullcontext()
            )

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
                        stacked["prompt_embeds"],
                        stacked["negative_prompt_embeds"],
                        stacked["pooled_prompt_embeds"],
                        stacked["negative_pooled_prompt_embeds"],
                        stacked["add_time_ids"],
                        args.sample_guidance_scale,
                        args.train_cfg,
                        args.train_clip_range,
                        args.train_adv_clip_max,
                        args.sample_eta,
                        autocast_cm,
                    )
                    optimizer.zero_grad()
                    accelerator.backward(loss)
                    optimizer.step()
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", loss.item(), global_step)
                        tb_writer.add_scalar("train/approx_kl", approx_kl.item(), global_step)
                        tb_writer.add_scalar("train/clipfrac", clipfrac.item(), global_step)
                    global_step += 1

            if accelerator.is_main_process and global_step > 0 and global_step % args.save_steps == 0:
                save_lora(out_dir / f"lora_step_{global_step}", pipe, accelerator)

    if accelerator.is_main_process:
        save_lora(out_dir / "lora_final", pipe, accelerator)
        tb_writer.close()


def save_lora(path: Path, pipe, accelerator: Accelerator):
    path.mkdir(parents=True, exist_ok=True)
    unet = accelerator.unwrap_model(pipe.unet)
    state_dict = get_peft_model_state_dict(unet)
    from safetensors.torch import save_file

    save_file(state_dict, path / "unet_lora.safetensors")
    logger.info("Saved LoRA to %s", path)


if __name__ == "__main__":
    main()
