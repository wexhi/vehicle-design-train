"""Periodic evaluation on eval_specs-style JSONL during SD3 GRPO training (one image per prompt)."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

from vehicle_design_train.grpo.sd3_rollout import ensure_flow_match_scheduler, sd3_flow_rollout_parallel
from vehicle_design_train.grpo_dataset import GrpoJsonlSample
from vehicle_design_train.rewards.composite import combine_rewards

logger = logging.getLogger(__name__)


def _prompt_subseed(prompt: str) -> int:
    return int(hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8], 16)


def default_eval_specs_jsonl_path() -> Path:
    """Repo `data/eval_specs.jsonl` next to package root."""
    return Path(__file__).resolve().parent.parent / "data" / "eval_specs.jsonl"


def resolve_eval_specs_jsonl_path(args: Any) -> Path | None:
    """Path to eval JSONL, or None if disabled / missing explicit path when required."""
    steps = int(getattr(args, "eval_steps", 0) or 0)
    if steps <= 0:
        return None
    raw = getattr(args, "eval_specs_jsonl", None)
    if raw and str(raw).strip():
        return Path(raw).expanduser().resolve()
    p = default_eval_specs_jsonl_path()
    return p


@torch.inference_mode()
def run_sd3_eval_specs_pass(
    *,
    pipe,
    args: Any,
    device: torch.device,
    eval_samples: list[GrpoJsonlSample],
    global_step: int,
    ir_scorer: Any | None,
    vqa_scorer: Any | None,
    tb_writer: SummaryWriter | None,
    out_dir: Path,
) -> None:
    """One eval pass: generate one image per eval spec, score, log TB + optional disk."""
    if not eval_samples:
        logger.warning("eval_steps>0 but eval sample list is empty; skip eval.")
        return

    ensure_flow_match_scheduler(pipe)
    inner = pipe.transformer
    was_training = inner.training
    inner.eval()

    n = len(eval_samples)
    ir_vals: list[float] = []
    vqa_vals: list[float] = []
    total_vals: list[float] = []
    tb_images: list[torch.Tensor] = []
    max_tb = max(0, int(getattr(args, "eval_max_tb_images", 0) or 0))

    try:
        for i, sample in enumerate(eval_samples):
            prompt = sample.prompt_en
            gen = torch.Generator(device=device).manual_seed(
                int(args.seed) + int(global_step) * 17_771 + i * 1_000_003 + 42
            )
            fast_seed = int(args.seed) + int(global_step) * 1_000_003 + _prompt_subseed(prompt)

            if args.flow_grpo_fast:
                results = sd3_flow_rollout_parallel(
                    pipe,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt or None,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.sample_num_steps,
                    guidance_scale=args.sample_guidance_scale,
                    num_parallel=1,
                    generators=[gen],
                    max_sequence_length=args.max_sequence_length,
                    noise_level=args.noise_level,
                    sde_type=args.sde_type,
                    output_type="pil",
                    flow_grpo_fast=True,
                    fast_branch_seed=fast_seed,
                    fast_sde_window_size=int(args.fast_sde_window_size),
                    fast_sde_range_lo=int(args.fast_sde_range_lo),
                    fast_sde_range_hi=int(args.fast_sde_range_hi),
                )
            else:
                results = sd3_flow_rollout_parallel(
                    pipe,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt or None,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.sample_num_steps,
                    guidance_scale=args.sample_guidance_scale,
                    num_parallel=1,
                    generators=[gen],
                    max_sequence_length=args.max_sequence_length,
                    noise_level=args.noise_level,
                    sde_type=args.sde_type,
                    output_type="pil",
                )

            img = results[0].images_pil[0]
            imgs = [img]
            prompts = [prompt]

            if ir_scorer is not None:
                r_ir_t = ir_scorer.score(imgs, prompts).to(device)
            else:
                r_ir_t = torch.zeros(1, device=device, dtype=torch.float32)

            if vqa_scorer is not None:
                vq, _det = vqa_scorer.score_rollout_group(
                    imgs, prompt, sample.judge_questions, geneval_metadata=sample.geneval
                )
                r_vqa_t = torch.tensor(vq, device=device, dtype=torch.float32)
            else:
                r_vqa_t = torch.zeros(1, device=device, dtype=torch.float32)

            r_total_t = combine_rewards(
                r_ir_t,
                r_vqa_t,
                float(args.weight_ir),
                float(args.weight_vqa),
            )
            ir_vals.append(float(r_ir_t[0].item()))
            vqa_vals.append(float(r_vqa_t[0].item()))
            total_vals.append(float(r_total_t[0].item()))

            if getattr(args, "eval_save_images", False):
                eval_dir = out_dir / "eval_samples" / f"step_{global_step:08d}"
                eval_dir.mkdir(parents=True, exist_ok=True)
                sid = sample.meta.get("sample_id") if sample.meta else None
                tag = str(sid) if sid else f"{i:04d}"
                img.save(eval_dir / f"{tag}.png")
                (eval_dir / f"{tag}_prompt.txt").write_text(prompt, encoding="utf-8")

            if tb_writer is not None and len(tb_images) < max_tb:
                tb_images.append(pil_to_tensor(img.convert("RGB")).float() / 255.0)
    finally:
        if was_training:
            inner.train()

    mean_ir = sum(ir_vals) / n
    mean_vqa = sum(vqa_vals) / n
    mean_total = sum(total_vals) / n

    logger.info(
        "[eval] step=%d specs=%d mean_r_ir=%.4f mean_r_vqa=%.4f mean_r_total=%.4f (weights ir=%s vqa=%s)",
        global_step,
        n,
        mean_ir,
        mean_vqa,
        mean_total,
        args.weight_ir,
        args.weight_vqa,
    )

    if tb_writer is None:
        return

    tb_writer.add_scalar("eval/count", float(n), global_step)
    if ir_scorer is not None:
        tb_writer.add_scalar("eval/mean_r_ir", mean_ir, global_step)
    if vqa_scorer is not None:
        tb_writer.add_scalar("eval/mean_r_vqa", mean_vqa, global_step)
    tb_writer.add_scalar("eval/mean_r_total_weighted", mean_total, global_step)
    if tb_images:
        tb_writer.add_images("eval/images", torch.stack(tb_images, dim=0), global_step)
