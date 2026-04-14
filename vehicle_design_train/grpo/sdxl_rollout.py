from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vehicle_design_train.grpo.ddim_logprob import ddim_step_with_logprob


@dataclass
class SdxlRolloutResult:
    """One sample rollout (batch size 1 image)."""

    images_pil: list  # list length 1 of PIL
    latents_traj: torch.Tensor  # (T+1, C, H, W)
    log_probs: torch.Tensor  # (T,)
    timesteps: torch.Tensor  # (T,)
    prompt_embeds: torch.Tensor  # (1, seq, dim) positive only (no CFG concat)
    negative_prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor
    add_time_ids: torch.Tensor  # before CFG concat — store positive branch (1, 6)
    text_encoder_projection_dim: int


def _ensure_ddim(pipe) -> None:
    from diffusers import DDIMScheduler

    if not isinstance(pipe.scheduler, DDIMScheduler):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


@torch.no_grad()
def sdxl_ddim_rollout(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    eta: float,
    generator: Optional[torch.Generator],
    output_type: str = "pil",
) -> SdxlRolloutResult:
    """
    DDIM rollout with per-step Gaussian log-probs (DDPO-compatible).
    Expects `pipe` to be StableDiffusionXLPipeline; swaps in DDIM if needed.
    """
    _ensure_ddim(pipe)
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    pipe._guidance_scale = guidance_scale
    pipe._guidance_rescale = 0.0
    pipe._clip_skip = None
    pipe._cross_attention_kwargs = None
    pipe._denoising_end = None
    pipe._interrupt = False

    neg = negative_prompt if negative_prompt is not None else ""

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=neg,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    )

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    latents = pipe.prepare_latents(
        1,
        pipe.unet.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        None,
    )

    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    add_time_ids = pipe._get_add_time_ids(
        (height, width),
        (0, 0),
        (height, width),
        dtype=prompt_embeds.dtype,
        text_encoder_projection_dim=text_encoder_projection_dim,
    )

    neg_add_time_ids = add_time_ids
    if do_cfg:
        prompt_embeds_cat = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds_cat = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids_cat = torch.cat([neg_add_time_ids, add_time_ids], dim=0)
    else:
        prompt_embeds_cat = prompt_embeds
        add_text_embeds_cat = add_text_embeds
        add_time_ids_cat = add_time_ids

    add_time_ids_cat = add_time_ids_cat.to(device).repeat(1, 1)

    prompt_embeds_cat = prompt_embeds_cat.to(device)
    add_text_embeds_cat = add_text_embeds_cat.to(device)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(guidance_scale - 1, device=device, dtype=latents.dtype).repeat(1)
        timestep_cond = pipe.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    all_latents = [latents]
    all_log_probs: list[torch.Tensor] = []

    for t in timesteps:
        latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        added_cond_kwargs = {"text_embeds": add_text_embeds_cat, "time_ids": add_time_ids_cat}
        noise_pred = pipe.unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds_cat,
            timestep_cond=timestep_cond,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[0]

        if do_cfg:
            nu, nt = noise_pred.chunk(2)
            noise_pred = nu + guidance_scale * (nt - nu)

        step_out = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            t,
            latents,
            eta=eta,
            prev_sample=None,
            generator=generator,
        )
        latents = step_out.prev_sample
        all_latents.append(latents)
        all_log_probs.append(step_out.log_probs)

        if latents.dtype != prompt_embeds.dtype and torch.backends.mps.is_available():
            latents = latents.to(prompt_embeds.dtype)

    latents_traj = torch.stack(all_latents, dim=0)
    log_probs_t = torch.stack(all_log_probs, dim=0).squeeze(-1)

    if output_type == "pil":
        needs_upcasting = pipe.vae.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False)
        if needs_upcasting:
            pipe.upcast_vae()
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])
    else:
        raise ValueError("Only output_type=pil supported for rewards")

    return SdxlRolloutResult(
        images_pil=image,
        latents_traj=latents_traj,
        log_probs=log_probs_t,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        add_time_ids=add_time_ids.to(device),
        text_encoder_projection_dim=text_encoder_projection_dim,
    )
