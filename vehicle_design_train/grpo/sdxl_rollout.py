from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from vehicle_design_train.grpo.ddim_logprob import ddim_step_with_logprob
from vehicle_design_train.grpo.ddp_utils import unwrap_unet


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


def ensure_ddim_scheduler(pipe) -> None:
    """Swap in DDIM if needed; rollout and train loss must use the same scheduler."""
    from diffusers import DDIMScheduler

    if not isinstance(pipe.scheduler, DDIMScheduler):
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)


@torch.no_grad()
def sdxl_ddim_rollout_parallel(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    eta: float,
    num_parallel: int,
    generators: Optional[list[Optional[torch.Generator]]],
    output_type: str = "pil",
) -> list[SdxlRolloutResult]:
    """
    DDIM rollout for `num_parallel` images (same prompt), one UNet batch per timestep.
    `generators` length must match `num_parallel` (entries may be None).
    """
    if num_parallel < 1:
        raise ValueError("num_parallel must be >= 1")
    if generators is None:
        gen_list: list[Optional[torch.Generator]] = [None] * num_parallel
    else:
        if len(generators) != num_parallel:
            raise ValueError(f"generators length {len(generators)} != num_parallel {num_parallel}")
        gen_list = list(generators)

    ensure_ddim_scheduler(pipe)
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
        num_images_per_prompt=num_parallel,
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

    unet_inner = unwrap_unet(pipe.unet)
    latents = pipe.prepare_latents(
        num_parallel,
        unet_inner.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        gen_list,
        None,
    )

    add_text_embeds = pooled_prompt_embeds
    if pipe.text_encoder_2 is None:
        text_encoder_projection_dim = int(pooled_prompt_embeds.shape[-1])
    else:
        text_encoder_projection_dim = pipe.text_encoder_2.config.projection_dim

    # Pipeline._get_add_time_ids reads self.unet.config; DDP-wrapped unet has no .config.
    _wrapped_unet = pipe.unet
    pipe.unet = unet_inner
    try:
        add_time_ids_1 = pipe._get_add_time_ids(
            (height, width),
            (0, 0),
            (height, width),
            dtype=prompt_embeds.dtype,
            text_encoder_projection_dim=text_encoder_projection_dim,
        ).to(device)
    finally:
        pipe.unet = _wrapped_unet
    add_time_ids = add_time_ids_1.repeat(num_parallel, 1)

    if do_cfg:
        prompt_embeds_cat = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds_cat = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids_cat = torch.cat([add_time_ids, add_time_ids], dim=0)
    else:
        prompt_embeds_cat = prompt_embeds
        add_text_embeds_cat = add_text_embeds
        add_time_ids_cat = add_time_ids

    prompt_embeds_cat = prompt_embeds_cat.to(device)
    add_text_embeds_cat = add_text_embeds_cat.to(device)

    timestep_cond = None
    if unet_inner.config.time_cond_proj_dim is not None:
        gst = torch.tensor(guidance_scale - 1, device=device, dtype=latents.dtype).repeat(num_parallel)
        timestep_cond = pipe.get_guidance_scale_embedding(
            gst, embedding_dim=unet_inner.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    all_latents: list[torch.Tensor] = [latents.clone()]
    all_log_probs: list[torch.Tensor] = []

    for t in timesteps:
        latent_model_input = torch.cat([latents, latents], dim=0) if do_cfg else latents
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

        gen_for_ddim = gen_list if len(gen_list) > 1 else (gen_list[0] if gen_list else None)
        step_out = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            t,
            latents,
            eta=eta,
            prev_sample=None,
            generator=gen_for_ddim,
        )
        latents = step_out.prev_sample
        all_latents.append(latents.clone())
        all_log_probs.append(step_out.log_probs)

        if latents.dtype != prompt_embeds.dtype and torch.backends.mps.is_available():
            latents = latents.to(prompt_embeds.dtype)

    # (T+1, P, C, H, W)
    traj_batched = torch.stack(all_latents, dim=0)
    logp_batched = torch.stack(all_log_probs, dim=0)

    if output_type == "pil":
        needs_upcasting = pipe.vae.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False)
        if needs_upcasting:
            pipe.upcast_vae()
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        denorm = [True] * num_parallel
        images_out = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=denorm)
        if not isinstance(images_out, list):
            images_out = [images_out]
    else:
        raise ValueError("Only output_type=pil supported for rewards")

    pe = prompt_embeds
    npe = negative_prompt_embeds
    ppe = pooled_prompt_embeds
    nppe = negative_pooled_prompt_embeds
    ati = add_time_ids

    results: list[SdxlRolloutResult] = []
    for i in range(num_parallel):
        lt = traj_batched[:, i]
        lp = logp_batched[:, i].reshape(-1)
        results.append(
            SdxlRolloutResult(
                images_pil=[images_out[i]],
                latents_traj=lt,
                log_probs=lp,
                timesteps=timesteps,
                prompt_embeds=pe[i : i + 1],
                negative_prompt_embeds=npe[i : i + 1],
                pooled_prompt_embeds=ppe[i : i + 1],
                negative_pooled_prompt_embeds=nppe[i : i + 1],
                add_time_ids=ati[i : i + 1],
                text_encoder_projection_dim=text_encoder_projection_dim,
            )
        )
    return results


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
    g = [generator]
    return sdxl_ddim_rollout_parallel(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        eta=eta,
        num_parallel=1,
        generators=g,
        output_type=output_type,
    )[0]
