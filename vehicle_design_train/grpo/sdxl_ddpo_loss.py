from __future__ import annotations

from typing import Any

import torch

from vehicle_design_train.grpo.ddim_logprob import ddim_step_with_logprob


def sdxl_ddpo_calculate_loss(
    pipe,
    latents: torch.Tensor,
    timesteps: torch.Tensor,
    next_latents: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor,
    guidance_scale: float,
    train_cfg: bool,
    clip_range: float,
    adv_clip_max: float,
    eta: float,
    autocast_cm: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    DDPO-style clipped loss for SDXL UNet. Embeddings are single-sample positive/negative
    (1, seq, dim); duplicated to batch B via repeat.
    """
    device = latents.device
    b = latents.shape[0]

    if train_cfg:
        neg_e = negative_prompt_embeds.repeat(b, 1, 1)
        pos_e = prompt_embeds.repeat(b, 1, 1)
        encoder_hidden_states = torch.cat([neg_e, pos_e], dim=0)

        neg_p = negative_pooled_prompt_embeds.repeat(b, 1)
        pos_p = pooled_prompt_embeds.repeat(b, 1)
        add_text_embeds = torch.cat([neg_p, pos_p], dim=0)

        tid = add_time_ids.to(device)
        add_time_ids_b = torch.cat([tid.repeat(b, 1), tid.repeat(b, 1)], dim=0)
    else:
        encoder_hidden_states = prompt_embeds.repeat(b, 1, 1)
        add_text_embeds = pooled_prompt_embeds.repeat(b, 1)
        add_time_ids_b = add_time_ids.to(device).repeat(b, 1)

    timestep_cond = None
    if pipe.unet.config.time_cond_proj_dim is not None:
        gst = torch.tensor(guidance_scale - 1, device=device, dtype=latents.dtype).repeat(b)
        timestep_cond = pipe.get_guidance_scale_embedding(
            gst, embedding_dim=pipe.unet.config.time_cond_proj_dim
        ).to(dtype=latents.dtype)

    with autocast_cm:
        if train_cfg:
            latent_in = torch.cat([latents, latents], dim=0)
            t_in = torch.cat([timesteps, timesteps], dim=0)
            if timestep_cond is not None:
                timestep_cond_in = torch.cat([timestep_cond, timestep_cond], dim=0)
            else:
                timestep_cond_in = None
        else:
            latent_in = latents
            t_in = timesteps
            timestep_cond_in = timestep_cond

        added = {"text_embeds": add_text_embeds, "time_ids": add_time_ids_b}
        noise_pred = pipe.unet(
            latent_in,
            t_in,
            encoder_hidden_states=encoder_hidden_states,
            timestep_cond=timestep_cond_in,
            added_cond_kwargs=added,
            return_dict=False,
        )[0]

        if train_cfg:
            nu, nt = noise_pred.chunk(2)
            noise_pred = nu + guidance_scale * (nt - nu)

        step_out = ddim_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timesteps,
            latents,
            eta=eta,
            prev_sample=next_latents,
            generator=None,
        )
        log_prob = step_out.log_probs

    adv = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    ratio = torch.exp(log_prob - old_log_probs)
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    loss = torch.mean(torch.maximum(unclipped, clipped))
    approx_kl = 0.5 * torch.mean((log_prob - old_log_probs) ** 2)
    clipfrac = torch.mean((torch.abs(ratio - 1.0) > clip_range).float())
    return loss, approx_kl, clipfrac
