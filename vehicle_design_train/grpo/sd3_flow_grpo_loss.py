from __future__ import annotations

from typing import Any

import torch

from vehicle_design_train.grpo.sd3_sde_with_logprob import sde_step_with_logprob


def sd3_flow_grpo_calculate_loss(
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
    guidance_scale: float,
    train_cfg: bool,
    clip_range: float,
    adv_clip_max: float,
    noise_level: float,
    sde_type: str,
    autocast_cm: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = latents.device
    b = latents.shape[0]
    sched = pipe.scheduler

    loss_terms: list[torch.Tensor] = []
    kl_terms: list[torch.Tensor] = []
    clip_terms: list[torch.Tensor] = []

    with autocast_cm:
        for i in range(b):
            li = latents[i : i + 1]
            nli = next_latents[i : i + 1]
            ti = timesteps[i : i + 1].reshape(-1).to(device=device, dtype=sched.timesteps.dtype)
            old_lp = old_log_probs[i : i + 1]
            adv_i = advantages[i : i + 1]

            if train_cfg:
                neg_e = negative_prompt_embeds[0:1]
                pos_e = prompt_embeds[0:1]
                encoder_hidden_states = torch.cat([neg_e, pos_e], dim=0)
                neg_p = negative_pooled_prompt_embeds[0:1]
                pos_p = pooled_prompt_embeds[0:1]
                pooled_cat = torch.cat([neg_p, pos_p], dim=0)
                latent_in = torch.cat([li, li], dim=0)
                ts_in = ti[0].expand(latent_in.shape[0]).to(device=device, dtype=latent_in.dtype)
            else:
                encoder_hidden_states = prompt_embeds[0:1]
                pooled_cat = pooled_prompt_embeds[0:1]
                latent_in = li
                ts_in = ti[0].expand(latent_in.shape[0]).to(device=device, dtype=latent_in.dtype)

            noise_pred = pipe.transformer(
                hidden_states=latent_in,
                timestep=ts_in,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_cat,
                joint_attention_kwargs=None,
                return_dict=False,
            )[0]

            if train_cfg:
                nu, nt = noise_pred.chunk(2)
                noise_pred = nu + guidance_scale * (nt - nu)

            _prev, log_prob, _, _ = sde_step_with_logprob(
                sched,
                noise_pred.float(),
                ti,
                li.float(),
                noise_level=noise_level,
                prev_sample=nli.float(),
                generator=None,
                sde_type=sde_type,
            )

            adv = torch.clamp(adv_i, -adv_clip_max, adv_clip_max)
            ratio = torch.exp(log_prob - old_lp)
            unclipped = -adv * ratio
            clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            loss_terms.append(torch.maximum(unclipped, clipped))
            kl_terms.append(0.5 * (log_prob - old_lp) ** 2)
            clip_terms.append((torch.abs(ratio - 1.0) > clip_range).float())

    loss = torch.stack(loss_terms).mean()
    approx_kl = torch.stack(kl_terms).mean()
    clipfrac = torch.stack(clip_terms).mean()
    return loss, approx_kl, clipfrac
