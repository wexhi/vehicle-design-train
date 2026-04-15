from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import calculate_shift, retrieve_timesteps

from vehicle_design_train.grpo.ddp_utils import unwrap_module
from vehicle_design_train.grpo.sd3_sde_with_logprob import sde_step_with_logprob


def flow_match_ode_step(
    scheduler: FlowMatchEulerDiscreteScheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Deterministic Flow-Match Euler update (matches FlowMatchEulerDiscreteScheduler.step when not stochastic)."""
    model_output = model_output.float()
    sample = sample.float()
    if not isinstance(timestep, torch.Tensor):
        timestep = torch.tensor([timestep], device=sample.device, dtype=scheduler.timesteps.dtype)
    elif timestep.ndim == 0:
        timestep = timestep.unsqueeze(0)
    step_index = [scheduler.index_for_timestep(t) for t in timestep]
    prev_step_index = [step + 1 for step in step_index]
    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(-1, *([1] * (len(sample.shape) - 1)))
    dt = sigma_prev - sigma
    return sample + dt * model_output


def _pick_fast_sde_window(
    *,
    num_steps: int,
    window_size: int,
    range_lo: int,
    range_hi: int,
    rng: random.Random,
) -> tuple[int, int]:
    """Match flow_grpo `sd3_pipeline_with_logprob_fast.py` window selection."""
    if window_size <= 0:
        return 0, max(num_steps - 1, 0)
    hi_start = range_hi - window_size
    if hi_start < range_lo:
        raise ValueError(
            f"fast_sde_window: need range_hi - fast_sde_window_size >= range_lo; got range [{range_lo}, {range_hi}), size {window_size}"
        )
    start = rng.randint(range_lo, hi_start)
    return start, start + window_size


@dataclass
class Sd3RolloutResult:
    images_pil: list
    latents_traj: torch.Tensor  # (T+1, C, H, W)
    log_probs: torch.Tensor  # (T,)
    timesteps: torch.Tensor  # (T,) scheduler timesteps used
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    negative_pooled_prompt_embeds: torch.Tensor
    rollout_mu: torch.Tensor  # (1,) float — dynamic shift mu for set_timesteps on train ranks
    # Flow-GRPO-Fast: True only on SDE transitions (train PPO/GRPO only on these indices)
    sde_trainable_mask: Optional[torch.Tensor] = None  # (T,) bool


def ensure_flow_match_scheduler(pipe) -> None:
    """Official Flow-GRPO uses custom SDE in sde_step_with_logprob, not scheduler.stochastic_sampling."""
    if not isinstance(pipe.scheduler, FlowMatchEulerDiscreteScheduler):
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)


def _scheduler_mu_for_resolution(pipe, height: int, width: int) -> float | None:
    if not pipe.scheduler.config.get("use_dynamic_shifting", False):
        return None
    lh = int(height) // pipe.vae_scale_factor
    lw = int(width) // pipe.vae_scale_factor
    inner = unwrap_module(pipe.transformer)
    ps = int(inner.config.patch_size)
    image_seq_len = (lh // ps) * (lw // ps)
    return float(
        calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.16),
        )
    )


def _broadcast_latents_to_group(latents: torch.Tensor, g: int) -> torch.Tensor:
    if latents.shape[0] == g:
        return latents
    if latents.shape[0] == 1:
        return latents.expand(g, -1, -1, -1)
    raise ValueError(f"expected latent batch 1 or {g}, got {latents.shape[0]}")


@torch.no_grad()
def _sd3_flow_rollout_parallel_fast(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    num_parallel: int,
    generators: list[Optional[torch.Generator]],
    max_sequence_length: int,
    noise_level: float,
    sde_type: str,
    output_type: str,
    fast_branch_seed: int,
    fast_sde_window_size: int,
    fast_sde_range_lo: int,
    fast_sde_range_hi: int,
) -> list[Sd3RolloutResult]:
    """Flow-GRPO-Fast: shared ODE prefix, SDE only on a random contiguous window (see flow_grpo sd3_pipeline_with_logprob_fast)."""
    g = num_parallel
    if g < 1:
        raise ValueError("num_parallel must be >= 1")

    ensure_flow_match_scheduler(pipe)
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    pipe._guidance_scale = guidance_scale
    pipe._clip_skip = None
    pipe._joint_attention_kwargs = None
    pipe._interrupt = False

    neg = negative_prompt if negative_prompt is not None else ""

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=neg,
        negative_prompt_2=neg,
        negative_prompt_3=neg,
        do_classifier_free_guidance=do_cfg,
        device=device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
    )

    neg_g = negative_prompt_embeds.expand(g, -1, -1)
    pos_g = prompt_embeds.expand(g, -1, -1)
    npool_g = negative_pooled_prompt_embeds.expand(g, -1)
    pool_g = pooled_prompt_embeds.expand(g, -1)
    if do_cfg:
        enc_cat_g = torch.cat([neg_g, pos_g], dim=0)
        pool_cat_g = torch.cat([npool_g, pool_g], dim=0)
        enc_cat_1 = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pool_cat_1 = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        enc_cat_g = pos_g
        pool_cat_g = pool_g
        enc_cat_1 = prompt_embeds
        pool_cat_1 = pooled_prompt_embeds

    mu = _scheduler_mu_for_resolution(pipe, height, width)
    sched_kw: dict = {}
    if mu is not None:
        sched_kw["mu"] = mu

    timestep_device = device if device.type != "xla" else "cpu"
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        timestep_device,
        **sched_kw,
    )
    tn = len(timesteps)
    range_hi = fast_sde_range_hi if fast_sde_range_hi > 0 else tn
    range_hi = min(max(range_hi, 0), tn)
    range_lo = min(max(fast_sde_range_lo, 0), max(tn - 1, 0))

    rng = random.Random(int(fast_branch_seed) & 0xFFFF_FFFF)
    s_start, s_end = _pick_fast_sde_window(
        num_steps=tn,
        window_size=fast_sde_window_size,
        range_lo=range_lo,
        range_hi=range_hi,
        rng=rng,
    )

    trainable = torch.zeros(tn, dtype=torch.bool, device=device)
    trainable[s_start:s_end] = True

    inner = unwrap_module(pipe.transformer)
    latents = pipe.prepare_latents(
        1,
        inner.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generators[0],
        None,
    ).float()

    all_latents: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []

    all_latents.append(_broadcast_latents_to_group(latents, g).clone())

    for i, t in enumerate(timesteps):
        if i == s_start and latents.shape[0] == 1:
            latents = latents.expand(g, -1, -1, -1).clone()

        use_g = latents.shape[0] == g
        enc_use = enc_cat_g if use_g else enc_cat_1
        pool_use = pool_cat_g if use_g else pool_cat_1
        latent_in = torch.cat([latents, latents], dim=0) if do_cfg else latents
        ts = t.expand(latent_in.shape[0]).to(device=device, dtype=latents.dtype)

        noise_pred = pipe.transformer(
            hidden_states=latent_in,
            timestep=ts,
            encoder_hidden_states=enc_use,
            pooled_projections=pool_use,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_cfg:
            nu, nt = noise_pred.chunk(2)
            noise_pred = nu + guidance_scale * (nt - nu)

        ts_batch = t.expand(latents.shape[0]).to(device=latents.device, dtype=pipe.scheduler.timesteps.dtype)
        in_win = s_start <= i < s_end

        if in_win:
            gen_for_noise = generators if use_g else (generators[0] if generators else None)
            latents, log_prob, _, _ = sde_step_with_logprob(
                pipe.scheduler,
                noise_pred.float(),
                ts_batch,
                latents.float(),
                noise_level=noise_level,
                prev_sample=None,
                generator=gen_for_noise,
                sde_type=sde_type,
            )
        else:
            nxt = flow_match_ode_step(pipe.scheduler, noise_pred.float(), ts_batch, latents.float())
            log_prob = torch.zeros(latents.shape[0], device=device, dtype=torch.float32)
            latents = nxt

        latents = latents.to(dtype=prompt_embeds.dtype)
        all_latents.append(_broadcast_latents_to_group(latents, g).clone())
        lp = log_prob.reshape(-1)
        if lp.shape[0] == 1 and g > 1:
            lp = lp.expand(g)
        elif lp.shape[0] != g:
            raise RuntimeError(f"log_prob batch {lp.shape[0]} != group {g}")
        all_log_probs.append(lp)

    traj_batched = torch.stack(all_latents, dim=0)
    logp_batched = torch.stack(all_log_probs, dim=0)

    if output_type == "pil":
        dec = (latents / pipe.vae.config.scaling_factor) + getattr(pipe.vae.config, "shift_factor", 0.0)
        needs_upcasting = pipe.vae.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False)
        if needs_upcasting:
            pipe.upcast_vae()
        image = pipe.vae.decode(dec, return_dict=False)[0]
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        denorm = [True] * g
        images_out = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=denorm)
        if not isinstance(images_out, list):
            images_out = [images_out]
    else:
        raise ValueError("Only output_type=pil supported for rewards")

    mu_t = torch.tensor([mu if mu is not None else 0.0], device=device, dtype=torch.float32)

    results: list[Sd3RolloutResult] = []
    for i in range(g):
        results.append(
            Sd3RolloutResult(
                images_pil=[images_out[i]],
                latents_traj=traj_batched[:, i],
                log_probs=logp_batched[:, i].reshape(-1),
                timesteps=timesteps,
                prompt_embeds=pos_g[i : i + 1],
                negative_prompt_embeds=neg_g[i : i + 1],
                pooled_prompt_embeds=pool_g[i : i + 1],
                negative_pooled_prompt_embeds=npool_g[i : i + 1],
                rollout_mu=mu_t,
                sde_trainable_mask=trainable,
            )
        )
    return results


@torch.no_grad()
def sd3_flow_rollout_parallel(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    num_parallel: int,
    generators: Optional[list[Optional[torch.Generator]]],
    max_sequence_length: int,
    noise_level: float,
    sde_type: str = "sde",
    output_type: str = "pil",
    *,
    flow_grpo_fast: bool = False,
    fast_branch_seed: int = 0,
    fast_sde_window_size: int = 2,
    fast_sde_range_lo: int = 0,
    fast_sde_range_hi: int = -1,
) -> list[Sd3RolloutResult]:
    if num_parallel < 1:
        raise ValueError("num_parallel must be >= 1")
    if generators is None:
        gen_list: list[Optional[torch.Generator]] = [None] * num_parallel
    else:
        if len(generators) != num_parallel:
            raise ValueError(f"generators length {len(generators)} != num_parallel {num_parallel}")
        gen_list = list(generators)

    if flow_grpo_fast:
        if any(g is None for g in gen_list):
            raise ValueError("flow_grpo_fast requires a non-None torch.Generator per group member (distinct SDE noise).")
        return _sd3_flow_rollout_parallel_fast(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_parallel=num_parallel,
            generators=gen_list,
            max_sequence_length=max_sequence_length,
            noise_level=noise_level,
            sde_type=sde_type,
            output_type=output_type,
            fast_branch_seed=fast_branch_seed,
            fast_sde_window_size=fast_sde_window_size,
            fast_sde_range_lo=fast_sde_range_lo,
            fast_sde_range_hi=fast_sde_range_hi,
        )

    ensure_flow_match_scheduler(pipe)
    device = pipe._execution_device
    do_cfg = guidance_scale > 1.0
    pipe._guidance_scale = guidance_scale
    pipe._clip_skip = None
    pipe._joint_attention_kwargs = None
    pipe._interrupt = False

    neg = negative_prompt if negative_prompt is not None else ""

    prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,
        prompt_3=prompt,
        negative_prompt=neg,
        negative_prompt_2=neg,
        negative_prompt_3=neg,
        do_classifier_free_guidance=do_cfg,
        device=device,
        num_images_per_prompt=num_parallel,
        max_sequence_length=max_sequence_length,
    )

    if do_cfg:
        enc_cat = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        pool_cat = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    else:
        enc_cat = prompt_embeds
        pool_cat = pooled_prompt_embeds

    mu = _scheduler_mu_for_resolution(pipe, height, width)
    sched_kw: dict = {}
    if mu is not None:
        sched_kw["mu"] = mu

    timestep_device = device if device.type != "xla" else "cpu"
    timesteps, _ = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        timestep_device,
        **sched_kw,
    )

    inner = unwrap_module(pipe.transformer)
    latents = pipe.prepare_latents(
        num_parallel,
        inner.config.in_channels,
        height,
        width,
        prompt_embeds.dtype,
        device,
        gen_list,
        None,
    ).float()

    all_latents: list[torch.Tensor] = [latents.clone()]
    all_log_probs: list[torch.Tensor] = []

    for _step_i, t in enumerate(timesteps):
        latent_in = torch.cat([latents, latents], dim=0) if do_cfg else latents
        ts = t.expand(latent_in.shape[0]).to(device=device, dtype=latents.dtype)

        noise_pred = pipe.transformer(
            hidden_states=latent_in,
            timestep=ts,
            encoder_hidden_states=enc_cat,
            pooled_projections=pool_cat,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        if do_cfg:
            nu, nt = noise_pred.chunk(2)
            noise_pred = nu + guidance_scale * (nt - nu)

        gen_for_noise = gen_list if len(gen_list) > 1 else (gen_list[0] if gen_list else None)
        ts_batch = t.expand(latents.shape[0]).to(device=latents.device, dtype=pipe.scheduler.timesteps.dtype)
        latents, log_prob, _, _ = sde_step_with_logprob(
            pipe.scheduler,
            noise_pred.float(),
            ts_batch,
            latents,
            noise_level=noise_level,
            prev_sample=None,
            generator=gen_for_noise,
            sde_type=sde_type,
        )
        latents = latents.to(dtype=prompt_embeds.dtype)
        all_latents.append(latents.clone())
        all_log_probs.append(log_prob)

    traj_batched = torch.stack(all_latents, dim=0)
    logp_batched = torch.stack(all_log_probs, dim=0)

    if output_type == "pil":
        dec = (latents / pipe.vae.config.scaling_factor) + getattr(pipe.vae.config, "shift_factor", 0.0)
        needs_upcasting = pipe.vae.dtype == torch.float16 and getattr(pipe.vae.config, "force_upcast", False)
        if needs_upcasting:
            pipe.upcast_vae()
        image = pipe.vae.decode(dec, return_dict=False)[0]
        if needs_upcasting:
            pipe.vae.to(dtype=torch.float16)
        denorm = [True] * num_parallel
        images_out = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=denorm)
        if not isinstance(images_out, list):
            images_out = [images_out]
    else:
        raise ValueError("Only output_type=pil supported for rewards")

    mu_t = torch.tensor([mu if mu is not None else 0.0], device=device, dtype=torch.float32)

    results: list[Sd3RolloutResult] = []
    for i in range(num_parallel):
        results.append(
            Sd3RolloutResult(
                images_pil=[images_out[i]],
                latents_traj=traj_batched[:, i],
                log_probs=logp_batched[:, i].reshape(-1),
                timesteps=timesteps,
                prompt_embeds=prompt_embeds[i : i + 1],
                negative_prompt_embeds=negative_prompt_embeds[i : i + 1],
                pooled_prompt_embeds=pooled_prompt_embeds[i : i + 1],
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[i : i + 1],
                rollout_mu=mu_t,
            )
        )
    return results


@torch.no_grad()
def sd3_flow_rollout(
    pipe,
    prompt: str,
    negative_prompt: Optional[str],
    height: int,
    width: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: Optional[torch.Generator],
    max_sequence_length: int,
    noise_level: float,
    sde_type: str = "sde",
    output_type: str = "pil",
) -> Sd3RolloutResult:
    g = [generator]
    return sd3_flow_rollout_parallel(
        pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_parallel=1,
        generators=g,
        max_sequence_length=max_sequence_length,
        noise_level=noise_level,
        sde_type=sde_type,
        output_type=output_type,
    )[0]
