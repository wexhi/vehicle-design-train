# DDIM step with Gaussian log-probability, adapted from HuggingFace TRL
# (trl/models/modeling_sd_base.py) under Apache-2.0.

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from dataclasses import dataclass

from diffusers.utils.torch_utils import randn_tensor


def _left_broadcast(input_tensor: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError("broadcast shape mismatch")
    return input_tensor.reshape(input_tensor.shape + (1,) * (len(shape) - input_ndim)).broadcast_to(shape)


def _gather_alpha_prod(scheduler: Any, timestep: torch.Tensor) -> torch.Tensor:
    t_cpu = timestep.cpu().long().reshape(-1)
    out = torch.empty_like(t_cpu, dtype=scheduler.alphas_cumprod.dtype)
    valid = t_cpu >= 0
    if valid.any():
        out[valid] = scheduler.alphas_cumprod.gather(0, t_cpu[valid])
    if (~valid).any():
        out[~valid] = scheduler.final_alpha_cumprod
    return out.reshape(timestep.shape).to(timestep.device)


def _get_variance(scheduler: Any, timestep: torch.Tensor, prev_timestep: torch.Tensor) -> torch.Tensor:
    alpha_prod_t = _gather_alpha_prod(scheduler, timestep)
    alpha_prod_t_prev = _gather_alpha_prod(scheduler, prev_timestep)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


@dataclass
class DDIMStepOutput:
    prev_sample: torch.Tensor
    log_probs: torch.Tensor


def ddim_step_with_logprob(
    scheduler: Any,
    model_output: torch.FloatTensor,
    timestep: torch.Tensor,
    sample: torch.FloatTensor,
    eta: float = 0.0,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
) -> DDIMStepOutput:
    """One DDIM reverse step and log p(prev_sample | sample) (mean over spatial dims)."""
    if scheduler.num_inference_steps is None:
        raise ValueError("Call set_timesteps before rollout.")

    prev_timestep = timestep - scheduler.config.num_train_timesteps // scheduler.num_inference_steps

    alpha_prod_t = _gather_alpha_prod(scheduler, timestep)
    alpha_prod_t_prev = _gather_alpha_prod(scheduler, prev_timestep)
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    if scheduler.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        pred_epsilon = model_output
    elif scheduler.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
        pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
    else:
        raise ValueError(f"prediction_type {scheduler.config.prediction_type} not supported for GRPO rollout")

    if scheduler.config.thresholding:
        pred_original_sample = scheduler._threshold_sample(pred_original_sample)
    elif getattr(scheduler.config, "clip_sample", False):
        pred_original_sample = pred_original_sample.clamp(
            -scheduler.config.clip_sample_range,
            scheduler.config.clip_sample_range,
        )

    variance = _get_variance(scheduler, timestep, prev_timestep)
    std_dev_t = eta * variance ** (0.5)
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)
    deterministic_mask = std_dev_t <= 1e-12

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise
        prev_sample = torch.where(deterministic_mask, prev_sample_mean, prev_sample)

    log_prob = torch.zeros_like(prev_sample_mean)
    stochastic_mask = ~deterministic_mask
    if stochastic_mask.any():
        std_safe = torch.where(stochastic_mask, std_dev_t, torch.ones_like(std_dev_t))
        log_prob = torch.where(
            stochastic_mask,
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_safe**2))
            - torch.log(std_safe)
            - torch.log(torch.sqrt(torch.as_tensor(2 * np.pi, device=sample.device, dtype=sample.dtype))),
            log_prob,
        )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return DDIMStepOutput(prev_sample.type(sample.dtype), log_prob)
