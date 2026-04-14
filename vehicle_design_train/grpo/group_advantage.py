"""Group-relative advantages (GRPO-style) for rewards shaped (group_size,)."""

from __future__ import annotations

import numpy as np
import torch


def compute_group_advantages(
    rewards: torch.Tensor | np.ndarray | list[float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """A_i = (R_i - mean(R)) / (std(R) + eps). Same formula as batch-wide DDPO when group = batch."""
    if isinstance(rewards, list):
        r = torch.tensor(rewards, dtype=torch.float32)
    elif isinstance(rewards, np.ndarray):
        r = torch.from_numpy(rewards.astype(np.float32))
    else:
        r = rewards.float() if rewards.dtype != torch.float32 else rewards
    mean = r.mean()
    std = r.std(unbiased=False)
    return (r - mean) / (std + eps)
