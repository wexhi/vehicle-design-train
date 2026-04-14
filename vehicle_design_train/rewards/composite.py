from __future__ import annotations

import torch


def combine_rewards(
    r_ir: torch.Tensor,
    r_vqa: torch.Tensor,
    weight_ir: float,
    weight_vqa: float,
) -> torch.Tensor:
    """Linear blend of raw reward heads; relative scaling happens in group advantages."""
    return weight_ir * r_ir + weight_vqa * r_vqa
