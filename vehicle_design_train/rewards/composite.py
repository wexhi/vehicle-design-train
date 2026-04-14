from __future__ import annotations

import torch


def normalize_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


def combine_rewards(
    r_ir: torch.Tensor,
    r_vqa: torch.Tensor,
    weight_ir: float,
    weight_vqa: float,
    normalize_ir: bool = True,
    normalize_vqa: bool = False,
) -> torch.Tensor:
    """Linear blend. Optionally min-max normalize ImageReward within the given tensor (e.g. per group)."""
    if normalize_ir:
        r_ir = normalize_minmax(r_ir)
    if normalize_vqa:
        r_vqa = normalize_minmax(r_vqa)
    return weight_ir * r_ir + weight_vqa * r_vqa
