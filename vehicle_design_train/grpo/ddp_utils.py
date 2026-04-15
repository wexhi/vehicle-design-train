"""Helpers when UNet is wrapped in DistributedDataParallel."""

from __future__ import annotations

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_module(module: nn.Module) -> nn.Module:
    """Access inner module when wrapped in DDP (for `.config`, `.dtype`, etc.)."""
    return module.module if isinstance(module, DDP) else module


def unwrap_unet(unet: nn.Module) -> nn.Module:
    """Access inner UNet for `.config`, `.dtype`, etc. Forward can stay on the wrapper."""
    return unwrap_module(unet)
