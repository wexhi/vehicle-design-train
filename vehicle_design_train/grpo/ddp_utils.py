"""Helpers when UNet is wrapped in DistributedDataParallel."""

from __future__ import annotations

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def unwrap_unet(unet: nn.Module) -> nn.Module:
    """Access inner UNet for `.config`, `.dtype`, etc. Forward can stay on the wrapper."""
    return unet.module if isinstance(unet, DDP) else unet
