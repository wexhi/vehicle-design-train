from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class ImageRewardScorer:
    """THUDM ImageReward (NeurIPS 2023) — human-preference RM scalar per (image, prompt)."""

    def __init__(self, device: torch.device | str, dtype: torch.dtype = torch.float16):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import ImageReward as ir_mod

        self._model = ir_mod.load("ImageReward-v1.0", device=self.device)
        logger.info("Loaded ImageReward-v1.0 on %s", self.device)

    @torch.inference_mode()
    def score(
        self,
        images: list["Image.Image"],
        prompts: list[str],
    ) -> torch.Tensor:
        if len(images) != len(prompts):
            raise ValueError("images and prompts length mismatch")
        self._ensure_model()
        scores = [float(self._model.score(prompt, img)) for prompt, img in zip(prompts, images)]
        return torch.tensor(scores, device=self.device, dtype=torch.float32)
