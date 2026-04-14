from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)


class ImageRewardScorer:
    """THUDM ImageReward (NeurIPS 2023) — human-preference RM scalar per (image, prompt)."""

    def __init__(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float16,
        max_batch_size: Optional[int] = None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self._model = None

    def _ensure_model(self):
        if self._model is not None:
            return
        import ImageReward as ir_mod

        self._model = ir_mod.load("ImageReward-v1.0", device=self.device)
        self._model.eval()
        logger.info("Loaded ImageReward-v1.0 on %s", self.device)

    def _forward_batch_one_prompt(self, prompt: str, images: list["Image.Image"]) -> torch.Tensor:
        """One batched forward: same text, multiple images. Returns float32 CPU vector (B,)."""
        model = self._model
        param = next(model.parameters())
        device, dtype = param.device, param.dtype

        blip = model.blip
        text_input = blip.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        text_input = text_input.to(device)
        b = len(images)
        input_ids = text_input.input_ids.expand(b, -1)
        attention_mask = text_input.attention_mask.expand(b, -1)

        image_batch = torch.stack([model.preprocess(img) for img in images], dim=0).to(device=device, dtype=dtype)
        image_embeds = blip.visual_encoder(image_batch)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        text_output = blip.text_encoder(
            input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        txt_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = model.mlp(txt_features)
        rewards = (rewards - model.mean) / model.std
        return rewards.view(-1).detach().cpu().to(torch.float32)

    def _score_same_prompt_in_chunks(self, prompt: str, images: list["Image.Image"]) -> torch.Tensor:
        if not images:
            return torch.empty(0, dtype=torch.float32)
        limit = self.max_batch_size if self.max_batch_size is not None and self.max_batch_size > 0 else len(images)
        parts: list[torch.Tensor] = []
        for start in range(0, len(images), limit):
            chunk = images[start : start + limit]
            parts.append(self._forward_batch_one_prompt(prompt, chunk))
        return torch.cat(parts, dim=0)

    @torch.inference_mode()
    def score(
        self,
        images: list["Image.Image"],
        prompts: list[str],
    ) -> torch.Tensor:
        if len(images) != len(prompts):
            raise ValueError("images and prompts length mismatch")
        n = len(images)
        if n == 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        self._ensure_model()

        by_prompt: dict[str, list[tuple[int, "Image.Image"]]] = defaultdict(list)
        for i, (p, img) in enumerate(zip(prompts, images)):
            by_prompt[p].append((i, img))

        flat = [0.0] * n
        for prompt, indexed in by_prompt.items():
            idxs = [t[0] for t in indexed]
            imgs = [t[1] for t in indexed]
            scores_cpu = self._score_same_prompt_in_chunks(prompt, imgs)
            for j, global_i in enumerate(idxs):
                flat[global_i] = float(scores_cpu[j])

        return torch.tensor(flat, device=self.device, dtype=torch.float32)
