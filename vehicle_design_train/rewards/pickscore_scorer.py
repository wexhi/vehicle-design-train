"""PickScore reward aligned with yifan123/flow_grpo `pickscore_scorer.py` (human-preference / Pick-a-Pic)."""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

__all__ = ["PickScoreScorer", "FlowGrpoPickScoreReward"]


class PickScoreScorer(torch.nn.Module):
    """Same scoring as Flow-GRPO: CLIP-H + PickScore_v1, logits scaled and divided by 26."""

    def __init__(
        self,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float32,
        processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_id: str = "yuvalkirstain/PickScore_v1",
    ):
        super().__init__()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.processor = CLIPProcessor.from_pretrained(processor_id)
        self.model = CLIPModel.from_pretrained(model_id).eval().to(device=self.device, dtype=self.dtype)
        self.model.requires_grad_(False)

    @torch.no_grad()
    def forward(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:
        if len(prompts) != len(images):
            raise ValueError("prompts and images must have the same length")
        if not images:
            return torch.empty(0, device=self.device, dtype=self.dtype)

        image_inputs = self.processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}

        text_inputs = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)

        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        scores = scores.diag()
        scores = scores / 26.0
        return scores

    def __call__(self, prompts: list[str], images: list[Image.Image]) -> torch.Tensor:  # type: ignore[override]
        return self.forward(prompts, images)


class FlowGrpoPickScoreReward:
    """Drop-in for VQA scorers: fills `r_vqa` with official PickScore (ignores JSONL judge questions)."""

    def __init__(
        self,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        processor_id: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        model_id: str = "yuvalkirstain/PickScore_v1",
        max_batch_size: Optional[int] = None,
    ):
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = dtype
        self.processor_id = processor_id
        self.model_id = model_id
        self.max_batch_size = max_batch_size
        self._scorer: Optional[PickScoreScorer] = None

    def _ensure(self) -> PickScoreScorer:
        if self._scorer is None:
            logger.info(
                "Loading PickScore (Flow-GRPO): processor=%s model=%s device=%s",
                self.processor_id,
                self.model_id,
                self.device,
            )
            self._scorer = PickScoreScorer(
                device=self.device,
                dtype=self.dtype,
                processor_id=self.processor_id,
                model_id=self.model_id,
            )
        return self._scorer

    @torch.inference_mode()
    def score_rollout_group(
        self,
        images: list[Image.Image],
        prompt_en: str,
        judge_questions: Optional[list[dict[str, Any]]],
        *,
        geneval_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        del judge_questions  # Flow-GRPO pickscore task uses caption only
        del geneval_metadata
        if not images:
            return [], []
        scorer = self._ensure()
        n = len(images)
        limit = self.max_batch_size if self.max_batch_size is not None and self.max_batch_size > 0 else n
        parts: list[torch.Tensor] = []
        for start in range(0, n, limit):
            chunk = images[start : start + limit]
            prompts = [prompt_en] * len(chunk)
            parts.append(scorer(prompts, chunk).detach().float().cpu())
        flat = torch.cat(parts, dim=0)
        scores = [float(flat[i].item()) for i in range(flat.shape[0])]
        details = [
            {
                "reward": "pickscore",
                "flow_grpo_norm": "divide_by_26",
                "score": scores[i],
            }
            for i in range(len(scores))
        ]
        return scores, details
