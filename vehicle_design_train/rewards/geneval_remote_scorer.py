"""GenEval reward via HTTP pickle API (yifan123/reward-server `app_geneval.py`), aligned with flow_grpo `geneval_score`."""

from __future__ import annotations

import io
import logging
import pickle
import time
import urllib.error
import urllib.request
from typing import Any, Literal, Optional

from PIL import Image

logger = logging.getLogger(__name__)

__all__ = ["GenevalRemoteScorer"]

RewardField = Literal["score", "accuracy", "strict_accuracy"]


def _pil_to_jpeg_bytes(image: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class GenevalRemoteScorer:
    """POST pickle {images, meta_datas, only_strict} → scores / rewards / strict_rewards (reward-server)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18085",
        *,
        only_strict: bool = False,
        reward_field: str = "score",
        timeout_sec: float = 120.0,
        max_batch_size: int = 64,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.only_strict = only_strict
        if reward_field not in ("score", "accuracy", "strict_accuracy"):
            raise ValueError(f"geneval_reward_field must be score|accuracy|strict_accuracy, got {reward_field!r}")
        self.reward_field: RewardField = reward_field  # type: ignore[assignment]
        self.timeout_sec = float(timeout_sec)
        self.max_batch_size = max(1, int(max_batch_size))
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_sec = float(retry_backoff_sec)

    def _post_batch(
        self,
        jpeg_list: list[bytes],
        meta_list: list[dict[str, Any]],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        payload = pickle.dumps(
            {
                "images": jpeg_list,
                "meta_datas": meta_list,
                "only_strict": self.only_strict,
            }
        )
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.base_url + "/",
                    data=payload,
                    method="POST",
                    headers={"Content-Type": "application/octet-stream"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    raw = resp.read()
                data = pickle.loads(raw)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, pickle.UnpicklingError, EOFError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    raise
                delay = min(self.retry_backoff_sec * (2**attempt), 60.0)
                logger.warning(
                    "GenEval server request failed (%s), retry in %.1fs (%d/%d)",
                    e,
                    delay,
                    attempt + 1,
                    self.max_retries,
                )
                time.sleep(delay)
        scores_f = data.get("scores")
        rewards_f = data.get("rewards")
        strict_f = data.get("strict_rewards")
        if self.reward_field == "score":
            out_list = scores_f
        elif self.reward_field == "accuracy":
            out_list = rewards_f
        else:
            out_list = strict_f
        if not isinstance(out_list, list) or len(out_list) != len(jpeg_list):
            raise RuntimeError(
                f"GenEval bad response: reward_field={self.reward_field}, "
                f"expected len {len(jpeg_list)}, got {type(out_list).__name__} len={len(out_list) if isinstance(out_list, list) else 'n/a'}"
            )
        row_scores = [float(x) for x in out_list]
        details = []
        for i in range(len(row_scores)):
            d: dict[str, Any] = {
                "reward": "geneval_remote",
                "field": self.reward_field,
                "value": row_scores[i],
                "only_strict": self.only_strict,
            }
            if isinstance(scores_f, list) and i < len(scores_f):
                d["score"] = float(scores_f[i])
            if isinstance(rewards_f, list) and i < len(rewards_f):
                d["accuracy"] = float(rewards_f[i])
            if isinstance(strict_f, list) and i < len(strict_f):
                d["strict_accuracy"] = float(strict_f[i])
            details.append(d)
        return row_scores, details

    def score_rollout_group(
        self,
        images: list[Image.Image],
        prompt_en: str,
        judge_questions: Optional[list[dict[str, Any]]],
        *,
        geneval_metadata: Optional[dict[str, Any]] = None,
    ) -> tuple[list[float], list[dict[str, Any]]]:
        del judge_questions
        del prompt_en  # prompt must be inside geneval_metadata (reward-server uses metadata['prompt'])
        n = len(images)
        if n == 0:
            return [], []
        if geneval_metadata is None:
            logger.warning(
                "GenEval: missing geneval_metadata for this sample (add JSONL field geneval / geneval_metadata); returning 0."
            )
            return [0.0] * n, [{"error": "missing_geneval_metadata"} for _ in range(n)]

        meta_one = dict(geneval_metadata)
        all_scores: list[float] = []
        all_details: list[dict[str, Any]] = []

        for start in range(0, n, self.max_batch_size):
            chunk = images[start : start + self.max_batch_size]
            jpeg_list = [_pil_to_jpeg_bytes(im) for im in chunk]
            meta_list = [dict(meta_one) for _ in chunk]
            part_s, part_d = self._post_batch(jpeg_list, meta_list)
            all_scores.extend(part_s)
            all_details.extend(part_d)

        return all_scores, all_details
