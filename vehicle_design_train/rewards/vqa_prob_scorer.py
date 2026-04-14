from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional

from PIL import Image

from vehicle_design_train.rewards.vqa_common import (
    DEFAULT_GLOBAL_TEMPLATE_EN,
    VqaRolloutGroupMixin,
    _logprob_yes_from_structure,
    pil_to_data_url_jpeg,
)

logger = logging.getLogger(__name__)

# Back-compat re-exports
__all__ = ["DEFAULT_GLOBAL_TEMPLATE_EN", "DashScopeVqaProbScorer"]


def _message_content(resp: Any) -> str:
    out = getattr(resp, "output", None)
    if out is None:
        return ""
    choices = getattr(out, "choices", None) or (out.get("choices") if isinstance(out, dict) else None)
    if not choices:
        return str(getattr(out, "text", "") or (out.get("text") if isinstance(out, dict) else "") or "")
    c0 = choices[0]
    if isinstance(c0, dict):
        msg = c0.get("message") or {}
        content = msg.get("content")
    else:
        msg = getattr(c0, "message", None)
        content = getattr(msg, "content", None) if msg is not None else None
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts).strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def _choice_logprob_blob(resp: Any) -> Any:
    out = getattr(resp, "output", None)
    if out is None:
        return None
    choices = getattr(out, "choices", None) or (out.get("choices") if isinstance(out, dict) else None)
    if not choices:
        return getattr(out, "logprobs", None) or (out.get("logprobs") if isinstance(out, dict) else None)
    c0 = choices[0]
    if isinstance(c0, dict):
        return (
            c0.get("logprobs")
            or c0.get("message", {}).get("logprobs")
            or c0.get("content", {}).get("logprobs")
        )
    return getattr(c0, "logprobs", None)


class DashScopeVqaProbScorer(VqaRolloutGroupMixin):
    """VQAScore-style scores via DashScope multimodal API (user-confirmed logprobs available)."""

    def __init__(
        self,
        model: str = "qwen3.5-35b-a3b",
        api_key: Optional[str] = None,
        global_question_template_en: Optional[str] = None,
        global_weight: float = 1.0,
        judge_weight: float = 1.0,
        max_tokens: int = 8,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        retry_max_backoff_sec: float = 30.0,
        extra_call_kwargs: Optional[dict[str, Any]] = None,
        request_logprobs: bool = True,
        top_logprobs: int = 5,
        max_workers: int = 8,
    ):
        from dashscope import MultiModalConversation

        self._mmc = MultiModalConversation
        self.model = model
        self.api_key = api_key
        self.global_question_template_en = global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
        self.global_weight = global_weight
        self.judge_weight = judge_weight
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.retry_max_backoff_sec = retry_max_backoff_sec
        self.extra_call_kwargs = extra_call_kwargs or {}
        self.request_logprobs = request_logprobs
        self.top_logprobs = max(0, min(5, int(top_logprobs)))
        self.max_workers = max(1, int(max_workers))

    def _call_once(self, image: Image.Image, question: str) -> Any:
        data_url = pil_to_data_url_jpeg(image)
        messages = [
            {
                "role": "system",
                "content": [{"text": "You answer only with Yes or No unless instructed otherwise."}],
            },
            {
                "role": "user",
                "content": [
                    {"image": data_url},
                    {"text": question},
                ],
            },
        ]
        kwargs: dict[str, Any] = {"max_tokens": self.max_tokens}
        if self.request_logprobs:
            kwargs["result_format"] = "message"
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = self.top_logprobs
        kwargs.update(self.extra_call_kwargs)
        return self._mmc.call(model=self.model, messages=messages, api_key=self.api_key, **kwargs)

    def _should_retry_status(self, status_code: Optional[int]) -> bool:
        if status_code is None:
            return True
        return status_code in {408, 409, 429} or status_code >= 500

    def _call(self, image: Image.Image, question: str) -> Any:
        last_exc: Exception | None = None
        last_resp: Any = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._call_once(image, question)
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = min(self.retry_backoff_sec * (2**attempt), self.retry_max_backoff_sec)
                logger.warning("DashScope VQA call failed, retrying in %.1fs (%d/%d): %s", delay, attempt + 1, self.max_retries, e)
                time.sleep(delay)
                continue

            status_code = getattr(resp, "status_code", 200)
            if status_code == 200 or not self._should_retry_status(status_code):
                return resp

            last_resp = resp
            if attempt >= self.max_retries:
                break
            delay = min(self.retry_backoff_sec * (2**attempt), self.retry_max_backoff_sec)
            logger.warning(
                "DashScope VQA returned retryable status=%s, retrying in %.1fs (%d/%d)",
                status_code,
                delay,
                attempt + 1,
                self.max_retries,
            )
            time.sleep(delay)

        if last_exc is not None:
            raise last_exc
        return last_resp

    def score_one_question(
        self,
        image: Image.Image,
        question: str,
        expected_answer: str,
        meta: str = "",
    ) -> tuple[float, dict[str, Any]]:
        try:
            resp = self._call(image, question)
        except Exception as e:
            logger.warning("DashScope VQA call failed (%s): %s", meta, e)
            return 0.0, {"error": str(e), "meta": meta}

        if getattr(resp, "status_code", 200) != 200:
            logger.warning(
                "DashScope VQA non-OK (%s): code=%s msg=%s",
                meta,
                getattr(resp, "code", ""),
                getattr(resp, "message", ""),
            )
            return 0.0, {"error": "api_status", "meta": meta}

        blob = _choice_logprob_blob(resp)
        prob = None
        if blob is not None:
            prob = _logprob_yes_from_structure(blob, expected_answer)
        detail: dict[str, Any] = {"meta": meta, "text": _message_content(resp)}
        if prob is not None:
            detail["prob_source"] = "logprobs"
            return float(prob), detail

        try:
            raw = json.loads(json.dumps(resp.output, default=lambda o: o.__dict__ if hasattr(o, "__dict__") else str(o)))
            prob = _logprob_yes_from_structure(raw, expected_answer)
        except Exception:
            prob = None
        if prob is not None:
            detail["prob_source"] = "logprobs_nested"
            return float(prob), detail

        logger.warning(
            "DashScope VQA missing usable logprobs for expected=%r (%s); text=%r",
            expected_answer,
            meta,
            detail["text"][:200],
        )
        return 0.0, detail
