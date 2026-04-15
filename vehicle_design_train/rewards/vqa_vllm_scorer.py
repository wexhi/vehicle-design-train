"""VQA log-prob scoring via OpenAI-compatible Chat Completions (e.g. vLLM serving Qwen/Qwen3.5-9B)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Optional

from PIL import Image

from vehicle_design_train.rewards.vqa_common import (
    DEFAULT_GLOBAL_TEMPLATE_EN,
    VqaRolloutGroupMixin,
    _logprob_yes_from_structure,
    _norm_answer,
    pil_to_data_url_jpeg,
)

logger = logging.getLogger(__name__)

__all__ = ["VllmOpenAiVqaProbScorer", "VllmOpenAiStructuredVqaScorer"]

STRUCTURED_VQA_SYSTEM_EN = """You are an image question-answering judge. The user will show one image and one question.
1. Reason carefully inside <Thought>...</Thought> (chain-of-thought is allowed).
2. After your thought, output exactly one final judgment as either <Answer>yes</Answer> or <Answer>no</Answer>.
Inside <Answer> use only the lowercase word yes or no. Do not add other text inside the <Answer> tag."""


def _parse_answer_yes_no(text: str) -> Optional[str]:
    """Return 'yes' or 'no' from model output, or None."""
    if not text or not text.strip():
        return None
    m = re.search(r"<Answer>\s*(yes|no)\s*</Answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).lower()
    m2 = re.search(r"<Answer>\s*([yn])\s*</Answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        return "yes" if m2.group(1).lower() == "y" else "no"
    loose = re.search(
        r"(?:^|\n)\s*(?:final\s*)?answer\s*[:：]\s*(yes|no)\s*(?:\n|$)",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if loose:
        return loose.group(1).lower()
    return None


def _binary_reward_matches_expected(parsed: Optional[str], expected_answer: str) -> float:
    if parsed is None:
        return 0.0
    exp = _norm_answer(expected_answer)
    got = _norm_answer(parsed)
    if exp == "yes":
        return 1.0 if got in ("yes", "y") else 0.0
    if exp == "no":
        return 1.0 if got in ("no", "n") else 0.0
    return 1.0 if got == exp else 0.0


def _silence_verbose_http_loggers() -> None:
    """OpenAI client uses httpx; default INFO logs every POST (noisy during VQA)."""
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _choice_text(choice: Any) -> str:
    msg = getattr(choice, "message", None)
    if msg is None:
        return ""
    c = getattr(msg, "content", None)
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
        return " ".join(parts).strip()
    return ""


def _first_token_logprob_blobs(choice: Any) -> list[dict[str, Any]]:
    """Flatten first generated token + top_logprobs into token/logprob dicts for _logprob_yes_from_structure."""
    lp = getattr(choice, "logprobs", None)
    if lp is None:
        return []
    content = getattr(lp, "content", None)
    if not content:
        return []
    first = content[0]
    items: list[dict[str, Any]] = []
    top = getattr(first, "top_logprobs", None) or []
    for t in top:
        tok = getattr(t, "token", None)
        logp = getattr(t, "logprob", None)
        if tok is not None and logp is not None:
            items.append({"token": tok, "logprob": float(logp)})
    tok = getattr(first, "token", None)
    logp = getattr(first, "logprob", None)
    if tok is not None and logp is not None:
        items.append({"token": tok, "logprob": float(logp)})
    return items


class VllmOpenAiVqaProbScorer(VqaRolloutGroupMixin):
    """
    Uses the OpenAI Python client against a vLLM (or compatible) server.
    See https://huggingface.co/Qwen/Qwen3.5-9B — image + text messages, optional ``enable_thinking: False``.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-9B",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        global_question_template_en: Optional[str] = None,
        global_weight: float = 1.0,
        judge_weight: float = 1.0,
        max_tokens: int = 8,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        retry_max_backoff_sec: float = 60.0,
        max_workers: int = 8,
        top_logprobs: int = 5,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        timeout_sec: float = 120.0,
        extra_create_kwargs: Optional[dict[str, Any]] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "vLLM/OpenAI VQA requires the `openai` package. Install with: uv pip install openai"
            ) from e

        _silence_verbose_http_loggers()
        self._client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout_sec)
        self.model = model
        self.global_question_template_en = global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
        self.global_weight = global_weight
        self.judge_weight = judge_weight
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.retry_max_backoff_sec = retry_max_backoff_sec
        self.max_workers = max(1, int(max_workers))
        self.top_logprobs = max(1, min(20, int(top_logprobs)))
        self.temperature = float(temperature)
        self.enable_thinking = bool(enable_thinking)
        self.extra_create_kwargs = dict(extra_create_kwargs or {})

    def _create(self, image: Image.Image, question: str) -> Any:
        data_url = pil_to_data_url_jpeg(image)
        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You answer only with Yes or No unless instructed otherwise.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": question},
                ],
            },
        ]

        extra_body: dict[str, Any] = dict(self.extra_create_kwargs)
        if not self.enable_thinking:
            ctk = dict(extra_body.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            extra_body["chat_template_kwargs"] = ctk

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self._client.chat.completions.create(**kwargs)

    def _call_with_retry(self, image: Image.Image, question: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._create(image, question)
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = min(self.retry_backoff_sec * (2**attempt), self.retry_max_backoff_sec)
                logger.warning("OpenAI/vLLM VQA call failed, retrying in %.1fs (%d/%d): %s", delay, attempt + 1, self.max_retries, e)
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def score_one_question(
        self,
        image: Image.Image,
        question: str,
        expected_answer: str,
        meta: str = "",
    ) -> tuple[float, dict[str, Any]]:
        try:
            resp = self._call_with_retry(image, question)
        except Exception as e:
            logger.warning("OpenAI/vLLM VQA call failed (%s): %s", meta, e)
            return 0.0, {"error": str(e), "meta": meta}

        choices = getattr(resp, "choices", None) or []
        if not choices:
            return 0.0, {"error": "no_choices", "meta": meta}

        choice = choices[0]
        text = _choice_text(choice)
        detail: dict[str, Any] = {"meta": meta, "text": text}

        blobs = _first_token_logprob_blobs(choice)
        prob = _logprob_yes_from_structure(blobs, expected_answer) if blobs else None
        if prob is not None:
            detail["prob_source"] = "openai_logprobs"
            return float(prob), detail

        logger.warning(
            "OpenAI/vLLM VQA missing usable logprobs for expected=%r (%s); text=%r",
            expected_answer,
            meta,
            text[:200],
        )
        return 0.0, detail


class VllmOpenAiStructuredVqaScorer(VqaRolloutGroupMixin):
    """
    vLLM + formatted reply (qwenvl-style tags): <Thought>...</Thought> then <Answer>yes|no</Answer>.
    Reward is 1.0 if parsed answer matches JSONL ``expected_answer`` (yes/no), else 0.0. No logprobs.
    Default ``enable_thinking=True`` for Qwen3 chat_template on vLLM.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen3.5-9B",
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "EMPTY",
        global_question_template_en: Optional[str] = None,
        global_weight: float = 1.0,
        judge_weight: float = 1.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
        retry_backoff_sec: float = 2.0,
        retry_max_backoff_sec: float = 120.0,
        max_workers: int = 8,
        temperature: float = 0.0,
        enable_thinking: bool = True,
        timeout_sec: float = 300.0,
        system_prompt_en: str = STRUCTURED_VQA_SYSTEM_EN,
        extra_create_kwargs: Optional[dict[str, Any]] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "vLLM/OpenAI VQA requires the `openai` package. Install with: uv pip install openai"
            ) from e

        _silence_verbose_http_loggers()
        self._client = OpenAI(base_url=base_url.rstrip("/"), api_key=api_key, timeout=timeout_sec)
        self.model = model
        self.global_question_template_en = global_question_template_en or DEFAULT_GLOBAL_TEMPLATE_EN
        self.global_weight = global_weight
        self.judge_weight = judge_weight
        self.max_tokens = max(64, int(max_tokens))
        self.max_retries = max_retries
        self.retry_backoff_sec = retry_backoff_sec
        self.retry_max_backoff_sec = retry_max_backoff_sec
        self.max_workers = max(1, int(max_workers))
        self.temperature = float(temperature)
        self.enable_thinking = bool(enable_thinking)
        self.system_prompt_en = system_prompt_en
        self.extra_create_kwargs = dict(extra_create_kwargs or {})

    def _create(self, image: Image.Image, question: str) -> Any:
        data_url = pil_to_data_url_jpeg(image)
        msgs: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt_en},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": question},
                ],
            },
        ]

        extra_body: dict[str, Any] = dict(self.extra_create_kwargs)
        ctk = dict(extra_body.get("chat_template_kwargs") or {})
        ctk["enable_thinking"] = self.enable_thinking
        extra_body["chat_template_kwargs"] = ctk

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body

        return self._client.chat.completions.create(**kwargs)

    def _call_with_retry(self, image: Image.Image, question: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._create(image, question)
            except Exception as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = min(self.retry_backoff_sec * (2**attempt), self.retry_max_backoff_sec)
                logger.warning(
                    "OpenAI/vLLM structured VQA failed, retry in %.1fs (%d/%d): %s",
                    delay,
                    attempt + 1,
                    self.max_retries,
                    e,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def score_one_question(
        self,
        image: Image.Image,
        question: str,
        expected_answer: str,
        meta: str = "",
    ) -> tuple[float, dict[str, Any]]:
        try:
            resp = self._call_with_retry(image, question)
        except Exception as e:
            logger.warning("OpenAI/vLLM structured VQA call failed (%s): %s", meta, e)
            return 0.0, {"error": str(e), "meta": meta}

        choices = getattr(resp, "choices", None) or []
        if not choices:
            return 0.0, {"error": "no_choices", "meta": meta}

        text = _choice_text(choices[0])
        parsed = _parse_answer_yes_no(text)
        reward = _binary_reward_matches_expected(parsed, expected_answer)
        detail: dict[str, Any] = {
            "meta": meta,
            "text": text,
            "parsed_answer": parsed,
            "expected": expected_answer,
            "prob_source": "structured_yes_no",
        }
        if parsed is None:
            detail["error"] = "unparseable_answer"
            logger.warning(
                "Structured VQA could not parse <Answer>yes|no</Answer> (%s); tail=%r",
                meta,
                text[-400:] if text else "",
            )
        return float(reward), detail
