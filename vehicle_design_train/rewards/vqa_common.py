"""Shared VQA probability helpers and rollout-group scheduling for DashScope / vLLM backends."""

from __future__ import annotations

import base64
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_GLOBAL_TEMPLATE_EN = (
    'Does this image reflect the following description: "{prompt}"? Please answer yes or no.'
)


def pil_to_data_url_jpeg(image: Image.Image, quality: int = 92) -> str:
    rgb = image.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _norm_answer(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _expected_tokens(expected: str) -> list[str]:
    e = _norm_answer(expected)
    if e == "yes":
        return ["yes", "y"]
    if e == "no":
        return ["no", "n"]
    return [e]


# Normalized token ids for binary Yes/No VQA; used so the denominator includes both classes.
_YES_NO_POOL = frozenset({"yes", "y", "no", "n"})


def _is_binary_yes_no_expected(expected: str) -> bool:
    return _norm_answer(expected) in _YES_NO_POOL


def _logprob_yes_from_structure(obj: Any, expected: str) -> Optional[float]:
    """Read P(expected class | top-logprob support) from nested token/logprob blobs.

    For yes/no questions, the denominator must include both affirmative and negative
    tokens that appear in the API's top_logprobs; otherwise rel ⊆ want makes num == z
    and the score is always 1.0.
    """
    import math

    want_list = _expected_tokens(expected)
    want_set = set(want_list)
    found: list[tuple[str, float]] = []

    def walk(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, dict):
            if "token" in x and "logprob" in x:
                try:
                    found.append((str(x["token"]), float(x["logprob"])))
                except (TypeError, ValueError):
                    pass
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    if not found:
        return None
    by_tok: dict[str, float] = {}
    for t, lp in found:
        key = _norm_answer(t)
        by_tok[key] = max(by_tok.get(key, float("-inf")), lp)

    if _is_binary_yes_no_expected(expected):
        rel = [k for k in by_tok if k in _YES_NO_POOL]
    else:
        rel = [k for k in by_tok if k in want_set]
    if not rel:
        rel = list(by_tok.keys())
    m = max(by_tok[k] for k in rel)
    exps = {k: math.exp(by_tok[k] - m) for k in rel}
    z = sum(exps.values())
    num = sum(exps[k] for k in rel if k in want_set)
    if num == 0.0:
        return None
    return float(num / z)


class VqaRolloutGroupMixin:
    """Requires: global_question_template_en, global_weight, judge_weight, max_workers, score_one_question."""

    global_question_template_en: str
    global_weight: float
    judge_weight: float
    max_workers: int

    def _weighted_vqa_from_details(
        self,
        details: dict[str, Any],
        judges: list[dict[str, Any]],
    ) -> float:
        items: list[tuple[float, float, str]] = []
        g_det = details.get("global")
        if not isinstance(g_det, dict):
            g_det = {}
        items.append((float(g_det.get("_prob", 0.0)), self.global_weight, "global"))
        for i, jq in enumerate(judges):
            q = jq.get("question_en") or jq.get("question")
            if not q:
                continue
            jd = details["judges"][i]
            if isinstance(jd, dict) and jd.get("skipped"):
                continue
            if not isinstance(jd, dict):
                items.append((0.0, self.judge_weight, f"judge_{i}"))
            else:
                items.append((float(jd.get("_prob", 0.0)), self.judge_weight, f"judge_{i}"))
        wsum = sum(w for _, w, _ in items)
        if wsum <= 0:
            return 0.0
        return float(sum(p * w for p, w, _ in items) / wsum)

    def _strip_prob_from_details(self, d: dict[str, Any]) -> None:
        d.pop("_prob", None)

    def score_rollout_group(
        self,
        images: list[Image.Image],
        prompt_en: str,
        judge_questions: Optional[list[dict[str, Any]]],
    ) -> tuple[list[float], list[dict[str, Any]]]:
        judges = list(judge_questions or [])
        global_q = self.global_question_template_en.format(prompt=prompt_en)
        n_img = len(images)

        for i, jq in enumerate(judges):
            q = jq.get("question_en") or jq.get("question")
            if not q:
                logger.warning("Skipping judge %d: missing question_en", i)

        details_list: list[dict[str, Any]] = []
        for _gi in range(n_img):
            jslots: list[Any] = []
            for i, jq in enumerate(judges):
                q = jq.get("question_en") or jq.get("question")
                if not q:
                    jslots.append({"skipped": True})
                else:
                    jslots.append(None)
            details_list.append({"global": None, "judges": jslots})

        tasks: list[tuple[int, str, int, Image.Image, str, str, str]] = []
        for gi, im in enumerate(images):
            tasks.append((gi, "global", -1, im, global_q, "yes", "global"))
            for i, jq in enumerate(judges):
                q = jq.get("question_en") or jq.get("question")
                if not q:
                    continue
                exp = str(jq.get("expected_answer", "yes")).strip()
                tasks.append((gi, "judge", i, im, q, exp, f"judge_{i}"))

        def run_task(
            t: tuple[int, str, int, Image.Image, str, str, str],
        ) -> tuple[int, str, int, float, dict[str, Any]]:
            gi, kind, ji, im, q, exp, meta = t
            prob, det = self.score_one_question(im, q, exp, meta=meta)
            det = dict(det)
            det["_prob"] = prob
            return gi, kind, ji, prob, det

        if self.max_workers <= 1:
            results = [run_task(t) for t in tasks]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=min(self.max_workers, max(1, len(tasks)))) as ex:
                futs = {ex.submit(run_task, t): t for t in tasks}
                for fut in as_completed(futs):
                    results.append(fut.result())

        for gi, kind, ji, _prob, det in results:
            if kind == "global":
                details_list[gi]["global"] = det
            else:
                details_list[gi]["judges"][ji] = det

        scores: list[float] = []
        for gi in range(n_img):
            d = details_list[gi]
            if d["global"] is None:
                d["global"] = {"error": "missing_global", "_prob": 0.0}
            for i, slot in enumerate(d["judges"]):
                q = judges[i].get("question_en") or judges[i].get("question")
                if not q:
                    continue
                if slot is None:
                    d["judges"][i] = {"error": "missing_judge_result", "_prob": 0.0}
            scores.append(self._weighted_vqa_from_details(d, judges))

        for d in details_list:
            self._strip_prob_from_details(d.get("global") or {})
            for j in d.get("judges") or []:
                if isinstance(j, dict):
                    self._strip_prob_from_details(j)

        return scores, details_list

    def score_sample(
        self,
        image: Image.Image,
        prompt_en: str,
        judge_questions: Optional[list[dict[str, Any]]],
    ) -> tuple[float, dict[str, Any]]:
        scores, details_list = self.score_rollout_group([image], prompt_en, judge_questions)
        return scores[0], details_list[0]
