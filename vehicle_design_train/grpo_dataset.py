from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


@dataclass
class GrpoJsonlSample:
    prompt_en: str
    judge_questions: list[dict[str, Any]]
    meta: dict[str, Any]


def iter_grpo_jsonl(
    path: str | Path,
    prompt_field: str = "prompt_en",
    max_samples: Optional[int] = None,
) -> Iterator[GrpoJsonlSample]:
    path = Path(path)
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("Skip line %d: JSON error %s", line_no, e)
                continue
            prompt = obj.get(prompt_field)
            if not prompt or not isinstance(prompt, str):
                logger.warning("Skip line %d: missing %s", line_no, prompt_field)
                continue
            jr = obj.get("judge_requirements") or {}
            judges = jr.get("judge_questions") or []
            if not isinstance(judges, list):
                judges = []
            meta = {k: obj.get(k) for k in ("sample_id", "uuid", "task_group", "dataset_version")}
            yield GrpoJsonlSample(prompt_en=prompt, judge_questions=judges, meta={k: v for k, v in meta.items() if v})
            n += 1
            if max_samples is not None and n >= max_samples:
                break


def list_grpo_jsonl(path: str | Path, **kwargs) -> list[GrpoJsonlSample]:
    return list(iter_grpo_jsonl(path, **kwargs))
