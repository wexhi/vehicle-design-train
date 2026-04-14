"""Build a Hugging Face `datasets.Dataset` from `processed.jsonl` annotations."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

from datasets import Dataset, Features, Image, Value


def _parse_training_text(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        return json.loads(raw)
    raise TypeError(f"training_text must be dict or JSON str, got {type(raw)}")


def _rewrite_path(p: str, prefix_old: str | None, prefix_new: str | None) -> str:
    if prefix_old and prefix_new and p.startswith(prefix_old):
        return prefix_new + p[len(prefix_old) :]
    return p


def iter_filtered_records(
    jsonl_path: str | os.PathLike[str],
    *,
    path_prefix_old: str | None = None,
    path_prefix_new: str | None = None,
    require_single_car: bool = True,
    require_zero_persons: bool = True,
    require_complete_vehicle: bool = False,
    caption_field: str = "positive_prompt_en",
) -> Iterator[tuple[str, str]]:
    """Yield (absolute_image_path, caption) for each kept row."""
    path = Path(jsonl_path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            cleaning = row.get("cleaning") or {}
            if require_single_car and cleaning.get("car_count") != 1:
                continue
            if require_zero_persons and cleaning.get("person_count", 0) != 0:
                continue
            core = row.get("core_annotation") or {}
            if require_complete_vehicle and not core.get("has_complete_vehicle", False):
                continue
            tt = _parse_training_text(row.get("training_text"))
            cap = tt.get(caption_field)
            if not cap or not str(cap).strip():
                continue
            img = row.get("image_path")
            if not img:
                continue
            img = _rewrite_path(str(img), path_prefix_old, path_prefix_new)
            if not Path(img).is_file():
                continue
            yield img, str(cap).strip()


def dataset_from_annotation_jsonl(
    jsonl_path: str | os.PathLike[str],
    *,
    path_prefix_old: str | None = None,
    path_prefix_new: str | None = None,
    require_single_car: bool = True,
    require_zero_persons: bool = True,
    require_complete_vehicle: bool = False,
    caption_field: str = "positive_prompt_en",
    seed: int = 42,
    train_ratio: float = 1.0,
) -> Dataset:
    """
    Build a dataset with columns `image` (HF Image) and `text` (caption).

    If train_ratio < 1, splits into train/validation by shuffled indices (returns train only
    when train_ratio==1). For training script we only expose the train split via caller.
    """
    records = list(
        iter_filtered_records(
            jsonl_path,
            path_prefix_old=path_prefix_old,
            path_prefix_new=path_prefix_new,
            require_single_car=require_single_car,
            require_zero_persons=require_zero_persons,
            require_complete_vehicle=require_complete_vehicle,
            caption_field=caption_field,
        )
    )

    def gen() -> Iterator[dict[str, Any]]:
        for img_path, text in records:
            yield {"image": img_path, "text": text}

    features = Features({"image": Image(), "text": Value("string")})
    ds = Dataset.from_generator(gen, features=features)
    if train_ratio < 1.0:
        split = ds.train_test_split(test_size=1.0 - train_ratio, seed=seed)
        return split["train"]
    return ds
