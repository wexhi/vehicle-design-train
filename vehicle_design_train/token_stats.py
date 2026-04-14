"""Report CLIP tokenizer length stats for captions in annotation JSONL (SDXL tokenizer 1 = 77 tokens)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from vehicle_design_train.jsonl_dataset import iter_filtered_records


def main() -> None:
    p = argparse.ArgumentParser(description="Token length stats for positive_prompt_en (or --caption-field).")
    p.add_argument(
        "--annotation-jsonl",
        type=Path,
        required=True,
        help="Path to processed.jsonl",
    )
    p.add_argument(
        "--pretrained-model",
        type=Path,
        required=True,
        help="SDXL base folder (uses subfolder tokenizer/).",
    )
    p.add_argument(
        "--caption-field",
        type=str,
        default="positive_prompt_en",
        help="Key inside training_text JSON.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on rows for quick runs.",
    )
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.pretrained_model), subfolder="tokenizer", use_fast=False)

    lengths: list[int] = []
    for i, (_, cap) in enumerate(
        iter_filtered_records(
            args.annotation_jsonl,
            caption_field=args.caption_field,
        )
    ):
        if args.max_samples is not None and i >= args.max_samples:
            break
        ids = tok(cap, truncation=False, add_special_tokens=True)["input_ids"]
        lengths.append(len(ids))

    if not lengths:
        print("No captions found (check filters and paths).")
        return

    lengths.sort()
    n = len(lengths)

    def pct(q: float) -> int:
        return lengths[int(q * (n - 1))]

    over_77 = sum(1 for L in lengths if L > 77)
    summary = {
        "count": n,
        "tokenizer_subfolder": "tokenizer",
        "model_max_length": tok.model_max_length,
        "min_len": lengths[0],
        "p50_len": pct(0.5),
        "p90_len": pct(0.9),
        "p99_len": pct(0.99),
        "max_len": lengths[-1],
        "truncated_if_clip_77": over_77,
        "fraction_over_77": round(over_77 / n, 4),
        "note": "SDXL also uses tokenizer_2 (OpenCLIP); this stats primary CLIP only.",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    bucket = Counter()
    for L in lengths:
        if L <= 64:
            bucket["<=64"] += 1
        elif L <= 77:
            bucket["65-77"] += 1
        elif L <= 128:
            bucket["78-128"] += 1
        else:
            bucket[">128"] += 1
    print("buckets:", dict(bucket))


if __name__ == "__main__":
    main()
