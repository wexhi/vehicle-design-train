"""Generate a small grid of images from a JSON prompt list (base SDXL + optional LoRA)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="SDXL inference grid from eval_prompts.json.")
    ap.add_argument("--pretrained-model", type=Path, required=True, help="Local SDXL diffusers folder.")
    ap.add_argument("--lora-path", type=Path, default=None, help="Directory containing saved LoRA weights.")
    ap.add_argument("--prompts-json", type=Path, required=True, help="JSON file: array of strings or {\"prompts\":[...]}.")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for PNGs.")
    ap.add_argument("--num-per-prompt", type=int, default=1, help="Images per prompt.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
        help="Pipeline torch dtype (bf16 recommended on Ampere+).",
    )
    args = ap.parse_args()

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    raw = json.loads(args.prompts_json.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        prompts = [str(x) for x in raw]
    else:
        prompts = [str(x) for x in raw["prompts"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(args.pretrained_model),
        torch_dtype=torch_dtype,
    )
    if args.lora_path is not None:
        pipe.load_lora_weights(str(args.lora_path))

    g = torch.Generator(device=pipe.device).manual_seed(args.seed)
    for i, prompt in enumerate(prompts):
        for k in range(args.num_per_prompt):
            seed = args.seed + i * 1000 + k
            g.manual_seed(seed)
            image = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                generator=g,
            ).images[0]
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
            path = args.out_dir / f"{i:03d}_{k}_{safe}.png"
            image.save(path)
            print("saved", path)


if __name__ == "__main__":
    main()
