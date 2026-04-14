"""Optional metrics: CLIP image-text similarity and notes on FID (heavy deps)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def main() -> None:
    ap = argparse.ArgumentParser(
        description="CLIP cosine similarity between generated images and their prompts (optional eval)."
    )
    ap.add_argument("--images-dir", type=Path, required=True, help="Directory of PNG/JPG outputs.")
    ap.add_argument("--pretrained-clip", type=str, default="openai/clip-vit-large-patch14", help="HF CLIP id.")
    ap.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="Optional JSON: list of {filename, prompt} aligned to generations.",
    )
    args = ap.parse_args()

    try:
        model = CLIPModel.from_pretrained(args.pretrained_clip)
        proc = CLIPProcessor.from_pretrained(args.pretrained_clip)
    except Exception as e:
        print("Failed to load CLIP:", e)
        raise SystemExit(1) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    manifest = None
    if args.manifest_json and args.manifest_json.is_file():
        manifest = json.loads(args.manifest_json.read_text(encoding="utf-8"))

    if manifest:
        pairs = [(item["filename"], item["prompt"]) for item in manifest]
    else:
        # Filename-only mode: use stem as prompt (weak; prefer manifest).
        pairs = [(p.name, p.stem.replace("_", " ")) for p in sorted(args.images_dir.glob("*")) if p.suffix.lower() in (".png", ".jpg", ".jpeg")]

    sims: list[float] = []
    for fname, prompt in pairs:
        path = args.images_dir / fname if not str(fname).startswith("/") else Path(fname)
        if not path.is_file():
            continue
        img = Image.open(path).convert("RGB")
        inputs = proc(text=[prompt], images=[img], return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
            # image_embeds and text_embeds are normalized in some versions; use logits_per_image
            logits = out.logits_per_image.squeeze(0).item()
            sims.append(float(logits))

    if not sims:
        print("No image/prompt pairs scored.")
        return

    mean_s = sum(sims) / len(sims)
    print(json.dumps({"count": len(sims), "mean_clip_logit_scale_diag": round(mean_s, 4)}, indent=2))
    print(
        "FID: not computed here (install pytorch-fid or torchmetrics and provide a reference folder). "
        "See https://github.com/mseitzer/pytorch-fid"
    )


if __name__ == "__main__":
    main()
