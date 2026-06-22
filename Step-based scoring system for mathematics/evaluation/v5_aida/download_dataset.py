"""Download 100 handwritten math samples from Aida-Calculus (HF mirror)."""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "deepcopy/Aida-Calculus-Math-Handwriting"
N = 100
SEED = 20260619  # different seed to avoid overlap with CROHME/MathWriting

GT_KEYS = ("latex", "equation", "formula", "text", "ground_truth")


def main() -> None:
    print(f"Loading {DATASET_ID} ...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED)

    manifest: list[dict] = []
    skipped = 0
    for i, item in enumerate(ds):
        if len(manifest) >= N:
            break
        img = item.get("image")
        if not isinstance(img, Image.Image):
            skipped += 1
            continue
        gt = ""
        for k in GT_KEYS:
            v = item.get(k)
            if v:
                gt = str(v).strip()
                break
        if not gt:
            skipped += 1
            continue

        sid = f"ai-{len(manifest):04d}"
        img_path = IMG_DIR / f"{sid}.png"
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, format="PNG")
        manifest.append(
            {
                "id": sid,
                "image_path": str(img_path.relative_to(DATA_DIR.parent)),
                "ground_truth_latex": gt,
            }
        )

    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} samples (skipped {skipped}) -> {out}")
    for m in manifest[:3]:
        print(f"  {m['id']}: {m['ground_truth_latex'][:80]}")


if __name__ == "__main__":
    main()
