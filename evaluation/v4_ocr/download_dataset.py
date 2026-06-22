"""Download 100 handwritten math samples from MathWriting (HF mirror).

Output:
    data/images/mw-0000.png ... mw-0099.png
    data/manifest.json — [{id, image_path, ground_truth_latex}, ...]
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "deepcopy/MathWriting-human"
N = 100
SEED = 20260618


def main() -> None:
    print(f"Loading {DATASET_ID} (this may pull a few hundred MB on first run)...")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    ds = ds.shuffle(seed=SEED)

    manifest: list[dict] = []
    for i, item in enumerate(ds):
        if i >= N:
            break
        # MathWriting-human field names: 'image' (PIL) and 'latex' or 'formula' or 'text'.
        # Try the common keys.
        img = item.get("image")
        if not isinstance(img, Image.Image):
            print(f"  [{i}] no image, skipping")
            continue
        gt = (
            item.get("latex")
            or item.get("formula")
            or item.get("text")
            or item.get("ground_truth")
            or ""
        )
        gt = str(gt).strip()
        if not gt:
            print(f"  [{i}] empty ground truth, skipping")
            continue

        sample_id = f"mw-{i:04d}"
        img_path = IMG_DIR / f"{sample_id}.png"
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, format="PNG")
        manifest.append(
            {
                "id": sample_id,
                "image_path": str(img_path.relative_to(DATA_DIR.parent)),
                "ground_truth_latex": gt,
            }
        )

    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved {len(manifest)} samples -> {manifest_path}")
    print(f"First 3 ground truths:")
    for m in manifest[:3]:
        print(f"  {m['id']}: {m['ground_truth_latex']}")


if __name__ == "__main__":
    main()
