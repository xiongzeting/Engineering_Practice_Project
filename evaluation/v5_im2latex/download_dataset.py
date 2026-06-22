"""Download 100 printed-formula samples from IM2LATEX-100K (test split).

Printed (not handwritten) — establishes MiMo's recognition ceiling on clean
typeset formulas, decoupled from handwriting noise.
"""
from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

DATASET_ID = "yuntian-deng/im2latex-100k"
SPLIT = "test"
N = 100
SEED = 20260620


def main() -> None:
    print(f"Loading {DATASET_ID} [{SPLIT}] ...")
    ds = load_dataset(DATASET_ID, split=SPLIT, streaming=True)
    ds = ds.shuffle(seed=SEED)

    manifest: list[dict] = []
    for item in ds:
        if len(manifest) >= N:
            break
        img = item.get("image")
        formula = str(item.get("formula", "") or "").strip()
        if not isinstance(img, Image.Image) or not formula:
            continue
        sid = f"im-{len(manifest):04d}"
        img_path = IMG_DIR / f"{sid}.png"
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path, format="PNG")
        manifest.append(
            {
                "id": sid,
                "image_path": str(img_path.relative_to(DATA_DIR.parent)),
                "ground_truth_latex": formula,
            }
        )

    out = DATA_DIR / "manifest.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(manifest)} samples -> {out}")
    for m in manifest[:3]:
        print(f"  {m['id']}: {m['ground_truth_latex'][:80]}")


if __name__ == "__main__":
    main()
