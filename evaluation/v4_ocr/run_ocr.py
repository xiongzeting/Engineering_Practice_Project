"""Run production OCR on each MathWriting sample.

Calls vision_only_ocr (the live production OCR) per image, writes per-sample JSON.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.vision_ocr import vision_only_ocr  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "per_sample"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 4


def run_one(item: dict) -> dict:
    sid = item["id"]
    img_path = DATA_DIR.parent / item["image_path"]
    if not img_path.exists():
        return {"id": sid, "ok": False, "error": f"image missing: {img_path}"}

    image_bytes = img_path.read_bytes()

    # One retry on transient errors.
    res = None
    last_err = ""
    for attempt in range(2):
        t0 = time.perf_counter()
        try:
            res = vision_only_ocr(image_bytes)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            res = {"ok": False, "error": last_err}
        elapsed = time.perf_counter() - t0
        if res.get("ok"):
            break
        last_err = res.get("error", "unknown")
        if attempt == 0:
            time.sleep(1.5)

    return {
        "id": sid,
        "ground_truth": item["ground_truth_latex"],
        "ok": bool(res.get("ok")),
        "error": None if res.get("ok") else last_err,
        "ocr_text": res.get("ocr_text", ""),
        "notes": res.get("notes", ""),
        "engine": res.get("engine", ""),
        "elapsed_s": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    if args.limit:
        manifest = manifest[: args.limit]

    print(f"OCR'ing {len(manifest)} samples with {args.workers} workers...")
    done = 0
    failed = 0
    t_start = time.perf_counter()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, item): item for item in manifest}
        for fut in cf.as_completed(futures):
            item = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "id": item["id"],
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }

            out_path = RESULTS_DIR / f"{result['id']}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            done += 1
            if not result.get("ok"):
                failed += 1
            if done % 10 == 0 or done == len(manifest):
                elapsed = time.perf_counter() - t_start
                rate = done / max(elapsed, 0.01)
                eta = (len(manifest) - done) / max(rate, 0.01)
                print(f"  [{done}/{len(manifest)}] failed={failed} elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

    print(f"\nDone. {done} samples, {failed} failed. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
