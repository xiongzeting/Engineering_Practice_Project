"""v2 detection runner: concurrent execution with retry-aware detector.

For each injected problem, run the binary detector and write per-problem JSON.
Failures (after 3 retries) are recorded with ok=False; metrics counts them.
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

from evaluation.v2_detection.detector import detect_errors  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "per_problem"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 4


def run_one(problem: dict) -> dict:
    pid = problem["id"]
    t0 = time.perf_counter()
    res = detect_errors(problem)
    elapsed = time.perf_counter() - t0

    return {
        "id": pid,
        "elapsed_s": round(elapsed, 2),
        "ok": res.get("ok", False),
        "error": res.get("error") if not res.get("ok") else None,
        "verdict": res.get("verdict"),
        "ground_truth": {
            "is_control": problem.get("is_control", False),
            "has_error": problem.get("has_error", False),
            "corruption_type": problem.get("corruption_type"),
            "corruption_index": problem.get("corruption_index"),  # 0-indexed in inject
        },
        "problem_text": problem.get("problem"),
        "steps": problem.get("steps"),
        "attempts": res.get("attempts"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = json.loads((DATA_DIR / "math_injected.json").read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    print(f"Detecting on {len(data)} problems with {args.workers} workers...")
    done = 0
    failed = 0
    t_start = time.perf_counter()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, p): p for p in data}
        for fut in cf.as_completed(futures):
            problem = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "id": problem["id"],
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "verdict": None,
                    "ground_truth": {
                        "is_control": problem.get("is_control", False),
                        "has_error": problem.get("has_error", False),
                        "corruption_type": problem.get("corruption_type"),
                        "corruption_index": problem.get("corruption_index"),
                    },
                }

            out_path = RESULTS_DIR / f"{result['id']}.json"
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            done += 1
            if not result.get("ok"):
                failed += 1
            if done % 10 == 0 or done == len(data):
                elapsed = time.perf_counter() - t_start
                rate = done / max(elapsed, 0.01)
                eta = (len(data) - done) / max(rate, 0.01)
                print(f"  [{done}/{len(data)}] failed={failed} elapsed={elapsed:.0f}s rate={rate:.2f}/s eta={eta:.0f}s")

    print(f"\nDone. {done} problems, {failed} failed. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
