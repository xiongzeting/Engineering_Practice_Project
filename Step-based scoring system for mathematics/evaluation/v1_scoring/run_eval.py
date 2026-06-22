"""Run DeepSeek step-scoring on injected MATH problems.

For each problem, treats the problem statement as the question and asks the
scorer to grade each step. Results saved as per-problem JSON for later metrics.

Concurrency: 4 threads (configurable via --workers).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import time
from pathlib import Path

# Project root on sys.path so app.* imports work.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.schemas import StepItem
from app.services.scorer import _llm_score_steps

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "per_problem"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_SCORE = 20  # per-problem max; per-step allocation by LLM
WORKERS = 4


def build_step_items(steps: list[dict]) -> list[StepItem]:
    return [
        StepItem(index=i + 1, raw=s["text"], normalized=s["text"], has_equation=True, confidence=0.9)
        for i, s in enumerate(steps)
    ]


def grade_one(problem: dict) -> dict:
    pid = problem["id"]
    steps = problem.get("steps") or []
    if not steps:
        return {"id": pid, "ok": False, "error": "no steps"}

    step_items = build_step_items(steps)
    ocr_text = problem.get("problem", "") + "\n\n" + "\n".join(s["text"] for s in steps)

    t0 = time.perf_counter()
    res = _llm_score_steps(
        ocr_text=ocr_text,
        steps=step_items,
        reference=problem.get("problem"),
        max_score=float(MAX_SCORE),
    )
    elapsed = time.perf_counter() - t0

    if not res.get("ok"):
        return {"id": pid, "ok": False, "error": res.get("error", "unknown"), "elapsed_s": elapsed}

    return {
        "id": pid,
        "ok": True,
        "scores": [s.model_dump() for s in res["scores"]],
        "total": res.get("total"),
        "feedback": res.get("feedback"),
        "elapsed_s": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--limit", type=int, default=0, help="limit N problems for smoke test")
    args = parser.parse_args()

    data = json.loads((DATA_DIR / "math_injected.json").read_text(encoding="utf-8"))
    if args.limit:
        data = data[: args.limit]

    print(f"Scoring {len(data)} problems with {args.workers} workers...")
    done = 0
    failed = 0
    t_start = time.perf_counter()

    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_to_problem = {pool.submit(grade_one, p): p for p in data}
        for fut in cf.as_completed(future_to_problem):
            problem = future_to_problem[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = {"id": problem["id"], "ok": False, "error": f"{type(e).__name__}: {e}"}

            # Merge problem metadata (steps, corruption info) with scoring result.
            merged = {
                **{k: v for k, v in problem.items() if k != "steps"},
                "problem_text": problem["problem"],
                "solution": problem["solution"],
                "steps_meta": problem["steps"],
                "scoring": result,
            }
            out_path = RESULTS_DIR / f"{problem['id']}.json"
            out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

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
