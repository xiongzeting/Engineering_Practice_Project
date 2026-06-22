"""Download MATH dataset and sample 200 multi-step problems stratified by Level 1-3."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LEVEL_QUOTAS = {1: 40, 2: 80, 3: 80}
SEED = 20260616
CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

# Filter for multi-step solutions — MATH Level 1-3 has many 1-line answers
# which aren't useful for step-level scoring evaluation.
MIN_SOLUTION_LINES = 3
ASY_RE = re.compile(r"\[asy\][\s\S]*?\[/asy\]", re.IGNORECASE)


def _parse_level(item: dict) -> int | None:
    raw = item.get("level")
    if raw is None:
        return None
    s = str(raw).strip()
    if s.isdigit():
        return int(s)
    if s.startswith("Level "):
        try:
            return int(s.split()[-1])
        except (ValueError, IndexError):
            return None
    return None


def _has_multiple_steps(solution: str) -> bool:
    cleaned = ASY_RE.sub("", solution)
    lines = [l for l in cleaned.split("\n") if l.strip()]
    return len(lines) >= MIN_SOLUTION_LINES


def main() -> None:
    by_level: dict[int, list[dict]] = {lvl: [] for lvl in LEVEL_QUOTAS}
    for cfg in CONFIGS:
        print(f"Loading config: {cfg}")
        ds = load_dataset("EleutherAI/hendrycks_math", cfg, split="train")
        for item in ds:
            lvl = _parse_level(item)
            if lvl not in LEVEL_QUOTAS:
                continue
            problem = (item.get("problem") or "").strip()
            solution = (item.get("solution") or "").strip()
            if not problem or not solution:
                continue
            if not _has_multiple_steps(solution):
                continue
            by_level[lvl].append({
                "id": f"math-l{lvl}-{len(by_level[lvl]):04d}",
                "problem": problem,
                "solution": solution,
                "level": lvl,
                "type": str(item.get("type") or cfg),
            })

    rng = random.Random(SEED)
    sampled: list[dict] = []
    for lvl, quota in LEVEL_QUOTAS.items():
        pool = by_level[lvl]
        if len(pool) < quota:
            print(f"  WARNING: level {lvl} only has {len(pool)} items, taking all")
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, quota))
        print(f"  level {lvl}: pool={len(pool)}, sampled={min(quota, len(pool))}")

    out_path = DATA_DIR / "math_raw.json"
    out_path.write_text(json.dumps(sampled, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(sampled)} samples -> {out_path}")


if __name__ == "__main__":
    main()
