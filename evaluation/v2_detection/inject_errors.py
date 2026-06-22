"""v2 error injection: no delete type (design fix — deletion isn't a real error
since the remaining steps are still mathematically correct).

Types: numeric / sign / variable / insert.
Control ratio: 15%.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 20260617  # same seed as v1 for reproducibility

CONTROL_RATIO = 0.15
INJECT_WEIGHTS = {
    "numeric": 0.45,
    "sign": 0.30,
    "variable": 0.10,
    "insert": 0.15,
}

NON_STEP_PATTERNS = [
    re.compile(r"\[asy\]", re.IGNORECASE),
    re.compile(r"\[/asy\]", re.IGNORECASE),
]
MATH_HINT_TOKENS = ["=", "\\frac", "\\sqrt", "\\cdot", "\\boxed", "\\sum", "\\int",
                    "\\Rightarrow", "\\implies", "\\to", "^", "\\alpha", "\\beta", "\\theta", "\\pi"]


def is_math_step(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3 or len(s) > 500:
        return False
    for p in NON_STEP_PATTERNS:
        if p.search(s):
            return False
    if "$" in s and s[-1] != "$":
        return True
    return any(tok in s for tok in MATH_HINT_TOKENS)


def clean_solution(sol: str) -> str:
    return re.sub(r"\[asy\][\s\S]*?\[/asy\]", "", sol, flags=re.IGNORECASE)


def split_into_steps(solution: str) -> list[str]:
    cleaned = clean_solution(solution)
    out = []
    for raw in cleaned.split("\n"):
        s = raw.strip()
        if s and is_math_step(s):
            out.append(s)
    return out


def inject_numeric(step: str, rng: random.Random) -> str:
    nums = re.findall(r"\d+", step)
    if not nums:
        return step
    target = rng.choice(nums)
    n = int(target)
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, -5])
    new_n = max(0, n + delta)
    return step.replace(target, str(new_n), 1)


def inject_sign(step: str, rng: random.Random) -> str:
    matches = list(re.finditer(r"([a-zA-Z0-9\}\])\\]+)\s*([+\-])\s*([a-zA-Z0-9\\{]+)", step))
    if not matches:
        return step
    m = rng.choice(matches)
    new_sign = "-" if m.group(2) == "+" else "+"
    return step[: m.start(2)] + new_sign + step[m.end(2):]


def inject_variable(step: str, rng: random.Random) -> str:
    candidates = re.findall(r"(?<![a-zA-Z\\])([a-zA-Z])(?![a-zA-Z])", step)
    candidates = [c for c in candidates if c not in {"e", "i", "j"}]
    if not candidates:
        return step
    old = rng.choice(candidates)
    used = set(re.findall(r"\\([a-zA-Z]+)", step))
    new = rng.choice([c for c in "xyzabcdfghkmnpqrstuvw" if c != old and c not in used])
    pattern = re.compile(rf"(?<![a-zA-Z\\]){re.escape(old)}(?![a-zA-Z])")
    return pattern.sub(new, step, count=1)


INSERT_TEMPLATES = [
    "Therefore $\\boxed{0}$.",
    "So we have $1 = 2$.",
    "This gives $x = \\frac{0}{0}$.",
    "Hence $\\sqrt{-1} = 1$.",
    "Thus $2 + 2 = 5$.",
    "Therefore $\\boxed{\\text{undefined}}$.",
]


def inject_insert(steps: list[str], idx: int, rng: random.Random) -> tuple[list[str], int]:
    wrong = rng.choice(INSERT_TEMPLATES)
    return steps[:idx] + [wrong] + steps[idx:], idx


def build_injected(problem: dict, rng: random.Random) -> dict:
    steps_raw = split_into_steps(problem["solution"])

    is_control = rng.random() < CONTROL_RATIO or len(steps_raw) < 2
    if is_control:
        return {
            **problem,
            "steps": [{"text": t, "corrupted": False, "corruption_type": None} for t in steps_raw],
            "has_error": False,
            "is_control": True,
        }

    types = list(INJECT_WEIGHTS.keys())
    rng.shuffle(types)
    for error_type in types:
        idx = rng.randrange(len(steps_raw))

        if error_type == "insert":
            new_texts, wrong_idx = inject_insert(steps_raw, idx, rng)
            return {
                **problem,
                "steps": [{"text": t, "corrupted": i == wrong_idx, "corruption_type": ("insert" if i == wrong_idx else None)}
                          for i, t in enumerate(new_texts)],
                "has_error": True,
                "is_control": False,
                "corruption_type": "insert",
                "corruption_index": wrong_idx,
            }

        target_text = steps_raw[idx]
        if error_type == "numeric":
            new_text = inject_numeric(target_text, rng)
        elif error_type == "sign":
            new_text = inject_sign(target_text, rng)
        elif error_type == "variable":
            new_text = inject_variable(target_text, rng)
        else:
            new_text = target_text

        if new_text != target_text:
            steps_out = [{"text": t, "corrupted": False, "corruption_type": None} for t in steps_raw]
            steps_out[idx] = {"text": new_text, "corrupted": True, "corruption_type": error_type}
            return {
                **problem,
                "steps": steps_out,
                "has_error": True,
                "is_control": False,
                "corruption_type": error_type,
                "corruption_index": idx,
            }

    return {
        **problem,
        "steps": [{"text": t, "corrupted": False, "corruption_type": None} for t in steps_raw],
        "has_error": False,
        "is_control": True,
        "corruption_type": "noop",
    }


def main() -> None:
    data = json.loads((DATA_DIR / "math_raw.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    out: list[dict] = []
    counts: dict[str, int] = {}
    for prob in data:
        item = build_injected(prob, rng)
        out.append(item)
        key = "control" if item.get("is_control") else item.get("corruption_type", "?")
        counts[key] = counts.get(key, 0) + 1

    out_path = DATA_DIR / "math_injected.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    step_counts = [len(x["steps"]) for x in out]
    print(f"Wrote {len(out)} problems -> {out_path}")
    print(f"Steps: min={min(step_counts)} max={max(step_counts)} avg={sum(step_counts)/len(step_counts):.1f}")
    print(f"Type distribution: {counts}")


if __name__ == "__main__":
    main()
