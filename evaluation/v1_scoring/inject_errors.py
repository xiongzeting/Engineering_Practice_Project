"""Split MATH solutions into steps and inject synthetic errors.

Output: evaluation/data/math_injected.json — list of problems, each with
`steps` (list of {text, corrupted, corruption_type}) and `has_error` flag.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
SEED = 20260617

# Control-group ratio (no error injected). Used to measure false-positive rate.
CONTROL_RATIO = 0.15

# Error type distribution (applied to non-control problems).
INJECT_WEIGHTS = {
    "numeric": 0.40,
    "sign": 0.25,
    "variable": 0.15,
    "delete": 0.10,
    "insert": 0.10,
}

# Lines containing these substrings are not pure math steps.
NON_STEP_PATTERNS = [
    re.compile(r"\[asy\]", re.IGNORECASE),
    re.compile(r"\[/asy\]", re.IGNORECASE),
    re.compile(r"^Final Answer", re.IGNORECASE),
    re.compile(r"^The answer is", re.IGNORECASE),
]

# Lines that look like math steps contain at least one of these.
MATH_HINT_TOKENS = ["=", "\\frac", "\\sqrt", "\\cdot", "\\boxed", "\\sum", "\\int",
                    "\\Rightarrow", "\\implies", "\\to", "\\therefore", "\\because",
                    "^", "\\alpha", "\\beta", "\\theta", "\\pi"]


def is_math_step(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3:
        return False
    if len(s) > 500:  # likely a long asy drawing or paragraph
        return False
    for p in NON_STEP_PATTERNS:
        if p.search(s):
            return False
    # Permissive: any line with $...$ math, or = sign, or common math tokens.
    if "$" in s and "$" != s[-1]:  # has inline/block math
        return True
    return any(tok in s for tok in MATH_HINT_TOKENS)


def clean_solution(sol: str) -> str:
    # Strip [asy]...[/asy] blocks entirely (they're figure code, not math).
    sol = re.sub(r"\[asy\][\s\S]*?\[/asy\]", "", sol, flags=re.IGNORECASE)
    return sol


def split_into_steps(solution: str) -> list[str]:
    cleaned = clean_solution(solution)
    lines = []
    for raw in cleaned.split("\n"):
        s = raw.strip()
        if not s:
            continue
        # A line may contain multiple math statements separated by periods.
        # Heuristic: if line has multiple "$...$ =" patterns, keep as one step.
        if is_math_step(s):
            lines.append(s)
    return lines


# ---------- Injectors ----------

def inject_numeric(step: str, rng: random.Random) -> str:
    nums = re.findall(r"\d+", step)
    if not nums:
        return step
    target = rng.choice(nums)
    n = int(target)
    # Small perturbation: ±1 to ±5, or 10-30% off.
    delta = rng.choice([-3, -2, -1, 1, 2, 3, 5, -5])
    new_n = max(0, n + delta) if n + delta >= 0 else n + abs(delta)
    return step.replace(target, str(new_n), 1)


def inject_sign(step: str, rng: random.Random) -> str:
    # Flip first + or - that's between two tokens (not at line start).
    matches = list(re.finditer(r"([a-zA-Z0-9\}\])\\]+)\s*([+\-])\s*([a-zA-Z0-9\\{]+)", step))
    if not matches:
        return step
    m = rng.choice(matches)
    new_sign = "-" if m.group(2) == "+" else "+"
    return step[: m.start(2)] + new_sign + step[m.end(2):]


def inject_variable(step: str, rng: random.Random) -> str:
    # Find single-letter variable names not inside LaTeX commands.
    candidates = re.findall(r"(?<![a-zA-Z\\])([a-zA-Z])(?![a-zA-Z])", step)
    candidates = [c for c in candidates if c not in {"e", "i", "j"}]  # keep constants
    if not candidates:
        return step
    old = rng.choice(candidates)
    # Avoid common LaTeX commands.
    used = set(re.findall(r"\\([a-zA-Z]+)", step))
    new = rng.choice([c for c in "xyzabcdfghkmnpqrstuvw" if c != old and c not in used])
    # Replace first standalone occurrence.
    pattern = re.compile(rf"(?<![a-zA-Z\\]){re.escape(old)}(?![a-zA-Z])")
    return pattern.sub(new, step, count=1)


def inject_delete(steps: list[str], idx: int, rng: random.Random) -> list[str]:
    # Return new list with step at idx removed (caller handles marking).
    return [s for i, s in enumerate(steps) if i != idx]


INSERT_TEMPLATES = [
    "Therefore $\\boxed{0}$.",
    "So we have $1 = 2$.",
    "This gives $x = \\frac{0}{0}$.",
    "Hence $\\sqrt{-1} = 1$.",
    "Thus $2 + 2 = 5$.",
]


def inject_insert(steps: list[str], idx: int, rng: random.Random) -> tuple[list[str], int]:
    # Insert a wrong step BEFORE idx; returns (new_steps, wrong_step_index).
    wrong = rng.choice(INSERT_TEMPLATES)
    new_steps = steps[:idx] + [wrong] + steps[idx:]
    return new_steps, idx


def build_injected(problem: dict, rng: random.Random) -> dict:
    steps_raw = split_into_steps(problem["solution"])
    steps = [{"text": t, "corrupted": False, "corruption_type": None} for t in steps_raw]

    # Too few steps to corrupt meaningfully → treat as control.
    is_control = rng.random() < CONTROL_RATIO or len(steps_raw) < 2
    if is_control:
        return {
            **problem,
            "steps": steps,
            "has_error": False,
            "is_control": True,
        }

    # Try each error type in random order until one succeeds.
    types = list(INJECT_WEIGHTS.keys())
    rng.shuffle(types)
    for error_type in types:
        target_idx = rng.randrange(len(steps_raw))

        if error_type == "delete" and len(steps_raw) >= 2:
            new_texts = inject_delete(steps_raw, target_idx, rng)
            return {
                **problem,
                "steps": [{"text": t, "corrupted": False, "corruption_type": None} for t in new_texts],
                "has_error": True,
                "is_control": False,
                "corruption_type": "delete",
                "corruption_index": target_idx,
            }

        if error_type == "insert":
            new_texts, wrong_idx = inject_insert(steps_raw, target_idx, rng)
            return {
                **problem,
                "steps": [{"text": t, "corrupted": i == wrong_idx, "corruption_type": ("insert" if i == wrong_idx else None)}
                          for i, t in enumerate(new_texts)],
                "has_error": True,
                "is_control": False,
                "corruption_type": "insert",
                "corruption_index": wrong_idx,
            }

        target_text = steps_raw[target_idx]
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
            steps_out[target_idx] = {"text": new_text, "corrupted": True, "corruption_type": error_type}
            return {
                **problem,
                "steps": steps_out,
                "has_error": True,
                "is_control": False,
                "corruption_type": error_type,
                "corruption_index": target_idx,
            }
        # else: try next error type or next index

    # All injections failed — fall back to control.
    return {
        **problem,
        "steps": steps,
        "has_error": False,
        "is_control": True,
        "corruption_type": "noop",
    }


def main() -> None:
    raw_path = DATA_DIR / "math_raw.json"
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    rng = random.Random(SEED)

    out: list[dict] = []
    type_counts: dict[str, int] = {}
    for prob in data:
        item = build_injected(prob, rng)
        out.append(item)
        key = "control" if item.get("is_control") else item.get("corruption_type", "?")
        type_counts[key] = type_counts.get(key, 0) + 1

    out_path = DATA_DIR / "math_injected.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    step_counts = [len(x["steps"]) for x in out]
    print(f"Wrote {len(out)} problems -> {out_path}")
    print(f"Step count: min={min(step_counts)} max={max(step_counts)} avg={sum(step_counts)/len(step_counts):.1f}")
    print(f"  skipped (no steps extractable): {sum(1 for x in out if not x['steps'])}")
    print(f"  type distribution: {type_counts}")


if __name__ == "__main__":
    main()
