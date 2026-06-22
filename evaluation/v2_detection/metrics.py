"""v2 binary detection metrics.

Two views:
1. Problem-level: did the model's has_error match ground truth has_error?
   - TP / FP / FN / TN, accuracy, precision, recall, F1.
2. Step-level (for corrupted problems): did the model flag the injected step?
   - Per-type recall, per-type false-positive count (other steps flagged).

API failures are counted as worst-case (verdict treated as has_error=False
with no steps flagged). This gives an honest lower bound instead of hiding
flaky behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent / "results" / "per_problem"


def _verdict(result: dict) -> dict:
    v = result.get("verdict") or {}
    return {
        "has_error": bool(v.get("has_error", False)) if result.get("ok") else False,
        "error_steps": list(v.get("error_steps", [])) if result.get("ok") else [],
        "error_types": dict(v.get("error_types", {})) if result.get("ok") else {},
    }


def aggregate() -> dict:
    files = sorted(RESULTS_DIR.glob("*.json"))
    if not files:
        raise SystemExit(f"no per_problem results in {RESULTS_DIR}")

    records = [json.loads(f.read_text(encoding="utf-8")) for f in files]

    # --- problem-level binary classification ---
    tp = fp = fn = tn = 0
    # Per-type step-level (only on non-control corrupted problems with ok=True)
    per_type_total = defaultdict(int)
    per_type_step_hit = defaultdict(int)  # injected step flagged
    per_type_step_fp = defaultdict(int)   # non-injected step flagged
    per_type_step_fp_total = defaultdict(int)  # count of non-injected steps
    # Failures
    n_failures = 0
    failure_breakdown = defaultdict(int)
    # Type-confusion: ground-truth injected type -> model's predicted category
    type_confusion = defaultdict(lambda: defaultdict(int))

    # Control false-positive rate (per-step)
    control_total_steps = 0
    control_flagged_steps = 0
    control_problem_fp = 0
    control_problem_total = 0

    for r in records:
        gt = r.get("ground_truth") or {}
        is_control = bool(gt.get("is_control"))
        gt_has_error = bool(gt.get("has_error"))
        v = _verdict(r)
        pred_has_error = v["has_error"]

        if not r.get("ok"):
            n_failures += 1
            # Worst-case: treat as has_error=False, no steps flagged.
            err_str = (r.get("error") or "unknown")[:80]
            failure_breakdown[err_str] += 1
            pred_has_error = False

        # problem-level
        if pred_has_error and gt_has_error:
            tp += 1
        elif pred_has_error and not gt_has_error:
            fp += 1
        elif not pred_has_error and gt_has_error:
            fn += 1
        else:
            tn += 1

        if is_control:
            control_problem_total += 1
            control_total_steps += len(r.get("steps") or [])
            control_flagged_steps += len(v["error_steps"])
            if v["error_steps"]:
                control_problem_fp += 1
            continue

        if not gt_has_error:
            continue  # noop/control-ish, skip step-level

        ctype = gt.get("corruption_type") or "?"
        corr_idx_0 = gt.get("corruption_index")
        per_type_total[ctype] += 1

        if corr_idx_0 is None:
            continue
        injected_idx = corr_idx_0 + 1  # convert 0-indexed -> 1-indexed
        all_steps = r.get("steps") or []
        n_steps = len(all_steps)

        if injected_idx in v["error_steps"]:
            per_type_step_hit[ctype] += 1
            pred_cat = v["error_types"].get(str(injected_idx), "other")
            type_confusion[ctype][pred_cat] += 1
        else:
            type_confusion[ctype]["__miss__"] += 1

        non_injected = [s for s in v["error_steps"] if s != injected_idx]
        per_type_step_fp[ctype] += len(non_injected)
        per_type_step_fp_total[ctype] += max(0, n_steps - 1)

    n = len(records)
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    per_type_step_recall = {
        t: (per_type_step_hit[t] / per_type_total[t] if per_type_total[t] else 0.0)
        for t in per_type_total
    }
    per_type_step_fpr = {
        t: (
            per_type_step_fp[t] / per_type_step_fp_total[t]
            if per_type_step_fp_total[t] else 0.0
        )
        for t in per_type_total
    }
    control_step_fpr = (
        control_flagged_steps / control_total_steps if control_total_steps else 0.0
    )
    control_problem_fpr = (
        control_problem_fp / control_problem_total if control_problem_total else 0.0
    )

    return {
        "n": n,
        "n_failures": n_failures,
        "failure_rate": round(n_failures / n if n else 0.0, 4),
        "failure_breakdown": dict(failure_breakdown),
        "problem_level": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "per_type": {
            t: {
                "n": per_type_total[t],
                "step_recall": round(per_type_step_recall[t], 4),
                "step_fpr": round(per_type_step_fpr[t], 4),
                "step_fp_count": per_type_step_fp[t],
                "step_fp_total": per_type_step_fp_total[t],
            }
            for t in sorted(per_type_total)
        },
        "control": {
            "n_problems": control_problem_total,
            "n_steps": control_total_steps,
            "step_fpr": round(control_step_fpr, 4),
            "problem_fpr": round(control_problem_fpr, 4),
        },
        "type_confusion": {t: dict(type_confusion[t]) for t in sorted(type_confusion)},
    }


def main() -> None:
    m = aggregate()
    out_path = Path(__file__).parent / "results" / "summary.json"
    out_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
