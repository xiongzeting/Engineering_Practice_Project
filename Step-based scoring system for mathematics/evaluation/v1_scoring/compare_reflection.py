"""Compare v1_scoring metrics across two per_problem result dirs.

Usage:
    python compare_reflection.py <dir_a> <dir_b> [--label-a A] [--label-b B]

Computes per-step precision/recall/F1 + control/corrupted score distributions
for each dir, then prints a side-by-side delta.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = Path(__file__).parent / "results"
MAX_SCORE = 20
DELETE_DETECTION_THRESHOLD = MAX_SCORE * 0.95


def load_dir(dir_name: str) -> list[dict]:
    p = RESULTS_ROOT / dir_name
    if not p.is_dir():
        raise SystemExit(f"missing dir: {p}")
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(p.glob("*.json"))]


def compute_metrics(items: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    by_type: dict[str, dict[str, int]] = {}
    control_scores: list[float] = []
    corrupted_scores: list[float] = []
    delete_detected = 0
    delete_total = 0
    n_failed = 0
    n_ok = 0
    type_confusion: dict[str, dict[str, int]] = {}

    for item in items:
        scoring = item.get("scoring") or {}
        if not scoring.get("ok"):
            n_failed += 1
            continue
        n_ok += 1
        scores = scoring.get("scores") or []
        steps_meta = item.get("steps_meta") or []
        n = max(len(scores), len(steps_meta))
        fair = MAX_SCORE / max(1, len(scores))
        is_control = bool(item.get("is_control"))
        total = float(scoring.get("total") or 0.0)
        if is_control:
            control_scores.append(total)
        else:
            corrupted_scores.append(total)
            ctype = item.get("corruption_type") or "unknown"
            if ctype == "delete":
                delete_total += 1
                if total < DELETE_DETECTION_THRESHOLD:
                    delete_detected += 1

        for i in range(n):
            meta = steps_meta[i] if i < len(steps_meta) else {}
            sc = scores[i] if i < len(scores) else {}
            actual_corrupt = bool(meta.get("corrupted"))
            score = float(sc.get("score", 0.0))
            pred_wrong = score < fair
            if actual_corrupt and pred_wrong:
                tp += 1
                gt_type = meta.get("corruption_type") or "unknown"
                pr_type = (sc.get("reason") or "")[:0] or "unknown"
                # Don't classify predicted type from reason (v1 didn't either).
            elif actual_corrupt and not pred_wrong:
                fn += 1
            elif not actual_corrupt and pred_wrong:
                fp += 1
            else:
                tn += 1

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "n_items": len(items),
        "n_ok": n_ok,
        "n_failed": n_failed,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "control_score_mean": round(statistics.mean(control_scores), 3) if control_scores else None,
        "control_score_stdev": round(statistics.stdev(control_scores), 3) if len(control_scores) > 1 else None,
        "corrupted_score_mean": round(statistics.mean(corrupted_scores), 3) if corrupted_scores else None,
        "corrupted_score_stdev": round(statistics.stdev(corrupted_scores), 3) if len(corrupted_scores) > 1 else None,
        "cohens_d": round(
            (statistics.mean(control_scores) - statistics.mean(corrupted_scores))
            / max(
                statistics.pstdev(control_scores + corrupted_scores) or 1.0,
                1e-9,
            ),
            3,
        ) if control_scores and corrupted_scores else None,
        "delete_detection_rate": round(delete_detected / max(delete_total, 1), 4) if delete_total else None,
    }


def print_comparison(a: dict, b: dict, label_a: str, label_b: str) -> None:
    keys = ["n_items", "n_ok", "n_failed", "tp", "fp", "fn", "tn",
            "precision", "recall", "f1",
            "control_score_mean", "corrupted_score_mean", "cohens_d",
            "delete_detection_rate"]
    print(f"\n{'metric':<28s} {label_a:>20s} {label_b:>20s} {'delta':>12s}")
    print("-" * 84)
    for k in keys:
        va = a.get(k)
        vb = b.get(k)
        if va is None or vb is None:
            continue
        delta = vb - va if isinstance(va, (int, float)) and isinstance(vb, (int, float)) else ""
        if isinstance(delta, float):
            delta_str = f"{delta:+.4f}"
        elif isinstance(delta, int):
            delta_str = f"{delta:+d}"
        else:
            delta_str = str(delta)
        print(f"{k:<28s} {str(va):>20s} {str(vb):>20s} {delta_str:>12s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    args = ap.parse_args()
    a = compute_metrics(load_dir(args.dir_a))
    b = compute_metrics(load_dir(args.dir_b))
    print(f"\n=== {args.label_a} ({args.dir_a}) ===")
    print(json.dumps(a, indent=2, ensure_ascii=False))
    print(f"\n=== {args.label_b} ({args.dir_b}) ===")
    print(json.dumps(b, indent=2, ensure_ascii=False))
    print_comparison(a, b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
