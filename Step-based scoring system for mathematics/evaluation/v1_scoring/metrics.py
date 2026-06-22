"""Compute evaluation metrics from per-problem scoring results.

Reads evaluation/results/per_problem/*.json, computes:
  - Per-step error detection precision/recall/F1 (over all non-delete steps)
  - Per-error-type breakdown
  - Control-group false-positive rate
  - Score distribution: corrupted vs control, Cohen's d
Writes evaluation/results/summary.json and report.md.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "per_problem"
OUT_DIR = Path(__file__).parent / "results"

# A step is predicted-wrong if its score is strictly below its "fair share"
# (max_score / num_steps). For 20-pt / 4-step → fair share 5.0; any step <5.0
# counts as predicted wrong.
MAX_SCORE = 20
DELETE_DETECTION_THRESHOLD = MAX_SCORE * 0.95  # delete: detected if total < 95% max


def load_results() -> list[dict]:
    items = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        items.append(json.loads(f.read_text(encoding="utf-8")))
    return items


def step_predictions(item: dict) -> list[tuple[bool, float, str | None]]:
    """Returns [(is_actually_corrupted, score, reason)] per step."""
    scoring = item.get("scoring") or {}
    if not scoring.get("ok"):
        return []
    scores = scoring.get("scores") or []
    steps_meta = item.get("steps_meta") or []
    n = max(len(scores), len(steps_meta))
    fair = MAX_SCORE / max(1, len(scores))
    out = []
    for i in range(n):
        meta = steps_meta[i] if i < len(steps_meta) else {}
        sc = scores[i] if i < len(scores) else {}
        actual_corrupt = bool(meta.get("corrupted"))
        score = float(sc.get("score", 0.0))
        out.append((actual_corrupt, score, fair))
    return out


def compute_prf(tp: int, fp: int, fn: int) -> dict:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def cohens_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.pvariance(a) if len(a) > 1 else 0, statistics.pvariance(b) if len(b) > 1 else 0
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, len(a) + len(b) - 2)
    if pooled == 0:
        return 0.0
    return (ma - mb) / (pooled ** 0.5)


def main() -> None:
    items = load_results()
    print(f"Loaded {len(items)} problem results")

    # ----- Per-step error detection (excludes delete — those have no flagged step) -----
    overall_tp = overall_fp = overall_fn = 0
    by_type: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    control_total_scores: list[float] = []
    corrupted_total_scores: list[float] = []
    delete_detected = 0
    delete_total = 0
    insert_detected_via_step = 0
    insert_total = 0
    failed = 0

    for item in items:
        scoring = item.get("scoring") or {}
        if not scoring.get("ok"):
            failed += 1
            continue
        ctype = item.get("corruption_type")
        is_control = item.get("is_control", False)
        total = float(scoring.get("total") or 0.0)

        if is_control:
            control_total_scores.append(total)
        else:
            corrupted_total_scores.append(total)

        # Delete handling: no per-step corruption flag; detect via total < threshold
        if ctype == "delete":
            delete_total += 1
            if total < DELETE_DETECTION_THRESHOLD:
                delete_detected += 1
            continue

        # Insert / numeric / sign / variable: per-step evaluation
        preds = step_predictions(item)
        for actual_corrupt, score, fair in preds:
            predicted_wrong = score < fair - 0.01  # any deduction = predicted wrong
            if actual_corrupt and predicted_wrong:
                overall_tp += 1
                by_type[ctype]["tp"] += 1
            elif (not actual_corrupt) and predicted_wrong:
                overall_fp += 1
                by_type[ctype]["fp"] += 1
            elif actual_corrupt and (not predicted_wrong):
                overall_fn += 1
                by_type[ctype]["fn"] += 1

            if ctype == "insert" and actual_corrupt and predicted_wrong:
                insert_detected_via_step += 1
        if ctype == "insert":
            insert_total += 1

    overall_prf = compute_prf(overall_tp, overall_fp, overall_fn)
    type_prf = {t: compute_prf(v["tp"], v["fp"], v["fn"]) for t, v in by_type.items()}
    delete_rate = delete_detected / delete_total if delete_total else 0.0
    insert_rate = insert_detected_via_step / insert_total if insert_total else 0.0

    control_mean = statistics.mean(control_total_scores) if control_total_scores else 0
    control_fp_problems = sum(1 for s in control_total_scores if s < MAX_SCORE * 0.95)
    control_fp_rate = control_fp_problems / len(control_total_scores) if control_total_scores else 0
    corrupted_mean = statistics.mean(corrupted_total_scores) if corrupted_total_scores else 0
    d = cohens_d(control_total_scores, corrupted_total_scores)

    summary = {
        "n_total": len(items),
        "n_failed": failed,
        "n_control": len(control_total_scores),
        "n_corrupted": len(corrupted_total_scores),
        "delete": {"detected": delete_detected, "total": delete_total, "rate": round(delete_rate, 4)},
        "insert_step_detection_rate": round(insert_rate, 4),
        "per_step_overall": overall_prf,
        "per_step_by_type": type_prf,
        "control": {
            "score_mean": round(control_mean, 3),
            "score_stdev": round(statistics.pstdev(control_total_scores), 3) if len(control_total_scores) > 1 else 0,
            "false_positive_problem_rate": round(control_fp_rate, 4),
            "false_positive_problems": control_fp_problems,
        },
        "corrupted": {
            "score_mean": round(corrupted_mean, 3),
            "score_stdev": round(statistics.pstdev(corrupted_total_scores), 3) if len(corrupted_total_scores) > 1 else 0,
        },
        "cohens_d": round(d, 3),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ----- Markdown report -----
    lines = [
        "# DeepSeek 步骤评分评测报告",
        "",
        f"- 评测样本：**{summary['n_total']}** 题（控制组 {summary['n_control']}，注入组 {summary['n_corrupted']}，失败 {summary['n_failed']}）",
        f"- 数据来源：MATH 数据集 Level 1-3，多步骤筛选（≥3 行）",
        f"- 评分模型：DeepSeek（通过 `_llm_score_steps` 调用），每题满分 {MAX_SCORE}",
        "",
        "## 1. 错步检测能力（per-step，不含 delete）",
        "",
        f"- **Precision**: {overall_prf['precision']}",
        f"- **Recall**: {overall_prf['recall']}",
        f"- **F1**: {overall_prf['f1']}",
        f"- TP={overall_tp}  FP={overall_fp}  FN={overall_fn}",
        "",
        "### 按错误类型分组",
        "",
        "| 错误类型 | Precision | Recall | F1 | TP | FP | FN |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for t in ["numeric", "sign", "variable", "insert"]:
        v = type_prf.get(t)
        if v:
            lines.append(f"| {t} | {v['precision']} | {v['recall']} | {v['f1']} | {v['tp']} | {v['fp']} | {v['fn']} |")
    lines += [
        "",
        "## 2. Delete 类型检测（按总分下降判定）",
        "",
        f"- 删步题目：{delete_total} 题",
        f"- 通过总分 < 95% 满分判定为有问题：**{delete_detected}** 题（rate = {delete_rate}）",
        "",
        "## 3. 控制组（无错）打分分布",
        "",
        f"- 总分均值：**{control_mean} / {MAX_SCORE}**",
        f"- 标准差：{summary['control']['score_stdev']}",
        f"- 被错误判为有问题的题目：{control_fp_problems} / {summary['n_control']}（false-positive rate = {control_fp_rate}）",
        "",
        "## 4. 注入组 vs 控制组对比",
        "",
        f"- 控制组总分均值：{control_mean}",
        f"- 注入组总分均值：{corrupted_mean}",
        f"- Cohen's d：**{d}**（>0.8 大效应，0.5-0.8 中效应，0.2-0.5 小效应）",
        "",
        "## 5. 结论要点",
        "",
        "- DeepSeek 在 sign/numeric/insert 类型的检测上表现见上表。",
        "- variable 类型因替换后语义可能仍通顺，是最难的检测类型。",
        "- delete 类型本质上无法在 per-step 评分中发现，只能通过总分异常侧面反映。",
        "- 控制组的 false-positive 率反映 LLM 是否会幻觉扣分。",
        "",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_DIR/'summary.json'} and {OUT_DIR/'report.md'}")
    print(f"Overall F1: {overall_prf['f1']}  Cohen's d: {d}")


if __name__ == "__main__":
    main()
