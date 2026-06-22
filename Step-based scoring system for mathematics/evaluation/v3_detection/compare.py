"""Compare v2_detection and v3_detection summary.json side-by-side.

Reads ../v2_detection/results/summary.json and ./results/summary.json,
writes ./results/comparison.md.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V2 = ROOT.parent / "v2_detection" / "results" / "summary.json"
V3 = ROOT / "results" / "summary.json"
OUT = ROOT / "results" / "comparison.md"

TYPE_CN = {
    "numeric": "数值",
    "sign": "符号",
    "variable": "变量",
    "insert": "插入错步",
}


def row(label: str, v2v, v3v, fmt="{:.3f}", delta=True) -> str:
    s2 = fmt.format(v2v) if isinstance(v2v, (int, float)) else str(v2v)
    s3 = fmt.format(v3v) if isinstance(v3v, (int, float)) else str(v3v)
    out = f"| {label} | {s2} | {s3} |"
    if delta and isinstance(v2v, (int, float)) and isinstance(v3v, (int, float)):
        d = v3v - v2v
        sign = "+" if d > 0 else ""
        out += f" {sign}{d:.3f} |"
    else:
        out += " — |"
    return out


def main() -> None:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    v3 = json.loads(V3.read_text(encoding="utf-8"))

    p2, p3 = v2["problem_level"], v3["problem_level"]
    c2, c3 = v2["control"], v3["control"]
    t2, t3 = v2["per_type"], v3["per_type"]

    lines: list[str] = []
    lines.append("# v2 vs v3 对比报告\n")
    lines.append("- v2：基线 prompt（风格/跳步不算错，连锁不重罚）")
    lines.append("- v3：v2 + 变量一致性专章（未声明换名 = variable 错）")
    lines.append("- 数据集完全相同（200 题、同一份 math_injected.json、同 SEED）")
    lines.append("- Δ = v3 - v2，正数代表 v3 更好\n")

    lines.append("## 1. 题目级二分类\n")
    lines.append("| 指标 | v2 | v3 | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(row("样本数", p2["tp"] + p2["fp"] + p2["fn"] + p2["tn"], p3["tp"] + p3["fp"] + p3["fn"] + p3["tn"], fmt="{}"))
    lines.append(row("API 失败", v2["n_failures"], v3["n_failures"], fmt="{}"))
    lines.append(row("Accuracy", p2["accuracy"], p3["accuracy"]))
    lines.append(row("Precision", p2["precision"], p3["precision"]))
    lines.append(row("Recall", p2["recall"], p3["recall"]))
    lines.append(row("F1", p2["f1"], p3["f1"]))
    lines.append(row("TP", p2["tp"], p3["tp"], fmt="{}"))
    lines.append(row("FP", p2["fp"], p3["fp"], fmt="{}"))
    lines.append(row("FN", p2["fn"], p3["fn"], fmt="{}"))
    lines.append(row("TN", p2["tn"], p3["tn"], fmt="{}"))
    lines.append("")

    lines.append("## 2. 控制组假阳率（最关键的回归指标）\n")
    lines.append("| 指标 | v2 | v3 | Δ |")
    lines.append("|---|---|---|---|")
    lines.append(row("控制组题数", c2["n_problems"], c3["n_problems"], fmt="{}"))
    lines.append(row("控制组步数", c2["n_steps"], c3["n_steps"], fmt="{}"))
    lines.append(row("步级假阳率", c2["step_fpr"], c3["step_fpr"]))
    lines.append(row("题级假阳率", c2["problem_fpr"], c3["problem_fpr"]))
    lines.append("")

    lines.append("## 3. 分错步类型召回（v3 的主攻目标）\n")
    lines.append("| 注入类型 | v2 召回 | v3 召回 | Δ | v2 题数 | v3 题数 |")
    lines.append("|---|---|---|---|---|---|")
    all_types = sorted(set(t2) | set(t3))
    for t in all_types:
        r2 = t2.get(t, {}).get("step_recall", 0.0)
        r3 = t3.get(t, {}).get("step_recall", 0.0)
        n2 = t2.get(t, {}).get("n", 0)
        n3 = t3.get(t, {}).get("n", 0)
        cn = TYPE_CN.get(t, t)
        d = r3 - r2
        sign = "+" if d > 0 else ""
        lines.append(f"| {cn} ({t}) | {r2:.3f} | {r3:.3f} | {sign}{d:.3f} | {n2} | {n3} |")
    lines.append("")

    lines.append("## 4. 分错步类型 — 其他步误判率（FPR）\n")
    lines.append("| 注入类型 | v2 FPR | v3 FPR | Δ |")
    lines.append("|---|---|---|---|")
    for t in all_types:
        f2 = t2.get(t, {}).get("step_fpr", 0.0)
        f3 = t3.get(t, {}).get("step_fpr", 0.0)
        cn = TYPE_CN.get(t, t)
        d = f3 - f2
        sign = "+" if d > 0 else ""
        lines.append(f"| {cn} ({t}) | {f2:.3f} | {f3:.3f} | {sign}{d:.3f} |")
    lines.append("")

    lines.append("## 5. 解读\n")
    var_d = t3.get("variable", {}).get("step_recall", 0.0) - t2.get("variable", {}).get("step_recall", 0.0)
    ctrl_d = c3["step_fpr"] - c2["step_fpr"]
    p_d = p3["precision"] - p2["precision"]
    r_d = p3["recall"] - p2["recall"]
    f1_d = p3["f1"] - p2["f1"]

    verdict = []
    if var_d >= 0.10:
        verdict.append(f"✓ 主攻目标达成：variable 召回提升 {var_d:+.3f}（{t2.get('variable', {}).get('step_recall', 0):.2f} → {t3.get('variable', {}).get('step_recall', 0):.2f}）")
    elif var_d > 0:
        verdict.append(f"△ variable 召回小幅提升 {var_d:+.3f}，未达 0.10 阈值")
    else:
        verdict.append(f"✗ variable 召回反而下降 {var_d:+.3f}，v3 prompt 未生效或副作用")

    if ctrl_d > 0.005:
        verdict.append(f"⚠ 控制组步级假阳率上升 {ctrl_d:+.3f}（{c2['step_fpr']:.3f} → {c3['step_fpr']:.3f}）—— 变量一致性章节引入了误判，需要警惕")
    else:
        verdict.append(f"✓ 控制组步级假阳率守住（Δ {ctrl_d:+.3f}），变量约束没有引入误判")

    if f1_d > 0.02:
        verdict.append(f"✓ F1 整体提升 {f1_d:+.3f}")
    elif f1_d < -0.02:
        verdict.append(f"✗ F1 整体下降 {f1_d:+.3f}，v3 prompt 综合劣化")
    else:
        verdict.append(f"△ F1 基本持平（Δ {f1_d:+.3f}）")

    for v in verdict:
        lines.append(f"- {v}")
    lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    print()
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
