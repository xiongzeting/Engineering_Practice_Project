"""Generate Chinese markdown report from v2_detection summary.json."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY = RESULTS_DIR / "summary.json"
OUT = RESULTS_DIR / "report.md"

TYPE_CN = {
    "numeric": "数值",
    "sign": "符号",
    "variable": "变量",
    "insert": "插入错步",
    "calculation": "计算",
    "logic": "逻辑",
    "other": "其他",
}


def render(m: dict) -> str:
    pl = m["problem_level"]
    ctrl = m["control"]
    lines: list[str] = []
    lines.append("# v2 错误检测能力评测报告\n")
    lines.append(f"- 数据集：MATH 200 题（DeepSeek v4-pro 检测，二分类 + 错步定位）")
    lines.append(f"- API 失败：{m['n_failures']} / {m['n']}（{m['failure_rate']*100:.1f}%），失败按 worst-case（视为漏报）计入 recall")
    lines.append("")

    lines.append("## 1. 整体二分类（题目级 has_error）\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 样本数 | {m['n']} |")
    lines.append(f"| TP / FP / FN / TN | {pl['tp']} / {pl['fp']} / {pl['fn']} / {pl['tn']} |")
    lines.append(f"| Accuracy | {pl['accuracy']:.3f} |")
    lines.append(f"| Precision | {pl['precision']:.3f} |")
    lines.append(f"| Recall | {pl['recall']:.3f} |")
    lines.append(f"| F1 | {pl['f1']:.3f} |")
    lines.append("")

    lines.append("## 2. 控制组假阳率（正确步骤被误判）\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 控制组题数 | {ctrl['n_problems']} |")
    lines.append(f"| 控制组步数 | {ctrl['n_steps']} |")
    lines.append(f"| 步级假阳率 | {ctrl['step_fpr']:.3f} |")
    lines.append(f"| 题级假阳率（任意步被误判） | {ctrl['problem_fpr']:.3f} |")
    lines.append("")

    lines.append("## 3. 分错步类型定位能力\n")
    lines.append("仅统计注入错的题目。step_recall = 模型把注入的错步标出来的比例；step_fpr = 模型把其他正常步误标为错的比例。\n")
    lines.append("| 注入类型 | 题数 | 错步定位召回 | 其他步误判率 | 误判步数 / 总正常步 |")
    lines.append("|---|---|---|---|---|")
    for t, info in sorted(m["per_type"].items()):
        cn = TYPE_CN.get(t, t)
        lines.append(
            f"| {cn} ({t}) | {info['n']} | {info['step_recall']:.3f} | "
            f"{info['step_fpr']:.3f} | {info['step_fp_count']} / {info['step_fp_total']} |"
        )
    lines.append("")

    lines.append("## 4. 错误类型分类混淆（注入类型 → 模型归类）\n")
    lines.append("| 注入类型 | 模型归类分布 |")
    lines.append("|---|---|")
    for t, dist in sorted(m["type_confusion"].items()):
        cn = TYPE_CN.get(t, t)
        parts = []
        for cat, cnt in sorted(dist.items(), key=lambda kv: -kv[1]):
            label = "漏报" if cat == "__miss__" else TYPE_CN.get(cat, cat)
            parts.append(f"{label}={cnt}")
        lines.append(f"| {cn} ({t}) | {', '.join(parts) if parts else '-'} |")
    lines.append("")

    if m["failure_breakdown"]:
        lines.append("## 5. API 失败原因分布\n")
        lines.append("| 错误 | 次数 |")
        lines.append("|---|---|")
        for err, cnt in sorted(m["failure_breakdown"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{err}` | {cnt} |")
        lines.append("")

    lines.append("## 6. 解读要点\n")
    lines.append(
        f"- F1 = {pl['f1']:.3f}：题目级『是否有错』的整体判断能力。"
        f"Precision = {pl['precision']:.3f}（误报率 {1 - pl['precision']:.2%}），"
        f"Recall = {pl['recall']:.3f}（漏报率 {1 - pl['recall']:.2%}，含 API 失败的最坏情况）。"
    )
    lines.append(
        f"- 控制组步级假阳率 = {ctrl['step_fpr']:.3f}：每 100 个完全正确的步骤中，"
        f"模型平均误标 {ctrl['step_fpr']*100:.1f} 个为错。这是当前 prompt 风格的关键风险点。"
    )
    best_type = max(m["per_type"].items(), key=lambda kv: kv[1]["step_recall"])
    worst_type = min(m["per_type"].items(), key=lambda kv: kv[1]["step_recall"])
    lines.append(
        f"- 最易检出的错步类型：{TYPE_CN.get(best_type[0], best_type[0])}（召回 {best_type[1]['step_recall']:.3f}）；"
        f"最不易检出：{TYPE_CN.get(worst_type[0], worst_type[0])}（召回 {worst_type[1]['step_recall']:.3f}）。"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    m = json.loads(SUMMARY.read_text(encoding="utf-8"))
    OUT.write_text(render(m), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
