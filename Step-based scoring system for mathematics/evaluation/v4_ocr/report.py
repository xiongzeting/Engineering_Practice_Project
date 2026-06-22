"""Generate Chinese markdown report from v4_ocr summary + per_sample cases."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
SUMMARY = RESULTS_DIR / "summary.json"
PER_SAMPLE = RESULTS_DIR / "per_sample"
OUT = RESULTS_DIR / "report.md"


def load_all_cases() -> list[dict]:
    cases: list[dict] = []
    for f in sorted(PER_SAMPLE.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        gt = r.get("ground_truth", "") or ""
        pred = r.get("ocr_text", "") or ""
        if not r.get("ok"):
            cases.append({
                "id": r["id"], "gt": gt, "pred": "",
                "cer": 1.0, "bleu": 0.0,
                "error": r.get("error", "unknown"),
            })
            continue
        # Recompute CER for sorting consistency.
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from metrics import normalize_latex, edit_distance, bleu4
        gt_n = normalize_latex(gt)
        pred_n = normalize_latex(pred)
        cer = edit_distance(pred_n, gt_n) / max(len(gt_n), 1)
        cases.append({
            "id": r["id"],
            "gt": gt,
            "pred": pred,
            "cer": round(cer, 4),
            "bleu": round(bleu4(pred_n, gt_n), 4),
        })
    return cases


def render(m: dict, cases: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# v4 OCR 端到端评测报告\n")
    lines.append("- 数据集：MathWriting-human（HF 镜像），随机 100 条真实学生手写数学表达式")
    lines.append("- OCR 引擎：生产环境 `vision_only_ocr`（OpenAI 兼容 vision API）")
    lines.append("- 方法论瑕疵：MathWriting 是单表达式级别，生产 OCR prompt 是为整页作业设计的；")
    lines.append(r"  指标计算前对 GT 和 OCR 输出做激进 LaTeX 归一化（剥 `$`、统一 `\frac`、删 `\left/\right` 等）。")
    lines.append("")

    lines.append("## 1. 整体指标\n")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    lines.append(f"| 样本数 | {m['n']} |")
    lines.append(f"| OCR 失败 | {m['n_failed']}（{m['failure_rate']*100:.1f}%） |")
    lines.append(f"| CER 平均 | {m['cer_mean']:.4f} |")
    lines.append(f"| CER 中位 | {m['cer_median']:.4f} |")
    lines.append(f"| BLEU-4 平均 | {m['bleu_mean']:.4f} |")
    lines.append(f"| 完全匹配率（归一化后） | {m['exact_match_rate']*100:.1f}% |")
    lines.append(f"| CER≤0.1 比例（基本读对） | {m['cer_le_01_rate']*100:.1f}% |")
    lines.append("")

    lines.append("## 2. CER 分桶分布\n")
    lines.append("| 桶 | 样本数 | 占比 |")
    lines.append("|---|---|---|")
    for bucket, count in m["buckets"].items():
        pct = count / m["n"] * 100 if m["n"] else 0
        lines.append(f"| {bucket} | {count} | {pct:.1f}% |")
    lines.append("")

    # Worst 10 by CER
    worst = sorted(cases, key=lambda c: -c["cer"])[:10]
    lines.append("## 3. 最差 10 条（CER 由高到低）\n")
    lines.append("用于肉眼判断错因：是 Unicode 误识别、符号丢失、多识别了说明文字，还是模型完全没读懂？\n")
    for c in worst:
        lines.append(f"### {c['id']}  CER={c['cer']:.3f}  BLEU={c.get('bleu', 0):.3f}")
        if "error" in c:
            lines.append(f"- OCR 失败：`{c['error']}`")
        lines.append(f"- **GT**：`{c['gt']}`")
        lines.append(f"- **OCR**：`{c['pred']}`")
        lines.append("")

    lines.append("## 4. 解读\n")
    cer_mean = m["cer_mean"]
    em = m["exact_match_rate"]
    bleu = m["bleu_mean"]
    fail = m["failure_rate"]
    if cer_mean < 0.05:
        verdict_cer = f"✓ CER 平均 {cer_mean:.3f} 处于优秀区间（<0.05），OCR 在手写单表达式上工作良好"
    elif cer_mean < 0.15:
        verdict_cer = f"△ CER 平均 {cer_mean:.3f} 处于可接受区间（0.05–0.15），但有改进空间"
    else:
        verdict_cer = f"✗ CER 平均 {cer_mean:.3f} 偏高（>0.15），OCR 在手写识别上有实质问题"
    lines.append(f"- {verdict_cer}。")
    lines.append(
        f"- 完全匹配率 {em*100:.1f}%，BLEU-4 平均 {bleu:.3f}。"
        f"对照 v2/v3 的 F1=0.91–0.93（OCR 完美时的上界），"
        f"OCR 错误对最终评分的实际影响需要 v5 端到端评测才能完整量化。"
    )
    if fail > 0.05:
        lines.append(f"- ⚠ OCR 失败率 {fail*100:.1f}%，需要检查 vision API 的稳定性（重试 / 超时配置）。")
    return "\n".join(lines) + "\n"


def main() -> None:
    m = json.loads(SUMMARY.read_text(encoding="utf-8"))
    cases = load_all_cases()
    OUT.write_text(render(m, cases), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
