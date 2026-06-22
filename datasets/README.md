# 评测数据集目录

本目录是 9 个评测数据集的**物理存放位置**。所有 `evaluation/vX_xxx/data/`
都是指回这里的反向软链接——评测脚本仍按原路径 `Path(__file__).parent / "data"`
读取，但真实数据集中在本目录，便于统一管理与归档。

## 数据集总览

| 目录 | 数据源 | 类型 | 样本数 | 用途 |
| --- | --- | --- | --- | --- |
| `v1_scoring_math/`     | MATH（Hendrycks）         | 文本 · 合成 | 200 | 逐步**评分准确率** |
| `v2_detection_math/`   | MATH（同 v1）             | 文本 · 合成 | 200 | **错误检测**（v2 修正版） |
| `v3_detection_math/`   | MATH（同 v1）             | 文本 · 合成 | 200 | **错误检测**（v3 复测） |
| `v4_ocr_mathwriting/`  | deepcopy/MathWriting-human | 图片 · 手写 | 100 | OCR（真实手写公式）|
| `v4b_ocr_mathwriting/` | ← 同 v4                   | — | — | OCR（旧对照组） |
| `v4c_ocr_mathwriting/` | ← 同 v4                   | — | — | OCR（旧对照组） |
| `v5_aida_calculus/`    | deepcopy/Aida-Calculus    | 图片 · 手写 | 100 | OCR（大学微积分） |
| `v5_im2latex/`         | yuntian-deng/im2latex-100k| 图片 · 印刷 | 100 | OCR（印刷公式） |
| `v6_crohme/`           | linxy/LaTeX_OCR           | 图片 · 手写 | 100 | OCR（CROHME 风格手写）|

> v4b / v4c 是 `v4_ocr_mathwriting` 的同名软链接，保留是因为早期跑过两次
> 对照实验；新实验请直接用 `v4_ocr_mathwriting`。

---

## 文本类数据集（v1 / v2 / v3）

这三个目录都包含两个文件：

- **`math_raw.json`** —— 从 MATH 数据集按 Level 1–3 分层抽样的 200 道题，
  字段：`id` / `problem` / `solution` / `level` / `type`。
- **`math_injected.json`** —— 把每道题的 `solution` 按行切成步骤，并按
  概率注入合成错误，字段：`id` / `steps[]`（每个步骤带 `corrupted`
  bool 与 `corruption_type`）/ `has_error` bool。

### v1：评分准确率（scoring）

- **任务**：对每道题的每一步打分（0–1），与"是否被注入错误"的 ground
  truth 对齐，算 F1。
- **错误类型分布**（注入器权重）：`numeric 40%` / `sign 25%` /
  `variable 15%` / `delete 10%` / `insert 10%`。
- **控制组**：15% 的题目不注入任何错误，用于测假阳率。
- **跑分入口**：`evaluation/v1_scoring/run_eval.py`，跑完后
  `evaluation/v1_scoring/results/` 存每题 JSON，
  `metrics.py` + `compare_reflection.py` 算 F1 与反思前后对比。

### v2：错误检测（detection · 修正版）

- **任务**：把检测器当二分类器，输出"这道题是否含错步"。
- **设计修正**：移除了 `delete` 类型——"删掉一步"在数学上仍然正确，
  不算真实错误，会污染标注。
- **错误分布**：`numeric 45%` / `sign 30%` / `variable 10%` / `insert 15%`。
- **控制组**：15% 无错误。
- 与 v1 用同一 SEED（`20260617`）以便逐步对照。

### v3：错误检测（复测）

- 与 v2 注入配置完全相同；用作"v2 的稳定复测"——若 v2/v3 结果差异大，
  说明检测器对随机种子过敏，需要补 robustness 测试。

---

## 图片类数据集（v4 / v5 / v6）

每个目录的统一结构：

```
<name>/
├── manifest.json      # [{id, image_path, ground_truth_latex}, ...]
└── images/            # 100 张 PNG
    ├── xx-0000.png
    ├── xx-0001.png
    └── ...
```

`manifest.json` 里 `image_path` 是相对评测目录的相对路径（
`data/images/xx-0000.png`），评测脚本读 manifest 时按相对路径定位——
因此 `evaluation/vX_xxx/data/` 必须仍然能解析（已由反向软链接保证）。

### v4：MathWriting（真实手写 · 基线）

- **来源**：HuggingFace `deepcopy/MathWriting-human`，SEED `20260618`。
- **特点**：真实人类手写公式，含连笔、字距不均、轻度歪斜——贴近学生
  作业，是系统的主 OCR 基线。
- **GT**：标准 LaTeX 字符串。

### v4b / v4c：v4 的对照组

- 早期实验复用 v4 数据；现已合并为单一软链接。**新实验不要用这两个名字**，
  保留只是因为旧 results/ 里有引用。

### v5 Aida-Calculus：大学微积分

- **来源**：`deepcopy/Aida-Calculus-Math-Handwriting`，SEED `20260619`。
- **特点**：手写但内容偏大学微积分（`lim` / `∫` / 偏导），公式复杂度
  显著高于 K1-12。用于测系统在超纲内容上的退化行为。
- **GT**：LaTeX。

### v5 im2latex：印刷公式

- **来源**：`yuntian-deng/im2latex-100k` 的 `test` split，SEED `20260620`。
- **特点**：**印刷体**（LaTeX 源码渲染而成），无手写噪声。用于剥离
  手写扰动，单独测系统对"清晰印刷公式"的识别上限——理论上限。
- **GT**：LaTeX。

### v6 CROHME：在线手写

- **来源**：`linxy/LaTeX_OCR`，SEED `20260621`。
- **特点**：CROHME 风格的在线手写（笔迹轨迹渲染为图片），比 MathWriting
  更接近学生实时书写，符号间距与笔画更不规则。
- **GT**：LaTeX。

---

## 如何重新生成

文本数据集（v1/v2/v3）：

```bash
cd evaluation/v1_scoring && python download_math.py   # 拉原始 MATH
cd evaluation/v1_scoring && python inject_errors.py    # 注入错误
# v2/v3 同理（inject_errors.py 的权重与 delete 配置不同）
```

图片数据集（v4/v5/v6）：

```bash
cd evaluation/v4_ocr && python download_dataset.py
# v5_aida / v5_im2latex / v6_crohme 同理
```

> 首次跑下载脚本会从 HuggingFace 拉源数据，可能需要几百 MB 带宽。
> `datasets` 库与 `Pillow` 是必需依赖（见 `requirements.txt`）。

---

## 与生产代码的关系

- OCR 评测调的是 `app/services/vision_ocr.py::vision_only_ocr`，
  即生产线上同一个函数——保证评测与真实用户体验一致。
- 评分评测调的是 `app/services/scorer.py` 的评分链路；F1=0.93 那条基线
  就是在 `v1_scoring_math` 上跑出来的（详见 memory 里的"scorer 保护"
  记录）。
