"""把 OCR 出来的纯文本切成一行行的 ``StepItem``。

评分流程的第一步——把一整段 ``"解: 原式= ...\\n= ...\\n∴ ..."`` 切成独立的
步骤，每个步骤都带上 ``normalized``（去前缀后）、``has_equation``（是否含
数学符号）和 ``confidence``（OCR 置信度估计）。

切分规则：
- 按连续换行或大量缩进作为步骤分隔；
- 去掉 ``Step N`` / ``1.`` / ``(1)`` / ``第 2 步`` 这类题内序号前缀；
- 过滤掉纯注释行（括注 ``(直接积分)``、`(偶倍奇零)` 等方法标签）；
- 特殊情况：如果整段只有一行但含 ``=`` 链，再按等号切分成多步，
  方便后续逐步打分。

这个模块是 :mod:`app.services.scorer` 的入口预处理。
"""
from __future__ import annotations

import re

from app.schemas import StepItem

# 步骤前缀正则：匹配 ``1.`` / ``(1)`` / ``第 2 步`` 等题内序号。
STEP_PREFIX_RE = re.compile(r"^\s*(?:\(?\d+\)|\d+[.)]|第\s*\d+\s*步[:：]?)\s*")
# 分隔符正则：连续换行 或 连续 ≥2 个空格都算步骤边界。
SEPARATOR_RE = re.compile(r"(?:\r?\n)+|(?:\s{2,})")
# 数学符号正则：用来判定该步是否含数学表达式（has_equation）。
MATH_TOKEN_RE = re.compile(r"(=|\\int|\\frac|\\sum|\\sin|\\cos|\\tan|\\ln|\\sqrt|\^|_|\+|\-|\*|/)")
# 括注正则：整行就是 ``(直接积分)``、``（偶倍奇零）`` 这种方法标签。
ANNOTATION_RE = re.compile(r"^\s*[（(][^（）()]{1,14}[）)]\s*$")


def _normalize_step(text: str) -> str:
    """步骤归一化：strip + 去掉序号前缀 + 压缩空白。"""
    text = text.strip()
    text = STEP_PREFIX_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_annotation_line(text: str) -> bool:
    """判断一行是否是「注释/方法标签」而非真正的解题步骤。

    判定标准：
    - 空行 → True；
    - 整行就是括注（``(直接积分)``）→ True；
    - 纯中文短语、不含数学符号、长度短 → True（视为标题/方法标签）。

    这类行会被 :func:`split_steps` 过滤掉，避免污染打分。
    """
    t = text.strip()
    if not t:
        return True
    if ANNOTATION_RE.match(t):
        return True
    # 纯中文短语且不含明显数学符号，通常是标题/方法标签。
    zh_count = len(re.findall(r"[\u4e00-\u9fff]", t))
    if zh_count >= 2 and not MATH_TOKEN_RE.search(t):
        compact = re.sub(r"[\u4e00-\u9fff]", "", t)
        if len(compact.strip()) == 0 or len(t) <= 16:
            return True
    return False


def split_steps(ocr_text: str) -> list[StepItem]:
    """把一段 OCR 文本切成 :class:`StepItem` 列表。

    主流程：
    1. 按 :data:`SEPARATOR_RE`（换行/连续缩进）切段；
    2. 每段做归一化 + 注释行过滤；
    3. 生成 StepItem（``has_equation`` 由 :data:`MATH_TOKEN_RE` 判断，
       ``confidence`` 按「是否有序号前缀」给 0.85 或 0.75）。

    特例：**单行等号链拆分**——若整段被切成 1 步且包含 ``=``，按 ``=``
    切成多个连续的等式步骤，便于逐步评分（例如 ``a=b=c=d`` 拆成
    ``a=b`` / ``b=c`` / ``c=d`` 三步）。
    """
    candidates = [seg.strip() for seg in SEPARATOR_RE.split(ocr_text) if seg.strip()]
    steps: list[StepItem] = []
    for c in candidates:
        if len(c) < 2:
            continue
        normalized = _normalize_step(c)
        if not normalized:
            continue
        if _is_annotation_line(normalized):
            continue
        steps.append(
            StepItem(
                index=len(steps) + 1,
                raw=c,
                normalized=normalized,
                has_equation=bool(MATH_TOKEN_RE.search(normalized)),
                confidence=0.85 if STEP_PREFIX_RE.match(c) else 0.75,
            )
        )

    # 特例：单行等号链 → 按等号拆成多步，便于逐步评分。
    if len(steps) == 1 and "=" in steps[0].normalized:
        chain = [x.strip() for x in steps[0].normalized.split("=") if x.strip()]
        if len(chain) > 2:
            eq_steps: list[StepItem] = []
            left = chain[0]
            for idx, right in enumerate(chain[1:], start=1):
                text = f"{left} = {right}"
                eq_steps.append(
                    StepItem(
                        index=idx,
                        raw=text,
                        normalized=text,
                        has_equation=True,
                        confidence=0.7,  # 等号链拆出来的步骤置信度略低
                    )
                )
                left = right
            return eq_steps
    return steps
