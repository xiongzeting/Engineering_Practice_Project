from __future__ import annotations

import re

from app.schemas import StepItem

STEP_PREFIX_RE = re.compile(r"^\s*(?:\(?\d+\)|\d+[.)]|第\s*\d+\s*步[:：]?)\s*")
SEPARATOR_RE = re.compile(r"(?:\r?\n)+|(?:\s{2,})")
MATH_TOKEN_RE = re.compile(r"(=|\\int|\\frac|\\sum|\\sin|\\cos|\\tan|\\ln|\\sqrt|\^|_|\+|\-|\*|/)")
ANNOTATION_RE = re.compile(r"^\s*[（(][^（）()]{1,14}[）)]\s*$")


def _normalize_step(text: str) -> str:
    text = text.strip()
    text = STEP_PREFIX_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_annotation_line(text: str) -> bool:
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

    # OCR 文本中如果是单行等号链，进一步拆分保障顺序化
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
                        confidence=0.7,
                    )
                )
                left = right
            return eq_steps
    return steps
