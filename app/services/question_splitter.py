"""题号识别：从 OCR 文本里切出每道题。

匹配 ``第 N 题`` / ``N.`` / ``N、`` 这三类中文作业最常见的题号写法，
:func:`split_text_into_questions` 按这些题号把整段 OCR 文本切成
``[(题号, 该题文本), ...]`` 列表。找不到题号就当作一整道题。

这个模块服务于两条路径：
1. cnocr/pix2text OCR 完后按题切，再逐题评分；
2. 视觉 LLM 路径如果 review 失败 fallback 到文本切分。
"""
from __future__ import annotations

import re

# 题号识别正则：按优先级匹配所有常见写法。
# 注意：括号格式 ``(\d+)`` 和数字开头格式都要保留，否则用 (1)(2)(3) 编号
# 的作业会被全部合并成"1 道题"。
QUESTION_START_RES = [
    re.compile(r"^\s*第\s*(\d+)\s*题"),                     # 第 1 题
    re.compile(r"^\s*\(\s*(\d+)\s*\)\s*\S"),                # (1)xxx 括号题号 + 后跟内容
    re.compile(r"^\s*\(\s*(\d+)\s*\)\s*$"),                 # (1) 单独占行
    re.compile(r"^\s*(\d+)\s*[.、)]\s*\S"),                 # 1. / 1、 / 1) 后跟内容
    re.compile(r"^\s*(\d+)\s*[.、)]\s*$"),                  # 1. 单独一行（题号占行）
]


def detect_question_number(line: str) -> int | None:
    """从一行文本里识别题号。匹配范围 [1, 200]，超出视为误识别。"""
    text = line.strip()
    if not text:
        return None
    for pattern in QUESTION_START_RES:
        m = pattern.match(text)
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 200:
                    return n
            except (ValueError, IndexError):
                continue
    return None


def split_text_into_questions(ocr_text: str) -> list[tuple[int, str]]:
    """按题号切分 OCR 文本，返回 ``[(题号, 该题文本), ...]``。

    逐行扫描：发现题号就开新段，否则把当前行追加到当前段。最后一段会
    在 ``_flush()`` 里收尾。

    如果整段都没识别到题号，视为「只有一题」，整段返回 ``[(1, text)]``。
    """
    if not ocr_text or not ocr_text.strip():
        return []
    lines = ocr_text.splitlines()
    questions: list[tuple[int, list[str]]] = []
    current_qno: int | None = None
    current_buf: list[str] = []

    def _flush() -> None:
        """把当前 buffer 收尾成一段，追加到 questions。"""
        nonlocal current_buf
        if current_buf and any(s.strip() for s in current_buf):
            text = "\n".join(s for s in current_buf if s.strip()).strip()
            if text:
                qno = current_qno if current_qno is not None else 1
                questions.append((qno, text))
        current_buf = []

    for ln in lines:
        qno = detect_question_number(ln)
        if qno is not None:
            # 新题号出现：冲掉上一段，开始新段。
            _flush()
            current_qno = qno
            current_buf = [ln]
        else:
            current_buf.append(ln)
    _flush()  # 冲掉最后一段

    if not questions:
        # 整段都没题号：当作一整道题。
        return [(1, ocr_text.strip())]
    return questions
