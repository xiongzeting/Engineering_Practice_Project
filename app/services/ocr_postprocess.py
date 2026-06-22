"""OCR 后处理：片段分类、低置信打标、题号切分、阅读顺序整理。

OCR 引擎（cnocr / latex-ocr / pix2text）吐出来的是「片段列表」，每段
带 ``text``、``type``（text/formula）、``bbox``。本模块把这些原始片段
整理成最终 OCR 文本：

- :func:`attach_low_confidence_flag` —— 启发式判断哪些片段「中文夹公式」
  或「LaTeX 括号不闭合」，打 ``low_confidence=True`` 标，供后续视觉纠错；
- :func:`build_ocr_text` —— 把片段拼成干净的整段文本；
- :func:`group_segments_by_question` —— 按 ``1.``/``第 N 题`` 这类题号
  把片段分到各题，方便按题评分；
- :func:`reorder_segments_reading` —— 处理双栏排版，按「先左栏从上到下，
  再右栏」重排片段顺序。
"""
from __future__ import annotations

import re
from typing import Any


# 题号识别正则（如 ``1.`` / ``第 2 题``），用于按题分组。
QUESTION_START_RE = re.compile(r"^\s*(?:第\s*)?(\d{1,2})(?:\s*题|\s*[.、)](?![\d.]))")
# 常见 LaTeX 命令（用于判断片段是公式还是文字）。
LATEX_CMD_RE = re.compile(r"\\\\(?:int|frac|sin|cos|ln|sqrt|pi|sum|lim|cdot|times|left|right|begin|end)")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
MATH_OP_RE = re.compile(r"[=+\-*/^_]")


def _bbox(seg: dict[str, Any]) -> list[int]:
    """取片段的 bbox（``[x1,y1,x2,y2]``），缺失或异常时回退到全 0。"""
    box = seg.get("bbox") or [0, 0, 0, 0]
    if len(box) != 4:
        return [0, 0, 0, 0]
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def reorder_segments_reading(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按阅读顺序重排片段：单栏 → 从上到下；双栏 → 先左栏后右栏。

    双栏检测的启发式：
    - 片段数 ≥ 8；
    - 横向中心点 spread > max_x × 0.35；
    - 左右两栏各自 ≥ 3 个片段。

    满足即视为双栏，否则按单栏处理。
    """
    if not segments:
        return []

    items = [dict(s) for s in segments]
    centers = []
    max_x = 0
    for s in items:
        x1, y1, x2, y2 = _bbox(s)
        max_x = max(max_x, x2)
        centers.append((x1 + x2) / 2)

    # 双栏检测（朴素但稳健的启发式）。
    two_col = False
    if len(centers) >= 8 and max_x > 0:
        cmin, cmax = min(centers), max(centers)
        spread = cmax - cmin
        if spread > max_x * 0.35:
            split = (cmin + cmax) / 2
            left_n = sum(1 for c in centers if c <= split)
            right_n = len(centers) - left_n
            two_col = left_n >= 3 and right_n >= 3

    if two_col:
        # 双栏：左栏先（按 y 升序），再右栏。
        split = (min(centers) + max(centers)) / 2
        left = [s for s in items if (_bbox(s)[0] + _bbox(s)[2]) / 2 <= split]
        right = [s for s in items if s not in left]
        left.sort(key=lambda s: (_bbox(s)[1], _bbox(s)[0]))
        right.sort(key=lambda s: (_bbox(s)[1], _bbox(s)[0]))
        ordered = left + right
    else:
        # 单栏：直接按 (y, x) 排。
        ordered = sorted(items, key=lambda s: (_bbox(s)[1], _bbox(s)[0]))

    # 重置序号为阅读顺序下的 1..N。
    for i, s in enumerate(ordered, start=1):
        s["index"] = i
    return ordered


def refine_segment_type(text: str, raw_type: str) -> str:
    """根据文本内容重新判定片段类型（TEXT/FORMULA/UNKNOWN）。

    OCR 引擎给的 type 不够准（常把中文夹公式判成 FORMULA）。本函数按
    中文字符数、LaTeX 命令数、运算符数、数字字符数重新打标：
    - 含中文且无 LaTeX → TEXT；
    - LaTeX 命令 ≥1，或 (运算符+数字)≥4 且无中文 → FORMULA；
    - 中文 ≥2 且运算符 ≤3 → TEXT；
    - 其他保持原 raw_type。
    """
    t = text.strip()
    if not t:
        return raw_type

    zh = len(CHINESE_RE.findall(t))
    latex = len(LATEX_CMD_RE.findall(t))
    op = len(MATH_OP_RE.findall(t))
    digit = sum(ch.isdigit() for ch in t)

    if zh >= 1 and latex == 0:
        return "TEXT"
    if latex > 0 or (op + digit >= 4 and zh == 0):
        return "FORMULA"
    if zh >= 2 and latex == 0 and op <= 3:
        return "TEXT"
    return raw_type or "UNKNOWN"


def is_noisy_segment(text: str, seg_type: str, score: float) -> bool:
    """判断片段是否是噪声（应被丢弃）。

    判定标准：
    - 空 / 仅 ``$`` / ``$$`` / ``\\[`` / ``\\]``；
    - 非 TEXT 但超长（>200 字符，多半是误识别）；
    - 低置信度（<0.35）且较长（>25 字符）；
    - TEXT 类型但既无中文又很短（≤2 字符，多半是噪声字符）；
    - LaTeX ``\\begin`` 没有对应的 ``\\end``。
    """
    t = text.strip()
    if not t:
        return True

    if t in {"$$", "$", "\\[", "\\]"}:
        return True

    if len(t) > 200 and seg_type != "TEXT":
        return True

    if score > 0 and score < 0.35 and len(t) > 25:
        return True
    if seg_type == "TEXT" and not CHINESE_RE.search(t) and len(t) <= 2:
        return True

    # \begin{...} 没有配对的 \end{...} → LaTeX 残缺，视为噪声。
    if t.count("\\begin") >= 1 and t.count("\\end") == 0:
        return True

    return False


def attach_low_confidence_flag(segments: list[dict[str, Any]], threshold: float = 0.78) -> list[dict[str, Any]]:
    """给每个片段附 ``low_confidence`` 与 ``noisy`` 标志，并重判定 type。

    - ``low_confidence=True``：OCR score 低于 ``threshold``（默认 0.78），
      后续可能需要视觉纠错；
    - ``noisy=True``：噪声片段，应被丢弃。
    """
    out: list[dict[str, Any]] = []
    for seg in reorder_segments_reading(segments):
        score = float(seg.get("score", 0.0) or 0.0)
        text = str(seg.get("text", "") or "").strip()
        raw_type = str(seg.get("type", "UNKNOWN") or "UNKNOWN").upper()
        refined_type = refine_segment_type(text, raw_type)

        item = dict(seg)
        item["type_raw"] = raw_type
        item["type"] = refined_type
        item["low_confidence"] = score < threshold if score > 0 else False
        item["noisy"] = is_noisy_segment(text, refined_type, score)
        out.append(item)
    return out


def build_ocr_text(segments: list[dict[str, Any]]) -> str:
    """把片段拼成干净的 OCR 文本字符串。

    丢噪声片段；FORMULA 直接拼接；TEXT 仅当中文数 1-18 时保留（太长的
    段落多半是误识别，不进评分文本）。
    """
    # 先按题号重排，避免片段顺序乱导致题号定位错。
    ordered = reorder_by_question_number(segments)
    lines: list[str] = []
    for seg in ordered:
        if seg.get("noisy"):
            continue
        text = str(seg.get("text", "") or "").strip()
        if not text:
            continue

        seg_type = str(seg.get("type", "UNKNOWN")).upper()
        if seg_type == "FORMULA":
            lines.append(text)
            continue

        # 保留短中文提示，但太长的段落多半是噪声，不进评分文本。
        zh_len = len(CHINESE_RE.findall(text))
        if seg_type == "TEXT" and 1 <= zh_len <= 18:
            lines.append(text)

    return "\n".join(lines)


def group_segments_by_question(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把片段按题号分组，每组生成 ``{question_no, segments, text}``。

    无题号的片段归入当前题；没有显式题号时按 1, 2, 3 顺序自动编号。
    """
    groups: list[dict[str, Any]] = []
    current = {"question_no": 0, "segments": []}

    for seg in segments:
        if seg.get("noisy"):
            continue
        text = str(seg.get("text", "")).strip()
        cleaned = text.lstrip("$").strip()
        match = QUESTION_START_RE.match(cleaned)

        if match:
            # 新题号：把当前组收尾，开新组。
            if current["segments"]:
                groups.append(current)
            current = {"question_no": int(match.group(1)), "segments": [seg]}
        else:
            current["segments"].append(seg)

    if current["segments"]:
        groups.append(current)

    # 没识别到题号的组，按 1, 2, ... 自动编号。
    for idx, g in enumerate(groups, start=1):
        if not g["question_no"]:
            g["question_no"] = idx
        g["text"] = build_ocr_text(g["segments"])

    groups.sort(key=lambda g: g["question_no"])
    return groups


def reorder_by_question_number(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """先按题号分组，再按题号升序把片段展平重排。

    用于解决「OCR 片段顺序与题号顺序不一致」的情况（常见于双栏图片被
    按行扫，而不是按栏扫）。
    """
    ordered = reorder_segments_reading(segments)
    grouped: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    qnos: list[int] = []

    for seg in ordered:
        if seg.get("noisy"):
            continue
        text = str(seg.get("text", "")).strip().lstrip("$")
        m = QUESTION_START_RE.match(text)
        if m:
            if cur:
                grouped.append(cur)
            cur = [seg]
            qnos.append(int(m.group(1)))
        else:
            cur.append(seg)
    if cur:
        grouped.append(cur)

    if not grouped:
        return ordered

    # 至少识别到 2 个题号才做按号重排，否则维持原阅读顺序。
    if len(qnos) >= 2:
        paired = list(zip(grouped, qnos + [999] * (len(grouped) - len(qnos))))
        paired.sort(key=lambda x: x[1])
        flat: list[dict[str, Any]] = []
        for grp, _ in paired:
            flat.extend(grp)
        return flat
    return ordered
