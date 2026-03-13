from __future__ import annotations

import re
from typing import Any


QUESTION_START_RE = re.compile(r"^\s*(?:第\s*)?(\d{1,2})\s*[.、)\]:：]?\s*")
LATEX_CMD_RE = re.compile(r"\\\\(?:int|frac|sin|cos|ln|sqrt|pi|sum|lim|cdot|times|left|right|begin|end)")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
MATH_OP_RE = re.compile(r"[=+\-*/^_]")


def _bbox(seg: dict[str, Any]) -> list[int]:
    box = seg.get("bbox") or [0, 0, 0, 0]
    if len(box) != 4:
        return [0, 0, 0, 0]
    return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def reorder_segments_reading(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []

    items = [dict(s) for s in segments]
    centers = []
    max_x = 0
    for s in items:
        x1, y1, x2, y2 = _bbox(s)
        max_x = max(max_x, x2)
        centers.append((x1 + x2) / 2)

    # Two-column detection (simple, robust heuristic).
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
        split = (min(centers) + max(centers)) / 2
        left = [s for s in items if (_bbox(s)[0] + _bbox(s)[2]) / 2 <= split]
        right = [s for s in items if s not in left]
        left.sort(key=lambda s: (_bbox(s)[1], _bbox(s)[0]))
        right.sort(key=lambda s: (_bbox(s)[1], _bbox(s)[0]))
        ordered = left + right
    else:
        ordered = sorted(items, key=lambda s: (_bbox(s)[1], _bbox(s)[0]))

    for i, s in enumerate(ordered, start=1):
        s["index"] = i
    return ordered


def refine_segment_type(text: str, raw_type: str) -> str:
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

    if t.count("\\begin") >= 1 and t.count("\\end") == 0:
        return True

    return False


def attach_low_confidence_flag(segments: list[dict[str, Any]], threshold: float = 0.78) -> list[dict[str, Any]]:
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
    # Reorder by detected question number first, then by visual order.
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

        # Keep short Chinese hints but avoid very long paragraph noise in scoring text.
        zh_len = len(CHINESE_RE.findall(text))
        if seg_type == "TEXT" and 1 <= zh_len <= 18:
            lines.append(text)

    return "\n".join(lines)


def group_segments_by_question(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    current = {"question_no": 0, "segments": []}

    for seg in segments:
        if seg.get("noisy"):
            continue
        text = str(seg.get("text", "")).strip()
        cleaned = text.lstrip("$").strip()
        match = QUESTION_START_RE.match(cleaned)

        if match:
            if current["segments"]:
                groups.append(current)
            current = {"question_no": int(match.group(1)), "segments": [seg]}
        else:
            current["segments"].append(seg)

    if current["segments"]:
        groups.append(current)

    for idx, g in enumerate(groups, start=1):
        if not g["question_no"]:
            g["question_no"] = idx
        g["text"] = build_ocr_text(g["segments"])

    groups.sort(key=lambda g: g["question_no"])
    return groups


def reorder_by_question_number(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
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

    if len(qnos) >= 2:
        paired = list(zip(grouped, qnos + [999] * (len(grouped) - len(qnos))))
        paired.sort(key=lambda x: x[1])
        flat: list[dict[str, Any]] = []
        for grp, _ in paired:
            flat.extend(grp)
        return flat
    return ordered
