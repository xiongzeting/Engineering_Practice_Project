"""版面分析：从原图里抠出每一行/每一个公式块。

在 cnocr + latex-ocr 双引擎流水线里，需要先知道「这块区域是文字还是
公式」才能选合适的引擎。本模块用 OpenCV 的形态学操作找连通区域，
:func:`_merge_boxes` 把相邻的bbox合并，:func:`_sort_reading_order`
按「上→下、左→右」排成阅读顺序，最后 :func:`segment_formula_regions`
返回 (x, y, w, h) 元组列表。
"""
from __future__ import annotations

import cv2
import numpy as np


def _merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = 8) -> list[tuple[int, int, int, int]]:
    """把互相靠近 / 微重叠的 bbox 合并成大 bbox。

    判定逻辑（同时满足两条）：
    - x/y 方向投影重叠都 > ``-gap``（即允许间距不超过 gap）；
    - 水平或垂直方向有一侧"贴得很近"（``close_h`` / ``close_v``），或者
      真的投影重叠。

    外层 ``while changed`` 反复合并，直到一轮下来没有新合并为止。
    返回 ``[(x, y, w, h), ...]``。
    """
    merged = boxes[:]
    changed = True
    while changed:
        changed = False
        output: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)
        for i, a in enumerate(merged):
            if used[i]:
                continue
            ax, ay, aw, ah = a
            x1, y1, x2, y2 = ax, ay, ax + aw, ay + ah
            # 找所有可与 a 合并的 j，逐个并入 a。
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                bx, by, bw, bh = merged[j]
                bx1, by1, bx2, by2 = bx, by, bx + bw, by + bh
                overlap_x = min(x2, bx2) - max(x1, bx1)
                overlap_y = min(y2, by2) - max(y1, by1)
                close_h = abs(by1 - y1) <= gap or abs(y2 - by2) <= gap
                close_v = abs(bx1 - x1) <= gap or abs(x2 - bx2) <= gap
                if (overlap_x > -gap and overlap_y > -gap) and (close_h or close_v or overlap_x > 0 or overlap_y > 0):
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    used[j] = True
                    changed = True
            used[i] = True
            output.append((x1, y1, x2 - x1, y2 - y1))
        merged = output
    return merged


def _sort_reading_order(boxes: list[tuple[int, int, int, int]], image_h: int) -> list[tuple[int, int, int, int]]:
    """按「行优先，行内按 x 升序」把 bbox 排成阅读顺序。

    思路：按中心点 y 把盒子分到若干"行"（同一行的 y 差在 ``row_tol`` 内），
    每次遇到落在容差内的现有行就并进去，并增量更新行中心 y（加权平均）。
    最后按行 y 排序，行内按 x 排序展平返回。

    ``row_tol`` 自适应：至少 14 像素，最大为图高 / 80。
    """
    if not boxes:
        return boxes
    row_tol = max(14, image_h // 80)
    rows: list[dict] = []

    for b in sorted(boxes, key=lambda x: (x[1] + x[3] // 2, x[0])):
        cy = b[1] + b[3] // 2
        placed = False
        for row in rows:
            if abs(cy - row["y"]) <= row_tol:
                row["boxes"].append(b)
                # 增量更新行中心 y，保证后续相近的盒子能继续并进来。
                row["y"] = int((row["y"] * row["count"] + cy) / (row["count"] + 1))
                row["count"] += 1
                placed = True
                break
        if not placed:
            rows.append({"y": cy, "boxes": [b], "count": 1})

    rows.sort(key=lambda r: r["y"])
    ordered: list[tuple[int, int, int, int]] = []
    for row in rows:
        ordered.extend(sorted(row["boxes"], key=lambda x: x[0]))
    return ordered


def segment_formula_regions(image_bytes: bytes, max_segments: int = 80) -> list[tuple[int, int, int, int]]:
    """从原图里抠出每一行 / 每一个公式的 bbox。

    流水线：
    1. **解码 + 二值化（反相）**：高斯模糊去噪后做自适应二值化，
       ``THRESH_BINARY_INV`` 让文字为白、背景为黑（便于找连通域）；
    2. **形态学闭运算**：用一个扁宽的矩形 kernel（宽 ≈ w/42，高 ≈ h/360）
       把同一行的字符粘到一起，形成"行带"；
    3. **轮廓提取 + 过滤**：丢弃太小（面积/高/宽不足）或太大（几乎整页，
       多半是边框）的轮廓；
    4. **合并 + 排序**：调 :func:`_merge_boxes` 合并相邻盒子，
       :func:`_sort_reading_order` 排成阅读顺序。

    最终返回前 ``max_segments`` 个 bbox（防止极端情况吐出几千个碎块）。
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        return []

    h, w = gray.shape
    area = h * w

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 9
    )
    # 形态学闭运算：扁宽 kernel 把同行字符粘成"行带"，便于后续当一行识别。
    kernel_w = max(18, w // 42)
    kernel_h = max(3, h // 360)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 阈值自适应图大小：太小肯定不是文字，太大可能是边框/表格。
    min_area = max(260, int(area * 0.00008))
    min_h = max(14, h // 140)
    min_w = max(28, w // 120)

    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        x, y, bw_, bh_ = cv2.boundingRect(cnt)
        box_area = bw_ * bh_
        if box_area < min_area:
            continue
        if bh_ < min_h or bw_ < min_w:
            continue
        # 占满 95% 宽 + 35% 高的轮廓多半是整页边框，丢弃。
        if bw_ > w * 0.95 and bh_ > h * 0.35:
            continue
        boxes.append((x, y, bw_, bh_))

    boxes = _merge_boxes(boxes, gap=max(8, w // 180))
    boxes = _sort_reading_order(boxes, h)
    return boxes[:max_segments]
