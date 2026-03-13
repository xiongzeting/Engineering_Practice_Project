from __future__ import annotations

import cv2
import numpy as np


def _merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int = 8) -> list[tuple[int, int, int, int]]:
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
    kernel_w = max(18, w // 42)
    kernel_h = max(3, h // 360)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, kernel_h))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        if bw_ > w * 0.95 and bh_ > h * 0.35:
            continue
        boxes.append((x, y, bw_, bh_))

    boxes = _merge_boxes(boxes, gap=max(8, w // 180))
    boxes = _sort_reading_order(boxes, h)
    return boxes[:max_segments]
