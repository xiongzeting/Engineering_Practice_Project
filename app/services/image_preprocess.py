from __future__ import annotations

import cv2
import numpy as np


def _estimate_skew_angle(binary_inv: np.ndarray) -> float:
    coords = np.column_stack(np.where(binary_inv > 0))
    if coords.shape[0] < 60:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(gray, (3, 3), 0)
    bw_inv = cv2.adaptiveThreshold(
        denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 10
    )

    angle = _estimate_skew_angle(bw_inv)
    if abs(angle) > 0.4 and abs(angle) < 15:
        h, w = gray.shape[:2]
        m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        gray = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Contrast enhancement for faint handwriting.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ok, out = cv2.imencode(".png", enhanced)
    if not ok:
        return image_bytes
    return out.tobytes()
