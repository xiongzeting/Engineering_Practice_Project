"""图像预处理：去噪、纠斜、增强对比度。

OCR 之前对扫描件 / 拍照图做统一处理，让后续识别更稳：
- 先转灰度再自适应二值化，去除浅色背景；
- :func:`_estimate_skew_angle` 用 ``cv2.minAreaRect`` 估整页的倾斜角；
- 旋转回正后再做一次轻度锐化，突出笔画边缘。

只在 cnocr / pix2text 流水线里用，纯视觉 LLM 路径（
:mod:`vision_ocr`）会把原图直接喂给多模态模型，不做 OpenCV 预处理。
"""
from __future__ import annotations

import cv2
import numpy as np


def _estimate_skew_angle(binary_inv: np.ndarray) -> float:
    """估计二值反相图的整页倾斜角度（度）。

    思路：把所有前景点（>0 的像素）拿出来，用 ``cv2.minAreaRect``
    拟合最小外接矩形，矩形的 ``angle`` 就是倾斜角。

    前景点数 < 60 直接返回 0（图基本空白，估计毫无意义）。
    OpenCV 的 rect 角度范围是 [-90, 0)，若 < -45 需要换算到 [0, 90)
    才符合「顺时针为正」的直觉。
    """
    coords = np.column_stack(np.where(binary_inv > 0))
    if coords.shape[0] < 60:
        return 0.0
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle
    return float(angle)


def preprocess_image_bytes(image_bytes: bytes) -> bytes:
    """对原始图片字节做去噪 / 纠斜 / 对比度增强。

    流水线（任意一步失败都优雅回退到原图）：
    1. **解码**：``cv2.imdecode`` 把 bytes 解成 BGR 矩阵；
    2. **灰度 + 去噪**：转灰度后做 3×3 高斯模糊，去掉扫描纹/细噪点；
    3. **自适应二值化（反相）**：高斯加权 + 31×31 block + C=10，
       对光照不均的拍照作业友好，反相后文字是白点（便于点集拟合）；
    4. **估计倾斜角**：调 :func:`_estimate_skew_angle`，角落在
       [0.4°, 15°] 才校正——太小无意义，太大可能本来就不是文字行；
    5. **旋转**：以中心为锚做仿射变换，``BORDER_REPLICATE`` 避免黑边；
    6. **CLAHE 对比度增强**：局部直方图均衡，让浅色笔画更清晰。

    最终结果编码成 PNG bytes 返回，喂给后续 OCR 引擎。
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes  # 解码失败：优雅回退

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

    # 对浅色手写做对比度增强（CLAHE：局部自适应直方图均衡）。
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    ok, out = cv2.imencode(".png", enhanced)
    if not ok:
        return image_bytes
    return out.tobytes()
