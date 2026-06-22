"""视觉 LLM 二次纠错（OCR 后处理）。

和 :mod:`ocr_corrector` 相对——这里**带着原图区域**调用多模态模型，
让模型「看着」图来判断 OCR 结果是否合理。当 OCR 给出的置信度低
（:func:`attach_low_confidence_flag` 打标的片段）时，会针对那些片段
单独抠图、走 :func:`_call_vision_llm` 重新识别。

代价是 API 调用次数 = 低置信片段数，比纯文本纠错贵得多。所以只在
``use_vision_correction=true`` 时启用，默认关闭。
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import requests
from PIL import Image

from app.config import settings


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    """三层 fallback 的 JSON 解析器（视觉纠错版）。

    1. 去掉模型可能加的 markdown 代码块包裹（```json ... ```）；
    2. 直接 ``json.loads``；
    3. 用正则抠最外层 ``{...}`` 再 parse。

    任一步成功即返回；都失败返回 None。
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _extract_chat_content(payload: dict[str, Any]) -> str:
    """从 ``/chat/completions`` 响应里抽取 ``choices[0].message.content``。

    兼容两种 content 形态：纯字符串或 list-of-blocks（OpenAI 新版）。
    """
    choices = payload.get("choices", [])
    if not choices:
        return ""
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _crop_to_data_url(image_bytes: bytes, bbox: list[int], pad: int = 6) -> str | None:
    """按 bbox 从原图裁出一小块，编码成 ``data:image/png;base64,...``。

    多模态 API 接受两种图片入参：URL 或 data URL。我们没法对外暴露图床，
    所以用 data URL 直接把 base64 内嵌进请求体。

    - ``pad``：bbox 四周外扩若干像素，避免笔画贴边被切掉；
    - 边界裁剪到 [0, w] / [0, h]；
    - 裁剪后尺寸为 0（bbox 异常）返回 None。
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    if len(bbox) != 4:
        return None
    w, h = image.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # 四周外扩 pad 像素，并裁到合法范围。
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop((x1, y1, x2, y2))
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _call_vision_llm(image_url: str, ocr_text: str, seg_type: str, idx: int) -> dict[str, Any] | None:
    """把单个低置信片段（含裁剪图）发给视觉 LLM 做二次识别。

    请求格式（多模态 chat completions）：
    - system：约束"只返回 JSON: ``{corrected_text, corrected_type, confidence, keep}``"；
    - user：包含文本（原 OCR + 类型 + 序号 + 规则）+ 图片 data URL。

    成功返回解析后的 dict，失败（无 key / 网络 / 解析）返回 None。
    """
    if not settings.llm_vision_api_key:
        return None
    url = f"{settings.llm_vision_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_vision_model,
        "temperature": 0.0,  # 纠错要稳定，不要创造性
        "max_tokens": 800,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是数学作业OCR纠错器。根据图片纠正当前单个OCR块。"
                    "返回严格JSON: {corrected_text, corrected_type, confidence, keep}. "
                    "corrected_type 仅可为 TEXT 或 FORMULA。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"块序号: {idx}\n"
                            f"当前类型: {seg_type}\n"
                            f"当前OCR: {ocr_text}\n"
                            "规则: 中文自然可读优先；公式保持简洁LaTeX；若是噪声则 keep=false。"
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.llm_vision_api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.vision_timeout_sec,
        )
        if not resp.ok:
            return None
        text = _extract_chat_content(resp.json())
        parsed = _safe_json_parse(text)
        return parsed
    except Exception:
        return None


def apply_vision_correction(
    image_bytes: bytes | None, segments: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """对低置信片段批量做视觉 LLM 二次识别。

    挑选规则（任一满足即送纠错）：
    - ``low_confidence=True``（OCR score < 0.78）；
    - ``noisy=True``（疑似噪声）；
    - ``type=='LINE'``（pix2text 给出的行级片段，可能拼错）。

    单片段流程：
    1. :func:`_crop_to_data_url` 按 bbox 裁图；
    2. :func:`_call_vision_llm` 调模型，拿到 ``{keep, confidence, corrected_text, corrected_type}``；
    3. ``keep=False`` → 标 noisy（丢弃）；
    4. ``confidence<0.35`` 或空文本 → 维持原状；
    5. 否则覆盖 text/type，把 score 上取（避免越改越低），打 ``vision_corrected=True``。

    返回 ``(updated_segments, summary)``，summary 用于日志和前端展示。
    单片段最多 ``vision_max_segments``（默认 20）次调用，避免成本失控。
    """
    if not image_bytes or not settings.llm_vision_api_key:
        # 没图或没 key：完全跳过，原样返回。
        return segments, {"enabled": False, "corrected_count": 0}

    updated = [dict(s) for s in segments]
    # 只挑低置信/噪声/行级片段送视觉纠错——其它片段本来就稳。
    candidates = [
        s for s in updated if s.get("low_confidence") or s.get("noisy") or str(s.get("type", "")).upper() == "LINE"
    ]
    candidates = candidates[: settings.vision_max_segments]

    corrected_count = 0
    attempted = 0
    for seg in candidates:
        idx = int(seg.get("index", 0))
        bbox = seg.get("bbox") or [0, 0, 0, 0]
        data_url = _crop_to_data_url(image_bytes, bbox)
        if not data_url:
            continue  # bbox 异常或裁图失败：跳过
        attempted += 1
        parsed = _call_vision_llm(
            image_url=data_url,
            ocr_text=str(seg.get("text", "")),
            seg_type=str(seg.get("type", "UNKNOWN")),
            idx=idx,
        )
        if not parsed:
            continue

        keep = bool(parsed.get("keep", True))
        conf = float(parsed.get("confidence", 0.0) or 0.0)
        new_text = str(parsed.get("corrected_text", "") or "").strip()
        new_type = str(parsed.get("corrected_type", "") or "").strip().upper()
        if not keep:
            # 模型明确说"这是噪声"：标 noisy 让后续流水线丢弃。
            seg["noisy"] = True
            corrected_count += 1
            continue
        if conf < 0.35 or not new_text:
            continue  # 模型自己都不确定：保持原状
        # 用纠错结果覆盖。score 取 max，防止模型给低分反而把原高分拉低。
        seg["text"] = new_text
        if new_type in ("TEXT", "FORMULA"):
            seg["type"] = new_type
        seg["score"] = max(float(seg.get("score", 0.0) or 0.0), conf)
        seg["low_confidence"] = conf < 0.78
        seg["vision_corrected"] = True
        corrected_count += 1

    return updated, {"enabled": True, "attempted": attempted, "corrected_count": corrected_count}
