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
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    if len(bbox) != 4:
        return None
    w, h = image.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
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
    if not settings.llm_api_key:
        return None
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.llm_model,
        "temperature": 0.0,
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
                "Authorization": f"Bearer {settings.llm_api_key}",
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
    if not image_bytes or not settings.llm_api_key:
        return segments, {"enabled": False, "corrected_count": 0}

    updated = [dict(s) for s in segments]
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
            continue
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
            seg["noisy"] = True
            corrected_count += 1
            continue
        if conf < 0.35 or not new_text:
            continue

        seg["text"] = new_text
        if new_type in ("TEXT", "FORMULA"):
            seg["type"] = new_type
        seg["score"] = max(float(seg.get("score", 0.0) or 0.0), conf)
        seg["low_confidence"] = conf < 0.78
        seg["vision_corrected"] = True
        corrected_count += 1

    return updated, {"enabled": True, "attempted": attempted, "corrected_count": corrected_count}
