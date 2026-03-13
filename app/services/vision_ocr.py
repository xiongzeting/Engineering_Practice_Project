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


def _image_bytes_to_data_url(image_bytes: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def vision_only_ocr(image_bytes: bytes, return_debug: bool = False) -> dict[str, Any]:
    if not settings.llm_api_key:
        return {"ok": False, "error": "LLM_API_KEY 未配置。"}
    image_url = _image_bytes_to_data_url(image_bytes)
    if not image_url:
        return {"ok": False, "error": "图片读取失败。"}

    prompt = (
        "请对整张数学作业图片进行OCR，输出清晰有序文本。\n"
        "要求：\n"
        "1) 保留题号顺序（从上到下，左到右）。\n"
        "2) 中文说明保持中文，不要误写成数字或符号。\n"
        "3) 公式用简洁LaTeX表达（不要无意义超长串）。\n"
        "4) 删除明显噪声片段。\n"
        "返回严格JSON：{\"lines\": [\"...\"], \"notes\": \"...\"}\n"
    )
    body = {
        "model": settings.llm_model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": "你是数学OCR助手，只返回JSON。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    try:
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {settings.llm_api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=settings.vision_timeout_sec,
        )
        if not resp.ok:
            return {"ok": False, "error": f"LLM请求失败: {resp.status_code} {resp.text[:200]}"}
        raw = _extract_chat_content(resp.json())
        parsed = _safe_json_parse(raw)
        if not parsed:
            return {
                "ok": False,
                "error": "LLM返回无法解析为JSON。",
                "raw_preview": raw[: settings.llm_debug_max_chars],
            }
        lines = parsed.get("lines", [])
        if not isinstance(lines, list):
            lines = []
        ocr_text = "\n".join(str(x).strip() for x in lines if str(x).strip()).strip()
        result: dict[str, Any] = {
            "ok": True,
            "engine": f"vision-only:{settings.llm_model}",
            "ocr_text": ocr_text,
            "notes": str(parsed.get("notes", "") or ""),
            "llm_configured": True,
        }
        if return_debug:
            result["llm_debug"] = {
                "request_path": "/chat/completions",
                "request_preview": prompt[: settings.llm_debug_max_chars],
                "response_preview": raw[: settings.llm_debug_max_chars],
                "parsed_preview": json.dumps(parsed, ensure_ascii=False)[: settings.llm_debug_max_chars],
            }
        return result
    except Exception as e:
        return {"ok": False, "error": f"调用异常: {type(e).__name__}: {e}"}
