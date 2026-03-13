from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.config import settings


def _extract_output_text(payload: dict[str, Any]) -> str:
    out = payload.get("output", [])
    chunks: list[str] = []
    for item in out:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunks).strip()


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
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    text = text.strip()
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


def llm_correct_ocr_text(
    ocr_text: str,
    segments: list[dict[str, Any]],
    timeout: int = 45,
    return_debug: bool = False,
    debug_max_chars: int = 1200,
) -> dict[str, Any] | None:
    if not settings.llm_api_key:
        return None
    if not ocr_text.strip():
        return None

    payload = {
        "task": "修正数学作业OCR结果中的中文与公式混淆，保持题目顺序。",
        "rules": [
            "优先保证中文自然可读，避免把中文错改成数字或符号。",
            "公式使用简洁LaTeX，保留核心符号与运算结构。",
            "删除明显噪声片段（孤立$$、无意义乱码）。",
            "保持原有题号顺序，不要新增题目。",
            "仅返回 JSON：{corrected_lines:[...], notes:\"...\"}",
        ],
        "ocr_text": ocr_text,
        "segments": [
            {
                "index": s.get("index"),
                "type": s.get("type"),
                "score": s.get("score"),
                "text": s.get("text"),
            }
            for s in segments[:160]
        ],
    }
    responses_body = {
        "model": settings.llm_model,
        "input": json.dumps(payload, ensure_ascii=False),
    }
    chat_body = {
        "model": settings.llm_model,
        "messages": [
            {
                "role": "system",
                "content": "你是数学作业OCR纠错器。只返回JSON，不要解释。",
            },
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        "temperature": 0.1,
    }

    try:
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        text = ""
        request_path = ""
        error_note = ""

        resp = requests.post(
            f"{settings.llm_base_url.rstrip('/')}/responses",
            headers=headers,
            json=responses_body,
            timeout=timeout,
        )
        if resp.ok:
            request_path = "/responses"
            data = resp.json()
            text = _extract_output_text(data)
        else:
            error_note = f"/responses:{resp.status_code}"
            chat_resp = requests.post(
                f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=chat_body,
                timeout=timeout,
            )
            if not chat_resp.ok:
                return None
            request_path = "/chat/completions"
            data = chat_resp.json()
            text = _extract_chat_content(data)

        parsed = _safe_json_parse(text)
        if not parsed:
            return None
        lines = parsed.get("corrected_lines", [])
        if not isinstance(lines, list):
            return None
        cleaned = "\n".join(str(x).strip() for x in lines if str(x).strip()).strip()
        if not cleaned:
            return None
        notes = str(parsed.get("notes", "")).strip()
        out: dict[str, Any] = {
            "corrected_text": cleaned,
            "notes": notes,
        }
        if return_debug:
            out["debug"] = {
                "request_path": request_path,
                "fallback_note": error_note,
                "request_preview": json.dumps(payload, ensure_ascii=False)[:debug_max_chars],
                "response_preview": text[:debug_max_chars],
                "parsed_preview": json.dumps(parsed, ensure_ascii=False)[:debug_max_chars],
            }
        return out
    except Exception:
        return None
