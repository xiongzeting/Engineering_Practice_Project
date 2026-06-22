"""文本 LLM 二次纠错（OCR 后处理）。

调用 GPT 类文本模型（我们用 deepseek-v4-pro）对纯文本 OCR 结果做后处理，
主要解决：中文/数学符号混淆、LaTeX 括号不闭合、题号识别错乱等。是
``use_llm_correction=True`` 路径的实现。

输入：OCR 出的纯文本；输出：纠错后的纯文本。和 :mod:`vision_corrector`
的区别是——这里**不再看图**，只对文本做语义级修复，便宜很多。
"""
from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.config import settings


def _extract_output_text(payload: dict[str, Any]) -> str:
    """从 ``/responses`` 风格的响应里抽取文本。

    Responses API 的结构是 ``payload['output']`` 是一个 message 列表，
    每个 message 的 ``content`` 里可能有 ``output_text`` 块。把所有
    output_text 文本拼起来返回。
    """
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
    """从 ``/chat/completions`` 风格的响应里抽取文本。

    ``choices[0].message.content`` 有两种形态：纯字符串，或 OpenAI 新版
    的 list-of-blocks（每个块带 ``type``/``text``）。两种都要处理，
    否则不同供应商会踩坑。
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
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    """三层 fallback 的 JSON 解析器。

    1. 直接 ``json.loads``：模型守规矩、整段是 JSON 时直接命中；
    2. 用正则 ``\\{[\\s\\S]*\\}`` 抠出最外层 ``{...}`` 再 parse：
       应对模型前后加了解释文字（"以下是结果：{...}"）的情况；
    3. 都失败就返回 None，由调用方决定 fallback。
    """
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
    """用文本 LLM 对纯文本 OCR 结果做语义级纠错。

    与 :mod:`vision_corrector` 的区别：这里**不再看图**，只读 OCR 文本，
    适合做轻量修复（中文/公式混淆、LaTeX 括号、题号错乱）。便宜很多，
    但能力远弱于视觉模型路径。

    流程：
    1. 构造任务 payload（``task`` + ``rules`` + ``ocr_text`` + ``segments``），
       rules 强制"只返回 JSON、不增题"；
    2. 优先 POST ``/responses``，失败再回退到 ``/chat/completions``；
    3. 用 :func:`_safe_json_parse` 抽 JSON，读 ``corrected_lines`` 拼回文本；
    4. ``return_debug=True`` 时额外返回请求/响应预览，供前端调试面板展示。

    无 key / 空 OCR / 任意异常都返回 ``None``，由调用方优雅降级。
    """
    if not settings.llm_score_api_key:
        return None  # 没配 key：直接跳过，由上游走其他路径
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
        # segments 截断到前 160 段，避免 token 爆炸。
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
    # 两种请求体：Responses API 用 input 字段；Chat Completions 用 messages。
    responses_body = {
        "model": settings.llm_score_model,
        "input": json.dumps(payload, ensure_ascii=False),
    }
    chat_body = {
        "model": settings.llm_score_model,
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
        "temperature": 0.1,  # 低温度：纠错任务要稳定，不要创造性
        "max_tokens": 2500,
    }

    try:
        headers = {
            "Authorization": f"Bearer {settings.llm_score_api_key}",
            "Content-Type": "application/json",
        }

        text = ""
        request_path = ""
        error_note = ""

        # 先打 Responses API。它是 OpenAI 较新的接口，结构化输出更稳。
        resp = requests.post(
            f"{settings.llm_score_base_url.rstrip('/')}/responses",
            headers=headers,
            json=responses_body,
            timeout=timeout,
        )
        if resp.ok:
            request_path = "/responses"
            data = resp.json()
            text = _extract_output_text(data)
        else:
            # Responses 失败（旧供应商可能不支持），回退到 Chat Completions。
            error_note = f"/responses:{resp.status_code}"
            chat_resp = requests.post(
                f"{settings.llm_score_base_url.rstrip('/')}/chat/completions",
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
            return None  # 模型输出不是 JSON：放弃，让上游用原 OCR 文本
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
            # 调试模式：落库 + 前端面板用，按 debug_max_chars 截断。
            out["debug"] = {
                "request_path": request_path,
                "fallback_note": error_note,
                "request_preview": json.dumps(payload, ensure_ascii=False)[:debug_max_chars],
                "response_preview": text[:debug_max_chars],
                "parsed_preview": json.dumps(parsed, ensure_ascii=False)[:debug_max_chars],
            }
        return out
    except Exception:
        return None  # 网络/解析异常：静默降级，不影响主流程
