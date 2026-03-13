from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from typing import Any

import requests

from app.config import settings
from app.schemas import StepItem, StepScore


def _basic_validity_score(text: str) -> float:
    score = 100.0
    if text.count("(") != text.count(")"):
        score -= 20
    if not any(ch.isdigit() for ch in text) and "=" not in text:
        score -= 15
    if re.search(r"[+\-*/^=]{2,}", text):
        score -= 10
    return max(score, 0.0)


def _rule_score_steps(steps: list[StepItem], reference_steps: list[StepItem]) -> tuple[list[StepScore], float, str]:
    step_scores: list[StepScore] = []
    ref_texts = [s.normalized for s in reference_steps]

    for step in steps:
        validity = _basic_validity_score(step.normalized)
        similarity = 75.0
        reason = "步骤结构基本合理。"
        if ref_texts:
            best = 0.0
            for ref in ref_texts:
                best = max(best, SequenceMatcher(None, step.normalized, ref).ratio())
            similarity = best * 100
            if best < 0.35:
                reason = "与参考解法差异较大，建议检查推导或计算。"
            elif best < 0.65:
                reason = "与参考解法部分一致，可检查中间变形。"
            else:
                reason = "与参考解法高度一致。"

        step_score = round(0.55 * validity + 0.45 * similarity, 2)
        step_scores.append(StepScore(index=step.index, score=step_score, reason=reason))

    total = round(sum(s.score for s in step_scores) / len(step_scores), 2) if step_scores else 0.0
    feedback = "建议重点核对低于 60 分的步骤。"
    return step_scores, total, feedback


def _llm_score_steps(
    ocr_text: str, steps: list[StepItem], reference: str | None, timeout: int | None = None
) -> dict[str, Any]:
    if not settings.llm_api_key:
        return {"ok": False, "error": "未配置LLM_API_KEY。"}

    step_payload = [{"index": s.index, "text": s.normalized} for s in steps]
    ocr_excerpt = ocr_text[:2000]
    prompt = {
        "task": "你是严格的数学步骤评分器。",
        "rules": [
            "按步骤顺序评分，先判断每一步是否数学上正确。",
            "若某一步正确且表述完整，该步应给100分；不要无理由给90-99。",
            "若首次出现关键逻辑错误（根错误），在该步明显扣分。",
            "根错误之后，若后续步骤只是沿用该错误但推导形式正确，不要重复重罚，可给60-85分。",
            "不要因为同一处前置错误，在后续每步都再次大幅扣分。",
            "每步reason尽量简短（不超过18个汉字）。",
            "以 JSON 返回：step_scores[{index,score,reason}], total_score, feedback",
        ],
        "ocr_text_excerpt": ocr_excerpt,
        "steps": step_payload,
        "reference_solution": reference or "",
    }
    responses_body = {
        "model": settings.llm_model,
        "input": json.dumps(prompt, ensure_ascii=False),
    }
    chat_body = {
        "model": settings.llm_model,
        "messages": [
            {"role": "system", "content": "你是数学步骤评分器。只返回JSON，不要解释。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "stream": settings.llm_score_use_stream,
    }
    try:
        read_timeout = None if settings.llm_score_no_read_timeout else (timeout if timeout is not None else settings.llm_score_timeout_sec)
        use_timeout: tuple[int, int | None] = (settings.llm_score_connect_timeout_sec, read_timeout)
        headers = {
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        diag: dict[str, Any] = {
            "timeout_sec": use_timeout,
            "base_url": settings.llm_base_url,
            "model": settings.llm_model,
            "stream": settings.llm_score_use_stream,
            "prefer_chat": settings.llm_score_prefer_chat,
            "attempts": [],
        }

        text = ""
        request_path = ""
        first = "chat" if settings.llm_score_prefer_chat else "responses"
        second = "responses" if first == "chat" else "chat"
        last_status: dict[str, int] = {}

        for mode in (first, second):
            if mode == "chat":
                t0 = time.perf_counter()
                chat_resp = requests.post(
                    f"{settings.llm_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=chat_body,
                    stream=settings.llm_score_use_stream,
                    timeout=use_timeout,
                )
                last_status["chat"] = int(chat_resp.status_code)
                diag["attempts"].append(
                    {
                        "path": "/chat/completions",
                        "ok": bool(chat_resp.ok),
                        "status_code": int(chat_resp.status_code),
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "response_preview": (chat_resp.text or "")[:280],
                    }
                )
                if not chat_resp.ok:
                    continue
                if settings.llm_score_use_stream:
                    text = _extract_chat_stream_content(chat_resp)
                else:
                    data = chat_resp.json()
                    text = _extract_chat_content(data)
                request_path = "/chat/completions"
                break

            t1 = time.perf_counter()
            resp = requests.post(
                f"{settings.llm_base_url.rstrip('/')}/responses",
                headers=headers,
                json=responses_body,
                timeout=use_timeout,
            )
            last_status["responses"] = int(resp.status_code)
            diag["attempts"].append(
                {
                    "path": "/responses",
                    "ok": bool(resp.ok),
                    "status_code": int(resp.status_code),
                    "elapsed_ms": int((time.perf_counter() - t1) * 1000),
                    "response_preview": (resp.text or "")[:280],
                }
            )
            if not resp.ok:
                continue
            data = resp.json()
            text = _extract_output_text(data)
            request_path = "/responses"
            break

        if not text:
            return {
                "ok": False,
                "error": (
                    "LLM请求失败: "
                    f"/chat/completions={last_status.get('chat', -1)}, "
                    f"/responses={last_status.get('responses', -1)}"
                ),
                "diag": diag,
            }

        parsed = _safe_json_parse(text)
        if not parsed:
            return {
                "ok": False,
                "error": "LLM返回内容无法解析为JSON。",
                "request_path": request_path,
                "raw_preview": text[:400],
                "diag": diag,
            }

        raw_scores = parsed.get("step_scores", [])
        if not isinstance(raw_scores, list):
            return {
                "ok": False,
                "error": "LLM返回缺少step_scores数组。",
                "request_path": request_path,
                "diag": diag,
            }

        step_scores = _normalize_step_scores(raw_scores, steps)
        if len(step_scores) != len(steps):
            return {
                "ok": False,
                "error": f"LLM评分步数不匹配: returned={len(step_scores)}, expected={len(steps)}",
                "request_path": request_path,
                "diag": diag,
            }

        total = round(sum(s.score for s in step_scores) / len(step_scores), 2) if step_scores else 0.0
        feedback = str(parsed.get("feedback", "")).strip() or "已按步骤正确性评分。"
        return {
            "ok": True,
            "scores": step_scores,
            "total": max(0.0, min(100.0, total)),
            "feedback": feedback,
            "meta": {
                "llm_used": True,
                "request_path": request_path,
                "llm_error": "",
                "llm_diag": diag,
            },
        }
    except Exception as e:
        return {"ok": False, "error": f"LLM调用异常: {type(e).__name__}: {e}"}


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
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return ""


def _extract_chat_stream_content(resp: requests.Response) -> str:
    parts: list[str] = []
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        choices = obj.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
        token = delta.get("content", "")
        if isinstance(token, str) and token:
            parts.append(token)
    return "".join(parts).strip()


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


def _normalize_step_scores(raw_scores: list[dict[str, Any]], steps: list[StepItem]) -> list[StepScore]:
    score_map: dict[int, tuple[float, str]] = {}
    for item in raw_scores:
        try:
            idx = int(item["index"])
            score = max(0.0, min(100.0, float(item["score"])))
            reason = str(item.get("reason", "")).strip() or "已评分。"
            score_map[idx] = (score, reason)
        except Exception:
            continue

    out: list[StepScore] = []
    for s in steps:
        score, reason = score_map.get(s.index, (70.0, "模型未返回该步评分，按中性分处理。"))
        out.append(StepScore(index=s.index, score=round(score, 2), reason=reason))

    # 策略1：若整体几乎全对，直接给满分，避免“全对却非100”的体验。
    if out and min(x.score for x in out) >= 95.0:
        for x in out:
            x.score = 100.0
            if "正确" not in x.reason:
                x.reason = "步骤正确，给满分。"
        return out

    # 策略2：连锁错误不重复重罚（首次关键错误后，后续步骤最低给到60）。
    first_major_err = next((i for i, x in enumerate(out) if x.score < 60.0), None)
    if first_major_err is not None:
        for i in range(first_major_err + 1, len(out)):
            if out[i].score < 60.0:
                out[i].score = 60.0
                out[i].reason = "受前序错误影响，当前步不重复重罚。"

    return out


def score_steps(
    ocr_text: str, steps: list[StepItem], reference_steps: list[StepItem], reference_raw: str | None, use_llm: bool
) -> tuple[list[StepScore], float, str, str, dict[str, Any]]:
    if use_llm:
        llm_res = _llm_score_steps(ocr_text=ocr_text, steps=steps, reference=reference_raw)
        if llm_res.get("ok"):
            scores = llm_res["scores"]
            total = llm_res["total"]
            feedback = llm_res["feedback"]
            meta = llm_res["meta"]
            meta.update({"scoring_mode": "llm"})
            return scores, total, feedback, "llm+rule", meta
        llm_error = str(llm_res.get("error", "LLM评分未成功，已回退到规则评分。"))
        llm_diag = llm_res.get("diag", {})
    else:
        llm_error = ""
        llm_diag = {}

    scores, total, feedback = _rule_score_steps(steps=steps, reference_steps=reference_steps)
    if use_llm and not llm_error and settings.llm_api_key:
        llm_error = "LLM评分未成功，已回退到规则评分。"
    elif use_llm and not settings.llm_api_key:
        llm_error = "未配置LLM_API_KEY，已使用规则评分。"
    return scores, total, feedback, "rule-based", {
        "llm_used": False,
        "request_path": "",
        "llm_error": llm_error,
        "llm_diag": llm_diag,
        "scoring_mode": "rule",
    }
