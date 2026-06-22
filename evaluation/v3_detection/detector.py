"""v3 binary error detection — adds variable-consistency section to v2 prompt.

Delta vs v2: targets the variable-class recall gap (v2: 0.63). The new prompt
explicitly tells the model that switching a variable name mid-solution without
a declaration ("设 y = ...") is an error, with worked examples. Tradeoff to
watch: false positives on legitimate new-variable introductions.

Output contract and retry policy unchanged from v2.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from app.config import settings


PROMPT_SYSTEM = "你是数学解题错误检测器。严格按数学对错判断，只返回 JSON，不要解释。"

PROMPT_TEMPLATE = """你的任务：逐步检查学生的解题过程，判断哪一步有数学错误。

【判错原则】
- 只判断数学对错（计算、符号、变量、逻辑推导）。
- 风格、简洁度、跳步、缺中间过渡 — 一律不算错。
- 一道题可以有多步错，也可以完全没有错。
- 若整道题完全正确，has_error=false。

【变量一致性（重点）】
- 解题过程中变量名必须前后一致。
- 若某步突然把前文的变量 x 写成 y（或换成其他字母/符号），且该步没有"设 y=…""令 y=…""y := …"的显式声明，视为 variable 错。
- 合法引入新变量（"设 y = 2x"、"令 t = x+1"、"换元 u = x²"）不算错。
- 例：
  错：第 1 步"3x = 9"、第 3 步突然写"3y = 9"且无声明。
  对：第 3 步"设 y = x，则 3y = 9"。
- 不同字母但数学等价（如大小写 X 与 x、希腊字母 α 与 a）也算错。

【错误类型分类】（仅用于错步）
- calculation：数值/算式计算错误（如 2+3=6）
- sign：正负号、加减号、不等号方向错误
- variable：变量替换、变量名混淆（含未声明换名）、代入错误
- logic：逻辑推导、公式记错、变形方向反、量纲错误
- other：上述之外的数学错误

【输入】
题目：{problem}

学生解题步骤：
{steps_block}

【输出格式】严格 JSON，键名固定：
{{
  "has_error": true/false,
  "error_steps": [3, 5],            // 1-indexed，空数组表示无错
  "error_types": {{"3": "calculation", "5": "sign"}}
}}

若 has_error=false，error_steps 必为 []，error_types 必为 {{}}。"""


def _build_user_prompt(problem: dict) -> str:
    steps = problem.get("steps") or []
    if not steps:
        raise ValueError("problem has no steps")
    block = "\n".join(f"步骤 {i + 1}：{s['text']}" for i, s in enumerate(steps))
    return PROMPT_TEMPLATE.format(
        problem=problem.get("problem", "").strip(),
        steps_block=block,
    )


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_chat_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    msg = choices[0].get("message") or {}
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
        choices = obj.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {} if isinstance(choices[0], dict) else {}
        token = delta.get("content", "")
        if isinstance(token, str) and token:
            parts.append(token)
    return "".join(parts).strip()


def _normalize_verdict(parsed: dict[str, Any], n_steps: int) -> dict[str, Any]:
    has_error_raw = parsed.get("has_error")
    if isinstance(has_error_raw, str):
        has_error_raw = has_error_raw.strip().lower() in {"true", "1", "yes"}
    has_error = bool(has_error_raw)

    raw_steps = parsed.get("error_steps") or []
    if not isinstance(raw_steps, list):
        raw_steps = []
    steps: list[int] = []
    for s in raw_steps:
        try:
            idx = int(s)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= n_steps and idx not in steps:
            steps.append(idx)
    steps.sort()

    raw_types = parsed.get("error_types") or {}
    if not isinstance(raw_types, dict):
        raw_types = {}
    types: dict[str, str] = {}
    valid_categories = {"calculation", "sign", "variable", "logic", "other"}
    for k, v in raw_types.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if idx not in steps:
            continue
        cat = str(v).strip().lower()
        if cat not in valid_categories:
            cat = "other"
        types[str(idx)] = cat

    if not steps:
        has_error = False
    else:
        has_error = True
        for idx in steps:
            types.setdefault(str(idx), "other")

    return {
        "has_error": has_error,
        "error_steps": steps,
        "error_types": types,
    }


def _extract_output_text(payload: dict[str, Any]) -> str:
    out = payload.get("output") or []
    chunks: list[str] = []
    for item in out:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                chunks.append(str(content.get("text", "")))
    return "\n".join(chunks).strip()


def _call_llm_once(user_prompt: str) -> tuple[bool, str, dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {settings.llm_score_api_key}",
        "Content-Type": "application/json",
    }
    chat_body = {
        "model": settings.llm_score_model,
        "messages": [
            {"role": "system", "content": PROMPT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 1500,
        "response_format": {"type": "json_object"},
        "stream": settings.llm_score_use_stream,
    }
    responses_body = {
        "model": settings.llm_score_model,
        "input": json.dumps(
            {"system": PROMPT_SYSTEM, "user": user_prompt}, ensure_ascii=False
        ),
    }
    read_timeout = None if settings.llm_score_no_read_timeout else settings.llm_score_timeout_sec
    use_timeout: tuple[int, int | None] = (
        settings.llm_score_connect_timeout_sec,
        read_timeout,
    )

    diag: dict[str, Any] = {"attempts": []}
    text = ""
    last_status: dict[str, int] = {}

    first = "chat" if settings.llm_score_prefer_chat else "responses"
    second = "responses" if first == "chat" else "chat"

    for mode in (first, second):
        if mode == "chat":
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{settings.llm_score_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=chat_body,
                    stream=settings.llm_score_use_stream,
                    timeout=use_timeout,
                )
            except requests.RequestException as e:
                diag["attempts"].append({"path": "/chat/completions", "ok": False, "error": f"{type(e).__name__}: {e}"})
                continue
            last_status["chat"] = int(resp.status_code)
            diag["attempts"].append(
                {
                    "path": "/chat/completions",
                    "ok": bool(resp.ok),
                    "status_code": int(resp.status_code),
                    "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                }
            )
            if not resp.ok:
                continue
            if settings.llm_score_use_stream:
                text = _extract_chat_stream_content(resp)
            else:
                try:
                    text = _extract_chat_content(resp.json())
                except Exception:
                    text = ""
            break

        t1 = time.perf_counter()
        try:
            resp = requests.post(
                f"{settings.llm_score_base_url.rstrip('/')}/responses",
                headers=headers,
                json=responses_body,
                timeout=use_timeout,
            )
        except requests.RequestException as e:
            diag["attempts"].append({"path": "/responses", "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue
        last_status["responses"] = int(resp.status_code)
        diag["attempts"].append(
            {
                "path": "/responses",
                "ok": bool(resp.ok),
                "status_code": int(resp.status_code),
                "elapsed_ms": int((time.perf_counter() - t1) * 1000),
            }
        )
        if not resp.ok:
            continue
        try:
            text = _extract_output_text(resp.json())
        except Exception:
            text = ""
        break

    diag["last_status"] = last_status
    if not text:
        return False, (
            f"empty body (chat={last_status.get('chat', -1)}, "
            f"responses={last_status.get('responses', -1)})"
        ), diag
    return True, text, diag


MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0


def detect_errors(problem: dict) -> dict[str, Any]:
    steps = problem.get("steps") or []
    if not steps:
        return {"ok": False, "error": "no steps", "verdict": None}

    prompt = _build_user_prompt(problem)
    last_err = ""
    diag_hist: list[dict[str, Any]] = []
    for attempt in range(1, MAX_RETRIES + 1):
        ok, text_or_err, diag = _call_llm_once(prompt)
        diag_hist.append({"attempt": attempt, "ok": ok, **diag})
        if not ok:
            last_err = text_or_err
            time.sleep(RETRY_BACKOFF_SEC * attempt)
            continue

        parsed = _safe_json_parse(text_or_err)
        if not parsed:
            last_err = "unparseable JSON"
            time.sleep(RETRY_BACKOFF_SEC * attempt)
            continue

        verdict = _normalize_verdict(parsed, n_steps=len(steps))
        return {
            "ok": True,
            "verdict": verdict,
            "attempts": diag_hist,
            "raw_preview": text_or_err[:400],
        }

    return {
        "ok": False,
        "error": last_err or "unknown",
        "verdict": None,
        "attempts": diag_hist,
    }


# Save the prompt template for reference / diff vs v2
if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).parent / "prompts" / "v3_prompt.txt"
    out.write_text(f"SYSTEM:\n{PROMPT_SYSTEM}\n\nUSER TEMPLATE:\n{PROMPT_TEMPLATE}\n", encoding="utf-8")
    print(f"Wrote {out}")
