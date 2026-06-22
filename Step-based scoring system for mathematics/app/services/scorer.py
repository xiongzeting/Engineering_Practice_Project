"""步骤评分核心：规则法 + LLM 两轮反思。

打分以「步骤」为粒度，每道题内部独立打分，最后汇总成整卷 0-100 分。
有两条路径：

- **LLM 路径**：:func:`_llm_score_steps` 用 ``deepseek-v4-pro`` 给每步
  打分 + 写出扣分理由；再可选走第二轮「反思」——把首轮结果和原题重新
  喂给 LLM 让它自检。失败、超时、JSON 解析错都会自动降级到规则法。
- **规则路径**：:func:`_rule_score_steps` 用正则 + 公式相似度做兜底，
  保证任何情况下都有分数输出。

对外暴露两个入口：
- :func:`score_questions` —— 老路径，输入整段 OCR 文本，内部切题切步骤；
- :func:`score_one_problem` —— 新路径，消费 review 已经分好的单题
  （``question_text + step_lines``），不再 re-split。前端有 ``problems``
  时走这条，避免题干被当步骤打分。

辅助：
- :func:`_step_appears_in_corpus` —— 用 ``SequenceMatcher`` 检测 LLM 是否
  「脑补」了 OCR 里没出现的步骤，命中就丢掉；
- :func:`_invoke_score_llm_with_retry` —— 指数级延长超时的重试包装。
"""
from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from typing import Any

import requests

from app.config import settings
from app.schemas import QuestionGrade, StepItem, StepScore
from app.services.question_splitter import split_text_into_questions
from app.services.step_parser import split_steps, MATH_TOKEN_RE, _is_annotation_line


def _basic_validity_score(text: str) -> float:
    """规则评分的基础有效性分（0-100）。

    从 100 分起扣：
    - 括号不匹配扣 20（左右括号数量不一致，常见 OCR 漏识别）；
    - 既无数字也无等号扣 15（不像数学表达式，可能是一段文字）；
    - 出现连续运算符（如 ``+-``、``*/``、``^=``）扣 10（OCR 把多个符号粘连）。

    最后用 :func:`max` 兜底为 0，保证不会出现负分。
    """
    score = 100.0
    if text.count("(") != text.count(")"):  # 括号配对失败
        score -= 20
    if not any(ch.isdigit() for ch in text) and "=" not in text:  # 既无数字也无等号
        score -= 15
    if re.search(r"[+\-*/^=]{2,}", text):  # 连续运算符，OCR 粘连的典型现象
        score -= 10
    return max(score, 0.0)


def _build_cognitive_weight_demo(max_score: float, n_steps: int) -> list[float]:
    """生成一份「非均匀」的分步权重示例，总和等于 ``max_score``。

    模仿一线老师的给分习惯：**关键步骤**给最大头，**最终答案**给一个像样的
    份额，剩余中间步骤均分剩下的。所有值都对齐到 0.5 的倍数（老师阅卷的
    常见扣分粒度）。

    分支：
    - ``n_steps <= 0``：返回空列表（没有步骤，权重无意义）；
    - ``n_steps == 1``：唯一一步直接给满分；
    - ``n_steps == 2``：关键步 60% + 最终答案 40%；
    - ``n_steps >= 3``：关键步 35% + 最终答案 20% + 中间均分 45%。

    最后做一次 ``drift``（取整漂移）补偿：若因四舍五入导致和不等于
    ``max_score``，把差额全部塞到关键步上（最大块改动相对不敏感）。
    """
    if n_steps <= 0:
        return []
    if n_steps == 1:
        return [round(max_score * 2) / 2]
    if n_steps == 2:
        # 只有「关键步 + 最终答案」两步：6/4 开。
        key = round(max_score * 0.6 * 2) / 2
        final = round((max_score - key) * 2) / 2
        return [max(0.5, key), max(0.5, final)]

    # 3 步以上：关键步 35% / 最终答案 20% / 中间步骤均分 45%。
    key_share = 0.35
    final_share = 0.20
    mid_total_share = max(0.0, 1.0 - key_share - final_share)
    mid_count = n_steps - 2
    mid_each_share = mid_total_share / mid_count

    raw: list[float] = [max_score * key_share]
    for _ in range(mid_count):
        raw.append(max_score * mid_each_share)
    raw.append(max_score * final_share)

    # 全部对齐到 0.5 的倍数（老师手批的真实粒度）。
    snapped = [round(x * 2) / 2 for x in raw]
    # 修复因对齐带来的漂移，保证总和精确等于 max_score。
    drift = round(max_score - sum(snapped), 2)
    if abs(drift) >= 0.5:
        # 把漂移整个塞到关键步（值最大，对相对比例最不敏感）。
        snapped[0] = round((snapped[0] + drift) * 2) / 2
    return [max(0.5, x) for x in snapped]


def _rule_score_steps(steps: list[StepItem], reference_steps: list[StepItem], max_score: float) -> tuple[list[StepScore], float, str]:
    """规则评分兜底路径。

    LLM 不可用、超时、返回非法 JSON 时走这里。打分逻辑非常朴素：

    - **基础有效性**：:func:`_basic_validity_score` 给每步一个 0-100 的
      「这看起来像不像数学表达式」分。
    - **参考解相似度**：若提供了 ``reference_steps``，用
      :class:`SequenceMatcher` 算每步与参考解的最优相似度（0-1），放大到
      100；没有参考解就固定给 75 分（中性基线）。
    - 两者加权（**有效性 0.55 + 相似度 0.45**）得到该步的 0-100 分值，
      再乘以 ``per_step_max``（每步满分）得到实际得分。

    最后做一次「总和上限」：若所有步骤得分之和超过 ``max_score``，按比例
    缩放回 ``max_score``（snap-safe，因为缩放后再对齐到 0.5）。
    """
    step_scores: list[StepScore] = []
    ref_texts = [s.normalized for s in reference_steps]
    # 每步满分 = 题目总分 / 步数（默认均分，规则路径不去模拟教师权重）。
    per_step_max = (max_score / len(steps)) if steps else max_score

    for step in steps:
        validity = _basic_validity_score(step.normalized)
        similarity = 75.0  # 中性基线，无参考解时使用
        reason = "步骤结构基本合理。"
        if ref_texts:
            # 取该步与所有参考步骤中相似度最高的一个，作为该步的相似分。
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

        # 0-100 制加权：有效性 55% + 相似度 45%。
        step_100 = 0.55 * validity + 0.45 * similarity
        step_score = _snap_to_half_point(step_100 / 100.0 * per_step_max)
        step_scores.append(StepScore(index=step.index, score=step_score, reason=reason))

    # 硬性上限：步骤总和不得超过题目总分，超过就按比例缩放（snap 安全）。
    total = round(sum(s.score for s in step_scores), 2) if step_scores else 0.0
    if max_score > 0 and total > max_score + 0.01:
        scale = max_score / total
        for s in step_scores:
            s.score = _snap_to_half_point(s.score * scale)
        total = round(sum(s.score for s in step_scores), 2)
    feedback = "已按规则评分（LLM 不可用时的兜底）。"
    return step_scores, total, feedback


def _build_initial_score_prompt(
    ocr_text: str,
    steps: list[StepItem],
    reference: str | None,
    max_score: float,
    question_text: str | None,
) -> dict[str, Any]:
    """构造「首轮评分」的 LLM 输入 prompt。

    核心 11 条评分规则（写在 ``scoring_rule`` 里，对模型来说最关键）：
    1. 先看题干再判步骤；``reference_solution`` 仅作辅助；
    2. 只抄题未作答 = 0 分；
    3. 步骤权重之和必须等于 ``max_score``，且不得默认均分（按重要性）；
    4. 逐步评分细则：完全正确=weight，计算小错≈80%，方法错≤50%，无意义=0；
    5. 不扣书写风格分（跳步、简写不扣）；
    6. 连锁错误不重扣；
    7. 变量突变按计算小错处理；
    8. **解法完整性**：有公式但没代入 / 没求值 = 扣 overall_penalty；
    9. 所有数值必须对齐到 0.5 的倍数；
    10. ``total_score = sum(step_scores) - overall_penalty``；
    11. ``reason`` ≤18 汉字，``feedback`` ≤60 汉字。

    最后给出严格的 JSON 输出模板，让模型只能按格式回。
    """
    step_payload = [{"index": s.index, "text": s.normalized} for s in steps]
    # OCR 文本截断到 2000 字符，避免 prompt 超长触发 LLM 截断。
    ocr_excerpt = ocr_text[:2000]
    clean_question = (question_text or "").strip()
    return {
        "task": (
            f"你是数学步骤评分器。本题总分为 {max_score} 分，必须严格按 {max_score} 分制评分，"
            f"禁止按 100 分制。只返回 JSON，不要解释，不要输出 Markdown。"
        ),
        "question": clean_question,
        "question_max_score": max_score,
        "ocr_text_excerpt": ocr_excerpt,
        "steps": step_payload,
        "reference_solution": reference or "",
        "scoring_rule": (
            "评分规则：\n\n"
            "1. 先根据 question 理解题目要求，再判断 steps 的数学正确性。"
            "reference_solution 仅作辅助；steps 与 OCR 冲突时，优先按 steps 评分。\n\n"
            "2. 若所有 steps 都只是重复题目或抄写条件，没有任何解题动作，则本题 0 分，"
            "feedback 写\"该题未作答，仅识别到题目陈述。\"\n\n"
            f"3. 必须给每步分配 weight，所有 weight 之和必须等于 {max_score}。"
            "不要默认均分；按步骤重要性分配：\n"
            "   - 关键公式、关键变形、构造方程、换元、最终答案：高权重；\n"
            "   - 代入、化简、移项、因式分解、中间推导：中权重；\n"
            "   - 简单算术、抄写条件、辅助说明：低权重。\n"
            f"   只有 1 步时，该步 weight = {max_score}。\n\n"
            "4. 逐步评分：\n"
            "   - 完全正确：score = weight，reason 写\"完全正确\"。\n"
            "   - 计算小错（数字、正负号、简单代入、变量误写）：给约 80% weight。\n"
            "   - 方法正确但局部变形、化简或条件处理有误：给 50%-80% weight。\n"
            "   - 公式错误、方法错误、定理误用、推导方向错误：给 0%-50% weight。\n"
            "   - 无关或无数学意义的步骤：给 0 分。\n\n"
            "5. 只按数学对错评分。跳步、简写、书写风格、缺少解释，不扣分。\n\n"
            "6. 连锁错误不重复重扣。若后续步骤只是沿用前一步错误结果，但推导形式正确，"
            "可以给该步满分。\n"
            "   例外：若前面是关键公式或关键方法错误，导致后续数学对象根本改变，"
            "则后续相关步骤最高给 80% weight。\n\n"
            "7. 变量突然改变且无\"设/令/换元\"说明，按计算小错处理。"
            "明显 OCR 误识别且不影响数学含义，不扣分。\n\n"
            "8. 解法完整性：\n"
            "   - 若题目要求求解、求值、计算、求导、求概率、求面积、求最值等具体结果，"
            "必须有明确最终答案（含数值或简化表达式）。\n"
            "   - 形式化的代入记号（如 $[F(x)]_a^b$、$\\big|_a^b$）本身只是\"将要求值\"的标记，"
            "不等于已算出结果；学生必须进一步写出 $F(b)-F(a)$ 的具体数值或简化式才算完整。\n"
            f"   - 主要推导正确但缺少最终答案：overall_penalty = {max_score} 的 20%。\n"
            f"   - 只列公式或方程，未代入、未求解、未化简：overall_penalty = {max_score} 的 40%。\n"
            f"   - 主要推导尚未完成：overall_penalty = {max_score} 的 50%。\n"
            "   - 反例：题目\"求 $\\int_0^1 \\frac{2}{x+1} dx$\"，学生写到 "
            "$2[\\ln(x+1)]_0^1$ 就停了，没有给出 $2\\ln 2$ → 视为缺少最终答案，"
            f"overall_penalty = {max_score} 的 20%，feedback 必须写\"缺少最终答案\"。\n"
            "   - 反例：题目\"求 $\\int_0^{\\pi/2} \\sin^4 x \\cos^2 x dx$\"，"
            "学生写了一堆 Wallis 公式乘积但没化简出最终数值 → 同上扣 20%。\n"
            "   - 完整性扣分只用于\"已写步骤基本正确但解法没完成\"的情况，不要对同一错误重复扣分。\n"
            "   - 若扣 overall_penalty，feedback 必须说明\"缺少最终答案\"\"未代入求解\""
            "或\"解法未完成\"。\n\n"
            "9. 所有 weight、score、overall_penalty、total_score 必须是 0.5 的倍数"
            "（例如 0、0.5、1、1.5、2）。若按比例计算后不是 0.5 的倍数，四舍五入到最近的 0.5 分。"
            "优先使用整数分；只有在整数分不能合理体现扣分时，才使用 x.5。\n\n"
            "10. total_score = sum(step_scores.score) - overall_penalty。"
            f"若小于 0，则 total_score = 0；若超过 {max_score}，则 total_score = {max_score}。\n\n"
            "11. reason 不超过 18 个汉字；feedback 不超过 60 个汉字。\n\n"
            "只输出如下 JSON：\n"
            "{\n"
            "  \"step_scores\": [\n"
            "    {\"index\": 1, \"weight\": 0.0, \"score\": 0.0, \"reason\": \"完全正确\"}\n"
            "  ],\n"
            "  \"overall_penalty\": 0.0,\n"
            "  \"total_score\": 0.0,\n"
            "  \"feedback\": \"整体评价。\"\n"
            "}"
        ),
    }


def _build_reflection_prompt(
    initial_parsed: dict[str, Any],
    ocr_text: str,
    steps: list[StepItem],
    reference: str | None,
    max_score: float,
    question_text: str | None,
) -> dict[str, Any]:
    """构造「自评反思」prompt：把模型首轮自己的评分丢回给它，让它
    扮演严格复审员重新核对。

    设计保守：**只在发现明确错误时才修订**，模棱两可时维持原评，避免
    「反思反而改坏了」。

    要求模型输出与首轮相同字段 + 一个额外的 ``reflection_reasoning`` 字段
    （≤80 汉字），说明这次复审修订了什么、为什么，供评测脚本统计
    ``reflection_changed`` 比例。
    """
    step_payload = [{"index": s.index, "text": s.normalized} for s in steps]
    ocr_excerpt = ocr_text[:2000]
    clean_question = (question_text or "").strip()
    # 只回传必要字段，不让模型把自己上一轮的 reflection_reasoning 也回吐。
    initial_snapshot = {
        "step_scores": initial_parsed.get("step_scores") or [],
        "overall_penalty": initial_parsed.get("overall_penalty", 0.0),
        "total_score": initial_parsed.get("total_score", 0.0),
        "feedback": initial_parsed.get("feedback", ""),
    }
    return {
        "task": (
            f"你刚刚为下面这道题（总分 {max_score} 分）给出了一份初步评分。"
            "现在请作为严格的复审员，重新核对每一步：1) 步骤的数学对错判定是否合理；"
            "2) 每步 score 是否符合评分规则（完全正确=weight；计算小错≈80%weight；方法错≤50%weight）；"
            "3) overall_penalty 是否漏扣或重扣（完整性反例：$2[\\ln(x+1)]_0^1$ 没算到 $2\\ln 2$ 应扣 20%）；"
            "4) 是否存在连锁错误被重复扣分。"
            "复审原则：仅在发现明确错误时才修订；模棱两可时维持原评。"
            f"必须严格按 {max_score} 分制评分，禁止按 100 分制。只返回 JSON，不要解释，不要输出 Markdown。"
        ),
        "question": clean_question,
        "question_max_score": max_score,
        "ocr_text_excerpt": ocr_excerpt,
        "steps": step_payload,
        "reference_solution": reference or "",
        "initial_scoring": initial_snapshot,
        "output_rules": (
            "输出 JSON 必须包含与初次评分相同的字段（step_scores / overall_penalty / total_score / feedback），"
            "并额外增加一个字段 reflection_reasoning（不超过 80 个汉字），说明本次复审是否修订了什么、为什么。"
            "若维持原评，reflection_reasoning 写\"复审后维持原评\"，"
            "step_scores/overall_penalty/total_score 与初次一致。"
            "所有数值仍必须是 0.5 的倍数。"
            "reason 不超过 18 个汉字；feedback 不超过 60 个汉字。"
        ),
    }


def _invoke_score_llm_with_retry(
    prompt_dict: dict[str, Any],
    steps: list[StepItem],
    max_score: float,
    timeout: int | None = None,
) -> tuple[dict[str, Any] | None, list[StepScore] | None, dict[str, Any]]:
    """调用评分 LLM，带多轮重试与两种 HTTP 路径回退。

    策略：
    - **双路径回退**：优先 ``/chat/completions``（OpenAI 兼容最广），
      失败再试 ``/responses``（Anthropic/Claude 风格）。
    - **多轮重试**：最多 4 轮（``MAX_ATTEMPTS``），每轮之间睡
      ``RETRY_BACKOFF_SEC * (attempt+1)`` 秒（线性退避）。
    - **递增超时**：读超时每轮放大 ``1× → 1.5× → 2× → 3×``，避免长
      step 列表或反思轮因为慢响应被反复超时砍掉。

    成功时返回 ``(parsed_json, normalized_step_scores, diag)``；失败时
    ``(None, None, diag)``，``diag`` 携带所有失败原因、HTTP 状态码、耗时、
    响应片段等，便于事后排查「为什么降级到规则评分」。
    """
    # /responses 路径的请求体：把整个 prompt 作为 input 字符串塞进去。
    responses_body = {
        "model": settings.llm_score_model,
        "input": json.dumps(prompt_dict, ensure_ascii=False),
    }
    # /chat/completions 路径的请求体：system 锁死「严格 max_score 分制」。
    chat_body = {
        "model": settings.llm_score_model,
        "messages": [
            {"role": "system", "content": (
                f"你是数学步骤评分器。本题总分为 {max_score} 分，必须严格按 {max_score} 分制评分，"
                f"禁止按 100 分制。只返回 JSON，不要解释，不要输出 Markdown。"
            )},
            {"role": "user", "content": json.dumps(prompt_dict, ensure_ascii=False)},
        ],
        "temperature": 0.0,  # 评分要稳定可复现，温度拉到 0
        "max_tokens": 6000,
        "response_format": {"type": "json_object"},  # 强制 JSON 输出
        "stream": settings.llm_score_use_stream,
    }
    # 读超时基础值：llm_score_no_read_timeout=True 表示完全不设读超时（等到底）。
    base_read_timeout = None if settings.llm_score_no_read_timeout else (timeout if timeout is not None else settings.llm_score_timeout_sec)
    headers = {
        "Authorization": f"Bearer {settings.llm_score_api_key}",
        "Content-Type": "application/json",
    }
    # diag 记录这次调用的所有诊断信息，落库到 llm_failures 表以便事后排查。
    diag: dict[str, Any] = {
        "timeout_sec": (settings.llm_score_connect_timeout_sec, base_read_timeout),
        "base_url": settings.llm_score_base_url,
        "model": settings.llm_score_model,
        "stream": settings.llm_score_use_stream,
        "prefer_chat": settings.llm_score_prefer_chat,
        "attempts": [],
        "retries": 0,
        "last_error": "",
        "request_path": "",
        "timeout_seen": False,
    }

    # 双路径：先试 prefer_chat 配置的那条，另一条做兜底。
    first = "chat" if settings.llm_score_prefer_chat else "responses"
    second = "responses" if first == "chat" else "chat"

    MAX_ATTEMPTS = 4
    RETRY_BACKOFF_SEC = 1.5
    # 渐进式读超时：1× → 1.5× → 2× → 3× 基础值，给慢响应更多机会。
    TIMEOUT_MULTIPLIERS = (1.0, 1.5, 2.0, 3.0)
    parsed: dict[str, Any] | None = None
    step_scores: list[StepScore] | None = None

    for attempt in range(MAX_ATTEMPTS):
        diag["retries"] = attempt
        # 本轮的读超时倍率（前几轮更紧，后几轮放宽）。
        mult = TIMEOUT_MULTIPLIERS[min(attempt, len(TIMEOUT_MULTIPLIERS) - 1)]
        read_timeout = None if base_read_timeout is None else int(round(base_read_timeout * mult))
        use_timeout: tuple[int, int | None] = (settings.llm_score_connect_timeout_sec, read_timeout)
        diag["timeout_sec"] = use_timeout
        text = ""
        last_status: dict[str, int] = {}

        for mode in (first, second):
            # 每轮先试优先路径，失败再试另一条路径。
            if mode == "chat":
                t0 = time.perf_counter()
                try:
                    chat_resp = requests.post(
                        f"{settings.llm_score_base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json=chat_body,
                        stream=settings.llm_score_use_stream,
                        timeout=use_timeout,
                    )
                except requests.exceptions.Timeout as exc:
                    diag["timeout_seen"] = True
                    diag["attempts"].append({
                        "attempt": attempt + 1, "path": "/chat/completions",
                        "ok": False, "status_code": -1, "timeout": True,
                        "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                        "response_preview": f"timeout: {type(exc).__name__}",
                    })
                    continue
                last_status["chat"] = int(chat_resp.status_code)
                diag["attempts"].append(
                    {
                        "attempt": attempt + 1,
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
                diag["request_path"] = "/chat/completions"
                break

            t1 = time.perf_counter()
            try:
                resp = requests.post(
                    f"{settings.llm_score_base_url.rstrip('/')}/responses",
                    headers=headers,
                    json=responses_body,
                    timeout=use_timeout,
                )
            except requests.exceptions.Timeout as exc:
                diag["timeout_seen"] = True
                diag["attempts"].append({
                    "attempt": attempt + 1, "path": "/responses",
                    "ok": False, "status_code": -1, "timeout": True,
                    "elapsed_ms": int((time.perf_counter() - t1) * 1000),
                    "response_preview": f"timeout: {type(exc).__name__}",
                })
                continue
            last_status["responses"] = int(resp.status_code)
            diag["attempts"].append(
                {
                    "attempt": attempt + 1,
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
            diag["request_path"] = "/responses"
            break

        if not text:
            # 两条路径都失败：记一下 HTTP 状态码，决定是否继续重试。
            diag["last_error"] = (
                "LLM请求失败: "
                f"/chat/completions={last_status.get('chat', -1)}, "
                f"/responses={last_status.get('responses', -1)}"
            )
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

        # 拿到文本后尝试解析为 JSON；解析失败可能是模型输出了 markdown 代码块。
        candidate = _safe_json_parse(text)
        if not candidate:
            diag["last_error"] = "LLM返回内容无法解析为JSON。"
            diag["raw_preview"] = text[:400]
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

        raw_scores = candidate.get("step_scores", [])
        if not isinstance(raw_scores, list):
            diag["last_error"] = "LLM返回缺少step_scores数组。"
            diag["raw_preview"] = text[:400]
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

        # 规范化（对齐 0.5 / 上限缩放）后，必须与 steps 数量一致才算成功。
        candidate_scores = _normalize_step_scores(raw_scores, steps, max_score)
        if len(candidate_scores) != len(steps):
            diag["last_error"] = f"LLM评分步数不匹配: returned={len(candidate_scores)}, expected={len(steps)}"
            diag["raw_preview"] = text[:400]
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
            continue

        # 全部校验通过，接受本轮结果。
        parsed = candidate
        step_scores = candidate_scores
        break

    return parsed, step_scores, diag


def _finalize_score_result(
    parsed: dict[str, Any],
    step_scores: list[StepScore],
    max_score: float,
) -> tuple[float, float, str]:
    """把 LLM 返回的 JSON + 规范化后的 step_scores 收敛成最终的
    ``(total, overall_penalty, feedback)``。

    - ``overall_penalty`` 来自模型对「完整性」的判断（见评分规则 #8），
      负值或异常值都归零；
    - ``total = sum(step_scores.score) - overall_penalty``，再做 0.5 对齐
      和 ``[0, max_score]`` 钳制；
    - ``feedback`` 缺失时给一个中性默认值，避免前端显示空白。
    """
    try:
        overall_penalty_raw = float(parsed.get("overall_penalty", 0.0) or 0.0)
    except (TypeError, ValueError):
        overall_penalty_raw = 0.0
    overall_penalty = max(0.0, _snap_to_half_point(overall_penalty_raw))
    raw_total = round(sum(s.score for s in step_scores), 2) - overall_penalty
    # 钳制到 [0, max_score]，并对齐 0.5。
    total = max(0.0, min(max_score, _snap_to_half_point(raw_total)))
    feedback = str(parsed.get("feedback", "")).strip() or "已按步骤正确性评分。"
    return total, overall_penalty, feedback


def _llm_score_steps(
    ocr_text: str, steps: list[StepItem], reference: str | None, max_score: float, timeout: int | None = None,
    question_text: str | None = None,
) -> dict[str, Any]:
    """LLM 评分主流程（两轮：首轮评分 + 可选自评反思）。

    返回字典结构：
    - 成功：``{"ok": True, "scores": [...], "total": float, "feedback": str,
      "meta": {...}}``，``meta`` 里含 ``llm_used``、``reflection_used``、
      ``reflection_changed`` 等审计字段。
    - 失败：``{"ok": False, "error": str, ...}``，调用方应据此决定是否
      降级到规则路径。

    反思轮失败不致命：会静默回退到首轮结果，并在 ``reflection_reasoning``
    里记 ``reflection_unavailable`` 原因，保证流程不中断。
    """
    if not settings.llm_score_api_key:
        return {"ok": False, "error": "未配置LLM_SCORE_API_KEY。"}

    try:
        # 第 1 轮：首轮评分（保留 v8 prompt 的原始行为）。
        initial_prompt = _build_initial_score_prompt(
            ocr_text=ocr_text, steps=steps, reference=reference,
            max_score=max_score, question_text=question_text,
        )
        initial_parsed, initial_step_scores, initial_diag = _invoke_score_llm_with_retry(
            initial_prompt, steps, max_score, timeout=timeout,
        )
        if initial_parsed is None or initial_step_scores is None:
            return {
                "ok": False,
                "error": initial_diag.get("last_error") or "LLM评分未成功。",
                "request_path": initial_diag.get("request_path", ""),
                "raw_preview": initial_diag.get("raw_preview", ""),
                "diag": initial_diag,
            }
        initial_total, initial_penalty, initial_feedback = _finalize_score_result(
            initial_parsed, initial_step_scores, max_score,
        )

        # 第 2 轮：可选自评反思（critique-and-revise），开关由 settings 控制。
        reflection_used = False
        reflection_changed = False
        reflection_reasoning = ""
        reflection_diag: dict[str, Any] = {}
        final_parsed = initial_parsed
        final_step_scores = initial_step_scores

        if settings.llm_score_reflection_enabled:
            refl_prompt = _build_reflection_prompt(
                initial_parsed=initial_parsed,
                ocr_text=ocr_text, steps=steps, reference=reference,
                max_score=max_score, question_text=question_text,
            )
            refl_parsed, refl_step_scores, reflection_diag = _invoke_score_llm_with_retry(
                refl_prompt, steps, max_score, timeout=timeout,
            )
            if refl_parsed is not None and refl_step_scores is not None:
                reflection_used = True
                # reflection_reasoning 限制 500 字以内，防止日志膨胀。
                reflection_reasoning = str(refl_parsed.get("reflection_reasoning", "") or "").strip()[:500]
                final_parsed = refl_parsed
                final_step_scores = refl_step_scores
            else:
                # 反思轮失败：静默回退到首轮结果。
                reflection_reasoning = f"reflection_unavailable: {reflection_diag.get('last_error', '')[:120]}"

        final_total, final_penalty, final_feedback = _finalize_score_result(
            final_parsed, final_step_scores, max_score,
        )

        # 仅当反思轮产生了实际分数变化（>=0.5 分），才算 reflection_changed=True。
        # 小于 0.5 的波动视为对齐误差，不算修订。
        if reflection_used:
            reflection_changed = (
                abs(final_total - initial_total) >= 0.5
                or any(
                    abs(a.score - b.score) >= 0.5
                    for a, b in zip(final_step_scores, initial_step_scores)
                )
                or abs(final_penalty - initial_penalty) >= 0.5
            )

        meta = {
            "llm_used": True,
            "request_path": initial_diag.get("request_path", ""),
            "llm_error": "",
            "llm_diag": initial_diag,
            "overall_penalty": final_penalty,
            # 反思审计字段，评测脚本和报告都会读这几个字段统计反思贡献。
            "reflection_enabled": bool(settings.llm_score_reflection_enabled),
            "reflection_used": reflection_used,
            "reflection_changed": reflection_changed,
            "reflection_reasoning": reflection_reasoning,
            "initial_total": initial_total,
            "initial_overall_penalty": initial_penalty,
            "revised_total": final_total,
            "revised_overall_penalty": final_penalty,
            "llm_diag_reflection": reflection_diag,
        }
        return {
            "ok": True,
            "scores": final_step_scores,
            "total": final_total,
            "feedback": final_feedback,
            "meta": meta,
        }
    except Exception as e:
        # 顶层兜底：任何意外异常都降级为「失败返回」，由调用方走规则路径。
        return {"ok": False, "error": f"LLM调用异常: {type(e).__name__}: {e}"}


def _extract_output_text(payload: dict[str, Any]) -> str:
    """从 ``/responses`` 风格响应里抽取纯文本。

    Anthropic/Claude 风格的响应结构是 ``payload["output"]`` 下若干个
    ``type=="message"`` 的 item，每个 item 的 ``content`` 里再嵌套若干
    ``type=="output_text"`` 的 content。这里把所有 output_text 拼接成一段。
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
    """从 ``/chat/completions`` 非流式响应里抽取助手回复文本。

    OpenAI 风格的响应结构是 ``payload["choices"][0]["message"]["content"]``。
    content 可能是字符串（标准）或 list（多模态，含若干 ``{"type":"text"}``
    项），这里都做兼容。
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


def _extract_chat_stream_content(resp: requests.Response) -> str:
    """从 ``/chat/completions`` 流式响应（SSE）里拼接出完整文本。

    SSE 帧形如 ``data: {"choices":[{"delta":{"content":"xxx"}}]}``，
    逐帧累积 ``delta.content``，遇到 ``[DONE]`` 结束。
    """
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
    """容错 JSON 解析：先直接 ``json.loads``，失败再用正则抠出最外层花括号。

    大模型经常把 JSON 包在 ```json ... ``` 代码块里或者前后带解释文字，
    正则兜底能显著提高解析成功率。
    """
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底：正则抠出最外层 {...}，再试一次。
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _snap_to_half_point(score: float) -> float:
    """把分数对齐到 0.5 的倍数——老师阅卷的真实扣分粒度。"""
    s = max(0.0, float(score))
    return round(s * 2) / 2


def _normalize_step_scores(raw_scores: list[dict[str, Any]], steps: list[StepItem], max_score: float) -> list[StepScore]:
    """把 LLM 返回的 step_scores 数组规范化成与 ``steps`` 一一对应的列表。

    处理顺序：
    1. 把 raw_scores 按 ``index`` 索引成 dict，方便查；
    2. 按 ``steps`` 的顺序逐项查表：查到就用，查不到给中性分（``per_step_max * 0.7``）
       + 标注「模型未返回该步评分」；
    3. 每步分数对齐 0.5；若对齐产生偏差且 reason 里没有「正确」「中性」字样，
       追加「已归整为 X 分」；
    4. 全部完成后做一次总和上限缩放（不得超过 ``max_score``）。
    """
    score_map: dict[int, tuple[float, str]] = {}
    for item in raw_scores:
        try:
            idx = int(item["index"])
            score = max(0.0, float(item["score"]))
            reason = str(item.get("reason", "")).strip() or "已评分。"
            score_map[idx] = (score, reason)
        except Exception:
            # 单条坏数据不影响其他步骤。
            continue

    per_step_max = (max_score / len(steps)) if steps else max_score
    out: list[StepScore] = []
    for s in steps:
        # 查不到时给中性分 = 每步满分的 70%，避免漏报某步时直接 0 分。
        raw_score, reason = score_map.get(s.index, (per_step_max * 0.7, "模型未返回该步评分，按中性分处理。"))
        snapped = _snap_to_half_point(raw_score)
        # 对齐带来明显差异且 reason 没有相关字样时，在 reason 里说明。
        if abs(snapped - raw_score) >= 0.01 and "正确" not in reason and "中性" not in reason:
            reason = f"{reason}（已归整为 {snapped} 分）"
        out.append(StepScore(index=s.index, score=snapped, reason=reason))

    # 硬性上限：所有步骤得分之和不得超过本题总分，按比例缩放。
    total = sum(x.score for x in out)
    if max_score > 0 and total > max_score + 0.01:
        scale = max_score / total
        for x in out:
            x.score = _snap_to_half_point(x.score * scale)

    return out


def score_steps(
    ocr_text: str, steps: list[StepItem], reference_steps: list[StepItem], reference_raw: str | None, use_llm: bool, max_score: float = 100.0,
    question_text: str | None = None,
) -> tuple[list[StepScore], float, str, str, dict[str, Any]]:
    """对一组已经切好的 StepItem 打分。

    主入口：先试 LLM 路径（:func:`_llm_score_steps`），失败就降级到
    规则路径（:func:`_rule_score_steps`）。返回 ``(step_scores, total,
    feedback, engine, diag_meta)``，engine 字段标记最终用了哪条路径，
    方便评测时区分「LLM 贡献的」和「规则兜底的」。
    """
    if use_llm:
        llm_res = _llm_score_steps(ocr_text=ocr_text, steps=steps, reference=reference_raw, max_score=max_score, question_text=question_text)
        if llm_res.get("ok"):
            scores = llm_res["scores"]
            total = llm_res["total"]
            feedback = llm_res["feedback"]
            meta = llm_res["meta"]
            meta.update({"scoring_mode": "llm"})
            return scores, total, feedback, "llm+rule", meta
        llm_error = str(llm_res.get("error", "LLM评分未成功，已回退到规则评分。"))
        llm_diag = llm_res.get("diag", {})
        # Persist the failure so we can diagnose silent rule-fallback cases
        # (the user-facing flow hides LLM internals, so we'd otherwise never
        # know why scoring dropped to rule-based).
        try:
            from app.services.db import save_llm_failure
            save_llm_failure(
                user_id=None,
                endpoint="/api/grade",
                stage="score-llm-fallback",
                error=(
                    f"llm_score_fallback: steps={len(steps)} max_score={max_score} "
                    f"err={llm_error[:200]} "
                    f"qtext_len={len(question_text or '')} ocr_len={len(ocr_text or '')}"
                ),
                raw_preview=str(llm_res.get("raw_preview") or json.dumps(llm_diag, ensure_ascii=False)[:1500])[:1500],
            )
        except Exception:
            pass
    else:
        llm_error = ""
        llm_diag = {}

    scores, total, feedback = _rule_score_steps(steps=steps, reference_steps=reference_steps, max_score=max_score)
    if use_llm and not llm_error and settings.llm_score_api_key:
        llm_error = "LLM评分未成功，已回退到规则评分。"
    elif use_llm and not settings.llm_score_api_key:
        llm_error = "未配置LLM_SCORE_API_KEY，已使用规则评分。"
    return scores, total, feedback, "rule-based", {
        "llm_used": False,
        "request_path": "",
        "llm_error": llm_error,
        "llm_diag": llm_diag,
        "scoring_mode": "rule",
    }


def score_questions(
    ocr_text: str,
    question_max_scores: dict[int, float] | None,
    reference_raw: str | None,
    use_llm: bool,
    question_text: str | None = None,
) -> tuple[list[QuestionGrade], list[StepItem], list[StepScore], float, float, str, str, dict[str, Any]]:
    """Grade an OCR text by first splitting into per-question chunks, then scoring each independently.

    Returns:
        questions, flat_steps, flat_step_scores, total_score (0-100), total_max_score,
        aggregated_feedback, engine, grading_meta.
    """
    pairs = split_text_into_questions(ocr_text)
    if not pairs:
        pairs = [(1, ocr_text)]

    n = len(pairs)
    # Default max_score: equal split summing to 100.
    default_each = 100.0 / n if n else 100.0

    questions: list[QuestionGrade] = []
    flat_steps: list[StepItem] = []
    flat_scores: list[StepScore] = []
    step_index_offset = 0
    engines_seen: set[str] = set()
    last_meta: dict[str, Any] = {}
    feedback_parts: list[str] = []

    for qno, q_text in pairs:
        max_score = float(question_max_scores.get(qno, default_each)) if question_max_scores else default_each
        steps = split_steps(q_text)
        if not steps:
            questions.append(QuestionGrade(
                qno=qno, max_score=max_score, score=0.0,
                steps=[], step_scores=[], feedback="未检测到可评分步骤。",
            ))
            feedback_parts.append(f"第 {qno} 题：未检测到可评分步骤。")
            continue
        step_scores, q_total, q_feedback, q_engine, q_meta = score_steps(
            ocr_text=q_text, steps=steps, reference_steps=[], reference_raw=reference_raw, use_llm=use_llm, max_score=max_score,
            question_text=question_text,
        )
        scaled_score = round(max(0.0, min(max_score, q_total)), 2)
        # Rebase step indices to global flat namespace for backward-compat consumers.
        for s in step_scores:
            flat_scores.append(StepScore(index=s.index + step_index_offset, score=s.score, reason=s.reason))
        for s in steps:
            flat_steps.append(StepItem(
                index=s.index + step_index_offset, raw=s.raw, normalized=s.normalized,
                has_equation=s.has_equation, confidence=s.confidence,
            ))
        step_index_offset += len(steps)
        engines_seen.add(q_engine)
        last_meta = q_meta
        questions.append(QuestionGrade(
            qno=qno, max_score=max_score, score=scaled_score,
            steps=steps, step_scores=step_scores, feedback=q_feedback,
        ))
        feedback_parts.append(f"第 {qno} 题：{q_feedback}")

    total_max_score = round(sum(q.max_score for q in questions), 2)
    raw_sum = sum(q.score for q in questions)
    total_score = round(raw_sum / total_max_score * 100, 2) if total_max_score > 0 else 0.0
    total_score = max(0.0, min(100.0, total_score))
    engine = "llm+rule" if "llm+rule" in engines_seen else ("rule-based" if engines_seen else "rule-based")
    aggregated_feedback = "\n\n".join(feedback_parts) if feedback_parts else "已按步骤评分。"
    meta: dict[str, Any] = dict(last_meta or {})
    meta["questions_meta"] = [
        {"qno": q.qno, "max_score": q.max_score, "score": q.score, "steps": len(q.steps)}
        for q in questions
    ]
    return questions, flat_steps, flat_scores, total_score, total_max_score, aggregated_feedback, engine, meta


def _build_steps_from_lines(step_lines: list[str]) -> list[StepItem]:
    """把「已经分好的步骤字符串列表」直接转成 StepItem 列表。

    新路径（前端传 ``problems`` 时）专用，**不再** 经过 OCR 文本再切题/
    再切步骤，避免题干被当步骤打分的历史 bug。

    仍然过滤两类无效行：
    - 空行或长度 < 2 的短行；
    - 注释行（如「（直接积分）」「由题意得」），用
      :func:`step_parser._is_annotation_line` 判定。
    """
    steps: list[StepItem] = []
    for ln in step_lines:
        s = str(ln or "").strip()
        if not s or len(s) < 2:
            continue
        # 归一化空白：连续空白压成一个空格。
        normalized = re.sub(r"\s+", " ", s)
        if _is_annotation_line(normalized):
            continue
        has_eq = bool(MATH_TOKEN_RE.search(normalized))
        steps.append(StepItem(
            index=len(steps) + 1,
            raw=s,
            normalized=normalized,
            has_equation=has_eq,
            confidence=0.85,  # 固定置信度（已经是 review 后的产物，不再走 OCR 置信度）
        ))
    return steps


def score_one_problem(
    qno: int,
    question_text: str,
    step_lines: list[str],
    max_score: float,
    reference_raw: str | None,
    use_llm: bool,
) -> tuple[QuestionGrade, list[StepItem], list[StepScore], str, str, dict[str, Any]]:
    """对「一道已经分好题干的题」做打分。

    这是新路径，前端通过 ``problems`` 字段把每题的 ``question_text`` 和
    ``step_lines`` 直接传过来，省掉了 OCR 文本再切题再切步骤的歧义——
    尤其避免了「题干被当步骤打分」这个历史问题。

    内部流程：
    1. :func:`_build_steps_from_lines` 把 step_lines 转成 StepItem 列表，
       并用 :func:`_step_appears_in_corpus` 过滤掉 LLM review 阶段脑补
       出来的、原文里没有的步骤；
    2. 调 :func:`score_steps` 走规则/LLM 打分；
    3. 包成 :class:`QuestionGrade` 返回。

    返回 tuple 的最后几个字段是为了让上层（main.py）能把元数据塞进
    grading_records.grading_meta_json 做诊断。
    """
    steps = _build_steps_from_lines(step_lines)
    if not steps:
        return (
            QuestionGrade(qno=qno, max_score=max_score, score=0.0,
                          steps=[], step_scores=[],
                          feedback="未检测到可评分步骤。"),
            [], [], "rule-based", "未检测到可评分步骤。", {"scoring_mode": "rule", "empty": True},
        )
    # 防御：丢弃 review LLM 可能脑补出来的步骤。真实 OCR 出来的步骤在
    # question_text 或 step_lines 中必然有近乎一致的原型；如果某步与
    # 整个语料的相似度都很低，多半是模型「自己续写」的。
    ocr_corpus = " ".join([question_text] + [str(s) for s in step_lines]).strip()
    if ocr_corpus:
        kept: list[StepItem] = []
        dropped_fabrications: list[str] = []
        for s in steps:
            if _step_appears_in_corpus(s.normalized, ocr_corpus):
                kept.append(s)
            else:
                dropped_fabrications.append(s.raw)
        if dropped_fabrications:
            # 落库一份诊断记录，方便事后统计脑补比例。
            try:
                from app.services.db import save_llm_failure
                save_llm_failure(
                    user_id=None,
                    endpoint="/api/grade",
                    stage="score-drop-fabricated",
                    error=f"qno={qno} dropped={len(dropped_fabrications)} fabricated steps",
                    raw_preview=str(dropped_fabrications)[:1200],
                )
            except Exception:
                pass
        # 只有「丢弃后仍剩步骤」时才真的丢；否则保留原样，避免把 OCR 碎片解法误删光。
        if kept:
            steps = kept
            # 丢弃后重新编号 1..N，避免索引断档。
            for i, s in enumerate(steps, start=1):
                s.index = i
    # 把题干 + 步骤拼成 LLM 上下文里的 ocr_text 字段。
    ocr_text_for_llm = "\n".join([question_text] + [s.raw for s in steps])
    step_scores, q_total, q_feedback, q_engine, q_meta = score_steps(
        ocr_text=ocr_text_for_llm, steps=steps, reference_steps=[], reference_raw=reference_raw,
        use_llm=use_llm, max_score=max_score,
        question_text=(question_text or "").strip() or None,
    )
    scaled = round(max(0.0, min(max_score, q_total)), 2)
    grade = QuestionGrade(
        qno=qno, max_score=max_score, score=scaled,
        steps=steps, step_scores=step_scores, feedback=q_feedback,
    )
    return grade, steps, step_scores, q_engine, q_feedback, q_meta


def _step_appears_in_corpus(step_text: str, corpus: str) -> bool:
    """判断某一步是否真的来自 OCR 语料，用于检测 LLM 是否「脑补」了步骤。

    策略：
    - 把 ``step_text`` 与 ``corpus`` 的每一行都去掉空白和 ``$``；
    - 逐行比对：要么是子串包含（一个是另一个的片段），要么相似度
      ``ratio() >= 0.75``（容忍 review 阶段字符顺序的小幅变化）。

    若 ``step_text`` 或 ``corpus`` 为空，或者步骤过短（<6 字符），保守
    视为「在语料中」，避免误删。

    典型反例：学生只写到 ``= 2 ln(x+1)|_0^1``，模型脑补了 ``= 2 ln 2``
    作为下一步——后者在任何 OCR 行里都找不到，会被本函数判负并丢弃。
    """
    if not step_text or not corpus:
        return True
    norm_step = re.sub(r"[\s$]+", "", step_text)
    if len(norm_step) < 6:
        return True
    corpus_lines = [re.sub(r"[\s$]+", "", ln) for ln in corpus.splitlines() if ln.strip()]
    if not corpus_lines:
        return True
    # 复用 SequenceMatcher 实例，每次只换第二条序列，比每次新建更快。
    matcher = SequenceMatcher(None, norm_step, "", autojunk=False)
    for ln in corpus_lines:
        if not ln:
            continue
        matcher.set_seq2(ln)
        ratio = matcher.ratio()
        # 子串包含 OR 高相似度，都算「在语料中」。
        if ratio >= 0.75 or norm_step in ln or ln in norm_step:
            return True
    return False
