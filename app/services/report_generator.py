"""精确报告生成（异步）。

批改完成后前端先拿到简版结果（分数 + 反馈），「精确报告」是一份完整
Markdown 文档——含每题每步的详细评语、知识点归类、错因分析、复习建议。
生成需要再调一次 LLM，比较慢，所以走 ``BackgroundTasks`` 异步生成，
前端通过 ``record_id`` 轮询 ``/api/detail-report/{id}`` 获取状态。

状态机：``pending`` → ``ready``（成功）/ ``failed``（LLM 失败）。
失败可以 ``POST /api/detail-report/{id}/regenerate`` 重试。
"""
from __future__ import annotations

import json
import time
from typing import Any

import requests

from app.config import settings
from app.services.scorer import (
    _extract_chat_content,
    _extract_output_text,
    _safe_json_parse,
)


def _build_prompt(grading: dict[str, Any]) -> dict[str, Any]:
    """把 grading 结果整理成喂给 LLM 的 prompt payload。

    payload 包含五块：
    - **task**：身份设定（中国中学/大学数学老师）+ 输出约束（严格 JSON）；
    - **input_summary**：总分、满分、整体反馈、题数；
    - **questions**：逐题展开，每题含 qno/满分/得分/反馈/步骤列表
      （每步给 ``max_score_approx``，由该题满分按步数均分估算而来）；
    - **knowledge_graph**：KG 节点目录，标注每个节点是否被本次作业触及
      （``is_touched``）或是否是错题节点（``is_error_node``），引导 LLM
      用规范化的知识点名；
    - **output_schema + hard_rules**：示例 JSON 结构 + 5 条硬性约束
      （中文 / 字数上限 / 至少 1 条 warning 等）。

    最终返回 dict，调用方会 ``json.dumps`` 后作为 user message 发出去。
    """
    questions = grading.get("questions") or []
    q_payload = []
    for q in questions:
        steps = q.get("steps") or []
        # 把 step_scores 列表转成 {index: score_obj}，方便按 index 查。
        scores = {int(s["index"]): s for s in (q.get("step_scores") or [])}
        step_dump = []
        for s in steps:
            idx = int(s["index"])
            sc = scores.get(idx)
            step_dump.append(
                {
                    "index": idx,
                    "text": s.get("normalized") or s.get("raw") or "",
                    "score": float(sc["score"]) if sc else 0.0,
                    # 该题满分按步数均分，让 LLM 知道每步的"参考满分"。
                    "max_score_approx": float(q.get("max_score", 0)) / max(1, len(steps)),
                    "reason": (sc.get("reason") if sc else "") or "",
                }
            )
        q_payload.append(
            {
                "qno": int(q.get("qno", 0)),
                "max_score": float(q.get("max_score", 0)),
                "score": float(q.get("score", 0)),
                "feedback": q.get("feedback") or "",
                "steps": step_dump,
            }
        )

    # 注入 KG 上下文：让 LLM 用规范化的 KP 名（避免它自创名称）。
    kg_report = grading.get("kg_report") or {}
    kg_nodes_catalog: list[dict[str, Any]] = []
    if isinstance(kg_report, dict) and kg_report.get("ok"):
        touched = set(kg_report.get("touched_nodes") or [])
        error_nodes = set(kg_report.get("error_nodes") or [])
        for node in kg_report.get("all_nodes") or []:
            try:
                nid = node.get("id")
                if not nid:
                    continue
                kg_nodes_catalog.append({
                    "id": nid,
                    "name": node.get("name", ""),
                    "stage": node.get("stage", ""),
                    "is_touched": nid in touched,
                    "is_error_node": nid in error_nodes,
                })
            except Exception:
                continue

    return {
        "task": (
            "你是中国中学/大学数学老师，基于已完成的逐步批改结果，撰写一份精确的学情分析报告。"
            "请严格按 JSON 输出，不要任何额外文字。"
        ),
        "input_summary": {
            "total_score": float(grading.get("total_score", 0)),
            "total_max_score": float(grading.get("total_max_score", 100)),
            "overall_feedback": grading.get("feedback") or "",
            "question_count": len(q_payload),
        },
        "questions": q_payload,
        "knowledge_graph": {
            "available_nodes": kg_nodes_catalog,
            "touched_node_ids": sorted((kg_report or {}).get("touched_nodes") or []),
            "error_node_ids": sorted((kg_report or {}).get("error_nodes") or []),
            "hint": (
                "knowledge_points / warnings.point 尽量使用 knowledge_graph.available_nodes 中 is_touched=true 节点的 name；"
                "is_error_node=true 的节点是该学生本次出错的薄弱知识点，应在 warnings 中优先覆盖。"
            ) if kg_nodes_catalog else "（本次未启用 KG，可自由命名知识点）",
        },
        "output_schema": {
            "overview": "（一段话，整体评价学生表现，不超过 80 字）",
            "question_summaries": [
                {
                    "qno": 1,
                    "score": 18,
                    "max_score": 20,
                    "wrong_steps": [{"index": 3, "issue": "（≤30字，指出具体错误）"}],
                    "knowledge_points": ["一元一次方程", "移项"],
                    "comment": "（≤40字，给学生的本题评语）",
                }
            ],
            "knowledge_mastery": [
                {"name": "正负号运算", "level": "掌握良好|需巩固|需重新学习", "evidence": "（≤30字依据）"}
            ],
            "warnings": [
                {
                    "severity": "high|medium|low",
                    "point": "知识点名称",
                    "issue": "（≤40字描述问题）",
                    "affected_questions": [2, 3],
                    "suggestion": "（≤50字复习建议）",
                }
            ],
            "review_plan": "（一段话，后续学习建议，不超过 100 字）",
        },
        "hard_rules": [
            "wrong_steps 只包含得分明显低于该步满分的步骤（约 < 70%）。",
            "warnings 至少 1 条；若全部正确，给一条 'low' 级别的巩固建议。",
            "knowledge_mastery 覆盖所有题目涉及的主要知识点（去重后通常 3-6 个）。",
            "knowledge_points 名称优先采用 knowledge_graph.available_nodes 中的 name 字段。",
            "所有文字用中文，简洁具体，不要套话。",
        ],
    }


def _call_llm(prompt: dict[str, Any]) -> str:
    """调用评分 LLM 生成精确报告，返回模型原始文本。

    双路 fallback：
    1. **Chat Completions**（``response_format=json_object`` 强制 JSON 模式）；
    2. **Responses API**（用 ``input`` 字段，OpenAI 较新接口）。

    超时设置（connect, read）来自 ``settings.llm_score_*_timeout_sec``。
    两条路任一返回非空文本即返回；都失败抛 RuntimeError，由调用方决定
    是否落库为 failed。
    """
    if not settings.llm_score_api_key:
        raise RuntimeError("未配置LLM_SCORE_API_KEY")

    headers = {
        "Authorization": f"Bearer {settings.llm_score_api_key}",
        "Content-Type": "application/json",
    }
    timeout = (settings.llm_score_connect_timeout_sec, settings.llm_score_timeout_sec)
    prompt_json = json.dumps(prompt, ensure_ascii=False)

    # Chat Completions：messages + response_format=json_object，主流供应商都支持。
    chat_body = {
        "model": settings.llm_score_model,
        "messages": [
            {"role": "system", "content": "你是数学学情分析助手。只返回JSON。"},
            {"role": "user", "content": prompt_json},
        ],
        "temperature": 0.2,  # 报告任务稍微给点创造性，比纯评分 0.0 高一点
        "max_tokens": 4000,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    # Responses：OpenAI 新接口，作为 fallback。
    responses_body = {"model": settings.llm_score_model, "input": prompt_json}

    last_err = ""
    for mode in ("chat", "responses"):
        try:
            if mode == "chat":
                t0 = time.perf_counter()
                resp = requests.post(
                    f"{settings.llm_score_base_url.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=chat_body,
                    timeout=timeout,
                )
                if not resp.ok:
                    last_err = f"chat HTTP {resp.status_code}: {resp.text[:200]}"
                    continue
                text = _extract_chat_content(resp.json())
            else:
                t1 = time.perf_counter()
                resp = requests.post(
                    f"{settings.llm_score_base_url.rstrip('/')}/responses",
                    headers=headers,
                    json=responses_body,
                    timeout=timeout,
                )
                if not resp.ok:
                    last_err = f"responses HTTP {resp.status_code}: {resp.text[:200]}"
                    continue
                text = _extract_output_text(resp.json())
            if text:
                return text
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
    raise RuntimeError(f"LLM调用失败: {last_err}")


def _format_question_summary(q: dict[str, Any]) -> str:
    """把 LLM 返回的单题 summary 渲染成 Markdown 片段。

    输出格式：题号 + 得分、错步列表（带 ❌）、知识点 chip、评语。
    全对则显示 ✅。
    """
    qno = q.get("qno", "?")
    score = q.get("score", 0)
    max_score = q.get("max_score", 0)
    lines = [f"### 第 {qno} 题（{score}/{max_score}）"]

    wrong = q.get("wrong_steps") or []
    if wrong:
        for w in wrong:
            idx = w.get("index", "?")
            issue = (w.get("issue") or "").strip()
            lines.append(f"- ❌ 步骤 {idx}：{issue}")
    else:
        lines.append("- ✅ 全部步骤正确")

    kps = q.get("knowledge_points") or []
    if kps:
        lines.append(f"- 知识点：{('、'.join(kps))}")

    comment = (q.get("comment") or "").strip()
    if comment:
        lines.append(f"- 评语：{comment}")

    return "\n".join(lines)


def _format_mastery_table(items: list[dict[str, Any]]) -> str:
    """把 knowledge_mastery 列表渲染成 3 列 Markdown 表格。"""
    if not items:
        return "（无）"
    lines = ["| 知识点 | 掌握度 | 依据 |", "| --- | --- | --- |"]
    for it in items:
        name = (it.get("name") or "").strip() or "—"
        level = (it.get("level") or "").strip() or "—"
        evidence = (it.get("evidence") or "").strip() or "—"
        lines.append(f"| {name} | {level} | {evidence} |")
    return "\n".join(lines)


def _format_warnings(items: list[dict[str, Any]]) -> str:
    """把 warnings 渲染成"高/中/低"级别的复习提示列表。

    severity ``high/medium/low`` 映射为中文"高/中/低"。每条带：
    - 知识点名 + 问题描述；
    - 若有 affected_questions，附"涉及 第 N 题"；
    - 若有 suggestion，单独缩进一行。
    """
    if not items:
        return "（本次作业未发现明显薄弱点）"
    sev_label = {"high": "高", "medium": "中", "low": "低"}
    lines = []
    for i, w in enumerate(items, 1):
        sev = sev_label.get((w.get("severity") or "low").lower(), "低")
        point = (w.get("point") or "").strip() or "—"
        issue = (w.get("issue") or "").strip()
        affected = w.get("affected_questions") or []
        affected_str = "、".join(f"第 {n} 题" for n in affected) if affected else "—"
        suggestion = (w.get("suggestion") or "").strip()
        lines.append(f"{i}. 【{sev}】**{point}** — {issue}")
        if affected_str != "—":
            lines[-1] += f"（涉及 {affected_str}）"
        if suggestion:
            lines.append(f"   - 建议：{suggestion}")
    return "\n".join(lines)


def _format_kg_section(kg_report: dict[str, Any] | None) -> str | None:
    """渲染知识图谱诊断区块：薄弱知识点表 + 前置依赖链。

    仅当 ``kg_report.ok=True`` 且有 touched 节点时才返回内容；否则返回
    None，调用方据此判断要不要把这一节写进 Markdown。

    包含的小区块：
    - 概览（涉及知识点数、薄弱节点、LLM 补标数）；
    - 薄弱知识点详情表（编号 / 名 / 学段 / 前置依赖）；
    - 超纲知识点（大学内容，不计入掌握度）；
    - 前置依赖链（最多展示 20 条边，超出折叠）。
    """
    if not isinstance(kg_report, dict) or not kg_report.get("ok"):
        return None
    touched = list(kg_report.get("touched_nodes") or [])
    error_nodes = list(kg_report.get("error_nodes") or [])
    all_nodes = {n.get("id"): n for n in (kg_report.get("all_nodes") or []) if isinstance(n, dict)}
    sub_edges = kg_report.get("sub_edges") or []
    if not touched:
        return None

    stage_label = {"primary": "小学", "middle": "初中", "high": "高中"}
    lines = ["> 本节由知识图谱模块自动生成，结合规则匹配与大模型二次判别。", ""]

    lines.append(f"- **本次涉及知识点**：{len(touched)} 个")
    if error_nodes:
        weak_pairs = []
        for nid in error_nodes:
            node = all_nodes.get(nid) or {}
            weak_pairs.append(f"{nid}·{node.get('name', '')}")
        lines.append(f"- **薄弱知识点（含错步）**：{('、'.join(weak_pairs))}")
    if kg_report.get("llm_used"):
        lines.append(
            f"- **大模型补标**：已对 {kg_report.get('llm_ambiguous', 0)} 个规则未命中的步骤做 LLM 二次判别"
            f"（耗时 {kg_report.get('llm_elapsed_ms', 0)} ms）"
        )
    lines.append("")

    if error_nodes:
        # 薄弱节点详情表：展示学段 + 前置依赖，帮学生回看缺失基础。
        lines.append("### 薄弱知识点详情")
        lines.append("")
        lines.append("| 编号 | 名称 | 学段 | 前置依赖 |")
        lines.append("| --- | --- | --- | --- |")
        for nid in error_nodes:
            node = all_nodes.get(nid) or {}
            prereqs = node.get("prerequisites") or []
            prereq_str = "、".join(
                f"{p}·{(all_nodes.get(p) or {}).get('name', '')}" for p in prereqs
            ) or "无"
            stage = stage_label.get(node.get("stage", ""), "—")
            lines.append(f"| {nid} | {node.get('name', '')} | {stage} | {prereq_str} |")
        lines.append("")

    unmapped = kg_report.get("unmapped_topics") or []
    if unmapped:
        # 超纲知识点：超出 K1-12 本体（大学微积分/线代等），只展示不计入掌握度。
        lines.append("### 超纲 / 暂未纳入本体的知识点")
        lines.append("")
        lines.append("> 下列知识点超出 K1-12 本体范围（如大学微积分、线性代数等），仅作展示，不计入图谱掌握度。")
        lines.append("")
        chips = "、".join(f"{it['name']}（{it['count']}）" for it in unmapped if isinstance(it, dict) and it.get("name"))
        lines.append(chips or "—")
        lines.append("")

    if sub_edges:
        # 前置依赖链：最多展示 20 条边，超出折叠成"另有 N 条边省略"。
        lines.append("### 前置依赖链（本次作业涉及）")
        lines.append("")
        for e in sub_edges[:20]:
            f_name = (all_nodes.get(e.get("from")) or {}).get("name", e.get("from"))
            t_name = (all_nodes.get(e.get("to")) or {}).get("name", e.get("to"))
            lines.append(f"- {f_name} → {t_name}")
        if len(sub_edges) > 20:
            lines.append(f"- …（另有 {len(sub_edges) - 20} 条边省略）")
        lines.append("")

    return "\n".join(lines)


def _assembly_markdown(grading: dict[str, Any], parsed: dict[str, Any]) -> str:
    """把 LLM 返回的 JSON 拼装成最终 Markdown 报告。

    Markdown 顺序：
    1. 标题 + 卷面分 + 生成时间；
    2. 总览（LLM 写的整体评价）；
    3. 分题明细（每题独立小节）；
    4. 知识图谱诊断（仅当 KG 可用时插入）；
    5. 知识点掌握情况表；
    6. 错误预警（重点复习）；
    7. 后续学习建议。

    任一段缺失都走优雅 fallback（显示"模型未返回"）。
    """
    total = float(grading.get("total_score", 0))
    total_max = float(grading.get("total_max_score", 100))
    overview = (parsed.get("overview") or "").strip() or "（模型未返回总览）"
    q_summaries = parsed.get("question_summaries") or []
    mastery = parsed.get("knowledge_mastery") or []
    warnings = parsed.get("warnings") or []
    review_plan = (parsed.get("review_plan") or "").strip() or "（模型未返回学习建议）"

    sections = [
        "# 数学作业精确批改报告",
        "",
        f"> 卷面分 **{total}/{total_max}**  ·  生成时间 {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 总览",
        "",
        overview,
        "",
        "## 分题明细",
        "",
    ]
    if q_summaries:
        for q in q_summaries:
            sections.append(_format_question_summary(q))
            sections.append("")
    else:
        sections.append("（模型未返回分题明细）")
        sections.append("")

    kg_section = _format_kg_section(grading.get("kg_report"))
    if kg_section:
        sections.append("## 知识图谱诊断")
        sections.append("")
        sections.append(kg_section)

    sections.append("## 知识点掌握情况")
    sections.append("")
    sections.append(_format_mastery_table(mastery))
    sections.append("")
    sections.append("## ⚠ 错误预警（重点复习）")
    sections.append("")
    sections.append(_format_warnings(warnings))
    sections.append("")
    sections.append("## 后续学习建议")
    sections.append("")
    sections.append(review_plan)
    sections.append("")
    return "\n".join(sections)


def generate_detail_report(grading: dict[str, Any]) -> str:
    """精确报告生成主入口（同步实现，由 BackgroundTasks 包一层异步调用）。

    流程：
    1. :func:`_build_prompt` 把 grading 整理成结构化 prompt；
    2. :func:`_call_llm` 调评分 LLM 拿到原始文本；
    3. :func:`_safe_json_parse` 解析 JSON；
    4. :func:`_assembly_markdown` 拼成 Markdown。

    任意一步失败抛 RuntimeError，调用方（main.py 里的 background task）
    会落库为 ``status='failed'``，前端可触发 regenerate 重试。
    """
    prompt = _build_prompt(grading)
    raw = _call_llm(prompt)
    parsed = _safe_json_parse(raw)
    if not parsed:
        raise RuntimeError("LLM 返回内容无法解析为 JSON")
    return _assembly_markdown(grading, parsed)
