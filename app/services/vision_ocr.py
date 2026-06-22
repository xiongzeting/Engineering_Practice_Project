"""纯视觉 LLM OCR 流水线（项目主力 OCR 路径）。

直接把作业图片 base64 喂给多模态模型（生产用 mimo-v2.5），让模型同时
完成「识别 + 分类 + 排版」，输出已经分好题的 JSON。

两阶段设计：
1. **第一阶段（视觉 LLM）**：模型对图直接出 JSON，包含 ``question_lines``
   和 ``step_lines``（或扁平的 ``lines``）。已经做了 Unicode→LaTeX 归一化。
2. **第二阶段（文本 LLM review）**：:func:`review_ocr_with_text_llm`
   把第一阶段所有行丢给一个更强的文本模型做后处理——重排序、丢弃噪声
   （试卷标题 / 学生姓名）、按题组织成 ``[{q:[...], s:[...]}, ...]``。

返回给前端的最终结构（:func:`vision_only_ocr`）：
``{ocr_text, question_text, steps_text, problems: [{qno, question_text, step_lines}, ...]}``。
``problems`` 字段是每题的题干+步骤，前端和 scorer 都直接消费它，省掉了
原来「文本再切题」的歧义。
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


_UNICODE_MATH_MAP: dict[str, str] = {
    "∫": r"\int", "∬": r"\iint", "∮": r"\oint",
    "∑": r"\sum", "∏": r"\prod", "⋃": r"\bigcup",
    "√": r"\sqrt", "∛": r"\sqrt[3]", "∜": r"\sqrt[4]",
    "×": r"\times", "÷": r"\div", "·": r"\cdot", "⋅": r"\cdot",
    "±": r"\pm", "∓": r"\mp",
    "≤": r"\le", "≥": r"\ge", "≦": r"\leqq", "≧": r"\geqq",
    "≠": r"\ne", "≈": r"\approx", "≡": r"\equiv", "∝": r"\propto",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "⟹": r"\Longrightarrow", "⟺": r"\Longleftrightarrow",
    "∞": r"\infty", "∂": r"\partial", "∇": r"\nabla",
    "π": r"\pi", "θ": r"\theta", "α": r"\alpha", "β": r"\beta", "γ": r"\gamma",
    "δ": r"\delta", "ε": r"\epsilon", "λ": r"\lambda", "μ": r"\mu", "ν": r"\nu",
    "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau", "φ": r"\varphi", "ω": r"\omega",
    "Δ": r"\Delta", "Σ": r"\Sigma", "Π": r"\Pi", "Ω": r"\Omega", "Φ": r"\Phi",
    "∈": r"\in", "∉": r"\notin", "∀": r"\forall", "∃": r"\exists",
    "∪": r"\cup", "∩": r"\cap", "∅": r"\emptyset",
    "⊂": r"\subset", "⊃": r"\supset", "⊆": r"\subseteq",
    "°": r"^{\circ}", "∠": r"\angle", "⊥": r"\perp", "∥": r"\parallel",
    "ℕ": r"\mathbb{N}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
    "ℝ": r"\mathbb{R}", "ℂ": r"\mathbb{C}",
    "…": r"\ldots", "⋯": r"\cdots", "⋮": r"\vdots", "⋱": r"\ddots",
    "≤": r"\le", "≥": r"\ge",
}

_SUPERSCRIPTS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
                 "⁺": "+", "⁻": "-", "⁼": "=", "⁽": "(", "⁾": ")", "ⁿ": "n", "ⁱ": "i"}
_SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9",
               "₊": "+", "₋": "-", "₌": "=", "₍": "(", "₎": ")", "ₙ": "n", "ₓ": "x", "ᵢ": "i", "ⱼ": "j", "ₐ": "a", "ₘ": "m", "ₖ": "k"}
_FRACTIONS = {"½": r"\frac{1}{2}", "⅓": r"\frac{1}{3}", "⅔": r"\frac{2}{3}",
              "¼": r"\frac{1}{4}", "¾": r"\frac{3}{4}", "⅕": r"\frac{1}{5}",
              "⅖": r"\frac{2}{5}", "⅗": r"\frac{3}{5}", "⅘": r"\frac{4}{5}",
              "⅙": r"\frac{1}{6}", "⅚": r"\frac{5}{6}", "⅛": r"\frac{1}{8}",
              "⅜": r"\frac{3}{8}", "⅝": r"\frac{5}{8}", "⅞": r"\frac{7}{8}"}


def _normalize_unicode_math(text: str) -> str:
    """把 ``$...$`` 数学段里的 Unicode 数学符号转成 LaTeX 命令。

    转换分三类，按顺序执行：
    1. **上下标**：连续的上标（如 ``x²`` → ``x^2``）或下标（如 ``aₙ`` →
       ``a_n``）合并成 ``^{...}`` / ``_{...}``；但若前一个字符已经是
       ``^`` 或 ``_``（例如 ``x^²``），跳过避免双重包裹；
    2. **分数**：``½`` → ``\frac{1}{2}``；
    3. **命名符号**：``∑`` → ``\\sum``、``∫`` → ``\\int`` 等。

    只处理 ``$...$`` 之间的内容，避免破坏正文。
    """
    if "$" not in text:
        return text

    def _convert_run(s: str) -> str:
        if not s:
            return ""
        # 1) 上下标：把连续字符打包进 ^{...}/_{...}，但要排除前面已有 ^/_ 的情况。
        def _make_repl(prefix: str, mapping: dict[str, str]):
            def _repl(m: re.Match[str]) -> str:
                chars = m.group(1)
                translated = "".join(mapping.get(c, c) for c in chars)
                return f"{prefix}{{{translated}}}" if len(translated) > 1 else f"{prefix}{translated}"
            return _repl
        if _SUPERSCRIPTS:
            cls = "".join(re.escape(c) for c in _SUPERSCRIPTS)
            s = re.sub(rf"(?<![\^_])([{cls}]+)", _make_repl("^", _SUPERSCRIPTS), s)
        if _SUBSCRIPTS:
            cls = "".join(re.escape(c) for c in _SUBSCRIPTS)
            s = re.sub(rf"(?<![\^_])([{cls}]+)", _make_repl("_", _SUBSCRIPTS), s)
        # 2) Unicode 分数字符 → \frac{...}{...}。
        for u, ltx in _FRACTIONS.items():
            s = s.replace(u, ltx)
        # 3) 命名数学符号 → 对应 LaTeX 命令。
        for u, ltx in _UNICODE_MATH_MAP.items():
            s = s.replace(u, ltx)
        return s

    # 逐字符扫描，用 $ 切换 in_math 状态，只对段内内容做转换。
    out_parts: list[str] = []
    in_math = False
    math_buf: list[str] = []
    for ch in text:
        if ch == "$":
            if in_math:
                out_parts.append(_convert_run("".join(math_buf)))
                math_buf = []
                out_parts.append("$")
                in_math = False
            else:
                out_parts.append("$")
                in_math = True
        else:
            if in_math:
                math_buf.append(ch)
            else:
                out_parts.append(ch)
    if math_buf:
        out_parts.append(_convert_run("".join(math_buf)))
    return "".join(out_parts)


_BRACKET_CHARS = set("()[]{}（）［］｛｝〔〕〈〉《》「」『』")
_PUNCT_ONLY = set("()[]{}（）。，；：、,.;;:!?！？·-—_/\u3000")

_FRAG_DROP_CHARS = _BRACKET_CHARS | _PUNCT_ONLY


def _cleanup_fragment_lines(lines: list[str]) -> list[str]:
    """清洗 OCR 输出中的「碎片行」。

    规则：
    - 丢空行；
    - 丢全是括号/标点的行（``(``、``）`` 这类常被 LLM 单独行输出）；
    - 丢与前一行完全相同的行（OCR 重复输出）；
    - 丢「短行且是前一行子串」的行：常见于 ``直接积分`` vs ``(直接积分)``
      这种被错误拆分的情况，前提是不含 ``$``（数学段不轻易合并）。
    """
    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        # 全是括号/标点的行：丢掉。
        if all(c in _FRAG_DROP_CHARS for c in s):
            continue
        if out:
            prev = out[-1].strip()
            # 完全相同的相邻行：丢掉。
            if s == prev:
                continue
            # 短行且与上一行互为子串：合并掉。
            if len(s) <= 12 and len(prev) <= 24 and (prev in s or s in prev) and "$" not in s:
                continue
        out.append(ln)
    return out


def _split_packed_math(line: str) -> list[str]:
    """把一行里挤了多个 ``$...$`` 数学段的 OCR 结果拆成多行。

    典型场景：mimo 把 ``$a=1$ $b=2$ $c=3$`` 当成一行输出，前端按行显示
    时会粘成一坨。本函数按 ``$...$`` 切分：第一段 + 前导文字保留为
    一行，之后每个新的 ``$...$`` 段单独成行。

    ``$`` 数量 < 4（不足两个完整段）时直接返回原行。
    """
    if line.count("$") < 4:
        return [line]
    parts = re.split(r"(\$[^$]+\$)", line)
    result: list[str] = []
    buffer = ""
    seen_math = False
    for p in parts:
        if not p.strip():
            continue
        is_math = p.startswith("$") and p.endswith("$") and len(p) >= 2
        if is_math:
            if not seen_math:
                # 第一个数学段：与前面的前导文字拼在同一行。
                buffer += p
                seen_math = True
            else:
                # 后续每个数学段：另起一行。
                if buffer.strip():
                    result.append(buffer.strip())
                buffer = p
        else:
            buffer += p
    if buffer.strip():
        result.append(buffer.strip())
    return result or [line]


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    """容错 JSON 解析，专门给 OCR LLM 输出用。

    三层兜底：
    1. 直接 :func:`json.loads`；
    2. 抠掉 markdown 代码块后再解析；
    3. 用正则找到最外层 ``{...}``，并尝试「补全未闭合的括号」+
       「去掉多余逗号」再解析（应对模型输出被截断）。
    """
    text = text.strip()
    if text.startswith("```"):
        # 抠掉 markdown 代码块外壳（```json ... ```）。
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # 兜底 1：从文本里抠出最外层 {...}。
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # 兜底 2：截断修复——补全未闭合的中括号和大括号，再去掉多余的尾逗号。
    opens = candidate.count("[") - candidate.count("]")
    braces = candidate.count("{") - candidate.count("}")
    repaired = candidate + ("]" * max(opens, 0)) + ("}" * max(braces, 0))
    repaired = re.sub(r",(\s*[\]}])", r"\1", repaired)
    try:
        return json.loads(repaired)
    except Exception:
        return None


def _extract_chat_content(payload: dict[str, Any]) -> str:
    """从 ``/chat/completions`` 响应里抽取助手回复纯文本。

    兼容两种 content 格式：字符串或 list-of-``{type, text}``（多模态）。
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


def _image_bytes_to_data_url(image_bytes: bytes) -> str | None:
    """把任意格式图片字节流转成 ``data:image/png;base64,...`` URL。

    多模态 LLM 通常要 data URL 而非 raw base64；这里用 PIL 先解码、
    统一转 RGB 再重编码成 PNG，保证原图即使是 JPG/BMP 也能被接受。
    解码失败返回 None。
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return None
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def review_ocr_with_text_llm(raw_lines: list[str]) -> dict[str, Any] | None:
    """Second-pass review using the text LLM (deepseek-v4-pro, same as scorer).

    The vision model (mimo-v2.5) frequently returns flat `lines` with wrong
    reading order or mixed classification. This function asks a stronger text
    model to clean up and return PER-PROBLEM pairs so the downstream frontend
    splitter can associate each question with its own solution.

    Returns: {"problems": [{"q": [...], "s": [...]}, ...], "dropped": [...]}
    on success, None on any failure (caller falls back to first-pass + heuristic).
    """
    if not settings.llm_score_api_key:
        return None
    cleaned_lines = [str(x).strip() for x in raw_lines if str(x).strip()]
    if not cleaned_lines:
        return None

    # Pass lines WITHOUT a "N." prefix. Earlier the prefix was meant as a
    # reference for the model, but it leaked into the output and the downstream
    # per-problem splitter mistook every numbered solution line for a new
    # question. The model classifies fine from plain text.
    joined = "\n".join(cleaned_lines)
    prompt = (
        "下面是从一张数学作业图片 OCR 出来的若干行。顺序可能错乱，可能混有试卷名称、页眉页脚、学生姓名等无关内容。\n"
        "请整理后输出 JSON。\n\n"
        "整理要求：\n"
        "1. 按图片中的题目为单位组织：每道独立题目对应一个对象，含 q（题干行列表）和 s（该题的解题步骤行列表）。\n"
        "2. q 包括：题号（如 1. / 15. / 第 1 题）、分值（如 (13分)）、已知条件、子问 (1)(2)(3)。\n"
        "3. s 包括：学生写的推导过程。常见开头：解、∵、∴、原式=、令、代入、化简、= 等。\n"
        "4. 子问 (1)(2)(3) 是同一道大题的一部分，必须放进同一个对象的 q 里，不要拆成独立题目。\n"
        "5. 多道独立题目：每道题对应一个对象，按图片阅读顺序排列。同一对象的 q 行必须连续，s 行也必须连续。\n"
        "6. 噪声丢弃：试卷标题（如\"定积分100题\"、\"期中试卷\"）、页眉页脚、学生姓名、班级、日期、页码等，放 dropped 数组。\n"
        "7. 不要改写数学表达式，只做分类和排序。\n"
        "8. 【严格抄录】q 和 s 中的每一行必须能在 OCR 输入行中找到对应（允许合并被错误拆分的同一行）。"
        "禁止自行推导、计算、补全、改写或添加 OCR 中没有出现的步骤。"
        "例如 OCR 只有 \"= 2 ln(x+1)|_0^1\"，绝不可以在 s 里追加 \"= 2 ln 2\"；"
        "学生没写的最终答案必须保持缺失，由评分阶段判断是否扣分。\n"
        "9. 【禁止给行编号】q 和 s 的每一行都直接抄录 OCR 原文，不要在行首添加 \"1.\" \"2.\" 这样的序号；"
        "只有当原文里题号（如 \"1.\"、\"15.\"）本身就是题干一部分时才保留。\n\n"
        f"OCR 行：\n{joined}\n\n"
        "返回 JSON（禁止 markdown 代码块，禁止解释）：\n"
        '{"problems": [{"q": ["题干行1", "题干行2"], "s": ["解题行1", "解题行2"]}], "dropped": ["被丢弃的行"]}'
    )
    body = {
        "model": settings.llm_score_model,
        "temperature": 0.0,
        "max_tokens": 8000,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "你是数学作业 OCR 后处理助手。只返回 JSON，不要解释。"},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {settings.llm_score_api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            f"{settings.llm_score_base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
            timeout=settings.llm_score_timeout_sec,
        )
    except Exception:
        return None
    if not resp.ok:
        return None
    raw = _extract_chat_content(resp.json())
    parsed = _safe_json_parse(raw)
    if not isinstance(parsed, dict):
        return None
    problems_raw = parsed.get("problems")
    if not isinstance(problems_raw, list) or not problems_raw:
        # Backward-compat: if model returned old {question, steps} schema,
        # treat as one problem with all of question as q and all of steps as s.
        q = parsed.get("question")
        s = parsed.get("steps")
        if isinstance(q, list) or isinstance(s, list):
            problems_raw = [{"q": q if isinstance(q, list) else [], "s": s if isinstance(s, list) else []}]
        else:
            return None
    problems: list[dict[str, list[str]]] = []
    # Defensive: if the model still echoes our old "N." prefix on solution
    # lines, strip it. A real question stem keeps its own natural "1." / "15."
    # prefix because that's part of the OCR original.
    def _strip_spurious_prefix(line: str, is_q: bool) -> str:
        s = str(line).strip()
        if is_q:
            return s  # never touch question stems
        # Only strip if the line clearly isn't itself numbered in the original
        # (solution lines starting with "= ", "解", "∵", "∴" shouldn't have N.).
        m = re.match(r"^(\d+)[.、)]\s+(.*)$", s)
        if m and re.match(r"^[=\s（(∴∵解原式令代入化简]", m.group(2)):
            return m.group(2)
        return s

    for p in problems_raw:
        if not isinstance(p, dict):
            continue
        q = p.get("q")
        s = p.get("s")
        q_lines = [str(x).strip() for x in (q or []) if str(x).strip()]
        s_lines = [_strip_spurious_prefix(x, is_q=False) for x in (s or []) if str(x).strip()]
        if not q_lines and not s_lines:
            continue
        problems.append({"q": q_lines, "s": s_lines})
    if not problems:
        return None
    dropped_raw = parsed.get("dropped") or []
    dropped = [str(x) for x in dropped_raw if str(x).strip()]
    return {"problems": problems, "dropped": dropped}


def _detect_qno_from_lines(lines: list[str]) -> int | None:
    """从前几行里提取题号（如 ``1.`` / ``15.`` / ``第 1 题``）。

    只扫前 3 行，题号范围限定 [1, 200]，超出视为误识别返回 None。
    """
    if not lines:
        return None
    for ln in lines[:3]:
        m = re.match(r"^\s*(?:第\s*)?(\d{1,3})(?:\s*题|\s*[.、)](?![\d.]))", str(ln or ""))
        if m:
            try:
                n = int(m.group(1))
                if 1 <= n <= 200:
                    return n
            except (ValueError, IndexError):
                continue
    return None


def _classify_lines_heuristic(lines: list[str]) -> tuple[list[str], list[str]]:
    """「兜底分类」：LLM 只返回 ``lines`` 数组（没分类）时，用正则恢复
    (question, steps) 结构。

    策略：
    1. 给每行打标签——Q（题干标志，如 ``1.``、``第 1 题``）/
       S（解题标志，如 ``解:``、``∵``、``∴``）/ O（其他）；
    2. **顺序修复**：若所有 Q 都在所有 S 之后（模型颠倒了输出），把
       列表旋转到「第一个 Q 在开头」。这是 mimo-v2.5 的典型 bug
       （解法先出，题干反而在最后）；
    3. 从第一个 S 切分：之前的都算 question，从第一个 S 起都算 steps。

    若没有 S 标志，无法分类，所有行都丢进 steps（保留内容，不强行拆）。
    """
    if not lines:
        return [], []

    q_marker = re.compile(r"^\s*(?:第\s*)?\d{1,3}(?:\s*题|\s*[.、)](?![\d.]))")
    s_marker = re.compile(
        r"^\s*(?:解\s*[:：]?|∵|∴|原式|\\because|\\therefore|令|由|代入|化简)\b"
    )

    def _tag(ln: str) -> str:
        t = str(ln or "").strip()
        if not t:
            return "O"
        cleaned = t.lstrip("$").strip()
        if q_marker.match(cleaned):
            return "Q"
        if s_marker.match(cleaned):
            return "S"
        return "O"

    tags = [_tag(ln) for ln in lines]
    q_indices = [i for i, x in enumerate(tags) if x == "Q"]
    s_indices = [i for i, x in enumerate(tags) if x == "S"]

    # 顺序修复：Q 全在 S 后面 → 模型颠倒了，旋转到第一个 Q 开头。
    if q_indices and s_indices and min(q_indices) > max(s_indices):
        rotate_at = q_indices[0]
        lines = list(lines[rotate_at:]) + list(lines[:rotate_at])
        tags = list(tags[rotate_at:]) + list(tags[:rotate_at])
        s_indices = [i for i, x in enumerate(tags) if x == "S"]

    if not s_indices:
        # 没有 S 标志：无法分类，全部丢给 steps，保证内容不丢。
        return [], list(lines)

    # 从第一个 S 切分：前面是题干，后面是步骤。
    split_at = s_indices[0]
    return list(lines[:split_at]), list(lines[split_at:])


def vision_only_ocr(image_bytes: bytes, return_debug: bool = False) -> dict[str, Any]:
    """纯视觉 LLM OCR 主入口。把图片 base64 喂给多模态 LLM（mimo-v2.5），
    再用文本 LLM（deepseek-v4-pro）做后处理。

    流程：
    1. **首轮（视觉 LLM）**：图片 → ``{question, steps}`` JSON。带 3 轮
       重试，每轮退避 ``2×(attempt+1)`` 秒；
    2. **次轮 review（文本 LLM）**：把首轮所有行交给 deepseek-v4-pro
       重排序、丢噪声（试卷标题/姓名）、按题组织成
       ``[{q, s}, ...]``。开关由环境变量 ``LLM_OCR_REVIEW`` 控制，默认开；
    3. **结构化输出**：构造 ``problems=[{qno, question_text, step_lines}]``，
       前端和 scorer 都直接消费这个字段，不再二次切题。

    ``return_debug=True`` 时额外返回 ``llm_debug`` 字段，含请求预览、响应
    预览、解析后的 JSON、重试诊断。
    """
    if not settings.llm_vision_api_key:
        return {"ok": False, "error": "LLM_VISION_API_KEY 未配置。"}
    image_url = _image_bytes_to_data_url(image_bytes)
    if not image_url:
        return {"ok": False, "error": "图片读取失败。"}

    prompt = (
        "TASK: OCR this math image into typed JSON.\n\n"
        "STRICT OUTPUT FORMAT (return ONLY this JSON, no prose, no markdown fence):\n"
        "{\n"
        "  \"question\": [\"...\", \"...\", ...],\n"
        "  \"steps\":    [\"...\", \"...\", ...],\n"
        "  \"notes\":    \"\"\n"
        "}\n\n"
        "CLASSIFICATION RULES:\n"
        "- Put a line in `question` if it is part of the PROBLEM STATEMENT: 题号 (e.g. \"1.\", \"15.\", \"第 1 题\"), 分值 (e.g. \"(13分)\"), 已知条件, the question being asked, and subquestion markers like \"(1)\", \"(2)\".\n"
        "- Put a line in `steps` if it is part of the STUDENT'S SOLUTION. Step starters include: \"解\", \"解:\", \"解：\", \"∵\", \"∴\", \"原式=\", \"=\", \"令\", \"由...得\", \"代入\", \"化简\", etc.\n"
        "- If a line is ambiguous, put it in `steps`.\n\n"
        "READING ORDER (CRITICAL):\n"
        "- Process the image strictly TOP-TO-BOTTOM, LEFT-TO-RIGHT.\n"
        "- The first lines of the image (typically the question header) MUST appear FIRST in the `question` array.\n"
        "- The solution lines that appear BELOW the question in the image MUST appear in `steps`.\n"
        "- NEVER output solution lines before question lines.\n\n"
        "SEPARATE PROBLEMS:\n"
        "- If the image contains multiple independent problems (each with its own number like 1. 2. 3.), keep each problem's question stem and solution as ADJACENT items in their respective arrays.\n"
        "- Example: image has 3 problems → `question` has 3 entries (one per problem's stem), `steps` has all solution lines in order.\n"
        "- Subquestions (1)(2)(3) within ONE problem stay TOGETHER in `question` (do not split into separate problems).\n\n"
        "FORMATTING:\n"
        "- Wrap each math expression in $...$.\n"
        "- Plain text lines (e.g. \"求 a;\", \"解:\") are NOT wrapped.\n\n"
        "EXAMPLE — if image shows:\n"
        "  15. (13分) 已知函数 f(x) = ln x + x² + ax + 2\n"
        "  在点 (2, f(2)) 处的切线与直线 2x+3y=0 垂直\n"
        "  (1) 求 a;\n"
        "  解: ∵ f(x) = ln x + x² + ax + 2\n"
        "  ∴ f'(x) = ...\n"
        "then output exactly:\n"
        "  {\"question\": [\"15. (13分) 已知函数 $f(x)=\\\\ln x+x^2+ax+2$\", \"在点 $(2,f(2))$ 处的切线与直线 $2x+3y=0$ 垂直\", \"(1) 求 $a$;\"], \"steps\": [\"解: $\\\\because f(x)=\\\\ln x+x^2+ax+2$\", \"$\\\\therefore f'(x)=\\\\frac{1}{x}+2x+a$\"], \"notes\": \"\"}\n\n"
        "DO NOT return a flat `lines` array. DO NOT omit any of the three keys."
    )
    body = {
        "model": settings.llm_vision_model,
        "temperature": 0.0,
        "max_tokens": 16000,
        "messages": [
            {"role": "system", "content": "你是数学 OCR 助手，只返回 JSON。"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    }
    url = f"{settings.llm_vision_base_url.rstrip('/')}/chat/completions"
    MAX_ATTEMPTS = 3
    RETRY_BACKOFF_SEC = 2.0
    last_error = ""
    last_raw_preview = ""
    attempt_diag: list[dict[str, Any]] = []
    try:
        parsed: dict[str, Any] | None = None
        raw = ""
        for attempt in range(MAX_ATTEMPTS):
            attempt_diag.append({"attempt": attempt + 1})
            try:
                resp = requests.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {settings.llm_vision_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=settings.vision_timeout_sec,
                )
            except Exception as exc:
                last_error = f"调用异常: {type(exc).__name__}: {exc}"
                attempt_diag[-1]["error"] = last_error
                if attempt < MAX_ATTEMPTS - 1:
                    import time as _t
                    _t.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue

            attempt_diag[-1]["status_code"] = int(resp.status_code)
            attempt_diag[-1]["ok"] = bool(resp.ok)

            if not resp.ok:
                last_error = f"LLM请求失败: {resp.status_code} {resp.text[:200]}"
                attempt_diag[-1]["response_preview"] = (resp.text or "")[:200]
                if attempt < MAX_ATTEMPTS - 1:
                    import time as _t
                    _t.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue

            raw = _extract_chat_content(resp.json())
            candidate = _safe_json_parse(raw)
            if not candidate:
                last_error = "LLM返回无法解析为JSON。"
                last_raw_preview = raw[: settings.llm_debug_max_chars]
                attempt_diag[-1]["raw_preview"] = last_raw_preview
                if attempt < MAX_ATTEMPTS - 1:
                    import time as _t
                    _t.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue

            # Shape sanity: must have at least one of question/steps/lines.
            if not any(isinstance(candidate.get(k), list) and candidate.get(k) for k in ("question", "steps", "lines")):
                last_error = "LLM返回缺少 question/steps 数组。"
                last_raw_preview = raw[: settings.llm_debug_max_chars]
                attempt_diag[-1]["raw_preview"] = last_raw_preview
                if attempt < MAX_ATTEMPTS - 1:
                    import time as _t
                    _t.sleep(RETRY_BACKOFF_SEC * (attempt + 1))
                continue

            parsed = candidate
            break

        if parsed is None:
            return {
                "ok": False,
                "error": last_error or "OCR 未成功。",
                "raw_preview": last_raw_preview,
                "attempts": attempt_diag,
            }

        question_raw = parsed.get("question", [])
        steps_raw = parsed.get("steps", [])
        # Backward-compat: older responses may still send a single `lines` array.
        lines_raw = parsed.get("lines", [])
        if not isinstance(question_raw, list):
            question_raw = []
        if not isinstance(steps_raw, list):
            steps_raw = []
        if not isinstance(lines_raw, list):
            lines_raw = []

        def _normalize_list(arr: list[Any]) -> list[str]:
            out: list[str] = []
            for x in arr:
                s = _normalize_unicode_math(str(x).strip())
                if s:
                    out.extend(_split_packed_math(s))
            return _cleanup_fragment_lines(out)

        question_lines = _normalize_list(question_raw)
        step_lines = _normalize_list(steps_raw)
        legacy_lines = _normalize_list(lines_raw)

        # Second-pass review with text LLM (deepseek-v4-pro). The vision model
        # often returns flat lines or mis-classifies; review fixes that and
        # drops noise (paper title, headers, etc.). Gated by env var.
        import os as _os
        review_enabled = _os.getenv("LLM_OCR_REVIEW", "1").lower() in ("1", "true", "yes")
        review_used = False
        review_dropped: list[str] = []
        review_problems: list[dict[str, list[str]]] = []
        if review_enabled:
            # Give review ALL lines from the first pass so it has full context.
            review_input = list(question_raw) + list(steps_raw) + list(lines_raw)
            reviewed = review_ocr_with_text_llm(review_input)
            if reviewed is not None:
                review_dropped = reviewed.get("dropped") or []
                for p in reviewed["problems"]:
                    review_problems.append({
                        "q": _normalize_list(p.get("q") or []),
                        "s": _normalize_list(p.get("s") or []),
                    })
                review_used = True

        if review_used and review_problems:
            # Interleave Q1,S1,Q2,S2,... so the downstream per-problem splitter
            # picks up question markers (1./2./15.) and associates each stem
            # with its own solution. No 【题目】/【解题步骤】 headers — those were
            # being mis-detected as a standalone question and caused all steps
            # to collapse into the last group.
            interleave: list[str] = []
            for p in review_problems:
                interleave.extend(p["q"])
                interleave.extend(p["s"])
            question_lines = [ln for p in review_problems for ln in p["q"]]
            step_lines = [ln for p in review_problems for ln in p["s"]]
            question_text = "\n".join(question_lines).strip()
            steps_text = "\n".join(step_lines).strip()
            ocr_text = "\n".join(interleave).strip()
        elif question_lines or step_lines:
            question_text = "\n".join(question_lines).strip()
            steps_text = "\n".join(step_lines).strip()
            combined_parts: list[str] = []
            if question_text:
                combined_parts.append(f"【题目】\n{question_text}")
            if steps_text:
                combined_parts.append(f"【解题步骤】\n{steps_text}")
            ocr_text = "\n\n".join(combined_parts).strip()
        else:
            # Legacy path: model returned only `lines`. Try to recover typed
            # structure via heuristic classification. This handles the case where
            # mimo-v2.5 dumps everything as flat lines (sometimes in wrong order,
            # e.g. solution first, question stem at the end).
            question_lines, step_lines = _classify_lines_heuristic(legacy_lines)
            question_text = "\n".join(question_lines).strip()
            steps_text = "\n".join(step_lines).strip()
            combined_parts: list[str] = []
            if question_text:
                combined_parts.append(f"【题目】\n{question_text}")
            if steps_text:
                combined_parts.append(f"【解题步骤】\n{steps_text}")
            ocr_text = "\n\n".join(combined_parts).strip() or "\n".join(legacy_lines).strip()

        # Build the per-problem structure for downstream scoring. When review
        # was used, problems come straight from the LLM (clean q/s split). When
        # review was skipped or returned nothing, fall back to deriving problems
        # from the typed/heuristic question_lines + step_lines so the frontend
        # and scorer still get a usable list.
        problems_out: list[dict[str, Any]] = []
        if review_used and review_problems:
            for idx, p in enumerate(review_problems, start=1):
                q_lines = list(p.get("q") or [])
                s_lines = list(p.get("s") or [])
                if not q_lines and not s_lines:
                    continue
                qno = _detect_qno_from_lines(q_lines) or idx
                problems_out.append({
                    "qno": qno,
                    "question_text": "\n".join(q_lines).strip(),
                    "step_lines": s_lines,
                })
        elif question_lines or step_lines:
            # Typed/heuristic path: we lost per-problem pairing, so emit one
            # problem wrapping everything. The downstream scorer will still
            # benefit from having the stem separated when q_lines starts with
            # a numbered marker.
            qno = _detect_qno_from_lines(question_lines) or 1
            problems_out.append({
                "qno": qno,
                "question_text": "\n".join(question_lines).strip(),
                "step_lines": step_lines,
            })

        result: dict[str, Any] = {
            "ok": True,
            "engine": f"vision-only:{settings.llm_vision_model}" + ("+review" if review_used else ""),
            "ocr_text": ocr_text,
            "question_text": question_text,
            "steps_text": steps_text,
            "problems": problems_out,
            "notes": str(parsed.get("notes", "") or ""),
            "llm_configured": True,
            "retries": sum(1 for a in attempt_diag if a.get("attempt", 0) > 1),
            "review_used": review_used,
            "review_dropped_count": len(review_dropped),
        }
        # Env-gated debug: persist raw LLM response so we can diagnose ordering /
        # classification bugs without making the user re-run with return_debug=True.
        import os
        if os.getenv("LLM_VISION_DEBUG"):
            try:
                from app.services.db import save_llm_failure
                parsed_summary = {
                    "question_count": len(question_lines),
                    "step_count": len(step_lines),
                    "legacy_count": len(legacy_lines),
                    "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else [],
                    "first_few_questions": question_lines[:3],
                    "first_few_steps": step_lines[:3],
                }
                save_llm_failure(
                    user_id=None,
                    endpoint="/api/ocr-vision-only",
                    stage="ocr-debug",
                    error=f"parsed_summary: {json.dumps(parsed_summary, ensure_ascii=False)[:1500]}",
                    raw_preview=raw[:3500],
                )
            except Exception:
                pass
        # Always log a tiny parsed-shape trace (cheap) so we can diagnose
        # typed-vs-legacy regressions without re-running with debug flag.
        try:
            from app.services.db import save_llm_failure
            parsed_shape = {
                "parsed_keys": list(parsed.keys()) if isinstance(parsed, dict) else [],
                "raw_question_count": len(parsed.get("question") or []) if isinstance(parsed, dict) else 0,
                "raw_steps_count": len(parsed.get("steps") or []) if isinstance(parsed, dict) else 0,
                "raw_lines_count": len(parsed.get("lines") or []) if isinstance(parsed, dict) else 0,
                "normalized_question_count": len(question_lines),
                "normalized_step_count": len(step_lines),
                "normalized_legacy_count": len(legacy_lines),
                "ocr_text_first_120": ocr_text[:120],
            }
            save_llm_failure(
                user_id=None,
                endpoint="/api/ocr-vision-only",
                stage="ocr-trace",
                error=f"shape: {json.dumps(parsed_shape, ensure_ascii=False)}",
                raw_preview="",
            )
        except Exception:
            pass
        if return_debug:
            result["llm_debug"] = {
                "request_path": "/chat/completions",
                "request_preview": prompt[: settings.llm_debug_max_chars],
                "response_preview": raw[: settings.llm_debug_max_chars],
                "parsed_preview": json.dumps(parsed, ensure_ascii=False)[: settings.llm_debug_max_chars],
                "attempts": attempt_diag,
            }
        return result
    except Exception as e:
        return {"ok": False, "error": f"调用异常: {type(e).__name__}: {e}", "attempts": attempt_diag}
