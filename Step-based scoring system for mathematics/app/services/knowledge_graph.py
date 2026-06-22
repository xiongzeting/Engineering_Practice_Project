"""Knowledge graph module: 40 K1-K12 math knowledge points with prerequisite DAG.

Source of truth: `data/kg_ontology.yaml` (version-controllable, admin-editable).
Loaded into NetworkX DiGraph on first access; module-level singleton.

CRUD operations (upsert/delete) write-through to YAML atomically (temp + rename).
Error→knowledge-point mapping: rule-first (regex on step text + inferred error
type from reason), LLM refinement layer refines ambiguous steps in one batch.

Key invariants:
- `scorer.py` is never imported here — KG enrichment is a pure post-processor.
- All public functions tolerate bad input (return safe defaults) so wrapping
  `enrich_report_with_kg` in try/except at call site is double protection.
"""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

import networkx as nx
import requests
import yaml

from app.config import settings

ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "data" / "kg_ontology.yaml"

STAGE_ORDER = {"primary": 0, "middle": 1, "high": 2}
VALID_ERROR_TYPES = {"calculation", "sign", "variable", "logic", "other"}

_graph: nx.DiGraph | None = None
_lock = threading.Lock()


# === Loading ===

def load_graph(force: bool = False) -> nx.DiGraph:
    """懒加载知识图谱单例。``force=True`` 时强制重新读 YAML（写入后用）。"""
    global _graph
    if _graph is None or force:
        with _lock:
            if _graph is None or force:
                _graph = _build_graph()
    return _graph


def _build_graph() -> nx.DiGraph:
    """从 ``data/kg_ontology.yaml`` 构造 :class:`networkx.DiGraph`。

    - 检查每个节点都有 ``id``；
    - 检查 id 不重复；
    - 检查 ``prerequisites`` 引用的节点都存在；
    - 检查整张图是 DAG（无环）。

    任何一条违反都抛 ValueError，避免脏 YAML 让系统进入不一致状态。
    """
    if not ONTOLOGY_PATH.is_file():
        raise FileNotFoundError(f"ontology missing: {ONTOLOGY_PATH}")
    data = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{ONTOLOGY_PATH}: expected list of nodes")
    g = nx.DiGraph()
    seen: set[str] = set()
    # 第 1 遍：建节点。
    for node in data:
        if not isinstance(node, dict) or "id" not in node:
            continue
        nid = str(node["id"])
        if nid in seen:
            raise ValueError(f"duplicate node id: {nid}")
        seen.add(nid)
        g.add_node(nid, **node)
    # 第 2 遍：建先修边（prerequisite → node）。
    for node in data:
        nid = str(node["id"])
        for prereq in node.get("prerequisites") or []:
            prereq = str(prereq)
            if prereq not in g:
                raise ValueError(f"node {nid} -> unknown prereq {prereq}")
            g.add_edge(prereq, nid)
    # DAG 校验：先修关系不能成环。
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("ontology has cycle — fix prerequisites")
    return g


# === Read ===

def get_all_nodes() -> list[dict]:
    """返回所有 KG 节点（按 id 排序）。"""
    g = load_graph()
    return [_node_payload(g, n) for n in sorted(g.nodes)]


def get_node(node_id: str) -> dict | None:
    """按 id 取单个节点。无则返回 None。"""
    g = load_graph()
    if node_id not in g:
        return None
    return _node_payload(g, node_id)


def get_all_edges() -> list[dict]:
    """返回所有先修边 ``{from, to}``。前端画图用。"""
    g = load_graph()
    return [{"from": u, "to": v} for u, v in g.edges]


def _node_payload(g: nx.DiGraph, nid: str) -> dict:
    """从图中抽出一个节点的「对外字典」：所有属性 + id + prerequisites 列表。"""
    attrs = dict(g.nodes[nid])
    attrs["id"] = nid
    # predecessors 是「先修节点」列表；NetworkX 存的是反边。
    attrs["prerequisites"] = list(g.predecessors(nid))
    return attrs


# === Mutation (write-through) ===

def upsert_node(data: dict) -> dict:
    """新增或更新节点（按 id 匹配）。必填字段：``id`` / ``name`` / ``stage``。

    更新时做浅合并：保留 YAML 里已有但本次未传的字段，再用新值覆盖。
    写完后 :func:`load_graph` ``force=True`` 重载图，保证后续读一致。
    """
    required = {"id", "name", "stage"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    with _lock:
        nodes = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8")) or []
        normalized = _normalize_node(data)
        nid = normalized["id"]
        # 找现有节点位置；找不到就追加。
        idx = next((i for i, n in enumerate(nodes) if str(n.get("id")) == nid), None)
        if idx is None:
            nodes.append(normalized)
        else:
            nodes[idx] = {**nodes[idx], **normalized}
        _atomic_write_yaml(nodes)
    load_graph(force=True)
    node = get_node(nid)
    assert node is not None
    return node


def delete_node(node_id: str) -> bool:
    """删除节点。返回是否真的删了（False = 节点不存在）。

    还会顺手把其他节点 ``prerequisites`` 中对它的引用清掉，避免悬挂引用。
    """
    with _lock:
        nodes = yaml.safe_load(ONTOLOGY_PATH.read_text(encoding="utf-8")) or []
        new_nodes = [n for n in nodes if str(n.get("id")) != str(node_id)]
        if len(new_nodes) == len(nodes):
            return False
        # 清理其他节点对被删节点的先修引用。
        for n in new_nodes:
            prereqs = n.get("prerequisites") or []
            n["prerequisites"] = [p for p in prereqs if str(p) != str(node_id)]
        _atomic_write_yaml(new_nodes)
    load_graph(force=True)
    return True


def _normalize_node(data: dict) -> dict:
    """把前端传进来的节点 dict 规范化成 YAML 期望的结构。

    - ``stage`` 必须是 ``primary`` / ``middle`` / ``high`` 之一，否则回退到 ``primary``；
    - ``error_type_hints`` 过滤掉非法错因值；
    - 所有字段都强制字符串化（id/name）或整数化（grade）。
    """
    stage = data.get("stage", "primary")
    if stage not in STAGE_ORDER:
        stage = "primary"
    hints = [h for h in (data.get("error_type_hints") or []) if h in VALID_ERROR_TYPES]
    return {
        "id": str(data["id"]),
        "name": str(data["name"]),
        "stage": stage,
        "grade": int(data.get("grade") or 0),
        "prerequisites": [str(p) for p in (data.get("prerequisites") or [])],
        "error_type_hints": hints,
        "keyword_patterns": [str(p) for p in (data.get("keyword_patterns") or [])],
        "description": str(data.get("description") or ""),
    }


def _atomic_write_yaml(nodes: list[dict]) -> None:
    """原子写 YAML：先写到 ``.yaml.tmp``，再 :func:`Path.replace` 替换原文件。

    ``replace`` 在同一文件系统下是原子操作，避免写一半进程崩溃导致 YAML 损坏。
    """
    ONTOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ONTOLOGY_PATH.with_suffix(".yaml.tmp")
    tmp.write_text(
        yaml.safe_dump(nodes, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    tmp.replace(ONTOLOGY_PATH)


# === Per-class KG overrides ===
# 教师无权改基础 ontology（只有管理员能改）。教师只能在自己的班级层级
# 叠一层 override：'upsert' 表示新增/替换本班节点；'delete' 表示隐藏
# 基础节点。最终的有效 KG = 基础 ∪ 班级 upsert − 班级 delete。

def _override_conn():
    """打开覆盖层的 SQLite 连接（复用 db 模块的连接工厂）。"""
    from app.services import db
    return db._connect()


def list_class_overrides(class_id: int) -> list[dict]:
    """列出某班级的所有 KG 覆盖记录（含 op 与 payload）。"""
    import json as _json
    with _override_conn() as conn:
        rows = conn.execute(
            "SELECT node_id, op, payload_json, updated_at FROM class_kg_overrides WHERE class_id=? ORDER BY node_id",
            (class_id,),
        ).fetchall()
    out = []
    for r in rows:
        try:
            payload = _json.loads(r["payload_json"] or "{}")
        except Exception:
            payload = {}
        out.append({
            "node_id": r["node_id"],
            "op": r["op"],
            "payload": payload,
            "updated_at": r["updated_at"],
        })
    return out


def upsert_class_override(class_id: int, data: dict) -> dict:
    """班级层级「新增/替换」节点。必填 ``id`` / ``name`` / ``stage``。

    使用 ``ON CONFLICT(class_id, node_id) DO UPDATE`` 做 upsert：
    - 若该 (class, node) 已有 override 记录，覆盖其 op 和 payload；
    - 否则插入新记录。

    返回写入的 override 描述。
    """
    import json as _json
    required = {"id", "name", "stage"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    normalized = _normalize_node(data)
    nid = str(normalized["id"])
    with _override_conn() as conn:
        conn.execute(
            """
            INSERT INTO class_kg_overrides(class_id, node_id, op, payload_json)
            VALUES(?,?,?,?)
            ON CONFLICT(class_id, node_id) DO UPDATE SET op='upsert', payload_json=excluded.payload_json, updated_at=CURRENT_TIMESTAMP
            """,
            (class_id, nid, "upsert", _json.dumps(normalized, ensure_ascii=False)),
        )
        conn.commit()
    return {"class_id": class_id, "node_id": nid, "op": "upsert", "payload": normalized}


def delete_class_override(class_id: int, node_id: str) -> dict:
    """班级层级「隐藏」一个基础节点。

    仍写入 override 表，只是 ``op='delete'``；这样撤销时直接删 override
    记录即可（见 :func:`restore_class_override`）。
    """
    nid = str(node_id)
    with _override_conn() as conn:
        conn.execute(
            """
            INSERT INTO class_kg_overrides(class_id, node_id, op, payload_json)
            VALUES(?,?,?,?)
            ON CONFLICT(class_id, node_id) DO UPDATE SET op='delete', payload_json='{}', updated_at=CURRENT_TIMESTAMP
            """,
            (class_id, nid, "delete", "{}"),
        )
        conn.commit()
    return {"class_id": class_id, "node_id": nid, "op": "delete"}


def restore_class_override(class_id: int, node_id: str) -> bool:
    """撤销班级层级的某条 override（恢复成基础行为）。返回是否真的删了一条。"""
    with _override_conn() as conn:
        cur = conn.execute(
            "DELETE FROM class_kg_overrides WHERE class_id=? AND node_id=?",
            (class_id, str(node_id)),
        )
        conn.commit()
        return cur.rowcount > 0


def get_effective_class_nodes(class_id: int) -> list[dict]:
    """返回某班级的「有效 KG」：基础节点 ∪ 班级 upsert − 班级 delete。

    算法：
    1. 取基础节点列表；
    2. 对每个基础节点：
       - 命中 delete override → 跳过；
       - 命中 upsert override → 用 payload 覆盖；
       - 无 override → 原样返回；
    3. 剩下的 override（基础里没有的）当作「班级新增节点」追加。

    每个节点都标了 ``class_override``（是否被本班修改过）和 ``class_added``
    （是否是本班新加的），前端据此显示「本班新增」角标。
    """
    import copy
    base = get_all_nodes()
    overrides = {o["node_id"]: o for o in list_class_overrides(class_id)}
    effective = []
    for node in base:
        nid = str(node["id"])
        if nid in overrides:
            ov = overrides.pop(nid)  # 用掉一条
            if ov["op"] == "delete":
                continue  # 本班隐藏
            # upsert：合并 payload，保留 id。
            effective.append({**node, **ov["payload"], "id": nid, "class_override": True})
        else:
            effective.append({**node, "class_override": False})
    # 剩下的 override 都是基础里没有的新节点。
    for nid, ov in overrides.items():
        if ov["op"] == "upsert":
            payload = dict(ov["payload"])
            payload["id"] = nid
            payload["class_override"] = True
            payload["class_added"] = True
            effective.append(payload)
    return effective


# === Error type inference ===

_ERROR_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("sign", [r"符号", r"正负", r"负号", r"正号", r"\\s*-\\s*"]),
    ("variable", [r"变量", r"未知数", r"字母", r"把.{0,3}写成", r"变量替换", r"代换"]),
    ("logic", [r"逻辑", r"思路", r"方法错", r"公式错", r"公式记错", r"推导", r"方向"]),
    ("calculation", [r"计算", r"运算", r"算错", r"口算", r"加减乘除"]),
]


def infer_error_type(reason: str) -> str:
    """从评分 reason 文本里推断错因类型。

    按优先级匹配四类：``sign``（符号错）> ``variable``（变量错）>
    ``logic``（逻辑/公式错）> ``calculation``（计算错）。都不命中返回
    ``other``。每类下有若干正则，命中任一即归类。

    匹配顺序很关键——「正负号错」属于 sign 而不是 calculation，所以
    sign 放最前。
    """
    if not reason:
        return "other"
    text = str(reason)
    for etype, patterns in _ERROR_TYPE_PATTERNS:
        for p in patterns:
            try:
                if re.search(p, text):
                    return etype
            except re.error:
                continue
    return "other"


# === Out-of-scope detection (university-level content) ===

# 这一组正则用来识别「明显超出 K1-12 范围」的步骤（极限、积分、导数、
# 线性代数、概率论）。命中时跳过规则层，直接让 LLM 自由打标签
# （free_text_tags），避免把大学内容硬塞进中小学本体。
_OUT_OF_SCOPE_PATTERNS: list[str] = [
    # 极限
    r"\\?lim\s*[(\.]",
    r"\\?to\s*\\?infty",
    r"->\s*\\?infty",
    # 积分（LaTeX \int 或 Unicode ∫）
    r"\\?int[_\s^{]",
    r"∫",
    r"∬|∭",
    # 导数
    r"\\?dfrac\s*\{\s*d",
    r"\\?partial",
    r"d\s*/\s*dx",
    r"d\s*/\s*dt",
    r"[a-zA-Z]\s*'\s*\(",   # f'(x), g'(t)
    r"[a-zA-Z]''\s*\(",      # f''(x)
    # 求和 / 无穷
    r"\\?sum\s*[_{]",
    r"∑",
    r"\\?infty",
    r"∞",
    # 线性代数
    r"\\?nabla",
    r"\\?det\b",
    r"\\?mathbf\{",
    r"\\?alpha|\\?beta|\\?gamma",
    # 概率论
    r"\bE\s*\[", r"\bVar\s*\(",
    r"\bP\s*\(\s*[A-Z]",   # P(X), P(A)
    # Chinese math terms — university level
    r"\b矩阵\b|\b行列式\b|\b特征值\b|\b特征向量\b|\b向量空间\b|\b线性变换\b",
    r"\b导数\b|\b微分\b|\b积分\b|\b偏导\b|\b多元函数\b|\b全微分\b",
    r"\b极限\b|\b级数\b|\b收敛\b|\b泰勒\b|\b麦克劳林\b|\b洛必达\b|\b牛顿-莱布尼茨\b",
    r"\b概率密度\b|\b随机变量\b|\b期望\b|\b方差\b|\b协方差\b|\b正态分布\b|\b二项分布\b|\b泊松分布\b",
    r"\bdet\s*\(", r"\btrace\s*\(",
    r"\bn!\b|\b阶乘\b|\b排列\b|\b组合\b",
]


def is_likely_out_of_scope(text: str) -> bool:
    """启发式判断：这一步是否属于大学内容（微积分/线代/概率论），超出
    K1-12 本体覆盖范围？

    命中任一 ``_OUT_OF_SCOPE_PATTERNS`` 即视为超出。KG 后处理会用这个
    结果决定是否跳过规则层、让 LLM 自由打标签。
    """
    if not text:
        return False
    for p in _OUT_OF_SCOPE_PATTERNS:
        try:
            if re.search(p, text):
                return True
        except re.error:
            continue
    return False


# === Step → KP mapping (rule layer) ===

def map_step_to_kps(
    step_text: str,
    reason: str,
    score: float | None = None,
    max_score: float | None = None,
) -> list[tuple[str, float]]:
    """规则层：把一步解题文本映射到 KG 节点（多个候选），返回
    ``(node_id, confidence)`` 列表，按置信度从高到低排序。

    打分规则：
    - **关键词命中**：节点的 ``keyword_patterns`` 在 ``step_text + reason``
      里命中，置信度 0.6；
    - **错因命中**：若该步失分且 ``infer_error_type(reason)`` 在节点的
      ``error_type_hints`` 里，置信度 0.3；
    - 取两者最大值。

    没有任何命中返回空列表，由 LLM 层接管。
    """
    g = load_graph()
    haystack = f"{step_text} {reason}"
    is_error = (
        score is not None
        and max_score is not None
        and float(score) < float(max_score)
    )
    inferred_type = infer_error_type(reason) if is_error else None

    candidates: dict[str, float] = {}
    for nid, attrs in g.nodes(data=True):
        conf = 0.0
        # 关键词命中：0.6 分。
        for pat in attrs.get("keyword_patterns") or []:
            try:
                if re.search(pat, haystack):
                    conf = max(conf, 0.6)
                    break
            except re.error:
                continue
        # 错因命中：0.3 分（作为辅助信号）。
        if inferred_type and inferred_type in (attrs.get("error_type_hints") or []):
            conf = max(conf, 0.3)
        if conf > 0:
            candidates[nid] = conf

    if not candidates:
        return []
    return sorted(candidates.items(), key=lambda x: (-x[1], x[0]))


# === Report enrichment ===

def _call_kg_llm(prompt: dict[str, Any]) -> str:
    """对 LLM 做一次 ``/chat/completions`` 调用，返回助手文本。

    专用于 KG 标注场景：system 角色固定为「数学知识点标注员」，temperature=0，
    response_format 强制 json_object。失败抛 RuntimeError，调用方需 try/except。
    """
    if not settings.llm_score_api_key:
        raise RuntimeError("LLM_SCORE_API_KEY not configured")
    headers = {
        "Authorization": f"Bearer {settings.llm_score_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": settings.llm_score_model,
        "messages": [
            {"role": "system", "content": "你是数学知识点标注员。只返回 JSON，不要解释。"},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": 2500,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    timeout = (settings.llm_score_connect_timeout_sec, 25)
    resp = requests.post(
        f"{settings.llm_score_base_url.rstrip('/')}/chat/completions",
        headers=headers,
        json=body,
        timeout=timeout,
    )
    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("empty choices")
    msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = [str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type") == "text"]
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    """容错 JSON 解析（KG 专用）：直接解析失败时用正则兜底抽最外层花括号。"""
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


def refine_mappings_with_llm(
    ambiguous_steps: list[dict],
) -> dict[tuple[Any, Any], dict[str, list]]:
    """LLM 精修：对规则层没命中的「模糊步骤」批量做知识点标注。

    输入：``[{qno, index, text, reason, score, max_score}, ...]``。

    输出：``{(qno, index): {"kps": [(node_id, confidence)], "tags": [str]}}``。
    - ``kps``：**严格匹配**，``node_id`` 必须在当前 KG 中存在（否则丢弃）；
    - ``tags``：自由文本标签，给大学内容（极限/导数/矩阵）兜底用——这些
      本来就不在 K1-12 本体里。

    任何失败（API 错、JSON 解析错、KG 加载错）都返回空 dict，调用方静默
    回退到规则层结果。
    """
    if not settings.kg_llm_enabled or not settings.llm_score_api_key:
        return {}
    if not ambiguous_steps:
        return {}
    try:
        g = load_graph()
    except Exception:
        return {}

    # 构造给 LLM 的「本体目录」：每个节点附 name/stage/grade/description(≤80 字)。
    node_catalog = [
        {
            "id": nid,
            "name": attrs.get("name", ""),
            "stage": attrs.get("stage", ""),
            "grade": attrs.get("grade", 0),
            "description": (attrs.get("description") or "")[:80],
        }
        for nid, attrs in g.nodes(data=True)
    ]

    prompt = {
        "task": (
            "下面是学生数学解题的若干步骤，每步已经过规则匹配但未命中 K1-12 知识点。"
            "为每步同时返回：(a) 本体严格匹配的 kps（若存在）；"
            "(b) 自由命名的 free_text_tags（当涉及本体未覆盖的内容，如大学微积分/线性代数/概率论等）。"
            "只返回 JSON。"
        ),
        "ontology_scope": "K1-12（小学/初中/高中），不含大学内容。",
        "ontology_nodes": node_catalog,
        "steps": [
            {
                "qno": s.get("qno"),
                "index": s.get("index"),
                "text": (s.get("text") or "")[:300],
                "reason": (s.get("reason") or "")[:120],
            }
            for s in ambiguous_steps
        ],
        "rules": [
            "kps：严格从 ontology_nodes 选 0-3 个；id 必须出自列表；confidence 取 0.0-1.0；不得编造 id",
            "free_text_tags：当该步涉及本体未覆盖的知识点（如极限、导数、矩阵、随机变量）时，用 1-3 个中文短语命名",
            "kps 与 free_text_tags 可同时为空（题目陈述/无解题动作）；也可同时有值（本体命中 + 补充说明）",
            "reason 提示计算/符号/变量错误时，优先在 kps 中映射到对应 error_type_hints 的节点",
            "若该步完全属于大学内容（如 ∫、d/dx、lim），kps 应为空，只填 free_text_tags",
        ],
        "output_format": {
            "mappings": [
                {
                    "qno": 1, "index": 1,
                    "kps": [{"id": "M03", "confidence": 0.85}],
                    "free_text_tags": [],
                }
            ]
        },
    }

    try:
        raw = _call_kg_llm(prompt)
    except Exception:
        return {}
    parsed = _safe_json_parse(raw)
    if not parsed:
        return {}

    result: dict[tuple[Any, Any], dict[str, list]] = {}
    for m in parsed.get("mappings") or []:
        try:
            key = (m.get("qno"), int(m["index"]))
            kps: list[tuple[str, float]] = []
            for kp in m.get("kps") or []:
                nid = kp.get("id")
                if not nid or nid not in g.nodes:
                    continue
                try:
                    conf = max(0.0, min(1.0, float(kp.get("confidence", 0.5))))
                except (TypeError, ValueError):
                    conf = 0.5
                kps.append((str(nid), conf))
            kps = sorted(kps, key=lambda x: -x[1])[:3]
            tags_raw = m.get("free_text_tags") or []
            tags = [str(t).strip()[:24] for t in tags_raw if isinstance(t, (str, int, float)) and str(t).strip()]
            tags = tags[:3]
            if kps or tags:
                result[key] = {"kps": kps, "tags": tags}
        except (KeyError, TypeError, ValueError):
            continue
    return result


def enrich_report_with_kg(questions: list[dict] | None) -> dict:
    """给一份批改结果做「知识点归因」后处理。

    流程：
    1. **规则层**：对每一步调 :func:`map_step_to_kps`，按关键词 + 错因
       命中打分；
    2. **收集模糊步骤**：规则层没命中的步骤入列 ``ambiguous_steps``；
    3. **LLM 批量精修**：对模糊步骤做一次批量 LLM 调用
       (:func:`refine_mappings_with_llm`)，严格匹配本体 + 允许自由标签；
    4. **合并**：把 LLM 结果回填到 step_mappings；统计 touched/error 节点。

    整个函数对异常完全容忍（返回 ``{"ok": False, "error": ...}`` 或在
    最外层 try 包裹下保证不抛），因为它是批改流水线的最后一步，不该让
    KG 故障把整个评分带挂。
    """
    try:
        g = load_graph()
    except Exception as e:
        return {"ok": False, "error": f"ontology load failed: {e}"}

    step_mappings: list[dict] = []
    touched_nodes: set[str] = set()
    error_nodes: set[str] = set()
    unmapped_topics: dict[str, int] = {}
    ambiguous_steps: list[dict] = []
    mapping_keys: list[tuple[Any, Any]] = []

    for q in questions or []:
        if not isinstance(q, dict):
            continue
        qno = q.get("qno")
        steps = q.get("steps") or []
        step_scores = q.get("step_scores") or []
        q_max = float(q.get("max_score") or 0)
        per_step_max = q_max / max(len(steps), 1) if q_max else None

        score_lookup: dict[int, dict] = {}
        for ss in step_scores:
            if isinstance(ss, dict):
                idx = ss.get("index")
                if idx is not None:
                    score_lookup[int(idx)] = ss

        for step in steps:
            if not isinstance(step, dict):
                continue
            idx = step.get("index")
            ss = score_lookup.get(int(idx)) if idx is not None else None
            score = ss.get("score") if ss else None
            reason = str(ss.get("reason") or "") if ss else ""
            max_score = per_step_max
            text = str(step.get("text") or step.get("raw") or step.get("normalized") or "")
            # 超纲（大学内容）→ 跳过规则层，留给 LLM 打自由标签。
            out_of_scope = is_likely_out_of_scope(text)
            if out_of_scope:
                kps = []
            else:
                kps = map_step_to_kps(text, reason, score, max_score)
            top_kps = kps[:3]  # 每步最多 3 个知识点
            for nid, _conf in top_kps:
                touched_nodes.add(nid)
                # 若该步失分，把对应节点标为「错过的节点」。
                if score is not None and max_score and float(score) < float(max_score):
                    error_nodes.add(nid)
            step_mappings.append({
                "qno": qno,
                "index": idx,
                "score": score,
                "max_score": max_score,
                "kps": [{"id": nid, "confidence": round(c, 2)} for nid, c in top_kps],
                "tags": [],
                "source": "rule" if top_kps else ("out_of_scope" if out_of_scope else "none"),
            })
            mapping_keys.append((qno, idx))
            # 规则层没命中的非空步骤 → 收集进 ambiguous，稍后给 LLM。
            if not top_kps and text.strip():
                ambiguous_steps.append({
                    "qno": qno, "index": idx, "text": text,
                    "reason": reason, "score": score, "max_score": max_score,
                })

    # LLM 批量精修模糊步骤（单次调用）。
    llm_used = False
    llm_error = ""
    if ambiguous_steps:
        try:
            t0 = time.perf_counter()
            llm_map = refine_mappings_with_llm(ambiguous_steps)
            llm_used = bool(llm_map)
            for entry in step_mappings:
                if entry.get("source") not in ("none", "out_of_scope"):
                    continue
                key = (entry.get("qno"), entry.get("index"))
                if key not in llm_map:
                    continue
                payload = llm_map[key]
                kps = payload.get("kps") or []
                tags = payload.get("tags") or []
                entry["kps"] = [{"id": nid, "confidence": round(c, 2)} for nid, c in kps]
                entry["tags"] = tags
                if kps:
                    entry["source"] = "llm"
                    for nid, _conf in kps:
                        touched_nodes.add(nid)
                        if entry.get("score") is not None and entry.get("max_score") and float(entry["score"]) < float(entry["max_score"]):
                            error_nodes.add(nid)
                elif tags:
                    entry["source"] = "tags"
                    for tag in tags:
                        unmapped_topics[tag] = unmapped_topics.get(tag, 0) + 1
            llm_elapsed_ms = int((time.perf_counter() - t0) * 1000)
        except Exception as e:
            llm_error = f"{type(e).__name__}: {e}"
            llm_elapsed_ms = 0
    else:
        llm_elapsed_ms = 0

    sub_edges: list[dict] = []
    for nid in touched_nodes:
        for pred in g.predecessors(nid):
            if pred in touched_nodes:
                sub_edges.append({"from": pred, "to": nid})

    return {
        "ok": True,
        "step_mappings": step_mappings,
        "touched_nodes": sorted(touched_nodes),
        "error_nodes": sorted(error_nodes),
        "unmapped_topics": [
            {"name": name, "count": cnt}
            for name, cnt in sorted(unmapped_topics.items(), key=lambda x: (-x[1], x[0]))
        ],
        "all_nodes": [_node_payload(g, n) for n in sorted(g.nodes)],
        "all_edges": [{"from": u, "to": v} for u, v in g.edges],
        "sub_edges": sub_edges,
        "n_questions": len(questions or []),
        "n_steps_mapped": sum(1 for m in step_mappings if m["kps"] or m.get("tags")),
        "llm_used": llm_used,
        "llm_error": llm_error,
        "llm_elapsed_ms": llm_elapsed_ms,
        "llm_ambiguous": len(ambiguous_steps),
    }


# === User mastery ===

def compute_user_mastery(user_id: int, limit: int = 50) -> list[dict]:
    """扫用户最近 ``limit`` 次批改记录里的 kg_report，按知识点聚合出
    每个节点的掌握度。

    输出每条：``{id, name, stage, attempts, accuracy, state}``，其中
    ``state`` 按准确率 + 练习次数分四档：
    - ``gray``：0 次尝试（未练习）；
    - ``green``：准确率 ≥0.85 且次数 ≥2；
    - ``yellow``：准确率 ≥0.5；
    - ``red``：准确率 <0.5。

    前端 KG 面板用这组数据给节点着色（红/黄/绿/灰）。
    """
    from app.services.db import _connect

    stats: dict[str, dict] = {}
    try:
        with _connect() as conn:
            rows = conn.execute(
                """
                SELECT grading_result_json
                FROM grading_records
                WHERE user_id=? AND grading_result_json IS NOT NULL
                ORDER BY id DESC LIMIT ?
                """,
                (user_id, max(1, min(limit, 200))),
            ).fetchall()
        for row in rows:
            try:
                payload = json.loads(row["grading_result_json"] or "{}")
            except Exception:
                continue
            kg_report = payload.get("kg_report")
            if not isinstance(kg_report, dict):
                continue
            for sm in kg_report.get("step_mappings") or []:
                score = sm.get("score")
                max_score = sm.get("max_score")
                if score is None or not max_score:
                    continue
                for kp in sm.get("kps") or []:
                    nid = kp.get("id")
                    if not nid:
                        continue
                    s = stats.setdefault(
                        str(nid),
                        {"sum_score": 0.0, "sum_max": 0.0, "attempts": 0},
                    )
                    s["sum_score"] += float(score)
                    s["sum_max"] += float(max_score)
                    s["attempts"] += 1
    except Exception:
        pass

    g = load_graph()
    result: list[dict] = []
    for nid in sorted(g.nodes):
        attrs = dict(g.nodes[nid])
        s = stats.get(nid)
        if not s or s["attempts"] == 0:
            state = "gray"
            accuracy: float | None = None
            attempts = 0
        else:
            accuracy = s["sum_score"] / max(s["sum_max"], 1.0)
            attempts = s["attempts"]
            if accuracy >= 0.85 and attempts >= 2:
                state = "green"
            elif accuracy >= 0.5:
                state = "yellow"
            else:
                state = "red"
        result.append({
            "id": nid,
            "name": attrs.get("name", ""),
            "stage": attrs.get("stage", ""),
            "grade": attrs.get("grade", 0),
            "attempts": attempts,
            "accuracy": round(accuracy, 3) if accuracy is not None else None,
            "state": state,
            "prerequisites": list(g.predecessors(nid)),
        })
    return result
