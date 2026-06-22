"""SQLite 持久化层，所有 SQL 都集中在这里。

启动时 :func:`init_db` 创建表 + 跑幂等迁移（``ALTER TABLE ADD COLUMN``
会被 ``PRAGMA table_info`` 检查跳过）。业务层（main.py）只看到 Python
函数，例如 :func:`save_grading_record` / :func:`list_wrong_answers`，
不直接写 SQL。

数据模型概览：
- ``users`` —— id / username / password_hash / role(student|teacher|admin)
- ``ocr_records`` / ``grading_records`` —— 读取和批改历史
- ``grading_records.detail_report_md`` —— 精确报告（Markdown），异步生成
- ``wrong_answers`` —— 错题本，:func:`populate_wrong_answers_from_session`
  在每次批改后自动把失分题入库；``steps_json`` 存逐步的结构化数据。
- ``favorite_assignments`` —— 收藏的作业
- ``classes`` / ``user_classes`` —— 班级 + 多对多关联（学生可加多个班）
- ``class_kg_overrides`` —— 教师为本班定制的 KG 覆盖层
- ``llm_failures`` —— LLM 调用失败日志（仅教师/管理员可见）
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings


def _connect() -> sqlite3.Connection:
    """打开一个 SQLite 连接。

    - 自动创建父目录（首次运行时 ``outputs/`` 还不存在）；
    - ``row_factory = sqlite3.Row``：让结果行可以按列名取值
      （``row["username"]`` 而非 ``row[1]``），可读性更高。
    """
    db_file = Path(settings.db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """建表 + 幂等迁移。应用启动时 :func:`app.main` 会调一次。

    所有 ``CREATE TABLE`` 都带 ``IF NOT EXISTS``，老库不会被破坏；
    新增字段走 ``PRAGMA table_info(...)`` 检测后再 ``ALTER TABLE ADD COLUMN``，
    保证可以「老库直跑」。
    """
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'student',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS ocr_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              engine TEXT NOT NULL,
              ocr_text TEXT NOT NULL,
              steps_count INTEGER NOT NULL DEFAULT 0,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS grading_records (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              engine TEXT NOT NULL,
              total_score REAL NOT NULL,
              steps_count INTEGER NOT NULL DEFAULT 0,
              ocr_text TEXT NOT NULL,
              grading_meta_json TEXT NOT NULL DEFAULT '{}',
              grading_result_json TEXT,
              detail_report_md TEXT,
              detail_report_status TEXT NOT NULL DEFAULT 'pending',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS favorite_assignments (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              title TEXT NOT NULL,
              ocr_text TEXT NOT NULL,
              total_score REAL NOT NULL DEFAULT 0,
              feedback TEXT NOT NULL DEFAULT '',
              knowledge_tags_json TEXT NOT NULL DEFAULT '[]',
              report_json TEXT NOT NULL DEFAULT '{}',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS wrong_answers (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER NOT NULL,
              grading_record_id INTEGER,
              qno INTEGER NOT NULL,
              question_text TEXT NOT NULL DEFAULT '',
              step_summary TEXT NOT NULL DEFAULT '',
              score REAL NOT NULL,
              max_score REAL NOT NULL,
              kg_nodes_json TEXT NOT NULL DEFAULT '[]',
              error_type TEXT NOT NULL DEFAULT 'other',
              status TEXT NOT NULL DEFAULT 'new',
              note TEXT NOT NULL DEFAULT '',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              reviewed_at DATETIME,
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(grading_record_id) REFERENCES grading_records(id)
            );
            CREATE INDEX IF NOT EXISTS idx_wrong_answers_user_status
              ON wrong_answers(user_id, status);
            CREATE INDEX IF NOT EXISTS idx_wrong_answers_record
              ON wrong_answers(grading_record_id);

            CREATE TABLE IF NOT EXISTS llm_failures (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id INTEGER,
              endpoint TEXT NOT NULL,
              stage TEXT NOT NULL,
              error TEXT NOT NULL DEFAULT '',
              raw_preview TEXT NOT NULL DEFAULT '',
              attempts_json TEXT NOT NULL DEFAULT '[]',
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_llm_failures_created
              ON llm_failures(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_llm_failures_user
              ON llm_failures(user_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS classes (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              name TEXT NOT NULL,
              stage TEXT NOT NULL DEFAULT 'middle',
              grade INTEGER NOT NULL DEFAULT 0,
              description TEXT NOT NULL DEFAULT '',
              invite_code TEXT NOT NULL UNIQUE,
              creator_id INTEGER,
              created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              FOREIGN KEY(creator_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS user_classes (
              user_id INTEGER NOT NULL,
              class_id INTEGER NOT NULL,
              joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              PRIMARY KEY(user_id, class_id),
              FOREIGN KEY(user_id) REFERENCES users(id),
              FOREIGN KEY(class_id) REFERENCES classes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_user_classes_class
              ON user_classes(class_id);
            CREATE TABLE IF NOT EXISTS class_kg_overrides (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              class_id INTEGER NOT NULL,
              node_id TEXT NOT NULL,
              op TEXT NOT NULL DEFAULT 'upsert',
              payload_json TEXT NOT NULL DEFAULT '{}',
              updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(class_id, node_id),
              FOREIGN KEY(class_id) REFERENCES classes(id)
            );
            CREATE INDEX IF NOT EXISTS idx_class_kg_overrides_class
              ON class_kg_overrides(class_id);
            """
        )
        # 幂等迁移：老库建表时没有 detail_report 相关字段，统一在这里补齐。
        existing_cols = {row["name"] for row in conn.execute("PRAGMA table_info(grading_records)")}
        if "detail_report_md" not in existing_cols:
            conn.execute("ALTER TABLE grading_records ADD COLUMN detail_report_md TEXT")
        if "detail_report_status" not in existing_cols:
            conn.execute(
                "ALTER TABLE grading_records ADD COLUMN detail_report_status TEXT NOT NULL DEFAULT 'pending'"
            )
        if "grading_result_json" not in existing_cols:
            conn.execute("ALTER TABLE grading_records ADD COLUMN grading_result_json TEXT")
        # wrong_answers.steps_json：存逐步结构化数据 {text, reason, score, max_score, wrong}，
        # 让错题本能高亮「错在哪一步」。
        wa_cols = {row["name"] for row in conn.execute("PRAGMA table_info(wrong_answers)")}
        if "steps_json" not in wa_cols:
            conn.execute("ALTER TABLE wrong_answers ADD COLUMN steps_json TEXT NOT NULL DEFAULT '[]'")
        conn.commit()


def create_user(username: str, password_hash: str, role: str) -> int:
    """新建用户，返回新 id。调用前应做唯一性检查（见 main.register）。"""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users(username, password_hash, role) VALUES(?,?,?)",
            (username, password_hash, role),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    """按用户名查用户（登录路径用）。无则返回 None。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE username=?",
            (username,),
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict[str, Any] | None:
    """按 id 查用户（鉴权中间件用）。无则返回 None。"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash, role, created_at FROM users WHERE id=?",
            (user_id,),
        ).fetchone()
    return dict(row) if row else None


def save_ocr_record(user_id: int, engine: str, ocr_text: str, steps_count: int) -> int:
    """落库一条 OCR 记录，返回记录 id。"""
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO ocr_records(user_id, engine, ocr_text, steps_count) VALUES(?,?,?,?)",
            (user_id, engine, ocr_text, steps_count),
        )
        conn.commit()
        return int(cur.lastrowid)


def save_grading_record(
    user_id: int,
    engine: str,
    total_score: float,
    steps_count: int,
    ocr_text: str,
    grading_meta: dict[str, Any],
    grading_result: dict[str, Any] | None = None,
) -> int:
    """落库一条批改记录，返回记录 id。

    - ``grading_meta``：必填，存入 ``grading_meta_json``，包含 engine、
      llm_used、reflection_used 等审计字段；
    - ``grading_result``：可选，存入 ``grading_result_json``，含每题、
      每步的完整评分（供报告生成与详情回看使用）。
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO grading_records(
              user_id, engine, total_score, steps_count, ocr_text,
              grading_meta_json, grading_result_json
            )
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                user_id,
                engine,
                total_score,
                steps_count,
                ocr_text,
                json.dumps(grading_meta, ensure_ascii=False),
                json.dumps(grading_result, ensure_ascii=False) if grading_result else None,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_ocr_records(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    """列出某用户的 OCR 历史，按时间倒序。limit 钳制在 [1, 200]。"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, engine, steps_count, ocr_text, created_at
            FROM ocr_records
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
    return [dict(x) for x in rows]


def list_grading_records(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    """列出某用户的批改历史，按时间倒序。limit 钳制在 [1, 200]。"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, engine, total_score, steps_count, ocr_text, grading_meta_json, created_at
            FROM grading_records
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
    return [dict(x) for x in rows]


def get_grading_record(record_id: int, user_id: int | None = None) -> dict[str, Any] | None:
    """按 id 取单条批改记录。

    - ``user_id=None``：管理员视角，不限用户；
    - ``user_id=int``：普通用户视角，强制带上 ``user_id=?`` 限制，避免越权读取。
    """
    with _connect() as conn:
        if user_id is None:
            row = conn.execute(
                """
                SELECT id, user_id, engine, total_score, steps_count, ocr_text,
                       grading_meta_json, grading_result_json,
                       detail_report_md, detail_report_status, created_at
                FROM grading_records
                WHERE id=?
                """,
                (record_id,),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT id, user_id, engine, total_score, steps_count, ocr_text,
                       grading_meta_json, grading_result_json,
                       detail_report_md, detail_report_status, created_at
                FROM grading_records
                WHERE id=? AND user_id=?
                """,
                (record_id, user_id),
            ).fetchone()
    return dict(row) if row else None


def update_detail_report(record_id: int, markdown: str | None, status: str) -> bool:
    """更新 detail_report 字段。``status`` 取值：pending / generating /
    ready / failed。返回是否真的更新到了（False 表示 record_id 不存在）。
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            UPDATE grading_records
            SET detail_report_md=?, detail_report_status=?
            WHERE id=?
            """,
            (markdown, status, record_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_user_dashboard(user_id: int) -> dict[str, Any]:
    """用户首页 dashboard：OCR 数、批改数、平均分、最近一次批改、收藏数。

    所有聚合都在单次连接内完成，避免多次开连接的开销。
    """
    with _connect() as conn:
        ocr_count = conn.execute(
            "SELECT COUNT(*) AS c FROM ocr_records WHERE user_id=?",
            (user_id,),
        ).fetchone()["c"]
        grading_stats = conn.execute(
            """
            SELECT COUNT(*) AS c, AVG(total_score) AS avg_score, MAX(created_at) AS last_time
            FROM grading_records
            WHERE user_id=?
            """,
            (user_id,),
        ).fetchone()
        latest = conn.execute(
            """
            SELECT id, engine, total_score, steps_count, created_at
            FROM grading_records
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()
        favorite_count = conn.execute(
            "SELECT COUNT(*) AS c FROM favorite_assignments WHERE user_id=?",
            (user_id,),
        ).fetchone()["c"]
    return {
        "ocr_count": int(ocr_count or 0),
        "grading_count": int(grading_stats["c"] or 0),
        "favorite_count": int(favorite_count or 0),
        "average_score": round(float(grading_stats["avg_score"] or 0), 2),
        "last_grading_time": grading_stats["last_time"],
        "latest_grading": dict(latest) if latest else None,
    }


def save_favorite_assignment(
    user_id: int,
    title: str,
    ocr_text: str,
    total_score: float,
    feedback: str,
    knowledge_tags: list[str],
    report: dict[str, Any],
) -> int:
    """收藏一条作业（包含完整 report，便于以后复习时直接打开）。"""
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO favorite_assignments(
              user_id, title, ocr_text, total_score, feedback, knowledge_tags_json, report_json
            )
            VALUES(?,?,?,?,?,?,?)
            """,
            (
                user_id,
                title,
                ocr_text,
                total_score,
                feedback,
                json.dumps(knowledge_tags, ensure_ascii=False),
                json.dumps(report, ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_favorite_assignments(user_id: int, limit: int = 50) -> list[dict[str, Any]]:
    """列出用户收藏的作业。JSON 字段（knowledge_tags、report）会被解析成 dict
    方便前端直接使用，避免前端再 parse 一遍。"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, title, ocr_text, total_score, feedback, knowledge_tags_json, report_json, created_at
            FROM favorite_assignments
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, max(1, min(limit, 200))),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["knowledge_tags"] = json.loads(str(item.pop("knowledge_tags_json") or "[]"))
        except json.JSONDecodeError:
            item["knowledge_tags"] = []
        try:
            item["report"] = json.loads(str(item.pop("report_json") or "{}"))
        except json.JSONDecodeError:
            item["report"] = {}
        items.append(item)
    return items


def delete_favorite_assignment(user_id: int, favorite_id: int) -> bool:
    """取消收藏。带 user_id 限制防越权删除别人的收藏。"""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM favorite_assignments WHERE id=? AND user_id=?",
            (favorite_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_admin_overview() -> dict[str, Any]:
    """管理员 dashboard：用户分角色统计、总 OCR / 批改数、全局平均分、
    最近 8 条批改。"""
    with _connect() as conn:
        users = conn.execute(
            "SELECT role, COUNT(*) AS count FROM users GROUP BY role ORDER BY role"
        ).fetchall()
        totals = conn.execute(
            """
            SELECT
              (SELECT COUNT(*) FROM ocr_records) AS ocr_count,
              (SELECT COUNT(*) FROM grading_records) AS grading_count,
              (SELECT AVG(total_score) FROM grading_records) AS average_score
            """
        ).fetchone()
        recent_gradings = conn.execute(
            """
            SELECT g.id, u.username, u.role, g.engine, g.total_score, g.steps_count, g.created_at
            FROM grading_records g
            JOIN users u ON u.id = g.user_id
            ORDER BY g.id DESC
            LIMIT 8
            """
        ).fetchall()
    return {
        "users_by_role": [dict(row) for row in users],
        "ocr_count": int(totals["ocr_count"] or 0),
        "grading_count": int(totals["grading_count"] or 0),
        "average_score": round(float(totals["average_score"] or 0), 2),
        "recent_gradings": [dict(row) for row in recent_gradings],
    }


# ============================================================
# Admin: user management
# ============================================================

def list_all_users(limit: int = 200) -> list[dict[str, Any]]:
    """管理员：列出全部用户，附带每人的批改数与错题数（子查询内联）。"""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.created_at,
                   (SELECT COUNT(*) FROM grading_records WHERE user_id=u.id) AS grading_count,
                   (SELECT COUNT(*) FROM wrong_answers WHERE user_id=u.id) AS wrong_count
            FROM users u
            ORDER BY u.id ASC
            LIMIT ?
            """,
            (max(1, min(limit, 1000)),),
        ).fetchall()
    return [dict(r) for r in rows]


def update_user_role(user_id: int, role: str) -> bool:
    """管理员：调整用户角色。

    出于安全考虑，本函数**不允许**直接把人改成 ``admin``——
    admin 角色只能在注册时 bootstrap 出一位，之后只能由现任管理员在
    数据库里手动操作。这里只接受 ``student`` / ``teacher``。
    """
    if role not in {"student", "teacher"}:
        return False
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE users SET role=? WHERE id=?",
            (role, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def delete_user(user_id: int) -> bool:
    """删除用户并级联清理所有依赖记录。

    顺序很关键：先确认用户存在 → 删 wrong_answers、favorites、grading、
    ocr → 最后删 user。任何一步遗漏都会因为外键约束失败。
    """
    with _connect() as conn:
        cur = conn.execute("SELECT id FROM users WHERE id=?", (user_id,))
        if not cur.fetchone():
            return False
        conn.execute("DELETE FROM wrong_answers WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM favorite_assignments WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM grading_records WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM ocr_records WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM users WHERE id=?", (user_id,))
        conn.commit()
    return True


# ============================================================
# Wrong-answer book
# ============================================================

def populate_wrong_answers_from_session(
    user_id: int,
    grading_record_id: int,
    questions: list[dict[str, Any]],
    kg_report: dict[str, Any] | None,
) -> int:
    """每次批改后，把失分题（score < max_score）落库到 wrong_answers。

    工作流：
    1. 从 ``kg_report.step_mappings`` 建立 ``qno → [kg_node_id]`` 映射，
       用于把错题关联到知识图谱节点；
    2. 逐题构造逐步结构化数据（``steps_json``），标记错在哪一步
       （``wrong=True``）；用 :func:`knowledge_graph.infer_error_type`
       从 reason 文本里推断错因（计算/符号/变量/逻辑）；
    3. 满分的题直接跳过，不进错题本；
    4. 批量 ``executemany`` 插入。

    返回插入的行数。
    """
    # 第 1 步：构造 qno → KG 节点 id 列表的映射。
    qno_to_kg: dict[int, list[str]] = {}
    if kg_report and isinstance(kg_report, dict):
        for sm in kg_report.get("step_mappings") or []:
            try:
                qno = int(sm.get("qno") or 0)
            except (TypeError, ValueError):
                continue
            for kp in sm.get("kps") or []:
                nid = kp.get("id") if isinstance(kp, dict) else None
                if nid and nid not in qno_to_kg.setdefault(qno, []):
                    qno_to_kg[qno].append(str(nid))

    rows_to_insert: list[tuple[Any, ...]] = []
    for q in questions:
        if not isinstance(q, dict):
            continue
        try:
            qno = int(q.get("qno") or 0)
            score = float(q.get("score") or 0)
            max_score = float(q.get("max_score") or 0)
        except (TypeError, ValueError):
            continue
        # 满分或异常分值：跳过。
        if max_score <= 0 or score >= max_score - 0.01:
            continue
        question_text = str(q.get("question_text") or "")[:1000]
        # 第 2 步：构造逐步结构化记录。step_scores[].index 与 steps[].index 对齐。
        steps_in = [s for s in (q.get("steps") or []) if isinstance(s, dict)]
        scores_in = [s for s in (q.get("step_scores") or []) if isinstance(s, dict)]
        score_by_idx: dict[int, dict[str, Any]] = {}
        for ss in scores_in:
            try:
                score_by_idx[int(ss.get("index"))] = ss
            except (TypeError, ValueError):
                continue
        per_step_max = (max_score / len(steps_in)) if steps_in else 0.0
        step_records: list[dict[str, Any]] = []
        step_parts: list[str] = []
        error_type = "other"
        from app.services.knowledge_graph import infer_error_type
        for step in steps_in:
            text = str(step.get("text") or step.get("normalized") or "").strip()
            if not text:
                continue
            try:
                idx = int(step.get("index") or 0)
            except (TypeError, ValueError):
                idx = 0
            ss = score_by_idx.get(idx, {})
            reason = str(ss.get("reason") or "").strip()
            step_score = float(ss.get("score") if ss.get("score") is not None else per_step_max)
            # 标记该步是否失分：分数低于每步满分 - 0.01 即视为失分（错步）。
            wrong = step_score < per_step_max - 0.01
            step_records.append({
                "text": text,
                "reason": reason,
                "score": round(step_score, 2),
                "max_score": round(per_step_max, 2),
                "wrong": wrong,
            })
            step_parts.append(text)
            if reason:
                step_parts.append(f"[{reason}]")
                # 失分步：尝试从 reason 文本里推断更细的错因。
                if wrong:
                    inferred = infer_error_type(reason)
                    if inferred != "other":
                        error_type = inferred
        step_summary = "\n".join(step_parts)[:2000]
        kg_nodes = qno_to_kg.get(qno, [])
        rows_to_insert.append((
            user_id,
            grading_record_id,
            qno,
            question_text,
            step_summary,
            score,
            max_score,
            json.dumps(kg_nodes, ensure_ascii=False),
            error_type,
            json.dumps(step_records, ensure_ascii=False)[:8000],
        ))

    if not rows_to_insert:
        return 0
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO wrong_answers(
              user_id, grading_record_id, qno, question_text, step_summary,
              score, max_score, kg_nodes_json, error_type, steps_json
            )
            VALUES(?,?,?,?,?,?,?,?,?,?)
            """,
            rows_to_insert,
        )
        conn.commit()
    return len(rows_to_insert)


def list_wrong_answers(
    user_id: int,
    status: str | None = None,
    kg_node: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """列出用户的错题本，支持按 ``status`` / ``kg_node`` 过滤。

    - ``status``：new / reviewing / mastered；
    - ``kg_node``：按 KG 节点 id 过滤，用 ``LIKE`` 模糊匹配 JSON 数组
      （SQLite 没有 JSON 查询操作符，这里用字符串包含做近似）；
    - KG 节点 id 会通过 ``get_all_nodes()`` 解析成 ``{id, name}``，
      前端直接显示节点名称（如 M10 → 「一元二次方程」）。
    """
    sql = """
        SELECT id, grading_record_id, qno, question_text, step_summary,
               score, max_score, kg_nodes_json, error_type, status, note,
               created_at, reviewed_at, steps_json
        FROM wrong_answers
        WHERE user_id=?
    """
    params: list[Any] = [user_id]
    if status:
        sql += " AND status=?"
        params.append(status)
    if kg_node:
        # JSON 数组的字符串包含查询：kg_nodes_json 里包含 "node_id" 即命中。
        sql += " AND kg_nodes_json LIKE ?"
        params.append(f'%"{kg_node}"%')
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(limit, 500)))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    # 构造 id→name 映射，让前端可以直接渲染中文名（M10 → "一元二次方程"）。
    kg_names: dict[str, str] = {}
    try:
        from app.services.knowledge_graph import get_all_nodes
        for node in get_all_nodes():
            kg_names[str(node.get("id"))] = str(node.get("name") or node.get("id"))
    except Exception:
        pass
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            node_ids = json.loads(str(item.pop("kg_nodes_json") or "[]"))
        except json.JSONDecodeError:
            node_ids = []
        item["kg_nodes"] = [
            {"id": str(nid), "name": kg_names.get(str(nid), str(nid))}
            for nid in node_ids
        ]
        try:
            item["steps"] = json.loads(str(item.pop("steps_json") or "[]"))
        except json.JSONDecodeError:
            item["steps"] = []
        items.append(item)
    return items


def update_wrong_answer(
    user_id: int,
    wrong_id: int,
    status: str | None = None,
    note: str | None = None,
) -> dict[str, Any] | None:
    """更新错题的 status 或 note。

    - status 取值必须是 ``new`` / ``reviewing`` / ``mastered``，否则视为
      非法，返回 None；
    - 标记 ``mastered`` 时自动写 ``reviewed_at=CURRENT_TIMESTAMP``；
    - note 截断到 2000 字符；
    - 两个字段都为 None 时直接返回当前快照，不写库。
    """
    fields: list[str] = []
    params: list[Any] = []
    if status is not None:
        if status not in {"new", "reviewing", "mastered"}:
            return None
        fields.append("status=?")
        params.append(status)
        if status == "mastered":
            fields.append("reviewed_at=CURRENT_TIMESTAMP")
    if note is not None:
        fields.append("note=?")
        params.append(note[:2000])
    if not fields:
        return get_wrong_answer(user_id, wrong_id)
    params.extend([wrong_id, user_id])
    with _connect() as conn:
        cur = conn.execute(
            f"UPDATE wrong_answers SET {', '.join(fields)} WHERE id=? AND user_id=?",
            params,
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get_wrong_answer(user_id, wrong_id)


def get_wrong_answer(user_id: int, wrong_id: int) -> dict[str, Any] | None:
    """按 id 取单条错题。KG 节点和 steps 都会被解析成结构化字段。"""
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT id, grading_record_id, qno, question_text, step_summary,
                   score, max_score, kg_nodes_json, error_type, status, note,
                   created_at, reviewed_at, steps_json
            FROM wrong_answers
            WHERE id=? AND user_id=?
            """,
            (wrong_id, user_id),
        ).fetchone()
    if not row:
        return None
    item = dict(row)
    try:
        node_ids = json.loads(str(item.pop("kg_nodes_json") or "[]"))
    except json.JSONDecodeError:
        node_ids = []
    kg_names: dict[str, str] = {}
    try:
        from app.services.knowledge_graph import get_all_nodes
        for node in get_all_nodes():
            kg_names[str(node.get("id"))] = str(node.get("name") or node.get("id"))
    except Exception:
        pass
    item["kg_nodes"] = [
        {"id": str(nid), "name": kg_names.get(str(nid), str(nid))}
        for nid in node_ids
    ]
    try:
        item["steps"] = json.loads(str(item.pop("steps_json") or "[]"))
    except json.JSONDecodeError:
        item["steps"] = []
    return item


def delete_wrong_answer(user_id: int, wrong_id: int) -> bool:
    """删一条错题（用户主动移除或 mastered 后清理）。带 user_id 防越权。"""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM wrong_answers WHERE id=? AND user_id=?",
            (wrong_id, user_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_wrong_answer_stats(user_id: int) -> dict[str, Any]:
    """错题本的统计面板数据：总数、按 status 分组、按错因分组、Top-10
    高频 KG 节点。前端错题本首页用这份统计数据画饼图/柱图。"""
    with _connect() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS c FROM wrong_answers WHERE user_id=?",
            (user_id,),
        ).fetchone()["c"]
        by_status = conn.execute(
            """
            SELECT status, COUNT(*) AS c
            FROM wrong_answers WHERE user_id=?
            GROUP BY status
            """,
            (user_id,),
        ).fetchall()
        by_error = conn.execute(
            """
            SELECT error_type, COUNT(*) AS c
            FROM wrong_answers WHERE user_id=?
            GROUP BY error_type ORDER BY c DESC
            """,
            (user_id,),
        ).fetchall()
        # Top KG nodes by frequency.
        rows = conn.execute(
            "SELECT kg_nodes_json FROM wrong_answers WHERE user_id=?",
            (user_id,),
        ).fetchall()
    node_counts: dict[str, int] = {}
    for row in rows:
        try:
            nodes = json.loads(str(row["kg_nodes_json"] or "[]"))
        except json.JSONDecodeError:
            continue
        for nid in nodes:
            node_counts[str(nid)] = node_counts.get(str(nid), 0) + 1
    top_nodes = sorted(node_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
    kg_names: dict[str, str] = {}
    try:
        from app.services.knowledge_graph import get_all_nodes
        for node in get_all_nodes():
            kg_names[str(node.get("id"))] = str(node.get("name") or node.get("id"))
    except Exception:
        pass
    return {
        "total": int(total or 0),
        "by_status": {row["status"]: int(row["c"]) for row in by_status},
        "by_error_type": {row["error_type"]: int(row["c"]) for row in by_error},
        "top_kg_nodes": [
            {"id": nid, "name": kg_names.get(nid, nid), "count": cnt}
            for nid, cnt in top_nodes
        ],
    }


# ============================================================
# LLM failure log (admin/dev only — never surfaced to end users)
# ============================================================

def save_llm_failure(
    user_id: int | None,
    endpoint: str,
    stage: str,
    error: str,
    raw_preview: str = "",
    attempts: list | None = None,
) -> int:
    """落库一条 LLM 调用失败记录，仅给管理员/开发事后排查用。

    ``stage`` 是分类标签，目前有：
    - ``ocr-trace`` / ``ocr-debug``：OCR 阶段的失败；
    - ``score-llm-fallback``：评分阶段降级到规则路径；
    - ``score-drop-fabricated``：评分阶段丢弃了 LLM 脑补的步骤。

    字段长度都做了截断，避免长文本撑爆数据库。
    """
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO llm_failures(user_id, endpoint, stage, error, raw_preview, attempts_json)
            VALUES(?,?,?,?,?,?)
            """,
            (
                user_id,
                endpoint[:64],
                stage[:32],
                (error or "")[:4000],
                (raw_preview or "")[:4000],
                json.dumps(attempts or [], ensure_ascii=False),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def list_llm_failures(
    limit: int = 50,
    user_id: int | None = None,
    endpoint: str | None = None,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """列出 LLM 失败日志，按时间倒序。可按 user_id / endpoint / stage 过滤。"""
    sql = """
        SELECT f.id, f.user_id, u.username, f.endpoint, f.stage,
               f.error, f.raw_preview, f.attempts_json, f.created_at
        FROM llm_failures f
        LEFT JOIN users u ON u.id = f.user_id
        WHERE 1=1
    """
    params: list[Any] = []
    if user_id is not None:
        sql += " AND f.user_id=?"
        params.append(user_id)
    if endpoint:
        sql += " AND f.endpoint=?"
        params.append(endpoint)
    if stage:
        sql += " AND f.stage=?"
        params.append(stage)
    sql += " ORDER BY f.id DESC LIMIT ?"
    params.append(max(1, min(limit, 500)))
    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["attempts"] = json.loads(str(item.pop("attempts_json") or "[]"))
        except json.JSONDecodeError:
            item["attempts"] = []
        items.append(item)
    return items


def delete_old_llm_failures(keep_days: int = 30) -> int:
    """清理 ``keep_days`` 天之前的 LLM 失败日志，返回删除条数。

    建议用 CRON 每日调一次，避免 llm_failures 表无限增长。
    """
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM llm_failures WHERE created_at < datetime('now', ?)",
            (f"-{int(max(1, keep_days))} days",),
        )
        conn.commit()
        return int(cur.rowcount or 0)
