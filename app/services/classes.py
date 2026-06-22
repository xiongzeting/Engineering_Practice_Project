"""班级管理服务。

班级是「学生 ↔ 教师」的桥：教师创建班级得到 6 位邀请码，学生用邀请码
加入班级（多对多，一个学生可以同时在多个班）。教师只能看到自己创建的
班级里的学生，管理员能看所有班级。

核心概念：
- **邀请码**：:func:`_new_invite_code` 生成无歧义字符（去掉 ``0/O/1/I``）
  的 6 位码，碰撞时重试。
- **角色作用域**：:func:`list_classes` 按调用者角色返回不同范围——
  ``admin`` 看全部、``teacher`` 只看自己 creator_id 的、``student`` 只看
  自己 join 过的。
- **校验帮手**：:func:`is_teacher_of_class` / :func:`list_class_member_ids`
  被 main.py 的权限层调用。
- **班级报表**：:func:`get_class_report` 聚合成员、批改次数、平均分、
  错因分布，是教师后台「班级 → 查看报表」的数据源。
"""
from __future__ import annotations

import secrets
import string
from typing import Any

from app.services import db


_CODE_ALPHABET = string.ascii_uppercase + string.digits  # 字母+数字（下面会剥离易混淆字符）
_CODE_LENGTH = 6  # 6 位邀请码：够用、易朗读


def _new_invite_code() -> str:
    """生成一个全新的 6 位邀请码。

    - 去掉易混淆字符 ``0/O/1/I``（口头传达时不会听错）；
    - 用 :func:`secrets.choice` 而非 :func:`random.choice`，避免可预测；
    - 最多重试 32 次（每次碰撞概率极低，32 次仍碰撞视为异常）。
    """
    # 过滤掉视觉上易混淆的字符。
    alphabet = "".join(c for c in _CODE_ALPHABET if c not in {"0", "O", "1", "I"})
    for _ in range(32):
        code = "".join(secrets.choice(alphabet) for _ in range(_CODE_LENGTH))
        if not _code_exists(code):
            return code
    raise RuntimeError("无法生成唯一邀请码，请重试。")


def _code_exists(code: str) -> bool:
    """检查邀请码是否已被占用。"""
    with db._connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM classes WHERE invite_code=? LIMIT 1", (code,)
        ).fetchone()
    return row is not None


def _normalize_stage(stage: str) -> str:
    """把学段标准化为 ``primary`` / ``middle`` / ``high`` 之一，非法值归 ``middle``。"""
    return stage if stage in {"primary", "middle", "high"} else "middle"


def create_class(
    name: str,
    creator_id: int,
    stage: str = "middle",
    grade: int = 0,
    description: str = "",
) -> dict[str, Any]:
    """创建班级。班级名为空会抛 ValueError；成功返回新建班级的完整 dict。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("班级名称不能为空。")
    code = _new_invite_code()
    with db._connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO classes(name, stage, grade, description, invite_code, creator_id)
            VALUES(?,?,?,?,?,?)
            """,
            (name, _normalize_stage(stage), int(grade or 0), description.strip(), code, creator_id),
        )
        conn.commit()
        class_id = int(cur.lastrowid)
    return get_class(class_id)  # type: ignore[return-value]


def update_class(
    class_id: int,
    *,
    name: str | None = None,
    stage: str | None = None,
    grade: int | None = None,
    description: str | None = None,
) -> dict[str, Any] | None:
    """按关键字参数局部更新班级字段。``None`` 的字段不更新。返回更新后的快照。

    名称给空字符串会抛 ValueError（与其他保持一致）。
    """
    fields: list[str] = []
    params: list[Any] = []
    if name is not None:
        name = name.strip()
        if not name:
            raise ValueError("班级名称不能为空。")
        fields.append("name=?")
        params.append(name)
    if stage is not None:
        fields.append("stage=?")
        params.append(_normalize_stage(stage))
    if grade is not None:
        fields.append("grade=?")
        params.append(int(grade))
    if description is not None:
        fields.append("description=?")
        params.append(description.strip())
    if fields:
        params.append(class_id)
        with db._connect() as conn:
            conn.execute(
                f"UPDATE classes SET {', '.join(fields)} WHERE id=?", params
            )
            conn.commit()
    return get_class(class_id)


def regenerate_invite_code(class_id: int) -> dict[str, Any] | None:
    """重置班级邀请码（旧码立即失效）。班级不存在时返回 None。

    用于邀请码泄露后的快速轮换。
    """
    code = _new_invite_code()
    with db._connect() as conn:
        cur = conn.execute(
            "UPDATE classes SET invite_code=? WHERE id=?", (code, class_id)
        )
        conn.commit()
        if cur.rowcount == 0:
            return None
    return get_class(class_id)


def delete_class(class_id: int) -> bool:
    """删除班级及其成员关系（user_classes）。

    注意：不删除关联的批改记录和错题——那些是学生的资产，留在学生名下。
    """
    with db._connect() as conn:
        cur = conn.execute("DELETE FROM classes WHERE id=?", (class_id,))
        conn.execute("DELETE FROM user_classes WHERE class_id=?", (class_id,))
        conn.commit()
        return cur.rowcount > 0


def get_class(class_id: int) -> dict[str, Any] | None:
    """按 id 取单个班级。"""
    with db._connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, stage, grade, description, invite_code,
                   creator_id, created_at
            FROM classes WHERE id=?
            """,
            (class_id,),
        ).fetchone()
    return dict(row) if row else None


def get_class_by_invite_code(code: str) -> dict[str, Any] | None:
    """按邀请码查班级。会做 toUpperCase + strip 归一化，前端传小写也能查到。"""
    code = (code or "").strip().upper()
    if not code:
        return None
    with db._connect() as conn:
        row = conn.execute(
            """
            SELECT id, name, stage, grade, description, invite_code,
                   creator_id, created_at
            FROM classes WHERE invite_code=?
            """,
            (code,),
        ).fetchone()
    return dict(row) if row else None


def list_classes(user_id: int | None = None, role: str = "student") -> list[dict[str, Any]]:
    """按调用者角色返回不同范围的班级列表。

    - ``admin``：全部班级；
    - ``teacher``：自己 creator_id 名下的班级；
    - ``student``：自己加入过的班级（通过 user_classes 关联）。

    每条记录都附带 ``member_count``（班级成员人数）。
    """
    with db._connect() as conn:
        if role == "admin":
            rows = conn.execute(
                """
                SELECT c.id, c.name, c.stage, c.grade, c.description,
                       c.invite_code, c.creator_id, c.created_at,
                       (SELECT COUNT(*) FROM user_classes WHERE class_id=c.id) AS member_count
                FROM classes c
                ORDER BY c.id DESC
                """
            ).fetchall()
        elif role == "teacher":
            rows = conn.execute(
                """
                SELECT c.id, c.name, c.stage, c.grade, c.description,
                       c.invite_code, c.creator_id, c.created_at,
                       (SELECT COUNT(*) FROM user_classes WHERE class_id=c.id) AS member_count
                FROM classes c
                WHERE c.creator_id=?
                ORDER BY c.id DESC
                """,
                (user_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT c.id, c.name, c.stage, c.grade, c.description,
                       c.invite_code, c.creator_id, c.created_at,
                       (SELECT COUNT(*) FROM user_classes WHERE class_id=c.id) AS member_count
                FROM classes c
                JOIN user_classes uc ON uc.class_id=c.id
                WHERE uc.user_id=?
                ORDER BY c.id DESC
                """,
                (user_id,),
            ).fetchall()
    return [dict(r) for r in rows]


def list_teacher_class_ids(teacher_id: int) -> list[int]:
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT id FROM classes WHERE creator_id=?", (teacher_id,)
        ).fetchall()
    return [int(r["id"]) for r in rows]


def is_teacher_of_class(teacher_id: int, class_id: int) -> bool:
    with db._connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM classes WHERE id=? AND creator_id=? LIMIT 1",
            (class_id, teacher_id),
        ).fetchone()
    return row is not None


def list_class_member_ids(class_id: int) -> list[int]:
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT user_id FROM user_classes WHERE class_id=?", (class_id,)
        ).fetchall()
    return [int(r["user_id"]) for r in rows]


def list_class_members(class_id: int) -> list[dict[str, Any]]:
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT u.id, u.username, u.role, u.created_at, uc.joined_at,
                   (SELECT COUNT(*) FROM grading_records g WHERE g.user_id=u.id) AS grading_count,
                   (SELECT COUNT(*) FROM wrong_answers w WHERE w.user_id=u.id) AS wrong_count,
                   (SELECT AVG(total_score) FROM grading_records g WHERE g.user_id=u.id) AS avg_score
            FROM users u
            JOIN user_classes uc ON uc.user_id=u.id
            WHERE uc.class_id=?
            ORDER BY uc.joined_at DESC
            """,
            (class_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_users_in_teacher_classes(teacher_id: int) -> list[dict[str, Any]]:
    """列出教师所教的所有班级中的学生（跨班聚合）。

    每条记录附带 ``class_names``（GROUP_CONCAT 拼接的班级名，便于前端
    显示「该生在初二(1)班、初二(2)班」）。
    """
    with db._connect() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT u.id, u.username, u.role, u.created_at,
                   GROUP_CONCAT(DISTINCT c.name) AS class_names,
                   (SELECT COUNT(*) FROM grading_records g WHERE g.user_id=u.id) AS grading_count,
                   (SELECT COUNT(*) FROM wrong_answers w WHERE w.user_id=u.id) AS wrong_count,
                   (SELECT AVG(total_score) FROM grading_records g WHERE g.user_id=u.id) AS avg_score
            FROM users u
            JOIN user_classes uc ON uc.user_id=u.id
            JOIN classes c ON c.id=uc.class_id AND c.creator_id=?
            WHERE u.role='student'
            GROUP BY u.id
            ORDER BY u.id DESC
            """,
            (teacher_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def list_user_class_ids(user_id: int) -> list[int]:
    """返回某用户加入的所有班级 id。学生主页「我的班级」模块用。"""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT class_id FROM user_classes WHERE user_id=?",
            (user_id,),
        ).fetchall()
    return [int(r["class_id"]) for r in rows]


def join_class(user_id: int, invite_code: str) -> dict[str, Any]:
    """学生用邀请码加入班级。

    失败情况：
    - 邀请码无效或对应班级不存在：抛 ValueError；
    - 已经加入过该班级：抛 ValueError（前端应直接进入该班）。

    成功返回班级 dict。
    """
    cls = get_class_by_invite_code(invite_code)
    if not cls:
        raise ValueError("邀请码无效或班级不存在。")
    with db._connect() as conn:
        # 防重复加入：先查后插。
        existing = conn.execute(
            "SELECT 1 FROM user_classes WHERE user_id=? AND class_id=?",
            (user_id, cls["id"]),
        ).fetchone()
        if existing:
            raise ValueError("你已经加入过这个班级。")
        conn.execute(
            "INSERT INTO user_classes(user_id, class_id) VALUES(?,?)",
            (user_id, cls["id"]),
        )
        conn.commit()
    return cls


def leave_class(user_id: int, class_id: int) -> bool:
    """学生退出班级。返回是否真的退了（False = 本来就不在班里）。"""
    with db._connect() as conn:
        cur = conn.execute(
            "DELETE FROM user_classes WHERE user_id=? AND class_id=?",
            (user_id, class_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_class_report(class_id: int) -> dict[str, Any]:
    """班级报表聚合：成员、平均分、最近 10 次批改、错题数、错因分布。

    所有聚合都在单次连接内完成。班级无成员时返回空报表（前端可正常渲染）。
    错因分布按 5 类（calculation/sign/variable/logic/other）分组，前端用
    这组数据画柱状图。
    """
    with db._connect() as conn:
        # 先查成员 id 列表，后续的统计用 IN (placeholders) 一次性查完。
        members = conn.execute(
            """
            SELECT u.id, u.username
            FROM users u JOIN user_classes uc ON uc.user_id=u.id
            WHERE uc.class_id=?
            """,
            (class_id,),
        ).fetchall()
        member_ids = [int(m["id"]) for m in members]
        if not member_ids:
            # 班级没成员：返回空报表，前端照样能显示。
            return {
                "class_id": class_id,
                "member_count": 0,
                "members": [],
                "grading_count": 0,
                "avg_score": None,
                "recent_gradings": [],
                "wrong_answer_count": 0,
                "error_type_distribution": {},
            }
        # 动态构造占位符：?,?,? 用于 IN 子句。
        placeholders = ",".join("?" * len(member_ids))
        grading_row = conn.execute(
            f"""
            SELECT COUNT(*) AS c, AVG(total_score) AS avg_score
            FROM grading_records WHERE user_id IN ({placeholders})
            """,
            member_ids,
        ).fetchone()
        recent = conn.execute(
            f"""
            SELECT g.id, g.user_id, u.username, g.total_score, g.created_at
            FROM grading_records g JOIN users u ON u.id=g.user_id
            WHERE g.user_id IN ({placeholders})
            ORDER BY g.id DESC LIMIT 10
            """,
            member_ids,
        ).fetchall()
        wrong_row = conn.execute(
            f"""
            SELECT COUNT(*) AS c,
                   SUM(CASE WHEN error_type='calculation' THEN 1 ELSE 0 END) AS calculation,
                   SUM(CASE WHEN error_type='sign' THEN 1 ELSE 0 END) AS sign,
                   SUM(CASE WHEN error_type='variable' THEN 1 ELSE 0 END) AS variable,
                   SUM(CASE WHEN error_type='logic' THEN 1 ELSE 0 END) AS logic,
                   SUM(CASE WHEN error_type='other' THEN 1 ELSE 0 END) AS other
            FROM wrong_answers WHERE user_id IN ({placeholders})
            """,
            member_ids,
        ).fetchone()
    return {
        "class_id": class_id,
        "member_count": len(member_ids),
        "members": [dict(m) for m in members],
        "grading_count": int(grading_row["c"] or 0),
        "avg_score": (round(float(grading_row["avg_score"]), 1)
                      if grading_row["avg_score"] is not None else None),
        "recent_gradings": [dict(r) for r in recent],
        "wrong_answer_count": int(wrong_row["c"] or 0),
        "error_type_distribution": {
            "calculation": int(wrong_row["calculation"] or 0),
            "sign": int(wrong_row["sign"] or 0),
            "variable": int(wrong_row["variable"] or 0),
            "logic": int(wrong_row["logic"] or 0),
            "other": int(wrong_row["other"] or 0),
        },
    }
