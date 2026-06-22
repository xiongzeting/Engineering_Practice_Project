"""FastAPI 应用入口，定义全部 HTTP 路由与请求/响应模型。

本文件承担三层职责：
1. **应用初始化**：创建 FastAPI 实例，挂载静态文件目录，启动时调用
   :func:`app.services.db.init_db` 执行建表与幂等迁移。
2. **认证依赖**：通过 :func:`_current_user` 解析 Bearer 令牌，注入当前用户
   字典到每个受保护的端点。角色分层在这里生效——student 只能访问自己的
   数据，teacher 限定在自己创建的班级范围内，admin 继承 teacher 的全部
   权限并能操作系统级配置。
3. **业务路由**：按职责分组——
   - ``/api/auth/*``    注册/登录/当前用户/管理员引导状态
   - ``/api/history/*`` /api/favorites /api/wrong-answers 等「学生个人中心」数据
   - ``/api/ocr*``      OCR 识别（纯视觉 / 多引擎 / 纠错）
   - ``/api/grade``     步骤评分（同时触发错题自动入库）
   - ``/api/classes*``  班级：学生侧的加入/退出，通用查询
   - ``/api/admin/*``   教师 + 管理员：用户、班级、基础知识点管理
   - ``/api/teacher/*`` 教师限定：本班学生、班级 KG 定制层

请求体较大的端点（如批量错误注入 / 批改）使用 ``Body(...)``；含文件上传的
端点（如 OCR）使用 ``UploadFile`` + ``Form``。错题自动入库、精确报告生成
等较重的副作用走 ``BackgroundTasks``，避免阻塞 HTTP 响应。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Body, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import AuthResponse, AuthUser, DetailReportResponse, GradingResult, SaveSessionResponse
from app.services.auth import create_access_token, hash_password, parse_access_token, verify_password
from app.services.db import (
    create_user,
    delete_favorite_assignment,
    delete_user as db_delete_user,
    delete_wrong_answer,
    get_admin_overview,
    get_grading_record,
    get_user_by_id,
    get_user_by_username,
    get_user_dashboard,
    get_wrong_answer,
    get_wrong_answer_stats,
    init_db,
    list_all_users,
    list_favorite_assignments,
    list_grading_records,
    list_llm_failures,
    list_ocr_records,
    list_wrong_answers,
    populate_wrong_answers_from_session,
    save_favorite_assignment,
    save_grading_record,
    save_llm_failure,
    save_ocr_record,
    update_detail_report,
    update_user_role,
    update_wrong_answer,
)
from app.services.report_generator import generate_detail_report
from app.services.ocr_corrector import llm_correct_ocr_text
from app.services.ocr_postprocess import (
    attach_low_confidence_flag,
    build_ocr_text,
    group_segments_by_question,
)
from app.services.ocr_service import OCRService
from app.services.scorer import score_questions, score_steps, score_one_problem
from app.services.step_parser import split_steps
from app.services.vision_corrector import apply_vision_correction
from app.services.vision_ocr import vision_only_ocr

app = FastAPI(title="Math OCR & Step Grading API", version="0.2.0")
ocr_service = OCRService()
auth_scheme = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = Path(__file__).resolve().parent.parent
static_dir = base_dir / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")
init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/{page_name}")
def app_page(page_name: str) -> FileResponse:
    if page_name not in {"grading", "history", "favorites", "wrong-answers", "account", "admin"}:
        raise HTTPException(status_code=404, detail="页面不存在。")
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "llm_configured": bool(settings.llm_api_key),
        "llm_model": settings.llm_model,
        "llm_base_url": settings.llm_base_url,
    }


def _read_and_validate_image(image: UploadFile | None) -> bytes:
    if not image:
        return b""
    image_bytes = image.file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=400, detail=f"图片超过大小限制: {settings.max_upload_mb}MB")
    return image_bytes


def _current_user(credentials: HTTPAuthorizationCredentials | None = Depends(auth_scheme)) -> dict:
    """FastAPI 依赖：从 Bearer token 解析当前登录用户。

    流程：取 ``Authorization: Bearer <token>`` → 用
    :func:`parse_access_token` 校验签名和有效期 → 查 SQLite 取用户记录。
    任何一环失败都抛 401，让前端跳回登录。
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(status_code=401, detail="请先登录。")
    payload = parse_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="登录已失效，请重新登录。")
    user = get_user_by_id(int(payload["uid"]))
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在。")
    return user


@app.post("/api/auth/register", response_model=AuthResponse)
def register(
    username: str = Form(...),
    password: str = Form(...),
    role: str = Form(default="student"),
) -> AuthResponse:
    uname = username.strip()
    pwd = password.strip()
    user_role = role.strip().lower() if role else "student"
    if len(uname) < 3:
        raise HTTPException(status_code=400, detail="用户名至少3位。")
    if len(pwd) < 6:
        raise HTTPException(status_code=400, detail="密码至少6位。")
    if user_role not in {"student", "teacher"}:
        user_role = "student"
    if get_user_by_username(uname):
        raise HTTPException(status_code=400, detail="用户名已存在。")
    # Bootstrap: allow the first admin to self-register when no admin exists.
    if user_role == "admin":
        existing = list_all_users()
        if any(u.get("role") == "admin" for u in existing):
            raise HTTPException(status_code=403, detail="系统已存在管理员，请联系管理员授权。")
    uid = create_user(uname, hash_password(pwd), user_role)
    user = get_user_by_id(uid)
    if not user:
        raise HTTPException(status_code=500, detail="注册失败。")
    token = create_access_token(user)
    return AuthResponse(
        access_token=token,
        user=AuthUser(id=int(user["id"]), username=str(user["username"]), role=str(user["role"])),
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def login(
    username: str = Form(...),
    password: str = Form(...),
) -> AuthResponse:
    user = get_user_by_username(username.strip())
    if not user or not verify_password(password.strip(), str(user.get("password_hash", ""))):
        raise HTTPException(status_code=401, detail="用户名或密码错误。")
    token = create_access_token(user)
    return AuthResponse(
        access_token=token,
        user=AuthUser(id=int(user["id"]), username=str(user["username"]), role=str(user["role"])),
    )


@app.get("/api/auth/me", response_model=AuthUser)
def me(user: dict = Depends(_current_user)) -> AuthUser:
    return AuthUser(id=int(user["id"]), username=str(user["username"]), role=str(user["role"]))


@app.get("/api/auth/bootstrap-status")
def bootstrap_status() -> dict:
    """公开端点：告诉前端注册界面「管理员角色是否可注册」。

    若系统尚无管理员，允许首位注册者选 ``admin`` 角色（bootstrap 一位
    管理员）；之后该入口自动关闭。前端依据本字段决定是否显示「管理员」
    单选项。
    """
    has_admin = any(u.get("role") == "admin" for u in list_all_users())
    return {"admin_exists": has_admin}


@app.get("/api/history/ocr")
def ocr_history(limit: int = 20, user: dict = Depends(_current_user)) -> dict:
    """返回当前用户的 OCR 历史（按时间倒序）。"""
    return {"items": list_ocr_records(int(user["id"]), limit=limit)}


@app.get("/api/history/grading")
def grading_history(limit: int = 20, user: dict = Depends(_current_user)) -> dict:
    """返回当前用户的批改历史（按时间倒序）。"""
    return {"items": list_grading_records(int(user["id"]), limit=limit)}


@app.get("/api/favorites")
def favorites(limit: int = 50, user: dict = Depends(_current_user)) -> dict:
    """返回当前用户的收藏作业列表。"""
    return {"items": list_favorite_assignments(int(user["id"]), limit=limit)}


@app.post("/api/favorites")
def create_favorite(
    title: str = Form(...),
    ocr_text: str = Form(...),
    total_score: float = Form(default=0),
    feedback: str = Form(default=""),
    knowledge_tags_json: str = Form(default="[]"),
    report_json: str = Form(default="{}"),
    user: dict = Depends(_current_user),
) -> dict:
    """收藏一条作业。``knowledge_tags_json`` 和 ``report_json`` 是 JSON
    字符串（前端表单只能传文本），后端解析失败时回退到空列表/空 dict。"""
    clean_title = title.strip() or "未命名作业"
    clean_text = ocr_text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="收藏内容不能为空。")
    try:
        tags_raw: Any = json.loads(knowledge_tags_json or "[]")
        knowledge_tags = [str(x) for x in tags_raw] if isinstance(tags_raw, list) else []
    except Exception:
        knowledge_tags = []
    try:
        report_raw: Any = json.loads(report_json or "{}")
        report = report_raw if isinstance(report_raw, dict) else {}
    except Exception:
        report = {}
    favorite_id = save_favorite_assignment(
        user_id=int(user["id"]),
        title=clean_title,
        ocr_text=clean_text,
        total_score=float(total_score),
        feedback=feedback.strip(),
        knowledge_tags=knowledge_tags,
        report=report,
    )
    return {"id": favorite_id, "ok": True}


@app.delete("/api/favorites/{favorite_id}")
def remove_favorite(favorite_id: int, user: dict = Depends(_current_user)) -> dict:
    ok = delete_favorite_assignment(int(user["id"]), favorite_id)
    if not ok:
        raise HTTPException(status_code=404, detail="收藏不存在。")
    return {"ok": True}


@app.get("/api/dashboard")
def dashboard(user: dict = Depends(_current_user)) -> dict:
    """学生首页 dashboard：统计 + 能力位（是否配置 LLM、上传上限）。"""
    stats = get_user_dashboard(int(user["id"]))
    return {
        "user": {
            "id": int(user["id"]),
            "username": str(user["username"]),
            "role": str(user["role"]),
        },
        "stats": stats,
        "capabilities": {
            "llm_configured": bool(settings.llm_api_key),
            "llm_model": settings.llm_model,
            "max_upload_mb": settings.max_upload_mb,
        },
    }


@app.get("/api/admin/overview")
def admin_overview(user: dict = Depends(_current_user)) -> dict:
    """教师后台首页的概览：分角色用户数、总批改数、平均分、最近 8 条批改。

    角色权限：``teacher`` / ``admin``。学生访问返回 403。
    """
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可查看管理后台。")
    return get_admin_overview()


# ============================================================
# Admin: user management
# ============================================================

@app.get("/api/admin/users")
def admin_list_users(user: dict = Depends(_current_user)) -> dict:
    """列出全部用户（含每人的批改数与错题数）。

    角色：``teacher`` / ``admin``。教师只能在前端筛选出自己班级的学生，
    后端这里仍返回全部，由前端按需筛选（避免来回请求）。
    """
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可管理用户。")
    return {"users": list_all_users()}


@app.patch("/api/admin/users/{user_id}/role")
def admin_update_user_role(
    user_id: int,
    payload: dict = Body(...),
    user: dict = Depends(_current_user),
) -> dict:
    """修改某用户的角色。

    约束：
    - 不能改自己的角色（防误操作锁死自己）；
    - 角色必须是 ``student`` / ``teacher`` / ``admin`` 三者之一；
    - ``update_user_role`` 内部拒绝升到 ``admin``（只能 bootstrap 一位）。
    """
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可管理用户。")
    if int(user["id"]) == int(user_id):
        raise HTTPException(status_code=400, detail="不能修改自己的角色。")
    role = str(payload.get("role") or "").strip().lower()
    if role not in {"student", "teacher", "admin"}:
        raise HTTPException(status_code=400, detail="角色必须是 student / teacher / admin。")
    ok = update_user_role(user_id, role)
    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在。")
    return {"ok": True, "user_id": user_id, "role": role}


@app.delete("/api/admin/users/{user_id}")
def admin_delete_user(user_id: int, user: dict = Depends(_current_user)) -> dict:
    """删除用户及其所有关联数据（错题、收藏、批改、OCR）。不能删自己。"""
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可管理用户。")
    if int(user["id"]) == int(user_id):
        raise HTTPException(status_code=400, detail="不能删除自己。")
    ok = db_delete_user(user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="用户不存在。")
    return {"ok": True, "deleted": user_id}


# ============================================================
# Classes (学生分班 + 教师报表)
# ============================================================

@app.get("/api/classes")
def classes_list(user: dict = Depends(_current_user)) -> dict:
    """按调用者角色返回班级列表（见 :func:`classes.list_classes`）。"""
    from app.services import classes as classes_svc
    return {"items": classes_svc.list_classes(int(user["id"]), str(user.get("role")))}


@app.get("/api/classes/{class_id}")
def classes_get(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """取单个班级详情。

    权限：
    - 教师 / 管理员：全部可见；
    - 学生：只能看自己已加入的班级，否则 403。
    """
    from app.services import classes as classes_svc
    cls = classes_svc.get_class(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="班级不存在。")
    # 学生只能看自己加入的班级；教师/管理员无限制。
    if str(user.get("role")) not in {"teacher", "admin"}:
        if class_id not in classes_svc.list_user_class_ids(int(user["id"])):
            raise HTTPException(status_code=403, detail="你未加入该班级。")
    return cls


@app.post("/api/classes/join")
def classes_join(payload: dict = Body(...), user: dict = Depends(_current_user)) -> dict:
    """学生用邀请码加入班级。重复加入、无效码都会抛 400。"""
    from app.services import classes as classes_svc
    code = str(payload.get("invite_code") or "").strip()
    if not code:
        raise HTTPException(status_code=400, detail="请输入邀请码。")
    try:
        cls = classes_svc.join_class(int(user["id"]), code)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "class": cls}


@app.post("/api/classes/{class_id}/leave")
def classes_leave(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """学生退出班级。未加入时返回 404。"""
    from app.services import classes as classes_svc
    ok = classes_svc.leave_class(int(user["id"]), class_id)
    if not ok:
        raise HTTPException(status_code=404, detail="你未加入该班级。")
    return {"ok": True}


@app.get("/api/classes/{class_id}/report")
def classes_report(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """取班级报表（成员、批改数、平均分、错因分布）。仅教师/管理员可用。"""
    from app.services import classes as classes_svc
    if str(user.get("role")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可查看班级报表。")
    if not classes_svc.get_class(class_id):
        raise HTTPException(status_code=404, detail="班级不存在。")
    return classes_svc.get_class_report(class_id)


@app.post("/api/admin/classes")
def admin_create_class(
    payload: dict = Body(...), user: dict = Depends(_current_user)
) -> dict:
    """创建班级。教师默认把 ``creator_id`` 设为自己；管理员可以通过
    ``creator_id`` 字段把班级指派给其他教师。"""
    from app.services import classes as classes_svc
    if str(user.get("role")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可创建班级。")
    # 管理员可以把班级指派给其他教师（前端选 creator）。
    creator_id = int(user["id"])
    if str(user.get("role")) == "admin" and payload.get("creator_id"):
        try:
            creator_id = int(payload["creator_id"])
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="creator_id 必须是数字。")
        target = get_user_by_id(creator_id)
        if not target or str(target.get("role")) != "teacher":
            raise HTTPException(status_code=400, detail="creator_id 必须指向已注册教师。")
    try:
        cls = classes_svc.create_class(
            name=str(payload.get("name") or ""),
            creator_id=creator_id,
            stage=str(payload.get("stage") or "middle"),
            grade=int(payload.get("grade") or 0),
            description=str(payload.get("description") or ""),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "class": cls}


@app.patch("/api/admin/classes/{class_id}")
def admin_update_class(
    class_id: int, payload: dict = Body(...), user: dict = Depends(_current_user)
) -> dict:
    """局部更新班级字段（``name`` / ``stage`` / ``grade`` / ``description``）。"""
    from app.services import classes as classes_svc
    if str(user.get("role")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可修改班级。")
    try:
        cls = classes_svc.update_class(
            class_id,
            name=payload.get("name"),
            stage=payload.get("stage"),
            grade=payload.get("grade"),
            description=payload.get("description"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not cls:
        raise HTTPException(status_code=404, detail="班级不存在。")
    return {"ok": True, "class": cls}


@app.delete("/api/admin/classes/{class_id}")
def admin_delete_class(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """删除班级及其成员关系（不删学生的历史批改）。"""
    from app.services import classes as classes_svc
    if str(user.get("role")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可删除班级。")
    ok = classes_svc.delete_class(class_id)
    if not ok:
        raise HTTPException(status_code=404, detail="班级不存在。")
    return {"ok": True}


@app.post("/api/admin/classes/{class_id}/regenerate-invite")
def admin_regenerate_invite(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """重置班级邀请码。旧码立即失效。"""
    from app.services import classes as classes_svc
    if str(user.get("role")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可重置邀请码。")
    cls = classes_svc.regenerate_invite_code(class_id)
    if not cls:
        raise HTTPException(status_code=404, detail="班级不存在。")
    return {"ok": True, "class": cls}


@app.get("/api/admin/classes/{class_id}/members")
def admin_class_members(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """列出班级成员详情（用户信息 + 加入时间 + 批改数 + 错题数 + 平均分）。"""
    from app.services import classes as classes_svc
    if str(user.get("role")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师可查看班级成员。")
    if not classes_svc.get_class(class_id):
        raise HTTPException(status_code=404, detail="班级不存在。")
    return {"items": classes_svc.list_class_members(class_id)}


# ============================================================
# Teacher-scoped endpoints: limited to classes this teacher owns.
# Admin can call these too (admin inherits teacher powers).
# ============================================================

def _assert_teaches_class(user: dict, class_id: int) -> None:
    """权限守卫：断言「当前用户能管理这个班级」。

    - 管理员：任何存在的班级都通过；
    - 教师：必须是该班级的 ``creator_id``；
    - 学生：永远不通过。

    不通过抛 403。所有教师作用域的端点都用这个守卫。
    """
    from app.services import classes as classes_svc
    if str(user.get("role")) == "admin":
        if not classes_svc.get_class(class_id):
            raise HTTPException(status_code=404, detail="班级不存在。")
        return
    if str(user.get("role")) != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可访问该班级。")
    if not classes_svc.is_teacher_of_class(int(user["id"]), class_id):
        raise HTTPException(status_code=403, detail="你无权访问该班级。")


@app.get("/api/teacher/students")
def teacher_list_students(user: dict = Depends(_current_user)) -> dict:
    """Students in any class this teacher owns. Admin sees all students."""
    from app.services import classes as classes_svc
    if str(user.get("role")) == "admin":
        # All students system-wide.
        all_users = list_all_users()
        return {"items": [u for u in all_users if u.get("role") == "student"]}
    if str(user.get("role")) != "teacher":
        raise HTTPException(status_code=403, detail="仅教师可查看学生列表。")
    return {"items": classes_svc.list_users_in_teacher_classes(int(user["id"]))}


@app.get("/api/teacher/classes/{class_id}/kg")
def teacher_get_class_kg(class_id: int, user: dict = Depends(_current_user)) -> dict:
    """Effective KG for a class = base + this class's overrides."""
    from app.services import knowledge_graph as kg
    from app.services import classes as classes_svc
    _assert_teaches_class(user, class_id)
    overrides = kg.list_class_overrides(class_id)
    nodes = kg.get_effective_class_nodes(class_id)
    return {
        "class_id": class_id,
        "class_name": (classes_svc.get_class(class_id) or {}).get("name"),
        "nodes": nodes,
        "overrides": overrides,
    }


@app.post("/api/teacher/classes/{class_id}/kg/nodes")
def teacher_upsert_class_kg_node(
    class_id: int, payload: dict = Body(...), user: dict = Depends(_current_user)
) -> dict:
    from app.services import knowledge_graph as kg
    _assert_teaches_class(user, class_id)
    try:
        override = kg.upsert_class_override(class_id, payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "override": override}


@app.delete("/api/teacher/classes/{class_id}/kg/nodes/{node_id}")
def teacher_delete_class_kg_node(
    class_id: int, node_id: str, user: dict = Depends(_current_user)
) -> dict:
    from app.services import knowledge_graph as kg
    _assert_teaches_class(user, class_id)
    override = kg.delete_class_override(class_id, node_id)
    return {"ok": True, "override": override}


@app.post("/api/teacher/classes/{class_id}/kg/nodes/{node_id}/restore")
def teacher_restore_class_kg_node(
    class_id: int, node_id: str, user: dict = Depends(_current_user)
) -> dict:
    from app.services import knowledge_graph as kg
    _assert_teaches_class(user, class_id)
    ok = kg.restore_class_override(class_id, node_id)
    return {"ok": ok}


@app.get("/api/admin/llm-failures")
def admin_list_llm_failures(
    limit: int = 50,
    user_id: int | None = None,
    endpoint: str | None = None,
    stage: str | None = None,
    user: dict = Depends(_current_user),
) -> dict:
    """Teacher-only: backend log of LLM call failures. Diagnostic data
    (raw previews, error messages, retry traces) is never sent to students."""
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可查看 LLM 故障日志。")
    items = list_llm_failures(
        limit=limit,
        user_id=user_id,
        endpoint=endpoint or None,
        stage=stage or None,
    )
    return {"items": items, "count": len(items)}


# ============================================================
# Wrong-answer book
# ============================================================

@app.get("/api/wrong-answers")
def wrong_answers_list(
    status: str | None = None,
    kg_node: str | None = None,
    limit: int = 50,
    user: dict = Depends(_current_user),
) -> dict:
    return {"items": list_wrong_answers(
        int(user["id"]),
        status=status or None,
        kg_node=kg_node or None,
        limit=limit,
    )}


@app.get("/api/wrong-answers/stats")
def wrong_answers_stats(user: dict = Depends(_current_user)) -> dict:
    return get_wrong_answer_stats(int(user["id"]))


@app.get("/api/wrong-answers/{wrong_id}")
def wrong_answers_get(wrong_id: int, user: dict = Depends(_current_user)) -> dict:
    item = get_wrong_answer(int(user["id"]), wrong_id)
    if not item:
        raise HTTPException(status_code=404, detail="错题不存在或无权访问。")
    return item


@app.patch("/api/wrong-answers/{wrong_id}")
def wrong_answers_update(
    wrong_id: int,
    payload: dict = Body(...),
    user: dict = Depends(_current_user),
) -> dict:
    status = payload.get("status")
    note = payload.get("note")
    if status is None and note is None:
        raise HTTPException(status_code=400, detail="必须提供 status 或 note 字段。")
    item = update_wrong_answer(
        int(user["id"]),
        wrong_id,
        status=status,
        note=note,
    )
    if not item:
        raise HTTPException(status_code=404, detail="错题不存在或无权访问。")
    return {"ok": True, "item": item}


@app.delete("/api/wrong-answers/{wrong_id}")
def wrong_answers_delete(wrong_id: int, user: dict = Depends(_current_user)) -> dict:
    ok = delete_wrong_answer(int(user["id"]), wrong_id)
    if not ok:
        raise HTTPException(status_code=404, detail="错题不存在或无权访问。")
    return {"ok": True}


@app.post("/api/ocr")
def run_ocr(
    image: UploadFile | None = File(default=None),
    extracted_text: str | None = Form(default=None),
    use_llm_correction: bool = Form(default=False),
    use_vision_correction: bool = Form(default=False),
    return_llm_debug: bool = Form(default=False),
    user: dict = Depends(_current_user),
) -> dict:
    if not image and not (extracted_text and extracted_text.strip()):
        raise HTTPException(status_code=400, detail="请上传图片或提供识别文本。")

    image_bytes = _read_and_validate_image(image)
    ocr_result = ocr_service.extract(image_bytes=image_bytes, fallback_text=extracted_text)
    if not ocr_result.text:
        extra = f" 详细错误：{ocr_result.error}" if ocr_result.error else ""
        raise HTTPException(
            status_code=400,
            detail=f"OCR 未识别到有效文本。请检查图片清晰度，确认 Pix2Text 模型已下载，或手工输入文本。{extra}",
        )

    segments = attach_low_confidence_flag(ocr_result.segments, threshold=0.78)
    vision_stats = {"enabled": False, "corrected_count": 0}
    if use_vision_correction:
        vision_source = image_bytes if image_bytes else None
        segments, vision_stats = apply_vision_correction(vision_source, segments)
        segments = attach_low_confidence_flag(segments, threshold=0.78)
    question_groups = group_segments_by_question(segments)
    cleaned_text = build_ocr_text(segments).strip() or ocr_result.text
    corrected_by_llm = False
    correction_note = ""
    llm_debug: dict | None = None
    if use_llm_correction:
        llm_fix = llm_correct_ocr_text(
            cleaned_text,
            segments,
            return_debug=return_llm_debug,
            debug_max_chars=settings.llm_debug_max_chars,
        )
        if llm_fix:
            cleaned_text = llm_fix.get("corrected_text", cleaned_text)
            correction_note = str(llm_fix.get("notes", "") or "")
            llm_debug = llm_fix.get("debug")
            corrected_by_llm = True
    text_count = sum(1 for s in segments if str(s.get("type", "")).upper() in ("TEXT", "TITLE", "LINE"))
    formula_count = sum(1 for s in segments if str(s.get("type", "")).upper() in ("FORMULA", "ISOLATED", "EMBEDDING"))
    noisy_count = sum(1 for s in segments if bool(s.get("noisy")))

    steps = split_steps(cleaned_text)
    record_id = save_ocr_record(
        user_id=int(user["id"]),
        engine=str(ocr_result.engine),
        ocr_text=cleaned_text,
        steps_count=len(steps),
    )
    return {
        "ocr_text": cleaned_text,
        "ocr_text_raw": ocr_result.text,
        "engine": ocr_result.engine,
        "segments": segments,
        "question_groups": question_groups,
        "segment_stats": {
            "text_count": text_count,
            "formula_count": formula_count,
            "noisy_count": noisy_count,
            "total": len(segments),
        },
        "preprocessed": ocr_result.preprocessed,
        "saved_path": ocr_result.saved_path,
        "ocr_error": ocr_result.error,
        "corrected_by_llm": corrected_by_llm,
        "correction_note": correction_note,
        "llm_configured": bool(settings.llm_api_key),
        "llm_debug": llm_debug,
        "vision_correction": vision_stats,
        "steps": steps,
        "record_id": record_id,
    }


@app.post("/api/ocr-vision-only")
def run_ocr_vision_only(
    image: UploadFile = File(...),
    return_llm_debug: bool = Form(default=False),
    user: dict = Depends(_current_user),
) -> dict:
    image_bytes = _read_and_validate_image(image)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="请上传图片。")
    res = vision_only_ocr(image_bytes=image_bytes, return_debug=return_llm_debug)
    if not res.get("ok"):
        # Persist the real cause for admin/dev inspection; surface only a generic
        # message to the user so we don't leak LLM internals or prompt scaffolding.
        try:
            save_llm_failure(
                user_id=int(user["id"]) if user.get("id") else None,
                endpoint="/api/ocr-vision-only",
                stage="ocr",
                error=str(res.get("error") or res.get("diag") or ""),
                raw_preview=str(res.get("raw_preview") or "")[:1500],
                attempts=res.get("attempts") if isinstance(res.get("attempts"), list) else None,
            )
        except Exception:
            pass
        raise HTTPException(status_code=400, detail="OCR 识别失败，请重试或改用文本批改")

    raw_text = str(res.get("ocr_text", "") or "").strip()
    raw_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    pseudo_segments = [
        {
            "index": i + 1,
            "text": line,
            "type": "LINE",
            "bbox": [0, 0, 0, 0],
            "score": 0.0,
        }
        for i, line in enumerate(raw_lines)
    ]
    segments = attach_low_confidence_flag(pseudo_segments, threshold=0.78)
    question_groups = group_segments_by_question(segments)
    cleaned_text = build_ocr_text(segments).strip() or raw_text
    steps = split_steps(cleaned_text)

    text_count = sum(1 for s in segments if str(s.get("type", "")).upper() in ("TEXT", "TITLE", "LINE"))
    formula_count = sum(1 for s in segments if str(s.get("type", "")).upper() in ("FORMULA", "ISOLATED", "EMBEDDING"))
    noisy_count = sum(1 for s in segments if bool(s.get("noisy")))

    res.update(
        {
            "ocr_text_raw": raw_text,
            "ocr_text": cleaned_text,
            "question_text": str(res.get("question_text", "") or ""),
            "steps_text": str(res.get("steps_text", "") or ""),
            "problems": res.get("problems") or [],
            "segments": segments,
            "question_groups": question_groups,
            "segment_stats": {
                "text_count": text_count,
                "formula_count": formula_count,
                "noisy_count": noisy_count,
                "total": len(segments),
            },
            "steps": steps,
            "saved_path": "",
            "preprocessed": False,
        }
    )
    record_id = save_ocr_record(
        user_id=int(user["id"]),
        engine=str(res.get("engine", "vision-only")),
        ocr_text=str(res.get("ocr_text", "") or ""),
        steps_count=len(steps),
    )
    res["record_id"] = record_id
    return res


def _async_generate_detail_report(record_id: int, grading_payload: dict[str, Any]) -> None:
    try:
        md = generate_detail_report(grading_payload)
        update_detail_report(record_id, md, "ready")
    except Exception as e:
        err_msg = f"# 精确报告生成失败\n\n原因：{type(e).__name__}: {e}\n\n可点击\"重新生成\"重试。"
        update_detail_report(record_id, err_msg, "failed")


@app.post("/api/grade", response_model=GradingResult)
def grade_homework(
    extracted_text: str = Form(...),
    reference_solution: str | None = Form(default=None),
    use_llm: bool = Form(default=False),
    question_max_scores: str | None = Form(default=None),
    question_text: str | None = Form(default=None),
    step_lines: str | None = Form(default=None),
    qno: int | None = Form(default=None),
    user: dict = Depends(_current_user),
) -> GradingResult:
    ocr_text = extracted_text.strip()
    if not ocr_text and not (step_lines or "").strip():
        raise HTTPException(status_code=400, detail="请先执行 OCR 或手工输入公式文本。")

    parsed_max_scores: dict[int, float] = {}
    if question_max_scores:
        try:
            raw = json.loads(question_max_scores)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        parsed_max_scores[int(k)] = float(v)
                    except (ValueError, TypeError):
                        continue
        except json.JSONDecodeError:
            pass  # ignore malformed payload, fall back to default equal split

    # Preferred path: caller supplied explicit stem (question_text) + pre-split
    # step_lines (typically from the OCR review pass). This skips the regex
    # question-splitter entirely, so the stem is never scored as a step and
    # questions without standard numbering still work.
    preset_steps: list[str] = []
    if step_lines:
        try:
            parsed_steps = json.loads(step_lines)
            if isinstance(parsed_steps, list):
                preset_steps = [str(s) for s in parsed_steps if str(s).strip()]
        except json.JSONDecodeError:
            preset_steps = []

    if preset_steps:
        # Single-question grading with explicit step list. qno/max_score come
        # from the form (the frontend grades one question at a time).
        effective_qno = int(qno) if qno and int(qno) > 0 else 1
        max_score = parsed_max_scores.get(effective_qno, 100.0) if parsed_max_scores else 100.0
        grade, flat_steps, flat_scores, score_engine, q_feedback, q_meta = score_one_problem(
            qno=effective_qno,
            question_text=(question_text or "").strip(),
            step_lines=preset_steps,
            max_score=max_score,
            reference_raw=reference_solution,
            use_llm=use_llm,
        )
        total = round(max(0.0, min(100.0, grade.score / max_score * 100)), 2) if max_score > 0 else 0.0
        return GradingResult(
            ocr_text=ocr_text,
            steps=flat_steps,
            step_scores=flat_scores,
            total_score=total,
            feedback=q_feedback,
            engine=score_engine,
            grading_meta=q_meta,
            questions=[grade],
            total_max_score=max_score,
            record_id=None,
        )

    questions, flat_steps, flat_scores, total, total_max, feedback, score_engine, grading_meta = score_questions(
        ocr_text=ocr_text,
        question_max_scores=parsed_max_scores or None,
        reference_raw=reference_solution,
        use_llm=use_llm,
        question_text=(question_text or "").strip() or None,
    )

    if not flat_steps:
        raise HTTPException(status_code=400, detail="未检测到可评分的步骤，请确认公式文本。")

    return GradingResult(
        ocr_text=ocr_text,
        steps=flat_steps,
        step_scores=flat_scores,
        total_score=total,
        feedback=feedback,
        engine=f"{score_engine}",
        grading_meta=grading_meta,
        questions=questions,
        total_max_score=total_max,
        record_id=None,
    )


@app.post("/api/grading/session", response_model=SaveSessionResponse)
def save_grading_session(
    payload: dict = Body(...),
    background_tasks: BackgroundTasks = None,
    user: dict = Depends(_current_user),
) -> SaveSessionResponse:
    """Persist the aggregated session result as ONE grading_records row and kick off detail report generation."""
    questions = payload.get("questions") or []
    if not isinstance(questions, list) or not questions:
        raise HTTPException(status_code=400, detail="payload.questions 不能为空。")

    ocr_text = str(payload.get("ocr_text") or "")
    total_score = float(payload.get("total_score") or 0)
    total_max_score = float(payload.get("total_max_score") or 0)
    feedback = str(payload.get("feedback") or "")
    engine = str(payload.get("engine") or "llm+rule")
    steps_count = sum(len(q.get("steps") or []) for q in questions if isinstance(q, dict))

    grading_meta = {
        "scoring_mode": "session",
        "questions_meta": [
            {
                "qno": int(q.get("qno", 0)),
                "max_score": float(q.get("max_score", 0)),
                "score": float(q.get("score", 0)),
                "steps": len(q.get("steps") or []),
            }
            for q in questions
            if isinstance(q, dict)
        ],
    }
    grading_payload = {
        "ocr_text": ocr_text,
        "total_score": total_score,
        "total_max_score": total_max_score,
        "feedback": feedback,
        "questions": questions,
    }
    # Knowledge graph enrichment — never blocks grading.
    try:
        from app.services.knowledge_graph import enrich_report_with_kg
        grading_payload["kg_report"] = enrich_report_with_kg(questions)
    except Exception as e:
        grading_payload["kg_report_error"] = f"{type(e).__name__}: {e}"
    record_id = save_grading_record(
        user_id=int(user["id"]),
        engine=engine,
        total_score=total_score,
        steps_count=steps_count,
        ocr_text=ocr_text,
        grading_meta=grading_meta,
        grading_result=grading_payload,
    )
    # Auto-populate wrong-answer book for any non-full-credit questions.
    try:
        kg_report = grading_payload.get("kg_report") if isinstance(grading_payload.get("kg_report"), dict) else None
        populate_wrong_answers_from_session(
            user_id=int(user["id"]),
            grading_record_id=record_id,
            questions=questions,
            kg_report=kg_report,
        )
    except Exception as e:
        # Wrong-answer book is best-effort; never block grading.
        grading_payload["wrong_answers_error"] = f"{type(e).__name__}: {e}"
    if background_tasks is not None:
        background_tasks.add_task(_async_generate_detail_report, record_id, grading_payload)
    return SaveSessionResponse(record_id=record_id, status="pending")


@app.get("/api/grading/{record_id}/detail", response_model=DetailReportResponse)
def get_detail_report(record_id: int, user: dict = Depends(_current_user)) -> DetailReportResponse:
    row = get_grading_record(record_id, user_id=int(user["id"]))
    if not row:
        raise HTTPException(status_code=404, detail="批改记录不存在或无权访问。")
    return DetailReportResponse(
        status=str(row.get("detail_report_status") or "pending"),
        markdown=row.get("detail_report_md"),
        updated_at=str(row.get("created_at") or "") or None,
    )


@app.get("/api/grading/{record_id}/payload")
def get_grading_payload(record_id: int, user: dict = Depends(_current_user)) -> dict:
    """Return the full grading_result_json so frontend can restore questions/steps."""
    row = get_grading_record(record_id, user_id=int(user["id"]))
    if not row:
        raise HTTPException(status_code=404, detail="批改记录不存在或无权访问。")
    raw = row.get("grading_result_json")
    payload: dict[str, Any] = {}
    if raw:
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            payload = {}
    return {
        "ok": bool(payload),
        "ocr_text": payload.get("ocr_text") or "",
        "total_score": payload.get("total_score"),
        "total_max_score": payload.get("total_max_score"),
        "feedback": payload.get("feedback") or "",
        "questions": payload.get("questions") or [],
        "kg_report": payload.get("kg_report"),
        "engine": row.get("engine"),
        "created_at": str(row.get("created_at") or ""),
    }


@app.post("/api/ocr/split")
def split_ocr_text_endpoint(payload: dict = Body(...), user: dict = Depends(_current_user)) -> dict:
    """Split raw OCR text into question groups (used by history restore)."""
    from app.services.question_splitter import split_text_into_questions
    text = str(payload.get("text") or "")
    pairs = split_text_into_questions(text)
    groups = [{"qno": qno, "text": txt} for qno, txt in pairs]
    return {"ok": True, "question_groups": groups, "ocr_text": text}


@app.post("/api/grading/{record_id}/detail/regenerate", response_model=DetailReportResponse)
def regenerate_detail_report(
    record_id: int,
    background_tasks: BackgroundTasks,
    user: dict = Depends(_current_user),
) -> DetailReportResponse:
    row = get_grading_record(record_id, user_id=int(user["id"]))
    if not row:
        raise HTTPException(status_code=404, detail="批改记录不存在或无权访问。")

    raw_result = row.get("grading_result_json")
    grading_payload: dict[str, Any] | None = None
    if raw_result:
        try:
            grading_payload = json.loads(str(raw_result))
        except json.JSONDecodeError:
            grading_payload = None
    if not grading_payload:
        raise HTTPException(status_code=400, detail="该记录缺少原始批改数据，无法重新生成报告。")

    update_detail_report(record_id, None, "pending")
    background_tasks.add_task(_async_generate_detail_report, record_id, grading_payload)
    return DetailReportResponse(status="pending", markdown=None, updated_at=None)


# ============================================================
# Knowledge Graph endpoints
# ============================================================

@app.get("/api/kg/ontology")
def kg_ontology(user: dict = Depends(_current_user)) -> dict:
    """Full ontology (40 K1-K12 nodes + edges) for visualization."""
    from app.services.knowledge_graph import get_all_nodes, get_all_edges
    return {"nodes": get_all_nodes(), "edges": get_all_edges()}


@app.get("/api/kg/mastery")
def kg_mastery(limit: int = 50, user: dict = Depends(_current_user)) -> dict:
    """Current user's per-node mastery state."""
    from app.services.knowledge_graph import compute_user_mastery
    return {"nodes": compute_user_mastery(int(user["id"]), limit=limit)}


@app.get("/api/admin/kg/nodes")
def kg_admin_list_nodes(user: dict = Depends(_current_user)) -> dict:
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可管理知识图谱。")
    from app.services.knowledge_graph import get_all_nodes
    return {"nodes": get_all_nodes()}


@app.post("/api/admin/kg/nodes")
def kg_admin_upsert_node(payload: dict = Body(...), user: dict = Depends(_current_user)) -> dict:
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可管理知识图谱。")
    from app.services.knowledge_graph import upsert_node
    try:
        node = upsert_node(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "node": node}


@app.delete("/api/admin/kg/nodes/{node_id}")
def kg_admin_delete_node(node_id: str, user: dict = Depends(_current_user)) -> dict:
    if str(user.get("role", "")) not in {"teacher", "admin"}:
        raise HTTPException(status_code=403, detail="仅教师账号可管理知识图谱。")
    from app.services.knowledge_graph import delete_node
    deleted = delete_node(node_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"节点 {node_id} 不存在。")
    return {"ok": True, "deleted": node_id}
