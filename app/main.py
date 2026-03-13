from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import AuthResponse, AuthUser, GradingResult
from app.services.auth import create_access_token, hash_password, parse_access_token, verify_password
from app.services.db import (
    create_user,
    get_user_by_id,
    get_user_by_username,
    init_db,
    list_grading_records,
    list_ocr_records,
    save_grading_record,
    save_ocr_record,
)
from app.services.ocr_corrector import llm_correct_ocr_text
from app.services.ocr_postprocess import (
    attach_low_confidence_flag,
    build_ocr_text,
    group_segments_by_question,
)
from app.services.ocr_service import OCRService
from app.services.scorer import score_steps
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


@app.get("/api/history/ocr")
def ocr_history(limit: int = 20, user: dict = Depends(_current_user)) -> dict:
    return {"items": list_ocr_records(int(user["id"]), limit=limit)}


@app.get("/api/history/grading")
def grading_history(limit: int = 20, user: dict = Depends(_current_user)) -> dict:
    return {"items": list_grading_records(int(user["id"]), limit=limit)}


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
        raise HTTPException(status_code=400, detail=res.get("error", "多模态OCR失败"))

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


@app.post("/api/grade", response_model=GradingResult)
def grade_homework(
    extracted_text: str = Form(...),
    reference_solution: str | None = Form(default=None),
    use_llm: bool = Form(default=False),
    user: dict = Depends(_current_user),
) -> GradingResult:
    ocr_text = extracted_text.strip()
    if not ocr_text:
        raise HTTPException(status_code=400, detail="请先执行 OCR 或手工输入公式文本。")

    steps = split_steps(ocr_text)
    if not steps:
        raise HTTPException(status_code=400, detail="未检测到可评分的步骤，请确认公式文本。")

    reference_steps = split_steps(reference_solution) if reference_solution else []
    step_scores, total, feedback, score_engine, grading_meta = score_steps(
        ocr_text=ocr_text,
        steps=steps,
        reference_steps=reference_steps,
        reference_raw=reference_solution,
        use_llm=use_llm,
    )

    save_grading_record(
        user_id=int(user["id"]),
        engine=str(score_engine),
        total_score=float(total),
        steps_count=len(steps),
        ocr_text=ocr_text,
        grading_meta=grading_meta,
    )
    return GradingResult(
        ocr_text=ocr_text,
        steps=steps,
        step_scores=step_scores,
        total_score=total,
        feedback=feedback,
        engine=f"{score_engine}",
        grading_meta=grading_meta,
    )
