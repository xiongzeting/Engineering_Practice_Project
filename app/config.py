from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    app_name: str = os.getenv("APP_NAME", "math-ocr-grading")
    llm_api_key: str | None = os.getenv("LLM_API_KEY")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_debug_max_chars: int = int(os.getenv("LLM_DEBUG_MAX_CHARS", "1200"))
    llm_score_timeout_sec: int = int(os.getenv("LLM_SCORE_TIMEOUT_SEC", "90"))
    llm_score_connect_timeout_sec: int = int(os.getenv("LLM_SCORE_CONNECT_TIMEOUT_SEC", "15"))
    llm_score_no_read_timeout: bool = os.getenv("LLM_SCORE_NO_READ_TIMEOUT", "false").lower() in ("1", "true", "yes")
    llm_score_use_stream: bool = os.getenv("LLM_SCORE_USE_STREAM", "true").lower() in ("1", "true", "yes")
    llm_score_prefer_chat: bool = os.getenv("LLM_SCORE_PREFER_CHAT", "true").lower() in ("1", "true", "yes")
    vision_timeout_sec: int = int(os.getenv("VISION_TIMEOUT_SEC", "45"))
    vision_max_segments: int = int(os.getenv("VISION_MAX_SEGMENTS", "20"))
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))
    db_path: str = os.getenv("DB_PATH", "outputs/app.db")
    auth_secret: str = os.getenv("AUTH_SECRET", "change-this-in-production")
    auth_exp_minutes: int = int(os.getenv("AUTH_EXP_MINUTES", "1440"))


settings = Settings()
