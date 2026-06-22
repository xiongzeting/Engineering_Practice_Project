"""全局运行时配置。

启动时先调用 :func:`_load_dotenv` 把工作目录（或上级目录）的 ``.env`` 读入
``os.environ``（已存在的环境变量不会被覆盖，方便 Docker / CI 注入）。
然后 :class:`Settings` 以 dataclass 形式把所有配置集中暴露为
``settings.<字段>``。

配置分四组：
- **LLM 凭证**：``llm_*`` 用于文本模型，``llm_vision_*`` 用于多模态视觉 OCR，
  ``llm_score_*`` 用于评分模型。三组可以指向不同供应商——生产里我们用
  mimo-v2.5 做视觉、deepseek-v4-pro 做文本评分，它们各有不同的超时和限流。
- **网络超时**：评分模型走流式时往往比较慢，所以 ``llm_score_*`` 那一组
  允许独立配置连接 / 读超时，``llm_score_no_read_timeout=true`` 还能完全
  关闭读超时，应对超长响应。
- **上传/存储**：``max_upload_mb`` 单图大小上限；``db_path`` SQLite 路径，
  默认 ``outputs/app.db``。
- **特性开关**：``kg_llm_enabled`` 控制是否调 LLM 做知识点映射；
  ``llm_score_reflection_enabled`` 控制评分是否启用反思二轮。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    """Minimal .env loader (stdlib only). Does not override existing env vars."""
    cwd = Path.cwd()
    for candidate in [cwd / ".env", cwd.parent / ".env"]:
        if not candidate.is_file():
            continue
        for raw_line in candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
        break


_load_dotenv()


@dataclass
class Settings:
    """集中暴露所有运行时配置。

    所有字段都从环境变量读取，并提供合理默认值。三组 LLM 配置
    （``llm_*`` / ``llm_vision_*`` / ``llm_score_*``）允许指向不同
    供应商，未显式配置时自动回退到通用 ``llm_*``。
    """

    # 应用名（仅用于日志与诊断）。
    app_name: str = os.getenv("APP_NAME", "math-ocr-grading")

    # === 通用 LLM（默认凭证，视觉/评分未单独配置时回退到这里）===
    llm_api_key: str | None = os.getenv("LLM_API_KEY") or None
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # === 视觉 OCR LLM（多模态，识别手写作业图片）===
    llm_vision_api_key: str | None = (os.getenv("LLM_VISION_API_KEY") or os.getenv("LLM_API_KEY")) or None
    llm_vision_base_url: str = os.getenv("LLM_VISION_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_vision_model: str = os.getenv("LLM_VISION_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")

    # === 评分 LLM（用于 step scoring + KG 精修 + OCR review）===
    llm_score_api_key: str | None = (os.getenv("LLM_SCORE_API_KEY") or os.getenv("LLM_API_KEY")) or None
    llm_score_base_url: str = os.getenv("LLM_SCORE_BASE_URL") or os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    llm_score_model: str = os.getenv("LLM_SCORE_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini")

    # === 调试与诊断 ===
    llm_debug_max_chars: int = int(os.getenv("LLM_DEBUG_MAX_CHARS", "1200"))  # 日志/落库时的字符截断长度

    # === 评分 LLM 超时控制 ===
    llm_score_timeout_sec: int = int(os.getenv("LLM_SCORE_TIMEOUT_SEC", "90"))  # 读超时
    llm_score_connect_timeout_sec: int = int(os.getenv("LLM_SCORE_CONNECT_TIMEOUT_SEC", "15"))  # 连接超时
    llm_score_no_read_timeout: bool = os.getenv("LLM_SCORE_NO_READ_TIMEOUT", "false").lower() in ("1", "true", "yes")  # 完全关闭读超时
    llm_score_use_stream: bool = os.getenv("LLM_SCORE_USE_STREAM", "true").lower() in ("1", "true", "yes")  # 用 SSE 流式
    llm_score_prefer_chat: bool = os.getenv("LLM_SCORE_PREFER_CHAT", "true").lower() in ("1", "true", "yes")  # 优先 /chat/completions

    # === 视觉 OCR 超时与上限 ===
    vision_timeout_sec: int = int(os.getenv("VISION_TIMEOUT_SEC", "45"))
    vision_max_segments: int = int(os.getenv("VISION_MAX_SEGMENTS", "20"))

    # === 上传与持久化 ===
    max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "10"))  # 单图大小上限（MB）
    db_path: str = os.getenv("DB_PATH", "outputs/app.db")

    # === 鉴权 ===
    auth_secret: str = os.getenv("AUTH_SECRET", "change-this-in-production")  # HMAC 签名密钥
    auth_exp_minutes: int = int(os.getenv("AUTH_EXP_MINUTES", "1440"))  # Token 有效期（分钟），默认 24h

    # === 特性开关 ===
    kg_llm_enabled: bool = os.getenv("KG_LLM_ENABLED", "true").lower() in ("1", "true", "yes")  # KG LLM 精修
    llm_score_reflection_enabled: bool = os.getenv("LLM_SCORE_REFLECTION_ENABLED", "true").lower() in ("1", "true", "yes")  # 评分反思二轮


settings = Settings()
