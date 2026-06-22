"""Pydantic 请求/响应模型，集中定义对外 API 的数据契约。

每个模型对应一个跨进程边界的数据形状，避免在路由里写「裸 dict」导致
字段名漂移。重要约定：

- :class:`StepItem` 是评分流程的最小单位——一行解题步骤。``normalized``
  是剥掉 ``Step N`` / ``步骤 3`` 等前缀后的纯数学内容，``confidence``
  由 OCR 阶段填写，低置信度会触发视觉纠错。
- :class:`StepScore` 是 :class:`StepItem` 对应的打分，``index`` 必须对齐。
- :class:`QuestionGrade` 把一道题（题干 + 多个步骤 + 步骤分 + 题反馈）
  打包，:class:`GradingResult` 再聚合多道题 + 整卷总分，是 ``/api/grade``
  的返回体。
- :class:`DetailReportResponse` 配合前端的「精确报告轮询」——批改完成后
  Markdown 报告在后台异步生成，前端按 ``record_id`` 轮询状态。
- :class:`AuthUser` / :class:`AuthResponse` 是登录/注册响应，``role`` 字段
  目前有三档：``student`` / ``teacher`` / ``admin``。
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepItem(BaseModel):
    index: int = Field(..., description="1-based index")
    raw: str = Field(..., description="Step text from OCR")
    normalized: str = Field(..., description="Normalized step text")
    has_equation: bool = Field(default=False)
    confidence: float = Field(default=0.8)


class StepScore(BaseModel):
    index: int
    score: float = Field(..., ge=0)
    reason: str


class QuestionGrade(BaseModel):
    qno: int = Field(..., description="1-based question number")
    max_score: float = Field(..., ge=0)
    score: float = Field(..., ge=0)
    steps: list[StepItem]
    step_scores: list[StepScore]
    feedback: str


class GradingResult(BaseModel):
    ocr_text: str
    steps: list[StepItem]
    step_scores: list[StepScore]
    total_score: float = Field(..., ge=0, le=100)
    feedback: str
    engine: str = Field(..., description="rule-based or llm+rule")
    grading_meta: dict[str, Any] = Field(default_factory=dict)
    questions: list[QuestionGrade] = Field(default_factory=list)
    total_max_score: float = Field(default=100.0, ge=0)
    record_id: int | None = Field(default=None, description=" grading_records.id for detail-report polling")


class DetailReportResponse(BaseModel):
    status: str = Field(..., description="pending | ready | failed")
    markdown: str | None = None
    updated_at: str | None = None
    error: str | None = None


class SaveSessionResponse(BaseModel):
    record_id: int
    status: str = "pending"


class AuthUser(BaseModel):
    id: int
    username: str
    role: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser
