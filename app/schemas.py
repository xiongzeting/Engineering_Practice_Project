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
    score: float = Field(..., ge=0, le=100)
    reason: str


class GradingResult(BaseModel):
    ocr_text: str
    steps: list[StepItem]
    step_scores: list[StepScore]
    total_score: float = Field(..., ge=0, le=100)
    feedback: str
    engine: str = Field(..., description="rule-based or llm+rule")
    grading_meta: dict[str, Any] = Field(default_factory=dict)


class AuthUser(BaseModel):
    id: int
    username: str
    role: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: AuthUser
