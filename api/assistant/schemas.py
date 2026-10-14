from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MessageRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1, max_length=2000)


class PaperView(BaseModel):
    n: int                       # 1-based; this is what the student means by "paper 2"
    id: str
    title: str
    abstract: str = ""
    year: int | None = None
    citations: int = 0
    first_author: str = ""
    top_concept: str = ""
    doi: str = ""
    url: str = ""


class Citation(BaseModel):
    paper: int
    title: str


class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: str
    citations: list[Citation] = []


class Suggestion(BaseModel):
    label: str
    prefill: str


class MessageResponse(BaseModel):
    session_id: str
    trace_id: str
    intent: str
    confidence: float | None = None
    action: Literal["papers", "report", "answer", "clarify", "needs_papers"]
    topic: str | None = None
    brief: str | None = None
    message: AssistantMessage
    papers: list[PaperView] | None = None
    suggestions: list[Suggestion] | None = None
    degraded: list[str] = []
    debug: dict | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    correct_intent: Literal["fetch_sources", "create_report", "chat", "unclear"] | None = None
    note: str | None = Field(default=None, max_length=500)
