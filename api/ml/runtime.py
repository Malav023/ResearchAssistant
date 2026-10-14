from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

INTENTS = ("fetch_sources", "create_report", "chat", "unclear")


class MLError(Exception):
    def __init__(self, code: str, message: str, *, retryable: bool = True):
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable


@dataclass
class ClassifyResult:
    label: str
    confidence: float
    probs: dict[str, float] = field(default_factory=dict)
    version: str = ""
    synthetic: bool = False   # True for the keyword fallback: confidence is NOT a model probability

    @property
    def margin(self) -> float:
        v = sorted(self.probs.values(), reverse=True)
        return round(v[0] - v[1], 4) if len(v) > 1 else round(self.confidence, 4)


@dataclass
class ExtractResult:
    topic: str | None
    brief: str | None = None
    avg_logprob: float | None = None
    mode: str = "topic_only"
    version: str = ""


class MLRuntime(Protocol):
    name: str

    async def classify(self, text: str) -> ClassifyResult: ...
    async def extract(self, text: str, intent: str) -> ExtractResult: ...
    async def health(self) -> dict: ...
    async def warmup(self) -> None: ...
    async def aclose(self) -> None: ...
