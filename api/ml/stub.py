"""Keyword heuristics. Used (a) as ML_BACKEND=stub for UI/dev work with no models, and
(b) as the degraded fallback when the real classifier/extractor is unreachable."""
from __future__ import annotations

import re

from ra_core.rag.paper_ref import extract_paper_refs

from .runtime import INTENTS, ClassifyResult, ExtractResult

_REPORT = re.compile(r"\b(report|review|survey|summar(?:y|ise|ize)|overview|write[- ]?up)\b", re.I)
_FETCH = re.compile(r"\b(find|fetch|pull|get|search|look(?:ing)? for|latest|recent|newest|newer|sources?|literature|papers? (?:on|about))\b", re.I)
_CHAT_Q = re.compile(r"\b(which|what|how|why|did|does|do|explain|compare)\b.*\b(papers?|authors?|study|studies|article)\b", re.I)

_LEAD_FETCH = re.compile(
    r"^(?:please\s+)?(?:can you\s+)?(?:pull|find|get|fetch|search(?: for)?|show|give)(?: me)?(?: the)?"
    r"(?: (?:latest|newest|recent|newer))?(?: (?:papers|articles|work|research|sources|literature))?"
    r"(?: (?:on|about|for|regarding))?\s+", re.I)
_LEAD_REPORT = re.compile(
    r"^(?:please\s+)?(?:write|make|create|generate|i need)(?: me)?(?: an?)?(?: (?:full|short|quick))?"
    r"(?: (?:report|review|summary|overview|survey))?(?: (?:on|about|of|for))?\s+", re.I)


def heuristic_label(text: str) -> str:
    t = text.strip()
    if not t:
        return "unclear"
    if extract_paper_refs(t, 8) or _CHAT_Q.search(t):
        return "chat"
    if _REPORT.search(t):
        return "create_report"
    if _FETCH.search(t):
        return "fetch_sources"
    return "unclear"


def heuristic_classify(text: str) -> ClassifyResult:
    label = heuristic_label(text)
    conf = 0.75 if label != "unclear" else 0.9
    rest = (1 - conf) / (len(INTENTS) - 1)
    probs = {i: (conf if i == label else round(rest, 4)) for i in INTENTS}
    return ClassifyResult(label, conf, probs, version="keyword_heuristic", synthetic=True)


def heuristic_topic(text: str) -> str:
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = _LEAD_REPORT.sub("", _LEAD_FETCH.sub("", t, count=1), count=1)
    t = re.sub(r"[^\w\s+#\-]", "", t).strip()
    for tail in (" for my seminar", " for my class", " please"):
        if t.endswith(tail):
            t = t[: -len(tail)]
    return (t or re.sub(r"[^\w\s+#\-]", "", text.lower()).strip())[:120]


class StubMLRuntime:
    name = "stub"

    async def classify(self, text: str) -> ClassifyResult:
        return heuristic_classify(text)

    async def extract(self, text: str, intent: str) -> ExtractResult:
        return ExtractResult(topic=heuristic_topic(text), version="keyword_heuristic")

    async def health(self) -> dict:
        return {"ok": True, "backend": "stub", "note": "keyword heuristics, no models loaded"}

    async def warmup(self) -> None:
        return None

    async def aclose(self) -> None:
        return None
