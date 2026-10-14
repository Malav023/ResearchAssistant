import re

_WRITE = re.compile(r"\b(write|draft|generate|create|make|produce|prepare|compile|put together|turn|give me)\b", re.I)
_DOC = re.compile(r"\b(report|summary|summarise|summarize|review|overview|synthesis|write-?up)\b", re.I)
_SPECIFIC = re.compile(
    r"\b(paper|article|study)\s*#?\s*\d+\b|"
    r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th|last)\s+(paper|article|study)\b", re.I)
_DEICTIC = re.compile(r"\b(these|those|them|the papers|the sources|my papers|the above|fetched|"
                      r"all (?:five|5|of them))\b", re.I)


def apply_rules(text: str, intent: str, has_papers: bool) -> tuple[str, bool, str | None]:
    """Returns (intent, use_session_papers, rule_name). Call after the confidence gate."""
    wants_doc = bool(_WRITE.search(text) and _DOC.search(text))
    if not (has_papers and wants_doc) or _SPECIFIC.search(text):
        return intent, False, None            # "summarize paper 2" stays chat
    deictic = bool(_DEICTIC.search(text))
    if intent in ("chat", "unclear") and deictic:
        return "create_report", True, "report_over_session_papers"
    return intent, intent == "create_report" and deictic, None