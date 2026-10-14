"""Detect 'the second paper' / 'paper 3' / 'first and third papers' / 'last paper'."""
from __future__ import annotations

import re

_ORD = {
    "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7, "eighth": 8, "8th": 8,
}
_NUMW = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
_ORD_ALT = "|".join(sorted(_ORD, key=len, reverse=True))
_NUM_ALT = "|".join(sorted(_NUMW, key=len, reverse=True))
_SEP = r"(?:\s*(?:,|&|and|or|vs\.?|versus)\s*)"
_NOUN = r"(?:papers?|articles?|studies|study|ones?)"

_RE_ORD = re.compile(rf"\b((?:{_ORD_ALT})(?:{_SEP}(?:{_ORD_ALT}))*)\s+{_NOUN}\b", re.I)
_RE_NUM = re.compile(
    rf"\b(?:papers?|articles?|study)\s*(?:#|no\.?\s*|number\s+)?"
    rf"((?:\d+|{_NUM_ALT})(?:{_SEP}#?(?:\d+|{_NUM_ALT}))*)\b",
    re.I,
)
_RE_LAST = re.compile(r"\b(?:last|final)\s+(?:paper|article|study|one)\b", re.I)


def extract_paper_refs(query: str, n_papers: int) -> set[int]:
    refs: set[int] = set()
    for m in _RE_ORD.finditer(query):
        for tok in re.findall(r"[a-z0-9]+", m.group(1).lower()):
            if tok in _ORD:
                refs.add(_ORD[tok])
    for m in _RE_NUM.finditer(query):
        for tok in re.findall(r"[a-z0-9]+", m.group(1).lower()):
            n = int(tok) if tok.isdigit() else _NUMW.get(tok)
            if n:
                refs.add(n)
    if _RE_LAST.search(query) and n_papers:
        refs.add(n_papers)
    return {r for r in refs if 1 <= r <= n_papers}
