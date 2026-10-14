from __future__ import annotations

import re

from .types import Chunk, Paper

# Protect common abbreviations so "et al." doesn't end a sentence.
_ABBR_RE = re.compile(r"\b(e\.g|i\.e|et al|fig|figs|eq|eqs|vs|cf|approx)\.", re.I)
_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c(\[])')
_DOT = "\u2024"


def split_sentences(text: str) -> list[str]:
    text = " ".join((text or "").split())
    if not text:
        return []
    protected = _ABBR_RE.sub(lambda m: m.group(0)[:-1] + _DOT, text)
    return [s.replace(_DOT, ".").strip() for s in _SPLIT_RE.split(protected) if s.strip()]


def _pack(sentences: list[str], max_words: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_w = 0
    for s in sentences:
        w = len(s.split())
        if cur and cur_w + w > max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap:] if overlap else []
            cur_w = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_w += w
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def _truncate_words(sentences: list[str], cap: int) -> list[str]:
    out, total = [], 0
    for s in sentences:
        total += len(s.split())
        out.append(s)
        if total >= cap:
            break
    return out


def chunk_paper(
    paper: Paper,
    paper_idx: int,
    *,
    max_words: int = 170,
    max_chunks: int = 4,
    max_abstract_chunks: int = 2,
    intro_word_cap: int = 700,
) -> list[Chunk]:
    """Abstract + intro only, 2-4 chunks per paper (matches the locked design)."""
    abs_chunks = _pack(split_sentences(paper.abstract), max_words, overlap=0)[:max_abstract_chunks]
    intro_sents = _truncate_words(split_sentences(paper.introduction), intro_word_cap)
    intro_chunks = _pack(intro_sents, max_words, overlap=1)[: max(0, max_chunks - len(abs_chunks))]

    pieces = [("abstract", t) for t in abs_chunks] + [("introduction", t) for t in intro_chunks]
    return [
        Chunk(
            chunk_id=f"p{paper_idx}c{i}",
            paper_idx=paper_idx,
            paper_id=paper.id,
            title=paper.title,
            section=sec,
            text=text,
        )
        for i, (sec, text) in enumerate(pieces)
    ]
