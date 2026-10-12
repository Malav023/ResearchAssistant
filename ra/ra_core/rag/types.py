from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class Paper:
    """One fetched source. Order in the session list == the number the student sees."""
    id: str
    title: str
    abstract: str
    introduction: str = ""
    url: str = ""
    year: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Paper":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_idx: int       # 1-based, matches "first paper", "paper 2"
    paper_id: str
    title: str
    section: str         # "abstract" | "introduction"
    text: str

    @property
    def index_text(self) -> str:
        # Title is prepended so both BM25 and the embedder can match
        # "the federated learning paper" even when the chunk body never says it.
        return f"{self.title}. {self.text}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float                 # fused RRF score
    dense_rank: int | None       # 0-based rank for the primary query (None if absent)
    sparse_rank: int | None


def papers_fingerprint(papers: list[Paper]) -> str:
    """Order-sensitive: reordering papers changes what 'paper 2' means."""
    return hashlib.sha1("|".join(p.id for p in papers).encode()).hexdigest()[:16]
